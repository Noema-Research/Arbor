"""
Custom inference library for Arbor-o1 models.

This provides a simple, optimized inference interface for Arbor models
without requiring Hugging Face Transformers.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union
import json
from pathlib import Path

from ..modeling.model import ArborTransformer, ArborConfig
from ..data.tokenizers import ArborTokenizer


class ArborInference:
    """
    Simple inference engine for Arbor models.
    
    Provides easy-to-use methods for text generation, embedding extraction,
    and model interaction without heavy dependencies.
    """
    
    def __init__(
        self,
        model: ArborTransformer,
        tokenizer: ArborTokenizer,
        device: str = "auto"
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"🌱 Arbor Inference Engine initialized")
        print(f"   Device: {self.device}")
        print(f"   Parameters: {self.model.param_count():,}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: str = "auto"
    ) -> "ArborInference":
        """
        Load model and tokenizer from a checkpoint directory.
        
        Args:
            model_path: Path to model checkpoint directory
            device: Device to load model on
            
        Returns:
            ArborInference instance
        """
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
            config = ArborConfig(**config_dict)
        else:
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        # Load model
        model = ArborTransformer(config)
        checkpoint_path = model_path / "pytorch_model.bin"
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model weights not found at {checkpoint_path}")
        
        # Load tokenizer
        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = ArborTokenizer(str(tokenizer_path))
        else:
            # Fallback to default tokenizer
            tokenizer = ArborTokenizer("gpt2", vocab_size=config.vocab_size)
            print("⚠️  Using default tokenizer")
        
        return cls(model, tokenizer, device)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_tokens: Optional[List[str]] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            stop_tokens: List of stop tokens
            
        Returns:
            Generated text (string if num_return_sequences=1, list otherwise)
        """
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Prepare stop token ids
        stop_token_ids = []
        if stop_tokens:
            for token in stop_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)["input_ids"]
                stop_token_ids.extend(token_ids)
        
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            with torch.no_grad():
                generated_ids = self._generate_sequence(
                    input_ids=input_ids.clone(),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    stop_token_ids=stop_token_ids,
                    **kwargs
                )
                
                # Decode
                full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                new_text = full_text[len(prompt):].strip()
                generated_sequences.append(new_text)
        
        if num_return_sequences == 1:
            return generated_sequences[0]
        return generated_sequences
    
    def _generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        stop_token_ids: List[int],
        **kwargs
    ) -> torch.Tensor:
        """Internal sequence generation method."""
        
        batch_size, seq_len = input_ids.shape
        generated_tokens = 0
        
        while generated_tokens < max_new_tokens:
            # Forward pass
            outputs = self.model(input_ids, return_dict=True)
            logits = outputs.logits[:, -1, :]  # Get last token logits
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or take argmax
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append token
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens += 1
            
            # Check for stop tokens
            if next_token.item() in stop_token_ids:
                break
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return input_ids
    
    def embed(
        self,
        texts: Union[str, List[str]],
        layer: int = -1
    ) -> torch.Tensor:
        """
        Extract embeddings from texts.
        
        Args:
            texts: Input text(s)
            layer: Which layer to extract embeddings from (-1 for last)
            
        Returns:
            Embeddings tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer.encode(text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                
                outputs = self.model(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Get embeddings from specified layer
                hidden_states = outputs.hidden_states[layer]  # (batch, seq, hidden)
                
                # Mean pooling over sequence dimension
                embedding = hidden_states.mean(dim=1)  # (batch, hidden)
                embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)
    
    def score_text(
        self,
        text: str,
        stride: int = 512
    ) -> Dict[str, float]:
        """
        Compute perplexity and other scores for text.
        
        Args:
            text: Input text to score
            stride: Stride for sliding window (for long texts)
            
        Returns:
            Dictionary with scores
        """
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        seq_len = input_ids.size(1)
        
        nlls = []
        prev_end_loc = 0
        
        with torch.no_grad():
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + stride, seq_len)
                trg_len = end_loc - prev_end_loc
                
                input_chunk = input_ids[:, begin_loc:end_loc]
                target_ids = input_chunk.clone()
                target_ids[:, :-trg_len] = -100
                
                outputs = self.model(input_chunk, labels=target_ids, return_dict=True)
                neg_log_likelihood = outputs.loss * trg_len
                
                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
        
        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / (seq_len - 1))
        
        return {
            "perplexity": perplexity.item(),
            "log_likelihood": -total_nll.item(),
            "bits_per_character": (total_nll / len(text)).item(),
            "tokens": seq_len
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Simple chat interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        # Format messages into prompt
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "system":
                prompt += f"System: {content}\n"
        
        prompt += "Assistant:"
        
        # Generate response
        response = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_tokens=["User:", "\\n\\n"],
            **kwargs
        )
        
        return response.strip()
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """
        Save model, config, and tokenizer.
        
        Args:
            save_path: Directory to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = self.model.config.__dict__
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save model weights
        torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        
        # Save tokenizer
        tokenizer_path = save_path / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_path))
        
        print(f"✅ Model saved to {save_path}")


# Convenience functions
def load_model(model_path: str, device: str = "auto") -> ArborInference:
    """Load an Arbor model for inference."""
    return ArborInference.from_pretrained(model_path, device)


def generate_text(
    model_path: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    device: str = "auto"
) -> str:
    """Quick text generation."""
    inference = load_model(model_path, device)
    return inference.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
