"""
Arbor transformer architecture for the Arbor-o1 model.

The Arbor architecture implements dynamic growth capabilities,
while Arbor-o1 is the specific model implementation/release.
"""

from typing import List, Optional, Dict, Any, Union
import torch
import torch.nn as nn
from dataclasses import dataclass

from .layers import ExpandableEmbedding, count_parameters
from .block import ArborBlock, PositionalEncoding


@dataclass
class ArborConfig:
    """Configuration for the Arbor transformer architecture."""
    
    # Model architecture
    vocab_size: int = 10000
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    max_seq_length: int = 1024
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Activation and other settings
    activation: str = "gelu"
    causal: bool = True
    tie_word_embeddings: bool = True
    
    # Growth settings
    growth_enabled: bool = True
    
    # Adaptive context settings
    adaptive_context: bool = True
    context_router_layers: int = 3
    min_context_length: int = 1024
    max_context_length: int = None  # Will use max_seq_length
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        assert self.activation in ["gelu", "relu", "swish"], f"Unsupported activation: {self.activation}"
        
        # Set max context length if not specified
        if self.max_context_length is None:
            self.max_context_length = self.max_seq_length


class ArborTransformer(nn.Module):
    """
    Arbor transformer architecture with dynamic growth capabilities.
    
    This implements the core Arbor architecture used in the Arbor-o1 model.
    The architecture can expand its capacity during training by growing
    the hidden dimensions of feed-forward networks in selected layers.
    """
    
    def __init__(self, config: ArborConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = ExpandableEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.dim,
            padding_idx=None,  # TODO: Add padding token support
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            dim=config.dim,
            max_seq_length=config.max_seq_length,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ArborBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                ffn_dropout=config.ffn_dropout,
                layer_norm_eps=config.layer_norm_eps,
                activation=config.activation,
                causal=config.causal,
            )
            for _ in range(config.num_layers)
        ])
        
        # Adaptive context system
        if config.adaptive_context:
            from .adaptive_context import AdaptiveContextManager, AdaptiveContextConfig
            
            adaptive_config = AdaptiveContextConfig(
                vocab_size=config.vocab_size,
                max_seq_length=config.max_seq_length,
                router_max_length=512,  # Fast analysis
                min_context=config.min_context_length,
                max_context=config.max_context_length,
                growth_enabled=config.growth_enabled
            )
            
            self.context_manager = AdaptiveContextManager(adaptive_config)
            self.current_context_length = config.min_context_length
            print(f"🧠 Adaptive context system initialized")
            print(f"   Context range: {config.min_context_length:,} - {config.max_context_length:,}")
        else:
            self.context_manager = None
            self.current_context_length = config.max_seq_length
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        
        # Output projection (language modeling head)
        if config.tie_word_embeddings:
            # Share weights between input embeddings and output projection
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Track growth events
        self.growth_history: List[Dict[str, Any]] = []
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Token ids of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            labels: Labels for language modeling loss (batch, seq_len)
            return_dict: Whether to return a dictionary or just logits
            
        Returns:
            If return_dict=True: Dictionary with logits and optionally loss
            If return_dict=False: Just the logits tensor
        """
        batch_size, seq_len = input_ids.shape
        
        # Adaptive context window decision
        context_decision = None
        if self.context_manager is not None and self.training is False:  # Only during inference
            adapted_length, context_decision = self.context_manager.analyze_and_adapt(input_ids, self)
            
            # Truncate input if it exceeds adapted context length
            if seq_len > adapted_length:
                input_ids = input_ids[:, -adapted_length:]  # Keep most recent tokens
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -adapted_length:]
                if labels is not None:
                    labels = labels[:, -adapted_length:]
                seq_len = adapted_length
                
                print(f"🔄 Context adapted: {seq_len} tokens, task: {context_decision.task_type}")
                print(f"   Reasoning: {context_decision.reasoning}")
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Language modeling head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings
            logits = torch.matmul(x, self.token_embedding.embedding.weight.T)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": x,
            }
        else:
            return logits
    
    def grow(self, layer_indices: List[int], add_hidden: int) -> None:
        """
        Grow the model by expanding FFN hidden dimensions in specified layers.
        
        Args:
            layer_indices: List of layer indices to expand
            add_hidden: Number of hidden units to add to each layer
        """
        if add_hidden <= 0:
            return
        
        growth_event = {
            "layer_indices": layer_indices,
            "add_hidden": add_hidden,
            "old_param_count": self.param_count(),
        }
        
        # Expand selected layers
        for layer_idx in layer_indices:
            if 0 <= layer_idx < len(self.layers):
                old_ffn_dim = self.layers[layer_idx].ffn_dim
                self.layers[layer_idx].grow_ffn(add_hidden)
                print(f"Layer {layer_idx}: FFN {old_ffn_dim} -> {self.layers[layer_idx].ffn_dim}")
        
        growth_event["new_param_count"] = self.param_count()
        self.growth_history.append(growth_event)
        
        print(f"Model grown: {growth_event['old_param_count']:,} -> {growth_event['new_param_count']:,} parameters")
    
    def grow_vocabulary(self, add_vocab: int) -> None:
        """
        Expand the vocabulary size.
        
        Args:
            add_vocab: Number of vocabulary entries to add
        """
        if add_vocab <= 0:
            return
            
        old_vocab = self.config.vocab_size
        self.token_embedding.grow_vocab(add_vocab)
        self.config.vocab_size = self.token_embedding.vocab_size
        
        # Update language modeling head if not tied
        if self.lm_head is not None:
            old_lm_head = self.lm_head
            self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
            
            # Copy old weights
            with torch.no_grad():
                self.lm_head.weight[:old_vocab] = old_lm_head.weight
                # Initialize new weights
                std = old_lm_head.weight.std().item()
                self.lm_head.weight[old_vocab:].normal_(0, std)
        
        print(f"Vocabulary expanded: {old_vocab} -> {self.config.vocab_size}")
    
    def param_count(self) -> int:
        """Return total number of parameters."""
        total, _ = count_parameters(self)
        return total
    
    def trainable_param_count(self) -> int:
        """Return number of trainable parameters."""
        _, trainable = count_parameters(self)
        return trainable
    
    def get_growth_history(self) -> List[Dict[str, Any]]:
        """Return the history of growth events."""
        return self.growth_history.copy()
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting token ids (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token ids (batch, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                outputs = self.forward(input_ids, return_dict=True)
                logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    logits[logits < values[:, [-1]]] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def resize_position_embeddings(self, new_length: int):
        """Dynamically resize position embeddings for new context length."""
        if new_length == self.current_context_length:
            return
        
        print(f"🔄 Resizing position embeddings: {self.current_context_length} → {new_length}")
        
        # Resize positional encoding
        if hasattr(self.pos_encoding, 'resize'):
            self.pos_encoding.resize(new_length)
        
        self.current_context_length = new_length
    
    def set_max_position_embeddings(self, max_length: int):
        """Set maximum position embeddings for attention computation."""
        self.config.max_seq_length = max_length
        
        # Update each layer's attention mechanism
        for layer in self.layers:
            if hasattr(layer.attention, 'set_max_length'):
                layer.attention.set_max_length(max_length)
    
    def clear_attention_cache(self):
        """Clear attention caches when context length changes."""
        for layer in self.layers:
            if hasattr(layer.attention, 'clear_cache'):
                layer.attention.clear_cache()
    
    def get_context_info(self) -> Dict[str, any]:
        """Get information about current context configuration."""
        info = {
            "current_context_length": self.current_context_length,
            "max_context_length": self.config.max_context_length,
            "min_context_length": self.config.min_context_length,
            "adaptive_context_enabled": self.context_manager is not None,
        }
        
        if self.context_manager is not None:
            info.update({
                "context_router_active": True,
                "available_context_lengths": self.context_manager.config.context_lengths,
                "supported_task_types": self.context_manager.config.task_types,
            })
        
        return info
    
    def force_context_length(self, length: int):
        """Force a specific context length (bypass adaptive system)."""
        if length not in self.context_manager.config.context_lengths:
            print(f"⚠️  Warning: {length} not in supported lengths {self.context_manager.config.context_lengths}")
        
        self.resize_position_embeddings(length)
        self.set_max_position_embeddings(length)
        self.clear_attention_cache()
        
        print(f"🔧 Forced context length to {length:,} tokens")


def create_arbor_model(
    vocab_size: int = 10000,
    dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    ffn_dim: int = 2048,
    max_seq_length: int = 1024,
    **kwargs
) -> ArborTransformer:
    """
    Factory function to create an Arbor transformer model.
    
    Args:
        vocab_size: Vocabulary size
        dim: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward hidden dimension
        max_seq_length: Maximum sequence length
        **kwargs: Additional configuration options
        
    Returns:
        ArborTransformer model
    """
    config = ArborConfig(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_length=max_seq_length,
        **kwargs
    )
    
    return ArborTransformer(config)
