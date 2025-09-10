"""
Tokenization utilities and dataset helpers.
"""

from typing import List, Dict, Any, Optional, Union
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np


class ArborTokenizer:
    """
    Wrapper around HuggingFace tokenizers with Arbor-specific functionality.
    """
    
    def __init__(
        self,
        tokenizer_name_or_path: str = "gpt2",
        vocab_size: Optional[int] = None,
        max_length: int = 1024,
        padding_side: str = "right",
    ):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_length = max_length
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {tokenizer_name_or_path}: {e}")
            # Fallback to GPT-2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set padding side
        self.tokenizer.padding_side = padding_side
        
        # Resize vocabulary if requested
        if vocab_size is not None and vocab_size != len(self.tokenizer):
            self._resize_vocabulary(vocab_size)
        
        self.vocab_size = len(self.tokenizer)
        
    def _resize_vocabulary(self, new_vocab_size: int) -> None:
        """Resize the tokenizer vocabulary."""
        current_size = len(self.tokenizer)
        if new_vocab_size == current_size:
            return
        
        if new_vocab_size < current_size:
            # Truncate vocabulary (not recommended)
            print(f"Warning: Truncating vocabulary from {current_size} to {new_vocab_size}")
        else:
            # Expand vocabulary by adding special tokens
            num_new_tokens = new_vocab_size - current_size
            new_tokens = [f"<extra_token_{i}>" for i in range(num_new_tokens)]
            self.tokenizer.add_tokens(new_tokens)
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate long sequences
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if max_length is None:
            max_length = self.max_length
        
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def batch_decode(
        self,
        sequences: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.
        
        Args:
            sequences: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded texts
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens (not IDs)."""
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save tokenizer to directory."""
        self.tokenizer.save_pretrained(save_directory)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.tokenizer)


class StreamingTokenizer:
    """
    Tokenizer for streaming/online tokenization of large datasets.
    """
    
    def __init__(
        self,
        tokenizer: ArborTokenizer,
        chunk_size: int = 1000,
        overlap: int = 50,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def tokenize_stream(
        self,
        text_iterator,
        max_length: int = 1024,
    ):
        """
        Tokenize a stream of text documents.
        
        Args:
            text_iterator: Iterator yielding text documents
            max_length: Maximum sequence length
            
        Yields:
            Tokenized document dictionaries
        """
        for text in text_iterator:
            # Split long documents into chunks
            if len(text) > self.chunk_size:
                chunks = self._split_into_chunks(text)
                for chunk in chunks:
                    encoded = self.tokenizer.encode(
                        chunk,
                        max_length=max_length,
                        padding=False,
                        truncation=True,
                    )
                    yield encoded
            else:
                encoded = self.tokenizer.encode(
                    text,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                )
                yield encoded
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
            
            # Find a good breaking point (space or punctuation)
            break_point = self._find_break_point(chunk)
            if break_point > 0:
                chunk = chunk[:break_point]
                start += break_point - self.overlap
            else:
                start = end - self.overlap
            
            start = max(start, 0)
        
        return chunks
    
    def _find_break_point(self, text: str) -> int:
        """Find a good point to break text (space, newline, punctuation)."""
        break_chars = [' ', '\n', '.', '!', '?', ';', ',']
        
        # Search backwards from the end
        for i in range(len(text) - 1, max(0, len(text) - 100), -1):
            if text[i] in break_chars:
                return i + 1
        
        return len(text)


def create_synthetic_vocabulary(vocab_size: int = 1000) -> List[str]:
    """
    Create a synthetic vocabulary for testing.
    
    Args:
        vocab_size: Size of vocabulary to create
        
    Returns:
        List of vocabulary tokens
    """
    vocab = ["<pad>", "<unk>", "<bos>", "<eos>"]
    
    # Add common words
    common_words = [
        "the", "and", "to", "of", "a", "in", "for", "is", "on", "that",
        "by", "this", "with", "i", "you", "it", "not", "or", "be", "are",
        "from", "at", "as", "your", "all", "any", "can", "had", "her", "was",
        "one", "our", "out", "day", "get", "use", "man", "new", "now", "way",
        "may", "say", "each", "which", "she", "do", "how", "their", "if", "will",
    ]
    
    vocab.extend(common_words)
    
    # Add synthetic tokens
    remaining = vocab_size - len(vocab)
    for i in range(remaining):
        vocab.append(f"token_{i:04d}")
    
    return vocab[:vocab_size]


def create_tokenizer_from_vocab(
    vocab: List[str],
    save_path: Optional[str] = None,
) -> ArborTokenizer:
    """
    Create a tokenizer from a vocabulary list.
    
    Args:
        vocab: List of vocabulary tokens
        save_path: Optional path to save the tokenizer
        
    Returns:
        ArborTokenizer instance
    """
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        from transformers import PreTrainedTokenizerFast
        
        # Create a simple word-level tokenizer
        tokenizer = Tokenizer(models.WordLevel(vocab={token: i for i, token in enumerate(vocab)}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece(prefix="")
        
        # Convert to HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
        
        if save_path:
            hf_tokenizer.save_pretrained(save_path)
        
        # Wrap in ArborTokenizer
        arbor_tokenizer = ArborTokenizer("gpt2")  # Temporary
        arbor_tokenizer.tokenizer = hf_tokenizer
        arbor_tokenizer.vocab_size = len(vocab)
        
        return arbor_tokenizer
        
    except ImportError:
        # Fallback: use GPT-2 tokenizer and resize
        print("Warning: tokenizers library not available, using GPT-2 tokenizer")
        return ArborTokenizer("gpt2", vocab_size=len(vocab))


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching tokenized sequences.
    
    Args:
        batch: List of tokenized examples
        
    Returns:
        Batched tensors
    """
    # Extract sequences
    input_ids = [item["input_ids"].squeeze() for item in batch]
    
    # Find maximum length in batch
    max_len = max(seq.size(0) for seq in input_ids)
    
    # Pad sequences
    batch_input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, seq in enumerate(input_ids):
        seq_len = seq.size(0)
        batch_input_ids[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1
    
    result = {
        "input_ids": batch_input_ids,
        "attention_mask": attention_mask,
    }
    
    # Add labels for language modeling (shifted input_ids)
    result["labels"] = batch_input_ids.clone()
    
    return result


def estimate_tokens_per_second(
    text_length: int,
    tokenizer: ArborTokenizer,
    sample_text: str = "This is a sample sentence for token estimation.",
) -> float:
    """
    Estimate average tokens per character for a tokenizer.
    
    Args:
        text_length: Length of text in characters
        tokenizer: Tokenizer to use
        sample_text: Sample text for estimation
        
    Returns:
        Estimated number of tokens
    """
    # Tokenize sample text
    encoded = tokenizer.encode(sample_text, add_special_tokens=False)
    num_tokens = len(encoded["input_ids"][0])
    num_chars = len(sample_text)
    
    # Calculate ratio
    tokens_per_char = num_tokens / num_chars
    
    return text_length * tokens_per_char
