"""
Expandable layers and utilities for dynamic model growth.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def expand_linear(old_linear: nn.Linear, add_out: int) -> nn.Linear:
    """
    Expand a linear layer's output dimension.
    
    Args:
        old_linear: Existing linear layer
        add_out: Number of output features to add
        
    Returns:
        New linear layer with expanded output dimension
    """
    old_in, old_out = old_linear.in_features, old_linear.out_features
    new_out = old_out + add_out
    
    # Create new layer
    new_linear = nn.Linear(old_in, new_out, bias=old_linear.bias is not None)
    
    # Copy old weights
    with torch.no_grad():
        new_linear.weight[:old_out] = old_linear.weight
        if old_linear.bias is not None:
            new_linear.bias[:old_out] = old_linear.bias
            
        # Initialize new weights using Xavier uniform
        fan_in = old_in
        bound = math.sqrt(6.0 / fan_in)
        new_linear.weight[old_out:].uniform_(-bound, bound)
        if new_linear.bias is not None:
            new_linear.bias[old_out:].zero_()
    
    return new_linear


def expand_linear_incols(old_linear: nn.Linear, add_in: int) -> nn.Linear:
    """
    Expand a linear layer's input dimension.
    
    Args:
        old_linear: Existing linear layer
        add_in: Number of input features to add
        
    Returns:
        New linear layer with expanded input dimension
    """
    old_in, old_out = old_linear.in_features, old_linear.out_features
    new_in = old_in + add_in
    
    # Create new layer
    new_linear = nn.Linear(new_in, old_out, bias=old_linear.bias is not None)
    
    # Copy old weights and bias
    with torch.no_grad():
        new_linear.weight[:, :old_in] = old_linear.weight
        if old_linear.bias is not None:
            new_linear.bias = old_linear.bias.clone()
            
        # Initialize new weights using Xavier uniform
        fan_out = old_out
        bound = math.sqrt(6.0 / fan_out)
        new_linear.weight[:, old_in:].uniform_(-bound, bound)
    
    return new_linear


class ExpandableFFN(nn.Module):
    """
    Feed-forward network that can dynamically expand its hidden dimension.
    """
    
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int, 
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Initialize layers
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "swish":
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FFN."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x
    
    def grow(self, add_hidden: int) -> None:
        """
        Expand the hidden dimension of the FFN.
        
        Args:
            add_hidden: Number of hidden units to add
        """
        if add_hidden <= 0:
            return
            
        old_hidden = self.hidden_dim
        self.hidden_dim += add_hidden
        
        # Expand fc1 (output dimension)
        self.fc1 = expand_linear(self.fc1, add_hidden)
        
        # Expand fc2 (input dimension)  
        self.fc2 = expand_linear_incols(self.fc2, add_hidden)
        
        print(f"ExpandableFFN grown: {old_hidden} -> {self.hidden_dim} hidden units")
    
    def param_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}, dropout={self.dropout}"


class ExpandableEmbedding(nn.Module):
    """
    Embedding layer that can expand vocabulary size dynamically.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)
    
    def grow_vocab(self, add_vocab: int) -> None:
        """
        Expand vocabulary size.
        
        Args:
            add_vocab: Number of vocabulary entries to add
        """
        if add_vocab <= 0:
            return
            
        old_vocab = self.vocab_size
        new_vocab = old_vocab + add_vocab
        
        # Create new embedding
        new_embedding = nn.Embedding(
            new_vocab, self.embed_dim, padding_idx=self.padding_idx
        )
        
        # Copy old weights
        with torch.no_grad():
            new_embedding.weight[:old_vocab] = self.embedding.weight
            
            # Initialize new embeddings
            std = self.embedding.weight.std().item()
            new_embedding.weight[old_vocab:].normal_(0, std)
        
        self.embedding = new_embedding
        self.vocab_size = new_vocab
        
        print(f"Vocabulary expanded: {old_vocab} -> {new_vocab}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops_per_token(model: nn.Module, seq_length: int = 512) -> int:
    """
    Estimate FLOPs per token for a transformer model.
    
    This is a rough approximation based on standard transformer operations.
    """
    total_params, _ = count_parameters(model)
    
    # Rough estimate: 2 * params per forward pass (matrix multiply dominates)
    # Plus attention operations: 4 * d * seq_length for each layer
    
    flops_per_token = 2 * total_params
    
    # Try to estimate attention FLOPs if we can inspect the model
    if hasattr(model, 'layers') or hasattr(model, 'transformer'):
        try:
            # Get model dimensions
            if hasattr(model, 'config'):
                d_model = getattr(model.config, 'dim', 512)
                n_layers = getattr(model.config, 'layers', 6)
            else:
                # Fallback estimation
                d_model = 512
                n_layers = 6
                
            attention_flops = n_layers * 4 * d_model * seq_length
            flops_per_token += attention_flops
        except:
            # If we can't inspect, use the parameter-based estimate
            pass
    
    return flops_per_token
