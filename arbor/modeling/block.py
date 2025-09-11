"""
Transformer block with expandable components.
"""

from typing import Optional
import torch
import torch.nn as nn
import math

from .layers import ExpandableFFN


class ArborBlock(nn.Module):
    """
    Transformer block with expandable feed-forward network.
    
    Components:
    - Multi-head self-attention
    - Layer normalization  
    - Expandable feed-forward network
    - Residual connections
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
        causal: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = dim // num_heads
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=True,
            batch_first=True,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        # Expandable feed-forward network
        self.ffn = ExpandableFFN(dim, ffn_dim, ffn_dropout, activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask for autoregressive generation
        self.causal = causal
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal attention (overrides self.causal)
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # Handle causal attention
        use_causal = is_causal if is_causal is not None else self.causal
        
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=attention_mask,
            is_causal=use_causal and attention_mask is None,
        )
        
        x = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x
    
    def grow_ffn(self, add_hidden: int) -> None:
        """
        Expand the feed-forward network hidden dimension.
        
        Args:
            add_hidden: Number of hidden units to add
        """
        old_ffn_dim = self.ffn_dim
        self.ffn.grow(add_hidden)
        self.ffn_dim = self.ffn.hidden_dim
        print(f"ArborBlock FFN grown: {old_ffn_dim} -> {self.ffn_dim}")
    
    def grow_attention(self, add_heads: int) -> None:
        """
        Expand the number of attention heads.
        
        Args:
            add_heads: Number of attention heads to add
            
        Note: This requires careful weight initialization and preservation.
        """
        if add_heads <= 0:
            return
            
        old_num_heads = self.num_heads
        new_num_heads = old_num_heads + add_heads
        
        # Calculate dimensions
        head_dim = self.dim // old_num_heads
        new_dim_per_layer = new_num_heads * head_dim
        
        # Store old weights
        old_qkv = self.attention.qkv_proj
        old_output = self.attention.output_proj
        
        # Create new larger layers
        new_qkv = nn.Linear(self.dim, new_dim_per_layer * 3, bias=old_qkv.bias is not None)
        new_output = nn.Linear(new_dim_per_layer, self.dim, bias=old_output.bias is not None)
        
        # Copy existing weights to preserve learned representations
        with torch.no_grad():
            # For QKV projection: copy existing weights and initialize new heads
            old_qkv_dim = old_num_heads * head_dim
            
            # Q weights
            new_qkv.weight[:old_qkv_dim, :].copy_(old_qkv.weight[:old_qkv_dim, :])
            # K weights  
            new_qkv.weight[new_dim_per_layer:new_dim_per_layer+old_qkv_dim, :].copy_(
                old_qkv.weight[old_qkv_dim:old_qkv_dim*2, :])
            # V weights
            new_qkv.weight[new_dim_per_layer*2:new_dim_per_layer*2+old_qkv_dim, :].copy_(
                old_qkv.weight[old_qkv_dim*2:, :])
            
            # Initialize new heads with small random values
            nn.init.normal_(new_qkv.weight[old_qkv_dim:new_dim_per_layer, :], std=0.02)
            nn.init.normal_(new_qkv.weight[new_dim_per_layer+old_qkv_dim:new_dim_per_layer*2, :], std=0.02)
            nn.init.normal_(new_qkv.weight[new_dim_per_layer*2+old_qkv_dim:, :], std=0.02)
            
            # Copy bias if present
            if old_qkv.bias is not None:
                new_qkv.bias[:old_qkv_dim].copy_(old_qkv.bias[:old_qkv_dim])
                new_qkv.bias[new_dim_per_layer:new_dim_per_layer+old_qkv_dim].copy_(
                    old_qkv.bias[old_qkv_dim:old_qkv_dim*2])
                new_qkv.bias[new_dim_per_layer*2:new_dim_per_layer*2+old_qkv_dim].copy_(
                    old_qkv.bias[old_qkv_dim*2:])
                
            # For output projection: pad with zeros for new heads
            new_output.weight[:, :old_qkv_dim].copy_(old_output.weight)
            nn.init.zeros_(new_output.weight[:, old_qkv_dim:])  # Initialize new head outputs to zero
            
            if old_output.bias is not None:
                new_output.bias.copy_(old_output.bias)
        
        # Replace the layers
        self.attention.qkv_proj = new_qkv
        self.attention.output_proj = new_output
        self.num_heads = new_num_heads
        
        print(f"🧠 Expanded attention heads: {old_num_heads} → {new_num_heads}")
    
    def param_count(self) -> int:
        """Return total number of parameters in this block."""
        return sum(p.numel() for p in self.parameters())
    
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, num_heads={self.num_heads}, "
                f"ffn_dim={self.ffn_dim}, causal={self.causal}")


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """
    
    def __init__(self, dim: int, max_seq_length: int = 5000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, dim)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * 
            -(math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer models.
    
    This is a more advanced positional encoding that has shown
    better performance on long sequences.
    """
    
    def __init__(self, dim: int, max_seq_length: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Create rotation matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rotation matrices for the given sequence length.
        
        Returns:
            Tuple of (cos, sin) matrices for rotary encoding
        """
        if seq_len is None:
            seq_len = x.size(-2)
            
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        
        return cos, sin
