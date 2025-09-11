"""
Enterprise-Scale Arbor Architecture for 200B-400B Parameter Models.

This module implements the full-scale Arbor architecture designed for
enterprise deployment with distributed training and inference.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import math

from .model import ArborConfig, ArborTransformer
from .layers import ExpandableFFN
from .block import ArborBlock


@dataclass
class EnterpriseArborConfig(ArborConfig):
    """Extended configuration for enterprise-scale Arbor models (200B-400B parameters)."""
    
    # Enterprise scale architecture
    vocab_size: int = 128000                    # Hermes-4-405B vocabulary
    dim: int = 16384                           # Large hidden dimension
    num_layers: int = 120                      # Deep architecture
    num_heads: int = 128                       # High attention head count
    ffn_dim: int = 65536                       # Large FFN for capacity
    max_seq_length: int = 2097152              # 2M context length
    
    # Advanced attention mechanisms
    num_key_value_heads: int = 16              # Grouped-query attention
    rope_theta: float = 500000.0               # Extended RoPE base
    rope_scaling: Optional[Dict] = None        # RoPE scaling configuration
    attention_bias: bool = False               # Bias-free attention
    
    # Enterprise growth settings
    growth_enabled: bool = True
    growth_factor: float = 4.0                 # Higher growth factor
    max_growth_factor: float = 8.0             # Maximum total growth
    target_params: int = 400_000_000_000       # 400B parameter target
    
    # Distributed training configuration
    tensor_parallel_size: int = 8              # Tensor parallelism
    pipeline_parallel_size: int = 16           # Pipeline parallelism
    data_parallel_size: int = 32               # Data parallelism
    
    # Memory optimization
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = True
    cpu_offload: bool = True
    parameter_sharding: bool = True
    
    # Advanced features
    mixture_of_experts: bool = False           # Future: MoE support
    num_experts: int = 64                      # Number of experts
    expert_capacity: float = 1.25              # Expert capacity factor
    
    # Efficiency optimizations
    flash_attention: bool = True               # Flash attention
    fused_kernels: bool = True                 # Fused CUDA kernels
    bf16_training: bool = True                 # BF16 mixed precision
    
    def __post_init__(self):
        """Validate enterprise configuration."""
        super().__post_init__()
        
        # Validate distributed setup
        total_gpus = self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        assert total_gpus >= 64, f"Enterprise model requires 64+ GPUs, got {total_gpus}"
        
        # Validate model size
        estimated_params = self.estimate_parameters()
        assert estimated_params >= 100_000_000_000, f"Too small for enterprise: {estimated_params/1e9:.1f}B params"
        
    def estimate_parameters(self) -> int:
        """Estimate total parameter count."""
        # Attention parameters
        attention_params = self.num_layers * self.dim * self.dim * 4  # Q, K, V, O
        
        # FFN parameters (with potential MoE)
        if self.mixture_of_experts:
            ffn_params = self.num_layers * self.dim * self.ffn_dim * 2 * self.num_experts
        else:
            ffn_params = self.num_layers * self.dim * self.ffn_dim * 2
            
        # Embedding parameters
        embedding_params = self.vocab_size * self.dim * 2  # Input + output
        
        # Layer norm parameters
        norm_params = self.num_layers * self.dim * 4  # 2 per layer, before attention and FFN
        
        total = attention_params + ffn_params + embedding_params + norm_params
        return total


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention for efficient large-scale models."""
    
    def __init__(self, config: EnterpriseArborConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.dim // self.num_heads
        
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        assert self.num_heads % self.num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        
        self.num_queries_per_kv = self.num_heads // self.num_key_value_heads
        
        # Projections for grouped-query attention
        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.dim, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.dim, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=config.attention_bias)
        
        # RoPE embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_seq_length,
            base=config.rope_theta
        )
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Expand K, V for grouped-query attention
        key_states = repeat_kv(key_states, self.num_queries_per_kv)
        value_states = repeat_kv(value_states, self.num_queries_per_kv)
        
        # Attention computation
        if self.config.flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use flash attention if available
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, 
                attn_mask=attention_mask, dropout_p=0.0, is_causal=True
            )
        else:
            # Standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class MixtureOfExperts(nn.Module):
    """Mixture of Experts for enterprise-scale efficiency."""
    
    def __init__(self, config: EnterpriseArborConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.dim = config.dim
        self.ffn_dim = config.ffn_dim
        
        # Router for expert selection
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpandableFFN(
                dim=self.dim,
                ffn_dim=self.ffn_dim,
                dropout=config.ffn_dropout,
                activation=config.activation
            ) for _ in range(self.num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, dim)  # Flatten for routing
        
        # Router logits
        router_logits = self.router(hidden_states)
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        
        # Top-k expert selection (top-2 for load balancing)
        top_k = 2
        top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Route to experts
        final_output = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            expert_tokens = hidden_states[expert_mask]
            
            if expert_tokens.numel() > 0:
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)
                
                # Weight by router probabilities
                expert_weights = top_k_probs[expert_mask]
                expert_weights = expert_weights[top_k_indices[expert_mask] == expert_idx]
                expert_output = expert_output * expert_weights.unsqueeze(-1)
                
                # Add to final output
                final_output[expert_mask] += expert_output
        
        return final_output.view(batch_size, seq_len, dim)


class EnterpriseArborBlock(ArborBlock):
    """Enhanced Arbor block for enterprise scale with advanced features."""
    
    def __init__(self, config: EnterpriseArborConfig):
        # Initialize base block with enterprise config
        super().__init__(
            dim=config.dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            ffn_dropout=config.ffn_dropout,
            layer_norm_eps=config.layer_norm_eps,
            activation=config.activation,
            causal=True
        )
        
        self.config = config
        
        # Replace attention with grouped-query attention
        self.attention = GroupedQueryAttention(config)
        
        # Replace FFN with MoE if enabled
        if config.mixture_of_experts:
            self.ffn = MixtureOfExperts(config)
        
        # Gradient checkpointing support
        self.gradient_checkpointing = config.gradient_checkpointing


class EnterpriseArborTransformer(nn.Module):
    """Enterprise-scale Arbor transformer for 200B-400B parameters."""
    
    def __init__(self, config: EnterpriseArborConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EnterpriseArborBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        
        # Output projection
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
        # Distributed setup if needed
        if dist.is_initialized():
            self._setup_distributed()
    
    def _init_weights(self, module):
        """Initialize weights for enterprise scale."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _setup_distributed(self):
        """Setup distributed training configuration."""
        # Tensor parallelism for attention and FFN
        if self.config.tensor_parallel_size > 1:
            self._apply_tensor_parallelism()
            
        # Pipeline parallelism for layers
        if self.config.pipeline_parallel_size > 1:
            self._apply_pipeline_parallelism()
    
    def _apply_tensor_parallelism(self):
        """Apply tensor parallelism to model components."""
        # This would involve sharding attention and FFN weights
        # across tensor parallel ranks
        pass  # Implementation depends on specific TP library
    
    def _apply_pipeline_parallelism(self):
        """Apply pipeline parallelism to transformer layers."""
        # This would involve distributing layers across pipeline ranks
        pass  # Implementation depends on specific PP library
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through transformer layers
        all_hidden_states = []
        all_attentions = []
        next_cache = []
        
        for i, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training:
                # Use gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids)
            
            if self.config.activation_checkpointing:
                all_hidden_states.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': all_hidden_states if self.config.activation_checkpointing else None,
            'attentions': all_attentions,
            'past_key_values': next_cache if use_cache else None
        }
    
    def count_parameters(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def memory_footprint(self) -> Dict[str, int]:
        """Calculate memory footprint breakdown."""
        footprint = {}
        
        for name, module in self.named_modules():
            if hasattr(module, 'parameters'):
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    footprint[name] = module_params * 4  # Assume FP32, 4 bytes per param
        
        return footprint


# Helper functions for enterprise features

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for long context support."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key and value tensors n_rep times for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Model size configurations for different scales

def create_enterprise_config(target_params: int = 400_000_000_000) -> EnterpriseArborConfig:
    """Create enterprise configuration for target parameter count."""
    
    if target_params <= 200_000_000_000:  # 200B
        return EnterpriseArborConfig(
            dim=12288,
            num_layers=96,
            num_heads=96,
            ffn_dim=49152,
            num_key_value_heads=12,
            tensor_parallel_size=8,
            pipeline_parallel_size=8,
            data_parallel_size=4
        )
    else:  # 400B
        return EnterpriseArborConfig(
            dim=16384,
            num_layers=120,
            num_heads=128,
            ffn_dim=65536,
            num_key_value_heads=16,
            tensor_parallel_size=8,
            pipeline_parallel_size=16,
            data_parallel_size=4
        )


# Factory function for creating enterprise models

def create_enterprise_arbor(target_params: int = 400_000_000_000) -> EnterpriseArborTransformer:
    """Create enterprise Arbor model with specified parameter count."""
    config = create_enterprise_config(target_params)
    model = EnterpriseArborTransformer(config)
    
    actual_params = model.count_parameters()
    print(f"🏢 Created Enterprise Arbor model: {actual_params/1e9:.1f}B parameters")
    print(f"🎯 Target: {target_params/1e9:.1f}B, Actual: {actual_params/1e9:.1f}B")
    print(f"📊 Architecture: {config.num_layers} layers, {config.dim} hidden, {config.num_heads} heads")
    
    return model
