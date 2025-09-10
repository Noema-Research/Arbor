"""
Arbor-o1 Configuration for 500M-1B Parameter Growth Model

This configuration defines a model that starts at ~500M parameters and can grow 
to ~1B parameters through dynamic expansion. Optimized for Hugging Face deployment.
"""

from arbor.modeling.model import ArborConfig
from arbor.growth.manager import GrowthConfig


# Base model configuration (~500M parameters)
ARBOR_500M_1B_CONFIG = ArborConfig(
    # Model architecture
    vocab_size=128000,   # Hermes-4-405B vocabulary size (Llama 3.1 405B based)
    hidden_size=1024,   # Large enough for quality but efficient
    num_layers=24,      # Good depth for reasoning
    num_heads=16,       # Multi-head attention
    intermediate_size=4096,  # FFN size (4x hidden_size)
    
    # Position embeddings
    max_position_embeddings=131072,  # Support up to 128K tokens
    rope_theta=10000.0,              # RoPE base frequency
    rope_scaling={                   # Linear scaling for long context
        "type": "linear",
        "factor": 32.0
    },
    
    # Growth parameters
    growth_factor=2.0,      # Can double FFN size when growing
    max_growth_steps=8,     # Allow multiple growth steps
    growth_threshold=0.95,  # High threshold for careful growth
    
    # Training efficiency
    dropout=0.1,
    layer_norm_eps=1e-5,
    initializer_range=0.02,
    
    # Generation settings
    use_cache=True,
    pad_token_id=0,      # <pad> token
    bos_token_id=1,      # <s> token  
    eos_token_id=2,      # </s> token
    
    # Efficiency settings
    gradient_checkpointing=True,  # Save memory during training
    use_flash_attention=True,     # Faster attention if available
    use_sliding_window=False,     # Disable sliding window for full context
    attention_dropout=0.0,        # No attention dropout for stability
    
    # Long context optimizations
    efficient_attention=True,     # Use efficient attention implementations
    context_scaling="rope_linear", # How to scale to longer contexts
    
    # Model identification
    model_type="arbor",
    architectures=["ArborForCausalLM"],
    
    # HuggingFace compatibility
    tie_word_embeddings=False,
    torch_dtype="float16",  # Use half precision for efficiency
    
    # Custom Arbor parameters
    expandable_layers=[6, 12, 18],  # Which layers can expand
    growth_schedule="adaptive",      # Adaptive growth based on loss
    preserve_performance=True,       # Maintain performance during growth
)


# Growth configuration for training
GROWTH_CONFIG = GrowthConfig(
    # Growth triggers
    triggers={
        "loss_plateau": {
            "patience": 5,
            "min_delta": 0.01,
            "monitor": "train_loss"
        },
        "gradient_norm": {
            "threshold": 0.1,
            "window_size": 100
        },
        "perplexity": {
            "threshold": 15.0,
            "eval_steps": 1000
        }
    },
    
    # Growth strategy
    strategy="conservative",  # Careful, measured growth
    max_total_growth=2.0,     # Don't exceed 2x original size
    
    # Growth mechanics
    preserve_weights=True,
    reinitialize_new=True,
    learning_rate_schedule="warmup",
    
    # Validation
    validation_steps=500,
    growth_validation=True,
    rollback_on_failure=True,
)


# Training configuration for HF Trainer
TRAINING_CONFIG = {
    "output_dir": "./arbor-500m-1b-checkpoints",
    "overwrite_output_dir": True,
    
    # Training hyperparameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    
    # Learning rate schedule
    "lr_scheduler_type": "cosine",
    "warmup_steps": 1000,
    
    # Efficiency
    "fp16": True,
    "dataloader_num_workers": 4,
    "remove_unused_columns": False,
    
    # Logging and saving
    "logging_steps": 100,
    "save_steps": 2000,
    "eval_steps": 1000,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    # Evaluation
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    
    # Growth-specific
    "growth_check_steps": 2000,
    "growth_patience": 3,
    "growth_warmup_steps": 5000,
}


# Model card information for HuggingFace Hub
MODEL_CARD = {
    "model_name": "arbor-500m-1b",
    "description": "Arbor-o1 dynamic growth transformer (500M-1B parameters)",
    "tags": [
        "arbor",
        "dynamic-growth", 
        "transformer",
        "causal-lm",
        "text-generation"
    ],
    "language": "en",
    "license": "apache-2.0",
    "datasets": ["openwebtext", "bookcorpus", "c4"],
    "metrics": ["perplexity"],
    
    "model_details": {
        "architecture": "Arbor Dynamic Growth Transformer",
        "parameters": "420M-1.1B (dynamic)",
        "context_length": "4K-128K (scalable)",
        "vocabulary_size": 32000,
        "precision": "fp16",
        "tokenizer": "llama",
        "position_encoding": "rope_linear_scaling",
    },
    
    "intended_use": {
        "primary_use": "Text generation and completion",
        "use_cases": [
            "Creative writing assistance",
            "Code completion", 
            "Conversational AI",
            "Research in dynamic neural architectures"
        ],
        "limitations": [
            "May require growth during deployment for optimal performance",
            "English language primarily",
            "Not suitable for factual question answering without fine-tuning"
        ]
    },
    
    "training_details": {
        "training_data": "Filtered web text, books, and Common Crawl",
        "preprocessing": "SentencePiece tokenization with Llama vocabulary",
        "compute": "8x A100 GPUs",
        "carbon_footprint": "Estimated 50kg CO2 equivalent"
    }
}


def get_model_config():
    """Get the base model configuration."""
    return ARBOR_500M_1B_CONFIG


def get_growth_config():
    """Get the growth training configuration."""
    return GROWTH_CONFIG


def get_training_args():
    """Get HuggingFace training arguments."""
    return TRAINING_CONFIG


def get_model_card():
    """Get model card information for HF Hub."""
    return MODEL_CARD


def estimate_parameters(config=None):
    """
    Estimate parameter count for the model configuration.
    
    Returns:
        dict: Parameter counts at different growth stages
    """
    if config is None:
        config = ARBOR_500M_1B_CONFIG
    
    # Base parameter calculation
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    intermediate_size = config.intermediate_size
    max_position = config.max_position_embeddings
    
    # Embedding parameters
    token_embeddings = vocab_size * hidden_size
    position_embeddings = max_position * hidden_size
    
    # Transformer layer parameters (per layer)
    attention_params = (
        4 * hidden_size * hidden_size +  # Q, K, V, O projections
        4 * hidden_size                  # Biases
    )
    
    ffn_params = (
        2 * hidden_size * intermediate_size +  # Up and down projections
        intermediate_size + hidden_size        # Biases
    )
    
    layer_norm_params = 2 * hidden_size  # Pre-attention and pre-FFN
    
    layer_params = attention_params + ffn_params + layer_norm_params
    
    # Output layer
    output_params = vocab_size * hidden_size
    
    # Total base parameters
    base_params = (
        token_embeddings + 
        position_embeddings + 
        num_layers * layer_params + 
        output_params +
        hidden_size  # Final layer norm
    )
    
    # Growth calculations
    expandable_ffn_layers = len(config.expandable_layers) if hasattr(config, 'expandable_layers') else num_layers
    growth_factor = config.growth_factor
    
    # Parameters added per growth step (only FFN layers grow)
    ffn_growth_per_layer = hidden_size * intermediate_size * (growth_factor - 1)
    max_growth_params = expandable_ffn_layers * ffn_growth_per_layer * config.max_growth_steps
    
    return {
        "base_parameters": base_params,
        "max_parameters": base_params + max_growth_params,
        "growth_potential": max_growth_params,
        "base_parameters_millions": base_params / 1_000_000,
        "max_parameters_millions": (base_params + max_growth_params) / 1_000_000,
        "breakdown": {
            "embeddings": token_embeddings + position_embeddings,
            "transformer_layers": num_layers * layer_params,
            "output_layer": output_params,
            "potential_growth": max_growth_params
        }
    }


if __name__ == "__main__":
    # Print configuration summary
    config = get_model_config()
    params = estimate_parameters(config)
    
    print("🌱 Arbor-500M-1B Model Configuration")
    print("=" * 50)
    print(f"Base Parameters: {params['base_parameters_millions']:.1f}M")
    print(f"Max Parameters: {params['max_parameters_millions']:.1f}M")
    print(f"Growth Potential: {params['growth_potential'] / 1_000_000:.1f}M")
    print()
    print(f"Architecture: {config.num_layers} layers, {config.hidden_size} hidden")
    print(f"Context Length: {config.max_position_embeddings}")
    print(f"Vocabulary: {config.vocab_size}")
    print(f"Growth Factor: {config.growth_factor}x")
    print(f"Max Growth Steps: {config.max_growth_steps}")
    print()
    print("Parameter Breakdown:")
    for component, count in params['breakdown'].items():
        print(f"  {component}: {count / 1_000_000:.1f}M")
