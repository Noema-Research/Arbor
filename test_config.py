"""
Test script for Arbor 500M-1B configuration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now import the config
try:
    from configs.arbor_500m_1b import (
        ARBOR_500M_1B_CONFIG,
        GROWTH_CONFIG,
        TRAINING_CONFIG,
        estimate_parameters
    )
    
    print("🌱 Arbor-500M-1B Model Configuration")
    print("=" * 50)
    
    # Show parameter estimates
    params = estimate_parameters()
    
    print(f"Base Parameters: {params['base_parameters_millions']:.1f}M")
    print(f"Max Parameters: {params['max_parameters_millions']:.1f}M") 
    print(f"Growth Potential: {params['growth_potential'] / 1_000_000:.1f}M")
    print()
    print(f"Architecture: {ARBOR_500M_1B_CONFIG.num_layers} layers, {ARBOR_500M_1B_CONFIG.hidden_size} hidden")
    print(f"Context Length: {ARBOR_500M_1B_CONFIG.max_position_embeddings}")
    print(f"Vocabulary: {ARBOR_500M_1B_CONFIG.vocab_size}")
    print(f"Growth Factor: {ARBOR_500M_1B_CONFIG.growth_factor}x")
    print(f"Max Growth Steps: {ARBOR_500M_1B_CONFIG.max_growth_steps}")
    print()
    print("Parameter Breakdown:")
    for component, count in params['breakdown'].items():
        print(f"  {component}: {count / 1_000_000:.1f}M")
    
    print()
    print("🔧 Training Configuration:")
    print(f"  Batch Size: {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"  Mixed Precision: {TRAINING_CONFIG['fp16']}")
    
    print()
    print("🌱 Growth Configuration:")
    print(f"  Strategy: {GROWTH_CONFIG.strategy}")
    print(f"  Max Total Growth: {GROWTH_CONFIG.max_total_growth}x")
    print(f"  Preserve Weights: {GROWTH_CONFIG.preserve_weights}")
    
    print()
    print("✅ Configuration loaded successfully!")
    print()
    print("📝 To create HuggingFace model:")
    print("   python scripts/create_hf_model.py")
    print()
    print("📤 To upload to HuggingFace:")
    print("   python scripts/create_hf_model.py --upload --repo-name username/arbor-500m-1b")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Creating standalone configuration...")
    
    # Standalone configuration classes
    class ArborConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class GrowthConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Create config
    config = ArborConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
        max_position_embeddings=131072,
        rope_theta=10000.0,
        rope_scaling={"type": "linear", "factor": 32.0},
        growth_factor=2.0,
        max_growth_steps=8,
        growth_threshold=0.95,
        dropout=0.1,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        gradient_checkpointing=True,
        use_flash_attention=True,
        model_type="arbor",
        tie_word_embeddings=False,
        torch_dtype="float16",
    )
    
    # Calculate parameters
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    intermediate_size = config.intermediate_size
    max_position = config.max_position_embeddings
    
    # Parameter calculation
    token_embeddings = vocab_size * hidden_size
    position_embeddings = max_position * hidden_size
    
    attention_params = 4 * hidden_size * hidden_size + 4 * hidden_size
    ffn_params = 2 * hidden_size * intermediate_size + intermediate_size + hidden_size
    layer_norm_params = 2 * hidden_size
    layer_params = attention_params + ffn_params + layer_norm_params
    
    output_params = vocab_size * hidden_size
    
    base_params = (
        token_embeddings + position_embeddings + 
        num_layers * layer_params + output_params + hidden_size
    )
    
    # Growth calculation
    growth_factor = config.growth_factor
    ffn_growth_per_layer = hidden_size * intermediate_size * (growth_factor - 1)
    max_growth_params = num_layers * ffn_growth_per_layer * config.max_growth_steps
    
    print("🌱 Arbor-500M-1B Model Configuration")
    print("=" * 50)
    print(f"Base Parameters: {base_params / 1_000_000:.1f}M")
    print(f"Max Parameters: {(base_params + max_growth_params) / 1_000_000:.1f}M")
    print(f"Growth Potential: {max_growth_params / 1_000_000:.1f}M")
    print()
    print(f"Architecture: {config.num_layers} layers, {config.hidden_size} hidden")
    print(f"Context Length: {config.max_position_embeddings:,} (128K)")
    print(f"RoPE Scaling: {config.rope_scaling}")
    print(f"Vocabulary: {config.vocab_size}")
    print(f"Growth Factor: {config.growth_factor}x")
    print(f"Max Growth Steps: {config.max_growth_steps}")
    print()
    print("Parameter Breakdown:")
    print(f"  Embeddings: {(token_embeddings + position_embeddings) / 1_000_000:.1f}M")
    print(f"  Transformer layers: {(num_layers * layer_params) / 1_000_000:.1f}M")
    print(f"  Output layer: {output_params / 1_000_000:.1f}M")
    print(f"  Growth potential: {max_growth_params / 1_000_000:.1f}M")
    print()
    print("✅ Configuration created successfully!")
