"""
Quick setup script to create HuggingFace-ready Arbor model.

This creates a production-ready 500M-1B parameter model for HuggingFace Hub.
"""

import json
import torch
from pathlib import Path


def create_hf_config():
    """Create HuggingFace-compatible config.json"""
    config = {
        "_name_or_path": "arbor-500m-1b",
        "architectures": ["ArborForCausalLM"],
        "model_type": "arbor",
        
        # Model architecture
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 131072,
        
        # RoPE scaling for long context
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "linear",
            "factor": 32.0
        },
        
        # Growth parameters (Arbor-specific)
        "growth_factor": 2.0,
        "max_growth_steps": 8,
        "growth_threshold": 0.95,
        "expandable_layers": [6, 12, 18],
        
        # Standard transformer params
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "layer_norm_eps": 1e-5,
        "initializer_range": 0.02,
        "use_cache": True,
        
        # Token IDs
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        
        # Type and precision
        "torch_dtype": "float16",
        "transformers_version": "4.35.0",
        
        # Long context settings
        "use_sliding_window": False,
        "sliding_window": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        
        # Custom metadata
        "custom_metadata": {
            "base_parameters": "420M",
            "max_parameters": "1100M", 
            "growth_potential": "680M",
            "training_framework": "arbor-o1",
            "version": "1.0.0",
            "dynamic_growth": True,
            "tokenizer_type": "llama",
            "context_scaling": "rope_linear",
            "max_context_demo": "4K",
            "max_context_supported": "128K",
            "efficient_attention": True
        }
    }
    return config


def create_simple_readme():
    """Create a simple README for the model."""
    readme = """---
language: en
license: apache-2.0
library_name: transformers
tags:
- arbor
- dynamic-growth
- transformer
- causal-lm
- text-generation
pipeline_tag: text-generation
widget:
- text: "The future of artificial intelligence is"
  example_title: "AI Future"
- text: "Once upon a time"
  example_title: "Creative Writing"
---

# Arbor-500M-1B Dynamic Growth Model

A dynamic growth transformer that starts at 407M parameters and can expand to 1.2B parameters.

## Features

🌱 **Dynamic Growth**: Expands from 407M to 1.2B parameters as needed  
🚀 **Efficient**: Starts small, grows only when beneficial  
⚡ **HF Compatible**: Works with standard transformers library  

## Quick Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("your-username/arbor-500m-1b") 
model = AutoModelForCausalLM.from_pretrained("your-username/arbor-500m-1b")

# Generate text
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Dynamic growth
model.grow()  # Expand model when needed
```

## Model Details

- **Base Parameters**: 407M
- **Max Parameters**: 1.2B  
- **Context Length**: 2048
- **Vocabulary**: 50,257 (GPT-2 compatible)
- **Architecture**: 24 layers, 1024 hidden size, 16 heads

## Architecture

The Arbor architecture uses expandable feed-forward networks that can grow during training or inference when performance plateaus, allowing the model to adapt its capacity to the complexity of the task.
"""
    return readme


def main():
    """Create HuggingFace model files."""
    output_dir = Path("./arbor-500m-1b-hf")
    output_dir.mkdir(exist_ok=True)
    
    print("🌱 Creating Arbor-500M-1B for HuggingFace Hub")
    print("=" * 50)
    
    # Create config
    config = create_hf_config()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Created config.json")
    
    # Create README
    readme = create_simple_readme()
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)
    print(f"✅ Created README.md")
    
    # Create tokenizer config (Llama compatible)
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>", 
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "model_max_length": 131072,
        "tokenizer_class": "LlamaTokenizer",
        "legacy": False,
        "use_default_system_prompt": False
    }
    
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✅ Created tokenizer_config.json")
    
    # Create generation config
    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_length": 131072,
        "max_new_tokens": 4096,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    }
    
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)
    print(f"✅ Created generation_config.json")
    
    print()
    print("📁 Created files:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")
    
    print()
    print("📝 Next steps:")
    print("1. Add your trained model weights as 'pytorch_model.bin'")
    print("2. Add Llama tokenizer files (tokenizer.model, tokenizer.json)")
    print("3. Upload to HuggingFace Hub:")
    print("   huggingface-cli upload your-username/arbor-500m-1b ./arbor-500m-1b-hf")
    
    print()
    print("🔗 Or use the full creation script:")
    print("   python scripts/create_hf_model.py --upload --repo-name your-username/arbor-500m-1b")


if __name__ == "__main__":
    main()
