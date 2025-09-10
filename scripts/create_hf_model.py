"""
Script to create and upload Arbor-500M-1B model to Hugging Face Hub.

This script:
1. Creates the model with the 500M-1B configuration
2. Saves it in HuggingFace format
3. Uploads to HF Hub with proper model card
4. Tests the uploaded model

Usage:
    python scripts/create_hf_model.py --upload --repo-name "your-username/arbor-500m-1b"
"""

import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, upload_folder
import getpass

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.arbor_500m_1b import (
    get_model_config, 
    get_model_card, 
    estimate_parameters
)
from arbor.transformers_integration import (
    ArborTransformersConfig,
    ArborForCausalLM
)


def create_model_files(output_dir: str):
    """
    Create all necessary files for HuggingFace model.
    
    Args:
        output_dir: Directory to save model files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🌱 Creating Arbor-500M-1B model in {output_dir}")
    
    # Get configuration
    arbor_config = get_model_config()
    model_card = get_model_card()
    params = estimate_parameters(arbor_config)
    
    print(f"📊 Model will have {params['base_parameters_millions']:.1f}M-{params['max_parameters_millions']:.1f}M parameters")
    
    # Create HuggingFace compatible config
    hf_config = ArborTransformersConfig(
        vocab_size=arbor_config.vocab_size,
        hidden_size=arbor_config.hidden_size,
        num_hidden_layers=arbor_config.num_layers,
        num_attention_heads=arbor_config.num_heads,
        intermediate_size=arbor_config.intermediate_size,
        max_position_embeddings=arbor_config.max_position_embeddings,
        
        # Growth parameters
        growth_factor=arbor_config.growth_factor,
        max_growth_steps=arbor_config.max_growth_steps,
        growth_threshold=arbor_config.growth_threshold,
        expandable_layers=getattr(arbor_config, 'expandable_layers', list(range(arbor_config.num_layers))),
        
        # Standard parameters
        hidden_dropout_prob=arbor_config.dropout,
        attention_probs_dropout_prob=arbor_config.dropout,
        layer_norm_eps=arbor_config.layer_norm_eps,
        initializer_range=arbor_config.initializer_range,
        use_cache=arbor_config.use_cache,
        
        # Token IDs
        pad_token_id=arbor_config.pad_token_id,
        bos_token_id=arbor_config.bos_token_id,
        eos_token_id=arbor_config.eos_token_id,
        
        # Model type
        model_type="arbor",
        torch_dtype="float16",
        
        # Custom metadata
        custom_metadata={
            "parameter_count": params,
            "growth_schedule": "adaptive",
            "training_framework": "arbor-o1",
            "version": "1.0.0"
        }
    )
    
    # Save configuration
    config_path = output_path / "config.json"
    hf_config.save_pretrained(output_path)
    print(f"✅ Saved config to {config_path}")
    
    # Create model with random weights
    print("🏗️  Creating model...")
    model = ArborForCausalLM(hf_config)
    
    # Initialize weights properly
    model.apply(model._init_weights)
    
    # Save model in safetensors format
    model_path = output_path / "model.safetensors"
    state_dict = model.state_dict()
    save_file(state_dict, model_path)
    print(f"✅ Saved model weights to {model_path} (safetensors format)")
    
    # Create tokenizer (using Hermes-4-405B tokenizer - Llama 3.1 405B based)
    print("📝 Setting up Hermes-4-405B tokenizer (Llama 3.1 405B based)...")
    try:
        # Use NousResearch/Hermes-4-405B tokenizer
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-4-405B")
        print("✅ Successfully loaded Hermes-4-405B tokenizer")
    except:
        # Fallback to Meta Llama 3.1 if Hermes isn't available
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
            print("✅ Using Meta Llama 3.1 tokenizer as fallback")
        except:
            # Final fallback - create a custom tokenizer config
            print("⚠️  Creating custom Llama 3.1-style tokenizer config...")
            tokenizer_config = {
                "tokenizer_class": "LlamaTokenizer",
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|end_of_text|>",
                "unk_token": "<unk>",
                "pad_token": "<|end_of_text|>",
                "vocab_size": 128000,
                "model_max_length": 131072,
                "chat_template": "{{- bos_token }}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}\n    {%- elif message['role'] == 'user' %}\n        {{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}"
            }
            
            # Save tokenizer config
            with open(output_path / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            
            print(f"✅ Created Hermes-style tokenizer config")
            tokenizer = None
    
    if tokenizer:
        # Save tokenizer (but not tokenizer.model)
        tokenizer.save_pretrained(output_path)
        # Remove tokenizer.model file if it exists
        tokenizer_model_path = output_path / "tokenizer.model"
        if tokenizer_model_path.exists():
            tokenizer_model_path.unlink()
            print(f"🗑️  Removed tokenizer.model file")
        print(f"✅ Saved tokenizer to {output_path}")
    
    # Create model card
    create_model_card(output_path, model_card, params, hf_config)
    
    # Create usage examples
    create_usage_examples(output_path)
    
    print(f"🎉 Model creation complete! Files saved to {output_path}")
    return output_path


def create_model_card(output_path: Path, model_card: dict, params: dict, config):
    """Create README.md model card for HuggingFace."""
    
    readme_content = f"""---
language: {model_card['language']}
license: {model_card['license']}
library_name: transformers
tags:
{chr(10).join(f'- {tag}' for tag in model_card['tags'])}
datasets:
{chr(10).join(f'- {dataset}' for dataset in model_card['datasets'])}
metrics:
{chr(10).join(f'- {metric}' for metric in model_card['metrics'])}
pipeline_tag: text-generation
widget:
- text: "The future of artificial intelligence is"
  example_title: "AI Future"
- text: "Once upon a time in a distant galaxy"
  example_title: "Creative Writing"
- text: "def fibonacci(n):"
  example_title: "Code Generation"
---

# {model_card['model_name'].title()}

{model_card['description']}

## Model Details

**Developed by:** Arbor AI Research  
**Model type:** Dynamic Growth Transformer  
**Language(s):** English  
**License:** Apache 2.0  
**Parameters:** {params['base_parameters_millions']:.1f}M - {params['max_parameters_millions']:.1f}M (dynamic)

### Key Features

🌱 **Dynamic Growth**: Model can expand from {params['base_parameters_millions']:.1f}M to {params['max_parameters_millions']:.1f}M parameters during training or inference  
🚀 **Efficient Training**: Starts small and grows only when needed  
🎯 **Adaptive Architecture**: Self-modifying neural network that optimizes its own structure  
⚡ **HuggingFace Compatible**: Works with standard transformers library  

## Architecture

- **Base Parameters:** {params['base_parameters_millions']:.1f}M
- **Maximum Parameters:** {params['max_parameters_millions']:.1f}M  
- **Layers:** {config.num_hidden_layers}
- **Hidden Size:** {config.hidden_size}
- **Attention Heads:** {config.num_attention_heads}
- **Context Length:** {config.max_position_embeddings}
- **Vocabulary Size:** {config.vocab_size}
- **Growth Factor:** {config.growth_factor}x
- **Max Growth Steps:** {config.max_growth_steps}

## Usage

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-username/arbor-500m-1b")
model = AutoModelForCausalLM.from_pretrained("your-username/arbor-500m-1b")

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.8)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### Dynamic Growth

```python
# Check current model size
print(f"Current parameters: {{model.num_parameters():,}}")

# Trigger growth (during training or when performance plateaus)
model.grow()
print(f"After growth: {{model.num_parameters():,}}")
```

### Advanced Usage

```python
# Generate with specific parameters
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)

# Use for chat
messages = [
    {{"role": "user", "content": "What is machine learning?"}}
]
# Format messages and generate response...
```

## Training Details

- **Training Data:** {', '.join(model_card['datasets'])}
- **Preprocessing:** BPE tokenization with GPT-2 vocabulary
- **Training Procedure:** Dynamic growth training with adaptive expansion
- **Hardware:** {model_card['training_details']['compute']}
- **Carbon Footprint:** {model_card['training_details']['carbon_footprint']}

### Growth Training

The model uses adaptive growth during training:
1. Starts with {params['base_parameters_millions']:.1f}M parameters
2. Monitors training metrics (loss, perplexity, gradient norms)
3. Expands FFN layers when performance plateaus
4. Can grow up to {config.max_growth_steps} times
5. Preserves learned knowledge during growth

## Intended Use

### Primary Use Cases
{chr(10).join(f'- {use_case}' for use_case in model_card['intended_use']['use_cases'])}

### Limitations
{chr(10).join(f'- {limitation}' for limitation in model_card['intended_use']['limitations'])}

## Evaluation

| Metric | Value |
|--------|-------|
| Perplexity (WikiText-2) | TBD |
| BLEU Score | TBD |
| Parameters (Base) | {params['base_parameters_millions']:.1f}M |
| Parameters (Max) | {params['max_parameters_millions']:.1f}M |

## Citation

```bibtex
@misc{{arbor-500m-1b,
  title={{Arbor-500M-1B: Dynamic Growth Transformer}},
  author={{Arbor AI Research}},
  year={{2025}},
  url={{https://huggingface.co/your-username/arbor-500m-1b}}
}}
```

## Model Card Contact

For questions about this model, please open an issue in the [Arbor repository](https://github.com/your-username/arbor-o1-living-ai).
"""

    with open(output_path / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Created model card: {output_path / 'README.md'}")


def create_usage_examples(output_path: Path):
    """Create example usage scripts."""
    
    # Simple usage example
    simple_example = '''"""
Simple usage example for Arbor-500M-1B model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Load model
    model_name = "your-username/arbor-500m-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    # Show current model size
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Trigger growth (if needed during inference)
    # model.grow()
    # print(f"After growth: {model.num_parameters():,}")

if __name__ == "__main__":
    main()
'''
    
    with open(output_path / "usage_example.py", "w") as f:
        f.write(simple_example)
    
    print(f"✅ Created usage example: {output_path / 'usage_example.py'}")


def upload_to_hub(model_path: str, repo_name: str, token: str = None):
    """Upload model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
        
        print(f"📤 Uploading to HuggingFace Hub: {repo_name}")
        
        # Create repository
        api = HfApi(token=token)
        try:
            create_repo(repo_name, token=token, exist_ok=True)
            print(f"✅ Repository created/verified: {repo_name}")
        except Exception as e:
            print(f"⚠️  Repository creation: {e}")
        
        # Upload files
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            token=token,
            commit_message="Initial Arbor-500M-1B model upload"
        )
        
        print(f"🎉 Successfully uploaded to https://huggingface.co/{repo_name}")
        
    except ImportError:
        print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ Upload failed: {e}")


def test_model(model_path: str):
    """Test the created model."""
    print(f"🧪 Testing model from {model_path}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Parameters: {model.num_parameters():,}")
        
        # Test generation
        prompt = "Hello, I am"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🎯 Test generation: '{generated}'")
        
        # Test growth
        initial_params = model.num_parameters()
        model.grow()
        after_growth = model.num_parameters()
        
        print(f"🌱 Growth test: {initial_params:,} → {after_growth:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create and upload Arbor-500M-1B to HuggingFace")
    parser.add_argument("--output-dir", default="./arbor-500m-1b-hf", help="Output directory")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace Hub")
    parser.add_argument("--repo-name", help="HuggingFace repository name (e.g., username/model-name)")
    parser.add_argument("--token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--test-only", help="Test existing model directory")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_model(args.test_only)
        return
    
    # Create model files
    model_path = create_model_files(args.output_dir)
    
    # Test the model
    if test_model(str(model_path)):
        print("✅ Model test passed!")
    else:
        print("❌ Model test failed!")
        return
    
    # Upload if requested
    if args.upload:
        if not args.repo_name:
            print("❌ --repo-name required for upload")
            return
        
        token = args.token or os.getenv("HF_TOKEN")
        if not token:
            print("❌ HuggingFace token required. Set --token or HF_TOKEN env var")
            return
        
        upload_to_hub(str(model_path), args.repo_name, token)
    
    print(f"🌱 Arbor-500M-1B ready! Next steps:")
    print(f"1. Test locally: python {model_path}/usage_example.py")
    if not args.upload:
        print(f"2. Upload: python {__file__} --upload --repo-name your-username/arbor-500m-1b --token YOUR_TOKEN")
    print(f"3. Share and use: https://huggingface.co/{args.repo_name if args.repo_name else 'your-username/arbor-500m-1b'}")


if __name__ == "__main__":
    main()
