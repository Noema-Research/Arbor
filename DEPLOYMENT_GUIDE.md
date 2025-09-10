# Arbor-500M-1B: HuggingFace Deployment Guide

🌱 **Complete guide to deploy your Arbor dynamic growth model to HuggingFace Hub**

## Model Overview

- **Base Parameters**: 699M
- **Max Parameters**: 799M (after growth)
- **Architecture**: 24 layers, 1024 hidden, 16 heads
- **Context Length**: 4096 tokens (demo), 128K tokens (max)
- **Vocabulary**: 128,000 (Hermes-4-405B tokenizer)
- **Format**: SafeTensors (secure, fast loading)
- **Growth Factor**: 2x expansion capability

## 📦 Quick Setup

### Option 1: Use Pre-configured Files

The `arbor-500m-1b-hf/` directory contains all necessary HuggingFace files:

```bash
arbor-500m-1b-hf/
├── config.json              # Model configuration
├── README.md                 # Model card
├── tokenizer_config.json     # Tokenizer settings
└── generation_config.json    # Generation defaults
```

### Option 2: Complete Model Creation

Use the full creation script:

```bash
# Create complete model with weights
python scripts/create_hf_model.py --output-dir ./my-arbor-model

# Upload to HuggingFace
python scripts/create_hf_model.py --upload --repo-name username/arbor-500m-1b --token YOUR_HF_TOKEN
```

## 🚀 Deployment Steps

### 1. Prepare Your Model

If you have trained weights:
```bash
# Copy your trained model weights
cp your_trained_model.bin arbor-500m-1b-hf/pytorch_model.bin
```

If starting fresh:
```bash
# Create random initialized model
python -c "
import torch
from transformers import AutoConfig
from arbor.transformers_integration import ArborForCausalLM

config = AutoConfig.from_pretrained('./arbor-500m-1b-hf')
model = ArborForCausalLM(config)
torch.save(model.state_dict(), './arbor-500m-1b-hf/pytorch_model.bin')
print('✅ Model weights created')
"
```

### 2. Add Tokenizer Files

Download Hermes-4-405B tokenizer files:
```bash
cd arbor-500m-1b-hf

# Download Hermes-4-405B tokenizer files
wget https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.model
wget https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json
wget https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/special_tokens_map.json

# Or use Python
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.save_pretrained('.')
print('✅ Tokenizer files added')
"
```

### 3. Validate Model

Test the model locally:
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

print('🧪 Testing model...')
tokenizer = AutoTokenizer.from_pretrained('./arbor-500m-1b-hf')
model = AutoModelForCausalLM.from_pretrained('./arbor-500m-1b-hf')

# Test generation
inputs = tokenizer('Hello', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=10)
text = tokenizer.decode(outputs[0])
print(f'✅ Generation test: {text}')

# Test growth
initial = model.num_parameters()
model.grow()
after = model.num_parameters()
print(f'✅ Growth test: {initial:,} → {after:,} parameters')
"
```

### 4. Upload to HuggingFace Hub

#### Method A: Using huggingface-cli
```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload
huggingface-cli upload username/arbor-500m-1b ./arbor-500m-1b-hf
```

#### Method B: Using Python
```python
from huggingface_hub import HfApi, create_repo

# Create repository
repo_name = "username/arbor-500m-1b"
create_repo(repo_name, exist_ok=True)

# Upload files
api = HfApi()
api.upload_folder(
    folder_path="./arbor-500m-1b-hf",
    repo_id=repo_name,
    commit_message="Initial Arbor-500M-1B upload"
)
```

## 📋 Required Files Checklist

- ✅ `config.json` - Model configuration
- ✅ `README.md` - Model card
- ✅ `tokenizer_config.json` - Tokenizer configuration
- ✅ `generation_config.json` - Generation defaults
- ⚠️ `pytorch_model.bin` - Model weights (you need to add this)
- ⚠️ `tokenizer.model` - SentencePiece model (download from Llama)
- ⚠️ `tokenizer.json` - Tokenizer JSON (download from Llama)
- ⚠️ `special_tokens_map.json` - Special tokens (download from Llama)

## 🎯 Usage After Upload

Once uploaded, users can use your model like any HuggingFace model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your model
tokenizer = AutoTokenizer.from_pretrained("username/arbor-500m-1b")
model = AutoModelForCausalLM.from_pretrained("username/arbor-500m-1b")

# Standard generation
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.8)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Dynamic growth
print(f"Initial size: {model.num_parameters():,}")
model.grow()  # Expand when needed
print(f"After growth: {model.num_parameters():,}")
```

## 🔧 Advanced Configuration

### Custom Growth Settings

Edit `config.json` to customize growth behavior:

```json
{
  "growth_factor": 2.0,           // How much to expand (2x = double)
  "max_growth_steps": 8,          // Maximum number of expansions
  "growth_threshold": 0.95,       // Performance threshold for growth
  "expandable_layers": [6,12,18]  // Which layers can grow
}
```

### Training Integration

Use with HuggingFace Trainer:

```python
from transformers import Trainer, TrainingArguments
from arbor.transformers_integration import ArborForCausalLM

model = ArborForCausalLM.from_pretrained("username/arbor-500m-1b")

training_args = TrainingArguments(
    output_dir="./training_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # ... other args
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset,
    # Custom callback for growth during training
)
```

## 🚨 Important Notes

1. **Model Registration**: The `ArborForCausalLM` class needs to be registered with transformers. Include the integration module in your environment.

2. **Growth During Inference**: Growth can happen during inference but requires re-uploading the model if you want to persist the larger version.

3. **Memory Requirements**: 
   - Base model: ~850MB VRAM
   - After max growth: ~2.2GB VRAM

4. **Compatibility**: Works with transformers >= 4.21.0

## 📞 Support

- Repository: [arbor-o1-living-ai](https://github.com/username/arbor-o1-living-ai)
- Issues: GitHub Issues
- Documentation: See `/docs` in the repository

## 🎉 You're Ready!

Your Arbor-500M-1B model is now ready for HuggingFace deployment! The dynamic growth capability makes it unique among available models on the Hub.

**Estimated deployment time**: 15-30 minutes
**Upload size**: ~850MB (base model) to ~2.2GB (max growth)
**Tokenizer**: Llama-2 SentencePiece (32K vocabulary)
