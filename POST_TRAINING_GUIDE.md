# Arbor Post-Training System

A comprehensive post-training system that allows you to fine-tune, instruct-tune, or domain-adapt Arbor models either immediately after training or later using models from HuggingFace Hub.

## 🎯 Features

- **🔄 Immediate Post-Training**: Continue training right after main training
- **📥 HuggingFace Integration**: Download and post-train any Arbor model
- **🧠 Multiple Training Types**: Fine-tuning, instruction tuning, domain adaptation
- **⚡ Efficient Training**: LoRA support for memory-efficient training
- **🧊 Layer Freezing**: Preserve learned representations while adapting others
- **🎛️ Flexible Configuration**: YAML-based configuration system
- **📊 Progress Monitoring**: Full logging and checkpoint support

## 🚀 Quick Start

### Option 1: Immediate Post-Training

Add to your main training config:

```yaml
# In your main training_config.yaml
post_training:
  enabled: true
  type: "fine_tune"
  datasets:
    - name: "specialized_data"
      source: "your-dataset"
      split: "train[:1000]"
  learning_rate: 5e-6
  max_steps: 500
  lora_enabled: true
```

Run main training (post-training will happen automatically):
```bash
python train.py configs/training_config.yaml
```

### Option 2: Later Post-Training

Download a model and post-train it:

```bash
# Create post-training config
python post_train.py --create-config

# Quick post-training
python post_train.py --model your-username/your-arbor-model --type instruct --steps 1000

# Or use config file
python post_train.py configs/post_training_instruct.yaml
```

## 📋 Configuration Guide

### Basic Configuration

```yaml
# Model source
model_source: "huggingface"  # "local", "huggingface", "checkpoint"
model_path: "your-username/your-arbor-model"

# Training type
training_type: "fine_tune"   # "fine_tune", "instruct", "domain_adapt"

# Datasets
datasets:
  - name: "custom_data"
    source: "your-dataset"
    split: "train[:2000]"
    text_column: "text"
    max_length: 2048

# Training parameters
learning_rate: 1e-5
max_steps: 1000
per_device_batch_size: 4
```

### Advanced Features

```yaml
# LoRA Configuration
lora_enabled: true
lora_rank: 16
target_modules: ["q_proj", "v_proj", "o_proj"]

# Layer Freezing
freeze_layers: [0, 1, 2, 3]  # Freeze first 4 layers

# Adaptive Context
adaptive_context_enabled: true
context_adaptation_strength: 0.8

# Output Settings
output_dir: "./post_training_output"
save_merged_model: true
push_to_hub: true
hub_model_id: "your-username/arbor-specialized"
```

## 🎪 Training Types

### 1. Fine-Tuning (`fine_tune`)

**Use Case**: Adapt model to specific domain or task
**Datasets**: Domain-specific text data
**Settings**: Lower learning rate, moderate steps

```yaml
training_type: "fine_tune"
datasets:
  - name: "domain_data"
    source: "your-domain-dataset" 
    text_column: "text"
    max_length: 1024
learning_rate: 1e-5
max_steps: 1000
lora_enabled: true
lora_rank: 16
```

### 2. Instruction Tuning (`instruct`)

**Use Case**: Create instruction-following assistant
**Datasets**: Instruction-response pairs
**Settings**: Higher learning rate, more steps

```yaml
training_type: "instruct"
datasets:
  - name: "alpaca"
    source: "tatsu-lab/alpaca"
    split: "train[:5000]"
    max_length: 2048
learning_rate: 2e-5
max_steps: 2000
lora_rank: 32  # Higher rank for instructions
```

### 3. Domain Adaptation (`domain_adapt`)

**Use Case**: Specialize for specific domain (code, medical, legal)
**Datasets**: Domain-specific corpora
**Settings**: Conservative approach with layer freezing

```yaml
training_type: "domain_adapt"
datasets:
  - name: "code_data"
    source: "codeparrot/github-code-clean"
    text_column: "code"
    max_length: 4096
learning_rate: 5e-6
freeze_layers: [0, 1, 2, 3, 4, 5]  # Freeze many layers
```

## 🛠️ Command Line Usage

### Create Example Configs

```bash
# Create all example configurations
python post_train.py --create-config
```

This creates:
- `configs/post_training_fine_tune.yaml`
- `configs/post_training_instruct.yaml`
- `configs/post_training_immediate.yaml`

### Quick Post-Training

```bash
# Fine-tune a HuggingFace model
python post_train.py --model microsoft/DialoGPT-medium --type fine_tune --steps 500

# Instruction tune with more steps
python post_train.py --model your-username/arbor-base --type instruct --steps 2000

# Domain adapt a local model
python post_train.py --model ./trained_models/final --type code --steps 1000
```

### Advanced Usage

```bash
# Use specific config file
python post_train.py configs/post_training_instruct.yaml

# Override model path
python post_train.py configs/post_training_fine_tune.yaml --model-path-override ./new_model

# Create and run custom config
python -c "
from arbor.train.post_trainer import create_post_training_config
create_post_training_config('huggingface', 'your-model', 'instruct', output_file='my_config.yaml')
"
python post_train.py my_config.yaml
```

## 🧪 Example Workflows

### Workflow 1: Complete Training Pipeline

```bash
# 1. Main training
python train.py configs/training_config.yaml

# 2. Automatic post-training (if enabled in config)
# → Runs immediately after main training

# 3. Manual additional post-training
python post_train.py --model ./arbor-training-output/final_model --type instruct --steps 1000
```

### Workflow 2: Download and Specialize

```bash
# 1. Download base model and post-train
python post_train.py --model your-username/arbor-base --type fine_tune --steps 500

# 2. Further specialize the result
python post_train.py --model ./post_training_fine_tune/merged_model --type instruct --steps 1000

# 3. Create domain-specific version
python post_train.py --model ./post_training_instruct/merged_model --type domain_adapt --steps 2000
```

### Workflow 3: Efficient LoRA Training

```yaml
# Create efficient LoRA config
lora_enabled: true
lora_rank: 8              # Small rank for efficiency
target_modules: ["q_proj", "v_proj"]  # Limited modules
freeze_layers: [0, 1, 2, 3, 4, 5]     # Freeze many layers
save_merged_model: true    # Save full model at end
```

## 📊 Monitoring and Validation

### Progress Tracking

Post-training automatically provides:
- **Parameter counts** before/after training
- **Loss tracking** during training  
- **Checkpoint saving** at regular intervals
- **Evaluation metrics** if eval dataset provided
- **Memory usage** monitoring

### Model Validation

```python
# Test your post-trained model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./post_training_output/merged_model")
model = AutoModelForCausalLM.from_pretrained("./post_training_output/merged_model")

# Test generation
prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🔧 Advanced Features

### LoRA (Low-Rank Adaptation)

```yaml
lora_enabled: true
lora_rank: 16              # 4-64 typical range
lora_alpha: 32            # Scaling factor
lora_dropout: 0.1         # Dropout rate
target_modules:           # Which modules to adapt
  - "q_proj"              # Query projection
  - "v_proj"              # Value projection  
  - "o_proj"              # Output projection
  - "gate_proj"           # Gate projection (Arbor FFN)
  - "up_proj"             # Up projection (Arbor FFN)
  - "down_proj"           # Down projection (Arbor FFN)
```

### Layer Freezing Strategy

```yaml
# Conservative: Freeze early layers
freeze_layers: [0, 1, 2, 3]

# Moderate: Freeze middle layers
freeze_layers: [4, 5, 6, 7, 8]

# Aggressive: Freeze most layers
freeze_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Custom: Freeze specific pattern
freeze_layers: [0, 2, 4, 6]  # Every other layer
```

### Hardware Optimization

The post-training system automatically:
- **Detects available GPU memory**
- **Adjusts batch sizes** accordingly
- **Uses gradient checkpointing** for memory efficiency
- **Implements mixed precision** training
- **Monitors memory usage** during training

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors**:
```bash
# Check model path
ls ./trained_models/final_model
# Or verify HF model ID exists
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('your-model-id')"
```

**Memory Issues**:
```yaml
# Reduce batch size
per_device_batch_size: 1
gradient_accumulation_steps: 8

# Enable gradient checkpointing
gradient_checkpointing: true

# Use smaller LoRA rank
lora_rank: 4
```

**Dataset Loading Errors**:
```python
# Test dataset loading
from datasets import load_dataset
dataset = load_dataset("your-dataset", split="train[:10]")
print(dataset[0])
```

### Performance Tips

1. **Use LoRA** for memory efficiency
2. **Freeze early layers** to preserve learned representations
3. **Use smaller learning rates** than main training
4. **Monitor validation loss** to avoid overfitting
5. **Save merged models** for easy deployment
6. **Use gradient accumulation** for larger effective batch sizes

## 📚 Integration with Main Training

Post-training integrates seamlessly with the main Arbor training system:

```yaml
# In your main training config
post_training:
  enabled: true              # Auto-run after main training
  type: "fine_tune"
  datasets: [...]           # Specialized datasets
  learning_rate: 5e-6       # Lower than main training
  max_steps: 500            # Quick specialization
  lora_enabled: true        # Memory efficient
  output_dir: "./post_training_auto"
```

This provides a **complete training pipeline** from base model to specialized model in one command!

## 🎉 Success Stories

**Example Use Cases**:

1. **Code Assistant**: Base model → Code post-training → Programming assistant
2. **Domain Expert**: Base model → Medical data → Medical Q&A system  
3. **Instruction Following**: Base model → Instruction tuning → General assistant
4. **Multilingual**: English model → Target language data → Multilingual model

The post-training system makes it easy to create specialized, high-performance models for any domain! 🌱
