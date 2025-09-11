# Arbor Documentation

**Complete Technical Documentation for Arbor by Noema Research**

*Version 1.0 | September 2025*

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Adaptive Context System](#adaptive-context-system)
7. [Post-Training & Fine-Tuning](#post-training--fine-tuning)
8. [API Reference](#api-reference)
9. [Examples & Tutorials](#examples--tutorials)
10. [Performance & Optimization](#performance--optimization)
11. [Troubleshooting](#troubleshooting)
12. [Development & Contributing](#development--contributing)

---

## Overview

**Arbor** is a revolutionary transformer architecture developed by [Noema Research](https://github.com/Noema-Research) that features:

- 🧠 **Adaptive Context Windows**: Intelligent context scaling from 1K to 131K tokens
- 🌱 **Dynamic Neural Growth**: Architecture expands during training (699M→799M parameters)
- 🎯 **Task-Aware Routing**: AI-powered analysis for optimal resource allocation
- 🤗 **HuggingFace Integration**: Full compatibility with Transformers ecosystem
- 🔧 **Production Ready**: SafeTensors, YAML configuration, comprehensive tooling

### Key Innovations

1. **Two-Stage Adaptive Architecture**: Lightweight router + main transformer
2. **Real-Time Context Adaptation**: Context windows change based on task complexity
3. **Hardware-Aware Scaling**: Automatic optimization for available resources
4. **Comprehensive Post-Training**: Fine-tuning, instruction tuning, domain adaptation

---

## Architecture

### Core Components

```
🌳 Arbor Architecture
├── 🧠 Task Complexity Router (3-layer lightweight transformer)
│   ├── Task Type Detection (8 categories)
│   ├── Complexity Analysis (4 levels)
│   └── Context Recommendation (1K-131K tokens)
├── 🏗️ Main Transformer (24-layer dynamic architecture)
│   ├── Adaptive Context Manager
│   ├── Expandable FFN Layers
│   ├── Multi-Head Attention
│   └── Growth Monitoring System
└── 🎯 Integration Layer
    ├── HuggingFace Compatibility
    ├── SafeTensors Support
    └── YAML Configuration
```

### Model Specifications

| Component | Specification | Description |
|-----------|---------------|-------------|
| **Architecture** | Transformer Decoder | Causal language modeling |
| **Base Parameters** | 699M | Starting parameter count |
| **Growth Capacity** | 799M | Maximum after expansion |
| **Vocabulary** | 128K tokens | Hermes-4-405B tokenizer |
| **Context Range** | 1K - 131K | Adaptive window size |
| **Layers** | 24 | Transformer blocks |
| **Hidden Size** | 1024 | Embedding dimension |
| **Attention Heads** | 16 | Multi-head attention |
| **FFN Dimension** | 4096 | Feed-forward network |

### Dynamic Growth Mechanism

```python
class ExpandableFFN(nn.Module):
    \"\"\"Feed-forward network that can grow during training.\"\"\"
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_ffn_dim = config.ffn_dim
        self.max_ffn_dim = config.ffn_dim * config.growth_factor
        
        # Initial layers
        self.w1 = nn.Linear(config.dim, self.current_ffn_dim)
        self.w2 = nn.Linear(self.current_ffn_dim, config.dim)
        
    def expand(self, new_dim: int):
        \"\"\"Expand FFN to new dimension while preserving learned weights.\"\"\"
        if new_dim <= self.current_ffn_dim:
            return
            
        # Create new larger layers
        old_w1, old_w2 = self.w1, self.w2
        self.w1 = nn.Linear(self.config.dim, new_dim)
        self.w2 = nn.Linear(new_dim, self.config.dim)
        
        # Copy old weights to preserve learning
        with torch.no_grad():
            self.w1.weight[:, :].copy_(old_w1.weight)
            self.w1.bias[:].copy_(old_w1.bias)
            self.w2.weight[:self.config.dim, :].copy_(old_w2.weight)
            self.w2.bias.copy_(old_w2.bias)
            
        self.current_ffn_dim = new_dim
```

---

## Installation & Setup

### System Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher  
- **CUDA**: 11.8+ (for GPU training)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Storage**: 10GB+ free space
- **Internet**: Required for tokenizer and dataset downloads

### Quick Installation

```bash
# Clone repository
git clone https://github.com/Noema-Research/Arbor.git
cd Arbor

# Install dependencies
pip install torch transformers datasets wandb PyYAML safetensors

# Verify installation
python -c "from arbor.modeling.model import ArborTransformer; print('✅ Installation successful!')"
```

### Development Installation

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/ -v
```

### Environment Variables

```bash
# Required for HuggingFace integration
export HF_TOKEN="your_huggingface_token"

# Optional for experiment tracking
export WANDB_API_KEY="your_wandb_key"

# Optional custom cache directories
export HF_HOME="/custom/cache/path"
export TRANSFORMERS_CACHE="/custom/cache/transformers"
```

---

## Configuration

Arbor uses YAML-based configuration for training, making complex setups simple and reproducible.

### Main Configuration Structure

```yaml
# Core model architecture
model:
  vocab_size: 128000           # Hermes-4-405B vocabulary
  hidden_size: 1024           # Embedding dimension
  num_layers: 24              # Transformer layers
  num_attention_heads: 16     # Multi-head attention
  intermediate_size: 4096     # FFN hidden size
  max_position_embeddings: 131072  # Maximum context length
  
  # Growth configuration
  growth:
    enabled: true             # Enable dynamic growth
    factor: 2.0              # Maximum growth multiplier
    threshold: 0.95          # Utilization threshold for expansion
    
# Adaptive context system
adaptive_context:
  enabled: true               # Enable adaptive context
  router_config:
    hidden_size: 256         # Router model size
    num_layers: 3            # Router depth
    num_attention_heads: 8   # Router attention heads
  
  # Context length mapping
  task_contexts:
    chat: [1024, 4096]       # Min, max for chat tasks
    code: [4096, 16384]      # Min, max for code tasks
    reasoning: [8192, 32768] # Min, max for reasoning
    document: [16384, 131072] # Min, max for documents
    
# Training datasets
datasets:
  - name: "tinystories"
    source: "roneneldan/TinyStories" 
    split: "train[:50000]"
    text_column: "text"
    preprocessing:
      max_length: 1024
      
  - name: "openwebtext"
    source: "openwebtext"
    split: "train[:100000]"
    text_column: "text"
    preprocessing:
      max_length: 2048
      
# Training configuration
training:
  learning_rate: 2e-5
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  
  # Training schedule
  max_steps: 10000
  warmup_steps: 1000
  lr_scheduler_type: "cosine"
  
  # Batch configuration
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4
  dataloader_num_workers: 4
  
  # Memory optimization
  fp16: true
  gradient_checkpointing: true
  dataloader_pin_memory: true
  
  # Monitoring
  logging_steps: 10
  eval_steps: 500
  save_steps: 1000
  
# Post-training configuration (optional)
post_training:
  enabled: false             # Enable immediate post-training
  type: "fine_tune"         # fine_tune, instruct, domain_adapt
  datasets:
    - name: "custom_domain"
      source: "your-dataset"
      split: "train[:1000]"
  learning_rate: 5e-6
  max_steps: 500
  lora_enabled: true
  
# HuggingFace integration
huggingface:
  upload:
    enabled: true
    repository: "your-username/arbor-model"
    private: false
    token: "${HF_TOKEN}"
    
# Logging and monitoring
logging:
  wandb:
    enabled: true
    project: "arbor-experiments"
    entity: "your-wandb-entity"
    tags: ["arbor", "adaptive-context", "dynamic-growth"]
```

### Configuration Templates

#### Small Model (Testing)
```yaml
# configs/arbor_small.yaml
model:
  vocab_size: 10000
  hidden_size: 256
  num_layers: 6
  num_attention_heads: 8
  intermediate_size: 1024
  max_position_embeddings: 2048

adaptive_context:
  enabled: true
  task_contexts:
    chat: [512, 1024]
    code: [1024, 2048]
```

#### Production Model (Full Scale)
```yaml
# configs/arbor_production.yaml
model:
  vocab_size: 128000
  hidden_size: 1024
  num_layers: 24
  num_attention_heads: 16
  intermediate_size: 4096
  max_position_embeddings: 131072

training:
  max_steps: 100000
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
```

---

## Training

### Basic Training

```bash
# Train with default configuration
python train.py configs/training_config.yaml

# Train with custom configuration
python train.py configs/arbor_production.yaml

# Resume from checkpoint
python train.py configs/training_config.yaml --resume_from_checkpoint ./checkpoints/checkpoint-5000
```

### Training Process

1. **Initialization Phase**
   - Downloads fresh Hermes-4-405B tokenizer
   - Initializes Arbor model with growth capability
   - Sets up adaptive context router
   - Configures datasets and data loaders

2. **Training Phase**
   - Monitors FFN layer utilization
   - Expands layers when utilization > 95%
   - Logs growth events and parameter counts
   - Evaluates on validation set regularly

3. **Completion Phase**
   - Saves final model in SafeTensors format
   - Uploads to HuggingFace Hub (if configured)
   - Generates training summary report
   - Optionally runs post-training

### Monitoring Training

#### WandB Integration
```yaml
logging:
  wandb:
    enabled: true
    project: "arbor-experiments"
    tags: ["production", "adaptive-context"]
```

Tracks:
- Loss curves (training/validation)
- Model growth events
- Context adaptation statistics
- Hardware utilization
- Learning rate schedule

#### Console Output
```
Epoch 1/10, Step 100/10000
Train Loss: 2.45 | Val Loss: 2.52 | LR: 1.8e-5
🌱 Layer 15 expanded: 699M → 710M parameters
📊 Context adaptations: 45 (avg 4.2K tokens)
💾 Memory: 12.3GB / 16GB VRAM
⏱️  Speed: 1.2 it/s
```

### Growth Monitoring

```python
# Monitor model growth during training
class GrowthTracker:
    def __init__(self):
        self.growth_events = []
        self.param_history = []
    
    def log_growth(self, step, layer_id, old_params, new_params):
        event = {
            "step": step,
            "layer": layer_id,
            "params_before": old_params,
            "params_after": new_params,
            "growth_factor": new_params / old_params
        }
        self.growth_events.append(event)
        print(f"🌱 Step {step}: Layer {layer_id} grew {old_params:,} → {new_params:,}")
```

---

## Adaptive Context System

The adaptive context system is Arbor's most innovative feature, dynamically adjusting context windows based on task complexity.

### How It Works

1. **Input Analysis**: Router model analyzes input text
2. **Task Classification**: Identifies task type and complexity
3. **Context Recommendation**: Suggests optimal context length
4. **Hardware Check**: Ensures recommendation fits available resources
5. **Dynamic Adaptation**: Main model adjusts context window

### Task Types and Context Mapping

| Task Type | Description | Context Range | Examples |
|-----------|-------------|---------------|----------|
| 💬 **Chat** | Conversational interactions | 1K - 4K | "How are you?", "What's the weather?" |
| 💻 **Code** | Programming and technical content | 4K - 16K | "Write a Python function", "Debug this code" |
| 🧠 **Reasoning** | Complex logical problems | 8K - 32K | "Solve this math problem", "Analyze the logic" |
| 📄 **Document** | Long-form text processing | 16K - 131K | "Summarize this paper", "Analyze this document" |
| 🎨 **Creative** | Creative writing tasks | 2K - 16K | "Write a story", "Create a poem" |
| ❓ **Q&A** | Question answering | 1K - 8K | "What is...", "Explain the concept of..." |
| 📝 **Summary** | Text summarization | 4K - 32K | "Summarize the following...", "Key points:" |
| 🌐 **Translation** | Language translation | 2K - 8K | "Translate this text", "Convert to Spanish" |

### Router Model Architecture

```python
class TaskComplexityRouter(nn.Module):
    \"\"\"Lightweight model for task analysis and context recommendation.\"\"\"
    
    def __init__(self, config):
        super().__init__()
        
        # Lightweight architecture for speed
        self.embedding_dim = 256
        self.hidden_dim = 512
        self.num_layers = 3
        
        # Fast processing components
        self.embeddings = nn.Embedding(config.vocab_size, self.embedding_dim)
        self.pos_embeddings = nn.Embedding(1024, self.embedding_dim)  # Short context for analysis
        
        # Transformer layers for analysis
        self.layers = nn.ModuleList([
            RouterAttentionBlock(self.embedding_dim, 8, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # Classification heads
        self.task_classifier = nn.Linear(self.embedding_dim, 8)      # 8 task types
        self.complexity_head = nn.Linear(self.embedding_dim, 4)     # 4 complexity levels
        self.context_predictor = nn.Linear(self.embedding_dim, 8)   # 8 context length options
        
    def forward(self, input_ids, attention_mask=None):
        # Process first 256 tokens for fast analysis
        analysis_input = input_ids[:, :256]
        
        # Embed and add positional encoding
        x = self.embeddings(analysis_input)
        x = x + self.pos_embeddings(torch.arange(x.size(1), device=x.device))
        
        # Pass through router layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Pool and classify
        pooled = x.mean(dim=1)  # Simple mean pooling
        
        task_logits = self.task_classifier(pooled)
        complexity_logits = self.complexity_head(pooled)
        context_logits = self.context_predictor(pooled)
        
        return {
            "task_type": torch.softmax(task_logits, dim=-1),
            "complexity": torch.softmax(complexity_logits, dim=-1),
            "context_recommendation": torch.softmax(context_logits, dim=-1)
        }
```

### Context Adaptation Process

```python
class AdaptiveContextManager:
    \"\"\"Manages dynamic context window adaptation.\"\"\"
    
    def __init__(self, model, router, config):
        self.model = model
        self.router = router
        self.config = config
        
        # Context length options (powers of 2 for efficiency)
        self.context_options = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        
        # Task to context mapping
        self.task_contexts = {
            0: [0, 2],    # chat: 1K-4K
            1: [2, 4],    # code: 4K-16K  
            2: [3, 5],    # reasoning: 8K-32K
            3: [4, 7],    # document: 16K-131K
            4: [1, 4],    # creative: 2K-16K
            5: [0, 3],    # qa: 1K-8K
            6: [2, 5],    # summary: 4K-32K
            7: [1, 3],    # translation: 2K-8K
        }
    
    def analyze_and_adapt(self, input_ids, attention_mask=None):
        \"\"\"Analyze input and adapt context window.\"\"\"
        
        # Router analysis
        with torch.no_grad():
            router_output = self.router(input_ids, attention_mask)
        
        # Extract predictions
        task_pred = torch.argmax(router_output["task_type"], dim=-1).item()
        complexity_pred = torch.argmax(router_output["complexity"], dim=-1).item()
        
        # Determine context length
        min_idx, max_idx = self.task_contexts[task_pred]
        
        # Adjust based on complexity (0: simple, 3: expert)
        complexity_boost = complexity_pred * 0.25
        context_idx = min(max_idx, int(min_idx + (max_idx - min_idx) * complexity_boost))
        
        recommended_length = self.context_options[context_idx]
        
        # Hardware constraint check
        available_memory = torch.cuda.get_device_properties(0).total_memory
        current_memory = torch.cuda.memory_allocated(0)
        memory_ratio = current_memory / available_memory
        
        if memory_ratio > 0.8:  # Reduce context if memory constrained
            recommended_length = min(recommended_length, 8192)
        
        # Apply adaptation
        self.model.set_context_length(recommended_length)
        
        return {
            "task_type": task_pred,
            "complexity": complexity_pred,
            "context_length": recommended_length,
            "confidence": float(torch.max(router_output["task_type"]))
        }
```

### Usage Examples

```python
# Initialize adaptive context
from arbor.modeling.adaptive_context import AdaptiveContextManager

manager = AdaptiveContextManager(model, router, config)

# Example 1: Chat input
chat_input = "Hello, how are you doing today?"
result = manager.analyze_and_adapt(tokenizer.encode(chat_input, return_tensors="pt"))
# → task_type: 0 (chat), context_length: 1024

# Example 2: Code input  
code_input = "Write a Python function that implements a binary search algorithm..."
result = manager.analyze_and_adapt(tokenizer.encode(code_input, return_tensors="pt"))
# → task_type: 1 (code), context_length: 8192

# Example 3: Document analysis
doc_input = "Please analyze the following research paper and provide key insights..."
result = manager.analyze_and_adapt(tokenizer.encode(doc_input, return_tensors="pt"))
# → task_type: 3 (document), context_length: 32768
```

---

## Post-Training & Fine-Tuning

Arbor includes a comprehensive post-training system for specializing models after initial training.

### Post-Training Types

#### 1. Fine-Tuning
Adapt model to specific domain or use case.

```yaml
# configs/post_training_fine_tune.yaml
model_source: "huggingface"
model_path: "your-username/arbor-base"

training_type: "fine_tune"

datasets:
  - name: "domain_data"
    source: "your-domain-dataset"
    split: "train[:5000]"
    text_column: "text"
    max_length: 1024

training:
  learning_rate: 1e-5
  max_steps: 1000
  per_device_batch_size: 4

lora:
  enabled: true
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "o_proj"]
```

#### 2. Instruction Tuning
Create instruction-following assistant.

```yaml
# configs/post_training_instruct.yaml
training_type: "instruct"

datasets:
  - name: "alpaca"
    source: "tatsu-lab/alpaca"
    split: "train[:10000]"
    instruction_column: "instruction"
    input_column: "input"
    output_column: "output"
    max_length: 2048

training:
  learning_rate: 2e-5
  max_steps: 2000
  per_device_batch_size: 2

lora:
  enabled: true
  rank: 32  # Higher rank for instruction following
```

#### 3. Domain Adaptation
Specialize for specific domain (medical, legal, code, etc.).

```yaml
# configs/post_training_domain.yaml
training_type: "domain_adapt"

datasets:
  - name: "medical_texts"
    source: "medical-domain-dataset"
    split: "train[:20000]"
    text_column: "text"
    max_length: 4096

training:
  learning_rate: 5e-6  # Conservative learning rate
  max_steps: 3000

# Freeze early layers to preserve general knowledge
freeze_layers: [0, 1, 2, 3, 4, 5]

lora:
  enabled: true
  rank: 8  # Smaller rank for domain adaptation
```

### Post-Training CLI

```bash
# Quick post-training commands
python post_train.py --model your-username/arbor-base --type fine_tune --steps 1000

# Use configuration file
python post_train.py configs/post_training_instruct.yaml

# Create example configurations
python post_train.py --create-config

# Download HF model and post-train
python post_train.py --model microsoft/DialoGPT-medium --type instruct --steps 2000
```

### Immediate Post-Training

Automatically continue training after main training:

```yaml
# In main training config
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

### LoRA Configuration

```yaml
lora:
  enabled: true
  rank: 16                    # Adaptation rank (4-64)
  alpha: 32                  # Scaling factor
  dropout: 0.1               # LoRA dropout
  target_modules:            # Modules to adapt
    - "q_proj"               # Query projection
    - "v_proj"               # Value projection
    - "o_proj"               # Output projection
    - "gate_proj"            # Gate projection (FFN)
    - "up_proj"              # Up projection (FFN)
    - "down_proj"            # Down projection (FFN)
```

---

## API Reference

### Core Classes

#### ArborTransformer

```python
class ArborTransformer(nn.Module):
    \"\"\"Main Arbor transformer model with adaptive context and growth.\"\"\"
    
    def __init__(self, config: ArborConfig):
        \"\"\"Initialize Arbor model.
        
        Args:
            config: Model configuration
        \"\"\"
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        \"\"\"Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for training [batch_size, seq_len]
            
        Returns:
            ModelOutput with logits, loss, hidden_states
        \"\"\"
    
    def set_context_length(self, length: int):
        \"\"\"Dynamically set context window length.\"\"\"
    
    def expand_layer(self, layer_idx: int, growth_factor: float = 2.0):
        \"\"\"Expand a specific layer's capacity.\"\"\"
    
    def count_parameters(self) -> int:
        \"\"\"Count total trainable parameters.\"\"\"
```

#### ArborConfig

```python
@dataclass
class ArborConfig:
    \"\"\"Configuration class for Arbor models.\"\"\"
    
    # Architecture
    vocab_size: int = 128000
    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 131072
    
    # Growth settings
    growth_enabled: bool = True
    growth_factor: float = 2.0
    growth_threshold: float = 0.95
    
    # Adaptive context
    adaptive_context_enabled: bool = True
    router_hidden_size: int = 256
    router_num_layers: int = 3
    
    # Training
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    activation_function: str = "gelu"
```

#### TaskComplexityRouter

```python
class TaskComplexityRouter(nn.Module):
    \"\"\"Router model for task analysis and context recommendation.\"\"\"
    
    def __init__(self, config):
        \"\"\"Initialize router model.\"\"\"
    
    def forward(self, input_ids, attention_mask=None):
        \"\"\"Analyze input and return task predictions.
        
        Returns:
            dict: {
                'task_type': tensor,        # Task classification
                'complexity': tensor,       # Complexity level
                'context_recommendation': tensor  # Context length
            }
        \"\"\"
    
    def analyze_text(self, text: str, tokenizer) -> ContextDecision:
        \"\"\"High-level text analysis interface.\"\"\"
```

#### ArborTrainer

```python
class ArborTrainer:
    \"\"\"YAML-based trainer for Arbor models.\"\"\"
    
    def __init__(self, config_path: str):
        \"\"\"Initialize trainer from YAML config.\"\"\"
    
    def train(self):
        \"\"\"Run full training pipeline.\"\"\"
    
    def evaluate(self):
        \"\"\"Run evaluation on validation set.\"\"\"
    
    def save_model(self, output_dir: str):
        \"\"\"Save model in HuggingFace format.\"\"\"
```

#### ArborPostTrainer

```python
class ArborPostTrainer:
    \"\"\"Post-training system for fine-tuning and specialization.\"\"\"
    
    def __init__(self, config: dict):
        \"\"\"Initialize post-trainer.\"\"\"
    
    def load_model(self, model_path: str):
        \"\"\"Load base model for post-training.\"\"\"
    
    def setup_lora(self, lora_config: dict):
        \"\"\"Configure LoRA for efficient training.\"\"\"
    
    def train(self):
        \"\"\"Run post-training process.\"\"\"
    
    def merge_and_save(self, output_dir: str):
        \"\"\"Merge LoRA weights and save full model.\"\"\"
```

### Utility Functions

```python
# Model utilities
def count_parameters(model: nn.Module) -> int:
    \"\"\"Count trainable parameters in model.\"\"\"

def get_model_size_mb(model: nn.Module) -> float:
    \"\"\"Get model size in megabytes.\"\"\"

def save_model_safetensors(model: nn.Module, path: str):
    \"\"\"Save model using SafeTensors format.\"\"\"

# Data utilities
def load_dataset_from_config(dataset_config: dict):
    \"\"\"Load dataset from YAML configuration.\"\"\"

def create_data_collator(tokenizer, max_length: int):
    \"\"\"Create data collator for training.\"\"\"

# Training utilities
def setup_optimizer(model: nn.Module, config: dict):
    \"\"\"Setup optimizer from configuration.\"\"\"

def setup_scheduler(optimizer, config: dict):
    \"\"\"Setup learning rate scheduler.\"\"\"

def monitor_gpu_memory():
    \"\"\"Monitor GPU memory usage during training.\"\"\"
```

---

## Examples & Tutorials

### Basic Training Example

```python
# examples/basic_training.py
import yaml
from arbor.train.yaml_trainer import ArborTrainer

# Load configuration
with open("configs/training_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize trainer
trainer = ArborTrainer(config)

# Start training
trainer.train()

# Evaluate model
results = trainer.evaluate()
print(f"Final loss: {results['eval_loss']:.4f}")

# Save model
trainer.save_model("./trained_model")
```

### Custom Dataset Training

```python
# examples/custom_datasets.py
from datasets import Dataset
import torch
from arbor.modeling.model import ArborTransformer, ArborConfig

# Create custom dataset
texts = [
    "This is a sample text for training.",
    "Another example of training data.",
    # ... more texts
]

dataset = Dataset.from_dict({"text": texts})

# Initialize model
config = ArborConfig(
    vocab_size=10000,
    hidden_size=512,
    num_layers=12
)
model = ArborTransformer(config)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Loss: {loss.item():.4f}")
```

### Adaptive Context Demo

```python
# examples/adaptive_context_demo.py
from arbor.modeling.model import ArborTransformer
from arbor.modeling.adaptive_context import AdaptiveContextManager
from transformers import AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-4-405B")
model = ArborTransformer.from_pretrained("your-username/arbor-model")

# Initialize adaptive context
context_manager = AdaptiveContextManager(model, model.router, model.config)

# Test different inputs
inputs = [
    "Hello, how are you?",  # Simple chat
    "Write a Python function to sort a list",  # Code task
    "Analyze the economic implications of climate change...",  # Complex analysis
]

for text in inputs:
    tokens = tokenizer.encode(text, return_tensors="pt")
    result = context_manager.analyze_and_adapt(tokens)
    
    print(f"Input: {text[:50]}...")
    print(f"Task: {result['task_type']}, Context: {result['context_length']}")
    print()
```

### Post-Training Example

```python
# examples/post_training_demo.py
from arbor.train.post_trainer import ArborPostTrainer

# Configuration for instruction tuning
config = {
    "model_source": "huggingface",
    "model_path": "your-username/arbor-base",
    "training_type": "instruct",
    "datasets": [{
        "name": "alpaca",
        "source": "tatsu-lab/alpaca",
        "split": "train[:5000]"
    }],
    "training": {
        "learning_rate": 2e-5,
        "max_steps": 1000,
        "per_device_batch_size": 4
    },
    "lora": {
        "enabled": True,
        "rank": 16,
        "target_modules": ["q_proj", "v_proj"]
    }
}

# Initialize post-trainer
post_trainer = ArborPostTrainer(config)

# Run post-training
post_trainer.train()

# Save specialized model
post_trainer.merge_and_save("./instruction_tuned_model")
```

### Inference Example

```python
# examples/inference_demo.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("your-username/arbor-model")
model = AutoModelForCausalLM.from_pretrained("your-username/arbor-model")

# Enable adaptive context
model.eval()

def generate_response(prompt: str, max_length: int = 200):
    \"\"\"Generate response with adaptive context.\"\"\"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate with adaptive context
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]

# Test generation
prompts = [
    "Explain quantum computing:",
    "Write a Python function to calculate fibonacci:",
    "What are the benefits of renewable energy?"
]

for prompt in prompts:
    response = generate_response(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
```

---

## Performance & Optimization

### Training Performance

#### Memory Optimization

```yaml
# Memory-efficient training configuration
training:
  # Enable mixed precision
  fp16: true
  
  # Gradient checkpointing for memory savings
  gradient_checkpointing: true
  
  # Efficient data loading
  dataloader_pin_memory: true
  dataloader_num_workers: 4
  
  # Batch size optimization
  per_device_train_batch_size: 2  # Smaller batch size
  gradient_accumulation_steps: 16  # Larger effective batch
  
  # Memory-efficient optimizers
  optimizer: "adamw"
  weight_decay: 0.01
```

#### Speed Optimization

```yaml
# Speed-optimized configuration
training:
  # Compiled model (PyTorch 2.0+)
  torch_compile: true
  
  # Efficient attention
  use_flash_attention: true
  
  # Data loading optimization
  dataloader_num_workers: 8
  dataloader_pin_memory: true
  dataloader_persistent_workers: true
  
  # Reduced precision
  bf16: true  # Better than fp16 on modern hardware
```

### Inference Performance

#### Optimizations

```python
# Optimize model for inference
@torch.inference_mode()
def optimized_generate(model, tokenizer, prompt: str):
    \"\"\"Optimized generation with adaptive context.\"\"\"
    
    # Compile model for speed (PyTorch 2.0+)
    if not hasattr(model, '_compiled'):
        model = torch.compile(model)
        model._compiled = True
    
    # Use efficient attention
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Benchmarking

```python
# Performance benchmarking
import time
import torch

def benchmark_model(model, tokenizer, num_runs=100):
    \"\"\"Benchmark model performance.\"\"\"
    
    # Warmup
    prompt = "Hello, how are you?"
    for _ in range(10):
        _ = model.generate(**tokenizer(prompt, return_tensors="pt"), max_length=50)
    
    # Timing runs
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.generate(**tokenizer(prompt, return_tensors="pt"), max_length=50)
        torch.cuda.synchronize()  # Wait for GPU
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    tokens_per_second = 50 / avg_time  # 50 tokens generated
    
    print(f"Average generation time: {avg_time:.3f}s")
    print(f"Tokens per second: {tokens_per_second:.1f}")
    print(f"Memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
```

### Hardware Recommendations

#### Training Hardware

| Model Size | GPU Memory | Recommended Setup | Batch Size |
|------------|------------|-------------------|------------|
| **Small (100M)** | 8GB | RTX 3080, RTX 4070 | 16-32 |
| **Medium (500M)** | 16GB | RTX 4080, A4000 | 8-16 |
| **Large (700M+)** | 24GB+ | RTX 4090, A5000, A6000 | 4-8 |
| **Production** | 48GB+ | A100, H100 | 16-32 |

#### Inference Hardware

| Use Case | GPU Memory | Setup | Performance |
|----------|------------|-------|-------------|
| **Development** | 8GB+ | RTX 3080+ | 20-50 tokens/s |
| **Production** | 16GB+ | RTX 4080+ | 50-100 tokens/s |
| **High-throughput** | 24GB+ | A100 | 100-200 tokens/s |

---

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: `ImportError: No module named 'arbor'`
```bash
# Solution: Install in development mode
pip install -e .

# Or check Python path
python -c "import sys; print(sys.path)"
```

**Issue**: CUDA out of memory during training
```yaml
# Solution: Reduce batch size and enable optimizations
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true
  fp16: true
```

**Issue**: Slow training speed
```yaml
# Solution: Optimize data loading and use mixed precision
training:
  dataloader_num_workers: 8
  dataloader_pin_memory: true
  bf16: true  # Faster than fp16 on modern GPUs
  torch_compile: true
```

#### Model Loading Issues

**Issue**: SafeTensors loading error
```python
# Solution: Update transformers library
pip install --upgrade transformers

# Or load with legacy format
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    use_safetensors=False
)
```

**Issue**: Tokenizer download fails
```python
# Solution: Set up authentication
from huggingface_hub import login
login(token="your_hf_token")

# Or use environment variable
export HF_TOKEN="your_hf_token"
```

#### Training Issues

**Issue**: Loss not decreasing
```yaml
# Check learning rate
training:
  learning_rate: 1e-4  # Try different values: 1e-5, 2e-5, 5e-5

# Check gradient clipping
training:
  max_grad_norm: 1.0

# Monitor gradients
logging:
  log_gradients: true
```

**Issue**: Model not growing during training
```yaml
# Check growth configuration
model:
  growth:
    enabled: true
    threshold: 0.9  # Lower threshold for more growth
    
# Monitor utilization
logging:
  log_layer_utilization: true
```

#### Post-Training Issues

**Issue**: LoRA training fails
```python
# Solution: Check target modules
lora:
  target_modules: ["q_proj", "v_proj"]  # Start with basic modules
  rank: 8  # Lower rank for stability
```

**Issue**: Model performance degrades after post-training
```yaml
# Solution: Use conservative settings
training:
  learning_rate: 1e-6  # Very low learning rate
  
# Freeze more layers
freeze_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Debug Mode

Enable detailed logging for troubleshooting:

```yaml
# Debug configuration
logging:
  level: DEBUG
  log_gradients: true
  log_layer_utilization: true
  log_memory_usage: true
  save_attention_maps: true

training:
  log_every: 1  # Log every step
  eval_every: 10  # Frequent evaluation
```

### Performance Debugging

```python
# Profile training step
import torch.profiler

def profile_training_step(model, batch):
    \"\"\"Profile a single training step.\"\"\"
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export for visualization
    prof.export_chrome_trace("trace.json")
```

### Getting Help

1. **Check logs**: Look at training logs for specific error messages
2. **Monitor resources**: Use `nvidia-smi` to check GPU usage
3. **Validate config**: Use `python test_config.py` to validate YAML files
4. **Search issues**: Check GitHub issues for similar problems
5. **Ask community**: Use GitHub Discussions for questions

---

## Development & Contributing

### Development Setup

```bash
# Fork and clone repository
git fork https://github.com/Noema-Research/Arbor.git
git clone https://github.com/YOUR_USERNAME/Arbor.git
cd Arbor

# Create development environment
python -m venv arbor-dev
source arbor-dev/bin/activate  # or `arbor-dev\Scripts\activate` on Windows

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
python -m pytest tests/ -v
```

### Code Standards

#### Style Guidelines

```bash
# Format code with Black
black arbor/ tests/ examples/

# Sort imports with isort
isort arbor/ tests/ examples/

# Type checking with mypy
mypy arbor/

# Linting with flake8
flake8 arbor/ tests/
```

#### Testing Requirements

```python
# Test structure
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_model.py       # Model architecture tests
│   ├── test_layers.py      # Layer implementation tests
│   └── test_adaptive_context.py  # Context system tests
├── integration/            # Integration tests
│   ├── test_training.py    # End-to-end training tests
│   └── test_inference.py   # Inference pipeline tests
└── performance/            # Performance benchmarks
    ├── test_speed.py       # Speed benchmarks
    └── test_memory.py      # Memory usage tests

# Writing tests
import pytest
import torch
from arbor.modeling.model import ArborTransformer, ArborConfig

class TestArborModel:
    def test_model_initialization(self):
        \"\"\"Test basic model initialization.\"\"\"
        config = ArborConfig(vocab_size=1000, hidden_size=256)
        model = ArborTransformer(config)
        assert model.config.vocab_size == 1000
        assert model.config.hidden_size == 256
    
    def test_forward_pass(self):
        \"\"\"Test forward pass with dummy data.\"\"\"
        config = ArborConfig(vocab_size=1000, hidden_size=256, num_layers=4)
        model = ArborTransformer(config)
        
        input_ids = torch.randint(0, 1000, (2, 32))  # batch_size=2, seq_len=32
        outputs = model(input_ids)
        
        assert outputs.logits.shape == (2, 32, 1000)
        assert not torch.isnan(outputs.logits).any()
```

#### Documentation Standards

```python
def complex_function(param1: str, param2: int, param3: Optional[Dict] = None) -> Tuple[str, int]:
    \"\"\"One-line summary of the function.
    
    Longer description explaining what the function does, how it works,
    and any important details about its behavior.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter  
        param3: Optional parameter with default value
        
    Returns:
        Tuple containing:
            - str: Description of first return value
            - int: Description of second return value
            
    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string
        
    Example:
        >>> result = complex_function("hello", 42)
        >>> print(result)
        ("processed_hello", 84)
    \"\"\"
```

### Contributing Workflow

#### 1. Issue Creation
```markdown
**Issue Type**: [Bug Report / Feature Request / Documentation]

**Description**: Clear description of the issue or requested feature

**Steps to Reproduce** (for bugs):
1. Step one
2. Step two
3. Step three

**Expected Behavior**: What should happen

**Actual Behavior**: What actually happens

**Environment**:
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 2.0.1]
- CUDA: [e.g., 11.8]

**Additional Context**: Any other relevant information
```

#### 2. Pull Request Process

```bash
# Create feature branch
git checkout -b feature/amazing-new-feature

# Make changes with tests
# ... code changes ...

# Run full test suite
python -m pytest tests/ -v

# Check code quality
black arbor/ tests/
flake8 arbor/
mypy arbor/

# Commit with descriptive message
git commit -m "feat: add amazing new feature

- Implement core functionality
- Add comprehensive tests
- Update documentation
"

# Push and create PR
git push origin feature/amazing-new-feature
```

#### 3. Review Process

All PRs require:
- ✅ **Passing tests**: All CI checks must pass
- ✅ **Code review**: At least one maintainer approval
- ✅ **Documentation**: Update docs for new features
- ✅ **Changelog**: Add entry to CHANGELOG.md

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation locally
cd docs/
make html

# Serve documentation
python -m http.server 8000 -d _build/html/

# Open http://localhost:8000 in browser
```

### Release Process

```bash
# Update version
bump2version minor  # or major, patch

# Build package
python -m build

# Test package
python -m twine check dist/*

# Upload to PyPI (maintainers only)
python -m twine upload dist/*

# Create GitHub release
gh release create v1.2.0 --generate-notes
```

---

## Appendix

### Configuration Schema

Complete YAML configuration schema with all available options:

```yaml
# Model architecture configuration
model:
  # Basic architecture
  vocab_size: 128000                    # Vocabulary size
  hidden_size: 1024                     # Hidden dimension
  num_layers: 24                        # Number of transformer layers
  num_attention_heads: 16               # Number of attention heads
  intermediate_size: 4096               # FFN intermediate size
  max_position_embeddings: 131072       # Maximum sequence length
  
  # Regularization
  dropout: 0.1                          # General dropout rate
  attention_dropout: 0.1                # Attention dropout rate
  layer_norm_eps: 1e-5                  # Layer norm epsilon
  
  # Architecture options
  activation_function: "gelu"           # Activation function
  use_cache: true                       # Enable key-value caching
  tie_word_embeddings: true             # Tie input/output embeddings
  
  # Growth configuration
  growth:
    enabled: true                       # Enable dynamic growth
    factor: 2.0                         # Growth multiplier
    threshold: 0.95                     # Utilization threshold
    monitor_frequency: 100              # Steps between growth checks
    
# Adaptive context configuration
adaptive_context:
  enabled: true                         # Enable adaptive context
  
  # Router model configuration
  router_config:
    hidden_size: 256                    # Router hidden size
    num_layers: 3                       # Router depth
    num_attention_heads: 8              # Router attention heads
    dropout: 0.1                        # Router dropout
    
  # Task-specific context ranges
  task_contexts:
    chat: [1024, 4096]                  # Chat tasks
    code: [4096, 16384]                 # Code tasks
    reasoning: [8192, 32768]            # Reasoning tasks
    document: [16384, 131072]           # Document tasks
    creative: [2048, 16384]             # Creative tasks
    qa: [1024, 8192]                    # Q&A tasks
    summary: [4096, 32768]              # Summarization
    translation: [2048, 8192]           # Translation
    
  # Hardware constraints
  memory_constraints:
    max_memory_ratio: 0.8               # Max GPU memory usage
    fallback_context: 8192              # Fallback context length
    
# Dataset configuration
datasets:
  - name: "dataset1"                    # Dataset identifier
    source: "huggingface/dataset"       # HuggingFace dataset ID
    split: "train[:10000]"              # Dataset split
    text_column: "text"                 # Text column name
    
    # Preprocessing options
    preprocessing:
      max_length: 2048                  # Maximum sequence length
      min_length: 10                    # Minimum sequence length
      prefix: ""                        # Text prefix
      suffix: ""                        # Text suffix
      remove_columns: []                # Columns to remove
      
    # Filtering options
    filtering:
      min_words: 5                      # Minimum word count
      max_words: 1000                   # Maximum word count
      language: "en"                    # Language filter
      
# Training configuration
training:
  # Optimization
  learning_rate: 2e-5                   # Learning rate
  weight_decay: 0.01                    # Weight decay
  beta1: 0.9                           # Adam beta1
  beta2: 0.95                          # Adam beta2
  eps: 1e-8                            # Adam epsilon
  optimizer: "adamw"                    # Optimizer type
  
  # Schedule
  max_steps: 10000                      # Maximum training steps
  warmup_steps: 1000                    # Warmup steps
  lr_scheduler_type: "cosine"           # LR scheduler
  
  # Batch configuration
  per_device_train_batch_size: 4        # Per-device batch size
  per_device_eval_batch_size: 8         # Per-device eval batch size
  gradient_accumulation_steps: 4        # Gradient accumulation
  dataloader_num_workers: 4             # Data loader workers
  
  # Memory optimization
  fp16: false                           # FP16 training
  bf16: true                           # BF16 training
  gradient_checkpointing: true          # Gradient checkpointing
  dataloader_pin_memory: true           # Pin memory
  
  # Regularization
  max_grad_norm: 1.0                    # Gradient clipping
  label_smoothing: 0.0                  # Label smoothing
  
  # Monitoring
  logging_steps: 10                     # Logging frequency
  eval_steps: 500                       # Evaluation frequency
  save_steps: 1000                      # Save frequency
  save_total_limit: 3                   # Max checkpoints
  
  # Efficiency
  torch_compile: false                  # PyTorch compilation
  use_flash_attention: true             # Flash attention
  
# Post-training configuration (optional)
post_training:
  enabled: false                        # Enable post-training
  type: "fine_tune"                     # Type: fine_tune, instruct, domain_adapt
  
  # Post-training datasets
  datasets:
    - name: "specialized_data"
      source: "your-dataset"
      split: "train[:1000]"
      
  # Post-training parameters
  learning_rate: 5e-6                   # Lower learning rate
  max_steps: 500                        # Fewer steps
  per_device_batch_size: 2              # Smaller batch
  
  # LoRA configuration
  lora:
    enabled: true                       # Enable LoRA
    rank: 16                           # LoRA rank
    alpha: 32                          # LoRA alpha
    dropout: 0.1                       # LoRA dropout
    target_modules:                     # Target modules
      - "q_proj"
      - "v_proj"
      - "o_proj"
      
  # Layer freezing
  freeze_layers: []                     # Layers to freeze
  
  # Output configuration
  output_dir: "./post_training_output"  # Output directory
  save_merged_model: true               # Save merged model
  
# HuggingFace integration
huggingface:
  upload:
    enabled: true                       # Enable upload
    repository: "username/model-name"   # Repository name
    private: false                      # Private repository
    token: "${HF_TOKEN}"               # Authentication token
    
  download:
    cache_dir: null                     # Custom cache directory
    revision: "main"                    # Model revision
    
# Logging and monitoring
logging:
  # Weights & Biases
  wandb:
    enabled: true                       # Enable WandB
    project: "arbor-experiments"        # Project name
    entity: "your-team"                 # Team/entity
    tags: ["arbor", "transformer"]      # Tags
    notes: "Arbor training run"         # Run notes
    
  # Local logging
  output_dir: "./logs"                  # Log directory
  level: "INFO"                         # Log level
  
  # Advanced logging
  log_gradients: false                  # Log gradient norms
  log_layer_utilization: true          # Log layer utilization
  log_memory_usage: true               # Log memory usage
  save_attention_maps: false           # Save attention visualizations
  
# Hardware configuration
hardware:
  # GPU settings
  cuda_device: "auto"                   # CUDA device
  mixed_precision: "bf16"               # Mixed precision mode
  
  # Memory management
  max_memory_per_gpu: null              # Max memory per GPU
  memory_efficient_attention: true      # Memory efficient attention
  
  # Distributed training
  distributed: false                    # Enable distributed training
  num_gpus: 1                          # Number of GPUs
  
# Evaluation configuration
evaluation:
  strategy: "steps"                     # Evaluation strategy
  eval_steps: 500                       # Evaluation frequency
  eval_dataset_size: 1000               # Evaluation dataset size
  
  # Metrics
  metrics:
    - "perplexity"                      # Perplexity
    - "loss"                           # Loss
    
  # Generation evaluation
  generation:
    enabled: true                       # Enable generation eval
    prompts:                           # Evaluation prompts
      - "The quick brown fox"
      - "In a galaxy far, far away"
    max_length: 100                     # Max generation length
    temperature: 0.7                    # Generation temperature
```

### Environment Variables

```bash
# Required
export HF_TOKEN="your_huggingface_token"

# Optional but recommended
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="arbor-experiments"

# Custom cache directories
export HF_HOME="/custom/huggingface/cache"
export TRANSFORMERS_CACHE="/custom/transformers/cache"
export WANDB_CACHE_DIR="/custom/wandb/cache"

# CUDA settings
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TORCH_CUDA_ARCH_LIST="8.6"  # For RTX 30xx series

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS="8"

# Debugging
export TRANSFORMERS_VERBOSITY="info"
export DATASETS_VERBOSITY="info"
```

---

**End of Documentation**

*This documentation covers the complete Arbor system. For the latest updates and additional resources, visit the [Arbor GitHub repository](https://github.com/Noema-Research/Arbor).*

**Developed by [Noema Research](https://github.com/Noema-Research) | Version 1.0 | September 2025**
