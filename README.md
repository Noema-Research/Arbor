# 🌱 Arbor-o### Key Innovation:

**The Arbor architecture** represents a paradigm shift from **static** to **dynamic neural architectures**. Instead of pre-defining a fixed model size, the Arbor architecture lets models determine their own capacity needs based on the complexity of the learning task. Arbor-o1 is the first model built on this architecture.Living AI

**Dynamic Neural Networks That Grow During Training**

Arbor-o1 is a revolutionary language model built on the **Arbor architecture** - a transformer design that implements **dynamic capacity expansion**. The Arbor architecture enables neural networks to increase their size during training when encountering learning plateaus or challenges.

> **Note**: **Arbor-o1** is the model name/release, while **Arbor** refers to the underlying transformer architecture with dynamic growth capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Key Features

- **Dynamic Architecture Growth**: FFN layers expand when training plateaus or specific triggers fire
- **Smart Growth Management**: Multiple triggers (validation plateau, gradient norm, loss spikes)
- **Optimizer Safety**: Proper handling of new parameters in optimizer state
- **Production Ready**: Mixed precision, gradient accumulation, checkpointing, distributed training support
- **Research Friendly**: Comprehensive logging, growth event tracking, reproducible experiments

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/arbor-o1-living-ai.git
cd arbor-o1-living-ai

# Quick setup (checks dependencies and runs test)
python setup.py

# Or install manually
pip install torch numpy matplotlib tqdm omegaconf wandb seaborn pytest jupyter
```

### 2. Run Quick Demo

```bash
# Simple demo
python quick_demo.py

# Interactive notebook
jupyter notebook notebooks/demo.ipynb

# Full training example
python scripts/train.py --config configs/small.yaml
```

### Basic Usage

```python
from arbor.modeling.model import ArborTransformer, ArborConfig
from arbor.growth.manager import GrowthManager
from arbor.growth.triggers import PlateauTrigger

# Create a model using the Arbor architecture
config = ArborConfig(
    vocab_size=10000,
    n_embd=512,
    n_layer=6,
    n_head=8,
    d_ff=2048,  # This will expand during training!
    max_length=1024
)

model = ArborTransformer(config)
print(f"Initial size: {model.param_count():,} parameters")

# Set up growth management for Arbor-o1 training
growth_manager = GrowthManager(
    triggers=[PlateauTrigger(patience=100, threshold=0.01)],
    growth_factor=1.3,
    min_steps_between_growth=500
)

# Train with growth enabled
trainer = Trainer(
    model=model,
    growth_manager=growth_manager,
    config=training_config,
    device="cuda"
)

trainer.train(dataloader)
print(f"Final size: {model.param_count():,} parameters")
print(f"Growth events: {len(growth_manager.growth_history)}")
```

## 📁 Repository Structure

```
arbor-o1-living-ai/
├── 📦 arbor/                     # Core package
│   ├── 🧠 modeling/              # Model architecture
│   │   ├── layers.py             # Expandable layers
│   │   ├── blocks.py             # Transformer blocks
│   │   └── model.py              # ArborTransformer
│   ├── 🌱 growth/                # Growth management
│   │   ├── triggers.py           # Growth triggers
│   │   └── manager.py            # Growth coordination
│   ├── 🚂 train/                 # Training infrastructure
│   │   ├── trainer.py            # Main trainer class
│   │   ├── optimizer_utils.py    # Optimizer handling
│   │   └── checkpointing.py      # Save/load logic
│   ├── 📊 data/                  # Data handling
│   │   ├── tokenizers.py         # Tokenization
│   │   └── datasets.py           # Dataset classes
│   └── 🛠️ utils/                 # Utilities
│       ├── metrics.py            # Metrics computation
│       └── logging.py            # Logging setup
├── ⚙️ configs/                   # Configuration files
│   ├── small.yaml               # Small model config
│   ├── base.yaml                # Base model config
│   └── growth.yaml              # Growth-focused config
├── 📜 scripts/                   # Training scripts
│   ├── train.py                 # Main training script
│   ├── eval.py                  # Evaluation script
│   └── prep_data.py             # Data preparation
├── 🧪 tests/                     # Unit tests
│   ├── test_layers.py           # Layer tests
│   ├── test_growth_manager.py   # Growth system tests
│   └── test_smoke_train.py      # Training pipeline tests
├── 📓 notebooks/                 # Jupyter notebooks
│   └── demo.ipynb               # Interactive demo
├── 🔧 setup.py                  # Setup script
├── 🏃 run_tests.py              # Test runner
├── ⚡ quick_demo.py             # Simple demo
└── 📋 pyproject.toml            # Project configuration
```

## 📖 Documentation

### Core Components

#### 1. **ArborTransformer** - The Growing Architecture
```python
# Arbor architecture that can expand during training
model = ArborTransformer(config)

# Manual growth
model.grow(growth_factor=1.5)

# Check size
print(f"Parameters: {model.param_count():,}")
```

#### 2. **ExpandableFFN** - Dynamic Feed-Forward Networks
```python
# FFN layer that can increase hidden size
ffn = ExpandableFFN(d_model=512, d_ff=2048)

# Expand hidden dimensions
ffn.grow(new_d_ff=3072)
```

#### 3. **GrowthManager** - Orchestrates Expansion
```python
# Manages when and how growth occurs in Arbor architecture
manager = GrowthManager(
    triggers=[PlateauTrigger(), GradientNormTrigger()],
    growth_factor=1.3,
    min_steps_between_growth=100
)

# Monitor training and trigger growth
result = manager.step(loss=current_loss, grad_norm=current_grad_norm)
if result:
    manager.grow_model(model, result, optimizer)
```

#### 4. **Growth Triggers** - Smart Detection Systems

**PlateauTrigger**: Detects learning plateaus
```python
trigger = PlateauTrigger(
    patience=50,      # Wait 50 steps for improvement
    threshold=0.01    # Minimum improvement required
)
```

**GradientNormTrigger**: Responds to gradient issues
```python
trigger = GradientNormTrigger(
    threshold=5.0,    # Gradient norm threshold
    patience=5        # Violations before triggering
)
```

**LossSpikeTrigger**: Detects training instabilities
```python
trigger = LossSpikeTrigger(
    spike_threshold=1.5,    # 1.5x increase = spike
    history_length=10       # Compare with last 10 losses
)
```

### Training Pipeline

#### Configuration System
```yaml
# configs/custom.yaml
model:
  vocab_size: 50000
  n_embd: 768
  n_layer: 12
  n_head: 12
  d_ff: 3072
  max_length: 1024

training:
  max_steps: 10000
  learning_rate: 3e-4
  batch_size: 32
  use_amp: true

growth:
  enabled: true
  growth_factor: 1.25
  min_steps_between_growth: 200
  triggers:
    - type: plateau
      patience: 100
      threshold: 0.005
    - type: gradient_norm
      threshold: 10.0
      patience: 5
```

#### Training Script
```bash
# Train with config file
python scripts/train.py --config configs/custom.yaml --output_dir ./checkpoints

# Train with custom parameters
python scripts/train.py 
    --model.vocab_size 30000 
    --training.max_steps 5000 
    --growth.growth_factor 1.4 
    --output_dir ./experiments/run_1
```

#### Evaluation
```bash
# Evaluate trained model
python scripts/eval.py 
    --checkpoint_path ./checkpoints/checkpoint-1000.pt 
    --data_path ./data/eval_set.pt 
    --output_dir ./results

# Compare multiple models
python scripts/eval.py 
    --compare ./checkpoints/checkpoint-1000.pt ./checkpoints/checkpoint-2000.pt 
    --output_dir ./comparison
```

### Data Preparation

#### Synthetic Data (for demos)
```python
from arbor.data import SyntheticDataset, ArborTokenizer

tokenizer = ArborTokenizer("gpt2", vocab_size=10000)
dataset = SyntheticDataset(
    size=1000,
    vocab_size=10000,
    sequence_length=512,
    tokenizer=tokenizer,
    complexity="medium"
)
```

#### Custom Data
```python
# Prepare your own dataset
python scripts/prep_data.py 
    --input_file ./raw_data.txt 
    --output_file ./processed_data.pt 
    --vocab_size 50000 
    --sequence_length 1024
```

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py layers
python run_tests.py growth
python run_tests.py smoke

# Run with pytest directly
pytest tests/ -v
```

## 📊 Monitoring and Visualization

### Weights & Biases Integration
```python
# Automatic W&B logging during training
trainer = Trainer(
    model=model,
    config=training_config,
    run_name="arbor_experiment",
    project="arbor-o1"
)
```

### Growth Visualization
```python
# Analyze growth history
from arbor.utils.metrics import compute_growth_metrics

metrics = compute_growth_metrics(growth_manager.growth_history)
print(f"Growth rate: {metrics['growth_rate']:.2f}x")
print(f"Avg steps between growth: {metrics['avg_steps_between_growth']:.1f}")
```

## 🔬 Research Applications

### Adaptive Model Sizing
- **Problem**: How big should a model be for a given task?
- **Solution**: Start small and let the model determine its needs

### Efficient Large Model Training
- **Problem**: Large models waste compute on simple examples
- **Solution**: Gradual capacity increase as complexity demands

### Continual Learning
- **Problem**: Fixed models struggle with new domains
- **Solution**: Dynamic expansion to accommodate new knowledge

### Resource Optimization
- **Problem**: Over-provisioning compute for unknown requirements
- **Solution**: Just-in-time capacity allocation

## 🎯 Use Cases

1. **🚀 Research**: Explore dynamic architectures and adaptive training
2. **💡 Production**: Efficient model development with uncertain requirements
3. **🎓 Education**: Understand growth dynamics and capacity needs
4. **🔧 Experimentation**: Rapid prototyping with adaptive models

## 🛠️ Advanced Usage

### Custom Growth Triggers
```python
from arbor.growth.triggers import BaseTrigger

class CustomTrigger(BaseTrigger):
    def __init__(self, custom_threshold):
        super().__init__()
        self.threshold = custom_threshold
    
    def should_trigger(self, **metrics):
        # Your custom logic here
        custom_metric = metrics.get('custom_metric', 0)
        return custom_metric > self.threshold
    
    def reset(self):
        # Reset internal state
        pass
```

### Custom Growth Strategies
```python
# Custom growth factor per layer type
class LayerSpecificGrowthManager(GrowthManager):
    def grow_model(self, model, growth_info, optimizer=None):
        # Different growth rates for different layers
        for i, layer in enumerate(model.transformer.layers):
            layer_growth_factor = 1.2 + (i * 0.1)  # Deeper layers grow more
            layer.mlp.grow(int(layer.mlp.d_ff * layer_growth_factor))
        
        return model.param_count()
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **🐛 Bug Reports**: Use GitHub issues with detailed reproduction steps
2. **💡 Feature Requests**: Describe the use case and proposed implementation  
3. **🔧 Pull Requests**: Include tests and documentation updates
4. **📖 Documentation**: Help improve our docs and examples

### Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/yourusername/arbor-o1-living-ai.git
cd arbor-o1-living-ai
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests before submitting
python run_tests.py
```

## 📚 Citation

If you use Arbor-o1 in your research, please cite:

```bibtex
@software{arbor_o1_2024,
  title={Arbor-o1: Dynamic Neural Networks That Grow During Training},
  author={Arbor Research Team},
  year={2024},
  url={https://github.com/yourusername/arbor-o1-living-ai},
  note={A production-ready implementation of dynamic capacity expansion for transformer models}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by research in dynamic neural architectures
- Built on the excellent PyTorch ecosystem
- Community feedback and contributions

---

**🌱 Arbor-o1: Where AI Learns to Grow!**

*Join us in exploring the future of adaptive artificial intelligence.*

## 📦 Installation

```bash
git clone https://github.com/arbor-research/arbor-o1-living-ai.git
cd arbor-o1-living-ai
pip install -e .
```

For development with optional dependencies:
```bash
pip install -e ".[dev]"
```

## 🏃‍♂️ Quick Start

### 1. Run a Tiny Demo

```bash
# Generate synthetic data
python scripts/prep_data.py --output_dir data/synthetic --vocab_size 1000 --seq_length 512 --num_sequences 10000

# Train a small model with growth enabled
python scripts/train.py --config configs/arbor_small.yaml --exp demo_growth

# Train baseline (no growth) for comparison
python scripts/train.py --config configs/arbor_small.yaml --exp demo_baseline --growth.enabled false
```

### 2. Evaluate Models

```bash
# Evaluate a trained model
python scripts/eval.py --checkpoint_path checkpoints/demo_growth/final_model.pt --data_path data/synthetic

# Compare growth vs baseline
python scripts/eval.py --checkpoint_path checkpoints/demo_baseline/final_model.pt --data_path data/synthetic
```

### 3. Explore Growth Events

```bash
# Launch Jupyter notebook to visualize growth timeline
jupyter notebook notebooks/demo.ipynb
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=arbor --cov-report=html

# Run specific test categories
pytest tests/test_layers.py -v
pytest tests/test_growth_manager.py -v
pytest tests/test_smoke_train.py -v
```

## 🏗️ Architecture

### Core Components

1. **ExpandableFFN**: Feed-forward layers that can grow hidden dimensions
2. **ArborBlock**: Transformer block with expandable FFN
3. **ArborTransformer**: Full model with growth capabilities
4. **GrowthManager**: Orchestrates when and how to expand the model
5. **Trainer**: Training loop with growth event handling

### Growth Triggers

- **Plateau Trigger**: Expands when validation loss plateaus
- **Gradient Norm Trigger**: Expands when gradients become too small
- **Slice Spike Trigger**: Expands when specific data slices show loss spikes

### Model Growth Process

```
1. Monitor metrics during training
2. Trigger fires based on configured conditions
3. GrowthManager selects layers and expansion amount
4. Model expands (new parameters initialized)
5. Optimizer state updated for new parameters
6. Training continues with larger capacity
```

## 📊 Configuration

Three main configs are provided:

- `configs/arbor_small.yaml`: Tiny demo model (~10M params)
- `configs/arbor_base.yaml`: Development baseline (~100M params)  
- `configs/arbor_growth.yaml`: Growth-enabled configuration

### Example Growth Config

```yaml
model:
  layers: 6
  dim: 512
  ffn_dim: 2048
  heads: 8
  vocab_size: 10000

growth:
  enabled: true
  add_hidden: 256
  max_events: 6
  cooldown_steps: 5000
  triggers:
    - type: plateau
      window_steps: 1000
      eps: 0.001
```

## 🔬 Experiments

### Baseline vs Growth Comparison

```bash
# Run controlled experiment
python scripts/train.py --config configs/arbor_base.yaml --exp baseline_fixed
python scripts/train.py --config configs/arbor_growth.yaml --exp growth_enabled

# Compare results
python scripts/compare_experiments.py --exp1 baseline_fixed --exp2 growth_enabled
```

### Ablation Studies

Test different growth triggers:
```bash
# Plateau only
python scripts/train.py --config configs/ablation_plateau.yaml --exp plateau_only

# Gradient norm only  
python scripts/train.py --config configs/ablation_gradnorm.yaml --exp gradnorm_only

# Combined triggers
python scripts/train.py --config configs/ablation_combined.yaml --exp combined
```

## 📈 Metrics & Logging

All experiments log to Weights & Biases by default:

- Training/validation loss and perplexity
- Model parameter count over time
- Growth events with timestamps and reasons
- FLOPs estimates and efficiency metrics
- Gradient norms and learning rates

Set `WANDB_PROJECT` environment variable to organize experiments:
```bash
export WANDB_PROJECT="arbor-o1-experiments"
```

## 🛠️ Development

### Code Structure

```
arbor/
├── modeling/           # Core model components
│   ├── layers.py      # ExpandableFFN, utilities
│   ├── block.py       # ArborBlock 
│   └── model.py       # ArborTransformer
├── growth/            # Growth management
│   ├── manager.py     # GrowthManager
│   └── triggers.py    # Growth triggers
├── train/             # Training infrastructure  
│   ├── train_loop.py  # Trainer class
│   ├── optimizer_utils.py # Optimizer handling
│   └── checkpoint.py  # Save/load with growth
├── data/              # Data loading
└── utils/             # Utilities and metrics
```

### Adding New Growth Triggers

1. Implement trigger class in `arbor/growth/triggers.py`
2. Register in `GrowthManager.create_trigger()`
3. Add config schema and tests
4. Document in experiments section

### Scaling to Large Models

For models >1B parameters:

```bash
# Enable DeepSpeed ZeRO
python scripts/train.py --config configs/arbor_large.yaml --deepspeed configs/deepspeed_config.json

# Use FSDP
python scripts/train.py --config configs/arbor_large.yaml --fsdp
```

## 📋 Experiment Checklist

Planning your experiments? Use this checklist:

- [ ] Baseline fixed-size model training
- [ ] Growth-enabled training with same initial size
- [ ] Ablation study on growth triggers
- [ ] Comparison of growth policies (which layers to expand)
- [ ] Parameter efficiency analysis (params vs performance)
- [ ] Training stability analysis around growth events
- [ ] Scaling behavior with different initial sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run the test suite (`pytest`)
5. Format code (`black . && isort .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📚 Citation

If you use Arbor-o1 in your research, please cite:

```bibtex
@software{arbor_o1_2024,
  title={Arbor-o1: Dynamic Transformer Architecture with Capacity Growth},
  author={Arbor Research Team},
  year={2024},
  url={https://github.com/arbor-research/arbor-o1-living-ai}
}
```

## 🔮 Future Work

- Multi-dimensional growth (attention heads, layers, dimensions)
- Pruning capabilities for model compression
- Growth prediction using meta-learning
- Integration with neural architecture search
- Support for multimodal architectures

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ FAQ

**Q: How much overhead does growth add to training?**
A: Growth events are infrequent (every few thousand steps) and add minimal overhead (<1% typically).

**Q: Can I use this with my existing transformer code?**
A: Yes! The `ExpandableFFN` can be dropped into most transformer implementations.

**Q: Does this work with distributed training?**
A: Yes, we support both DeepSpeed and PyTorch FSDP with proper synchronization.

**Q: How do I choose growth hyperparameters?**
A: Start with our defaults and tune based on your dataset size and compute budget. See `EXPERIMENTS.md` for guidance.

---

**Made with 🌳 by the Arbor Research Team**
