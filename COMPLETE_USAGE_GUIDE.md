# 🌳 Arbor: Complete Usage Guide

Welcome to Arbor - the adaptive, growing transformer architecture with comprehensive agentic capabilities and real-time monitoring.

## 🚀 Quick Start

### 1. Basic Training with Dashboard

```bash
# Clone and setup
git clone <arbor-repo>
cd arbor-o1-living-ai

# Install dependencies  
pip install torch transformers streamlit plotly matplotlib psutil

# Launch training with dashboard
python launch_training_dashboard.py
```

**Result**: 
- Training starts automatically
- Dashboard opens at http://localhost:8501
- Watch your model grow in real-time!

### 2. Agentic AI Usage

```python
from arbor.agents import ArborAgent

# Create an intelligent agent
agent = ArborAgent(
    model_name="gpt-4",
    tools=["python", "bash", "web_search", "file_ops"]
)

# Execute complex tasks
result = agent.execute("Analyze this dataset and create visualizations")
print(result.content)
```

### 3. Layer Growth in Action

```python
from arbor.modeling import ArborTransformer, ArborConfig

# Configure adaptive growth
config = ArborConfig(
    num_layers=24,        # Start with 24 layers
    max_layers=64,        # Grow up to 64 layers
    layer_growth_enabled=True,
    layer_growth_threshold=0.92  # Grow when 92% utilized
)

model = ArborTransformer(config)
# Model automatically grows during training!
```

## 📊 Dashboard Features

### Live Metrics Tab
- **Training Loss**: Real-time loss curves with trend analysis
- **Learning Rate**: Adaptive learning rate scheduling
- **Gradient Norms**: Gradient clipping and stability monitoring
- **Performance**: Tokens/second, memory usage, GPU utilization

### Architecture Tab  
- **Visual Model**: Interactive model architecture display
- **Layer Utilization**: Per-layer activation heatmaps
- **Growth Events**: Timeline of parameter and layer additions
- **Resource Usage**: Memory and compute allocation

### Growth Tracking Tab
- **Parameter Evolution**: Watch parameters grow over time
- **Layer Timeline**: Visual layer addition history  
- **Efficiency Metrics**: Growth impact on performance
- **Capacity Planning**: Utilization trends and predictions

### Alerts Tab
- **Real-time Notifications**: Training anomalies and events
- **Severity Levels**: Info, Warning, Error, Critical alerts
- **Custom Thresholds**: Configurable alert conditions
- **Notification Methods**: Email, webhooks, dashboard alerts

### Analytics Tab
- **Training Statistics**: Comprehensive training summaries
- **Growth Analysis**: Parameter vs. performance correlations
- **System Metrics**: Hardware utilization and bottlenecks
- **Export Tools**: PDF reports and data downloads

## 🤖 Agentic Capabilities

### Available Tools

#### Code Execution
```python
# Python code execution with environment management
agent.execute_python("""
import pandas as pd
data = pd.read_csv('data.csv')
print(data.describe())
""")
```

#### System Operations
```python
# Bash command execution with safety checks
agent.execute_bash("find . -name '*.py' | head -10")
```

#### Web Intelligence
```python
# Web search and content retrieval
result = agent.web_search("latest AI research transformers")
agent.fetch_url("https://arxiv.org/abs/2023.xxxxx")
```

#### File Operations
```python
# Advanced file manipulation
agent.read_file("config.json", encoding="utf-8")
agent.write_file("output.txt", content)
agent.search_files(pattern="*.py", grep="class.*Transformer")
```

### Model Context Protocol (MCP)

```python
# Connect to external services via MCP
from arbor.agents.mcp import MCPClient

client = MCPClient("filesystem")
result = client.call_tool("read_file", {"path": "data.csv"})
```

### Advanced Agentic Workflows

```python
# Multi-step reasoning and execution
agent = ArborAgent(tools=["all"])

result = agent.execute("""
1. Analyze the CSV file in /data/sales.csv
2. Create visualizations showing trends
3. Generate a summary report 
4. Save results to /reports/analysis.html
""")
```

## 🌱 Dynamic Growth System

### Layer Growth Mechanics

The Arbor model intelligently adds layers when needed:

1. **Utilization Monitoring**: Tracks activation patterns across layers
2. **Growth Triggers**: Adds layers when utilization exceeds threshold
3. **Strategic Insertion**: Places new layers optimally in the architecture
4. **Performance Optimization**: Maintains efficiency during growth

### Growth Configuration

```python
config = ArborConfig(
    # Layer growth settings
    layer_growth_enabled=True,
    min_layers=24,                    # Starting point
    max_layers=64,                    # Growth limit
    layer_growth_threshold=0.92,      # Utilization trigger
    layer_growth_factor=4,            # Layers added per growth
    layer_growth_cooldown=100,        # Steps between growths
    
    # Parameter growth settings  
    growth_enabled=True,
    growth_factor=2.0,                # FFN expansion factor
    growth_threshold=0.90,            # Growth trigger
    growth_cooldown=50,               # Cooldown period
)
```

### Monitoring Growth

```python
# Access growth metrics during training
growth_stats = model.get_growth_statistics()
print(f"Current layers: {growth_stats['num_layers']}")
print(f"Growth events: {growth_stats['growth_events']}")
print(f"Average utilization: {growth_stats['avg_utilization']:.3f}")
```

## 📈 Training Integration

### Complete Training Loop

```python
from arbor.modeling import ArborTransformer, ArborConfig
from arbor.tracking import TrainingMonitor
from arbor.data import SyntheticDataset

# Setup model with growth
config = ArborConfig(num_layers=24, max_layers=64, layer_growth_enabled=True)
model = ArborTransformer(config)

# Setup tracking
monitor = TrainingMonitor(save_dir="logs", update_interval=1.0)
monitor.start_monitoring()

# Training loop with growth tracking
for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(batch, return_dict=True)
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Track metrics (includes growth monitoring)
    metrics = monitor.log_training_step(
        step=step,
        train_loss=loss.item(),
        model_state=model.get_state_dict()
    )
    
    # Model grows automatically based on utilization!
```

### Advanced Training Features

```python
# Custom growth strategies
model.set_growth_strategy("progressive")  # Add layers progressively
model.set_growth_strategy("targeted")     # Add layers where needed most

# Manual growth control
if should_grow_condition():
    model.grow_layers(num_layers=4, position="optimal")

# Growth callbacks
def on_layer_growth(old_layers, new_layers, growth_info):
    print(f"🌱 Grew from {old_layers} to {new_layers} layers")
    
model.register_growth_callback(on_layer_growth)
```

## 🛠️ Advanced Configuration

### Environment Setup

```bash
# Install with all features
pip install torch transformers streamlit plotly matplotlib psutil requests beautifulsoup4

# For development
pip install pytest black flake8 mypy

# For agentic features
pip install model-context-protocol langchain openai anthropic
```

### Custom Agents

```python
from arbor.agents import BaseAgent, Tool

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.add_tool(CustomTool())
    
    def custom_reasoning(self, prompt):
        # Implement custom reasoning logic
        return self.llm.generate(prompt)

# Register custom tools
@Tool.register("custom_tool")
def my_custom_tool(arg1: str, arg2: int) -> str:
    """Custom tool that does something specific."""
    return f"Processed {arg1} with {arg2}"
```

### Production Deployment

```python
# Distributed training setup
from arbor.training import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    config=config,
    world_size=4,
    backend="nccl"
)

# Launch with monitoring
trainer.train_with_monitoring(
    dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    tracking_enabled=True,
    dashboard_port=8501
)
```

## 🎯 Use Cases

### 1. Research & Development
- **Adaptive Architecture Experiments**: Test growth strategies
- **Performance Analysis**: Monitor efficiency during growth
- **Ablation Studies**: Compare growth vs. static architectures

### 2. Production Training
- **Large-Scale Models**: Efficient scaling for production
- **Resource Optimization**: Adaptive resource allocation
- **Monitoring & Alerts**: Production-ready monitoring

### 3. Agentic Applications
- **Code Generation**: AI assistants for programming
- **Data Analysis**: Automated data science workflows
- **Content Creation**: Multi-modal content generation

### 4. Interactive Development
- **Live Training**: Watch models learn in real-time
- **Experiment Tracking**: Compare different configurations
- **Performance Debugging**: Identify training bottlenecks

## 📚 Additional Resources

### Documentation
- `docs/AGENTIC_AI_GUIDE.md` - Complete agentic AI documentation
- `docs/LAYER_GROWTH_IMPLEMENTATION.md` - Growth system details
- `examples/README.md` - Example usage and tutorials

### Code Structure
```
arbor-o1-living-ai/
├── arbor/
│   ├── modeling/          # Core transformer architecture  
│   ├── agents/           # Agentic AI capabilities
│   ├── tracking/         # Training monitoring & dashboard
│   └── data/            # Data utilities
├── examples/            # Usage examples
├── docs/               # Documentation
└── tests/              # Test suite
```

### Getting Help
1. **Check the examples**: `examples/` directory has working code
2. **Read the docs**: Comprehensive guides in `docs/`
3. **Run the dashboard**: Visual debugging with the Streamlit interface
4. **Enable debugging**: Set logging levels for detailed output

## 🎉 What's Next?

1. **Try the Quick Start**: Get familiar with the basic features
2. **Explore the Dashboard**: Understand the monitoring capabilities  
3. **Experiment with Growth**: Test different growth configurations
4. **Build Agents**: Create your own agentic AI applications
5. **Scale Up**: Use Arbor for your production training workflows

Welcome to the future of adaptive AI! 🚀
