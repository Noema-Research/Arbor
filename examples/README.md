# Arbor Examples

This directory contains example scripts and demonstrations of Arbor's capabilities.

## Available Examples

### 🎯 Training with Dashboard
- **File**: `training_with_dashboard.py`
- **Purpose**: Demonstrates how to use Arbor's training tracking dashboard
- **Features**: 
  - Live metrics visualization
  - Layer growth monitoring
  - Alert system integration
  - Performance tracking

### 🚀 Quick Launch
- **File**: `../launch_training_dashboard.py` (in root directory)
- **Purpose**: Easy launcher for training + dashboard
- **Usage**: Starts both training and Streamlit dashboard automatically

## Running the Examples

### Method 1: Manual Launch

1. **Start the dashboard** (in one terminal):
   ```bash
   streamlit run arbor/tracking/dashboard.py
   ```

2. **Run training** (in another terminal):
   ```bash
   python examples/training_with_dashboard.py
   ```

3. **View the dashboard**:
   - Open http://localhost:8501 in your browser
   - Watch live metrics and model architecture

### Method 2: Automatic Launch

1. **Use the launcher script**:
   ```bash
   python launch_training_dashboard.py
   ```

2. **View the dashboard**:
   - Dashboard automatically opens at http://localhost:8501
   - Both processes run simultaneously

## What You'll See

### 🖥️ Training Output
- Real-time loss metrics
- Layer growth events
- Parameter count updates
- Training progress

### 📊 Dashboard Features
- **Live Metrics**: Loss trends, learning rate, gradient norms
- **Architecture View**: Visual representation of model layers
- **Growth Tracking**: Layer and parameter growth over time  
- **Alerts**: Automatic notifications for training events
- **Analytics**: Performance statistics and trends

### 🔔 Alert Types
- High loss spikes
- Layer growth events
- Memory usage warnings
- Training anomalies

## Customizing the Examples

### Modify Training Parameters
Edit `training_with_dashboard.py`:
```python
config = ArborConfig(
    vocab_size=10000,
    dim=512,
    num_layers=24,          # Starting layers
    max_layers=64,          # Maximum layers
    layer_growth_threshold=0.92,  # Growth trigger
    # ... other parameters
)
```

### Customize Dashboard
The dashboard reads from `training_logs/` directory:
- Metrics are saved as JSON files
- Model states are tracked automatically
- Alerts are logged with timestamps

### Add Custom Metrics
In your training loop:
```python
# Log custom metrics
metrics = monitor.log_training_step(
    step=step,
    epoch=epoch,
    train_loss=loss.item(),
    custom_metrics={
        'accuracy': accuracy,
        'perplexity': perplexity,
        # Add your metrics here
    }
)
```

## Troubleshooting

### Common Issues

1. **Dashboard not loading**:
   - Check if Streamlit is installed: `pip install streamlit`
   - Verify port 8501 is available
   - Check firewall settings

2. **Import errors**:
   - Ensure you're in the project root directory
   - Install required dependencies: `pip install -r requirements.txt`
   - Check Python path includes the project

3. **No data in dashboard**:
   - Training must run for metrics to appear
   - Check `training_logs/` directory exists
   - Verify training script is logging metrics

### Performance Tips

1. **For faster training simulation**:
   - Reduce the delay in `training_with_dashboard.py`
   - Use smaller batch sizes
   - Limit the number of training steps

2. **For better dashboard performance**:
   - Adjust update intervals in the dashboard
   - Reduce the number of data points plotted
   - Use the refresh controls appropriately

## Next Steps

1. **Explore the dashboard**: Try different tabs and controls
2. **Modify parameters**: Experiment with growth settings
3. **Add custom metrics**: Track your own training metrics
4. **Integration**: Use the tracking system in your own training scripts

For more information, see the main documentation in the `docs/` directory.
