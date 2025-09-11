"""
Comprehensive metrics tracking for Arbor training
"""

import time
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific step."""
    step: int
    epoch: int
    timestamp: float
    
    # Loss metrics
    train_loss: float
    val_loss: Optional[float] = None
    
    # Learning metrics
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    
    # Model architecture
    num_layers: int = 0
    num_parameters: int = 0
    ffn_dimensions: List[int] = None
    
    # Layer utilization (for growth monitoring)
    layer_utilization: Dict[int, float] = None
    avg_utilization: float = 0.0
    max_utilization: float = 0.0
    
    # Performance metrics
    tokens_per_second: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    
    # Growth events
    growth_events: List[Dict] = None
    
    def __post_init__(self):
        if self.ffn_dimensions is None:
            self.ffn_dimensions = []
        if self.layer_utilization is None:
            self.layer_utilization = {}
        if self.growth_events is None:
            self.growth_events = []

class MetricsTracker:
    """Tracks and stores training metrics over time."""
    
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.metrics_history: List[TrainingMetrics] = []
        self.current_epoch = 0
        self.start_time = time.time()
        
        # Live metrics for dashboard
        self.live_metrics = {}
        
    def log_metrics(self, metrics: TrainingMetrics):
        """Log new training metrics."""
        metrics.timestamp = time.time()
        self.metrics_history.append(metrics)
        
        # Update live metrics
        self.live_metrics = asdict(metrics)
        
        # Save to disk periodically
        if len(self.metrics_history) % 10 == 0:
            self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to disk."""
        metrics_file = self.save_dir / "training_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump([asdict(m) for m in self.metrics_history], f, indent=2)
    
    def load_metrics(self) -> List[TrainingMetrics]:
        """Load metrics from disk."""
        metrics_file = self.save_dir / "training_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                self.metrics_history = [TrainingMetrics(**item) for item in data]
        
        return self.metrics_history
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame for analysis."""
        if not self.metrics_history:
            return pd.DataFrame()
        
        data = []
        for metrics in self.metrics_history:
            row = asdict(metrics)
            # Flatten layer utilization
            if metrics.layer_utilization:
                for layer_idx, util in metrics.layer_utilization.items():
                    row[f'layer_{layer_idx}_util'] = util
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_loss_trends(self) -> Dict[str, List]:
        """Get loss trends for plotting."""
        steps = [m.step for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history if m.val_loss is not None]
        val_steps = [m.step for m in self.metrics_history if m.val_loss is not None]
        
        return {
            'steps': steps,
            'train_loss': train_losses,
            'val_steps': val_steps,
            'val_loss': val_losses
        }
    
    def get_growth_timeline(self) -> Dict[str, List]:
        """Get model growth timeline."""
        steps = []
        parameters = []
        layers = []
        
        for metrics in self.metrics_history:
            steps.append(metrics.step)
            parameters.append(metrics.num_parameters / 1e6)  # Convert to millions
            layers.append(metrics.num_layers)
        
        return {
            'steps': steps,
            'parameters_m': parameters,
            'num_layers': layers
        }
    
    def get_utilization_trends(self) -> Dict[str, Any]:
        """Get layer utilization trends."""
        steps = []
        avg_utils = []
        max_utils = []
        layer_utils = {}
        
        for metrics in self.metrics_history:
            if metrics.layer_utilization:
                steps.append(metrics.step)
                avg_utils.append(metrics.avg_utilization)
                max_utils.append(metrics.max_utilization)
                
                # Track individual layers
                for layer_idx, util in metrics.layer_utilization.items():
                    if layer_idx not in layer_utils:
                        layer_utils[layer_idx] = {'steps': [], 'utilization': []}
                    layer_utils[layer_idx]['steps'].append(metrics.step)
                    layer_utils[layer_idx]['utilization'].append(util)
        
        return {
            'steps': steps,
            'avg_utilization': avg_utils,
            'max_utilization': max_utils,
            'layer_utilization': layer_utils
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            'avg_tokens_per_sec': np.mean([m.tokens_per_second for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage_gb for m in recent_metrics]),
            'avg_gpu_utilization': np.mean([m.gpu_utilization for m in recent_metrics]),
            'total_training_time': time.time() - self.start_time,
            'total_steps': len(self.metrics_history)
        }
    
    def get_growth_events(self) -> List[Dict]:
        """Get all growth events from training."""
        growth_events = []
        
        for metrics in self.metrics_history:
            if metrics.growth_events:
                for event in metrics.growth_events:
                    event['step'] = metrics.step
                    event['timestamp'] = metrics.timestamp
                    growth_events.append(event)
        
        return growth_events
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect training anomalies."""
        anomalies = []
        
        if len(self.metrics_history) < 10:
            return anomalies
        
        recent_losses = [m.train_loss for m in self.metrics_history[-10:]]
        avg_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        
        # Check for loss spikes
        latest_loss = recent_losses[-1]
        if latest_loss > avg_loss + 3 * std_loss:
            anomalies.append({
                'type': 'loss_spike',
                'step': self.metrics_history[-1].step,
                'value': latest_loss,
                'threshold': avg_loss + 3 * std_loss,
                'message': f'Training loss spike detected: {latest_loss:.4f}'
            })
        
        # Check for gradient explosion
        latest_grad_norm = self.metrics_history[-1].grad_norm
        if latest_grad_norm > 10.0:
            anomalies.append({
                'type': 'gradient_explosion',
                'step': self.metrics_history[-1].step,
                'value': latest_grad_norm,
                'threshold': 10.0,
                'message': f'Gradient explosion detected: {latest_grad_norm:.4f}'
            })
        
        # Check for memory issues
        latest_memory = self.metrics_history[-1].memory_usage_gb
        if latest_memory > 20.0:  # Assuming 24GB GPU
            anomalies.append({
                'type': 'high_memory',
                'step': self.metrics_history[-1].step,
                'value': latest_memory,
                'threshold': 20.0,
                'message': f'High memory usage: {latest_memory:.2f}GB'
            })
        
        return anomalies
