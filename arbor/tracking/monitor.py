"""
Training Monitor - Integrates metrics tracking, visualization, and alerts
"""

import threading
import time
from typing import Dict, Optional, Callable
import psutil
import GPUtil
from .metrics import MetricsTracker, TrainingMetrics
from .alerts import AlertSystem
from .visualizer import NetworkVisualizer, GrowthVisualizer

class TrainingMonitor:
    """Comprehensive training monitor that tracks everything."""
    
    def __init__(self, 
                 save_dir: str = "training_logs",
                 email_config: Optional[Dict] = None,
                 webhook_url: Optional[str] = None,
                 update_interval: float = 1.0):
        
        self.metrics_tracker = MetricsTracker(save_dir)
        self.alert_system = AlertSystem(email_config, webhook_url)
        self.network_visualizer = NetworkVisualizer()
        self.growth_visualizer = GrowthVisualizer()
        
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # System monitoring
        self.start_time = time.time()
        
    def start_monitoring(self):
        """Start background monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def log_training_step(self, 
                         step: int,
                         epoch: int,
                         train_loss: float,
                         learning_rate: float,
                         grad_norm: float,
                         model_state: Dict,
                         val_loss: Optional[float] = None) -> TrainingMetrics:
        """Log a training step with all metrics."""
        
        # Get system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Get GPU metrics if available
        gpu_utilization = 0.0
        memory_usage_gb = memory_info.used / (1024**3)
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_utilization = gpu.load * 100
                memory_usage_gb = gpu.memoryUsed / 1024  # Convert to GB
        except:
            pass
        
        # Create metrics object
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            num_layers=model_state.get('num_layers', 0),
            num_parameters=model_state.get('num_parameters', 0),
            ffn_dimensions=model_state.get('ffn_dimensions', []),
            layer_utilization=model_state.get('layer_utilization', {}),
            avg_utilization=model_state.get('avg_utilization', 0.0),
            max_utilization=model_state.get('max_utilization', 0.0),
            tokens_per_second=model_state.get('tokens_per_second', 0.0),
            memory_usage_gb=memory_usage_gb,
            gpu_utilization=gpu_utilization,
            growth_events=model_state.get('growth_events', [])
        )
        
        # Log metrics
        self.metrics_tracker.log_metrics(metrics)
        
        # Check for alerts
        history = [m.__dict__ for m in self.metrics_tracker.metrics_history[-10:]]
        alerts = self.alert_system.check_training_metrics(metrics.__dict__, history)
        
        # Check growth events
        if metrics.growth_events:
            growth_alerts = self.alert_system.check_growth_events(metrics.growth_events)
            alerts.extend(growth_alerts)
        
        return metrics
    
    def get_current_status(self) -> Dict:
        """Get current training status."""
        if not self.metrics_tracker.metrics_history:
            return {"status": "No training data available"}
        
        latest_metrics = self.metrics_tracker.metrics_history[-1]
        performance_stats = self.metrics_tracker.get_performance_stats()
        alert_summary = self.alert_system.get_alert_summary()
        
        return {
            "latest_metrics": latest_metrics.__dict__,
            "performance_stats": performance_stats,
            "alert_summary": alert_summary,
            "training_duration": time.time() - self.start_time,
            "total_steps": len(self.metrics_tracker.metrics_history)
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get all data needed for the dashboard."""
        loss_trends = self.metrics_tracker.get_loss_trends()
        growth_timeline = self.metrics_tracker.get_growth_timeline()
        utilization_trends = self.metrics_tracker.get_utilization_trends()
        performance_stats = self.metrics_tracker.get_performance_stats()
        growth_events = self.metrics_tracker.get_growth_events()
        anomalies = self.metrics_tracker.detect_anomalies()
        
        return {
            "loss_trends": loss_trends,
            "growth_timeline": growth_timeline,
            "utilization_trends": utilization_trends,
            "performance_stats": performance_stats,
            "growth_events": growth_events,
            "anomalies": anomalies,
            "alerts": self.alert_system.get_active_alerts(),
            "status": self.get_current_status()
        }
    
    def create_architecture_visualization(self, model_config: Dict) -> str:
        """Create and save architecture visualization."""
        if not self.metrics_tracker.metrics_history:
            return ""
        
        latest_metrics = self.metrics_tracker.metrics_history[-1]
        
        fig = self.network_visualizer.visualize_architecture(
            num_layers=latest_metrics.num_layers,
            hidden_dim=model_config.get('hidden_dim', 1024),
            num_heads=model_config.get('num_heads', 16),
            ffn_dims=latest_metrics.ffn_dimensions or [4096] * latest_metrics.num_layers,
            layer_utilization=latest_metrics.layer_utilization
        )
        
        # Save figure
        filename = f"architecture_step_{latest_metrics.step}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        return filename
    
    def export_training_report(self) -> str:
        """Export comprehensive training report."""
        dashboard_data = self.get_dashboard_data()
        
        # Create report
        report = f"""
# Arbor Training Report

## Training Summary
- Total Steps: {dashboard_data['status']['total_steps']}
- Training Duration: {dashboard_data['status']['training_duration']:.2f} seconds
- Current Status: {dashboard_data['status']['latest_metrics']['step']} steps completed

## Model Architecture
- Layers: {dashboard_data['status']['latest_metrics']['num_layers']}
- Parameters: {dashboard_data['status']['latest_metrics']['num_parameters'] / 1e6:.1f}M
- Average Utilization: {dashboard_data['status']['latest_metrics']['avg_utilization']:.3f}

## Performance Metrics
"""
        
        perf_stats = dashboard_data['performance_stats']
        for key, value in perf_stats.items():
            report += f"- {key.replace('_', ' ').title()}: {value:.2f}\n"
        
        report += f"\n## Growth Events\n"
        growth_events = dashboard_data['growth_events']
        for i, event in enumerate(growth_events[-5:]):  # Last 5 events
            report += f"{i+1}. Step {event['step']}: {event.get('message', 'Growth event')}\n"
        
        report += f"\n## Active Alerts\n"
        alerts = dashboard_data['alerts']
        if alerts:
            for alert in alerts[-5:]:  # Last 5 alerts
                report += f"- {alert.severity.value.upper()}: {alert.message}\n"
        else:
            report += "No active alerts\n"
        
        # Save report
        report_filename = f"training_report_{int(time.time())}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        return report_filename
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # System health checks could go here
                # For now, just sleep
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(self.update_interval)
    
    def add_custom_alert_handler(self, handler: Callable):
        """Add custom alert handler."""
        self.alert_system.add_alert_handler(handler)
    
    def set_alert_thresholds(self, thresholds: Dict):
        """Update alert thresholds."""
        self.alert_system.thresholds.update(thresholds)
