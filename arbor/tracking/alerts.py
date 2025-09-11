"""
Alert system for monitoring Arbor training
"""

import time
from enum import Enum
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class TrainingAlert:
    """Represents a training alert."""
    timestamp: float
    severity: AlertSeverity
    category: str
    message: str
    step: int
    metrics: Dict
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'step': self.step,
            'metrics': self.metrics,
            'resolved': self.resolved
        }

class AlertSystem:
    """Comprehensive alert system for training monitoring."""
    
    def __init__(self, 
                 email_config: Optional[Dict] = None,
                 webhook_url: Optional[str] = None):
        self.alerts: List[TrainingAlert] = []
        self.alert_handlers: List[Callable] = []
        self.email_config = email_config
        self.webhook_url = webhook_url
        
        # Alert thresholds
        self.thresholds = {
            'loss_spike_factor': 3.0,
            'gradient_norm_max': 10.0,
            'memory_usage_max': 20.0,  # GB
            'utilization_growth_threshold': 0.92,
            'learning_rate_min': 1e-8,
            'tokens_per_sec_min': 10.0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def add_alert_handler(self, handler: Callable[[TrainingAlert], None]):
        """Add a custom alert handler."""
        self.alert_handlers.append(handler)
    
    def check_training_metrics(self, current_metrics: Dict, history: List[Dict]) -> List[TrainingAlert]:
        """Check metrics and generate alerts."""
        new_alerts = []
        
        # Loss monitoring
        if len(history) >= 10:
            recent_losses = [m['train_loss'] for m in history[-10:]]
            avg_loss = sum(recent_losses) / len(recent_losses)
            std_loss = (sum((x - avg_loss) ** 2 for x in recent_losses) / len(recent_losses)) ** 0.5
            
            current_loss = current_metrics.get('train_loss', 0)
            if current_loss > avg_loss + self.thresholds['loss_spike_factor'] * std_loss:
                alert = TrainingAlert(
                    timestamp=time.time(),
                    severity=AlertSeverity.WARNING,
                    category="loss_anomaly",
                    message=f"Training loss spike detected: {current_loss:.4f} (threshold: {avg_loss + self.thresholds['loss_spike_factor'] * std_loss:.4f})",
                    step=current_metrics.get('step', 0),
                    metrics={'current_loss': current_loss, 'avg_loss': avg_loss, 'std_loss': std_loss}
                )
                new_alerts.append(alert)
        
        # Gradient monitoring
        grad_norm = current_metrics.get('grad_norm', 0)
        if grad_norm > self.thresholds['gradient_norm_max']:
            alert = TrainingAlert(
                timestamp=time.time(),
                severity=AlertSeverity.ERROR,
                category="gradient_explosion",
                message=f"Gradient explosion detected: {grad_norm:.4f}",
                step=current_metrics.get('step', 0),
                metrics={'grad_norm': grad_norm}
            )
            new_alerts.append(alert)
        
        # Memory monitoring
        memory_usage = current_metrics.get('memory_usage_gb', 0)
        if memory_usage > self.thresholds['memory_usage_max']:
            alert = TrainingAlert(
                timestamp=time.time(),
                severity=AlertSeverity.WARNING,
                category="high_memory",
                message=f"High memory usage: {memory_usage:.2f}GB",
                step=current_metrics.get('step', 0),
                metrics={'memory_usage_gb': memory_usage}
            )
            new_alerts.append(alert)
        
        # Layer utilization monitoring (for growth)
        layer_utilization = current_metrics.get('layer_utilization', {})
        if layer_utilization:
            high_util_layers = [idx for idx, util in layer_utilization.items() 
                              if util > self.thresholds['utilization_growth_threshold']]
            
            if len(high_util_layers) >= len(layer_utilization) * 0.8:  # 80% of layers
                alert = TrainingAlert(
                    timestamp=time.time(),
                    severity=AlertSeverity.INFO,
                    category="growth_opportunity",
                    message=f"Model ready for growth: {len(high_util_layers)} layers above {self.thresholds['utilization_growth_threshold']} utilization",
                    step=current_metrics.get('step', 0),
                    metrics={'high_util_layers': high_util_layers, 'total_layers': len(layer_utilization)}
                )
                new_alerts.append(alert)
        
        # Learning rate monitoring
        learning_rate = current_metrics.get('learning_rate', 0)
        if learning_rate < self.thresholds['learning_rate_min']:
            alert = TrainingAlert(
                timestamp=time.time(),
                severity=AlertSeverity.WARNING,
                category="low_learning_rate",
                message=f"Learning rate very low: {learning_rate:.2e}",
                step=current_metrics.get('step', 0),
                metrics={'learning_rate': learning_rate}
            )
            new_alerts.append(alert)
        
        # Performance monitoring
        tokens_per_sec = current_metrics.get('tokens_per_second', 0)
        if tokens_per_sec < self.thresholds['tokens_per_sec_min']:
            alert = TrainingAlert(
                timestamp=time.time(),
                severity=AlertSeverity.WARNING,
                category="low_performance",
                message=f"Low training speed: {tokens_per_sec:.2f} tokens/sec",
                step=current_metrics.get('step', 0),
                metrics={'tokens_per_second': tokens_per_sec}
            )
            new_alerts.append(alert)
        
        # Process new alerts
        for alert in new_alerts:
            self._process_alert(alert)
        
        return new_alerts
    
    def check_growth_events(self, growth_events: List[Dict]) -> List[TrainingAlert]:
        """Monitor growth events and create informational alerts."""
        new_alerts = []
        
        for event in growth_events:
            if event.get('timestamp', 0) > time.time() - 60:  # Recent events (last minute)
                alert = TrainingAlert(
                    timestamp=event.get('timestamp', time.time()),
                    severity=AlertSeverity.INFO,
                    category="model_growth",
                    message=f"Model growth event: {event.get('type', 'unknown')} at step {event.get('step', 0)}",
                    step=event.get('step', 0),
                    metrics=event
                )
                new_alerts.append(alert)
                self._process_alert(alert)
        
        return new_alerts
    
    def _process_alert(self, alert: TrainingAlert):
        """Process and dispatch an alert."""
        self.alerts.append(alert)
        self.logger.info(f"Alert generated: {alert.severity.value.upper()} - {alert.message}")
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
        
        # Send email for critical alerts
        if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] and self.email_config:
            self._send_email_alert(alert)
        
        # Send webhook notification
        if self.webhook_url:
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: TrainingAlert):
        """Send email notification for alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"Arbor Training Alert - {alert.severity.value.upper()}"
            
            body = f"""
Training Alert Generated:

Severity: {alert.severity.value.upper()}
Category: {alert.category}
Message: {alert.message}
Step: {alert.step}
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}

Metrics: {alert.metrics}

This is an automated message from the Arbor training monitoring system.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.sendmail(self.email_config['from'], self.email_config['to'], msg.as_string())
            server.quit()
            
            self.logger.info(f"Email alert sent for: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: TrainingAlert):
        """Send webhook notification for alert."""
        try:
            import requests
            
            payload = {
                'text': f"🚨 Arbor Training Alert: {alert.message}",
                'severity': alert.severity.value,
                'category': alert.category,
                'step': alert.step,
                'timestamp': alert.timestamp
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook alert sent for: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[TrainingAlert]:
        """Get current active alerts."""
        alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_index: int, resolution_message: str = ""):
        """Mark an alert as resolved."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
            self.logger.info(f"Alert resolved: {self.alerts[alert_index].message}")
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts."""
        total_alerts = len(self.alerts)
        active_alerts = len([a for a in self.alerts if not a.resolved])
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([a for a in self.alerts 
                                                 if a.severity == severity and not a.resolved])
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'severity_breakdown': severity_counts,
            'recent_alerts': self.get_active_alerts()[-5:]  # Last 5 alerts
        }
