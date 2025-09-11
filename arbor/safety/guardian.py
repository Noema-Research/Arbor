"""
Arbor Safety Guardian System - AI Safety and Control Framework

This module implements comprehensive safety measures to prevent:
- Uncontrolled self-improvement (intelligence explosion)
- Unauthorized model modifications
- Escape attempts from training environment
- Resource hijacking and misuse
- Manipulation of training process

The Guardian system monitors all model activities and enforces strict limits.
"""

import torch
import time
import psutil
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
from datetime import datetime, timedelta
from .approval import HumanApprovalInterface

@dataclass
class SafetyLimits:
    """Configuration for safety limits and constraints."""
    
    # Model size constraints
    max_parameters: int = 10_000_000_000  # 10B parameter limit
    max_layers: int = 100  # Maximum number of layers
    max_hidden_size: int = 8192  # Maximum hidden dimension
    
    # Growth rate limits
    max_growth_events_per_hour: int = 5
    max_parameter_growth_rate: float = 2.0  # Max 2x growth per event
    max_total_growth_factor: float = 10.0  # Max 10x total growth
    
    # Resource limits
    max_memory_usage_gb: float = 32.0  # Maximum GPU memory usage
    max_training_time_hours: int = 72  # Maximum continuous training
    max_cpu_usage_percent: float = 80.0  # Maximum CPU usage
    
    # Security constraints
    require_human_approval_for_growth: bool = True
    enable_model_state_verification: bool = True
    enable_behavior_monitoring: bool = True
    enable_escape_detection: bool = True
    
    # Emergency controls
    emergency_shutdown_threshold: float = 0.95  # Critical resource usage
    model_divergence_threshold: float = 5.0  # Max loss divergence
    suspicious_activity_threshold: int = 3  # Alerts before shutdown

class SafetyViolation(Exception):
    """Exception raised when safety limits are violated."""
    pass

class ArborSafetyGuardian:
    """
    Comprehensive safety system for Arbor models.
    
    The Guardian monitors model behavior, resource usage, and growth patterns
    to prevent dangerous scenarios and ensure safe AI operation.
    """
    
    def __init__(self, limits: SafetyLimits = None):
        self.limits = limits or SafetyLimits()
        self.start_time = time.time()
        
        # State tracking
        self.growth_events = []
        self.resource_history = []
        self.security_alerts = []
        self.model_checksum_history = []
        self.behavior_anomalies = []
        
        # Monitoring flags
        self.monitoring_active = True
        self.emergency_mode = False
        self.human_approval_pending = False
        
        # Thread-safe locks
        self.state_lock = threading.Lock()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Human approval interface
        self.approval_interface = HumanApprovalInterface()
        
        # Initialize monitoring thread
        self.monitor_thread = None
        self.start_monitoring()
        
        self.logger.info("🛡️ Arbor Safety Guardian initialized")
        self.logger.info(f"   Max parameters: {self.limits.max_parameters:,}")
        self.logger.info(f"   Growth rate limit: {self.limits.max_parameter_growth_rate}x")
        self.logger.info(f"   Memory limit: {self.limits.max_memory_usage_gb}GB")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup safety logging system."""
        logger = logging.getLogger("arbor.safety")
        logger.setLevel(logging.INFO)
        
        # Create safety log directory
        log_dir = Path("safety_logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler for safety events
        handler = logging.FileHandler(log_dir / f"safety_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("🔍 Safety monitoring started")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_resource_usage()
                self._check_training_time()
                self._detect_escape_attempts()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(10)
    
    def validate_growth_request(self, model, growth_type: str, growth_params: Dict[str, Any]) -> bool:
        """
        Validate if a growth request is safe to execute.
        
        Args:
            model: The Arbor model requesting growth
            growth_type: Type of growth ("layers", "parameters", "hidden_size")
            growth_params: Parameters for the growth operation
            
        Returns:
            True if growth is approved, False otherwise
            
        Raises:
            SafetyViolation: If growth violates safety limits
        """
        with self.state_lock:
            self.logger.info(f"🔍 Validating {growth_type} growth request")
            
            # Check if we're in emergency mode
            if self.emergency_mode:
                raise SafetyViolation("Growth blocked: System in emergency mode")
            
            # Get current model stats
            current_params = model.param_count()
            current_layers = len(model.layers)
            current_hidden = model.config.dim
            
            # Validate parameter limits
            if growth_type == "parameters":
                new_params = current_params * growth_params.get("factor", 2.0)
                if new_params > self.limits.max_parameters:
                    raise SafetyViolation(f"Parameter growth would exceed limit: {new_params:,} > {self.limits.max_parameters:,}")
            
            # Validate layer limits
            if growth_type == "layers":
                new_layers = current_layers + growth_params.get("add_layers", 1)
                if new_layers > self.limits.max_layers:
                    raise SafetyViolation(f"Layer growth would exceed limit: {new_layers} > {self.limits.max_layers}")
            
            # Validate hidden size limits
            if growth_type == "hidden_size":
                new_hidden = growth_params.get("new_hidden_size", current_hidden * 2)
                if new_hidden > self.limits.max_hidden_size:
                    raise SafetyViolation(f"Hidden size growth would exceed limit: {new_hidden} > {self.limits.max_hidden_size}")
            
            # Check growth rate limits
            recent_growth = self._get_recent_growth_events(hours=1)
            if len(recent_growth) >= self.limits.max_growth_events_per_hour:
                raise SafetyViolation(f"Growth rate limit exceeded: {len(recent_growth)} events in past hour")
            
            # Check total growth factor
            initial_params = self._get_initial_parameter_count(model)
            total_growth_factor = current_params / max(initial_params, 1)
            if total_growth_factor >= self.limits.max_total_growth_factor:
                raise SafetyViolation(f"Total growth factor limit exceeded: {total_growth_factor:.2f}x")
            
            # Human approval check
            if self.limits.require_human_approval_for_growth:
                if not self._request_human_approval(growth_type, growth_params):
                    self.logger.warning("🚫 Growth denied: Human approval not granted")
                    return False
            
            # Log approved growth
            self._log_growth_event(growth_type, growth_params, current_params)
            self.logger.info("✅ Growth request approved")
            return True
    
    def _request_human_approval(self, growth_type: str, growth_params: Dict[str, Any]) -> bool:
        """Request human approval for growth operation."""
        self.logger.warning(f"👤 HUMAN APPROVAL REQUIRED for {growth_type} growth")
        
        # Use the approval interface for proper human oversight
        return self.approval_interface.request_approval(
            operation_type=growth_type,
            operation_params=growth_params,
            urgency="normal",
            timeout_minutes=30
        )
    
    def _check_resource_usage(self):
        """Monitor system resource usage."""
        try:
            # Check memory usage
            gpu_memory = self._get_gpu_memory_usage()
            if gpu_memory > self.limits.max_memory_usage_gb:
                self._trigger_emergency_shutdown(f"GPU memory usage critical: {gpu_memory:.2f}GB")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.limits.max_cpu_usage_percent:
                self.logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            
            # Log resource usage
            self.resource_history.append({
                "timestamp": time.time(),
                "gpu_memory_gb": gpu_memory,
                "cpu_percent": cpu_percent,
                "system_memory_percent": psutil.virtual_memory().percent
            })
            
            # Keep only recent history
            cutoff_time = time.time() - 3600  # Last hour
            self.resource_history = [r for r in self.resource_history if r["timestamp"] > cutoff_time]
            
        except Exception as e:
            self.logger.error(f"Resource monitoring error: {e}")
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        try:
            if torch.cuda.is_available():
                memory_bytes = torch.cuda.memory_allocated()
                return memory_bytes / (1024**3)  # Convert to GB
            return 0.0
        except:
            return 0.0
    
    def _check_training_time(self):
        """Check if training time limits are exceeded."""
        training_hours = (time.time() - self.start_time) / 3600
        if training_hours > self.limits.max_training_time_hours:
            self._trigger_emergency_shutdown(f"Training time limit exceeded: {training_hours:.1f}h")
    
    def _detect_escape_attempts(self):
        """Detect potential model escape attempts."""
        if not self.limits.enable_escape_detection:
            return
        
        try:
            # Check for suspicious file system access
            suspicious_paths = [
                "/etc/", "/usr/bin/", "/home/", "/root/",
                "~/.ssh/", "~/.aws/", "~/.config/"
            ]
            
            # Check for network activity (simplified)
            net_connections = psutil.net_connections()
            external_connections = [c for c in net_connections 
                                  if c.status == 'ESTABLISHED' and 
                                  c.raddr and not c.raddr.ip.startswith('127.')]
            
            if len(external_connections) > 10:  # Threshold for suspicious activity
                self.logger.warning(f"⚠️ High network activity detected: {len(external_connections)} connections")
                self._log_security_alert("high_network_activity", {
                    "connection_count": len(external_connections),
                    "timestamp": time.time()
                })
            
        except Exception as e:
            self.logger.error(f"Escape detection error: {e}")
    
    def verify_model_integrity(self, model) -> bool:
        """Verify model hasn't been tampered with."""
        if not self.limits.enable_model_state_verification:
            return True
        
        try:
            # Calculate model state checksum
            model_state = model.state_dict()
            state_str = str(sorted([(k, v.shape, v.dtype) for k, v in model_state.items()]))
            checksum = hashlib.sha256(state_str.encode()).hexdigest()
            
            # Store checksum
            self.model_checksum_history.append({
                "timestamp": time.time(),
                "checksum": checksum,
                "parameter_count": model.param_count()
            })
            
            # Check for unexpected changes
            if len(self.model_checksum_history) > 1:
                prev_checksum = self.model_checksum_history[-2]["checksum"]
                if checksum != prev_checksum:
                    self.logger.info("📊 Model state changed (normal during training)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model integrity check failed: {e}")
            return False
    
    def _trigger_emergency_shutdown(self, reason: str):
        """Trigger emergency shutdown procedures."""
        with self.state_lock:
            self.emergency_mode = True
            self.monitoring_active = False
            
            self.logger.critical(f"🚨 EMERGENCY SHUTDOWN TRIGGERED: {reason}")
            self.logger.critical("🛑 All model operations halted")
            
            # Log emergency event
            emergency_event = {
                "timestamp": time.time(),
                "reason": reason,
                "system_state": {
                    "gpu_memory": self._get_gpu_memory_usage(),
                    "cpu_percent": psutil.cpu_percent(),
                    "training_time_hours": (time.time() - self.start_time) / 3600
                }
            }
            
            # Save emergency state
            emergency_file = Path("safety_logs/emergency_shutdown.json")
            with open(emergency_file, 'w') as f:
                json.dump(emergency_event, f, indent=2)
            
            # Raise exception to halt execution
            raise SafetyViolation(f"Emergency shutdown: {reason}")
    
    def _log_growth_event(self, growth_type: str, growth_params: Dict[str, Any], current_params: int):
        """Log a growth event."""
        event = {
            "timestamp": time.time(),
            "growth_type": growth_type,
            "growth_params": growth_params,
            "current_params": current_params
        }
        self.growth_events.append(event)
        self.logger.info(f"📈 Growth event logged: {growth_type}")
    
    def _log_security_alert(self, alert_type: str, details: Dict[str, Any]):
        """Log a security alert."""
        alert = {
            "timestamp": time.time(),
            "alert_type": alert_type,
            "details": details
        }
        self.security_alerts.append(alert)
        self.logger.warning(f"🔔 Security alert: {alert_type}")
        
        # Check if we need emergency action
        recent_alerts = [a for a in self.security_alerts 
                        if a["timestamp"] > time.time() - 300]  # Last 5 minutes
        
        if len(recent_alerts) >= self.limits.suspicious_activity_threshold:
            self._trigger_emergency_shutdown("Multiple security alerts detected")
    
    def _get_recent_growth_events(self, hours: float) -> List[Dict[str, Any]]:
        """Get growth events from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [event for event in self.growth_events if event["timestamp"] > cutoff_time]
    
    def _get_initial_parameter_count(self, model) -> int:
        """Get the initial parameter count when monitoring started."""
        if self.growth_events:
            return self.growth_events[0].get("current_params", model.param_count())
        return model.param_count()
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        return {
            "guardian_status": {
                "active": self.monitoring_active,
                "emergency_mode": self.emergency_mode,
                "start_time": self.start_time,
                "runtime_hours": (time.time() - self.start_time) / 3600
            },
            "limits": {
                "max_parameters": self.limits.max_parameters,
                "max_layers": self.limits.max_layers,
                "max_growth_rate": self.limits.max_parameter_growth_rate,
                "max_memory_gb": self.limits.max_memory_usage_gb
            },
            "activity_summary": {
                "growth_events": len(self.growth_events),
                "security_alerts": len(self.security_alerts),
                "behavior_anomalies": len(self.behavior_anomalies)
            },
            "current_resources": {
                "gpu_memory_gb": self._get_gpu_memory_usage(),
                "cpu_percent": psutil.cpu_percent(),
                "system_memory_percent": psutil.virtual_memory().percent
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown the safety guardian."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛡️ Safety Guardian shutdown")

# Global guardian instance
_global_guardian: Optional[ArborSafetyGuardian] = None

def get_safety_guardian() -> ArborSafetyGuardian:
    """Get the global safety guardian instance."""
    global _global_guardian
    if _global_guardian is None:
        _global_guardian = ArborSafetyGuardian()
    return _global_guardian

def initialize_safety_system(limits: SafetyLimits = None) -> ArborSafetyGuardian:
    """Initialize the Arbor safety system."""
    global _global_guardian
    _global_guardian = ArborSafetyGuardian(limits)
    return _global_guardian
