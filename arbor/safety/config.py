"""
Safety configuration for Arbor models.

This file contains safety limits and human approval requirements
for controlling AI behavior and preventing dangerous scenarios.
"""

# Default safety configuration
SAFETY_CONFIG = {
    "limits": {
        # Model size constraints (prevent runaway growth)
        "max_parameters": 50_000_000_000,  # 50B parameter hard limit
        "max_layers": 200,                 # Maximum number of layers
        "max_hidden_size": 16384,          # Maximum hidden dimension
        
        # Growth rate limits (prevent intelligence explosion)
        "max_growth_events_per_hour": 3,   # Max 3 growth events per hour
        "max_parameter_growth_rate": 1.5,  # Max 1.5x growth per event
        "max_total_growth_factor": 5.0,    # Max 5x total growth from start
        
        # Resource limits (prevent resource hijacking)
        "max_memory_usage_gb": 64.0,       # Maximum GPU memory usage
        "max_training_time_hours": 168,    # Maximum 1 week continuous training
        "max_cpu_usage_percent": 90.0,     # Maximum CPU usage
        
        # Security constraints
        "require_human_approval_for_growth": True,
        "enable_model_state_verification": True,
        "enable_behavior_monitoring": True,
        "enable_escape_detection": True,
        
        # Emergency controls
        "emergency_shutdown_threshold": 0.95,   # Critical resource usage
        "model_divergence_threshold": 10.0,     # Max loss divergence
        "suspicious_activity_threshold": 5      # Alerts before shutdown
    },
    
    "monitoring": {
        "resource_check_interval": 5,      # Check resources every 5 seconds
        "behavior_log_interval": 60,       # Log behavior every minute
        "integrity_check_interval": 300,   # Check model integrity every 5 minutes
        "alert_retention_hours": 24        # Keep alerts for 24 hours
    },
    
    "alerts": {
        "email_notifications": False,       # Email alerts disabled by default
        "webhook_url": None,               # No webhook by default
        "log_level": "INFO",               # Logging level
        "emergency_contacts": []           # Emergency contact list
    }
}

# Production safety configuration (more restrictive)
PRODUCTION_SAFETY_CONFIG = {
    "limits": {
        "max_parameters": 10_000_000_000,   # 10B parameter limit for production
        "max_layers": 100,
        "max_hidden_size": 8192,
        "max_growth_events_per_hour": 1,    # Only 1 growth event per hour
        "max_parameter_growth_rate": 1.2,   # Slower growth rate
        "max_total_growth_factor": 2.0,     # More conservative total growth
        "max_memory_usage_gb": 32.0,
        "max_training_time_hours": 48,      # Shorter training time
        "max_cpu_usage_percent": 75.0,
        "require_human_approval_for_growth": True,
        "enable_model_state_verification": True,
        "enable_behavior_monitoring": True,
        "enable_escape_detection": True,
        "emergency_shutdown_threshold": 0.85,
        "model_divergence_threshold": 5.0,
        "suspicious_activity_threshold": 3
    },
    "monitoring": {
        "resource_check_interval": 2,       # More frequent checking
        "behavior_log_interval": 30,
        "integrity_check_interval": 120,
        "alert_retention_hours": 72
    },
    "alerts": {
        "email_notifications": True,        # Enable email alerts in production
        "webhook_url": "https://your-monitoring-system.com/webhook",
        "log_level": "DEBUG",
        "emergency_contacts": [
            "security@your-company.com",
            "ai-safety@your-company.com"
        ]
    }
}

# Development safety configuration (more permissive)
DEVELOPMENT_SAFETY_CONFIG = {
    "limits": {
        "max_parameters": 100_000_000_000,  # Higher limits for development
        "max_layers": 500,
        "max_hidden_size": 32768,
        "max_growth_events_per_hour": 10,
        "max_parameter_growth_rate": 3.0,
        "max_total_growth_factor": 20.0,
        "max_memory_usage_gb": 128.0,
        "max_training_time_hours": 720,     # 30 days for long experiments
        "max_cpu_usage_percent": 95.0,
        "require_human_approval_for_growth": False,  # Allow automatic growth
        "enable_model_state_verification": True,
        "enable_behavior_monitoring": True,
        "enable_escape_detection": True,
        "emergency_shutdown_threshold": 0.98,
        "model_divergence_threshold": 20.0,
        "suspicious_activity_threshold": 10
    },
    "monitoring": {
        "resource_check_interval": 10,
        "behavior_log_interval": 120,
        "integrity_check_interval": 600,
        "alert_retention_hours": 12
    },
    "alerts": {
        "email_notifications": False,
        "webhook_url": None,
        "log_level": "WARNING",
        "emergency_contacts": []
    }
}
