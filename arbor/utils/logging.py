"""
Logging utilities and growth event tracking.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file to log to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ] + ([logging.FileHandler(log_file)] if log_file else [])
    )
    
    # Get logger for Arbor
    logger = logging.getLogger("arbor")
    logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def log_growth_event(
    step: int,
    old_param_count: int,
    new_param_count: int,
    growth_details: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log a growth event with detailed information.
    
    Args:
        step: Training step when growth occurred
        old_param_count: Parameter count before growth
        new_param_count: Parameter count after growth
        growth_details: Detailed information about the growth
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("arbor.growth")
    
    param_increase = new_param_count - old_param_count
    growth_ratio = new_param_count / old_param_count if old_param_count > 0 else 1.0
    
    logger.info(f"🌱 GROWTH EVENT at step {step}")
    logger.info(f"   Parameters: {old_param_count:,} → {new_param_count:,} (+{param_increase:,})")
    logger.info(f"   Growth ratio: {growth_ratio:.2f}x")
    logger.info(f"   Layers affected: {growth_details.get('layer_indices', [])}")
    logger.info(f"   Hidden units added: {growth_details.get('add_hidden', 0)}")
    logger.info(f"   Reason: {growth_details.get('reason', 'unknown')}")
    
    # Log to Weights & Biases if available
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "growth/step": step,
                "growth/old_param_count": old_param_count,
                "growth/new_param_count": new_param_count,
                "growth/param_increase": param_increase,
                "growth/growth_ratio": growth_ratio,
                "growth/layers_affected": len(growth_details.get('layer_indices', [])),
                "growth/hidden_units_added": growth_details.get('add_hidden', 0),
            }, step=step)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to log growth event to wandb: {e}")


def setup_wandb(
    project_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    tags: Optional[list] = None,
    notes: Optional[str] = None,
) -> None:
    """
    Initialize Weights & Biases logging.
    
    Args:
        project_name: W&B project name
        experiment_name: Experiment name/run name
        config: Configuration dictionary
        tags: Optional tags for the run
        notes: Optional notes for the run
    """
    try:
        import wandb
        
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=tags,
            notes=notes,
        )
        
        print(f"W&B logging initialized: {project_name}/{experiment_name}")
        
    except ImportError:
        print("Warning: wandb not available, skipping W&B logging")
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")


def log_experiment_start(
    config: Dict[str, Any],
    model_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log experiment start information.
    
    Args:
        config: Experiment configuration
        model_info: Model information
        dataset_info: Dataset information
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("arbor.experiment")
    
    logger.info("🚀 Starting Arbor experiment")
    logger.info("=" * 50)
    
    # Log model info
    logger.info("📊 Model Configuration:")
    logger.info(f"   Architecture: {model_info.get('type', 'ArborTransformer')}")
    logger.info(f"   Parameters: {model_info.get('total_parameters', 0):,}")
    logger.info(f"   Layers: {model_info.get('num_layers', 0)}")
    logger.info(f"   Dimension: {model_info.get('dim', 0)}")
    logger.info(f"   FFN dimension: {model_info.get('ffn_dim', 0)}")
    logger.info(f"   Attention heads: {model_info.get('num_heads', 0)}")
    
    # Log dataset info
    logger.info("📚 Dataset Information:")
    logger.info(f"   Type: {dataset_info.get('type', 'unknown')}")
    logger.info(f"   Size: {dataset_info.get('size', 0):,} examples")
    logger.info(f"   Vocabulary size: {dataset_info.get('vocab_size', 0):,}")
    logger.info(f"   Sequence length: {dataset_info.get('seq_length', 0)}")
    
    # Log training config
    logger.info("⚙️ Training Configuration:")
    logger.info(f"   Max steps: {config.get('max_steps', 0):,}")
    logger.info(f"   Batch size: {config.get('batch_size', 0)}")
    logger.info(f"   Learning rate: {config.get('learning_rate', 0):.2e}")
    logger.info(f"   Growth enabled: {config.get('growth', {}).get('enabled', False)}")
    
    if config.get('growth', {}).get('enabled', False):
        growth_config = config['growth']
        logger.info("🌱 Growth Configuration:")
        logger.info(f"   Add hidden: {growth_config.get('add_hidden', 0)}")
        logger.info(f"   Max events: {growth_config.get('max_events', 0)}")
        logger.info(f"   Cooldown steps: {growth_config.get('cooldown_steps', 0)}")
        logger.info(f"   Triggers: {len(growth_config.get('triggers', []))}")
    
    logger.info("=" * 50)


def log_experiment_end(
    results: Dict[str, Any],
    training_time: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log experiment completion information.
    
    Args:
        results: Final experiment results
        training_time: Total training time in seconds
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("arbor.experiment")
    
    logger.info("🏁 Experiment completed")
    logger.info("=" * 50)
    
    # Training summary
    logger.info("📈 Training Summary:")
    logger.info(f"   Final step: {results.get('final_step', 0):,}")
    logger.info(f"   Training time: {training_time:.1f}s ({training_time/3600:.2f}h)")
    logger.info(f"   Final loss: {results.get('final_val_loss', 0):.4f}")
    logger.info(f"   Best loss: {results.get('best_val_loss', 0):.4f}")
    logger.info(f"   Final perplexity: {results.get('final_val_perplexity', 0):.2f}")
    
    # Growth summary
    if results.get('growth_events', 0) > 0:
        logger.info("🌳 Growth Summary:")
        logger.info(f"   Growth events: {results.get('growth_events', 0)}")
        logger.info(f"   Final parameters: {results.get('final_param_count', 0):,}")
        
        initial_params = results.get('initial_param_count', results.get('final_param_count', 0))
        if initial_params > 0:
            growth_ratio = results.get('final_param_count', 0) / initial_params
            logger.info(f"   Growth ratio: {growth_ratio:.2f}x")
    
    logger.info("=" * 50)


def save_experiment_log(
    experiment_name: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    growth_history: list,
    save_dir: str = "experiment_logs",
) -> str:
    """
    Save a complete experiment log to JSON.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        results: Final results
        growth_history: List of growth events
        save_dir: Directory to save logs
        
    Returns:
        Path to saved log file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.json"
    log_path = os.path.join(save_dir, log_filename)
    
    experiment_log = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config": config,
        "results": results,
        "growth_history": growth_history,
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "arbor_version": "0.1.0",
        }
    }
    
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2, default=str)
    
    print(f"Experiment log saved: {log_path}")
    return log_path


def create_progress_callback(
    logger: Optional[logging.Logger] = None,
    log_every: int = 100,
):
    """
    Create a progress logging callback for training.
    
    Args:
        logger: Logger instance
        log_every: Log progress every N steps
        
    Returns:
        Callback function
    """
    if logger is None:
        logger = logging.getLogger("arbor.progress")
    
    def callback(step: int, metrics: Dict[str, Any]):
        if step % log_every == 0:
            log_str = f"Step {step:,}"
            
            # Add key metrics
            if "loss" in metrics:
                log_str += f" | Loss: {metrics['loss']:.4f}"
            if "lr" in metrics:
                log_str += f" | LR: {metrics['lr']:.2e}"
            if "param_count" in metrics:
                log_str += f" | Params: {metrics['param_count']:,}"
            if "tokens_per_sec" in metrics:
                log_str += f" | Tokens/s: {metrics['tokens_per_sec']:.0f}"
            
            logger.info(log_str)
    
    return callback


def log_model_architecture(
    model,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log detailed model architecture information.
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger("arbor.model")
    
    logger.info("🏗️ Model Architecture:")
    
    # Count parameters by module type
    param_counts = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        param_count = sum(p.numel() for p in module.parameters())
        
        if param_count > 0:
            if module_type not in param_counts:
                param_counts[module_type] = 0
            param_counts[module_type] += param_count
    
    # Log parameter breakdown
    total_params = sum(param_counts.values())
    logger.info(f"   Total parameters: {total_params:,}")
    
    for module_type, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / total_params if total_params > 0 else 0
        logger.info(f"   {module_type}: {count:,} ({percentage:.1f}%)")
    
    # Log memory estimate
    memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    logger.info(f"   Estimated memory: {memory_mb:.1f} MB")


class ExperimentTracker:
    """
    Class for tracking experiment metrics and events.
    """
    
    def __init__(self, experiment_name: str, save_dir: str = "experiment_logs"):
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.start_time = datetime.now()
        
        self.metrics_history = []
        self.growth_events = []
        self.checkpoints = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for a training step."""
        metrics_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(metrics_entry)
    
    def log_growth_event(self, step: int, growth_details: Dict[str, Any]) -> None:
        """Log a growth event."""
        growth_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **growth_details
        }
        self.growth_events.append(growth_entry)
    
    def log_checkpoint(self, step: int, checkpoint_path: str, metrics: Dict[str, Any]) -> None:
        """Log a checkpoint save."""
        checkpoint_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "path": checkpoint_path,
            "metrics": metrics,
        }
        self.checkpoints.append(checkpoint_entry)
    
    def save_summary(self) -> str:
        """Save experiment summary to file."""
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "metrics_history": self.metrics_history,
            "growth_events": self.growth_events,
            "checkpoints": self.checkpoints,
        }
        
        summary_path = os.path.join(self.save_dir, f"{self.experiment_name}_summary.json")
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary_path
