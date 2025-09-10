"""
Checkpointing utilities for saving and loading models with growth history.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import os
import json
from pathlib import Path

from ..modeling.model import ArborTransformer, ArborConfig
from ..growth.manager import GrowthManager


def save_checkpoint(
    model: ArborTransformer,
    optimizer: torch.optim.Optimizer,
    growth_manager: Optional[GrowthManager],
    step: int,
    loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a complete checkpoint including model, optimizer, and growth state.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        growth_manager: Growth manager (optional)
        step: Current training step
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint so far
        additional_info: Additional information to save
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint data
    checkpoint = {
        "step": step,
        "loss": loss,
        "model_config": model.config.__dict__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "growth_history": model.get_growth_history(),
    }
    
    # Add growth manager state if available
    if growth_manager is not None:
        checkpoint["growth_manager_state"] = growth_manager.save_state()
    
    # Add additional info
    if additional_info:
        checkpoint["additional_info"] = additional_info
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save as latest
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)
    
    # Save as best if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        torch.save(checkpoint, best_path)
        
        # Also save just the model for inference
        model_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save({
            "config": model.config.__dict__,
            "state_dict": model.state_dict(),
            "growth_history": model.get_growth_history(),
        }, model_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[ArborTransformer] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    growth_manager: Optional[GrowthManager] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model, optimizer, and growth state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into (optional, will create if None)
        optimizer: Optimizer to load state into (optional)
        growth_manager: Growth manager to load state into (optional)
        device: Device to load tensors to
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        Dictionary with loaded information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model if not provided
    if model is None:
        config = ArborConfig(**checkpoint["model_config"])
        model = ArborTransformer(config)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        model = model.to(device)
        
        # Restore growth history
        if "growth_history" in checkpoint:
            model.growth_history = checkpoint["growth_history"]
    
    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    # Load growth manager state
    if growth_manager is not None and "growth_manager_state" in checkpoint:
        try:
            growth_manager.load_state(checkpoint["growth_manager_state"])
        except Exception as e:
            print(f"Warning: Could not load growth manager state: {e}")
    
    result = {
        "model": model,
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float('inf')),
        "growth_history": checkpoint.get("growth_history", []),
    }
    
    if "additional_info" in checkpoint:
        result["additional_info"] = checkpoint["additional_info"]
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return result


def load_model_for_inference(
    model_path: str,
    device: str = "cpu",
) -> ArborTransformer:
    """
    Load a model for inference (without optimizer state).
    
    Args:
        model_path: Path to model file
        device: Device to load model to
        
    Returns:
        Loaded model ready for inference
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model data
    model_data = torch.load(model_path, map_location=device)
    
    # Create model
    config = ArborConfig(**model_data["config"])
    model = ArborTransformer(config)
    
    # Load weights
    model.load_state_dict(model_data["state_dict"])
    model = model.to(device)
    model.eval()
    
    # Restore growth history
    if "growth_history" in model_data:
        model.growth_history = model_data["growth_history"]
    
    print(f"Model loaded for inference from: {model_path}")
    return model


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Check for latest_checkpoint.pt first
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_path):
        return latest_path
    
    # Look for numbered checkpoints
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_step_") and file.endswith(".pt"):
            try:
                step = int(file.replace("checkpoint_step_", "").replace(".pt", ""))
                checkpoint_files.append((step, os.path.join(checkpoint_dir, file)))
            except ValueError:
                continue
    
    if checkpoint_files:
        # Return the checkpoint with the highest step number
        checkpoint_files.sort(key=lambda x: x[0])
        return checkpoint_files[-1][1]
    
    return None


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Config saved to: {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded from: {config_path}")
    return config


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 5,
    keep_best: bool = True,
) -> None:
    """
    Clean up old checkpoints to save disk space.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Get all numbered checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_step_") and file.endswith(".pt"):
            try:
                step = int(file.replace("checkpoint_step_", "").replace(".pt", ""))
                checkpoint_files.append((step, os.path.join(checkpoint_dir, file)))
            except ValueError:
                continue
    
    if len(checkpoint_files) <= keep_last_n:
        return  # Nothing to clean up
    
    # Sort by step number
    checkpoint_files.sort(key=lambda x: x[0])
    
    # Keep the last N checkpoints
    to_keep = set()
    for _, path in checkpoint_files[-keep_last_n:]:
        to_keep.add(path)
    
    # Keep special checkpoints
    special_files = ["latest_checkpoint.pt", "best_checkpoint.pt", "best_model.pt"]
    for special_file in special_files:
        special_path = os.path.join(checkpoint_dir, special_file)
        if os.path.exists(special_path):
            to_keep.add(special_path)
    
    # Remove old checkpoints
    removed_count = 0
    for _, path in checkpoint_files:
        if path not in to_keep:
            try:
                os.remove(path)
                removed_count += 1
            except OSError as e:
                print(f"Warning: Could not remove {path}: {e}")
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} old checkpoints from {checkpoint_dir}")


def export_growth_history(
    model: ArborTransformer,
    output_path: str,
) -> None:
    """
    Export growth history to a JSON file for analysis.
    
    Args:
        model: Model with growth history
        output_path: Path to save growth history
    """
    growth_history = model.get_growth_history()
    
    # Add model info
    export_data = {
        "model_config": model.config.__dict__,
        "final_param_count": model.param_count(),
        "growth_events": growth_history,
        "summary": {
            "num_events": len(growth_history),
            "total_param_increase": sum(
                event.get("param_increase", 0) for event in growth_history
            ),
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"Growth history exported to: {output_path}")


def create_checkpoint_manifest(checkpoint_dir: str) -> None:
    """
    Create a manifest file listing all checkpoints and their metadata.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    manifest = {
        "checkpoints": [],
        "created_at": str(torch.cuda.Event()),  # Timestamp
    }
    
    # Scan for checkpoint files
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".pt"):
            file_path = os.path.join(checkpoint_dir, file)
            try:
                # Load checkpoint metadata without loading full state
                checkpoint = torch.load(file_path, map_location="cpu")
                
                checkpoint_info = {
                    "filename": file,
                    "step": checkpoint.get("step", 0),
                    "loss": checkpoint.get("loss", float('inf')),
                    "size_mb": os.path.getsize(file_path) / (1024 * 1024),
                    "modified_time": os.path.getmtime(file_path),
                }
                
                if "growth_history" in checkpoint:
                    checkpoint_info["num_growth_events"] = len(checkpoint["growth_history"])
                
                manifest["checkpoints"].append(checkpoint_info)
                
            except Exception as e:
                print(f"Warning: Could not read checkpoint {file}: {e}")
    
    # Sort by step
    manifest["checkpoints"].sort(key=lambda x: x["step"])
    
    # Save manifest
    manifest_path = os.path.join(checkpoint_dir, "checkpoint_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    print(f"Checkpoint manifest created: {manifest_path}")
