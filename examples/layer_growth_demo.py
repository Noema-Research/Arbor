#!/usr/bin/env python3
"""
Arbor Layer Growth Demo
Demonstrates dynamic layer growth from 24 to 64 layers during training
"""

import torch
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from arbor.modeling.model import ArborTransformer, ArborConfig
from arbor.data.synthetic import SyntheticDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model_from_config(config: Dict[str, Any]) -> ArborTransformer:
    """Create model from configuration."""
    model_config = ArborConfig(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        ffn_dim=config['model']['ffn_dim'],
        max_seq_length=config['model']['max_seq_length'],
        dropout=config['model']['dropout'],
        attention_dropout=config['model']['attention_dropout'],
        ffn_dropout=config['model']['ffn_dropout'],
        layer_norm_eps=config['model']['layer_norm_eps'],
        activation=config['model']['activation'],
        causal=config['model']['causal'],
        tie_word_embeddings=config['model']['tie_word_embeddings'],
        
        # Growth parameters
        enable_growth=config['growth']['enabled'],
        max_growth_events=config['growth']['max_events'],
        add_hidden=config['growth']['add_hidden'],
        growth_cooldown_steps=config['growth']['cooldown_steps'],
        layer_selection=config['growth']['layer_selection'],
        layers_per_event=config['growth']['layers_per_event'],
        new_param_lr_multiplier=config['growth']['new_param_lr_multiplier'],
        
        # Layer growth parameters
        layer_growth_enabled=config['growth']['layer_growth_enabled'],
        min_layers=config['growth']['min_layers'],
        max_layers=config['growth']['max_layers'],
        layer_growth_threshold=config['growth']['layer_growth_threshold'],
        layer_growth_factor=config['growth']['layer_growth_factor'],
        layer_growth_cooldown=config['growth']['layer_growth_cooldown'],
    )
    
    return ArborTransformer(model_config)

def create_dataloader(config: Dict[str, Any], split: str = 'train') -> DataLoader:
    """Create dataloader from configuration."""
    data_config = config['data']
    dataset = SyntheticDataset(
        vocab_size=data_config['vocab_size'],
        seq_length=data_config['seq_length'],
        num_sequences=data_config['num_sequences']
    )
    
    # Split dataset
    total_size = len(dataset)
    val_size = int(total_size * data_config['val_split'])
    test_size = int(total_size * data_config['test_split'])
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    if split == 'train':
        dataset = train_dataset
    elif split == 'val':
        dataset = val_dataset
    else:
        dataset = test_dataset
    
    return DataLoader(
        dataset,
        batch_size=data_config['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['infrastructure'].get('num_workers', 0),
        pin_memory=config['infrastructure'].get('pin_memory', False)
    )

def train_step(model: ArborTransformer, batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
    """Perform single training step."""
    model.train()
    optimizer.zero_grad()
    
    # Prepare input and target
    input_ids = batch[:, :-1]
    targets = batch[:, 1:]
    
    # Forward pass
    outputs = model(input_ids, labels=targets, return_dict=True)
    loss = outputs['loss']
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def validate_model(model: ArborTransformer, val_loader: DataLoader) -> float:
    """Validate model performance."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            outputs = model(input_ids, labels=targets, return_dict=True)
            total_loss += outputs['loss'].item()
            num_batches += 1
    
    return total_loss / num_batches

def monitor_layer_utilization(model: ArborTransformer) -> Dict[str, float]:
    """Monitor layer utilization statistics."""
    if not hasattr(model, 'layer_utilization_history') or not model.layer_utilization_history:
        return {}
    
    # Get latest utilization data
    latest_stats = model.layer_utilization_history[-1]
    
    stats = {}
    for layer_idx, utilization in latest_stats.items():
        stats[f'layer_{layer_idx}_utilization'] = utilization
    
    # Calculate average utilization
    if latest_stats:
        stats['avg_utilization'] = sum(latest_stats.values()) / len(latest_stats)
        stats['max_utilization'] = max(latest_stats.values())
        stats['min_utilization'] = min(latest_stats.values())
    
    return stats

def plot_training_progress(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot training metrics and layer growth."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(metrics['steps'], metrics['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(metrics['val_steps'], metrics['val_loss'], label='Val Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Number of layers over time
    axes[0, 1].plot(metrics['steps'], metrics['num_layers'])
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Number of Layers')
    axes[0, 1].set_title('Layer Growth Over Time')
    axes[0, 1].grid(True)
    
    # Average layer utilization
    if 'avg_utilization' in metrics:
        axes[1, 0].plot(metrics['steps'], metrics['avg_utilization'])
        axes[1, 0].axhline(y=0.92, color='r', linestyle='--', label='Growth Threshold')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Average Layer Utilization')
        axes[1, 0].set_title('Layer Utilization')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Model parameters over time
    axes[1, 1].plot(metrics['steps'], metrics['total_params'])
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Total Parameters (M)')
    axes[1, 1].set_title('Model Size Growth')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training plots saved to {save_path}")
    
    plt.show()

def main():
    """Main training loop with layer growth demonstration."""
    # Load configuration
    config_path = "configs/arbor_layer_growth.yaml"
    config = load_config(config_path)
    
    logger.info("Starting Arbor Layer Growth Demo")
    logger.info(f"Initial layers: {config['model']['num_layers']}")
    logger.info(f"Target range: {config['growth']['min_layers']}-{config['growth']['max_layers']} layers")
    
    # Create model
    model = create_model_from_config(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    # Create data loaders
    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training metrics
    metrics = {
        'steps': [],
        'train_loss': [],
        'val_steps': [],
        'val_loss': [],
        'num_layers': [],
        'total_params': [],
        'avg_utilization': [],
        'growth_events': []
    }
    
    # Training loop
    step = 0
    max_steps = config['training']['max_steps']
    log_every = config['training']['log_every']
    eval_every = config['training']['eval_every']
    
    logger.info("Starting training...")
    
    for epoch in range(100):  # Large number, will break by max_steps
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # Training step
            train_loss = train_step(model, batch, optimizer)
            
            # Update metrics
            metrics['steps'].append(step)
            metrics['train_loss'].append(train_loss)
            metrics['num_layers'].append(len(model.layers))
            metrics['total_params'].append(sum(p.numel() for p in model.parameters()) / 1e6)
            
            # Monitor layer utilization
            utilization_stats = monitor_layer_utilization(model)
            if 'avg_utilization' in utilization_stats:
                metrics['avg_utilization'].append(utilization_stats['avg_utilization'])
            else:
                metrics['avg_utilization'].append(0.0)
            
            # Check for layer growth events
            if hasattr(model, 'growth_history') and model.growth_history:
                if len(model.growth_history) > len(metrics['growth_events']):
                    latest_event = model.growth_history[-1]
                    if 'layer_growth' in latest_event:
                        metrics['growth_events'].append({
                            'step': step,
                            'event': latest_event
                        })
                        logger.info(f"🌱 Layer growth at step {step}: {latest_event}")
            
            # Logging
            if step % log_every == 0:
                logger.info(
                    f"Step {step}/{max_steps} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Layers: {len(model.layers)} | "
                    f"Params: {metrics['total_params'][-1]:.1f}M"
                )
                
                if utilization_stats:
                    logger.info(
                        f"  Utilization - Avg: {utilization_stats.get('avg_utilization', 0):.3f} | "
                        f"Max: {utilization_stats.get('max_utilization', 0):.3f} | "
                        f"Min: {utilization_stats.get('min_utilization', 0):.3f}"
                    )
            
            # Validation
            if step % eval_every == 0:
                val_loss = validate_model(model, val_loader)
                metrics['val_steps'].append(step)
                metrics['val_loss'].append(val_loss)
                
                logger.info(f"Validation at step {step}: Loss = {val_loss:.4f}")
            
            step += 1
        
        if step >= max_steps:
            break
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE")
    logger.info("="*50)
    logger.info(f"Final layers: {len(model.layers)}")
    logger.info(f"Final parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    logger.info(f"Layer growth events: {len(metrics['growth_events'])}")
    
    for i, event in enumerate(metrics['growth_events']):
        logger.info(f"  Event {i+1} at step {event['step']}: {event['event']}")
    
    # Plot results
    plot_training_progress(metrics, "arbor_layer_growth_training.png")
    
    # Save final model
    torch.save(model.state_dict(), "arbor_layer_growth_final.pth")
    logger.info("Model saved to arbor_layer_growth_final.pth")

if __name__ == "__main__":
    main()
