#!/usr/bin/env python3
"""
Training script for Arbor-o1 models.

This script handles the complete training pipeline including:
- Configuration loading and validation
- Dataset preparation and loading
- Model initialization with optional growth
- Training loop with checkpointing
- Experiment tracking and logging
"""

import argparse
import os
import sys
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbor import ArborTransformer, create_arbor_model, GrowthManager, Trainer
from arbor.modeling import ArborConfig
from arbor.data import SyntheticTextDataset, ArborTokenizer, create_dataloader
from arbor.train import create_trainer
from arbor.utils import setup_logging, setup_wandb, log_experiment_start, log_experiment_end
from arbor.utils.metrics import compute_model_size_metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def prepare_dataset(config: Dict[str, Any], data_dir: Optional[str] = None) -> tuple:
    """
    Prepare training and validation datasets.
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, tokenizer)
    """
    data_config = config.get("data", {})
    
    if data_dir:
        # Load from prepared data directory
        print(f"Loading dataset from: {data_dir}")
        
        # Load dataset info
        info_path = os.path.join(data_dir, "dataset_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
            print(f"Dataset type: {dataset_info.get('type', 'unknown')}")
        
        # Load tokenizer
        tokenizer_dir = os.path.join(data_dir, "tokenizer")
        if os.path.exists(tokenizer_dir):
            tokenizer = ArborTokenizer(tokenizer_dir)
        else:
            # Fallback to creating tokenizer
            vocab_size = data_config.get("vocab_size", 1000)
            tokenizer = ArborTokenizer("gpt2", vocab_size=vocab_size)
        
        # Load datasets
        train_path = os.path.join(data_dir, "train.pt")
        val_path = os.path.join(data_dir, "val.pt")
        
        if os.path.exists(train_path):
            train_data = torch.load(train_path)
            if isinstance(train_data, dict) and "sequences" in train_data:
                # Create dataset from sequences
                class PreloadedDataset(torch.utils.data.Dataset):
                    def __init__(self, sequences, tokenizer):
                        self.sequences = sequences
                        self.tokenizer = tokenizer
                    
                    def __len__(self):
                        return len(self.sequences)
                    
                    def __getitem__(self, idx):
                        sequence = self.sequences[idx]
                        attention_mask = (sequence != tokenizer.pad_token_id).long()
                        return {
                            "input_ids": sequence.unsqueeze(0),
                            "attention_mask": attention_mask.unsqueeze(0),
                        }
                
                train_dataset = PreloadedDataset(train_data["sequences"], tokenizer)
            else:
                # Assume it's a list of examples
                class PreloadedDataset(torch.utils.data.Dataset):
                    def __init__(self, examples):
                        self.examples = examples
                    
                    def __len__(self):
                        return len(self.examples)
                    
                    def __getitem__(self, idx):
                        return self.examples[idx]
                
                train_dataset = PreloadedDataset(train_data)
        else:
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        # Load validation dataset
        if os.path.exists(val_path):
            val_data = torch.load(val_path)
            if isinstance(val_data, dict) and "sequences" in val_data:
                val_dataset = PreloadedDataset(val_data["sequences"], tokenizer)
            else:
                val_dataset = PreloadedDataset(val_data)
        else:
            print("Warning: No validation data found, using training data subset")
            val_dataset = None
    
    else:
        # Create synthetic dataset
        print("Creating synthetic dataset...")
        
        vocab_size = data_config.get("vocab_size", 1000)
        seq_length = data_config.get("seq_length", 512)
        num_sequences = data_config.get("num_sequences", 10000)
        val_split = data_config.get("val_split", 0.1)
        
        # Create tokenizer
        from arbor.data.tokenize import create_synthetic_vocabulary, create_tokenizer_from_vocab
        vocab = create_synthetic_vocabulary(vocab_size)
        tokenizer = create_tokenizer_from_vocab(vocab)
        
        # Create dataset
        full_dataset = SyntheticTextDataset(
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_sequences=num_sequences,
            tokenizer=tokenizer,
        )
        
        # Split dataset
        train_size = int(num_sequences * (1 - val_split))
        val_size = num_sequences - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    batch_size = data_config.get("batch_size", 32)
    num_workers = config.get("infrastructure", {}).get("num_workers", 2)
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    
    print(f"Training batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader, tokenizer


def create_model(config: Dict[str, Any], tokenizer: ArborTokenizer) -> ArborTransformer:
    """Create and initialize the Arbor model."""
    model_config = config.get("model", {})
    
    # Override vocab size from tokenizer
    model_config["vocab_size"] = len(tokenizer)
    
    # Create model
    model = create_arbor_model(**model_config)
    
    print(f"Created model with {model.param_count():,} parameters")
    
    return model


def setup_experiment(config: Dict[str, Any], experiment_name: str) -> None:
    """Setup experiment tracking and logging."""
    logging_config = config.get("logging", {})
    experiment_config = config.get("experiment", {})
    
    # Setup logging
    log_level = logging_config.get("log_level", "INFO")
    logger = setup_logging(log_level)
    
    # Setup Weights & Biases
    if logging_config.get("use_wandb", False):
        wandb_project = logging_config.get("wandb_project", "arbor-o1")
        
        # Merge experiment name with config name
        run_name = f"{experiment_name}_{experiment_config.get('name', 'run')}"
        
        setup_wandb(
            project_name=wandb_project,
            experiment_name=run_name,
            config=config,
            tags=experiment_config.get("tags", []),
            notes=experiment_config.get("notes", ""),
        )
    
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train Arbor-o1 models")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--exp",
        "--experiment",
        type=str,
        default="arbor_experiment",
        help="Experiment name for tracking"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing prepared dataset (optional)"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Override checkpoint directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run - just create model and print info"
    )
    
    # Growth overrides
    parser.add_argument(
        "--growth",
        type=str,
        choices=["true", "false"],
        help="Override growth enabled setting"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Override maximum training steps"
    )
    
    args = parser.parse_args()
    
    print("🌱 Arbor-o1 Training")
    print("=" * 50)
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.checkpoint_dir:
        config["logging"]["checkpoint_dir"] = args.checkpoint_dir
    
    if args.growth:
        config["growth"]["enabled"] = args.growth.lower() == "true"
    
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    
    config["training"]["resume_from_checkpoint"] = args.resume
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Setup experiment tracking
    logger = setup_experiment(config, args.exp)
    
    # Prepare dataset
    print("\n📚 Preparing dataset...")
    train_dataloader, val_dataloader, tokenizer = prepare_dataset(config, args.data_dir)
    
    # Create model
    print("\n🏗️ Creating model...")
    model = create_model(config, tokenizer)
    model_info = compute_model_size_metrics(model)
    
    # Dataset info for logging
    dataset_info = {
        "type": config.get("data", {}).get("type", "unknown"),
        "size": len(train_dataloader.dataset),
        "vocab_size": len(tokenizer),
        "seq_length": config.get("data", {}).get("seq_length", 512),
        "batch_size": config.get("data", {}).get("batch_size", 32),
    }
    
    # Log experiment start
    log_experiment_start(config, model_info, dataset_info, logger)
    
    if args.dry_run:
        print("\n🔍 Dry run completed!")
        print(f"Model: {model_info['total_parameters']:,} parameters")
        print(f"Estimated memory: {model_info['parameter_memory_mb']:.1f} MB")
        print(f"Dataset: {dataset_info['size']:,} examples")
        return
    
    # Create trainer
    print("\n🚀 Setting up trainer...")
    trainer = create_trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config["training"],
        device=device,
    )
    
    # Override growth manager config if needed
    if trainer.growth_manager:
        trainer.growth_manager.config.update(config.get("growth", {}))
    
    # Start training
    print("\n🏃‍♀️ Starting training...")
    start_time = time.time()
    
    try:
        results = trainer.train()
        training_time = time.time() - start_time
        
        # Log experiment end
        log_experiment_end(results, training_time, logger)
        
        # Print final summary
        print("\n✅ Training completed successfully!")
        print(f"Final validation loss: {results.get('final_val_loss', 0):.4f}")
        print(f"Final parameter count: {results.get('final_param_count', 0):,}")
        print(f"Growth events: {results.get('growth_events', 0)}")
        print(f"Training time: {training_time:.1f}s ({training_time/3600:.2f}h)")
        
        # Save final results
        checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints")
        results_path = os.path.join(checkpoint_dir, f"{args.exp}_results.json")
        
        final_results = {
            "experiment_name": args.exp,
            "config_path": args.config,
            "training_time": training_time,
            **results
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.1f}s")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
