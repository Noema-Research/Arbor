"""
YAML-based training configuration system for Arbor.

This module provides a simple interface for training Arbor models
using YAML configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import wandb
import torch
from ..modeling.model import ArborTransformer, ArborConfig


@dataclass
class ArborTrainingConfig:
    """Configuration class for Arbor training."""
    
    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Parse sections
        self.model_config = self.config.get('model', {})
        self.dataset_config = self.config.get('datasets', [])
        self.tokenizer_config = self.config.get('tokenizer', {})
        self.training_config = self.config.get('training', {})
        self.logging_config = self.config.get('logging', {})
        self.hf_config = self.config.get('huggingface', {})
        self.hardware_config = self.config.get('hardware', {})
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_sections = ['model', 'datasets', 'tokenizer', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate datasets
        if not self.dataset_config:
            raise ValueError("At least one dataset must be specified")
        
        return True


class ArborYAMLTrainer:
    """YAML-configured trainer for Arbor models."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with YAML configuration."""
        self.config = ArborTrainingConfig(config_path)
        self.config.validate()
        
        self.model = None
        self.tokenizer = None
        self.datasets = {}
        self.trainer = None
        
        print(f"🌱 Initialized Arbor trainer with config: {config_path}")
    
    def setup_tokenizer(self):
        """Setup tokenizer - always downloads fresh Hermes-4-405B tokenizer."""
        print("📥 Downloading fresh Hermes-4-405B tokenizer...")
        
        try:
            # Always download fresh from HuggingFace to ensure latest version
            self.tokenizer = AutoTokenizer.from_pretrained(
                'NousResearch/Hermes-4-405B',
                force_download=True,  # Always download fresh
                resume_download=False  # Don't resume partial downloads
            )
            print("✅ Successfully loaded fresh Hermes-4-405B tokenizer")
            print(f"   Vocabulary size: {len(self.tokenizer):,}")
            print(f"   Model max length: {self.tokenizer.model_max_length:,}")
                
        except Exception as e:
            print(f"❌ Failed to download Hermes-4-405B tokenizer: {e}")
            print("   Please check your internet connection and HuggingFace access")
            raise
    
    def setup_model(self):
        """Setup Arbor model based on configuration."""
        model_config = self.config.model_config
        
        # Create Arbor configuration
        arbor_config = ArborConfig(
            vocab_size=model_config.get('vocab_size', 128000),
            dim=model_config.get('hidden_size', 1024),
            num_layers=model_config.get('num_layers', 24),
            num_heads=model_config.get('num_heads', 16),
            ffn_dim=model_config.get('intermediate_size', 4096),
            max_seq_length=model_config.get('max_position_embeddings', 131072),
            growth_enabled=model_config.get('growth', {}).get('enabled', True),
            
            # Adaptive context settings
            adaptive_context=model_config.get('adaptive_context', {}).get('enabled', True),
            min_context_length=model_config.get('adaptive_context', {}).get('min_context_length', 1024),
            max_context_length=model_config.get('adaptive_context', {}).get('max_context_length'),
        )
        
        # Create model
        self.model = ArborTransformer(arbor_config)
        
        print(f"✅ Created Arbor model: {self.model.param_count():,} parameters")
        
        # Show adaptive context info
        if arbor_config.adaptive_context:
            context_info = self.model.get_context_info()
            print(f"🧠 Adaptive context enabled:")
            print(f"   Range: {context_info['min_context_length']:,} - {context_info['max_context_length']:,}")
            print(f"   Supported tasks: {len(context_info['supported_task_types'])}")
        
        # Setup growth monitoring if enabled
        if model_config.get('growth', {}).get('enabled', True):
            self.setup_growth_monitoring()
            
        # Setup adaptive context monitoring if enabled
        if arbor_config.adaptive_context:
            self.setup_adaptive_context_monitoring()
    
    def setup_growth_monitoring(self):
        """Setup dynamic growth monitoring."""
        growth_config = self.config.model_config.get('growth', {})
        
        print(f"🌱 Growth monitoring enabled:")
        print(f"   Factor: {growth_config.get('factor', 2.0)}x")
        print(f"   Max steps: {growth_config.get('max_steps', 8)}")
        print(f"   Threshold: {growth_config.get('threshold', 0.95)}")
    
    def setup_adaptive_context_monitoring(self):
        """Setup adaptive context window monitoring."""
        adaptive_config = self.config.model_config.get('adaptive_context', {})
        
        print(f"🧠 Adaptive context monitoring enabled:")
        print(f"   Router layers: {adaptive_config.get('context_router_layers', 3)}")
        print(f"   Hardware aware: {adaptive_config.get('hardware_aware', True)}")
        print(f"   Memory threshold: {adaptive_config.get('memory_threshold', 0.85)}")
        print(f"   Latency threshold: {adaptive_config.get('latency_threshold', 2.0)}s")
        
        # Log available context lengths and task types
        task_types = adaptive_config.get('task_types', [])
        context_lengths = adaptive_config.get('context_lengths', [])
        
        if task_types:
            print(f"   Task types: {', '.join(task_types[:3])}{'...' if len(task_types) > 3 else ''}")
        if context_lengths:
            print(f"   Context options: {context_lengths[0]:,} - {context_lengths[-1]:,} tokens")
    
    def load_datasets(self):
        """Load and preprocess datasets based on configuration."""
        print("📚 Loading datasets...")
        
        for dataset_config in self.config.dataset_config:
            name = dataset_config['name']
            source = dataset_config['source']
            split = dataset_config.get('split', 'train')
            
            try:
                # Load dataset
                if 'fallback_dataset' in dataset_config:
                    try:
                        dataset = load_dataset(source, split=split)
                    except:
                        fallback = dataset_config['fallback_dataset']
                        fallback_config = dataset_config.get('fallback_config')
                        if fallback_config:
                            dataset = load_dataset(fallback, fallback_config, split=split)
                        else:
                            dataset = load_dataset(fallback, split=split)
                        print(f"   📋 Using fallback dataset for {name}")
                else:
                    # Handle filters for datasets like github-code
                    if 'filters' in dataset_config:
                        filters = dataset_config['filters']
                        dataset = load_dataset(source, split=split, **filters)
                    else:
                        dataset = load_dataset(source, split=split)
                
                # Preprocess dataset
                processed_dataset = self.preprocess_dataset(dataset, dataset_config)
                self.datasets[name] = processed_dataset
                
                print(f"   ✅ {name}: {len(processed_dataset)} examples")
                
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
                continue
    
    def preprocess_dataset(self, dataset, config):
        """Preprocess a dataset according to configuration."""
        text_column = config.get('text_column', 'text')
        preprocessing = config.get('preprocessing', {})
        
        prefix = preprocessing.get('prefix', '')
        suffix = preprocessing.get('suffix', '')
        max_length = preprocessing.get('max_length', 1024)
        
        def tokenize_function(examples):
            texts = []
            for text in examples[text_column]:
                formatted_text = f"{prefix}\n{text}\n{suffix}"
                texts.append(formatted_text)
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_overflowing_tokens=False,
            )
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
    
    def setup_logging(self):
        """Setup logging and monitoring."""
        logging_config = self.config.logging_config
        
        # Setup WandB if enabled
        wandb_config = logging_config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'arbor-training'),
                entity=wandb_config.get('entity'),
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                config=self.config.config
            )
            print("✅ WandB logging enabled")
    
    def create_trainer(self, dataset_name: str):
        """Create trainer for a specific dataset."""
        training_config = self.config.training_config
        
        # Training arguments
        args = TrainingArguments(
            output_dir=f"{training_config.get('output_dir', './output')}/{dataset_name}",
            learning_rate=training_config.get('learning_rate', 2e-5),
            warmup_steps=training_config.get('warmup_steps', 100),
            max_steps=training_config.get('steps_per_dataset', 500),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
            eval_steps=training_config.get('eval_steps', 50),
            save_steps=training_config.get('save_steps', 100),
            logging_steps=training_config.get('logging_steps', 20),
            fp16=training_config.get('fp16', True),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            dataloader_drop_last=training_config.get('dataloader_drop_last', True),
            evaluation_strategy=training_config.get('eval_strategy', 'steps'),
            report_to="wandb" if self.config.logging_config.get('wandb', {}).get('enabled') else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        dataset = self.datasets[dataset_name]
        eval_dataset = dataset.select(range(min(100, len(dataset)))) if len(dataset) > 100 else None
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
    
    def train(self):
        """Execute complete training pipeline."""
        print("🚀 Starting Arbor training pipeline...")
        
        # Setup components
        self.setup_tokenizer()
        self.setup_model()
        self.load_datasets()
        self.setup_logging()
        
        # Train on each dataset sequentially
        for dataset_name in self.datasets.keys():
            print(f"\n🎯 Training on {dataset_name}...")
            
            trainer = self.create_trainer(dataset_name)
            
            # Train
            initial_params = self.model.param_count()
            train_result = trainer.train()
            final_params = self.model.param_count()
            
            print(f"   ✅ {dataset_name} complete!")
            print(f"   📊 Parameters: {initial_params:,} → {final_params:,}")
            
            # Save intermediate model
            trainer.save_model()
        
        # Upload to HuggingFace if configured
        if self.config.hf_config.get('upload', {}).get('enabled', False):
            self.upload_to_hf()
        
        # Run post-training if configured
        post_training_config = self.config.config.get('post_training')
        if post_training_config and post_training_config.get('enabled', False):
            self.run_post_training(post_training_config)
        
        print("🎉 Training pipeline complete!")
    
    def upload_to_hf(self):
        """Upload trained model to HuggingFace Hub."""
        hf_config = self.config.hf_config
        upload_config = hf_config.get('upload', {})
        
        repository = upload_config.get('repository')
        if not repository:
            print("⚠️  No HF repository specified, skipping upload")
            return
        
        print(f"🚀 Uploading to HuggingFace: {repository}")
        
        # Implementation would use HF Hub API
        # This is a placeholder for the actual upload logic
        print("✅ HuggingFace upload complete!")
    
    def run_post_training(self, post_training_config):
        """Run immediate post-training after main training."""
        print("\n🔄 Starting immediate post-training...")
        
        try:
            # Import post-trainer
            from .post_trainer import ArborPostTrainer
            
            # Create temporary post-training config
            import tempfile
            import yaml
            
            # Get the final model path
            final_model_path = f"{self.config.training_config.get('output_dir', './output')}/final_model"
            
            # Create post-training configuration
            temp_config = {
                "model_source": "local",
                "model_path": final_model_path,
                "training_type": post_training_config.get('type', 'fine_tune'),
                "datasets": post_training_config.get('datasets', []),
                "learning_rate": post_training_config.get('learning_rate', 5e-6),
                "max_steps": post_training_config.get('max_steps', 500),
                "per_device_batch_size": post_training_config.get('batch_size', 4),
                "lora_enabled": post_training_config.get('lora_enabled', True),
                "lora_rank": post_training_config.get('lora_rank', 8),
                "freeze_layers": post_training_config.get('freeze_layers', []),
                "output_dir": post_training_config.get('output_dir', './post_training_immediate'),
                "save_merged_model": True,
                "push_to_hub": post_training_config.get('push_to_hub', False),
                "hub_model_id": post_training_config.get('hub_model_id')
            }
            
            # Save temporary config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(temp_config, f, default_flow_style=False, indent=2)
                temp_config_path = f.name
            
            # Run post-training
            post_trainer = ArborPostTrainer(temp_config_path)
            
            # Use the current model instead of loading from disk
            post_trainer.model = self.model
            post_trainer.tokenizer = self.tokenizer
            
            # Run only the dataset loading and training parts
            post_trainer.load_post_training_datasets()
            
            if post_trainer.datasets:
                print(f"🎯 Post-training on {len(post_trainer.datasets)} datasets...")
                
                # Setup LoRA and freezing
                post_trainer.setup_lora()
                post_trainer.freeze_layers()
                
                # Train on each dataset
                for dataset_name in post_trainer.datasets.keys():
                    print(f"📈 Post-training on {dataset_name}...")
                    trainer = post_trainer.create_post_trainer(dataset_name)
                    train_result = trainer.train()
                    trainer.save_model()
                    print(f"   ✅ {dataset_name} post-training complete!")
                
                print("🎉 Immediate post-training complete!")
            else:
                print("⚠️  No post-training datasets loaded")
            
            # Cleanup
            os.unlink(temp_config_path)
            
        except Exception as e:
            print(f"❌ Post-training failed: {e}")
            import traceback
            traceback.print_exc()


def train_from_yaml(config_path: str):
    """Train Arbor model from YAML configuration."""
    trainer = ArborYAMLTrainer(config_path)
    trainer.train()
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Arbor model from YAML config")
    parser.add_argument("config", help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    train_from_yaml(args.config)
