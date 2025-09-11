"""
Post-Training System for Arbor Models.

This module provides post-training capabilities that can be used either:
1. Immediately after main training (continuous pipeline)
2. Later with downloaded models from HuggingFace Hub

Post-training supports:
- Fine-tuning on specialized datasets
- Instruction tuning
- RLHF (Reinforcement Learning from Human Feedback)
- Domain adaptation
- Few-shot learning optimization
"""

import torch
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import wandb
from huggingface_hub import hf_hub_download, snapshot_download

# Import Arbor components
try:
    from ..modeling.model import ArborTransformer, ArborConfig
    from .yaml_trainer import ArborYAMLTrainer
except ImportError:
    # Handle relative import issues
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from modeling.model import ArborTransformer, ArborConfig
    from train.yaml_trainer import ArborYAMLTrainer


@dataclass
class PostTrainingConfig:
    """Configuration for post-training phase."""
    
    # Model source
    model_source: str = "local"  # "local", "huggingface", or "checkpoint"
    model_path: str = None       # Path or HF model ID
    
    # Post-training type
    training_type: str = "fine_tune"  # "fine_tune", "instruct", "rlhf", "domain_adapt"
    
    # Datasets for post-training
    datasets: List[Dict] = None
    
    # Training parameters
    learning_rate: float = 1e-5    # Lower LR for post-training
    warmup_steps: int = 50
    max_steps: int = 1000
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    eval_steps: int = 100
    save_steps: int = 200
    
    # Post-training specific settings
    freeze_layers: List[int] = None      # Which layers to freeze
    lora_enabled: bool = False           # Use LoRA for efficient fine-tuning
    lora_rank: int = 16                  # LoRA rank
    target_modules: List[str] = None     # Modules to apply LoRA to
    
    # Adaptive context during post-training
    adaptive_context_enabled: bool = True
    context_adaptation_strength: float = 0.8  # How aggressively to adapt context
    
    # Growth during post-training
    growth_enabled: bool = False         # Usually disabled for post-training
    growth_threshold: float = 0.98       # Higher threshold for post-training
    
    # Output settings
    output_dir: str = "./post_training_output"
    save_merged_model: bool = True       # Save full merged model (not just adapters)
    push_to_hub: bool = False
    hub_model_id: str = None
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = []
        if self.freeze_layers is None:
            self.freeze_layers = []
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ArborPostTrainer:
    """
    Post-training system for Arbor models.
    
    Supports multiple post-training scenarios:
    1. Immediate post-training after main training
    2. Loading from HuggingFace Hub for later post-training
    3. Loading from local checkpoints
    4. Various post-training techniques (fine-tuning, instruction tuning, etc.)
    """
    
    def __init__(self, config_path: str, base_model_path: Optional[str] = None):
        """
        Initialize post-trainer.
        
        Args:
            config_path: Path to post-training YAML configuration
            base_model_path: Optional path to base model (overrides config)
        """
        self.config_path = Path(config_path)
        self.load_config()
        
        if base_model_path:
            self.config.model_path = base_model_path
        
        self.model = None
        self.tokenizer = None
        self.datasets = {}
        self.original_model = None  # Keep reference to original for comparison
        
        print(f"🔄 Initialized Arbor Post-Trainer")
        print(f"   Config: {config_path}")
        print(f"   Type: {self.config.training_type}")
        print(f"   Model source: {self.config.model_source}")
    
    def load_config(self):
        """Load post-training configuration from YAML."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert dict to PostTrainingConfig
        self.config = PostTrainingConfig(**config_dict)
        
        print(f"✅ Loaded post-training configuration")
        print(f"   Training type: {self.config.training_type}")
        print(f"   Datasets: {len(self.config.datasets)}")
        print(f"   LoRA enabled: {self.config.lora_enabled}")
    
    def load_base_model(self):
        """Load the base model for post-training."""
        print(f"📥 Loading base model...")
        
        if self.config.model_source == "local":
            self._load_local_model()
        elif self.config.model_source == "huggingface":
            self._load_huggingface_model()
        elif self.config.model_source == "checkpoint":
            self._load_checkpoint_model()
        else:
            raise ValueError(f"Unknown model source: {self.config.model_source}")
        
        # Load tokenizer (always from HuggingFace for consistency)
        print("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'NousResearch/Hermes-4-405B',
            force_download=True
        )
        
        print(f"✅ Model and tokenizer loaded")
        print(f"   Parameters: {self.model.param_count():,}")
        
        # Show model info
        if hasattr(self.model, 'get_context_info'):
            context_info = self.model.get_context_info()
            print(f"   Adaptive context: {context_info['adaptive_context_enabled']}")
            if context_info['adaptive_context_enabled']:
                print(f"   Context range: {context_info['min_context_length']:,} - {context_info['max_context_length']:,}")
    
    def _load_local_model(self):
        """Load model from local directory."""
        model_path = Path(self.config.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path not found: {model_path}")
        
        # Check if it's an Arbor model or HuggingFace model
        config_file = model_path / "config.json"
        if config_file.exists():
            # HuggingFace format
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"   Loaded HuggingFace model from {model_path}")
        else:
            # Native Arbor model
            # Try to load Arbor config and model
            raise NotImplementedError("Native Arbor model loading not yet implemented")
    
    def _load_huggingface_model(self):
        """Load model from HuggingFace Hub."""
        model_id = self.config.model_path
        
        print(f"   Downloading from HuggingFace: {model_id}")
        
        try:
            # Try to load as Arbor model first
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True  # For custom models
            )
            print(f"   Successfully loaded from HuggingFace Hub")
            
        except Exception as e:
            print(f"   Error loading from HuggingFace: {e}")
            raise
    
    def _load_checkpoint_model(self):
        """Load model from training checkpoint."""
        checkpoint_path = Path(self.config.model_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
        
        # Load from checkpoint directory
        self.model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print(f"   Loaded from checkpoint: {checkpoint_path}")
    
    def setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        if not self.config.lora_enabled:
            return
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            print(f"🔧 Setting up LoRA...")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_rank,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=self.config.target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            print(f"✅ LoRA setup complete")
            print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            print(f"   LoRA rank: {self.config.lora_rank}")
            print(f"   Target modules: {self.config.target_modules}")
            
        except ImportError:
            print("❌ PEFT library not found. Install with: pip install peft")
            print("   Continuing without LoRA...")
            self.config.lora_enabled = False
    
    def freeze_layers(self):
        """Freeze specified layers during post-training."""
        if not self.config.freeze_layers:
            return
        
        print(f"🧊 Freezing layers: {self.config.freeze_layers}")
        
        frozen_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            
            # Check if this parameter belongs to a frozen layer
            for layer_idx in self.config.freeze_layers:
                if f".{layer_idx}." in name or f"layers.{layer_idx}" in name:
                    param.requires_grad = False
                    frozen_params += param.numel()
                    break
        
        print(f"✅ Layer freezing complete")
        print(f"   Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")
        print(f"   Trainable parameters: {total_params - frozen_params:,}")
    
    def load_post_training_datasets(self):
        """Load datasets for post-training."""
        print("📚 Loading post-training datasets...")
        
        for dataset_config in self.config.datasets:
            name = dataset_config['name']
            source = dataset_config['source']
            
            try:
                print(f"   Loading {name}...")
                
                # Handle different dataset sources
                if 'split' in dataset_config:
                    dataset = load_dataset(source, split=dataset_config['split'])
                else:
                    dataset = load_dataset(source)['train']
                
                # Apply post-training specific preprocessing
                processed_dataset = self._preprocess_post_training_dataset(dataset, dataset_config)
                self.datasets[name] = processed_dataset
                
                print(f"   ✅ {name}: {len(processed_dataset)} examples")
                
                # Show sample for post-training types
                if len(processed_dataset) > 0:
                    sample = processed_dataset[0]
                    if 'input_ids' in sample:
                        sample_text = self.tokenizer.decode(sample['input_ids'][:100])
                        print(f"      Sample: {sample_text[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
                continue
    
    def _preprocess_post_training_dataset(self, dataset, config):
        """Preprocess dataset for specific post-training type."""
        text_column = config.get('text_column', 'text')
        training_type = self.config.training_type
        
        if training_type == "instruct":
            return self._preprocess_instruction_dataset(dataset, config)
        elif training_type == "fine_tune":
            return self._preprocess_fine_tune_dataset(dataset, config)
        elif training_type == "domain_adapt":
            return self._preprocess_domain_adapt_dataset(dataset, config)
        else:
            return self._preprocess_fine_tune_dataset(dataset, config)
    
    def _preprocess_instruction_dataset(self, dataset, config):
        """Preprocess for instruction tuning."""
        def format_instruction(examples):
            formatted_texts = []
            
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples.get('input', [''] * len(examples['instruction']))[i]
                output = examples['output'][i]
                
                # Format as instruction-following
                if input_text:
                    formatted = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    formatted = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                
                formatted_texts.append(formatted)
            
            return self.tokenizer(
                formatted_texts,
                truncation=True,
                padding=False,
                max_length=config.get('max_length', 2048),
                return_overflowing_tokens=False
            )
        
        return dataset.map(format_instruction, batched=True, remove_columns=dataset.column_names)
    
    def _preprocess_fine_tune_dataset(self, dataset, config):
        """Preprocess for general fine-tuning."""
        text_column = config.get('text_column', 'text')
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding=False,
                max_length=config.get('max_length', 2048),
                return_overflowing_tokens=False
            )
        
        return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    def _preprocess_domain_adapt_dataset(self, dataset, config):
        """Preprocess for domain adaptation."""
        # Similar to fine-tuning but with domain-specific formatting
        return self._preprocess_fine_tune_dataset(dataset, config)
    
    def create_post_trainer(self, dataset_name: str):
        """Create HuggingFace trainer for post-training."""
        print(f"🔧 Creating post-trainer for {dataset_name}...")
        
        # Training arguments optimized for post-training
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/{dataset_name}",
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=20,
            fp16=True,
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable automatic logging
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Get dataset
        dataset = self.datasets[dataset_name]
        
        # Create small eval set
        if len(dataset) > 100:
            eval_size = min(100, len(dataset) // 10)
            eval_dataset = dataset.select(range(eval_size))
            train_dataset = dataset.select(range(eval_size, len(dataset)))
        else:
            eval_dataset = None
            train_dataset = dataset
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
    
    def run_post_training(self):
        """Execute the complete post-training pipeline."""
        print("🚀 Starting Arbor Post-Training Pipeline")
        print("=" * 50)
        
        # Load base model
        self.load_base_model()
        
        # Setup LoRA if enabled
        self.setup_lora()
        
        # Freeze layers if specified
        self.freeze_layers()
        
        # Load post-training datasets
        self.load_post_training_datasets()
        
        if not self.datasets:
            print("❌ No datasets loaded for post-training")
            return
        
        # Store initial parameters for comparison
        initial_params = self.model.param_count() if hasattr(self.model, 'param_count') else 0
        
        print(f"\n🎯 Starting post-training on {len(self.datasets)} datasets...")
        
        # Train on each dataset
        for dataset_name in self.datasets.keys():
            print(f"\n📈 Post-training on {dataset_name}...")
            
            trainer = self.create_post_trainer(dataset_name)
            
            # Train
            train_result = trainer.train()
            
            print(f"   ✅ {dataset_name} post-training complete!")
            print(f"   📊 Final loss: {train_result.training_loss:.4f}")
            
            # Save checkpoint
            trainer.save_model()
            
            # Save merged model if using LoRA
            if self.config.lora_enabled and self.config.save_merged_model:
                self._save_merged_model(trainer, dataset_name)
        
        # Final model info
        final_params = self.model.param_count() if hasattr(self.model, 'param_count') else 0
        if final_params and initial_params:
            print(f"\n📊 Parameter change: {initial_params:,} → {final_params:,}")
        
        # Push to hub if configured
        if self.config.push_to_hub:
            self._push_to_hub()
        
        print("🎉 Post-training pipeline complete!")
    
    def _save_merged_model(self, trainer, dataset_name):
        """Save merged LoRA model."""
        if not self.config.lora_enabled:
            return
        
        try:
            from peft import PeftModel
            
            print(f"💾 Saving merged model for {dataset_name}...")
            
            # Merge LoRA weights
            merged_model = trainer.model.merge_and_unload()
            
            # Save merged model
            output_path = Path(self.config.output_dir) / f"{dataset_name}_merged"
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            
            print(f"   ✅ Merged model saved to {output_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to save merged model: {e}")
    
    def _push_to_hub(self):
        """Push trained model to HuggingFace Hub."""
        if not self.config.hub_model_id:
            print("⚠️  No hub_model_id specified, skipping push to hub")
            return
        
        print(f"🚀 Pushing to HuggingFace Hub: {self.config.hub_model_id}")
        
        try:
            # Push the final model
            self.model.push_to_hub(self.config.hub_model_id)
            self.tokenizer.push_to_hub(self.config.hub_model_id)
            
            print(f"✅ Successfully pushed to {self.config.hub_model_id}")
            
        except Exception as e:
            print(f"❌ Failed to push to hub: {e}")


def create_post_training_config(
    model_source: str,
    model_path: str,
    training_type: str = "fine_tune",
    datasets: List[Dict] = None,
    output_file: str = "post_training_config.yaml"
):
    """
    Helper function to create post-training configuration files.
    
    Args:
        model_source: "local", "huggingface", or "checkpoint"
        model_path: Path or HuggingFace model ID
        training_type: "fine_tune", "instruct", "rlhf", "domain_adapt"
        datasets: List of dataset configurations
        output_file: Where to save the config
    """
    if datasets is None:
        # Default datasets based on training type
        if training_type == "instruct":
            datasets = [
                {
                    "name": "alpaca",
                    "source": "tatsu-lab/alpaca",
                    "split": "train[:1000]",
                    "max_length": 2048
                }
            ]
        else:
            datasets = [
                {
                    "name": "general",
                    "source": "roneneldan/TinyStories",
                    "split": "train[:1000]",
                    "text_column": "text",
                    "max_length": 1024
                }
            ]
    
    config = {
        "model_source": model_source,
        "model_path": model_path,
        "training_type": training_type,
        "datasets": datasets,
        "learning_rate": 1e-5,
        "max_steps": 1000,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lora_enabled": True,
        "lora_rank": 16,
        "output_dir": f"./post_training_{training_type}",
        "save_merged_model": True,
        "push_to_hub": False
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created post-training config: {output_file}")
    return output_file


# CLI interface
def main():
    """Command line interface for post-training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Arbor Post-Training System")
    parser.add_argument("config", help="Path to post-training YAML configuration")
    parser.add_argument("--model-path", help="Override model path from config")
    parser.add_argument("--create-config", action="store_true", help="Create example config")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_post_training_config(
            model_source="huggingface",
            model_path="your-username/your-arbor-model",
            training_type="fine_tune"
        )
        return
    
    # Run post-training
    post_trainer = ArborPostTrainer(args.config, args.model_path)
    post_trainer.run_post_training()


if __name__ == "__main__":
    main()
