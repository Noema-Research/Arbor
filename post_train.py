#!/usr/bin/env python3
"""
Simple Post-Training Script for Arbor Models.

This script allows users to post-train Arbor models either:
1. Immediately after main training
2. Later with models downloaded from HuggingFace

Usage:
    python post_train.py post_training_config.yaml
    python post_train.py --create-config
    python post_train.py --model your-username/your-model --type instruct
"""

import sys
import os
import argparse
from pathlib import Path

# Add arbor to path
sys.path.insert(0, str(Path(__file__).parent))

from arbor.train.post_trainer import ArborPostTrainer, create_post_training_config


def create_example_configs():
    """Create example post-training configurations."""
    configs = {
        "fine_tune": {
            "model_source": "huggingface",
            "model_path": "your-username/your-arbor-model",
            "training_type": "fine_tune",
            "datasets": [
                {
                    "name": "stories",
                    "source": "roneneldan/TinyStories",
                    "split": "train[:2000]",
                    "text_column": "text",
                    "max_length": 1024
                }
            ],
            "learning_rate": 1e-5,
            "max_steps": 500,
            "lora_enabled": True,
            "lora_rank": 16,
            "output_dir": "./post_training_fine_tune"
        },
        
        "instruct": {
            "model_source": "huggingface", 
            "model_path": "your-username/your-arbor-model",
            "training_type": "instruct",
            "datasets": [
                {
                    "name": "alpaca",
                    "source": "tatsu-lab/alpaca",
                    "split": "train[:1000]",
                    "max_length": 2048
                }
            ],
            "learning_rate": 2e-5,
            "max_steps": 1000,
            "lora_enabled": True,
            "lora_rank": 32,
            "output_dir": "./post_training_instruct"
        },
        
        "domain_adapt": {
            "model_source": "local",
            "model_path": "./trained_models/final_model",
            "training_type": "domain_adapt",
            "datasets": [
                {
                    "name": "code",
                    "source": "codeparrot/github-code-clean",
                    "split": "train[:1000]",
                    "text_column": "code",
                    "max_length": 4096
                }
            ],
            "learning_rate": 5e-6,
            "max_steps": 2000,
            "lora_enabled": True,
            "freeze_layers": [0, 1, 2, 3],  # Freeze first 4 layers
            "output_dir": "./post_training_code"
        }
    }
    
    print("📋 Creating example post-training configurations...")
    
    for config_type, config_data in configs.items():
        filename = f"configs/post_training_{config_type}.yaml"
        
        import yaml
        with open(filename, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"   ✅ {filename}")
    
    print("\n💡 Example configurations created!")
    print("   Edit the model_path in each config to point to your model")
    print("   Then run: python post_train.py configs/post_training_fine_tune.yaml")


def quick_post_train(model_path: str, training_type: str, steps: int = 500):
    """Quick post-training setup for immediate use."""
    print(f"🚀 Quick Post-Training Setup")
    print(f"   Model: {model_path}")
    print(f"   Type: {training_type}")
    print(f"   Steps: {steps}")
    
    # Create quick config
    config_file = f"quick_post_training_{training_type}.yaml"
    
    datasets_map = {
        "fine_tune": [
            {
                "name": "stories",
                "source": "roneneldan/TinyStories", 
                "split": "train[:1000]",
                "text_column": "text",
                "max_length": 1024
            }
        ],
        "instruct": [
            {
                "name": "alpaca_small",
                "source": "tatsu-lab/alpaca",
                "split": "train[:500]",
                "max_length": 2048
            }
        ],
        "code": [
            {
                "name": "python_code",
                "source": "codeparrot/github-code-clean",
                "split": "train[:500]", 
                "text_column": "code",
                "max_length": 2048
            }
        ]
    }
    
    # Determine model source
    if model_path.startswith("./") or model_path.startswith("/"):
        model_source = "local"
    elif "/" in model_path and not model_path.startswith("http"):
        model_source = "huggingface"
    else:
        model_source = "checkpoint"
    
    create_post_training_config(
        model_source=model_source,
        model_path=model_path,
        training_type=training_type,
        datasets=datasets_map.get(training_type, datasets_map["fine_tune"]),
        output_file=config_file
    )
    
    print(f"✅ Created quick config: {config_file}")
    print("🚀 Starting post-training...")
    
    # Run post-training
    post_trainer = ArborPostTrainer(config_file)
    post_trainer.run_post_training()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Arbor Post-Training System")
    parser.add_argument("config", nargs="?", help="Path to post-training YAML configuration")
    parser.add_argument("--create-config", action="store_true", help="Create example configurations")
    parser.add_argument("--model", help="Model path for quick post-training")
    parser.add_argument("--type", choices=["fine_tune", "instruct", "code"], default="fine_tune", 
                       help="Type of post-training")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--model-path-override", help="Override model path from config")
    
    args = parser.parse_args()
    
    # Create example configs
    if args.create_config:
        create_example_configs()
        return
    
    # Quick post-training mode
    if args.model:
        quick_post_train(args.model, args.type, args.steps)
        return
    
    # Regular post-training with config file
    if not args.config:
        print("❌ Please provide a configuration file or use --model for quick setup")
        print("\nExamples:")
        print("  python post_train.py configs/post_training_fine_tune.yaml")
        print("  python post_train.py --model your-username/your-model --type instruct")
        print("  python post_train.py --create-config")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    print("🔄 Arbor Post-Training System")
    print("=" * 40)
    print(f"📋 Config: {args.config}")
    if args.model_path_override:
        print(f"🔄 Model override: {args.model_path_override}")
    print("=" * 40)
    
    try:
        post_trainer = ArborPostTrainer(args.config, args.model_path_override)
        post_trainer.run_post_training()
        
        print("\n🎉 Post-training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Post-training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Post-training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
