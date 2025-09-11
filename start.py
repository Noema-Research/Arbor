#!/usr/bin/env python3
"""
Quick start script for Arbor training.

This script helps users get started quickly with minimal configuration.
"""

import os
import sys
import shutil
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print("🌱" + "=" * 50 + "🌱")
    print("   ARBOR O1 LIVING AI - QUICK START")
    print("🌱" + "=" * 50 + "🌱")
    print()


def check_dependencies():
    """Check if required packages are installed."""
    required = ['torch', 'transformers', 'datasets', 'yaml']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install torch transformers datasets PyYAML")
        return False
    
    print("✅ All dependencies found")
    return True


def setup_config():
    """Setup training configuration."""
    config_dir = Path("configs")
    config_file = config_dir / "my_training.yaml"
    example_file = config_dir / "example_config.yaml"
    
    if config_file.exists():
        print(f"✅ Config already exists: {config_file}")
        return str(config_file)
    
    if example_file.exists():
        shutil.copy(example_file, config_file)
        print(f"📋 Created config: {config_file}")
        print("   Edit this file to customize your training")
    else:
        print("❌ Example config not found")
        return None
    
    return str(config_file)


def check_internet():
    """Check internet connectivity for tokenizer download."""
    import urllib.request
    
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=5)
        print("✅ Internet connection available")
        print("   Will download fresh Hermes-4-405B tokenizer")
        return True
    except:
        print("❌ No internet connection")
        print("   Internet required to download Hermes-4-405B tokenizer")
        return False


def main():
    """Main function."""
    print_banner()
    
    print("🔍 Checking setup...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check internet connectivity
    if not check_internet():
        sys.exit(1)
    
    # Setup config
    config_file = setup_config()
    if not config_file:
        sys.exit(1)
    
    print("\n🚀 Ready to train!")
    print(f"   Config: {config_file}")
    print("\n📝 Next steps:")
    print(f"   1. Edit {config_file} to customize training")
    print("   2. Set environment variables (optional):")
    print("      export HF_TOKEN='your_token'")
    print("      export WANDB_API_KEY='your_key'")
    print(f"   3. Start training: python train.py {config_file}")
    print("\n💡 Tips:")
    print("   - Start with small datasets for testing")
    print("   - Enable WandB logging to monitor growth")
    print("   - Use HF upload to share your trained model")
    
    # Ask if user wants to start training now
    print("\n❓ Start training now? (y/n): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting training...")
        os.system(f"python train.py {config_file}")
    else:
        print("👋 Happy training! Run the command above when ready.")


if __name__ == "__main__":
    main()
