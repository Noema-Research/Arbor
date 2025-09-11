#!/usr/bin/env python3
"""
Simple training script for Arbor models using YAML configuration.

Usage:
    python train.py configs/training_config.yaml

This script provides a simplified interface for training Arbor models
with all configuration specified in a YAML file.
"""

import sys
import os
from pathlib import Path

# Add arbor to path
sys.path.insert(0, str(Path(__file__).parent))

from arbor.train.yaml_trainer import train_from_yaml


def main():
    """Main training entry point."""
    if len(sys.argv) != 2:
        print("Usage: python train.py <config.yaml>")
        print("\nExample:")
        print("  python train.py configs/training_config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Verify config exists
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    print("🌱 Arbor YAML Training System")
    print("=" * 40)
    print(f"📋 Config: {config_path}")
    print("=" * 40)
    
    try:
        trainer = train_from_yaml(config_path)
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
