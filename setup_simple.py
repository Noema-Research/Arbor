#!/usr/bin/env python3
"""
Simple setup script for Arbor-o1 Living AI.
This script sets up the environment without complex packaging.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_arbor():
    """Set up Arbor environment."""
    print("🌱 Setting up Arbor-o1 Living AI...")
    
    # Get the current directory
    arbor_dir = Path(__file__).parent.absolute()
    print(f"📁 Arbor directory: {arbor_dir}")
    
    # Check if we can import arbor
    sys.path.insert(0, str(arbor_dir))
    
    try:
        import arbor
        print("✅ Arbor import successful!")
        print(f"   Version: {arbor.__version__}")
        
        # Test basic functionality
        from arbor.modeling.model import ArborConfig, ArborTransformer
        
        config = ArborConfig(
            vocab_size=1000,
            dim=64,
            num_layers=2,
            num_heads=2,
            ffn_dim=128,
            max_seq_length=32
        )
        
        model = ArborTransformer(config)
        params = model.param_count()
        print(f"✅ Model creation test: {params:,} parameters")
        
        print("\n🎉 Arbor setup completed successfully!")
        print("\n📖 Next steps:")
        print("   1. Run the demo notebook: jupyter notebook notebooks/demo.ipynb")
        print("   2. Or run the setup checker: python arbor_setup_checker.py")
        print("\n💡 To use Arbor in other Python scripts, add this to the top:")
        print(f"   import sys; sys.path.append('{arbor_dir}')")
        print("   import arbor")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\n🔧 Try installing dependencies:")
        print("   pip install torch numpy tqdm transformers matplotlib datasets safetensors")
        return False

if __name__ == "__main__":
    success = setup_arbor()
    sys.exit(0 if success else 1)
