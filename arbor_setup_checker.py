#!/usr/bin/env python3
"""
Quick setup script for Arbor-o1 Living AI.

This script helps users get started with Arbor-o1 by:
- Checking dependencies
- Running a quick demo
- Setting up the environment
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
        ("omegaconf", "OmegaConf"),
    ]
    
    optional_packages = [
        ("wandb", "Weights & Biases"),
        ("seaborn", "Seaborn"),
        ("pytest", "pytest"),
    ]
    
    missing_required = []
    missing_optional = []
    
    print("\n📦 Checking dependencies...")
    
    # Check required packages
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (required)")
            missing_required.append(package)
    
    # Check optional packages
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} (optional)")
            missing_optional.append(package)
    
    return missing_required, missing_optional


def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        return True
    
    print(f"\n📥 Installing {len(packages)} packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + packages)
        print("✅ Installation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick functionality test...")
    
    try:
        # Test basic imports
        from arbor.modeling.model import ArborTransformer, ArborConfig
        from arbor.modeling.layers import ExpandableFFN
        from arbor.growth.manager import GrowthManager
        from arbor.growth.triggers import PlateauTrigger
        
        print("✅ Import test passed")
        
        # Test model creation
        config = ArborConfig(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=16
        )
        
        model = ArborTransformer(config)
        initial_params = model.param_count()
        
        print(f"✅ Model creation: {initial_params:,} parameters")
        
        # Test growth
        original_d_ff = model.transformer.layers[0].mlp.d_ff
        model.grow(growth_factor=1.2)
        new_d_ff = model.transformer.layers[0].mlp.d_ff
        new_params = model.param_count()
        
        print(f"✅ Growth test: {original_d_ff} → {new_d_ff} FFN size")
        print(f"✅ Parameter growth: {initial_params:,} → {new_params:,}")
        
        # Test generation
        import torch
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, 100, (1, 5))
            output = model(input_ids)
            
        print(f"✅ Forward pass: {output.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def create_demo_script():
    """Create a simple demo script."""
    demo_content = '''#!/usr/bin/env python3
"""
Simple Arbor-o1 demo script.
"""

import torch
from arbor.modeling.model import ArborTransformer, ArborConfig
from arbor.data import ArborTokenizer, SyntheticDataset, create_dataloader
from arbor.train import Trainer, TrainingConfig
from arbor.growth.manager import GrowthManager
from arbor.growth.triggers import PlateauTrigger

def main():
    print("🌱 Simple Arbor-o1 Demo")
    print("=" * 30)
    
    # Create small model
    config = ArborConfig(
        vocab_size=500,
        n_embd=64,
        n_layer=2,
        n_head=4,
        d_ff=128,
        max_length=32
    )
    
    model = ArborTransformer(config)
    tokenizer = ArborTokenizer("gpt2", vocab_size=500)
    
    print(f"📊 Initial model: {model.param_count():,} parameters")
    
    # Create dataset
    dataset = SyntheticDataset(
        size=100,
        vocab_size=500,
        sequence_length=32,
        tokenizer=tokenizer
    )
    
    dataloader = create_dataloader(dataset, batch_size=8, shuffle=True)
    
    # Setup growth
    growth_manager = GrowthManager(
        triggers=[PlateauTrigger(patience=10, threshold=0.05)],
        growth_factor=1.25,
        min_steps_between_growth=20
    )
    
    # Training config
    training_config = TrainingConfig(
        max_steps=100,
        learning_rate=1e-3,
        log_interval=20,
        use_amp=False
    )
    
    # Train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        growth_manager=growth_manager
    )
    
    trainer.train(dataloader)
    
    print(f"📈 Final model: {model.param_count():,} parameters")
    print(f"🌱 Growth events: {len(growth_manager.growth_history)}")
    print("✅ Demo completed!")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_demo.py", "w") as f:
        f.write(demo_content)
    
    print("✅ Created quick_demo.py")


def main():
    """Main setup function."""
    print("🌱 Arbor-o1 Living AI Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        
        response = input("Install missing packages? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            if not install_dependencies(missing_required):
                print("❌ Setup failed!")
                sys.exit(1)
        else:
            print("❌ Cannot proceed without required packages")
            sys.exit(1)
    
    if missing_optional:
        print(f"\n⚠️  Optional packages missing: {', '.join(missing_optional)}")
        response = input("Install optional packages? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            install_dependencies(missing_optional)
    
    # Run quick test
    if not run_quick_test():
        print("❌ Setup verification failed!")
        sys.exit(1)
    
    # Create demo script
    create_demo_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("   1. Run quick demo: python quick_demo.py")
    print("   2. Explore notebook: jupyter notebook notebooks/demo.ipynb")
    print("   3. Run tests: python run_tests.py")
    print("   4. Train custom model: python scripts/train.py --config configs/small.yaml")
    print("\n🌱 Welcome to Arbor-o1 - The Living AI!")


if __name__ == "__main__":
    main()
