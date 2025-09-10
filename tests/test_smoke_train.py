"""
Smoke tests for Arbor-o1 training pipeline.

Tests the complete training workflow including:
- Data loading and preparation
- Model initialization and training
- Growth events during training
- Checkpointing and recovery
"""

import pytest
import torch
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbor.modeling.model import ArborTransformer, ArborConfig
from arbor.data import ArborTokenizer, SyntheticDataset, create_dataloader
from arbor.train import Trainer, TrainingConfig
from arbor.growth.manager import GrowthManager
from arbor.growth.triggers import PlateauTrigger, GradientNormTrigger
from arbor.utils.checkpointing import save_checkpoint, load_checkpoint


class TestSmokeTraining:
    """Smoke tests for the complete training pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_smoke_basic_training(self):
        """Test basic training without growth."""
        # Create small model configuration
        model_config = ArborConfig(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=16
        )
        
        # Create model and tokenizer
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=100)
        
        # Create synthetic dataset
        dataset = SyntheticDataset(
            size=50,
            vocab_size=100,
            sequence_length=16,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            max_steps=10,
            learning_rate=1e-3,
            warmup_steps=2,
            log_interval=5,
            eval_interval=None,
            save_interval=None,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_amp=False,  # Disable AMP for simplicity
        )
        
        # Create trainer without growth
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device=self.device
        )
        
        # Train for a few steps
        trainer.train(dataloader)
        
        # Check that training completed
        assert trainer.step_count == 10
        assert len(trainer.loss_history) > 0
    
    def test_smoke_training_with_growth(self):
        """Test training with growth enabled."""
        # Create small model configuration
        model_config = ArborConfig(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=16
        )
        
        # Create model and tokenizer
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=100)
        
        # Create synthetic dataset
        dataset = SyntheticDataset(
            size=100,
            vocab_size=100,
            sequence_length=16,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        # Create growth manager with aggressive triggers for testing
        triggers = [
            PlateauTrigger(patience=3, threshold=0.1),
            GradientNormTrigger(threshold=2.0, patience=2)
        ]
        
        growth_manager = GrowthManager(
            triggers=triggers,
            growth_factor=1.2,
            min_steps_between_growth=5
        )
        
        # Create training configuration
        training_config = TrainingConfig(
            max_steps=30,
            learning_rate=1e-3,
            warmup_steps=3,
            log_interval=10,
            eval_interval=None,
            save_interval=None,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_amp=False,
        )
        
        # Create trainer with growth
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            growth_manager=growth_manager,
            device=self.device
        )
        
        original_params = model.param_count()
        
        # Train
        trainer.train(dataloader)
        
        # Check that training completed
        assert trainer.step_count == 30
        
        # Check if growth occurred (may or may not happen)
        if len(growth_manager.growth_history) > 0:
            assert model.param_count() > original_params
            print(f"Growth events: {len(growth_manager.growth_history)}")
            print(f"Parameter growth: {original_params} -> {model.param_count()}")
    
    def test_smoke_checkpointing(self):
        """Test training with checkpointing."""
        # Create model and config
        model_config = ArborConfig(
            vocab_size=50,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=8
        )
        
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=50)
        
        # Create dataset
        dataset = SyntheticDataset(
            size=40,
            vocab_size=50,
            sequence_length=8,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Create training config with checkpointing
        training_config = TrainingConfig(
            max_steps=20,
            learning_rate=1e-3,
            warmup_steps=2,
            log_interval=5,
            save_interval=10,  # Save every 10 steps
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_amp=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device=self.device,
            output_dir=self.temp_dir
        )
        
        # Train
        trainer.train(dataloader)
        
        # Check that checkpoint was saved
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint-*.pt"))
        assert len(checkpoint_files) > 0
        
        # Test loading checkpoint
        checkpoint_path = checkpoint_files[0]
        checkpoint_data = load_checkpoint(str(checkpoint_path))
        
        assert "model_state_dict" in checkpoint_data
        assert "step" in checkpoint_data
        assert checkpoint_data["step"] > 0
    
    def test_smoke_resume_training(self):
        """Test resuming training from checkpoint."""
        # Create model and config
        model_config = ArborConfig(
            vocab_size=50,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=8
        )
        
        model1 = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=50)
        
        # Create dataset
        dataset = SyntheticDataset(
            size=40,
            vocab_size=50,
            sequence_length=8,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Training config
        training_config = TrainingConfig(
            max_steps=15,
            learning_rate=1e-3,
            warmup_steps=2,
            log_interval=5,
            save_interval=10,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_amp=False,
        )
        
        # First training session
        trainer1 = Trainer(
            model=model1,
            tokenizer=tokenizer,
            config=training_config,
            device=self.device,
            output_dir=self.temp_dir
        )
        
        trainer1.train(dataloader)
        
        # Find checkpoint
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint-*.pt"))
        assert len(checkpoint_files) > 0
        checkpoint_path = checkpoint_files[0]
        
        # Create new model and resume training
        model2 = ArborTransformer(model_config)
        
        trainer2 = Trainer(
            model=model2,
            tokenizer=tokenizer,
            config=training_config,
            device=self.device,
            output_dir=self.temp_dir
        )
        
        # Resume from checkpoint
        trainer2.load_checkpoint(str(checkpoint_path))
        
        # Check that state was restored
        assert trainer2.step_count == trainer1.step_count
        
        # Continue training
        training_config.max_steps = trainer2.step_count + 10
        trainer2.train(dataloader)
        
        # Check that training continued
        assert trainer2.step_count > trainer1.step_count
    
    def test_smoke_mixed_precision(self):
        """Test training with mixed precision (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        # Create model and config
        model_config = ArborConfig(
            vocab_size=100,
            n_embd=64,
            n_layer=2,
            n_head=4,
            d_ff=128,
            max_length=16
        )
        
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=100)
        
        # Create dataset
        dataset = SyntheticDataset(
            size=50,
            vocab_size=100,
            sequence_length=16,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # Training config with AMP
        training_config = TrainingConfig(
            max_steps=15,
            learning_rate=1e-3,
            warmup_steps=2,
            log_interval=5,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            use_amp=True,  # Enable mixed precision
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device="cuda"
        )
        
        # Train
        trainer.train(dataloader)
        
        # Check that training completed
        assert trainer.step_count == 15
        assert trainer.scaler is not None  # AMP scaler should be created
    
    def test_smoke_gradient_accumulation(self):
        """Test training with gradient accumulation."""
        # Create model and config
        model_config = ArborConfig(
            vocab_size=50,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=8
        )
        
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=50)
        
        # Create dataset
        dataset = SyntheticDataset(
            size=60,
            vocab_size=50,
            sequence_length=8,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=2,  # Small batch size
            shuffle=True,
            num_workers=0
        )
        
        # Training config with gradient accumulation
        training_config = TrainingConfig(
            max_steps=10,
            learning_rate=1e-3,
            warmup_steps=2,
            log_interval=5,
            gradient_accumulation_steps=4,  # Accumulate over 4 steps
            max_grad_norm=1.0,
            use_amp=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device=self.device
        )
        
        # Train
        trainer.train(dataloader)
        
        # Check that training completed
        assert trainer.step_count == 10
        # Effective batch size should be 2 * 4 = 8
    
    def test_smoke_data_loading_edge_cases(self):
        """Test data loading with edge cases."""
        # Create very small dataset
        model_config = ArborConfig(
            vocab_size=20,
            n_embd=16,
            n_layer=1,
            n_head=2,
            d_ff=32,
            max_length=4
        )
        
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=20)
        
        # Very small dataset
        dataset = SyntheticDataset(
            size=5,  # Only 5 samples
            vocab_size=20,
            sequence_length=4,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        
        training_config = TrainingConfig(
            max_steps=8,  # More steps than data
            learning_rate=1e-3,
            warmup_steps=1,
            log_interval=3,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_amp=False,
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device=self.device
        )
        
        # Should handle cycling through small dataset
        trainer.train(dataloader)
        
        assert trainer.step_count == 8
    
    def test_smoke_model_generation(self):
        """Test model text generation after training."""
        # Create and train a small model
        model_config = ArborConfig(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=16
        )
        
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=100)
        
        dataset = SyntheticDataset(
            size=50,
            vocab_size=100,
            sequence_length=16,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        training_config = TrainingConfig(
            max_steps=20,
            learning_rate=1e-3,
            warmup_steps=2,
            log_interval=10,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            use_amp=False,
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device=self.device
        )
        
        # Train
        trainer.train(dataloader)
        
        # Test generation
        model.eval()
        with torch.no_grad():
            # Create a simple prompt
            prompt_ids = torch.randint(0, 100, (1, 5)).to(self.device)
            
            # Generate
            generated = model.generate(
                prompt_ids,
                max_new_tokens=10,
                temperature=1.0,
                do_sample=True
            )
            
            # Check output shape
            assert generated.shape[0] == 1
            assert generated.shape[1] == 15  # 5 prompt + 10 generated
            
            # Check that tokens are in valid range
            assert torch.all(generated >= 0)
            assert torch.all(generated < 100)


class TestSmokeErrors:
    """Test error handling in training pipeline."""
    
    def test_invalid_config_errors(self):
        """Test that invalid configurations raise appropriate errors."""
        # Invalid model config
        with pytest.raises((ValueError, AssertionError)):
            ArborConfig(vocab_size=0)  # Invalid vocab size
        
        with pytest.raises((ValueError, AssertionError)):
            ArborConfig(vocab_size=100, n_embd=0)  # Invalid embedding size
    
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        model_config = ArborConfig(
            vocab_size=50,
            n_embd=16,
            n_layer=1,
            n_head=2,
            d_ff=32,
            max_length=8
        )
        
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=50)
        
        # Put model on CPU but try to use CUDA device in trainer
        if torch.cuda.is_available():
            training_config = TrainingConfig(
                max_steps=1,
                learning_rate=1e-3,
            )
            
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                config=training_config,
                device="cuda"
            )
            
            # Model should be moved to CUDA automatically
            assert next(model.parameters()).device.type == "cuda"
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        model_config = ArborConfig(
            vocab_size=50,
            n_embd=16,
            n_layer=1,
            n_head=2,
            d_ff=32,
            max_length=8
        )
        
        model = ArborTransformer(model_config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=50)
        
        # Create empty dataset
        dataset = SyntheticDataset(
            size=0,  # Empty
            vocab_size=50,
            sequence_length=8,
            tokenizer=tokenizer
        )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        training_config = TrainingConfig(
            max_steps=5,
            learning_rate=1e-3,
        )
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            device="cpu"
        )
        
        # Should handle empty dataset gracefully
        trainer.train(dataloader)
        # Training should complete but step count may be 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
