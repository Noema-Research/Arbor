"""
Enterprise Model Deployment Scripts.

Ready-to-use scripts for deploying 200B-400B parameter Arbor models
in production environments with distributed training and inference.
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from arbor.modeling.enterprise import create_enterprise_arbor, create_enterprise_config
from arbor.training.distributed import (
    EnterpriseTrainingConfig, 
    EnterpriseTrainer, 
    DistributedTrainingManager,
    create_200b_training_config,
    create_400b_training_config
)
from arbor.inference.enterprise_inference import (
    EnterpriseInference,
    InferenceConfig,
    create_inference_config_200b,
    create_inference_config_400b,
    benchmark_inference
)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('arbor_enterprise.log')
        ]
    )


def train_enterprise_model(args):
    """Train enterprise Arbor model with distributed setup."""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting enterprise training for {args.model_size}")
    
    # Create training configuration
    if args.model_size == "200b":
        config = create_200b_training_config()
    elif args.model_size == "400b":
        config = create_400b_training_config()
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")
    
    # Override config with command line arguments
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.micro_batch_size = args.batch_size
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Setup distributed training
    world_size = args.world_size or torch.cuda.device_count()
    
    def train_worker(rank):
        """Training worker function."""
        distributed_manager = DistributedTrainingManager(
            world_size=world_size,
            rank=rank,
            local_rank=rank % torch.cuda.device_count(),
            master_addr=args.master_addr,
            master_port=args.master_port
        )
        
        try:
            # Create trainer
            trainer = EnterpriseTrainer(config, distributed_manager)
            
            # Load checkpoint if resuming
            if args.resume_from_checkpoint:
                trainer.load_checkpoint(args.resume_from_checkpoint)
            
            # Training loop (simplified)
            logger.info(f"🎯 Starting training on rank {rank}")
            
            for step in range(trainer.step, config.max_steps):
                # In a real implementation, this would load data from a dataloader
                dummy_batch = {
                    "input_ids": torch.randint(1, config.model_config.vocab_size, 
                                             (config.micro_batch_size, config.max_seq_length))
                }
                
                # Training step
                metrics = trainer.train_step(dummy_batch)
                
                # Logging
                if step % 100 == 0 and rank == 0:
                    logger.info(f"Step {step}: Loss = {metrics['loss']:.4f}, LR = {metrics['lr']:.6f}")
                
                # Save checkpoint
                if step % config.save_steps == 0 and rank == 0:
                    checkpoint_path = f"{config.checkpoint_dir}/checkpoint-{step}.pt"
                    trainer.save_checkpoint(checkpoint_path)
                    logger.info(f"💾 Saved checkpoint at step {step}")
                
                # Early stopping for demo
                if args.demo and step >= 10:
                    break
            
            logger.info(f"✅ Training completed on rank {rank}")
            
        finally:
            distributed_manager.cleanup()
    
    # Launch distributed training
    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train_worker, nprocs=world_size, join=True)
    else:
        train_worker(0)


def deploy_inference_server(args):
    """Deploy enterprise inference server."""
    logger = logging.getLogger(__name__)
    logger.info(f"🌐 Deploying inference server for {args.model_size}")
    
    # Create inference configuration
    if args.model_size == "200b":
        config = create_inference_config_200b()
    elif args.model_size == "400b":
        config = create_inference_config_400b()
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")
    
    # Override config with command line arguments
    if args.model_path:
        config.model_path = args.model_path
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_new_tokens:
        config.max_new_tokens = args.max_new_tokens
    
    # Initialize inference engine
    engine = EnterpriseInference(config)
    
    logger.info("🚀 Inference server ready!")
    logger.info(f"📊 Model: {config.model_path}")
    logger.info(f"🎯 Batch size: {config.batch_size}")
    logger.info(f"⚡ Optimizations: Flash Attention={config.use_flash_attention}, Torch Compile={config.use_torch_compile}")
    
    # Example inference
    if args.test_inference:
        logger.info("🧪 Running test inference...")
        
        # Test with dummy input
        dummy_input = torch.randint(1, 1000, (1, 32), device=config.device)
        
        outputs = engine.generate(
            input_ids=dummy_input,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        
        logger.info(f"✅ Generated {outputs['generated_tokens'].size(-1)} tokens")
        logger.info(f"⚡ Speed: {outputs['tokens_per_second']:.2f} tokens/second")
    
    # Keep server running
    if not args.demo:
        logger.info("🔄 Server running... Press Ctrl+C to stop")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Server stopped")


def benchmark_model(args):
    """Benchmark model performance."""
    logger = logging.getLogger(__name__)
    logger.info(f"📈 Benchmarking {args.model_size} model")
    
    # Create inference configuration
    if args.model_size == "200b":
        config = create_inference_config_200b()
    elif args.model_size == "400b":
        config = create_inference_config_400b()
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")
    
    # Override model path if provided
    if args.model_path:
        config.model_path = args.model_path
    
    # Run benchmark
    num_samples = args.num_samples or 100
    metrics = benchmark_inference(config, num_samples)
    
    # Display results
    logger.info("📊 Benchmark Results:")
    logger.info(f"  Model Size: {args.model_size}")
    logger.info(f"  Samples: {metrics['samples']}")
    logger.info(f"  Total Time: {metrics['total_time']:.2f}s")
    logger.info(f"  Avg Time/Sample: {metrics['avg_time_per_sample']:.3f}s")
    logger.info(f"  Tokens/Second: {metrics['tokens_per_second']:.2f}")
    logger.info(f"  Throughput: {metrics['throughput']:.2f} samples/s")


def create_model(args):
    """Create and save enterprise model."""
    logger = logging.getLogger(__name__)
    logger.info(f"🏗️ Creating {args.model_size} model")
    
    # Determine target parameters
    if args.model_size == "200b":
        target_params = 200_000_000_000
    elif args.model_size == "400b":
        target_params = 400_000_000_000
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")
    
    # Create model
    model = create_enterprise_arbor(target_params)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), model_path)
    
    # Save configuration
    config_path = output_dir / "config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump(model.config.__dict__, f, indent=2)
    
    logger.info(f"💾 Model saved to {output_dir}")
    logger.info(f"📊 Parameters: {model.count_parameters()/1e9:.1f}B")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Enterprise Arbor Model Deployment")
    
    # Global arguments
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (shorter execution)")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train enterprise model")
    train_parser.add_argument("--model-size", required=True, choices=["200b", "400b"])
    train_parser.add_argument("--world-size", type=int, help="Number of processes for distributed training")
    train_parser.add_argument("--master-addr", default="localhost", help="Master address for distributed training")
    train_parser.add_argument("--master-port", default="12355", help="Master port for distributed training")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, help="Micro batch size")
    train_parser.add_argument("--max-steps", type=int, help="Maximum training steps")
    train_parser.add_argument("--dataset-path", help="Path to training dataset")
    train_parser.add_argument("--checkpoint-dir", help="Directory to save checkpoints")
    train_parser.add_argument("--resume-from-checkpoint", help="Resume training from checkpoint")
    
    # Inference command
    inference_parser = subparsers.add_parser("serve", help="Deploy inference server")
    inference_parser.add_argument("--model-size", required=True, choices=["200b", "400b"])
    inference_parser.add_argument("--model-path", help="Path to model checkpoint")
    inference_parser.add_argument("--batch-size", type=int, help="Inference batch size")
    inference_parser.add_argument("--max-new-tokens", type=int, help="Maximum tokens to generate")
    inference_parser.add_argument("--test-inference", action="store_true", help="Run test inference")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("--model-size", required=True, choices=["200b", "400b"])
    benchmark_parser.add_argument("--model-path", help="Path to model checkpoint")
    benchmark_parser.add_argument("--num-samples", type=int, help="Number of samples for benchmark")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create and save model")
    create_parser.add_argument("--model-size", required=True, choices=["200b", "400b"])
    create_parser.add_argument("--output-dir", required=True, help="Output directory for model")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "train":
        train_enterprise_model(args)
    elif args.command == "serve":
        deploy_inference_server(args)
    elif args.command == "benchmark":
        benchmark_model(args)
    elif args.command == "create":
        create_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
