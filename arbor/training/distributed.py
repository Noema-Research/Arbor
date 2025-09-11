"""
Distributed Training Setup for Enterprise Arbor Models.

This module provides utilities for setting up distributed training
across multiple GPUs and nodes for 200B-400B parameter models.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Optional, Dict, Any, Callable
import logging

from .enterprise import EnterpriseArborTransformer, EnterpriseArborConfig, EnterpriseArborBlock


logger = logging.getLogger(__name__)


class DistributedTrainingManager:
    """Manages distributed training setup for enterprise Arbor models."""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str = "localhost",
        master_port: str = "12355",
        backend: str = "nccl"
    ):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        
        # Initialize distributed process group
        self._init_distributed()
        
    def _init_distributed(self):
        """Initialize distributed training environment."""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank
        )
        
        # Set CUDA device
        torch.cuda.set_device(self.local_rank)
        
        logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
    
    def setup_model_parallel(self, model: EnterpriseArborTransformer) -> EnterpriseArborTransformer:
        """Setup model parallelism (FSDP + optional tensor/pipeline parallelism)."""
        
        # FSDP configuration for parameter sharding
        fsdp_config = {
            "auto_wrap_policy": transformer_auto_wrap_policy,
            "mixed_precision": torch.distributed.fsdp.MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            "sharding_strategy": torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
            "cpu_offload": torch.distributed.fsdp.CPUOffload(offload_params=True),
            "backward_prefetch": torch.distributed.fsdp.BackwardPrefetch.BACKWARD_PRE,
            "forward_prefetch": True,
        }
        
        # Wrap model with FSDP
        model = FSDP(model, **fsdp_config)
        
        logger.info(f"Setup FSDP model parallelism on rank {self.rank}")
        return model
    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


class EnterpriseTrainingConfig:
    """Configuration for enterprise-scale distributed training."""
    
    def __init__(
        self,
        # Model configuration
        model_config: EnterpriseArborConfig,
        
        # Training hyperparameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        max_grad_norm: float = 1.0,
        
        # Training schedule
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        
        # Batch configuration
        micro_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        
        # Memory optimization
        use_gradient_checkpointing: bool = True,
        use_cpu_offload: bool = True,
        use_activation_checkpointing: bool = True,
        
        # Logging and monitoring
        log_level: str = "INFO",
        wandb_project: Optional[str] = None,
        tensorboard_log_dir: Optional[str] = None,
        
        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        resume_from_checkpoint: Optional[str] = None,
        
        # Data configuration
        dataset_path: str = "",
        max_seq_length: int = 2048,
        num_workers: int = 4,
    ):
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_cpu_offload = use_cpu_offload
        self.use_activation_checkpointing = use_activation_checkpointing
        self.log_level = log_level
        self.wandb_project = wandb_project
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_dir = checkpoint_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.dataset_path = dataset_path
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        
        # Calculate effective batch size
        self.effective_batch_size = (
            self.micro_batch_size * 
            self.gradient_accumulation_steps * 
            self.model_config.data_parallel_size
        )


class EnterpriseTrainer:
    """Enterprise-scale trainer for Arbor models with distributed training."""
    
    def __init__(
        self,
        config: EnterpriseTrainingConfig,
        distributed_manager: DistributedTrainingManager
    ):
        self.config = config
        self.distributed_manager = distributed_manager
        self.device = f"cuda:{distributed_manager.local_rank}"
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize metrics
        self.step = 0
        self.metrics = {}
        
    def _create_model(self) -> EnterpriseArborTransformer:
        """Create and setup the enterprise model."""
        model = EnterpriseArborTransformer(self.config.model_config)
        model = model.to(self.device)
        
        # Setup distributed training
        model = self.distributed_manager.setup_model_parallel(model)
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter grouping."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
        
        return optimizer
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                # Cosine decay after warmup
                progress = (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(self.optimizer, lr_lambda)
        return scheduler
    
    def _setup_logging(self):
        """Setup logging and monitoring."""
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize wandb if specified
        if self.config.wandb_project and self.distributed_manager.rank == 0:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    config=self.config.__dict__,
                    name=f"arbor-enterprise-{self.config.model_config.target_params//1e9:.0f}B"
                )
            except ImportError:
                logger.warning("wandb not available, skipping wandb logging")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        
        # Compute loss (causal language modeling)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step (if accumulation is complete)
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Calculate metrics
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "lr": self.scheduler.get_last_lr()[0],
            "step": self.step
        }
        
        self.step += 1
        return metrics
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        if self.distributed_manager.rank == 0:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "step": self.step,
                "config": self.config,
            }
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint["step"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} at step {self.step}")


def launch_distributed_training(
    train_fn: Callable,
    world_size: int,
    config: EnterpriseTrainingConfig,
    master_addr: str = "localhost",
    master_port: str = "12355"
):
    """Launch distributed training across multiple processes."""
    
    def worker(rank):
        # Setup distributed manager
        distributed_manager = DistributedTrainingManager(
            world_size=world_size,
            rank=rank,
            local_rank=rank % torch.cuda.device_count(),
            master_addr=master_addr,
            master_port=master_port
        )
        
        try:
            # Run training function
            train_fn(config, distributed_manager)
        finally:
            # Cleanup
            distributed_manager.cleanup()
    
    # Launch processes
    mp.spawn(worker, args=(), nprocs=world_size, join=True)


# Example usage and configuration templates

def create_200b_training_config() -> EnterpriseTrainingConfig:
    """Create training configuration for 200B parameter model."""
    from .enterprise import create_enterprise_config
    
    model_config = create_enterprise_config(target_params=200_000_000_000)
    
    return EnterpriseTrainingConfig(
        model_config=model_config,
        learning_rate=3e-4,
        micro_batch_size=1,
        gradient_accumulation_steps=32,
        max_steps=100000,
        warmup_steps=2000,
        max_seq_length=4096,
        use_gradient_checkpointing=True,
        use_cpu_offload=True,
        checkpoint_dir="./checkpoints/arbor-200b"
    )


def create_400b_training_config() -> EnterpriseTrainingConfig:
    """Create training configuration for 400B parameter model."""
    from .enterprise import create_enterprise_config
    
    model_config = create_enterprise_config(target_params=400_000_000_000)
    
    return EnterpriseTrainingConfig(
        model_config=model_config,
        learning_rate=1e-4,
        micro_batch_size=1,
        gradient_accumulation_steps=64,
        max_steps=200000,
        warmup_steps=4000,
        max_seq_length=8192,
        use_gradient_checkpointing=True,
        use_cpu_offload=True,
        checkpoint_dir="./checkpoints/arbor-400b"
    )
