"""
Main training loop implementation with growth support.
"""

from typing import Dict, Any, Optional, Callable, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import logging

from ..modeling.model import ArborTransformer
from ..growth.manager import GrowthManager
from .optimizer_utils import (
    compute_gradient_norm, 
    get_optimizer_lr, 
    add_parameters_to_optimizer,
    get_new_parameters
)
from .checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from ..utils.metrics import compute_perplexity, log_metrics
from ..utils.logging import setup_logging, log_growth_event


class Trainer:
    """
    Main trainer class for Arbor models with growth support.
    
    Handles training loop, validation, checkpointing, and integration
    with the growth manager for dynamic model expansion.
    """
    
    def __init__(
        self,
        model: ArborTransformer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        growth_manager: Optional[GrowthManager],
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.growth_manager = growth_manager
        self.config = config
        self.device = device
        
        # Training settings
        self.max_steps = config.get("max_steps", 10000)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.eval_every = config.get("eval_every", 500)
        self.save_every = config.get("save_every", 1000)
        self.log_every = config.get("log_every", 100)
        
        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        self.resume_from_checkpoint = config.get("resume_from_checkpoint", True)
        
        # State tracking
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        
        # Setup logging
        self.logger = setup_logging(config.get("log_level", "INFO"))
        
        # Track parameter names for growth detection
        self.param_names_before_growth = {name for name, _ in model.named_parameters()}
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Load checkpoint if resuming
        if self.resume_from_checkpoint:
            self._maybe_load_checkpoint()
    
    def _maybe_load_checkpoint(self) -> None:
        """Load checkpoint if available."""
        latest_checkpoint = find_latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            try:
                checkpoint_data = load_checkpoint(
                    latest_checkpoint,
                    model=self.model,
                    optimizer=self.optimizer,
                    growth_manager=self.growth_manager,
                    device=self.device,
                )
                
                self.step = checkpoint_data["step"]
                self.best_val_loss = checkpoint_data.get("loss", float('inf'))
                
                self.logger.info(f"Resumed from checkpoint at step {self.step}")
                
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training statistics and metrics
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Model parameters: {self.model.param_count():,}")
        
        self.model.train()
        
        # Training metrics
        total_loss = 0.0
        total_tokens = 0
        last_log_step = 0
        
        # Create progress bar
        pbar = tqdm(total=self.max_steps, initial=self.step, desc="Training")
        
        while self.step < self.max_steps:
            epoch_start_step = self.step
            
            for batch in self.train_dataloader:
                if self.step >= self.max_steps:
                    break
                
                # Training step
                step_metrics = self._training_step(batch)
                
                # Update metrics
                total_loss += step_metrics["loss"]
                total_tokens += step_metrics["tokens"]
                
                # Check for growth
                if self.growth_manager is not None:
                    growth_metrics = {
                        "val_loss": step_metrics.get("val_loss"),
                        "grad_norm": step_metrics["grad_norm"],
                        "perplexity": step_metrics.get("perplexity"),
                    }
                    
                    # Check if growth should occur
                    if self.growth_manager.step(growth_metrics):
                        self._handle_growth_event()
                
                # Validation and logging
                if self.step % self.eval_every == 0:
                    val_metrics = self._validation_step()
                    step_metrics.update(val_metrics)
                    
                    # Update best validation loss
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        step_metrics["is_best"] = True
                
                # Logging
                if self.step % self.log_every == 0:
                    self._log_metrics(step_metrics, total_loss, total_tokens, last_log_step)
                    last_log_step = self.step
                
                # Checkpointing
                if self.step % self.save_every == 0:
                    self._save_checkpoint(step_metrics.get("val_loss", total_loss), 
                                        step_metrics.get("is_best", False))
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{step_metrics['loss']:.4f}",
                    "lr": f"{get_optimizer_lr(self.optimizer):.2e}",
                    "params": f"{self.model.param_count():,}",
                })
                
                self.step += 1
            
            self.epoch += 1
        
        pbar.close()
        
        # Final validation and checkpoint
        final_val_metrics = self._validation_step()
        self._save_checkpoint(final_val_metrics["val_loss"], is_best=False)
        
        # Training summary
        training_time = time.time() - self.start_time
        
        summary = {
            "final_step": self.step,
            "final_epoch": self.epoch,
            "final_val_loss": final_val_metrics["val_loss"],
            "final_val_perplexity": final_val_metrics["val_perplexity"],
            "best_val_loss": self.best_val_loss,
            "training_time": training_time,
            "final_param_count": self.model.param_count(),
            "growth_events": len(self.model.get_growth_history()),
        }
        
        self.logger.info("Training completed!")
        self.logger.info(f"Final validation loss: {final_val_metrics['val_loss']:.4f}")
        self.logger.info(f"Final parameter count: {self.model.param_count():,}")
        
        return summary
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute one training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Step metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask")
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            loss = outputs["loss"] / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.step + 1) % self.gradient_accumulation_steps == 0:
            # Compute gradient norm before clipping
            grad_norm = compute_gradient_norm(self.model)
            
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
        else:
            grad_norm = 0.0
        
        # Calculate tokens processed
        tokens = input_ids.numel()
        
        return {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "grad_norm": grad_norm,
            "tokens": tokens,
            "learning_rate": get_optimizer_lr(self.optimizer),
        }
    
    def _validation_step(self) -> Dict[str, Any]:
        """
        Execute validation and return metrics.
        
        Returns:
            Validation metrics
        """
        if self.val_dataloader is None:
            return {"val_loss": 0.0, "val_perplexity": 0.0}
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                input_ids = batch["input_ids"]
                labels = batch.get("labels", input_ids)
                attention_mask = batch.get("attention_mask")
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                total_tokens += input_ids.numel()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = compute_perplexity(avg_loss)
        
        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "val_tokens": total_tokens,
        }
    
    def _handle_growth_event(self) -> None:
        """Handle a growth event by updating optimizer and logging."""
        self.logger.info("Growth event detected!")
        
        # Get new parameters added by growth
        current_param_names = {name for name, _ in self.model.named_parameters()}
        new_params = get_new_parameters(self.model, self.param_names_before_growth)
        
        if new_params:
            # Add new parameters to optimizer
            add_parameters_to_optimizer(
                self.optimizer, 
                new_params, 
                lr_multiplier=self.config.get("new_param_lr_multiplier", 1.0)
            )
            
            # Update parameter tracking
            self.param_names_before_growth = current_param_names
            
            # Log growth event
            if hasattr(log_growth_event, '__call__'):
                log_growth_event(
                    step=self.step,
                    old_param_count=self.model.get_growth_history()[-1]["old_param_count"],
                    new_param_count=self.model.param_count(),
                    growth_details=self.model.get_growth_history()[-1],
                )
        
        self.logger.info(f"Model now has {self.model.param_count():,} parameters")
    
    def _log_metrics(
        self, 
        step_metrics: Dict[str, Any], 
        total_loss: float, 
        total_tokens: int,
        last_log_step: int,
    ) -> None:
        """Log training metrics."""
        steps_since_log = self.step - last_log_step
        avg_loss = total_loss / max(steps_since_log, 1)
        
        metrics = {
            "step": self.step,
            "epoch": self.epoch,
            "loss": step_metrics["loss"],
            "avg_loss": avg_loss,
            "grad_norm": step_metrics["grad_norm"],
            "learning_rate": step_metrics["learning_rate"],
            "param_count": self.model.param_count(),
            "tokens_per_sec": total_tokens / max(time.time() - self.start_time, 1),
        }
        
        # Add validation metrics if available
        if "val_loss" in step_metrics:
            metrics.update({
                "val_loss": step_metrics["val_loss"],
                "val_perplexity": step_metrics["val_perplexity"],
            })
        
        # Add growth information
        if self.growth_manager:
            metrics["growth_events"] = len(self.model.get_growth_history())
            metrics["steps_since_growth"] = self.growth_manager.steps_since_last_growth
        
        # Log to wandb or other logging system
        log_metrics(metrics, step=self.step)
        
        # Console logging
        log_str = f"Step {self.step}: loss={metrics['loss']:.4f}"
        if "val_loss" in metrics:
            log_str += f", val_loss={metrics['val_loss']:.4f}"
        log_str += f", lr={metrics['learning_rate']:.2e}, params={metrics['param_count']:,}"
        
        self.logger.info(log_str)
    
    def _save_checkpoint(self, loss: float, is_best: bool = False) -> None:
        """Save a training checkpoint."""
        additional_info = {
            "epoch": self.epoch,
            "config": self.config,
            "training_time": time.time() - self.start_time,
        }
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            growth_manager=self.growth_manager,
            step=self.step,
            loss=loss,
            checkpoint_dir=self.checkpoint_dir,
            is_best=is_best,
            additional_info=additional_info,
        )
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Dataloader to evaluate on (uses validation set if None)
            
        Returns:
            Evaluation metrics
        """
        if dataloader is None:
            dataloader = self.val_dataloader
        
        if dataloader is None:
            raise ValueError("No dataloader provided for evaluation")
        
        self.logger.info("Starting evaluation...")
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                input_ids = batch["input_ids"]
                labels = batch.get("labels", input_ids)
                attention_mask = batch.get("attention_mask")
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                total_tokens += input_ids.numel()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches
        perplexity = compute_perplexity(avg_loss)
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_tokens": total_tokens,
        }
        
        self.logger.info(f"Evaluation completed: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")
        
        return metrics


def create_trainer(
    model: ArborTransformer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    config: Dict[str, Any],
    device: str = "cuda",
) -> Trainer:
    """
    Factory function to create a trainer with optimizer and scheduler.
    
    Args:
        model: The model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Training configuration
        device: Device to use
        
    Returns:
        Configured trainer
    """
    from .optimizer_utils import create_optimizer
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_type=config.get("optimizer", "adamw"),
        learning_rate=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
        betas=config.get("betas", (0.9, 0.999)),
    )
    
    # Create scheduler
    scheduler = None
    if config.get("use_scheduler", True):
        total_steps = config.get("max_steps", 10000)
        warmup_steps = config.get("warmup_steps", int(0.1 * total_steps))
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get("learning_rate", 1e-4),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )
    
    # Create growth manager
    growth_manager = None
    if config.get("growth", {}).get("enabled", False):
        growth_manager = GrowthManager(
            model=model,
            optimizer=optimizer,
            config=config["growth"],
        )
    
    return Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        growth_manager=growth_manager,
        config=config,
        device=device,
    )
