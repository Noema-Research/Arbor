"""
Optimizer utilities for handling parameter growth.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


def get_new_parameters(model: nn.Module, old_param_names: set) -> List[nn.Parameter]:
    """
    Get parameters that were added to the model since last check.
    
    Args:
        model: The PyTorch model
        old_param_names: Set of parameter names from before growth
        
    Returns:
        List of new parameters
    """
    current_param_names = {name for name, _ in model.named_parameters()}
    new_param_names = current_param_names - old_param_names
    
    new_params = []
    for name, param in model.named_parameters():
        if name in new_param_names:
            new_params.append(param)
    
    return new_params


def add_parameters_to_optimizer(
    optimizer: torch.optim.Optimizer, 
    new_parameters: List[nn.Parameter],
    lr_multiplier: float = 1.0,
) -> None:
    """
    Add new parameters to an existing optimizer.
    
    Args:
        optimizer: The optimizer to update
        new_parameters: List of new parameters to add
        lr_multiplier: Learning rate multiplier for new parameters
    """
    if not new_parameters:
        return
    
    # Get reference parameter group settings
    if len(optimizer.param_groups) > 0:
        reference_group = optimizer.param_groups[0]
        
        # Create new parameter group with same settings
        new_group = {
            key: value for key, value in reference_group.items() 
            if key != 'params'
        }
        new_group['params'] = new_parameters
        
        # Apply learning rate multiplier
        if 'lr' in new_group:
            new_group['lr'] *= lr_multiplier
            
    else:
        # Fallback if no existing parameter groups
        new_group = {
            'params': new_parameters,
            'lr': 1e-4 * lr_multiplier,
        }
    
    # Add the new parameter group
    optimizer.add_param_group(new_group)
    
    print(f"Added {len(new_parameters)} parameters to optimizer "
          f"with lr_multiplier={lr_multiplier}")


def reset_optimizer_state_for_new_params(
    optimizer: torch.optim.Optimizer,
    new_parameters: List[nn.Parameter],
) -> None:
    """
    Reset/initialize optimizer state for new parameters.
    
    Args:
        optimizer: The optimizer
        new_parameters: List of new parameters
    """
    for param in new_parameters:
        if param in optimizer.state:
            # Clear existing state
            del optimizer.state[param]
        
        # Initialize fresh state based on optimizer type
        optimizer.state[param] = {}
        
        if isinstance(optimizer, torch.optim.AdamW):
            optimizer.state[param]['step'] = 0
            optimizer.state[param]['exp_avg'] = torch.zeros_like(param)
            optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param)
            
        elif isinstance(optimizer, torch.optim.Adam):
            optimizer.state[param]['step'] = 0
            optimizer.state[param]['exp_avg'] = torch.zeros_like(param)
            optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param)
            
        elif isinstance(optimizer, torch.optim.SGD):
            if optimizer.defaults.get('momentum', 0) > 0:
                optimizer.state[param]['momentum_buffer'] = torch.zeros_like(param)
                
        elif isinstance(optimizer, torch.optim.RMSprop):
            optimizer.state[param]['step'] = 0
            optimizer.state[param]['square_avg'] = torch.zeros_like(param)
            if optimizer.defaults.get('momentum', 0) > 0:
                optimizer.state[param]['momentum_buffer'] = torch.zeros_like(param)
            if optimizer.defaults.get('centered', False):
                optimizer.state[param]['grad_avg'] = torch.zeros_like(param)


def warmup_new_parameters(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    new_parameters: List[nn.Parameter],
    dataloader: torch.utils.data.DataLoader,
    warmup_steps: int = 100,
    warmup_lr: float = 1e-5,
    device: str = "cuda",
) -> None:
    """
    Perform warmup training on new parameters while keeping old parameters frozen.
    
    Args:
        model: The model
        optimizer: The optimizer
        new_parameters: List of new parameters to warm up
        dataloader: Training dataloader
        warmup_steps: Number of warmup steps
        warmup_lr: Learning rate for warmup
        device: Device to use
    """
    model.train()
    
    # Store original requires_grad states
    original_states = {}
    for name, param in model.named_parameters():
        original_states[name] = param.requires_grad
        if param not in new_parameters:
            param.requires_grad = False  # Freeze old parameters
    
    # Create temporary optimizer for warmup
    warmup_optimizer = torch.optim.AdamW(new_parameters, lr=warmup_lr)
    
    print(f"Starting warmup for {len(new_parameters)} new parameters...")
    
    step_count = 0
    for batch in dataloader:
        if step_count >= warmup_steps:
            break
            
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
        else:
            input_ids = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else input_ids
        
        # Forward pass
        outputs = model(input_ids, labels=labels, return_dict=True)
        loss = outputs["loss"]
        
        # Backward pass
        warmup_optimizer.zero_grad()
        loss.backward()
        warmup_optimizer.step()
        
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"Warmup step {step_count}/{warmup_steps}, loss: {loss.item():.4f}")
    
    # Restore original requires_grad states
    for name, param in model.named_parameters():
        param.requires_grad = original_states[name]
    
    print(f"Warmup completed after {step_count} steps")


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the L2 norm of model gradients.
    
    Args:
        model: The model
        
    Returns:
        Gradient norm as a float
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    total_norm = total_norm ** 0.5
    return total_norm


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from optimizer.
    
    Args:
        optimizer: The optimizer
        
    Returns:
        Current learning rate
    """
    if len(optimizer.param_groups) > 0:
        return optimizer.param_groups[0]['lr']
    return 0.0


def scale_optimizer_lr(optimizer: torch.optim.Optimizer, scale_factor: float) -> None:
    """
    Scale the learning rate of all parameter groups.
    
    Args:
        optimizer: The optimizer
        scale_factor: Factor to multiply learning rate by
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale_factor
    
    print(f"Scaled optimizer learning rate by {scale_factor}")


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: The model
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Adam beta parameters
        eps: Adam epsilon
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    parameters = model.parameters()
    
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def get_parameter_counts_by_layer(model: nn.Module) -> Dict[str, int]:
    """
    Get parameter counts broken down by layer/module.
    
    Args:
        model: The model
        
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    layer_counts = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                layer_counts[name] = param_count
    
    return layer_counts
