"""
Metrics computation and evaluation utilities.
"""

from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import math


def compute_perplexity(loss: Union[float, torch.Tensor]) -> float:
    """
    Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    
    return math.exp(loss)


def compute_accuracy(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    ignore_index: int = -100
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        labels: True labels (batch, seq_len)
        ignore_index: Index to ignore in accuracy computation
        
    Returns:
        Accuracy as a float
    """
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid tokens
    mask = (labels != ignore_index)
    
    if mask.sum() == 0:
        return 0.0
    
    # Compute accuracy only on valid tokens
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_token_level_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute comprehensive token-level metrics.
    
    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        labels: True labels (batch, seq_len)
        ignore_index: Index to ignore in computation
        
    Returns:
        Dictionary of metrics
    """
    # Compute loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Compute accuracy
    accuracy = compute_accuracy(logits, labels, ignore_index)
    
    # Compute perplexity
    perplexity = compute_perplexity(loss)
    
    # Compute top-k accuracies
    top5_accuracy = compute_top_k_accuracy(logits, labels, k=5, ignore_index=ignore_index)
    top10_accuracy = compute_top_k_accuracy(logits, labels, k=10, ignore_index=ignore_index)
    
    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "perplexity": perplexity,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
    }


def compute_top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100,
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        labels: True labels (batch, seq_len)
        k: Number of top predictions to consider
        ignore_index: Index to ignore in computation
        
    Returns:
        Top-k accuracy as a float
    """
    # Get top-k predictions
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    
    # Expand labels to match top-k predictions
    labels_expanded = labels.unsqueeze(-1).expand(-1, -1, k)
    
    # Check if true label is in top-k predictions
    correct = (top_k_preds == labels_expanded).any(dim=-1)
    
    # Create mask for valid tokens
    mask = (labels != ignore_index)
    
    if mask.sum() == 0:
        return 0.0
    
    # Compute accuracy only on valid tokens
    top_k_correct = correct & mask
    accuracy = top_k_correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def compute_sequence_level_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    ignore_index: int = -100,
) -> Dict[str, Any]:
    """
    Compute sequence-level metrics.
    
    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        labels: True labels (batch, seq_len)
        tokenizer: Tokenizer for decoding
        ignore_index: Index to ignore in computation
        
    Returns:
        Dictionary of metrics and examples
    """
    batch_size = logits.size(0)
    predictions = torch.argmax(logits, dim=-1)
    
    exact_matches = 0
    bleu_scores = []
    
    for i in range(batch_size):
        pred_seq = predictions[i]
        true_seq = labels[i]
        
        # Remove ignore tokens
        valid_mask = (true_seq != ignore_index)
        pred_seq = pred_seq[valid_mask]
        true_seq = true_seq[valid_mask]
        
        # Exact match
        if torch.equal(pred_seq, true_seq):
            exact_matches += 1
        
        # BLEU score (simplified)
        try:
            pred_text = tokenizer.decode(pred_seq.tolist())
            true_text = tokenizer.decode(true_seq.tolist())
            
            # Simple word-level BLEU-1
            pred_words = pred_text.split()
            true_words = true_text.split()
            
            if len(true_words) > 0:
                overlap = len(set(pred_words) & set(true_words))
                bleu_1 = overlap / len(true_words)
                bleu_scores.append(bleu_1)
        except:
            # Skip if decoding fails
            pass
    
    return {
        "exact_match_ratio": exact_matches / batch_size,
        "avg_bleu_1": np.mean(bleu_scores) if bleu_scores else 0.0,
        "num_sequences": batch_size,
    }


def compute_model_size_metrics(model: nn.Module) -> Dict[str, Any]:
    """
    Compute model size and efficiency metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of size metrics
    """
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory usage (rough estimate)
    param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Model depth
    num_layers = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            num_layers += 1
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_memory_mb": param_memory_mb,
        "num_layers": num_layers,
        "parameters_per_layer": total_params / max(num_layers, 1),
    }


def compute_growth_metrics(growth_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metrics about model growth events.
    
    Args:
        growth_history: List of growth events
        
    Returns:
        Dictionary of growth metrics
    """
    if not growth_history:
        return {
            "num_growth_events": 0,
            "total_parameter_increase": 0,
            "growth_rate": 0.0,
        }
    
    num_events = len(growth_history)
    
    # Parameter growth
    initial_params = growth_history[0]["old_param_count"]
    final_params = growth_history[-1]["new_param_count"]
    total_increase = final_params - initial_params
    growth_rate = (final_params / initial_params) if initial_params > 0 else 1.0
    
    # Growth timing
    steps = [event["step"] for event in growth_history]
    if len(steps) > 1:
        avg_steps_between_growth = np.mean(np.diff(steps))
    else:
        avg_steps_between_growth = 0
    
    # Parameter increase per event
    param_increases = [event.get("param_increase", 0) for event in growth_history]
    avg_param_increase = np.mean(param_increases) if param_increases else 0
    
    return {
        "num_growth_events": num_events,
        "initial_parameters": initial_params,
        "final_parameters": final_params,
        "total_parameter_increase": total_increase,
        "growth_rate": growth_rate,
        "avg_steps_between_growth": avg_steps_between_growth,
        "avg_param_increase_per_event": avg_param_increase,
        "growth_efficiency": total_increase / max(num_events, 1),
    }


def compute_efficiency_metrics(
    model: nn.Module,
    loss: float,
    training_time: float,
    num_tokens: int,
) -> Dict[str, float]:
    """
    Compute training efficiency metrics.
    
    Args:
        model: The trained model
        loss: Final loss value
        training_time: Training time in seconds
        num_tokens: Total number of tokens processed
        
    Returns:
        Dictionary of efficiency metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "tokens_per_second": num_tokens / training_time if training_time > 0 else 0,
        "params_per_token": total_params / num_tokens if num_tokens > 0 else 0,
        "loss_per_param": loss / total_params if total_params > 0 else float('inf'),
        "training_efficiency": (1.0 / loss) / total_params if total_params > 0 and loss > 0 else 0,
    }


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
    logger=None,
) -> None:
    """
    Log metrics to various logging systems.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
        prefix: Prefix for metric names
        logger: Logger instance
    """
    # Console logging
    if logger:
        log_str = f"Step {step}: " if step is not None else ""
        log_str += ", ".join([f"{prefix}{k}={v:.4f}" if isinstance(v, float) else f"{prefix}{k}={v}" 
                             for k, v in metrics.items()])
        logger.info(log_str)
    
    # Weights & Biases logging
    try:
        import wandb
        if wandb.run is not None:
            prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            wandb.log(prefixed_metrics, step=step)
    except ImportError:
        pass
    except Exception as e:
        if logger:
            logger.warning(f"Failed to log to wandb: {e}")


def create_metrics_summary(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    model_metrics: Dict[str, Any],
    growth_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a comprehensive metrics summary.
    
    Args:
        train_metrics: Training metrics
        val_metrics: Validation metrics
        model_metrics: Model size metrics
        growth_metrics: Growth metrics (optional)
        
    Returns:
        Combined metrics summary
    """
    summary = {
        "training": train_metrics,
        "validation": val_metrics,
        "model": model_metrics,
    }
    
    if growth_metrics:
        summary["growth"] = growth_metrics
    
    # Compute derived metrics
    summary["derived"] = {
        "overfitting_ratio": val_metrics.get("loss", 0) / max(train_metrics.get("loss", 1e-8), 1e-8),
        "validation_perplexity": compute_perplexity(val_metrics.get("loss", 0)),
        "parameter_efficiency": val_metrics.get("loss", float('inf')) / model_metrics.get("total_parameters", 1),
    }
    
    return summary


def compare_models(
    model1_metrics: Dict[str, Any],
    model2_metrics: Dict[str, Any],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
) -> Dict[str, Any]:
    """
    Compare metrics between two models.
    
    Args:
        model1_metrics: Metrics for first model
        model2_metrics: Metrics for second model
        model1_name: Name for first model
        model2_name: Name for second model
        
    Returns:
        Comparison results
    """
    comparison = {
        "models": {
            model1_name: model1_metrics,
            model2_name: model2_metrics,
        },
        "comparison": {},
    }
    
    # Compare key metrics
    key_metrics = ["loss", "perplexity", "accuracy", "total_parameters"]
    
    for metric in key_metrics:
        val1 = model1_metrics.get(metric)
        val2 = model2_metrics.get(metric)
        
        if val1 is not None and val2 is not None:
            if metric in ["loss"]:  # Lower is better
                winner = model1_name if val1 < val2 else model2_name
                improvement = abs(val1 - val2) / max(val1, val2)
            elif metric in ["perplexity"]:  # Lower is better
                winner = model1_name if val1 < val2 else model2_name
                improvement = abs(val1 - val2) / max(val1, val2)
            else:  # Higher is better
                winner = model1_name if val1 > val2 else model2_name
                improvement = abs(val1 - val2) / max(val1, val2)
            
            comparison["comparison"][metric] = {
                model1_name: val1,
                model2_name: val2,
                "winner": winner,
                "improvement": improvement,
            }
    
    return comparison
