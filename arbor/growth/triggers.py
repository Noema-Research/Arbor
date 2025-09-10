"""
Growth triggers for dynamic model expansion.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import torch
import numpy as np
from collections import deque


class GrowthTrigger(ABC):
    """Base class for growth triggers."""
    
    @abstractmethod
    def should_trigger(self, **kwargs) -> bool:
        """Check if the trigger condition is met."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the trigger state."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state for checkpointing."""
        pass
    
    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        pass


class PlateauTrigger(GrowthTrigger):
    """
    Trigger growth when validation loss plateaus.
    
    Fires when the best validation loss hasn't improved by more than
    `eps` for `window_steps` consecutive steps.
    """
    
    def __init__(self, window_steps: int = 1000, eps: float = 1e-4):
        self.window_steps = window_steps
        self.eps = eps
        
        self.best_loss = float('inf')
        self.steps_since_improvement = 0
        self.step_count = 0
        
    def should_trigger(self, val_loss: float, **kwargs) -> bool:
        """Check if validation loss has plateaued."""
        self.step_count += 1
        
        if val_loss < self.best_loss - self.eps:
            self.best_loss = val_loss
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        should_fire = self.steps_since_improvement >= self.window_steps
        
        if should_fire:
            print(f"PlateauTrigger fired: {self.steps_since_improvement} steps since improvement "
                  f"(best_loss={self.best_loss:.6f})")
        
        return should_fire
    
    def reset(self) -> None:
        """Reset trigger state after growth event."""
        self.steps_since_improvement = 0
        # Don't reset best_loss - we want to continue from current performance
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "best_loss": self.best_loss,
            "steps_since_improvement": self.steps_since_improvement,
            "step_count": self.step_count,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        self.best_loss = state["best_loss"]
        self.steps_since_improvement = state["steps_since_improvement"]
        self.step_count = state["step_count"]


class GradientNormTrigger(GrowthTrigger):
    """
    Trigger growth when gradient norms become too small.
    
    Fires when the average gradient norm over a window falls below
    a threshold, indicating potential underfitting.
    """
    
    def __init__(self, threshold: float = 1e-3, window_steps: int = 500):
        self.threshold = threshold
        self.window_steps = window_steps
        
        self.grad_norms = deque(maxlen=window_steps)
        self.step_count = 0
        
    def should_trigger(self, grad_norm: float, **kwargs) -> bool:
        """Check if gradient norm is too small."""
        self.step_count += 1
        self.grad_norms.append(grad_norm)
        
        if len(self.grad_norms) < self.window_steps:
            return False
        
        avg_grad_norm = np.mean(self.grad_norms)
        should_fire = avg_grad_norm < self.threshold
        
        if should_fire:
            print(f"GradientNormTrigger fired: avg_grad_norm={avg_grad_norm:.6f} < {self.threshold}")
        
        return should_fire
    
    def reset(self) -> None:
        """Reset trigger state after growth event."""
        self.grad_norms.clear()
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "grad_norms": list(self.grad_norms),
            "step_count": self.step_count,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        self.grad_norms = deque(state["grad_norms"], maxlen=self.window_steps)
        self.step_count = state["step_count"]


class LossSpikeTrigger(GrowthTrigger):
    """
    Trigger growth when loss spikes on specific data slices.
    
    Monitors loss on different data slices and triggers growth
    when a slice shows significantly higher loss than the baseline.
    """
    
    def __init__(
        self, 
        slice_fn: Callable[[torch.Tensor], torch.Tensor],
        ratio_threshold: float = 2.0,
        window_steps: int = 100,
        min_samples: int = 50
    ):
        self.slice_fn = slice_fn
        self.ratio_threshold = ratio_threshold
        self.window_steps = window_steps
        self.min_samples = min_samples
        
        self.baseline_losses = deque(maxlen=window_steps)
        self.slice_losses = deque(maxlen=window_steps)
        self.step_count = 0
        
    def should_trigger(
        self, 
        input_ids: torch.Tensor, 
        losses: torch.Tensor, 
        **kwargs
    ) -> bool:
        """
        Check if slice loss is significantly higher than baseline.
        
        Args:
            input_ids: Input token ids (batch, seq_len)
            losses: Per-sample losses (batch,)
        """
        self.step_count += 1
        
        # Get slice mask
        slice_mask = self.slice_fn(input_ids)
        
        if slice_mask.sum() == 0:
            return False  # No samples in slice
        
        # Compute losses
        baseline_loss = losses[~slice_mask].mean().item() if (~slice_mask).sum() > 0 else losses.mean().item()
        slice_loss = losses[slice_mask].mean().item()
        
        self.baseline_losses.append(baseline_loss)
        self.slice_losses.append(slice_loss)
        
        if len(self.slice_losses) < self.min_samples:
            return False
        
        # Check if slice loss is significantly higher
        avg_baseline = np.mean(self.baseline_losses)
        avg_slice = np.mean(self.slice_losses)
        
        if avg_baseline > 0:  # Avoid division by zero
            ratio = avg_slice / avg_baseline
            should_fire = ratio > self.ratio_threshold
            
            if should_fire:
                print(f"LossSpikeTrigger fired: slice_loss={avg_slice:.6f} / "
                      f"baseline_loss={avg_baseline:.6f} = {ratio:.2f} > {self.ratio_threshold}")
            
            return should_fire
        
        return False
    
    def reset(self) -> None:
        """Reset trigger state after growth event."""
        self.baseline_losses.clear()
        self.slice_losses.clear()
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "baseline_losses": list(self.baseline_losses),
            "slice_losses": list(self.slice_losses),
            "step_count": self.step_count,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        self.baseline_losses = deque(state["baseline_losses"], maxlen=self.window_steps)
        self.slice_losses = deque(state["slice_losses"], maxlen=self.window_steps)
        self.step_count = state["step_count"]


class PerplexityTrigger(GrowthTrigger):
    """
    Trigger growth when perplexity improvement stagnates.
    
    Similar to plateau trigger but works with perplexity instead of loss.
    """
    
    def __init__(self, window_steps: int = 1000, improvement_threshold: float = 0.01):
        self.window_steps = window_steps
        self.improvement_threshold = improvement_threshold
        
        self.best_perplexity = float('inf')
        self.steps_since_improvement = 0
        self.step_count = 0
        
    def should_trigger(self, perplexity: float, **kwargs) -> bool:
        """Check if perplexity improvement has stagnated."""
        self.step_count += 1
        
        # Calculate improvement ratio
        if self.best_perplexity < float('inf'):
            improvement_ratio = (self.best_perplexity - perplexity) / self.best_perplexity
        else:
            improvement_ratio = float('inf')
        
        if improvement_ratio > self.improvement_threshold:
            self.best_perplexity = perplexity
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        should_fire = self.steps_since_improvement >= self.window_steps
        
        if should_fire:
            print(f"PerplexityTrigger fired: {self.steps_since_improvement} steps since improvement "
                  f"(best_perplexity={self.best_perplexity:.2f})")
        
        return should_fire
    
    def reset(self) -> None:
        """Reset trigger state after growth event."""
        self.steps_since_improvement = 0
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "best_perplexity": self.best_perplexity,
            "steps_since_improvement": self.steps_since_improvement,
            "step_count": self.step_count,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        self.best_perplexity = state["best_perplexity"]
        self.steps_since_improvement = state["steps_since_improvement"]
        self.step_count = state["step_count"]


class CompositeTrigger(GrowthTrigger):
    """
    Composite trigger that combines multiple triggers with logical operators.
    
    Can use AND, OR, or custom logic to combine trigger conditions.
    """
    
    def __init__(self, triggers: List[GrowthTrigger], logic: str = "any"):
        """
        Args:
            triggers: List of individual triggers
            logic: "any" (OR), "all" (AND), or "majority"
        """
        self.triggers = triggers
        self.logic = logic
        
    def should_trigger(self, **kwargs) -> bool:
        """Check composite trigger condition."""
        trigger_results = [trigger.should_trigger(**kwargs) for trigger in self.triggers]
        
        if self.logic == "any":
            return any(trigger_results)
        elif self.logic == "all":
            return all(trigger_results)
        elif self.logic == "majority":
            return sum(trigger_results) > len(trigger_results) // 2
        else:
            raise ValueError(f"Unknown logic: {self.logic}")
    
    def reset(self) -> None:
        """Reset all sub-triggers."""
        for trigger in self.triggers:
            trigger.reset()
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "trigger_states": [trigger.get_state() for trigger in self.triggers],
            "logic": self.logic,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        trigger_states = state["trigger_states"]
        for trigger, trigger_state in zip(self.triggers, trigger_states):
            trigger.load_state(trigger_state)


# Utility functions for creating common slice functions
def create_length_slice_fn(min_length: int, max_length: Optional[int] = None):
    """Create a slice function that selects sequences by length."""
    def slice_fn(input_ids: torch.Tensor) -> torch.Tensor:
        # Count non-padding tokens (assuming 0 is padding)
        lengths = (input_ids != 0).sum(dim=1)
        if max_length is None:
            return lengths >= min_length
        else:
            return (lengths >= min_length) & (lengths <= max_length)
    return slice_fn


def create_token_slice_fn(target_tokens: List[int]):
    """Create a slice function that selects sequences containing specific tokens."""
    def slice_fn(input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for token in target_tokens:
            mask |= (input_ids == token).any(dim=1)
        return mask
    return slice_fn


def create_random_slice_fn(probability: float = 0.1):
    """Create a slice function that randomly selects sequences."""
    def slice_fn(input_ids: torch.Tensor) -> torch.Tensor:
        return torch.rand(input_ids.size(0), device=input_ids.device) < probability
    return slice_fn
