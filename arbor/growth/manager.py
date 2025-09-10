"""
Growth manager for orchestrating dynamic expansion in Arbor architecture.

Used by Arbor-o1 and other models built on the Arbor architecture.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import random

from .triggers import GrowthTrigger, PlateauTrigger, GradientNormTrigger, LossSpikeTrigger
from ..modeling.model import ArborTransformer


class GrowthManager:
    """
    Manages dynamic growth of the Arbor transformer architecture.
    
    The GrowthManager monitors training metrics and triggers model expansion
    when certain conditions are met. It handles the coordination between
    triggers, growth policies, and optimizer updates.
    """
    
    def __init__(
        self,
        model: ArborTransformer,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        
        # Growth settings
        self.enabled = config.get("enabled", True)
        self.add_hidden = config.get("add_hidden", 256)
        self.max_events = config.get("max_events", 6)
        self.cooldown_steps = config.get("cooldown_steps", 5000)
        self.layer_selection = config.get("layer_selection", "uniform")  # "uniform", "top_k", "bottom_k"
        self.layers_per_event = config.get("layers_per_event", 2)
        
        # State tracking
        self.growth_events: List[Dict[str, Any]] = []
        self.steps_since_last_growth = 0
        self.total_steps = 0
        
        # Initialize triggers
        self.triggers = self._create_triggers(config.get("triggers", []))
        
        # Growth policy
        self.growth_policy = self._create_growth_policy()
        
    def _create_triggers(self, trigger_configs: List[Dict[str, Any]]) -> List[GrowthTrigger]:
        """Create trigger objects from configuration."""
        triggers = []
        
        for trigger_config in trigger_configs:
            trigger_type = trigger_config["type"]
            trigger_params = {k: v for k, v in trigger_config.items() if k != "type"}
            
            if trigger_type == "plateau":
                triggers.append(PlateauTrigger(**trigger_params))
            elif trigger_type == "gradnorm":
                triggers.append(GradientNormTrigger(**trigger_params))
            elif trigger_type == "loss_spike":
                # Need to create slice function
                slice_type = trigger_params.pop("slice_type", "random")
                if slice_type == "random":
                    from .triggers import create_random_slice_fn
                    slice_fn = create_random_slice_fn(trigger_params.pop("probability", 0.1))
                elif slice_type == "length":
                    from .triggers import create_length_slice_fn
                    min_len = trigger_params.pop("min_length", 100)
                    max_len = trigger_params.pop("max_length", None)
                    slice_fn = create_length_slice_fn(min_len, max_len)
                else:
                    raise ValueError(f"Unknown slice_type: {slice_type}")
                
                triggers.append(LossSpikeTrigger(slice_fn=slice_fn, **trigger_params))
            else:
                raise ValueError(f"Unknown trigger type: {trigger_type}")
        
        return triggers
    
    def _create_growth_policy(self) -> Callable[[int], List[int]]:
        """Create a function that selects which layers to grow."""
        def select_layers(num_layers: int) -> List[int]:
            if self.layer_selection == "uniform":
                # Select random layers
                return random.sample(range(num_layers), min(self.layers_per_event, num_layers))
            elif self.layer_selection == "top_k":
                # Select top layers (closer to output)
                start_idx = max(0, num_layers - self.layers_per_event)
                return list(range(start_idx, num_layers))
            elif self.layer_selection == "bottom_k":
                # Select bottom layers (closer to input)
                return list(range(min(self.layers_per_event, num_layers)))
            elif self.layer_selection == "middle":
                # Select middle layers
                middle = num_layers // 2
                half_range = self.layers_per_event // 2
                start = max(0, middle - half_range)
                end = min(num_layers, middle + half_range + 1)
                return list(range(start, end))
            else:
                raise ValueError(f"Unknown layer_selection: {self.layer_selection}")
        
        return select_layers
    
    def should_grow(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if the model should grow based on current metrics.
        
        Args:
            metrics: Dictionary containing training metrics like:
                - val_loss: Validation loss
                - grad_norm: Gradient norm
                - perplexity: Model perplexity
                - input_ids: Input tokens (for slice triggers)
                - losses: Per-sample losses (for slice triggers)
                
        Returns:
            True if growth should occur
        """
        if not self.enabled:
            return False
        
        if len(self.growth_events) >= self.max_events:
            return False
        
        if self.steps_since_last_growth < self.cooldown_steps:
            return False
        
        # Check all triggers
        for trigger in self.triggers:
            if trigger.should_trigger(**metrics):
                return True
        
        return False
    
    def grow_model(self, reason: str = "trigger_fired") -> bool:
        """
        Grow the model by expanding selected layers.
        
        Args:
            reason: Reason for growth (for logging)
            
        Returns:
            True if growth occurred, False otherwise
        """
        if not self.enabled or len(self.growth_events) >= self.max_events:
            return False
        
        # Select layers to grow
        num_layers = len(self.model.layers)
        selected_layers = self.growth_policy(num_layers)
        
        if not selected_layers:
            return False
        
        # Record growth event
        growth_event = {
            "timestamp": datetime.now().isoformat(),
            "step": self.total_steps,
            "reason": reason,
            "layer_indices": selected_layers,
            "add_hidden": self.add_hidden,
            "old_param_count": self.model.param_count(),
            "old_ffn_dims": [self.model.layers[i].ffn_dim for i in selected_layers],
        }
        
        # Perform growth
        self.model.grow(selected_layers, self.add_hidden)
        
        # Update growth event with new information
        growth_event.update({
            "new_param_count": self.model.param_count(),
            "new_ffn_dims": [self.model.layers[i].ffn_dim for i in selected_layers],
            "param_increase": self.model.param_count() - growth_event["old_param_count"],
        })
        
        self.growth_events.append(growth_event)
        
        # Reset triggers after growth
        for trigger in self.triggers:
            trigger.reset()
        
        self.steps_since_last_growth = 0
        
        print(f"Growth event {len(self.growth_events)}: {growth_event}")
        
        return True
    
    def step(self, metrics: Dict[str, Any]) -> bool:
        """
        Process one training step and check for growth.
        
        Args:
            metrics: Training metrics for this step
            
        Returns:
            True if growth occurred
        """
        self.total_steps += 1
        self.steps_since_last_growth += 1
        
        if self.should_grow(metrics):
            return self.grow_model()
        
        return False
    
    def add_parameters_to_optimizer(self, new_parameters: List[nn.Parameter]) -> None:
        """
        Add new parameters to the optimizer after growth.
        
        Args:
            new_parameters: List of new parameters to add
        """
        if not new_parameters:
            return
        
        # Get default parameter group settings
        if len(self.optimizer.param_groups) > 0:
            default_group = self.optimizer.param_groups[0].copy()
            default_group['params'] = new_parameters
        else:
            # Fallback defaults for AdamW
            default_group = {
                'params': new_parameters,
                'lr': 1e-4,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0.01,
            }
        
        # Add new parameter group
        self.optimizer.add_param_group(default_group)
        
        # Initialize optimizer state for new parameters
        for param in new_parameters:
            if param not in self.optimizer.state:
                self.optimizer.state[param] = {}
                
                # Initialize state based on optimizer type
                if isinstance(self.optimizer, torch.optim.AdamW):
                    self.optimizer.state[param]['step'] = 0
                    self.optimizer.state[param]['exp_avg'] = torch.zeros_like(param)
                    self.optimizer.state[param]['exp_avg_sq'] = torch.zeros_like(param)
                elif isinstance(self.optimizer, torch.optim.SGD):
                    if 'momentum' in self.optimizer.defaults:
                        self.optimizer.state[param]['momentum_buffer'] = torch.zeros_like(param)
        
        print(f"Added {len(new_parameters)} new parameters to optimizer")
    
    def get_growth_summary(self) -> Dict[str, Any]:
        """Get a summary of all growth events."""
        if not self.growth_events:
            return {"num_events": 0}
        
        total_param_increase = sum(event["param_increase"] for event in self.growth_events)
        initial_params = self.growth_events[0]["old_param_count"]
        final_params = self.growth_events[-1]["new_param_count"]
        
        return {
            "num_events": len(self.growth_events),
            "total_param_increase": total_param_increase,
            "initial_params": initial_params,
            "final_params": final_params,
            "growth_ratio": final_params / initial_params if initial_params > 0 else 1.0,
            "events": self.growth_events,
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save growth manager state for checkpointing."""
        return {
            "growth_events": self.growth_events,
            "steps_since_last_growth": self.steps_since_last_growth,
            "total_steps": self.total_steps,
            "config": self.config,
            "trigger_states": [trigger.get_state() for trigger in self.triggers],
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load growth manager state from checkpoint."""
        self.growth_events = state["growth_events"]
        self.steps_since_last_growth = state["steps_since_last_growth"]
        self.total_steps = state["total_steps"]
        
        # Load trigger states
        if "trigger_states" in state:
            trigger_states = state["trigger_states"]
            for trigger, trigger_state in zip(self.triggers, trigger_states):
                trigger.load_state(trigger_state)
    
    def plot_growth_timeline(self, save_path: Optional[str] = None) -> None:
        """
        Plot the growth timeline showing parameter count over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.growth_events:
                print("No growth events to plot")
                return
            
            # Extract data for plotting
            steps = [0] + [event["step"] for event in self.growth_events]
            param_counts = [self.growth_events[0]["old_param_count"]] + \
                          [event["new_param_count"] for event in self.growth_events]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.step(steps, param_counts, where='post', linewidth=2, marker='o')
            
            # Mark growth events
            for i, event in enumerate(self.growth_events):
                plt.axvline(x=event["step"], color='red', linestyle='--', alpha=0.7)
                plt.text(event["step"], param_counts[i+1], f'Event {i+1}', 
                        rotation=90, verticalalignment='bottom')
            
            plt.xlabel('Training Step')
            plt.ylabel('Parameter Count')
            plt.title('Model Growth Timeline')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Growth timeline saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting")
    
    def __repr__(self) -> str:
        return (f"GrowthManager(enabled={self.enabled}, "
                f"events={len(self.growth_events)}/{self.max_events}, "
                f"triggers={len(self.triggers)})")
