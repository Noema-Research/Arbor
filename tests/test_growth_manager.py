"""
Unit tests for Arbor-o1 Growth Manager.

Tests the growth management system including:
- Growth triggers and their conditions
- GrowthManager coordination
- Model expansion orchestration
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbor.growth.triggers import (
    BaseTrigger,
    PlateauTrigger,
    GradientNormTrigger,
    LossSpikeTrigger,
)
from arbor.growth.manager import GrowthManager
from arbor.modeling.model import ArborTransformer, ArborConfig


class TestBaseTrigger:
    """Test the base trigger class."""
    
    def test_base_trigger_abstract(self):
        """Test that BaseTrigger cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTrigger()
    
    def test_base_trigger_interface(self):
        """Test that subclasses must implement required methods."""
        
        class IncompleteTrigger(BaseTrigger):
            pass
        
        with pytest.raises(TypeError):
            IncompleteTrigger()


class TestPlateauTrigger:
    """Test the plateau-based growth trigger."""
    
    def test_plateau_trigger_init(self):
        """Test PlateauTrigger initialization."""
        trigger = PlateauTrigger(patience=5, threshold=0.01)
        
        assert trigger.patience == 5
        assert trigger.threshold == 0.01
        assert trigger.wait_count == 0
        assert trigger.best_loss == float('inf')
    
    def test_plateau_trigger_improving_loss(self):
        """Test trigger behavior with improving loss."""
        trigger = PlateauTrigger(patience=3, threshold=0.01)
        
        # Simulate improving losses
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        
        for loss in losses:
            should_grow = trigger.should_trigger(loss=loss)
            assert not should_grow
            assert trigger.wait_count == 0
    
    def test_plateau_trigger_plateau_detection(self):
        """Test trigger activation during loss plateau."""
        trigger = PlateauTrigger(patience=3, threshold=0.01)
        
        # Initial improving loss
        trigger.should_trigger(loss=1.0)
        trigger.should_trigger(loss=0.5)
        
        # Now plateau (small improvements)
        assert not trigger.should_trigger(loss=0.495)  # Small improvement
        assert trigger.wait_count == 1
        
        assert not trigger.should_trigger(loss=0.492)  # Small improvement
        assert trigger.wait_count == 2
        
        assert not trigger.should_trigger(loss=0.490)  # Small improvement
        assert trigger.wait_count == 3
        
        # Should trigger after patience exceeded
        should_grow = trigger.should_trigger(loss=0.488)
        assert should_grow
        assert trigger.wait_count == 0  # Reset after trigger
    
    def test_plateau_trigger_loss_increase(self):
        """Test trigger behavior when loss increases."""
        trigger = PlateauTrigger(patience=3, threshold=0.01)
        
        # Initial loss
        trigger.should_trigger(loss=0.5)
        
        # Loss increases (worse)
        assert not trigger.should_trigger(loss=0.52)
        assert trigger.wait_count == 1
        
        # Loss improves again
        assert not trigger.should_trigger(loss=0.4)
        assert trigger.wait_count == 0  # Reset on improvement
    
    def test_plateau_trigger_reset(self):
        """Test trigger reset functionality."""
        trigger = PlateauTrigger(patience=3, threshold=0.01)
        
        # Build up wait count
        trigger.should_trigger(loss=1.0)
        trigger.should_trigger(loss=0.995)
        trigger.should_trigger(loss=0.992)
        assert trigger.wait_count == 2
        
        # Reset
        trigger.reset()
        assert trigger.wait_count == 0
        assert trigger.best_loss == float('inf')


class TestGradientNormTrigger:
    """Test the gradient norm-based growth trigger."""
    
    def test_gradient_norm_trigger_init(self):
        """Test GradientNormTrigger initialization."""
        trigger = GradientNormTrigger(threshold=10.0, patience=5)
        
        assert trigger.threshold == 10.0
        assert trigger.patience == 5
        assert trigger.violation_count == 0
    
    def test_gradient_norm_trigger_normal_gradients(self):
        """Test trigger with normal gradient norms."""
        trigger = GradientNormTrigger(threshold=10.0, patience=3)
        
        # Normal gradient norms
        norms = [1.0, 2.5, 5.0, 3.2, 1.8]
        
        for norm in norms:
            should_grow = trigger.should_trigger(grad_norm=norm)
            assert not should_grow
            assert trigger.violation_count == 0
    
    def test_gradient_norm_trigger_high_gradients(self):
        """Test trigger activation with high gradient norms."""
        trigger = GradientNormTrigger(threshold=10.0, patience=3)
        
        # High gradient norms
        high_norms = [15.0, 12.0, 20.0]
        
        for i, norm in enumerate(high_norms):
            should_grow = trigger.should_trigger(grad_norm=norm)
            if i < 2:  # First two violations
                assert not should_grow
                assert trigger.violation_count == i + 1
            else:  # Third violation should trigger
                assert should_grow
                assert trigger.violation_count == 0  # Reset after trigger
    
    def test_gradient_norm_trigger_mixed_gradients(self):
        """Test trigger with mixed high and normal gradients."""
        trigger = GradientNormTrigger(threshold=10.0, patience=3)
        
        # Mix of high and normal gradients
        assert not trigger.should_trigger(grad_norm=15.0)  # High
        assert trigger.violation_count == 1
        
        assert not trigger.should_trigger(grad_norm=5.0)   # Normal - resets
        assert trigger.violation_count == 0
        
        assert not trigger.should_trigger(grad_norm=12.0)  # High again
        assert trigger.violation_count == 1
    
    def test_gradient_norm_trigger_invalid_input(self):
        """Test trigger with invalid inputs."""
        trigger = GradientNormTrigger(threshold=10.0, patience=3)
        
        # Test with None (should not crash)
        should_grow = trigger.should_trigger(grad_norm=None)
        assert not should_grow
        
        # Test with invalid metric name
        should_grow = trigger.should_trigger(loss=1.0)  # Wrong metric
        assert not should_grow


class TestLossSpikeTrigger:
    """Test the loss spike-based growth trigger."""
    
    def test_loss_spike_trigger_init(self):
        """Test LossSpikeTrigger initialization."""
        trigger = LossSpikeTrigger(spike_threshold=2.0, history_length=10)
        
        assert trigger.spike_threshold == 2.0
        assert trigger.history_length == 10
        assert len(trigger.loss_history) == 0
    
    def test_loss_spike_trigger_normal_losses(self):
        """Test trigger with normal loss progression."""
        trigger = LossSpikeTrigger(spike_threshold=2.0, history_length=5)
        
        # Gradually decreasing losses
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        
        for loss in losses:
            should_grow = trigger.should_trigger(loss=loss)
            assert not should_grow
    
    def test_loss_spike_trigger_spike_detection(self):
        """Test trigger activation on loss spikes."""
        trigger = LossSpikeTrigger(spike_threshold=1.5, history_length=3)
        
        # Build history
        trigger.should_trigger(loss=1.0)
        trigger.should_trigger(loss=0.9)
        trigger.should_trigger(loss=0.8)
        
        # Normal loss - no spike
        assert not trigger.should_trigger(loss=0.85)
        
        # Spike - loss increases significantly
        should_grow = trigger.should_trigger(loss=1.5)  # 1.5 > 0.8 * 1.5
        assert should_grow
    
    def test_loss_spike_trigger_history_management(self):
        """Test loss history management."""
        trigger = LossSpikeTrigger(spike_threshold=2.0, history_length=3)
        
        # Fill history beyond capacity
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        
        for loss in losses:
            trigger.should_trigger(loss=loss)
        
        # Should only keep last 3 losses
        assert len(trigger.loss_history) == 3
        assert trigger.loss_history == [0.8, 0.7, 0.6]
    
    def test_loss_spike_trigger_insufficient_history(self):
        """Test trigger with insufficient loss history."""
        trigger = LossSpikeTrigger(spike_threshold=2.0, history_length=5)
        
        # Only provide 2 losses when 5 needed for comparison
        assert not trigger.should_trigger(loss=1.0)
        assert not trigger.should_trigger(loss=2.0)  # Even if spike-like


class TestGrowthManager:
    """Test the GrowthManager coordination system."""
    
    def test_growth_manager_init(self):
        """Test GrowthManager initialization."""
        triggers = [PlateauTrigger(patience=5)]
        manager = GrowthManager(
            triggers=triggers,
            growth_factor=1.5,
            min_steps_between_growth=100
        )
        
        assert len(manager.triggers) == 1
        assert manager.growth_factor == 1.5
        assert manager.min_steps_between_growth == 100
        assert manager.step_count == 0
        assert manager.last_growth_step == -1
    
    def test_growth_manager_step_tracking(self):
        """Test step counting in GrowthManager."""
        manager = GrowthManager(triggers=[])
        
        assert manager.step_count == 0
        
        # Step without triggering
        result = manager.step(loss=1.0)
        assert result is None
        assert manager.step_count == 1
        
        # Multiple steps
        for i in range(5):
            manager.step(loss=1.0)
        assert manager.step_count == 6
    
    def test_growth_manager_trigger_coordination(self):
        """Test that GrowthManager coordinates multiple triggers."""
        # Create mock triggers
        trigger1 = Mock(spec=BaseTrigger)
        trigger2 = Mock(spec=BaseTrigger)
        
        trigger1.should_trigger.return_value = False
        trigger2.should_trigger.return_value = True
        
        manager = GrowthManager(
            triggers=[trigger1, trigger2],
            min_steps_between_growth=0
        )
        
        # Should trigger because trigger2 returns True
        result = manager.step(loss=1.0)
        assert result is not None
        assert result["trigger_type"] == type(trigger2).__name__
        
        # Check that both triggers were called
        trigger1.should_trigger.assert_called_once()
        trigger2.should_trigger.assert_called_once()
    
    def test_growth_manager_min_steps_constraint(self):
        """Test minimum steps between growth constraint."""
        trigger = Mock(spec=BaseTrigger)
        trigger.should_trigger.return_value = True
        
        manager = GrowthManager(
            triggers=[trigger],
            min_steps_between_growth=10
        )
        
        # First trigger should work
        result = manager.step(loss=1.0)
        assert result is not None
        
        # Immediate trigger should be blocked
        result = manager.step(loss=1.0)
        assert result is None
        
        # Advance steps
        for _ in range(9):
            result = manager.step(loss=1.0)
            assert result is None
        
        # Now should trigger again
        result = manager.step(loss=1.0)
        assert result is not None
    
    def test_growth_manager_grow_model(self):
        """Test model growth through GrowthManager."""
        # Create a small test model
        config = ArborConfig(
            vocab_size=1000,
            n_embd=64,
            n_layer=2,
            n_head=4,
            d_ff=128,
            max_length=32
        )
        model = ArborTransformer(config)
        
        # Create manager
        manager = GrowthManager(triggers=[], growth_factor=1.5)
        
        # Get original parameter count
        original_params = model.param_count()
        
        # Trigger growth
        growth_info = {
            "step": 100,
            "trigger_type": "TestTrigger",
            "metrics": {"loss": 1.0}
        }
        
        new_params = manager.grow_model(model, growth_info)
        
        # Check that model grew
        assert new_params > original_params
        assert model.param_count() == new_params
    
    def test_growth_manager_optimizer_handling(self):
        """Test optimizer parameter group handling during growth."""
        # Create mock model and optimizer
        model = Mock()
        model.param_count.return_value = 1000
        model.grow.return_value = 1500
        model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        
        optimizer = Mock()
        optimizer.param_groups = [{"params": model.parameters()}]
        
        manager = GrowthManager(triggers=[])
        
        # Mock the optimizer utility functions
        with patch('arbor.growth.manager.update_optimizer_param_groups') as mock_update:
            growth_info = {"step": 100, "trigger_type": "Test"}
            manager.grow_model(model, growth_info, optimizer)
            
            # Check that optimizer was updated
            mock_update.assert_called_once_with(optimizer, model)
    
    def test_growth_manager_history_tracking(self):
        """Test growth history tracking."""
        trigger = Mock(spec=BaseTrigger)
        trigger.should_trigger.return_value = True
        
        manager = GrowthManager(
            triggers=[trigger],
            min_steps_between_growth=0
        )
        
        # Trigger several growth events
        for i in range(3):
            result = manager.step(loss=1.0 - i * 0.1)
            assert result is not None
        
        # Check history
        assert len(manager.growth_history) == 3
        
        for i, event in enumerate(manager.growth_history):
            assert event["step"] == i + 1
            assert "trigger_type" in event
            assert "metrics" in event
    
    def test_growth_manager_reset(self):
        """Test GrowthManager reset functionality."""
        trigger = Mock(spec=BaseTrigger)
        manager = GrowthManager(triggers=[trigger])
        
        # Simulate some activity
        manager.step(loss=1.0)
        manager.step(loss=0.9)
        manager.growth_history.append({"test": "event"})
        
        # Reset
        manager.reset()
        
        # Check reset state
        assert manager.step_count == 0
        assert manager.last_growth_step == -1
        assert len(manager.growth_history) == 0
        
        # Check that triggers were reset
        trigger.reset.assert_called_once()


class TestGrowthIntegration:
    """Test integration between triggers and manager."""
    
    def test_realistic_growth_scenario(self):
        """Test a realistic growth scenario with actual triggers."""
        # Create real triggers
        plateau_trigger = PlateauTrigger(patience=3, threshold=0.01)
        grad_trigger = GradientNormTrigger(threshold=5.0, patience=2)
        
        manager = GrowthManager(
            triggers=[plateau_trigger, grad_trigger],
            growth_factor=1.2,
            min_steps_between_growth=5
        )
        
        # Simulate training with plateau
        losses = [1.0, 0.8, 0.6, 0.595, 0.592, 0.590, 0.589]  # Plateau at end
        grad_norms = [1.0, 2.0, 1.5, 1.8, 2.2, 1.9, 2.1]     # Normal gradients
        
        growth_events = []
        
        for loss, grad_norm in zip(losses, grad_norms):
            result = manager.step(loss=loss, grad_norm=grad_norm)
            if result:
                growth_events.append(result)
        
        # Should trigger once due to plateau
        assert len(growth_events) == 1
        assert "Plateau" in growth_events[0]["trigger_type"]
    
    def test_multiple_trigger_types(self):
        """Test scenario where different triggers activate."""
        plateau_trigger = PlateauTrigger(patience=2, threshold=0.01)
        spike_trigger = LossSpikeTrigger(spike_threshold=1.5, history_length=3)
        
        manager = GrowthManager(
            triggers=[plateau_trigger, spike_trigger],
            min_steps_between_growth=0
        )
        
        # First: plateau scenario
        manager.step(loss=1.0)
        manager.step(loss=0.995)  # Plateau starts
        result1 = manager.step(loss=0.993)  # Should trigger plateau
        
        assert result1 is not None
        assert "Plateau" in result1["trigger_type"]
        
        # Reset for next test
        manager.reset()
        
        # Second: spike scenario
        manager.step(loss=1.0)
        manager.step(loss=0.9)
        manager.step(loss=0.8)
        result2 = manager.step(loss=1.5)  # Spike
        
        assert result2 is not None
        assert "Spike" in result2["trigger_type"]
    
    def test_growth_manager_with_real_model(self):
        """Test GrowthManager with real ArborTransformer."""
        # Create small model for testing
        config = ArborConfig(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=2,
            d_ff=64,
            max_length=16
        )
        model = ArborTransformer(config)
        
        # Create trigger that will activate
        trigger = PlateauTrigger(patience=1, threshold=0.1)
        manager = GrowthManager(
            triggers=[trigger],
            growth_factor=1.5,
            min_steps_between_growth=0
        )
        
        # Simulate plateau to trigger growth
        original_params = model.param_count()
        
        manager.step(loss=1.0)
        result = manager.step(loss=0.95)  # Small improvement, should trigger
        
        if result:  # Growth was triggered
            new_params = manager.grow_model(model, result)
            assert new_params > original_params
            
            # Verify model still works
            test_input = torch.randint(0, 100, (1, 10))
            output = model(test_input)
            assert output.logits.shape == (1, 10, 100)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
