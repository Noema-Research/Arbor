"""
Unit tests for Arbor-o1 expandable layers.

Tests the core expandable components including:
- ExpandableFFN layer functionality
- Parameter expansion mechanisms
- Weight copying and initialization
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbor.modeling.layers import ExpandableFFN, expand_linear


class TestExpandLinear:
    """Test the expand_linear utility function."""
    
    def test_expand_linear_basic(self):
        """Test basic linear layer expansion."""
        # Create a simple linear layer
        original = nn.Linear(10, 5)
        
        # Set specific weights for testing
        with torch.no_grad():
            original.weight.fill_(1.0)
            original.bias.fill_(0.5)
        
        # Expand to larger hidden size
        expanded = expand_linear(original, new_out_features=8)
        
        # Check dimensions
        assert expanded.in_features == 10
        assert expanded.out_features == 8
        
        # Check that original weights are preserved
        assert torch.allclose(expanded.weight[:5, :], original.weight)
        assert torch.allclose(expanded.bias[:5], original.bias)
        
        # Check that new weights are properly initialized (not all zeros/ones)
        new_weights = expanded.weight[5:, :]
        new_bias = expanded.bias[5:]
        
        assert not torch.allclose(new_weights, torch.zeros_like(new_weights))
        assert not torch.allclose(new_bias, torch.zeros_like(new_bias))
    
    def test_expand_linear_no_bias(self):
        """Test expansion of linear layer without bias."""
        original = nn.Linear(10, 5, bias=False)
        
        with torch.no_grad():
            original.weight.fill_(1.0)
        
        expanded = expand_linear(original, new_out_features=8)
        
        assert expanded.bias is None
        assert expanded.in_features == 10
        assert expanded.out_features == 8
        assert torch.allclose(expanded.weight[:5, :], original.weight)
    
    def test_expand_linear_same_size(self):
        """Test that expanding to same size returns equivalent layer."""
        original = nn.Linear(10, 5)
        
        with torch.no_grad():
            original.weight.fill_(1.0)
            original.bias.fill_(0.5)
        
        expanded = expand_linear(original, new_out_features=5)
        
        assert expanded.in_features == 10
        assert expanded.out_features == 5
        assert torch.allclose(expanded.weight, original.weight)
        assert torch.allclose(expanded.bias, original.bias)
    
    def test_expand_linear_smaller_size(self):
        """Test that expanding to smaller size raises error."""
        original = nn.Linear(10, 5)
        
        with pytest.raises(ValueError, match="Cannot shrink"):
            expand_linear(original, new_out_features=3)
    
    def test_expand_linear_preserves_gradients(self):
        """Test that expansion preserves gradient requirements."""
        original = nn.Linear(10, 5)
        original.weight.requires_grad_(True)
        original.bias.requires_grad_(True)
        
        expanded = expand_linear(original, new_out_features=8)
        
        assert expanded.weight.requires_grad
        assert expanded.bias.requires_grad


class TestExpandableFFN:
    """Test the ExpandableFFN layer."""
    
    def test_expandable_ffn_init(self):
        """Test ExpandableFFN initialization."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        assert ffn.d_model == 512
        assert ffn.d_ff == 2048
        assert ffn.fc1.in_features == 512
        assert ffn.fc1.out_features == 2048
        assert ffn.fc2.in_features == 2048
        assert ffn.fc2.out_features == 512
    
    def test_expandable_ffn_forward(self):
        """Test ExpandableFFN forward pass."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        # Create test input
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 512)
        
        # Forward pass
        output = ffn(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, 512)
        
        # Check that output is not all zeros (network is working)
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_expandable_ffn_grow(self):
        """Test ExpandableFFN growth functionality."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        # Store original weights
        original_fc1_weight = ffn.fc1.weight.clone()
        original_fc1_bias = ffn.fc1.bias.clone()
        original_fc2_weight = ffn.fc2.weight.clone()
        original_fc2_bias = ffn.fc2.bias.clone()
        
        # Test input
        x = torch.randn(2, 10, 512)
        original_output = ffn(x)
        
        # Grow the layer
        new_d_ff = 3072
        ffn.grow(new_d_ff)
        
        # Check new dimensions
        assert ffn.d_ff == new_d_ff
        assert ffn.fc1.out_features == new_d_ff
        assert ffn.fc2.in_features == new_d_ff
        
        # Check that original weights are preserved
        assert torch.allclose(ffn.fc1.weight[:2048, :], original_fc1_weight)
        assert torch.allclose(ffn.fc1.bias[:2048], original_fc1_bias)
        assert torch.allclose(ffn.fc2.weight[:, :2048], original_fc2_weight)
        assert torch.allclose(ffn.fc2.bias, original_fc2_bias)
        
        # Check that output is still reasonable (but may differ due to new params)
        new_output = ffn(x)
        assert new_output.shape == original_output.shape
        assert not torch.allclose(new_output, torch.zeros_like(new_output))
    
    def test_expandable_ffn_grow_same_size(self):
        """Test that growing to same size doesn't change anything."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        original_fc1_weight = ffn.fc1.weight.clone()
        original_fc1_bias = ffn.fc1.bias.clone()
        original_fc2_weight = ffn.fc2.weight.clone()
        original_fc2_bias = ffn.fc2.bias.clone()
        
        # Grow to same size
        ffn.grow(2048)
        
        # Check dimensions unchanged
        assert ffn.d_ff == 2048
        assert ffn.fc1.out_features == 2048
        assert ffn.fc2.in_features == 2048
        
        # Check weights unchanged
        assert torch.allclose(ffn.fc1.weight, original_fc1_weight)
        assert torch.allclose(ffn.fc1.bias, original_fc1_bias)
        assert torch.allclose(ffn.fc2.weight, original_fc2_weight)
        assert torch.allclose(ffn.fc2.bias, original_fc2_bias)
    
    def test_expandable_ffn_grow_smaller_fails(self):
        """Test that growing to smaller size fails."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        with pytest.raises(ValueError, match="Cannot shrink"):
            ffn.grow(1024)
    
    def test_expandable_ffn_parameter_count(self):
        """Test parameter counting."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        expected_params = (
            512 * 2048 + 2048 +  # fc1: weight + bias
            2048 * 512 + 512     # fc2: weight + bias
        )
        
        assert ffn.param_count() == expected_params
        
        # After growth
        ffn.grow(3072)
        expected_params_grown = (
            512 * 3072 + 3072 +  # fc1: weight + bias
            3072 * 512 + 512     # fc2: weight + bias
        )
        
        assert ffn.param_count() == expected_params_grown
    
    def test_expandable_ffn_device_consistency(self):
        """Test that growth preserves device placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        ffn = ffn.cuda()
        
        # Check all parameters are on CUDA
        assert all(p.device.type == 'cuda' for p in ffn.parameters())
        
        # Grow and check device consistency
        ffn.grow(3072)
        
        assert all(p.device.type == 'cuda' for p in ffn.parameters())
    
    def test_expandable_ffn_dtype_consistency(self):
        """Test that growth preserves dtype."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        ffn = ffn.half()  # Convert to half precision
        
        # Check all parameters are half precision
        assert all(p.dtype == torch.float16 for p in ffn.parameters())
        
        # Grow and check dtype consistency
        ffn.grow(3072)
        
        assert all(p.dtype == torch.float16 for p in ffn.parameters())
    
    def test_expandable_ffn_backward_compatibility(self):
        """Test that grown layer can still process original inputs correctly."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        # Test with various input shapes
        inputs = [
            torch.randn(1, 1, 512),      # Single token
            torch.randn(2, 10, 512),     # Small batch
            torch.randn(8, 100, 512),    # Larger batch
        ]
        
        for x in inputs:
            original_output = ffn(x)
            
            # Grow and test again
            ffn.grow(3072)
            new_output = ffn(x)
            
            # Check shape consistency
            assert new_output.shape == original_output.shape
            assert not torch.isnan(new_output).any()
            assert not torch.isinf(new_output).any()


class TestLayerIntegration:
    """Test integration between different layer components."""
    
    def test_multiple_ffn_growth(self):
        """Test growing multiple FFN layers."""
        ffn1 = ExpandableFFN(d_model=512, d_ff=2048)
        ffn2 = ExpandableFFN(d_model=512, d_ff=2048)
        
        # Test input
        x = torch.randn(2, 10, 512)
        
        # Forward through both layers
        intermediate = ffn1(x)
        original_output = ffn2(intermediate)
        
        # Grow both layers
        ffn1.grow(3072)
        ffn2.grow(3072)
        
        # Forward through grown layers
        intermediate_grown = ffn1(x)
        new_output = ffn2(intermediate_grown)
        
        # Check that pipeline still works
        assert new_output.shape == original_output.shape
        assert not torch.isnan(new_output).any()
    
    def test_gradients_after_growth(self):
        """Test that gradients flow correctly after growth."""
        ffn = ExpandableFFN(d_model=512, d_ff=2048)
        
        # Enable gradients
        ffn.train()
        
        # Test input with gradients
        x = torch.randn(2, 10, 512, requires_grad=True)
        
        # Forward and backward before growth
        output1 = ffn(x)
        loss1 = output1.sum()
        loss1.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert all(p.grad is not None for p in ffn.parameters())
        
        # Clear gradients
        ffn.zero_grad()
        x.grad = None
        
        # Grow the layer
        ffn.grow(3072)
        
        # Forward and backward after growth
        output2 = ffn(x)
        loss2 = output2.sum()
        loss2.backward()
        
        # Check gradients still flow correctly
        assert x.grad is not None
        assert all(p.grad is not None for p in ffn.parameters())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
