"""
Synthetic Data Generation for Arbor Training Examples

Simple synthetic dataset generator for demonstration purposes.
"""

import torch
import random
from torch.utils.data import Dataset
from typing import Optional

class SyntheticDataset(Dataset):
    """
    Simple synthetic dataset for language modeling tasks.
    
    Generates random sequences of tokens for training demonstration.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        seq_length: int = 1024,
        num_sequences: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            vocab_size: Size of vocabulary
            seq_length: Length of each sequence
            num_sequences: Number of sequences to generate
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_sequences = num_sequences
        
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Pre-generate data for consistency
        self.data = self._generate_sequences()
    
    def _generate_sequences(self) -> torch.Tensor:
        """Generate random sequences."""
        # Generate random token sequences
        sequences = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.num_sequences, self.seq_length),
            dtype=torch.long
        )
        
        return sequences
    
    def __len__(self) -> int:
        """Return dataset length."""
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item by index."""
        return self.data[idx]

class SyntheticClassificationDataset(Dataset):
    """
    Synthetic dataset for classification tasks.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 512,
        num_classes: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize classification dataset.
        
        Args:
            num_samples: Number of samples
            input_dim: Input feature dimension
            num_classes: Number of classes
            seed: Random seed
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate data
        self.features = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.labels[idx]

class SyntheticMultimodalDataset(Dataset):
    """
    Synthetic multimodal dataset with text and image-like features.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        text_seq_length: int = 256,
        vocab_size: int = 10000,
        image_channels: int = 3,
        image_size: int = 224,
        seed: Optional[int] = None
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            num_samples: Number of samples
            text_seq_length: Length of text sequences
            vocab_size: Text vocabulary size
            image_channels: Number of image channels
            image_size: Image height/width
            seed: Random seed
        """
        self.num_samples = num_samples
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate text data
        self.text_data = torch.randint(
            0, vocab_size, 
            (num_samples, text_seq_length)
        )
        
        # Generate image-like data
        self.image_data = torch.randn(
            num_samples, 
            image_channels, 
            image_size, 
            image_size
        )
        
        # Generate labels
        self.labels = torch.randint(0, 2, (num_samples,))  # Binary classification
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        return {
            'text': self.text_data[idx],
            'image': self.image_data[idx],
            'label': self.labels[idx]
        }
