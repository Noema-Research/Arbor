"""
Dataset classes and data loading utilities.
"""

from typing import List, Dict, Any, Optional, Union, Iterator
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import random
import numpy as np
import json
import os
from pathlib import Path

from .tokenize import ArborTokenizer, collate_fn


class SyntheticTextDataset(Dataset):
    """
    Synthetic text dataset for testing and development.
    
    Generates random sequences using a vocabulary to simulate
    language modeling data without requiring large text corpora.
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        seq_length: int = 512,
        num_sequences: int = 10000,
        tokenizer: Optional[ArborTokenizer] = None,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_sequences = num_sequences
        self.tokenizer = tokenizer
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Create vocabulary if tokenizer not provided
        if tokenizer is None:
            from .tokenize import create_synthetic_vocabulary, create_tokenizer_from_vocab
            vocab = create_synthetic_vocabulary(vocab_size)
            self.tokenizer = create_tokenizer_from_vocab(vocab)
        else:
            self.tokenizer = tokenizer
            self.vocab_size = len(tokenizer)
        
        # Generate sequences
        self.sequences = self._generate_sequences()
    
    def _generate_sequences(self) -> List[torch.Tensor]:
        """Generate synthetic text sequences."""
        sequences = []
        
        # Reserve special token IDs
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id or 1
        bos_id = self.tokenizer.bos_token_id or 2
        
        # Usable vocabulary range (excluding special tokens)
        vocab_start = 4
        vocab_end = min(self.vocab_size, len(self.tokenizer))
        
        for _ in range(self.num_sequences):
            # Random sequence length (with some variation)
            length = random.randint(
                max(10, self.seq_length // 2), 
                self.seq_length
            )
            
            # Generate sequence with some structure
            sequence = []
            
            # Add BOS token
            if bos_id is not None:
                sequence.append(bos_id)
            
            # Generate content with simple patterns
            for i in range(length - 2):  # Reserve space for BOS/EOS
                if i == 0 or random.random() < 0.7:
                    # Normal token
                    token_id = random.randint(vocab_start, vocab_end - 1)
                else:
                    # Sometimes repeat previous token (simple pattern)
                    token_id = sequence[-1]
                
                sequence.append(token_id)
            
            # Add EOS token
            if eos_id is not None:
                sequence.append(eos_id)
            
            # Pad to fixed length
            while len(sequence) < self.seq_length:
                sequence.append(pad_id)
            
            sequences.append(torch.tensor(sequence[:self.seq_length], dtype=torch.long))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Create attention mask (1 for non-padding tokens)
        attention_mask = (sequence != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": sequence.unsqueeze(0),  # Add batch dimension
            "attention_mask": attention_mask.unsqueeze(0),
        }


class TextFileDataset(Dataset):
    """
    Dataset for loading text from files.
    
    Loads text files and tokenizes them for language modeling.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: ArborTokenizer,
        max_length: int = 512,
        stride: int = 256,
        min_length: int = 10,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.min_length = min_length
        
        # Load and tokenize all texts
        self.examples = self._load_and_tokenize()
    
    def _load_and_tokenize(self) -> List[Dict[str, torch.Tensor]]:
        """Load text files and create tokenized examples."""
        examples = []
        
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Tokenize the entire text
                encoded = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    max_length=None,
                    padding=False,
                    truncation=False,
                )
                
                input_ids = encoded["input_ids"][0]  # Remove batch dimension
                
                # Split into overlapping chunks
                for i in range(0, len(input_ids), self.stride):
                    chunk = input_ids[i:i + self.max_length]
                    
                    if len(chunk) >= self.min_length:
                        # Pad if necessary
                        if len(chunk) < self.max_length:
                            padding = [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
                            chunk = torch.cat([chunk, torch.tensor(padding)])
                        
                        # Create attention mask
                        attention_mask = (chunk != self.tokenizer.pad_token_id).long()
                        
                        examples.append({
                            "input_ids": chunk,
                            "attention_mask": attention_mask,
                        })
                
            except Exception as e:
                print(f"Warning: Could not load file {file_path}: {e}")
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large text corpora.
    
    Loads and tokenizes text on-the-fly to handle datasets
    that don't fit in memory.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: ArborTokenizer,
        max_length: int = 512,
        buffer_size: int = 10000,
        shuffle: bool = True,
    ):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle = shuffle
    
    def _read_files(self) -> Iterator[str]:
        """Read lines from files."""
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            yield line
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized examples."""
        buffer = []
        
        for text in self._read_files():
            # Tokenize text
            try:
                encoded = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                )
                
                example = {
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                }
                
                buffer.append(example)
                
                # Yield from buffer when full
                if len(buffer) >= self.buffer_size:
                    if self.shuffle:
                        random.shuffle(buffer)
                    
                    for example in buffer:
                        yield example
                    
                    buffer = []
                    
            except Exception as e:
                print(f"Warning: Error tokenizing text: {e}")
        
        # Yield remaining examples
        if buffer:
            if self.shuffle:
                random.shuffle(buffer)
            
            for example in buffer:
                yield example


class JSONLDataset(Dataset):
    """
    Dataset for loading JSONL (JSON Lines) files.
    
    Each line should be a JSON object with a "text" field.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: ArborTokenizer,
        text_field: str = "text",
        max_length: int = 512,
    ):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
        
        # Load examples
        self.examples = self._load_examples()
    
    def _load_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Load and tokenize examples from JSONL file."""
        examples = []
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get(self.text_field, "")
                    
                    if text:
                        encoded = self.tokenizer.encode(
                            text,
                            max_length=self.max_length,
                            padding="max_length",
                            truncation=True,
                        )
                        
                        examples.append({
                            "input_ids": encoded["input_ids"][0],
                            "attention_mask": encoded["attention_mask"][0],
                        })
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings for language modeling.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )


def save_dataset_info(
    dataset: Dataset,
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save dataset information for reproducibility.
    
    Args:
        dataset: Dataset to describe
        save_path: Path to save info
        additional_info: Additional information to include
    """
    info = {
        "dataset_type": type(dataset).__name__,
        "size": len(dataset),
        "sample_example": dataset[0] if len(dataset) > 0 else None,
    }
    
    # Add dataset-specific info
    if hasattr(dataset, 'vocab_size'):
        info["vocab_size"] = dataset.vocab_size
    if hasattr(dataset, 'seq_length'):
        info["seq_length"] = dataset.seq_length
    if hasattr(dataset, 'tokenizer'):
        info["tokenizer_vocab_size"] = len(dataset.tokenizer)
    
    if additional_info:
        info.update(additional_info)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    print(f"Dataset info saved to: {save_path}")


def load_huggingface_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    tokenizer: ArborTokenizer,
    text_field: str = "text",
    max_length: int = 512,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load a dataset from HuggingFace Hub and convert to PyTorch Dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        subset: Subset/configuration name
        split: Dataset split to load
        tokenizer: Tokenizer to use
        text_field: Field containing text data
        max_length: Maximum sequence length
        cache_dir: Cache directory for downloaded data
        
    Returns:
        PyTorch Dataset
    """
    try:
        from datasets import load_dataset
        
        # Load HuggingFace dataset
        if subset:
            hf_dataset = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
        else:
            hf_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        # Convert to list of examples
        examples = []
        for item in hf_dataset:
            text = item.get(text_field, "")
            if text:
                encoded = tokenizer.encode(
                    text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
                
                examples.append({
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0],
                })
        
        # Create custom dataset class
        class HFDatasetWrapper(Dataset):
            def __init__(self, examples):
                self.examples = examples
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                return self.examples[idx]
        
        return HFDatasetWrapper(examples)
        
    except ImportError:
        raise ImportError("datasets library required for loading HuggingFace datasets")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset {dataset_name}: {e}")
