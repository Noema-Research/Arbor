#!/usr/bin/env python3
"""
Data preparation script for Arbor-o1.

This script generates synthetic datasets or processes real datasets
for training Arbor models.
"""

import argparse
import os
import json
from typing import Dict, Any
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbor.data import SyntheticTextDataset, ArborTokenizer, save_dataset_info
from arbor.data.tokenize import create_synthetic_vocabulary, create_tokenizer_from_vocab


def create_synthetic_data(
    output_dir: str,
    vocab_size: int = 1000,
    seq_length: int = 512,
    num_sequences: int = 10000,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Create synthetic text dataset for training and evaluation.
    
    Args:
        output_dir: Directory to save dataset
        vocab_size: Size of vocabulary
        seq_length: Sequence length
        num_sequences: Total number of sequences to generate
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility
        
    Returns:
        Dataset information dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating synthetic dataset with {num_sequences:,} sequences...")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Sequence length: {seq_length}")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create tokenizer
    print("Creating tokenizer...")
    vocab = create_synthetic_vocabulary(vocab_size)
    tokenizer = create_tokenizer_from_vocab(vocab)
    
    # Save tokenizer
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    try:
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"Tokenizer saved to: {tokenizer_dir}")
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
    
    # Create full dataset
    print("Generating sequences...")
    full_dataset = SyntheticTextDataset(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_sequences=num_sequences,
        tokenizer=tokenizer,
        seed=seed,
    )
    
    # Split dataset
    train_size = int(num_sequences * (1 - val_split - test_split))
    val_size = int(num_sequences * val_split)
    test_size = num_sequences - train_size - val_size
    
    print(f"Splitting dataset: train={train_size:,}, val={val_size:,}, test={test_size:,}")
    
    # Save splits
    for split, start_idx, end_idx in [
        ("train", 0, train_size),
        ("val", train_size, train_size + val_size),
        ("test", train_size + val_size, num_sequences),
    ]:
        if start_idx >= end_idx:
            continue
            
        split_sequences = full_dataset.sequences[start_idx:end_idx]
        split_path = os.path.join(output_dir, f"{split}.pt")
        
        torch.save({
            "sequences": split_sequences,
            "vocab_size": vocab_size,
            "seq_length": seq_length,
        }, split_path)
        
        print(f"Saved {split} split: {len(split_sequences):,} sequences to {split_path}")
    
    # Save dataset info
    dataset_info = {
        "type": "synthetic",
        "vocab_size": vocab_size,
        "seq_length": seq_length,
        "total_sequences": num_sequences,
        "splits": {
            "train": train_size,
            "val": val_size,
            "test": test_size,
        },
        "seed": seed,
        "created_at": str(torch.cuda.Event()),
    }
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    save_dataset_info(full_dataset, info_path, dataset_info)
    
    print(f"Dataset creation completed! Saved to: {output_dir}")
    return dataset_info


def download_and_process_real_data(
    dataset_name: str,
    output_dir: str,
    max_samples: int = 100000,
    seq_length: int = 512,
    vocab_size: int = 10000,
) -> Dict[str, Any]:
    """
    Download and process a real dataset from HuggingFace.
    
    Args:
        dataset_name: Name of HuggingFace dataset
        output_dir: Directory to save processed data
        max_samples: Maximum number of samples to process
        seq_length: Target sequence length
        vocab_size: Vocabulary size for tokenizer
        
    Returns:
        Dataset information dictionary
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Please install datasets and transformers: pip install datasets transformers")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    
    # Load dataset
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text_field = "text"
    elif dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext", split="train")
        text_field = "text"
    else:
        # Generic loading
        dataset = load_dataset(dataset_name)
        text_field = "text"  # Assume text field name
    
    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = ArborTokenizer("gpt2", vocab_size=vocab_size)
    
    # Process and save splits
    dataset_info = {
        "type": "real",
        "source": dataset_name,
        "vocab_size": vocab_size,
        "seq_length": seq_length,
        "max_samples": max_samples,
    }
    
    for split_name in dataset.keys():
        if split_name not in ["train", "validation", "test"]:
            continue
            
        split_data = dataset[split_name]
        
        print(f"Processing {split_name} split...")
        
        sequences = []
        processed_count = 0
        
        for item in tqdm(split_data, desc=f"Processing {split_name}"):
            if processed_count >= max_samples:
                break
                
            text = item.get(text_field, "")
            if len(text.strip()) < 10:  # Skip very short texts
                continue
            
            # Tokenize
            encoded = tokenizer.encode(
                text,
                max_length=seq_length,
                padding="max_length",
                truncation=True,
            )
            
            sequences.append({
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
            })
            
            processed_count += 1
        
        # Save split
        split_path = os.path.join(output_dir, f"{split_name}.pt")
        torch.save(sequences, split_path)
        
        print(f"Saved {split_name}: {len(sequences):,} sequences to {split_path}")
        dataset_info[f"{split_name}_size"] = len(sequences)
    
    # Save tokenizer and info
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset processing completed! Saved to: {output_dir}")
    return dataset_info


def create_demo_data(output_dir: str) -> Dict[str, Any]:
    """Create a small demo dataset for quick testing."""
    return create_synthetic_data(
        output_dir=output_dir,
        vocab_size=500,
        seq_length=256,
        num_sequences=1000,
        val_split=0.2,
        test_split=0.1,
        seed=42,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Arbor-o1 training")
    
    parser.add_argument(
        "--type",
        choices=["synthetic", "real", "demo"],
        default="synthetic",
        help="Type of dataset to create"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset"
    )
    
    # Synthetic data options
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1000,
        help="Vocabulary size for synthetic data"
    )
    
    parser.add_argument(
        "--seq_length",
        type=int,
        default=512,
        help="Sequence length"
    )
    
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=10000,
        help="Number of sequences to generate"
    )
    
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split ratio"
    )
    
    # Real data options
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Name of HuggingFace dataset for real data"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum samples to process from real dataset"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("🌱 Arbor-o1 Data Preparation")
    print("=" * 40)
    
    if args.type == "synthetic":
        dataset_info = create_synthetic_data(
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            num_sequences=args.num_sequences,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
        )
    elif args.type == "real":
        dataset_info = download_and_process_real_data(
            dataset_name=args.dataset_name,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
        )
    elif args.type == "demo":
        dataset_info = create_demo_data(args.output_dir)
    else:
        raise ValueError(f"Unknown data type: {args.type}")
    
    print("\n✅ Data preparation completed!")
    print(f"Dataset type: {dataset_info['type']}")
    print(f"Total sequences: {dataset_info.get('total_sequences', 'N/A'):,}")
    print(f"Vocabulary size: {dataset_info['vocab_size']:,}")
    print(f"Sequence length: {dataset_info['seq_length']}")
    print(f"Output directory: {args.output_dir}")
    
    print("\n🚀 Ready for training! Use this data with:")
    print(f"python scripts/train.py --data_dir {args.output_dir} --config configs/arbor_small.yaml")


if __name__ == "__main__":
    main()
