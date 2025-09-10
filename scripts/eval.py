#!/usr/bin/env python3
"""
Evaluation script for Arbor-o1 models.

This script handles model evaluation including:
- Loading trained models from checkpoints
- Computing comprehensive metrics
- Generating sample text
- Comparing different models
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arbor.train import load_model_for_inference, load_checkpoint
from arbor.data import ArborTokenizer, create_dataloader
from arbor.utils.metrics import (
    compute_token_level_metrics,
    compute_model_size_metrics,
    compute_growth_metrics,
    create_metrics_summary,
    compare_models,
)


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: str = "cuda",
) -> tuple:
    """
    Load model and tokenizer from checkpoint.
    
    Returns:
        Tuple of (model, tokenizer, metadata)
    """
    print(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = load_model_for_inference(checkpoint_path, device=device)
    
    # Try to load tokenizer from same directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
    
    if os.path.exists(tokenizer_dir):
        tokenizer = ArborTokenizer(tokenizer_dir)
    else:
        # Fallback: try parent directory or use default
        parent_tokenizer_dir = os.path.join(os.path.dirname(checkpoint_dir), "tokenizer")
        if os.path.exists(parent_tokenizer_dir):
            tokenizer = ArborTokenizer(parent_tokenizer_dir)
        else:
            # Use GPT-2 tokenizer and resize
            print("Warning: No tokenizer found, using GPT-2 with model vocab size")
            tokenizer = ArborTokenizer("gpt2", vocab_size=model.config.vocab_size)
    
    # Load metadata
    metadata = {}
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        metadata = {
            "step": checkpoint_data.get("step", 0),
            "loss": checkpoint_data.get("loss", float('inf')),
            "growth_history": checkpoint_data.get("growth_history", []),
            "config": model.config.__dict__,
        }
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
    
    print(f"Model loaded: {model.param_count():,} parameters")
    if metadata.get("growth_history"):
        print(f"Growth events: {len(metadata['growth_history'])}")
    
    return model, tokenizer, metadata


def evaluate_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: ArborTokenizer,
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        tokenizer: Tokenizer
        device: Device to use
        max_batches: Maximum number of batches to evaluate (for speed)
        
    Returns:
        Evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0
    num_batches = 0
    
    all_logits = []
    all_labels = []
    
    print("Evaluating on dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
            attention_mask = batch.get("attention_mask")
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Compute metrics
            batch_metrics = compute_token_level_metrics(logits, labels)
            
            total_loss += loss.item()
            total_accuracy += batch_metrics["accuracy"]
            total_tokens += input_ids.numel()
            num_batches += 1
            
            # Collect for sequence-level metrics (sample only)
            if batch_idx < 10:  # Only first 10 batches for efficiency
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
    
    # Compute average metrics
    avg_metrics = {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
        "perplexity": torch.exp(torch.tensor(total_loss / num_batches)).item(),
        "total_tokens": total_tokens,
        "num_batches": num_batches,
    }
    
    # Compute detailed metrics on sample
    if all_logits and all_labels:
        sample_logits = torch.cat(all_logits, dim=0)
        sample_labels = torch.cat(all_labels, dim=0)
        
        detailed_metrics = compute_token_level_metrics(sample_logits, sample_labels)
        avg_metrics.update({
            f"sample_{k}": v for k, v in detailed_metrics.items()
        })
    
    return avg_metrics


def generate_samples(
    model: nn.Module,
    tokenizer: ArborTokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    device: str = "cuda",
) -> List[Dict[str, str]]:
    """
    Generate text samples from prompts.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use
        
    Returns:
        List of generation results
    """
    model.eval()
    results = []
    
    print(f"Generating {len(prompts)} samples...")
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            
            # Generate
            try:
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
                
                # Decode
                full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generated_text = full_text[len(prompt):].strip()
                
                results.append({
                    "prompt": prompt,
                    "generated": generated_text,
                    "full_text": full_text,
                })
                
            except Exception as e:
                print(f"Warning: Generation failed for prompt '{prompt}': {e}")
                results.append({
                    "prompt": prompt,
                    "generated": "[Generation failed]",
                    "error": str(e),
                })
    
    return results


def evaluate_model(
    checkpoint_path: str,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "auto",
    max_batches: Optional[int] = None,
    generate_samples: bool = True,
) -> Dict[str, Any]:
    """
    Complete model evaluation pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to evaluation dataset
        output_dir: Directory to save results
        device: Device to use
        max_batches: Maximum batches for evaluation
        generate_samples: Whether to generate text samples
        
    Returns:
        Complete evaluation results
    """
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model, tokenizer, metadata = load_model_and_tokenizer(checkpoint_path, device)
    
    # Prepare results
    results = {
        "checkpoint_path": checkpoint_path,
        "device": device,
        "metadata": metadata,
    }
    
    # Model size metrics
    model_metrics = compute_model_size_metrics(model)
    results["model_metrics"] = model_metrics
    
    # Growth metrics if available
    if metadata.get("growth_history"):
        growth_metrics = compute_growth_metrics(metadata["growth_history"])
        results["growth_metrics"] = growth_metrics
    
    # Dataset evaluation
    if data_path and os.path.exists(data_path):
        print(f"\nEvaluating on dataset: {data_path}")
        
        # Load dataset
        if data_path.endswith(".pt"):
            data = torch.load(data_path)
            
            # Create dataset wrapper
            class EvalDataset(torch.utils.data.Dataset):
                def __init__(self, data, tokenizer):
                    if isinstance(data, dict) and "sequences" in data:
                        self.sequences = data["sequences"]
                    elif isinstance(data, list):
                        self.sequences = [item["input_ids"].squeeze() if "input_ids" in item else item 
                                        for item in data]
                    else:
                        self.sequences = data
                    
                    self.tokenizer = tokenizer
                
                def __len__(self):
                    return len(self.sequences)
                
                def __getitem__(self, idx):
                    seq = self.sequences[idx]
                    if isinstance(seq, dict):
                        return seq
                    else:
                        # Create attention mask
                        attention_mask = (seq != self.tokenizer.pad_token_id).long()
                        return {
                            "input_ids": seq.unsqueeze(0),
                            "attention_mask": attention_mask.unsqueeze(0),
                        }
            
            eval_dataset = EvalDataset(data, tokenizer)
            eval_dataloader = create_dataloader(
                eval_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            
            # Evaluate
            eval_metrics = evaluate_on_dataset(
                model, eval_dataloader, tokenizer, device, max_batches
            )
            results["evaluation_metrics"] = eval_metrics
            
            print(f"Evaluation loss: {eval_metrics['loss']:.4f}")
            print(f"Evaluation perplexity: {eval_metrics['perplexity']:.2f}")
            print(f"Evaluation accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Text generation samples
    if generate_samples:
        print("\nGenerating text samples...")
        
        sample_prompts = [
            "The quick brown fox",
            "In a world where",
            "Once upon a time",
            "The future of AI",
            "Today I learned that",
        ]
        
        generation_results = generate_samples(
            model, tokenizer, sample_prompts, 
            max_new_tokens=50, temperature=0.8, device=device
        )
        results["generation_samples"] = generation_results
        
        print("Sample generations:")
        for result in generation_results[:3]:  # Show first 3
            print(f"  Prompt: {result['prompt']}")
            print(f"  Generated: {result['generated']}")
            print()
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comprehensive results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {results_path}")
        
        # Save human-readable summary
        summary_path = os.path.join(output_dir, "evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Arbor-o1 Model Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Parameters: {model_metrics['total_parameters']:,}\n")
            f.write(f"Memory: {model_metrics['parameter_memory_mb']:.1f} MB\n\n")
            
            if "evaluation_metrics" in results:
                eval_metrics = results["evaluation_metrics"]
                f.write("Dataset Evaluation:\n")
                f.write(f"  Loss: {eval_metrics['loss']:.4f}\n")
                f.write(f"  Perplexity: {eval_metrics['perplexity']:.2f}\n")
                f.write(f"  Accuracy: {eval_metrics['accuracy']:.4f}\n")
                f.write(f"  Tokens: {eval_metrics['total_tokens']:,}\n\n")
            
            if "growth_metrics" in results:
                growth_metrics = results["growth_metrics"]
                f.write("Growth Summary:\n")
                f.write(f"  Events: {growth_metrics['num_growth_events']}\n")
                f.write(f"  Final params: {growth_metrics['final_parameters']:,}\n")
                f.write(f"  Growth rate: {growth_metrics['growth_rate']:.2f}x\n\n")
            
            if "generation_samples" in results:
                f.write("Generation Samples:\n")
                for i, sample in enumerate(results["generation_samples"][:3]):
                    f.write(f"  {i+1}. {sample['prompt']} -> {sample['generated']}\n")
        
        print(f"Summary saved to: {summary_path}")
    
    return results


def compare_checkpoints(
    checkpoint_paths: List[str],
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Compare multiple model checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        data_path: Path to evaluation dataset
        output_dir: Directory to save comparison results
        device: Device to use
        
    Returns:
        Comparison results
    """
    print(f"Comparing {len(checkpoint_paths)} models...")
    
    model_results = {}
    
    # Evaluate each model
    for i, checkpoint_path in enumerate(checkpoint_paths):
        model_name = f"model_{i+1}_{Path(checkpoint_path).stem}"
        print(f"\n--- Evaluating {model_name} ---")
        
        try:
            results = evaluate_model(
                checkpoint_path=checkpoint_path,
                data_path=data_path,
                device=device,
                max_batches=50,  # Limit for speed in comparison
                generate_samples=False,  # Skip generation for comparison
            )
            model_results[model_name] = results
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            model_results[model_name] = {"error": str(e)}
    
    # Create comparison
    comparison = {
        "models": model_results,
        "comparison_summary": {},
    }
    
    # Compare key metrics
    valid_models = {name: results for name, results in model_results.items() 
                   if "error" not in results}
    
    if len(valid_models) >= 2:
        model_names = list(valid_models.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Extract comparable metrics
                metrics1 = {}
                metrics2 = {}
                
                if "evaluation_metrics" in valid_models[model1]:
                    metrics1.update(valid_models[model1]["evaluation_metrics"])
                if "model_metrics" in valid_models[model1]:
                    metrics1.update(valid_models[model1]["model_metrics"])
                
                if "evaluation_metrics" in valid_models[model2]:
                    metrics2.update(valid_models[model2]["evaluation_metrics"])
                if "model_metrics" in valid_models[model2]:
                    metrics2.update(valid_models[model2]["model_metrics"])
                
                if metrics1 and metrics2:
                    comp_result = compare_models(metrics1, metrics2, model1, model2)
                    comparison["comparison_summary"][f"{model1}_vs_{model2}"] = comp_result
    
    # Save comparison results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        comparison_path = os.path.join(output_dir, "model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"Comparison results saved to: {comparison_path}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arbor-o1 models")
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to evaluation dataset (.pt file)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for evaluation"
    )
    
    parser.add_argument(
        "--max_batches",
        type=int,
        help="Maximum number of batches to evaluate (for speed)"
    )
    
    parser.add_argument(
        "--no_generation",
        action="store_true",
        help="Skip text generation samples"
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple checkpoints (provide multiple paths)"
    )
    
    args = parser.parse_args()
    
    print("🌱 Arbor-o1 Model Evaluation")
    print("=" * 40)
    
    if args.compare:
        # Compare multiple models
        results = compare_checkpoints(
            checkpoint_paths=args.compare,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
        )
        
        print("\n📊 Model comparison completed!")
        
    else:
        # Evaluate single model
        results = evaluate_model(
            checkpoint_path=args.checkpoint_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
            max_batches=args.max_batches,
            generate_samples=not args.no_generation,
        )
        
        print("\n✅ Evaluation completed!")
        
        # Print summary
        if "model_metrics" in results:
            metrics = results["model_metrics"]
            print(f"Model: {metrics['total_parameters']:,} parameters")
        
        if "evaluation_metrics" in results:
            eval_metrics = results["evaluation_metrics"]
            print(f"Loss: {eval_metrics['loss']:.4f}")
            print(f"Perplexity: {eval_metrics['perplexity']:.2f}")
            print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
        
        if "growth_metrics" in results:
            growth_metrics = results["growth_metrics"]
            print(f"Growth events: {growth_metrics['num_growth_events']}")
            print(f"Growth ratio: {growth_metrics['growth_rate']:.2f}x")


if __name__ == "__main__":
    main()
