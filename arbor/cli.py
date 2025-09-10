"""
Streamlined CLI for Arbor-o1 inference.

Usage examples:
    # Generate text
    python -m arbor.cli generate "The future of AI is" --max-tokens 50

    # Interactive chat
    python -m arbor.cli chat --model path/to/model

    # Score text
    python -m arbor.cli score "This is a test sentence" --model path/to/model

    # Extract embeddings
    python -m arbor.cli embed "Text to embed" --model path/to/model
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

from .inference import ArborInference, load_model


def generate_command(args):
    """Handle text generation command."""
    try:
        inference = load_model(args.model, args.device)
        
        result = inference.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.sample,
            num_return_sequences=args.num_sequences
        )
        
        if args.num_sequences == 1:
            print(f"Prompt: {args.prompt}")
            print(f"Generated: {result}")
        else:
            print(f"Prompt: {args.prompt}")
            for i, seq in enumerate(result, 1):
                print(f"Generation {i}: {seq}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def chat_command(args):
    """Handle interactive chat command."""
    try:
        inference = load_model(args.model, args.device)
        
        print("🌱 Arbor-o1 Chat Interface")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("-" * 40)
        
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        
        while True:
            user_input = input("\\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            response = inference.chat(
                messages=messages,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            print(f"Assistant: {response}")
            messages.append({"role": "assistant", "content": response})
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def score_command(args):
    """Handle text scoring command."""
    try:
        inference = load_model(args.model, args.device)
        
        scores = inference.score_text(args.text)
        
        print(f"Text: {args.text}")
        print("-" * 40)
        print(f"Perplexity: {scores['perplexity']:.2f}")
        print(f"Log Likelihood: {scores['log_likelihood']:.2f}")
        print(f"Bits per Character: {scores['bits_per_character']:.3f}")
        print(f"Tokens: {scores['tokens']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(scores, f, indent=2)
            print(f"\\nScores saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def embed_command(args):
    """Handle embedding extraction command."""
    try:
        inference = load_model(args.model, args.device)
        
        if args.file:
            with open(args.file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            texts = [args.text]
        
        embeddings = inference.embed(texts, layer=args.layer)
        
        print(f"Extracted embeddings for {len(texts)} text(s)")
        print(f"Embedding shape: {embeddings.shape}")
        
        if args.output:
            torch.save(embeddings, args.output)
            print(f"Embeddings saved to {args.output}")
        else:
            # Show first few dimensions for preview
            for i, text in enumerate(texts):
                embedding = embeddings[i]
                preview = embedding[:5].tolist()
                print(f"Text {i+1}: '{text[:50]}...' -> [{', '.join(f'{x:.3f}' for x in preview)}, ...]")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def benchmark_command(args):
    """Handle benchmarking command."""
    try:
        import time
        
        inference = load_model(args.model, args.device)
        
        print(f"🔬 Benchmarking Arbor-o1 Model")
        print(f"Device: {args.device}")
        print(f"Prompt: '{args.prompt}'")
        print(f"Tokens to generate: {args.max_tokens}")
        print("-" * 40)
        
        times = []
        
        for i in range(args.runs):
            start_time = time.time()
            
            result = inference.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=args.sample
            )
            
            end_time = time.time()
            duration = end_time - start_time
            times.append(duration)
            
            tokens_per_second = args.max_tokens / duration
            
            print(f"Run {i+1}: {duration:.2f}s ({tokens_per_second:.1f} tokens/s)")
            
            if i == 0:  # Show first result
                print(f"Sample output: {result[:100]}...")
        
        avg_time = sum(times) / len(times)
        avg_tokens_per_second = args.max_tokens / avg_time
        
        print("-" * 40)
        print(f"Average time: {avg_time:.2f}s")
        print(f"Average speed: {avg_tokens_per_second:.1f} tokens/s")
        print(f"Model parameters: {inference.model.param_count():,}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Arbor-o1 CLI for inference and interaction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m arbor.cli generate "Hello world" --model ./my_model
  python -m arbor.cli chat --model ./my_model --system-prompt "You are helpful"
  python -m arbor.cli score "Test sentence" --model ./my_model
  python -m arbor.cli embed "Text to embed" --model ./my_model --output embeddings.pt
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="./checkpoints/latest",
        help="Path to model checkpoint directory"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text from prompt")
    gen_parser.add_argument("prompt", type=str, help="Input prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    gen_parser.add_argument("--sample", action="store_true", help="Use sampling (vs greedy)")
    gen_parser.add_argument("--num-sequences", type=int, default=1, help="Number of sequences")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat interface")
    chat_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per response")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--system-prompt", type=str, help="System prompt for chat")
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score text (perplexity, etc.)")
    score_parser.add_argument("text", type=str, help="Text to score")
    score_parser.add_argument("--output", "-o", type=str, help="Save scores to JSON file")
    
    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Extract embeddings")
    embed_parser.add_argument("--text", type=str, help="Text to embed")
    embed_parser.add_argument("--file", type=str, help="File with texts (one per line)")
    embed_parser.add_argument("--layer", type=int, default=-1, help="Layer to extract from")
    embed_parser.add_argument("--output", "-o", type=str, help="Save embeddings to file")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    bench_parser.add_argument("--prompt", type=str, default="The future of AI", help="Test prompt")
    bench_parser.add_argument("--max-tokens", type=int, default=50, help="Tokens to generate")
    bench_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    bench_parser.add_argument("--sample", action="store_true", help="Use sampling")
    bench_parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to appropriate command handler
    if args.command == "generate":
        generate_command(args)
    elif args.command == "chat":
        chat_command(args)
    elif args.command == "score":
        score_command(args)
    elif args.command == "embed":
        embed_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)


if __name__ == "__main__":
    main()
