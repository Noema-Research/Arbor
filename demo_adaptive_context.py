#!/usr/bin/env python3
"""
Adaptive Context Demo for Arbor Models.

This script demonstrates the dynamic context window adaptation system.
Shows how the model analyzes different types of inputs and adapts its
context window based on task complexity and hardware constraints.
"""

import torch
import sys
from pathlib import Path

# Add arbor to path
sys.path.insert(0, str(Path(__file__).parent))

from arbor.modeling.model import ArborTransformer, ArborConfig
from arbor.modeling.adaptive_context import ContextDecision
from transformers import AutoTokenizer


def create_adaptive_arbor_model():
    """Create an Arbor model with adaptive context enabled."""
    config = ArborConfig(
        vocab_size=128000,
        dim=1024,
        num_layers=12,  # Smaller for demo
        num_heads=16,
        ffn_dim=4096,
        max_seq_length=131072,
        
        # Enable adaptive context
        adaptive_context=True,
        min_context_length=1024,
        max_context_length=32768,  # Limit for demo
        growth_enabled=True
    )
    
    model = ArborTransformer(config)
    return model


def load_tokenizer():
    """Load the Hermes tokenizer."""
    print("📥 Loading Hermes-4-405B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'NousResearch/Hermes-4-405B',
        force_download=False  # Use cache for demo
    )
    return tokenizer


def demo_task_examples():
    """Different types of tasks to test adaptive context."""
    return {
        "simple_chat": {
            "text": "Hello! How are you today?",
            "expected_task": "chat",
            "expected_context": "short"
        },
        
        "complex_reasoning": {
            "text": """Let me think step by step about this complex mathematical proof. 
            We need to prove that for any prime number p > 2, the equation x^2 ≡ -1 (mod p) 
            has a solution if and only if p ≡ 1 (mod 4). This involves understanding 
            quadratic residues, the Legendre symbol, and properties of finite fields...""",
            "expected_task": "reasoning",
            "expected_context": "long"
        },
        
        "code_task": {
            "text": """def implement_transformer_attention(query, key, value, mask=None):
    # Implementing scaled dot-product attention
    # This is a complex function that needs careful implementation
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights""",
            "expected_task": "code",
            "expected_context": "medium"
        },
        
        "document_processing": {
            "text": """This is a comprehensive research paper on natural language processing. 
            It covers multiple sections including introduction, related work, methodology, 
            experiments, results, discussion, and conclusions. The paper is approximately 
            20 pages long and contains detailed technical information about transformer 
            architectures, attention mechanisms, and their applications in various NLP tasks.
            
            Introduction:
            Natural language processing has evolved significantly over the past decade...
            
            Related Work:
            Previous approaches to language modeling include...
            
            Methodology:
            Our approach builds upon the transformer architecture...""",
            "expected_task": "document",
            "expected_context": "very_long"
        },
        
        "creative_writing": {
            "text": """Once upon a time, in a magical kingdom far away, there lived a young 
            princess named Luna who had the extraordinary ability to communicate with stars. 
            Every night, she would climb to the highest tower of the castle and whisper 
            secrets to the constellations above...""",
            "expected_task": "creative",
            "expected_context": "medium"
        }
    }


def analyze_task(model, tokenizer, task_name: str, task_data: dict):
    """Analyze a specific task with the adaptive context system."""
    print(f"\n🔍 Analyzing Task: {task_name}")
    print(f"Expected: {task_data['expected_task']} task, {task_data['expected_context']} context")
    print("-" * 60)
    
    # Tokenize input
    text = task_data["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]
    
    print(f"📝 Input length: {input_ids.shape[1]} tokens")
    
    # Get model's context info before adaptation
    initial_info = model.get_context_info()
    print(f"🏁 Initial context: {initial_info['current_context_length']:,} tokens")
    
    # Run inference to trigger adaptive context
    model.eval()
    with torch.no_grad():
        # This will trigger the adaptive context system
        outputs = model(input_ids, return_dict=True)
        
        # Get updated context info
        adapted_info = model.get_context_info()
        
        print(f"🎯 Adapted context: {adapted_info['current_context_length']:,} tokens")
        
        # Show logits shape to confirm processing
        logits_shape = outputs["logits"].shape
        print(f"📊 Output shape: {logits_shape}")
        
        return {
            "initial_context": initial_info['current_context_length'],
            "adapted_context": adapted_info['current_context_length'],
            "input_length": input_ids.shape[1],
            "task_name": task_name
        }


def hardware_constraint_demo(model, tokenizer):
    """Demonstrate hardware-aware context adaptation."""
    print("\n🖥️  Hardware Constraint Demo")
    print("=" * 60)
    
    # Create a very long input that would exceed memory
    long_text = "This is a test sentence. " * 1000  # Very long input
    inputs = tokenizer(long_text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]
    
    print(f"📏 Ultra-long input: {input_ids.shape[1]:,} tokens")
    
    # Show how the system adapts to hardware constraints
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_ids, return_dict=True)
            final_info = model.get_context_info()
            print(f"✅ Successfully processed with {final_info['current_context_length']:,} token context")
            print(f"📊 Final output shape: {outputs['logits'].shape}")
        except Exception as e:
            print(f"❌ Error: {e}")


def context_growth_demo(model, tokenizer):
    """Demonstrate manual context length adjustment."""
    print("\n📈 Context Growth Demo")
    print("=" * 60)
    
    # Test different context lengths manually
    test_lengths = [1024, 4096, 16384, 32768]
    
    sample_text = "The quick brown fox jumps over the lazy dog. " * 50
    inputs = tokenizer(sample_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    for length in test_lengths:
        print(f"\n🔧 Testing context length: {length:,}")
        
        try:
            # Force specific context length
            model.force_context_length(length)
            
            # Test inference
            with torch.no_grad():
                outputs = model(input_ids, return_dict=True)
                print(f"   ✅ Success! Output shape: {outputs['logits'].shape}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def main():
    """Main demo function."""
    print("🌱 Arbor Adaptive Context Window Demo")
    print("=" * 60)
    
    try:
        # Create model and tokenizer
        print("🔨 Creating adaptive Arbor model...")
        model = create_adaptive_arbor_model()
        
        tokenizer = load_tokenizer()
        
        # Show model info
        info = model.get_context_info()
        print(f"✅ Model created!")
        print(f"   Parameters: {model.param_count():,}")
        print(f"   Adaptive context: {info['adaptive_context_enabled']}")
        print(f"   Context range: {info['min_context_length']:,} - {info['max_context_length']:,}")
        print(f"   Available lengths: {info['available_context_lengths']}")
        print(f"   Supported tasks: {info['supported_task_types']}")
        
        # Test different task types
        tasks = demo_task_examples()
        results = []
        
        for task_name, task_data in tasks.items():
            try:
                result = analyze_task(model, tokenizer, task_name, task_data)
                results.append(result)
            except Exception as e:
                print(f"❌ Error analyzing {task_name}: {e}")
        
        # Summary
        print("\n📊 Results Summary")
        print("=" * 60)
        for result in results:
            print(f"{result['task_name']:<20} | "
                  f"Input: {result['input_length']:<4} | "
                  f"Initial: {result['initial_context']:<5} | "
                  f"Adapted: {result['adapted_context']:<5}")
        
        # Hardware demo
        hardware_constraint_demo(model, tokenizer)
        
        # Context growth demo
        context_growth_demo(model, tokenizer)
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Task-aware context adaptation")
        print("✅ Hardware constraint handling") 
        print("✅ Dynamic context resizing")
        print("✅ Multiple task type support")
        print("✅ Real-time adaptation during inference")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
