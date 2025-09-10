"""
Example usage of the Arbor custom inference library.

This shows how to use the simple, optimized inference interface
without requiring Hugging Face Transformers.
"""

import torch
from pathlib import Path

# If you've installed arbor-o1 as a package:
# from arbor import ArborInference, load_model, generate_text

# For development/local usage:
import sys
sys.path.append("..")
from arbor.inference import ArborInference, load_model, generate_text


def main():
    print("🌱 Arbor-o1 Custom Inference Example")
    print("=" * 50)
    
    # Example 1: Load model and generate text
    print("\\n1. Loading model...")
    
    # Replace with your actual model path
    model_path = "path/to/your/trained/model"
    
    # For this example, let's create a dummy inference object
    # In practice, you'd load from a real checkpoint
    try:
        inference = load_model(model_path)
    except FileNotFoundError:
        print("⚠️  Model path not found, creating example with dummy model...")
        
        # Create example inference object for demonstration
        from arbor.modeling.model import ArborTransformer, ArborConfig
        from arbor.data.tokenizers import ArborTokenizer
        
        config = ArborConfig(
            vocab_size=50257,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            max_position_embeddings=1024
        )
        
        model = ArborTransformer(config)
        tokenizer = ArborTokenizer("gpt2", vocab_size=config.vocab_size)
        
        inference = ArborInference(model, tokenizer, device="cpu")
    
    print("✅ Model loaded successfully!")
    
    # Example 2: Simple text generation
    print("\\n2. Generating text...")
    
    prompt = "The future of artificial intelligence is"
    generated = inference.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    
    # Example 3: Multiple generations
    print("\\n3. Multiple generations...")
    
    multiple_outputs = inference.generate(
        prompt="Once upon a time",
        max_new_tokens=30,
        temperature=1.0,
        num_return_sequences=3
    )
    
    for i, output in enumerate(multiple_outputs, 1):
        print(f"Generation {i}: {output}")
    
    # Example 4: Chat interface
    print("\\n4. Chat interface...")
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    response = inference.chat(
        messages=messages,
        max_new_tokens=100,
        temperature=0.7
    )
    
    print(f"User: What is machine learning?")
    print(f"Assistant: {response}")
    
    # Example 5: Text embeddings
    print("\\n5. Extracting embeddings...")
    
    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning models can process vast amounts of data.",
        "The weather is nice today."
    ]
    
    embeddings = inference.embed(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute similarity between first two texts
    similarity = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    print(f"Similarity between AI texts: {similarity:.3f}")
    
    similarity_weather = torch.cosine_similarity(embeddings[0], embeddings[2], dim=0)
    print(f"Similarity AI vs weather: {similarity_weather:.3f}")
    
    # Example 6: Text scoring (perplexity)
    print("\\n6. Text scoring...")
    
    test_text = "This is a well-formed sentence that should have reasonable perplexity."
    scores = inference.score_text(test_text)
    
    print(f"Text: {test_text}")
    print(f"Perplexity: {scores['perplexity']:.2f}")
    print(f"Bits per character: {scores['bits_per_character']:.3f}")
    print(f"Tokens: {scores['tokens']}")
    
    # Example 7: Batch processing
    print("\\n7. Batch processing...")
    
    prompts = [
        "The benefits of renewable energy include",
        "In the field of robotics, recent advances",
        "Climate change is affecting"
    ]
    
    batch_results = []
    for prompt in prompts:
        result = inference.generate(
            prompt=prompt,
            max_new_tokens=25,
            temperature=0.6
        )
        batch_results.append(result)
    
    for prompt, result in zip(prompts, batch_results):
        print(f"'{prompt}' → '{result}'")
    
    # Example 8: Save and load
    print("\\n8. Saving model...")
    
    save_path = "./saved_arbor_model"
    inference.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    
    # Load it back
    print("\\n9. Loading saved model...")
    try:
        loaded_inference = ArborInference.from_pretrained(save_path)
        print("✅ Model loaded successfully from save!")
        
        # Test generation
        test_gen = loaded_inference.generate("Hello", max_new_tokens=10)
        print(f"Test generation: Hello{test_gen}")
        
    except Exception as e:
        print(f"⚠️  Loading failed (expected in example): {e}")
    
    print("\\n" + "=" * 50)
    print("🌱 Arbor-o1 Custom Inference Example Complete!")


def simple_usage_example():
    """
    Simplified usage example showing the most common patterns.
    """
    print("\\n🚀 Simple Usage Examples")
    print("-" * 30)
    
    # Quick generation (if you have a saved model)
    model_path = "path/to/your/model"
    
    try:
        # One-liner text generation
        result = generate_text(
            model_path=model_path,
            prompt="The key to success is",
            max_tokens=30,
            temperature=0.8
        )
        print(f"Quick generation: {result}")
        
    except FileNotFoundError:
        print("⚠️  Quick generation requires a saved model")
    
    # Interactive loop (commented out for demo)
    """
    print("\\n💬 Interactive mode (uncomment to try):")
    inference = load_model(model_path)
    
    while True:
        user_input = input("\\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        response = inference.generate(
            prompt=f"User: {user_input}\\nAssistant:",
            max_new_tokens=50,
            temperature=0.7,
            stop_tokens=["User:", "\\n\\n"]
        )
        
        print(f"Assistant: {response}")
    """


if __name__ == "__main__":
    main()
    simple_usage_example()
