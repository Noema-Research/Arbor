"""
Example usage of Arbor with Hugging Face Transformers.
"""

from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

# Register Arbor model with transformers
from arbor.transformers_integration import (
    ArborTransformersConfig,
    ArborForCausalLM,
    ArborModel
)

def main():
    print("🌱 Arbor-o1 + Hugging Face Transformers Integration Demo")
    print("=" * 60)
    
    # 1. Create config compatible with transformers
    config = ArborTransformersConfig(
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        n_positions=1024,
        # Arbor-specific
        growth_enabled=True,
        growth_factor=1.3
    )
    
    print(f"📊 Created Arbor config:")
    print(f"   - Vocab size: {config.vocab_size:,}")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Layers: {config.num_hidden_layers}")
    print(f"   - Growth enabled: {config.growth_enabled}")
    
    # 2. Create model using transformers-style API
    model = ArborForCausalLM(config)
    print(f"\n🧠 Created Arbor model: {model.param_count():,} parameters")
    
    # 3. Use standard transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Standard transformers inference
    text = "The future of AI is"
    inputs = tokenizer(text, return_tensors="pt")
    
    print(f"\n📝 Input: '{text}'")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📄 Generated: '{generated_text}'")
    
    # 5. Demonstrate growth capability
    print(f"\n🌱 Before growth: {model.param_count():,} parameters")
    model.grow(growth_factor=1.2)
    print(f"🌿 After growth: {model.param_count():,} parameters")
    
    # 6. Still works with transformers API after growth
    with torch.no_grad():
        outputs_after_growth = model.generate(
            inputs.input_ids,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_after_growth = tokenizer.decode(outputs_after_growth[0], skip_special_tokens=True)
    print(f"📄 Generated after growth: '{generated_after_growth}'")
    
    print("\n✅ Arbor + Transformers integration working!")
    print("🎯 You can now use Arbor with standard transformers workflows")

if __name__ == "__main__":
    main()
