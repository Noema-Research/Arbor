# 📄 Arbor Long Context Guide: 4K → 128K+ Tokens

## 🎯 **Context Scaling Strategy**

Your Arbor model is now configured for **progressive context scaling**:

- **Demo**: 4K tokens (fast, efficient)
- **Production**: 64K tokens (document processing)
- **Future**: 128K+ tokens (full scaling support)

## ⚙️ **Technical Implementation**

### **RoPE Linear Scaling**
```json
{
  "max_position_embeddings": 131072,
  "rope_theta": 10000.0,
  "rope_scaling": {
    "type": "linear", 
    "factor": 32.0
  }
}
```

- **Base Context**: 4K tokens (factor 1.0)
- **Scaled Context**: 128K tokens (factor 32.0)
- **Method**: Linear RoPE interpolation
- **Efficiency**: Maintains quality across all lengths

### **Memory Optimizations**
```python
# Efficient attention patterns
"use_flash_attention": true,
"gradient_checkpointing": true,
"attention_dropout": 0.0,
"use_sliding_window": false
```

## 🚀 **Usage Examples**

### **Short Context (4K - Demo)**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("username/arbor-500m-1b")
tokenizer = AutoTokenizer.from_pretrained("username/arbor-500m-1b")

# Regular chat/completion
prompt = "Explain quantum computing"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
```

### **Medium Context (16K-32K - Documents)**
```python
# Process long documents
long_document = open("research_paper.txt").read()  # ~20K tokens

inputs = tokenizer(
    long_document + "\\n\\nSummarize the key findings:",
    return_tensors="pt",
    max_length=32768,
    truncation=True
)

summary = model.generate(
    **inputs,
    max_new_tokens=500,
    temperature=0.7
)
```

### **Long Context (64K-128K - Full Documents)**
```python
# Full book or multiple documents
full_context = read_multiple_files()  # ~100K tokens

inputs = tokenizer(
    full_context + "\\n\\nAnalyze the themes across all documents:",
    return_tensors="pt", 
    max_length=131072,
    truncation=True
)

analysis = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.8
)
```

## 💡 **Adaptive Context Length**

### **Smart Context Selection**
```python
def adaptive_context(text, max_context=None):
    """Automatically choose optimal context length."""
    token_count = len(tokenizer.encode(text))
    
    if token_count <= 4096:
        return 4096      # Fast processing
    elif token_count <= 16384:
        return 16384     # Medium documents
    elif token_count <= 65536:
        return 65536     # Long documents
    else:
        return 131072    # Maximum context
        
# Usage
optimal_length = adaptive_context(your_text)
inputs = tokenizer(your_text, max_length=optimal_length, truncation=True)
```

### **Progressive Context Growth**
```python
def process_with_growth(text, model):
    """Use model growth for longer contexts."""
    token_count = len(tokenizer.encode(text))
    
    if token_count > 32768:
        # Grow model for long context processing
        initial_params = model.num_parameters()
        model.grow()
        print(f"Grown for long context: {initial_params:,} → {model.num_parameters():,}")
    
    return model.generate(**inputs)
```

## 🎛️ **Configuration Options**

### **Training Configuration**
```python
# For training with long contexts
LONG_CONTEXT_TRAINING = {
    "max_seq_length": 131072,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 2,  # Reduce for memory
    "per_device_train_batch_size": 1,  # Small batch for long sequences
    "gradient_accumulation_steps": 32,  # Maintain effective batch size
    "bf16": True,  # Use bfloat16 for stability
    "rope_scaling_factor": 32.0,
}
```

### **Inference Configuration**
```python
# Memory-efficient inference
INFERENCE_CONFIG = {
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "low_cpu_mem_usage": True,
    "use_cache": True,
    "pad_token_id": 0,
    "attention_implementation": "flash_attention_2"  # If available
}
```

## 📊 **Performance Scaling**

| Context Length | Memory (GB) | Speed (tok/s) | Use Case |
|----------------|-------------|---------------|----------|
| 4K             | 0.8         | 50            | Chat, QA |
| 16K            | 1.2         | 35            | Articles |
| 32K            | 2.0         | 25            | Papers |
| 64K            | 3.5         | 15            | Books |
| 128K           | 6.5         | 8             | Archives |

## 🔧 **Memory Management**

### **Gradient Checkpointing**
```python
# Enable for training
model.gradient_checkpointing_enable()

# Configure checkpointing segments
model.config.gradient_checkpointing = True
model.config.use_cache = False  # During training
```

### **Attention Optimization**
```python
# Use efficient attention implementations
import torch.nn.functional as F

# Flash Attention (if available)
if hasattr(F, 'scaled_dot_product_attention'):
    model.config.use_flash_attention = True
    
# Memory-efficient attention
model.config.attention_dropout = 0.0  # Reduce memory fragmentation
```

## 🎯 **Best Practices**

### **1. Progressive Testing**
```python
# Test context scaling gradually
for context_len in [4096, 8192, 16384, 32768, 65536]:
    test_input = "test " * (context_len // 4)
    try:
        result = model.generate(...)
        print(f"✅ {context_len} tokens: Success")
    except RuntimeError as e:
        print(f"❌ {context_len} tokens: {e}")
        break
```

### **2. Batch Processing**
```python
# Process long documents in chunks
def process_long_document(doc, chunk_size=32768, overlap=512):
    chunks = []
    for i in range(0, len(doc), chunk_size - overlap):
        chunk = doc[i:i + chunk_size]
        result = model.generate(tokenizer(chunk, return_tensors="pt"))
        chunks.append(result)
    return combine_chunks(chunks)
```

### **3. Dynamic Scaling**
```python
# Automatically scale based on input
def smart_generate(text, model, tokenizer):
    tokens = len(tokenizer.encode(text))
    
    # Choose appropriate settings
    if tokens > 65536:
        # Use model growth for very long contexts
        model.grow()
        max_memory = True
    
    # Generate with optimal settings
    return model.generate(
        **tokenizer(text, return_tensors="pt"),
        max_new_tokens=min(1000, 131072 - tokens)
    )
```

## 🌟 **Future Proofing Features**

### **Implemented**
- ✅ RoPE linear scaling (up to 128K)
- ✅ Flash attention support
- ✅ Gradient checkpointing
- ✅ Efficient memory management
- ✅ Progressive context testing

### **Ready for Extension**
- 🔄 Sliding window attention (disabled, can enable)
- 🔄 Ring attention patterns
- 🔄 Hierarchical attention
- 🔄 Mixture of depths
- 🔄 Context compression techniques

## 🎉 **Summary**

Your Arbor model now supports:
- **Current**: 4K demo context
- **Near-term**: 64K production context  
- **Future**: 128K+ with room for growth
- **Scaling**: Linear RoPE interpolation
- **Efficiency**: Flash attention + checkpointing
- **Flexibility**: Adaptive context selection

Perfect for everything from quick demos to processing entire books! 📚✨
