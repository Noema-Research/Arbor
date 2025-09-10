---
language: en
license: apache-2.0
library_name: tran- **Parameters**: 699M (base) → 799M (grown) 
- **Architecture**: 24 layers, 1024 hidden, 16 attention heads
- **Context**: 4096 tokens (demo), up to 128K with scaling
- **Format**: SafeTensors (`.safetensors`) for secure model loading
- **Tokenizer**: Hermes-4-405B JSON-based tokenizer (128,000 vocab, no `.model` file)rs
tags:
- arbor
- dynamic-growth
- transformer
- causal-lm
- text-generation
pipeline_tag: text-generation
widget:
- text: "The future of artificial intelligence is"
  example_title: "AI Future"
- text: "Once upon a time"
  example_title: "Creative Writing"
---

# Arbor-500M-1B Dynamic Growth Model

A dynamic growth transformer that starts at 420M parameters and can expand to 1.1B parameters.

## Features

🌱 **Dynamic Growth**: Expands from 699M to 799M parameters as needed  
🚀 **Efficient**: Starts larger, grows strategically when beneficial  
⚡ **HF Compatible**: Works with standard transformers library  
🦙 **Hermes-4-405B tokenizer**: Uses NousResearch Hermes-4-405B tokenizer (128K vocab)
📄 **Long Context**: Supports up to 128K tokens with efficient scaling  

## Quick Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("your-username/arbor-500m-1b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your-username/arbor-500m-1b")

# Generate text (works with any context length up to 128K)
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Long context example
long_text = "Your very long document here..." * 1000  # Up to 128K tokens
inputs = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=65536)
outputs = model.generate(**inputs, max_new_tokens=100)

# Dynamic growth
model.grow()  # Expand model when needed
```

## Model Details

- **Base Parameters**: 420M
- **Max Parameters**: 1.1B  
- **Context Length**: 4K (demo), up to 128K (supported)
- **Format**: SafeTensors (`.safetensors`) for secure model loading
- **Tokenizer**: Llama 3.1 JSON-based tokenizer (no `.model` file)
- **Architecture**: 24 layers, 1024 hidden size, 16 heads
- **Position Encoding**: RoPE with linear scaling
- **Attention**: Efficient long-context attention

## Architecture

The Arbor architecture uses expandable feed-forward networks that can grow during training or inference when performance plateaus, allowing the model to adapt its capacity to the complexity of the task.
