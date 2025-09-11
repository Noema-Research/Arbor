## 📚 Documentation Index

### 🏗️ **Architecture & Implementation**
-| 🚀 **Quick Start** | [`README.md`](README.md#-quick-start) |
| 📚 **Complete Documentation** | [`DOCUMENTATION.md`](DOCUMENTATION.md) |
| 🤖 **Agentic AI Guide** | [`AGENTIC_AI_GUIDE.md`](AGENTIC_AI_GUIDE.md) |
| 🤖 **Agentic AI Examples** | [`examples/agent_usage.py`](examples/agent_usage.py) |
| 🏗️ **Layer Growth Demo** | [`examples/layer_growth_demo.py`](examples/layer_growth_demo.py) |README.md`](README.md) - Main documentation with architecture overview
- [`DOCUMENTATION.md`](DOCUMENTATION.md) - Complete technical documentation
- [`SCALING_GUIDE.md`](SCALING_GUIDE.md) - Complete scaling methodology (799M → 400B)
- [`LAYER_GROWTH_IMPLEMENTATION.md`](LAYER_GROWTH_IMPLEMENTATION.md) - Dynamic layer growth (24→64 layers)
- [`AGENTIC_AI_GUIDE.md`](AGENTIC_AI_GUIDE.md) - Comprehensive agentic capabilities guide
- [`DEPLOYMENT_COMPLETE.md`](DEPLOYMENT_COMPLETE.md) - Implementation status checklist

### 🤖 **Agentic AI Capabilities**
- [`arbor/agents/`](arbor/agents/) - Complete agentic system with tool calling
- [`arbor/agents/base_agent.py`](arbor/agents/base_agent.py) - Core agent with reasoning capabilities
- [`arbor/agents/code_executor.py`](arbor/agents/code_executor.py) - Secure code execution (Python, Bash, Docker)
- [`arbor/agents/mcp_client.py`](arbor/agents/mcp_client.py) - Model Context Protocol integration
- [`inference_agent.py`](inference_agent.py) - Interactive agentic interface
- [`examples/agent_usage.py`](examples/agent_usage.py) - Agent usage examples

### 🚀 **Enterprise Deployment**
- [`ENTERPRISE_DEPLOYMENT.md`](ENTERPRISE_DEPLOYMENT.md) - Complete 200B-400B deployment guide
- [`deploy.sh`](deploy.sh) - One-command deployment automation script
- [`scripts/enterprise_deploy.py`](scripts/enterprise_deploy.py) - Enterprise deployment CLI

### 🎭 **Multimodal Extensions**
- [`MULTIMODAL_ROADMAP.md`](MULTIMODAL_ROADMAP.md) - Vision, audio, video architecture plans
- [`arbor/modeling/multimodal.py`](arbor/modeling/multimodal.py) - Multimodal model architecture
- [`configs/multimodal_training_config.yaml`](configs/multimodal_training_config.yaml) - Multimodal training config

### 🔧 **Core Implementation Files**

#### Enterprise Architecture (200B-400B Parameters)
- [`arbor/modeling/enterprise.py`](arbor/modeling/enterprise.py) - Enterprise-scale model implementations
- [`arbor/training/distributed.py`](arbor/training/distributed.py) - Distributed training framework  
- [`arbor/inference/enterprise_inference.py`](arbor/inference/enterprise_inference.py) - High-performance inference

#### Core Architecture (Research → Production)
- [`arbor/modeling/model.py`](arbor/modeling/model.py) - Core Arbor transformer architecture
- [`arbor/modeling/block.py`](arbor/modeling/block.py) - Transformer blocks with dynamic growth
- [`arbor/modeling/layers.py`](arbor/modeling/layers.py) - Expandable FFN and attention layers

#### Training & Post-Training
- [`arbor/train/trainer.py`](arbor/train/trainer.py) - Main training orchestration
- [`arbor/train/post_trainer.py`](arbor/train/post_trainer.py) - Post-training system (LoRA, instruction tuning)
- [`arbor/data/context_router.py`](arbor/data/context_router.py) - Adaptive context routing system

### ⚙️ **Configuration Files**
- [`configs/training_config.yaml`](configs/training_config.yaml) - Standard training configuration
- [`configs/arbor_enterprise_scale.yaml`](configs/arbor_enterprise_scale.yaml) - 400B parameter enterprise config
- [`configs/arbor_layer_growth.yaml`](configs/arbor_layer_growth.yaml) - Layer growth demonstration config
- [`configs/agent_config.yaml`](configs/agent_config.yaml) - Agentic AI configuration
- [`configs/post_training_instruct.yaml`](configs/post_training_instruct.yaml) - Post-training configuration

### 🧪 **Examples & Demos**
- [`examples/layer_growth_demo.py`](examples/layer_growth_demo.py) - Dynamic layer growth demonstration
- [`examples/agent_usage.py`](examples/agent_usage.py) - Agentic AI examples and usage patterns
- [`examples/basic_training.py`](examples/basic_training.py) - Simple training script
- [`examples/inference_demo.py`](examples/inference_demo.py) - Generation examples

### 🧪 **Utilities & Scripts**
- [`train.py`](train.py) - Main training entry point
- [`post_train.py`](post_train.py) - Post-training entry point
- [`inference_agent.py`](inference_agent.py) - Interactive agentic interface
- [`utils/tokenizer_utils.py`](utils/tokenizer_utils.py) - Hermes-4-405B tokenizer utilities

---

## 🎯 Quick Navigation

| **What You Want** | **Go Here** |
|-------------------|-------------|
| 🚀 **Quick Start** | [`README.md`](README.md#-quick-start) |
| 🤖 **Agentic AI** | [`examples/agent_usage.py`](examples/agent_usage.py) |
| �️ **Layer Growth** | [`examples/layer_growth_demo.py`](examples/layer_growth_demo.py) |
| �🏢 **Enterprise Deployment** | [`ENTERPRISE_DEPLOYMENT.md`](ENTERPRISE_DEPLOYMENT.md) |
| 🎭 **Multimodal AI** | [`MULTIMODAL_ROADMAP.md`](MULTIMODAL_ROADMAP.md) |
| 📈 **Scaling to 200B-400B** | [`SCALING_GUIDE.md`](SCALING_GUIDE.md) |
| ✅ **Implementation Status** | [`DEPLOYMENT_COMPLETE.md`](DEPLOYMENT_COMPLETE.md) |
| 🔧 **Training Configuration** | [`configs/training_config.yaml`](configs/training_config.yaml) |
| 🏗️ **Layer Growth Config** | [`configs/arbor_layer_growth.yaml`](configs/arbor_layer_growth.yaml) |
| 🤖 **Agent Configuration** | [`configs/agent_config.yaml`](configs/agent_config.yaml) |
| 🎛️ **Enterprise Config** | [`configs/arbor_enterprise_scale.yaml`](configs/arbor_enterprise_scale.yaml) |
| 🎭 **Multimodal Config** | [`configs/multimodal_training_config.yaml`](configs/multimodal_training_config.yaml) |
| 🤖 **Model Architecture** | [`arbor/modeling/`](arbor/modeling/) |
| 🛠️ **Agent System** | [`arbor/agents/`](arbor/agents/) |
| 🎯 **Training System** | [`arbor/train/`](arbor/train/) |
| 🌐 **Deployment Scripts** | [`scripts/enterprise_deploy.py`](scripts/enterprise_deploy.py) |

---

**🌳 Arbor: Production-ready adaptive intelligence from 799M to 400B parameters**
