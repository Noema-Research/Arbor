## 📚 Documentation Index

### 🏗️ **Architecture & Implementation**
- [`README.md`](README.md) - Main documentation with architecture overview
- [`SCALING_GUIDE.md`](SCALING_GUIDE.md) - Complete scaling methodology (799M → 400B)
- [`DEPLOYMENT_COMPLETE.md`](DEPLOYMENT_COMPLETE.md) - Implementation status checklist

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
- [`configs/post_training_instruct.yaml`](configs/post_training_instruct.yaml) - Post-training configuration

### 🧪 **Utilities & Scripts**
- [`train.py`](train.py) - Main training entry point
- [`post_train.py`](post_train.py) - Post-training entry point
- [`utils/tokenizer_utils.py`](utils/tokenizer_utils.py) - Hermes-4-405B tokenizer utilities

---

## 🎯 Quick Navigation

| **What You Want** | **Go Here** |
|-------------------|-------------|
| 🚀 **Quick Start** | [`README.md`](README.md#-quick-start) |
| 🏢 **Enterprise Deployment** | [`ENTERPRISE_DEPLOYMENT.md`](ENTERPRISE_DEPLOYMENT.md) |
| 🎭 **Multimodal AI** | [`MULTIMODAL_ROADMAP.md`](MULTIMODAL_ROADMAP.md) |
| 📈 **Scaling to 200B-400B** | [`SCALING_GUIDE.md`](SCALING_GUIDE.md) |
| ✅ **Implementation Status** | [`DEPLOYMENT_COMPLETE.md`](DEPLOYMENT_COMPLETE.md) |
| 🔧 **Training Configuration** | [`configs/training_config.yaml`](configs/training_config.yaml) |
| 🎛️ **Enterprise Config** | [`configs/arbor_enterprise_scale.yaml`](configs/arbor_enterprise_scale.yaml) |
| 🎭 **Multimodal Config** | [`configs/multimodal_training_config.yaml`](configs/multimodal_training_config.yaml) |
| 🤖 **Model Architecture** | [`arbor/modeling/`](arbor/modeling/) |
| 🎯 **Training System** | [`arbor/train/`](arbor/train/) |
| 🌐 **Deployment Scripts** | [`scripts/enterprise_deploy.py`](scripts/enterprise_deploy.py) |

---

**🌳 Arbor: Production-ready adaptive intelligence from 799M to 400B parameters**
