#!/usr/bin/env python3
"""
Multimodal Training Script for Arbor.

This script extends the base Arbor training to support multimodal inputs
including images, audio, and video alongside text.
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from arbor.modeling.multimodal import (
    MultimodalConfig, 
    MultimodalArborTransformer,
    MULTIMODAL_CONFIGS
)
from arbor.modeling.enterprise import EnterpriseArborConfig
from arbor.train.trainer import ArborTrainer
from utils.tokenizer_utils import get_hermes_tokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultimodalTrainingConfig:
    """Configuration for multimodal training."""
    
    # Model configuration
    model_config: EnterpriseArborConfig
    multimodal_config: MultimodalConfig
    
    # Training configuration
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int = 10000
    warmup_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Batch configuration
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 4
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Multimodal-specific
    multimodal_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.1
    
    # Paths
    output_dir: str = "./checkpoints/arbor-multimodal"
    dataset_configs: List[Dict[str, Any]] = None
    
    # HuggingFace integration
    hf_upload_enabled: bool = False
    hf_repository: str = "noema-research/arbor-multimodal"
    hf_token: Optional[str] = None


class MultimodalDataCollator:
    """Data collator for multimodal training."""
    
    def __init__(self, tokenizer, multimodal_config: MultimodalConfig):
        self.tokenizer = tokenizer
        self.multimodal_config = multimodal_config
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate multimodal batch data.
        
        Args:
            batch: List of samples with text, images, audio, video
            
        Returns:
            Collated batch ready for model input
        """
        # Separate different modalities
        text_data = []
        image_data = []
        audio_data = []
        video_data = []
        
        for sample in batch:
            if 'text' in sample:
                text_data.append(sample['text'])
            if 'image' in sample and self.multimodal_config.enable_vision:
                image_data.append(sample['image'])
            if 'audio' in sample and self.multimodal_config.enable_audio:
                audio_data.append(sample['audio'])
            if 'video' in sample and self.multimodal_config.enable_video:
                video_data.append(sample['video'])
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text_data,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        batch_dict = {
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"]
        }
        
        # Process images if present
        if image_data and self.multimodal_config.enable_vision:
            # Stack images into batch tensor
            batch_dict["images"] = torch.stack(image_data)
        
        # Process audio if present  
        if audio_data and self.multimodal_config.enable_audio:
            # Pad and stack audio
            max_audio_length = max(audio.shape[-1] for audio in audio_data)
            padded_audio = []
            for audio in audio_data:
                if audio.shape[-1] < max_audio_length:
                    padding = max_audio_length - audio.shape[-1]
                    audio = torch.nn.functional.pad(audio, (0, padding))
                padded_audio.append(audio)
            batch_dict["audio"] = torch.stack(padded_audio)
        
        # Process video if present
        if video_data and self.multimodal_config.enable_video:
            # Stack video tensors
            batch_dict["videos"] = torch.stack(video_data)
        
        return batch_dict


class MultimodalDatasetLoader:
    """Load and preprocess multimodal datasets."""
    
    def __init__(self, config: MultimodalTrainingConfig):
        self.config = config
        self.tokenizer = get_hermes_tokenizer()
        
    def load_datasets(self) -> Dict[str, Any]:
        """Load configured multimodal datasets."""
        try:
            from datasets import load_dataset, DatasetDict
            from PIL import Image
            import librosa
            import cv2
        except ImportError as e:
            raise ImportError(f"Missing required dependencies: {e}")
        
        datasets = {}
        
        for dataset_config in self.config.dataset_configs or []:
            name = dataset_config.get("name", "unknown")
            source = dataset_config.get("source")
            split = dataset_config.get("split", "train")
            
            logger.info(f"Loading dataset: {name} from {source}")
            
            try:
                if source.startswith("local_"):
                    # Load local dataset
                    dataset = self._load_local_dataset(dataset_config)
                else:
                    # Load from HuggingFace
                    dataset = load_dataset(source, split=split, streaming=True)
                
                # Apply preprocessing
                dataset = self._preprocess_dataset(dataset, dataset_config)
                datasets[name] = dataset
                
                logger.info(f"✅ Loaded dataset: {name}")
                
            except Exception as e:
                logger.warning(f"❌ Failed to load dataset {name}: {e}")
                continue
        
        return datasets
    
    def _load_local_dataset(self, config: Dict[str, Any]):
        """Load dataset from local files."""
        # Implementation for local dataset loading
        logger.warning("Local dataset loading not yet implemented")
        return None
    
    def _preprocess_dataset(self, dataset, config: Dict[str, Any]):
        """Preprocess dataset for multimodal training."""
        
        def preprocess_sample(sample):
            """Preprocess a single sample."""
            processed = {}
            
            # Process text
            if "text_column" in config:
                text = sample.get(config["text_column"], "")
                processed["text"] = str(text)
            
            # Process images
            if "image_column" in config and self.config.multimodal_config.enable_vision:
                image_data = sample.get(config["image_column"])
                if image_data is not None:
                    processed["image"] = self._preprocess_image(image_data)
            
            # Process audio
            if "audio_column" in config and self.config.multimodal_config.enable_audio:
                audio_data = sample.get(config["audio_column"])
                if audio_data is not None:
                    processed["audio"] = self._preprocess_audio(audio_data)
            
            # Process video
            if "video_column" in config and self.config.multimodal_config.enable_video:
                video_data = sample.get(config["video_column"])
                if video_data is not None:
                    processed["video"] = self._preprocess_video(video_data)
            
            return processed
        
        return dataset.map(preprocess_sample)
    
    def _preprocess_image(self, image_data) -> torch.Tensor:
        """Preprocess image data."""
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Convert to PIL if needed
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                image = Image.open(image_data)
            else:
                image = image_data
            
            # Standard image preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            return transform(image)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            # Return zero tensor as fallback
            return torch.zeros(3, 224, 224)
    
    def _preprocess_audio(self, audio_data) -> torch.Tensor:
        """Preprocess audio data."""
        try:
            import librosa
            
            # Load audio if path provided
            if isinstance(audio_data, str):
                audio, sr = librosa.load(audio_data, sr=16000)
            else:
                audio = audio_data
                sr = 16000
            
            # Ensure consistent length (30 seconds max)
            max_length = 30 * sr
            if len(audio) > max_length:
                audio = audio[:max_length]
            elif len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)))
            
            return torch.from_numpy(audio).float()
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            # Return zero tensor as fallback
            return torch.zeros(30 * 16000)
    
    def _preprocess_video(self, video_data) -> torch.Tensor:
        """Preprocess video data."""
        try:
            import cv2
            
            # Load video frames
            if isinstance(video_data, str):
                cap = cv2.VideoCapture(video_data)
                frames = []
                frame_count = 0
                target_frames = 16
                
                while cap.read()[0] and frame_count < target_frames:
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (224, 224))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                        frame_count += 1
                
                cap.release()
                
                # Pad or truncate to 16 frames
                while len(frames) < target_frames:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                
                frames = frames[:target_frames]
                video_tensor = torch.from_numpy(np.array(frames)).float() / 255.0
                video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
                
                return video_tensor
            
        except Exception as e:
            logger.warning(f"Video preprocessing failed: {e}")
            # Return zero tensor as fallback
            return torch.zeros(16, 3, 224, 224)


class MultimodalLoss:
    """Multimodal loss computation."""
    
    def __init__(self, config: MultimodalTrainingConfig):
        self.config = config
        self.text_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.contrastive_loss_fn = torch.nn.CrossEntropyLoss()
    
    def compute_loss(self, model_outputs, batch, labels):
        """Compute multimodal training loss."""
        total_loss = 0.0
        loss_dict = {}
        
        # Text generation loss
        if "logits" in model_outputs:
            logits = model_outputs["logits"]
            text_loss = self.text_loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            total_loss += text_loss * self.config.multimodal_loss_weight
            loss_dict["text_loss"] = text_loss.item()
        
        # Contrastive loss for image-text alignment
        if "images" in batch and "contrastive_logits" in model_outputs:
            contrastive_logits = model_outputs["contrastive_logits"]
            batch_size = contrastive_logits.size(0)
            labels = torch.arange(batch_size, device=contrastive_logits.device)
            
            contrastive_loss = self.contrastive_loss_fn(contrastive_logits, labels)
            total_loss += contrastive_loss * self.config.contrastive_loss_weight
            loss_dict["contrastive_loss"] = contrastive_loss.item()
        
        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict


class MultimodalTrainer(ArborTrainer):
    """Extended trainer for multimodal Arbor models."""
    
    def __init__(self, config: MultimodalTrainingConfig):
        self.multimodal_config = config
        
        # Initialize multimodal model
        self.model = MultimodalArborTransformer(
            config.model_config,
            config.multimodal_config
        )
        
        # Initialize tokenizer
        self.tokenizer = get_hermes_tokenizer()
        
        # Initialize data components
        self.data_collator = MultimodalDataCollator(
            self.tokenizer,
            config.multimodal_config
        )
        
        self.dataset_loader = MultimodalDatasetLoader(config)
        self.loss_fn = MultimodalLoss(config)
        
        # Initialize base trainer
        super().__init__(None)  # We'll set the config manually
        
        logger.info("🎭 Multimodal trainer initialized")
    
    def load_datasets(self):
        """Load multimodal training datasets."""
        logger.info("📊 Loading multimodal datasets...")
        self.datasets = self.dataset_loader.load_datasets()
        
        if not self.datasets:
            logger.warning("⚠️ No datasets loaded, using dummy data")
            self.datasets = self._create_dummy_dataset()
        
        return self.datasets
    
    def _create_dummy_dataset(self):
        """Create dummy multimodal dataset for testing."""
        dummy_data = []
        
        for i in range(100):
            sample = {
                "text": f"This is sample text number {i} for multimodal training.",
            }
            
            # Add dummy image
            if self.multimodal_config.multimodal_config.enable_vision:
                sample["image"] = torch.randn(3, 224, 224)
            
            # Add dummy audio
            if self.multimodal_config.multimodal_config.enable_audio:
                sample["audio"] = torch.randn(30 * 16000)
            
            # Add dummy video
            if self.multimodal_config.multimodal_config.enable_video:
                sample["video"] = torch.randn(16, 3, 224, 224)
            
            dummy_data.append(sample)
        
        return {"dummy": dummy_data}
    
    def train_step(self, batch):
        """Execute a single multimodal training step."""
        self.model.train()
        
        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Compute loss
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        loss, loss_dict = self.loss_fn.compute_loss(outputs, batch, labels)
        
        # Backward pass
        loss.backward()
        
        # Update metrics
        self.update_metrics(loss_dict)
        
        return loss, loss_dict
    
    def update_metrics(self, loss_dict):
        """Update training metrics."""
        for key, value in loss_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save_model(self, output_dir: str):
        """Save multimodal model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.model.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        import json
        config_dict = {
            "model_config": self.multimodal_config.model_config.__dict__,
            "multimodal_config": self.multimodal_config.multimodal_config.__dict__,
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"💾 Multimodal model saved to {output_dir}")


def load_config(config_path: str) -> MultimodalTrainingConfig:
    """Load multimodal training configuration from YAML."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract model configuration
    model_config_dict = config_dict.get("model", {})
    model_config = EnterpriseArborConfig(**model_config_dict)
    
    # Extract multimodal configuration
    multimodal_config_dict = config_dict.get("multimodal", {})
    multimodal_config = MultimodalConfig(**multimodal_config_dict)
    
    # Extract training configuration
    training_config_dict = config_dict.get("training", {})
    
    # Create training config
    training_config = MultimodalTrainingConfig(
        model_config=model_config,
        multimodal_config=multimodal_config,
        **training_config_dict
    )
    
    # Add dataset configurations
    training_config.dataset_configs = config_dict.get("datasets", [])
    
    # HuggingFace configuration
    hf_config = config_dict.get("huggingface", {}).get("upload", {})
    training_config.hf_upload_enabled = hf_config.get("enabled", False)
    training_config.hf_repository = hf_config.get("repository", "noema-research/arbor-multimodal")
    training_config.hf_token = hf_config.get("token", os.getenv("HF_TOKEN"))
    
    return training_config


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train multimodal Arbor model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to multimodal training configuration YAML"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(MULTIMODAL_CONFIGS.keys()),
        help="Use a preset multimodal configuration"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Perform dry run without actual training"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"📄 Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Apply preset if specified
    if args.preset:
        logger.info(f"🎛️ Applying preset: {args.preset}")
        config.multimodal_config = MULTIMODAL_CONFIGS[args.preset]
    
    # Initialize trainer
    logger.info("🎭 Initializing multimodal trainer...")
    trainer = MultimodalTrainer(config)
    
    # Load datasets
    datasets = trainer.load_datasets()
    logger.info(f"📊 Loaded {len(datasets)} datasets")
    
    if args.dry_run:
        logger.info("🧪 Dry run mode - skipping actual training")
        return
    
    # Training loop
    logger.info("🚀 Starting multimodal training...")
    
    try:
        # Simple training loop (in practice, use more sophisticated trainer)
        for epoch in range(config.num_train_epochs):
            logger.info(f"📈 Epoch {epoch + 1}/{config.num_train_epochs}")
            
            # Training steps would go here
            # This is a simplified version
            for step in range(min(10, config.max_steps)):
                # Create dummy batch for demonstration
                dummy_batch = trainer.data_collator([
                    {"text": f"Training sample {step}"}
                ])
                
                loss, loss_dict = trainer.train_step(dummy_batch)
                
                if step % config.logging_steps == 0:
                    logger.info(f"Step {step}: Loss = {loss.item():.4f}")
            
            # Save checkpoint
            if (epoch + 1) % (config.save_steps // 100) == 0:
                save_dir = f"{config.output_dir}/checkpoint-epoch-{epoch + 1}"
                trainer.save_model(save_dir)
        
        # Final save
        trainer.save_model(config.output_dir)
        logger.info("✅ Multimodal training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        trainer.save_model(f"{config.output_dir}/interrupted")
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
