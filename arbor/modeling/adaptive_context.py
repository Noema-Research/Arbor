"""
Adaptive Context Window System for Arbor.

This module implements a dynamic context window that grows/shrinks based on task complexity.
Uses a small router model to analyze input and determine optimal context length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import math


@dataclass
class ContextDecision:
    """Decision from context router about optimal context length."""
    recommended_length: int
    confidence: float
    task_type: str
    complexity_score: float
    reasoning: str


class TaskComplexityRouter(nn.Module):
    """
    Small, fast model that analyzes input to determine task complexity and optimal context length.
    
    This router examines the input text and predicts:
    1. Task type (code, chat, document, reasoning, etc.)
    2. Complexity level (simple, medium, complex, expert)
    3. Recommended context window size
    4. Confidence in the recommendation
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Small embedding for fast processing
        self.embedding_dim = 256
        self.hidden_dim = 512
        
        # Lightweight transformer for analysis
        self.embeddings = nn.Embedding(config.vocab_size, self.embedding_dim)
        self.pos_embeddings = nn.Embedding(config.max_seq_length, self.embedding_dim)
        
        # Fast attention layers (2-3 layers only)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=self.hidden_dim,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)  # Very lightweight
        ])
        
        # Task classification head
        self.task_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, len(self.config.task_types))
        )
        
        # Complexity prediction head
        self.complexity_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1)  # Single complexity score
        )
        
        # Context length recommendation head
        self.context_recommender = nn.Sequential(
            nn.Linear(self.embedding_dim + len(self.config.task_types) + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, len(self.config.context_lengths))
        )
        
        # Hardware awareness
        self.hardware_adapter = nn.Linear(self.config.hardware_features, self.hidden_dim // 4)
        
    def forward(self, input_ids: torch.Tensor, hardware_info: Optional[torch.Tensor] = None) -> ContextDecision:
        """
        Analyze input and determine optimal context window.
        
        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            hardware_info: Hardware capability tensor [batch_size, hardware_features]
            
        Returns:
            ContextDecision with recommended context length and reasoning
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Truncate to router's max length for fast processing
        max_router_len = min(seq_len, self.config.router_max_length)
        input_ids = input_ids[:, :max_router_len]
        
        # Embeddings
        token_embeds = self.embeddings(input_ids)
        pos_ids = torch.arange(max_router_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embeddings(pos_ids)
        
        hidden_states = token_embeds + pos_embeds
        
        # Fast attention processing
        for layer in self.attention_layers:
            hidden_states = layer(hidden_states)
        
        # Global representation (mean pooling)
        global_repr = hidden_states.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Task type prediction
        task_logits = self.task_classifier(global_repr)
        task_probs = F.softmax(task_logits, dim=-1)
        task_type_idx = torch.argmax(task_probs, dim=-1)
        task_confidence = torch.max(task_probs, dim=-1)[0]
        
        # Complexity prediction
        complexity_score = torch.sigmoid(self.complexity_predictor(global_repr)).squeeze(-1)
        
        # Combine features for context recommendation
        features = torch.cat([global_repr, task_probs, complexity_score.unsqueeze(-1)], dim=-1)
        
        # Add hardware information if available
        if hardware_info is not None:
            hw_features = self.hardware_adapter(hardware_info)
            features = torch.cat([features, hw_features], dim=-1)
        
        # Context length recommendation
        context_logits = self.context_recommender(features)
        context_probs = F.softmax(context_logits, dim=-1)
        context_idx = torch.argmax(context_probs, dim=-1)
        context_confidence = torch.max(context_probs, dim=-1)[0]
        
        # Convert to actual context lengths
        recommended_lengths = [self.config.context_lengths[idx.item()] for idx in context_idx]
        task_types = [self.config.task_types[idx.item()] for idx in task_type_idx]
        
        # Create decisions for batch
        decisions = []
        for i in range(batch_size):
            decision = ContextDecision(
                recommended_length=recommended_lengths[i],
                confidence=min(task_confidence[i].item(), context_confidence[i].item()),
                task_type=task_types[i],
                complexity_score=complexity_score[i].item(),
                reasoning=self._generate_reasoning(
                    task_types[i], 
                    complexity_score[i].item(), 
                    recommended_lengths[i]
                )
            )
            decisions.append(decision)
        
        return decisions[0] if batch_size == 1 else decisions
    
    def _generate_reasoning(self, task_type: str, complexity: float, context_length: int) -> str:
        """Generate human-readable reasoning for the context decision."""
        reasoning_parts = []
        
        # Task type reasoning
        if task_type == "code":
            reasoning_parts.append("Detected code/programming task")
        elif task_type == "reasoning":
            reasoning_parts.append("Detected complex reasoning task")
        elif task_type == "document":
            reasoning_parts.append("Detected document processing task")
        elif task_type == "chat":
            reasoning_parts.append("Detected conversational task")
        else:
            reasoning_parts.append(f"Detected {task_type} task")
        
        # Complexity reasoning
        if complexity < 0.3:
            reasoning_parts.append("low complexity")
        elif complexity < 0.7:
            reasoning_parts.append("medium complexity")
        else:
            reasoning_parts.append("high complexity")
        
        # Context length reasoning
        if context_length <= 2048:
            reasoning_parts.append("short context sufficient")
        elif context_length <= 8192:
            reasoning_parts.append("medium context recommended")
        elif context_length <= 32768:
            reasoning_parts.append("long context needed")
        else:
            reasoning_parts.append("maximum context required")
        
        return " → ".join(reasoning_parts)


@dataclass
class AdaptiveContextConfig:
    """Configuration for adaptive context system."""
    
    # Task types the router can identify
    task_types: List[str] = None
    
    # Available context lengths (must match model capabilities)
    context_lengths: List[int] = None
    
    # Router model settings
    router_max_length: int = 512  # Fast analysis on first 512 tokens
    vocab_size: int = 128000
    max_seq_length: int = 131072
    
    # Hardware feature dimensions
    hardware_features: int = 8  # GPU memory, cores, etc.
    
    # Dynamic growth settings
    growth_enabled: bool = True
    min_context: int = 1024
    max_context: int = 131072
    growth_factor: float = 2.0
    
    # Performance thresholds
    memory_threshold: float = 0.85  # Max GPU memory usage
    latency_threshold: float = 2.0   # Max acceptable latency (seconds)
    
    def __post_init__(self):
        if self.task_types is None:
            self.task_types = [
                "chat",          # Conversational
                "code",          # Programming/code
                "reasoning",     # Complex reasoning
                "document",      # Document processing
                "creative",      # Creative writing
                "qa",           # Question answering
                "summarization", # Text summarization
                "translation"    # Language translation
            ]
        
        if self.context_lengths is None:
            self.context_lengths = [
                1024,    # Short context
                2048,    # Small context
                4096,    # Medium context
                8192,    # Large context
                16384,   # Very large context
                32768,   # Ultra large context
                65536,   # Maximum context
                131072   # Extreme context
            ]


class AdaptiveContextManager:
    """
    Manages dynamic context window adaptation during inference.
    
    This system:
    1. Uses router to analyze input and recommend context length
    2. Dynamically adjusts model's context window
    3. Monitors performance and adapts in real-time
    4. Handles hardware constraints gracefully
    """
    
    def __init__(self, config: AdaptiveContextConfig):
        self.config = config
        self.router = TaskComplexityRouter(config)
        self.current_context_length = config.min_context
        self.performance_history = []
        
        # Hardware monitoring
        self.gpu_memory_monitor = GPUMemoryMonitor()
        self.latency_monitor = LatencyMonitor()
        
    def analyze_and_adapt(self, input_ids: torch.Tensor, model) -> Tuple[int, ContextDecision]:
        """
        Analyze input and adapt context window dynamically.
        
        Args:
            input_ids: Input tokens
            model: The main Arbor model
            
        Returns:
            Tuple of (adapted_context_length, decision)
        """
        # Get hardware information
        hardware_info = self._get_hardware_info()
        
        # Route to get recommendation
        decision = self.router(input_ids, hardware_info)
        
        # Check hardware constraints
        feasible_length = self._check_hardware_constraints(
            decision.recommended_length, 
            model.param_count()
        )
        
        # Adapt model context if needed
        if feasible_length != self.current_context_length:
            self._adapt_model_context(model, feasible_length)
            self.current_context_length = feasible_length
        
        # Update decision with final length
        if feasible_length != decision.recommended_length:
            decision.reasoning += f" → hardware limited to {feasible_length}"
            decision.recommended_length = feasible_length
        
        return feasible_length, decision
    
    def _get_hardware_info(self) -> torch.Tensor:
        """Get current hardware capabilities as tensor."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            gpu_memory_free = torch.cuda.memory_reserved(0) / 1e9
            gpu_compute = torch.cuda.get_device_properties(0).multi_processor_count
        else:
            gpu_memory = 0.0
            gpu_memory_free = 0.0
            gpu_compute = 0.0
        
        # Create hardware feature vector
        hardware_features = torch.tensor([
            gpu_memory,           # Total GPU memory
            gpu_memory_free,      # Free GPU memory
            gpu_compute,          # Compute units
            torch.cuda.device_count() if torch.cuda.is_available() else 0,  # GPU count
            1.0,                  # CPU availability (placeholder)
            16.0,                 # RAM GB (placeholder)
            1.0,                  # Network speed (placeholder)
            1.0                   # Storage speed (placeholder)
        ], dtype=torch.float32)
        
        return hardware_features.unsqueeze(0)  # Add batch dimension
    
    def _check_hardware_constraints(self, requested_length: int, model_params: int) -> int:
        """Check if requested context length is feasible given hardware."""
        # Estimate memory requirements
        memory_per_token = model_params * 4 / 1e9  # 4 bytes per param, rough estimate
        estimated_memory = memory_per_token * requested_length
        
        # Get available memory
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            available_memory *= self.config.memory_threshold  # Safety margin
        else:
            available_memory = 8.0  # Assume 8GB for CPU
        
        # Find largest feasible context length
        feasible_lengths = [l for l in self.config.context_lengths if l <= requested_length]
        
        for length in reversed(feasible_lengths):
            estimated_mem = memory_per_token * length
            if estimated_mem <= available_memory:
                return length
        
        # Fallback to minimum context
        return self.config.min_context
    
    def _adapt_model_context(self, model, new_context_length: int):
        """Dynamically adapt the model's context window."""
        print(f"🔄 Adapting context window: {self.current_context_length} → {new_context_length}")
        
        # Update model's position embeddings if needed
        if hasattr(model, 'resize_position_embeddings'):
            model.resize_position_embeddings(new_context_length)
        
        # Update attention mask handling
        if hasattr(model, 'set_max_position_embeddings'):
            model.set_max_position_embeddings(new_context_length)
        
        # Clear attention caches
        if hasattr(model, 'clear_attention_cache'):
            model.clear_attention_cache()


class GPUMemoryMonitor:
    """Monitor GPU memory usage for context adaptation."""
    
    def __init__(self):
        self.memory_history = []
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage as fraction."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            usage = allocated / total
            self.memory_history.append(usage)
            return usage
        return 0.0
    
    def predict_memory_for_context(self, context_length: int, model_size: int) -> float:
        """Predict memory usage for a given context length."""
        # Simplified estimation
        base_memory = model_size * 4 / 1e9  # Model parameters in GB
        context_memory = context_length * model_size * 4 / (1e9 * 1000)  # Rough estimate
        return base_memory + context_memory


class LatencyMonitor:
    """Monitor inference latency for context adaptation."""
    
    def __init__(self):
        self.latency_history = []
    
    def measure_latency(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start_time
        self.latency_history.append(latency)
        return result, latency
    
    def get_average_latency(self, window: int = 10) -> float:
        """Get average latency over recent window."""
        if not self.latency_history:
            return 0.0
        recent = self.latency_history[-window:]
        return sum(recent) / len(recent)
