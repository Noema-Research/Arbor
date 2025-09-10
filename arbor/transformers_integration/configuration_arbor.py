"""
Hugging Face Transformers configuration for Arbor architecture.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

ARBOR_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class ArborTransformersConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of an Arbor model.
    
    This configuration inherits from PretrainedConfig to be compatible
    with Hugging Face Transformers while supporting Arbor's growth features.
    """
    
    model_type = "arbor"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        # Arbor-specific parameters
        growth_enabled=True,
        max_growth_steps=10,
        growth_factor=1.3,
        min_steps_between_growth=100,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        
        # Arbor-specific config
        self.growth_enabled = growth_enabled
        self.max_growth_steps = max_growth_steps
        self.growth_factor = growth_factor
        self.min_steps_between_growth = min_steps_between_growth

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property  
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
