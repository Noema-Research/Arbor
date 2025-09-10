"""
Arbor-o1: The Living AI
Core package initialization
"""

__version__ = "0.1.0"
__author__ = "Arbor Research Team"
__email__ = "research@arbor.ai"

from arbor.modeling.model import ArborTransformer, create_arbor_model
from arbor.growth.manager import GrowthManager
from arbor.train.train_loop import Trainer

__all__ = [
    "ArborTransformer",
    "create_arbor_model", 
    "GrowthManager",
    "Trainer",
]
