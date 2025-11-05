"""
CEGE: Conversational Emotion Graph Evolution
A novel architecture for Emotion Recognition in Conversation (ERC)
"""

from .model import (
    CEGEModel,
    SequentialContextEncoder,
    TemporalMemoryModule,
    DynamicGraphBuilder,
    TemporalGCNLayer,
    TemporalAttention,
    MaskedNLLLoss
)

from .dataloader import IEMOCAPDataset, get_IEMOCAP_loaders

__version__ = '1.0.0'
__all__ = [
    'CEGEModel',
    'SequentialContextEncoder',
    'TemporalMemoryModule',
    'DynamicGraphBuilder',
    'TemporalGCNLayer',
    'TemporalAttention',
    'MaskedNLLLoss',
    'IEMOCAPDataset',
    'get_IEMOCAP_loaders'
]
