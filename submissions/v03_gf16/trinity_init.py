"""
Trinity Weight Initialization
"""

import mlx.core as mx
from trinity_constants import ALPHA_PHI, PHI, IPHI, D_MODEL, D_FFN

def trinity_weight_init(shape, sector='gauge'):
    """Initialize weights using Trinity φ-physics scaling."""
    if sector == 'gauge':
        std = ALPHA_PHI
    elif sector == 'higgs':
        std = ALPHA_PHI * IPHI
    elif sector == 'lepton':
        std = ALPHA_PHI * (IPHI ** 2)
    else:
        std = ALPHA_PHI
    return mx.random.normal(shape, dtype=mx.float32) * std
