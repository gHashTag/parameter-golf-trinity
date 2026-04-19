"""
JEPA-T Predictor
"""

import mlx.core as mx
import mlx.nn as nn
from phi_attention import PhiSparseAttention, rms_norm
from trinity_constants import D_MODEL, D_FFN, PHI

class TransformerBlock(nn.Module):
    """φ-Transformer block."""
    
    def __init__(self):
        super().__init__()
        self.attn = PhiSparseAttention()
        self.mlp = nn.Sequential(
            nn.Linear(D_MODEL, D_FFN),
            nn.GELU(),
            nn.Linear(D_FFN, D_MODEL)
        )
        
    def __call__(self, x):
        x = x + self.attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x

class JEPATemporalPredictor(nn.Module):
    """JEPA-T predictor (8MB encoder + 8MB predictor)."""
    
    def __init__(self, n_encoder_layers=9, n_predictor_layers=4):
        super().__init__()
        self.encoder = nn.Sequential(*[TransformerBlock() for _ in range(n_encoder_layers)])
        self.predictor = nn.Sequential(
            nn.Linear(D_MODEL, int(D_MODEL * PHI)),
            nn.GELU(),
            nn.Linear(int(D_MODEL * PHI), D_MODEL)
        )
        
    def __call__(self, x_context, x_target):
        z_ctx = self.encoder(x_context)
        with mx.stopgradient():
            z_tgt = self.encoder(x_target)
        z_pred = self.predictor(z_ctx[:, -1:])
        loss = mx.mean((z_pred - z_tgt[:, :1]) ** 2)
        return loss

def compute_phi_scale(z):
    """Compute φ-optimized scaling for latent representation."""
    return 1.0 / (mx.std(z) * (PHI ** -0.5))
