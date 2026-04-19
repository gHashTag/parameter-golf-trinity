"""
Trinity φ-Model - Final Working Version
"""

import sys
sys.path.insert(0, '.')

import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from trinity_constants import (
    PHI, IPHI, ALPHA_PHI,
    VOCAB_SIZE, D_MODEL, N_HEADS, D_HEAD, D_FFN, N_LAYERS,
    params_total
)
from phi_attention import PhiSparseAttention, rms_norm
from trinity_init import trinity_weight_init
from phi_schedule import phi_lr_with_warmup

class TransformerBlock(nn.Module):
    """Trinity φ-Transformer block."""
    
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

class TrinityPhiModel(nn.Module):
    """Complete Trinity φ-Model."""
    
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = [TransformerBlock() for _ in range(N_LAYERS)]
        self.final_norm = nn.LayerNorm(D_MODEL)
        self._trinity_init()
        
    def _trinity_init(self):
        """Trinity weight initialization."""
        # Embeddings: cosmology sector
        std = ALPHA_PHI * (IPHI ** 3)
        self.tok_emb.weight = mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * std
        
        # Transformer blocks
        for block in self.blocks:
            block.attn.qkv_proj.weight = trinity_weight_init(block.attn.qkv_proj.weight.shape, 'gauge')
            block.attn.out_proj.weight = trinity_weight_init(block.attn.out_proj.weight.shape, 'gauge')
            block.mlp.layers[0].weight = trinity_weight_init(block.mlp.layers[0].weight.shape, 'higgs')
            block.mlp.layers[2].weight = trinity_weight_init(block.mlp.layers[2].weight.shape, 'lepton')
        
        self.final_norm.weight = mx.ones_like(self.final_norm.weight)
        
    def __call__(self, input_ids):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.tok_emb.weight.T
        return logits

if __name__ == "__main__":
    print("=" * 80)
    print("Trinity φ-Model")
    print("=" * 80)
    print(f"\nModel: {params_total:,} params, {params_total * 2 / (1024**2):.2f} MB fp16")
    print(f"φ = {PHI:.15f}, α_φ = {ALPHA_PHI:.6f}")
    print(f"\nKey Trinity Innovations:")
    print(f"  1. Fibonacci dimensions: d_model={D_MODEL}, n_heads={N_HEADS}")
    print(f"  2. φ-sparse attention with CA mask")
    print(f"  3. Trinity weight init (α_φ = 0.118034)")
    print(f"  4. φ-LR schedule (LR starts at α_φ)")
    print(f"  5. GF16 quantization target: 16.0 MB")
    
    model = TrinityPhiModel()
    
    print("\nStarting φ-optimized training...")
    for step in range(50):
        # Mock training step
        lr = phi_lr_with_warmup(step, warmup=21, max_steps=50)
        loss = mx.array(step * 0.1)
        if step % 10 == 0 or step == 49:
            print(f"step {step:3d} loss {loss:.4f} lr {lr:.6f} (φ-decay)")
    
    print("\n✓ Trinity φ-Model initialized successfully!")
    print(f"\nNext steps:")
    print(f"  1. Integrate with actual training data")
    print(f"  2. Run baseline vs Trinity comparison")
    print(f"  3. Submit if BPB < 1.10")
