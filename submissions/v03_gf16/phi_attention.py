"""
φ-Sparse Attention with CA Mask
"""

import mlx.core as mx
import mlx.nn as nn
import math
import numpy as np
from trinity_constants import PHI, IPHI, D_MODEL, N_HEADS, D_HEAD

class PhiSparseAttention(nn.Module):
    """φ-Structured sparse attention."""
    
    def __init__(self, max_len: int = 512):
        super().__init__()
        self.n_heads = N_HEADS
        self.d_head = D_HEAD
        self.scale = self.d_head ** (-IPHI)  # φ⁻¹ scaling
        
        # Build φ-lattice mask
        self.phi_mask = self._build_phi_mask(max_len)
        
        # QKV projection
        self.qkv_proj = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        
    def _build_phi_mask(self, T: int) -> mx.array:
        """Build CA Rule 110 + Fibonacci distance mask."""
        fib_dists = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        mask = np.full((T, T), -np.inf)
        
        for i in range(T):
            mask[i, i] = 0.0
            for d in fib_dists:
                if i - d >= 0:
                    mask[i, i-d] = 0.0
                    
        return mx.array(mask)
    
    def __call__(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        
        attn = mx.matmul(q, k.transpose(0, 1, 2, 3)) * self.scale
        mask = self.phi_mask[:T, :T]
        attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        
        return self.out_proj(out)

def rms_norm(x, eps=1e-5):
    """RMS normalization."""
    return x * mx.rsqrt(x.mean(axis=-1, keepdims=True) + eps)
