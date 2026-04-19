"""
Trinity Constants Module

Golden ratio constants derived from Trinity physics paper.
One identity φ²+φ⁻²=3 generates 42 physical formulas with Δ<0.1%.
"""

import math

# ==============================================================================
# FUNDAMENTAL φ CONSTANTS
# ==============================================================================

PHI = 1.6180339887498948482  # Golden ratio
IPHI = PHI - 1                # φ⁻¹ = 0.618...
PHI2 = PHI ** 2               # φ² = 2.618... (= φ+1)
ALPHA_PHI = PHI**(-3) / 2     # 0.118034 ≈ α_s(m_Z) PDG2024

# Verify Trinity Identity
assert abs(PHI**2 + PHI**(-2) - 3.0) < 1e-12, "Trinity identity φ²+φ⁻²=3 failed"

# ==============================================================================
# FIBONACCI SEQUENCE FOR ARCHITECTURE DIMENSIONS
# ==============================================================================

# F_n = (φ^n - ψ^n)/√5 where ψ = -1/φ
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]

def nearest_fib(n: int) -> int:
    """Find nearest Fibonacci number to n."""
    return min(FIB, key=lambda f: abs(f - n))

# ==============================================================================
# TRINITY SECTOR SCALES
# ==============================================================================

TRINITY_SECTORS = {
    'gauge': ALPHA_PHI,                    # α_φ = 0.118034 (strong coupling)
    'higgs': ALPHA_PHI * IPHI,            # α_φ × φ⁻¹ (Higgs boson)
    'lepton': ALPHA_PHI * IPHI**2,       # α_φ × φ⁻² (leptons)
    'cosmology': ALPHA_PHI * (IPHI**3),   # α_φ × φ⁻³ (Ω_Λ)
}

# ==============================================================================
# MODEL DIMENSIONS (FIBONACCI OR φ-MULTIPLES)
# ==============================================================================

# Based on 16MB budget and Trinity Identity
VOCAB_SIZE = 50257   # GPT-2 BPE (tied embeddings)
D_MODEL = nearest_fib(144)   # Fib(12) = 144
N_HEADS = nearest_fib(8)     # Fib(6) = 8
D_HEAD = D_MODEL // N_HEADS    # 18
D_FFN = int(D_MODEL * PHI)     # 144 × 1.618 ≈ 232 (near Fib(13)=233)
N_LAYERS = 9                   # log_φ(budget/layer_size)

print(f"Trinity Model Dimensions:")
print(f"  d_model  = {D_MODEL} (Fibonacci)")
print(f"  n_heads  = {N_HEADS} (Fibonacci)")
print(f"  d_head   = {D_HEAD}")
print(f"  d_ffn    = {D_FFN} (φ×d_model)")
print(f"  n_layers = {N_LAYERS}")

# ==============================================================================
# PARAMETER COUNT ESTIMATION
# ==============================================================================

# Embeddings (tied with output)
params_emb = VOCAB_SIZE * D_MODEL

# Attention per layer
# qkv_proj: 3 × d_model × d_model
# out_proj: d_model × d_model
params_attn_layer = 4 * (D_MODEL * D_MODEL)

# FFN per layer
# gate: d_model × d_ffn
# up: d_ffn × d_model
params_ffn_layer = 2 * (D_MODEL * D_FFN)

# Layer norm (2 per layer)
params_ln_layer = 4 * D_MODEL

# Total per layer
params_layer = params_attn_layer + params_ffn_layer + params_ln_layer

# Total parameters
params_total = params_emb + (N_LAYERS * params_layer)

print(f"\nTrinity Model Parameters:")
print(f"  embeddings: {params_emb:,}")
print(f"  per layer:  {params_layer:,}")
print(f"  total:      {params_total:,}")
print(f"  fp16 size:  {params_total * 2:,} bytes = {params_total * 2 / (1024**2):.2f} MB")
print(f"  GF16 target: 16.0 MB")
