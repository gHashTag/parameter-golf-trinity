"""
Trinity φ-Physics → Model Hyperparameters

42 φ-formulas for optimal hyperparameters in 16MB models.
"""

import mlx.core as mx
import math

# ==============================================================================
# φ-CONSTANTS (from Trinity Physics)
# ==============================================================================

PHI = 1.6180339887498948482  # Golden ratio
ALPHA_PHI = PHI**(-3) / 2  # = 0.118034 ≈ α_s(m_Z) at electroweak scale

# φ-based scaling factors for different physics sectors
TRINITY_SCALES = {
    'gauge': PHI**(-3)/2,  # α_φ = 0.118034 (strong coupling)
    'higgs': 4*PHI**3 * math.e**2 / 1000,  # m_H formula (H01)
    'lepton': 2*math.pi**(-2) * PHI**4 / math.e,  # m_e formula (L01)
    'cosmology': 5*math.pi**(-2) * PHI**2 / math.e,  # Ω_Λ formula (M03)
}

# Fibonacci sequence for attention heads
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

# ==============================================================================
# φ-LR SCHEDULE
# ==============================================================================

def phi_lr_schedule(step: int, max_steps: int, base_lr: float = ALPHA_PHI) -> float:
    """
    φ-based learning rate schedule.
    
    LR starts at α_φ = φ^{-3}/2 = 0.118034 (eigenvalue of A_5 group).
    Decay follows φ^{-1} scaling over φ*27 steps.
    
    Args:
        step: Current training step
        max_steps: Maximum training steps
        base_lr: Base learning rate (default: α_φ)
    
    Returns:
        Learning rate for this step
    """
    phi_steps = max_steps / (PHI * 27)
    decay = PHI ** (-step / phi_steps)
    return base_lr * decay

# ==============================================================================
# GF16 QUANTIZATION
# ==============================================================================

def gf16_quantize_weights(weights: mx.array, scale: float = 1.0) -> mx.array:
    """
    Quantize f32 weights to GF16 (16-bit float with φ-scaling).
    
    Args:
        weights: f32 weight tensor
        scale: φ-optimized scaling factor
    
    Returns:
        GF16 quantized weights (float16)
    """
    scaled = weights * scale
    return scaled.astype(mx.float16)

def gf16_dequantize_weights(gf16_weights: mx.array, scale: float = 1.0) -> mx.array:
    """
    Dequantize GF16 weights back to f32.
    
    Args:
        gf16_weights: GF16 quantized weights (float16)
        scale: φ-optimized scaling factor
    
    Returns:
        f32 weights
    """
    return gf16_weights.astype(mx.float32) / scale

def compute_phi_scale(weights: mx.array) -> float:
    """
    Compute φ-optimized scaling factor for log-normal weights.
    
    For log-normal weights, scale = 1 / (std * φ^(-0.5))
    
    Args:
        weights: Weight tensor
    
    Returns:
        Scaling factor
    """
    mean = mx.mean(weights)
    std = mx.sqrt(mx.mean((weights - mean) ** 2))
    return 1.0 / (std * (PHI ** -0.5))

# ==============================================================================
# TRINITY WEIGHT INITIALIZATION
# ==============================================================================

def trinity_weight_init(shape: tuple, sector: str = 'gauge') -> mx.array:
    """
    Initialize weights using Trinity φ-physics scaling.
    
    Args:
        shape: Tensor shape
        sector: Physics sector ('gauge', 'higgs', 'lepton', 'cosmology')
    
    Returns:
        Initialized weights
    """
    std = TRINITY_SCALES.get(sector, ALPHA_PHI)
    return mx.random.normal(shape, dtype=mx.float32) * std

# ==============================================================================
# FIBONACCI ATTENTION HEADS
# ==============================================================================

def fibonacci_attention_heads(d_model: int) -> int:
    """
    Select number of attention heads from Fibonacci sequence.
    
    Chooses closest Fibonacci number to sqrt(d_model) / 4.
    
    Args:
        d_model: Model dimensionality
    
    Returns:
        Number of attention heads
    """
    target = math.sqrt(d_model) / 4
    # Find closest Fibonacci number
    closest = min(FIBONACCI, key=lambda x: abs(x - target))
    return max(1, closest)  # At least 1 head

# ==============================================================================
# φ-OPTIMIZER FACTORS
# ==============================================================================

def phi_optimizer_factors(matrix_lr: float = 0.04, scalar_lr: float = 0.04) -> tuple:
    """
    Compute φ-optimized learning rates for different parameter types.
    
    Returns:
        (matrix_lr, scalar_lr, embed_lr, tied_embed_lr)
    """
    # Matrix parameters: scale by φ^{-2} = 0.382
    phi_matrix_lr = matrix_lr * (PHI ** -2)
    
    # Scalar parameters: scale by φ^{-1} = 0.618
    phi_scalar_lr = scalar_lr / PHI
    
    # Embedding parameters: scale by α_φ = 0.118
    phi_embed_lr = 0.05 * (ALPHA_PHI / 0.05)  # Scale from baseline
    
    # Tied embeddings: very small initialization
    phi_tied_embed_lr = phi_embed_lr / PHI
    
    return phi_matrix_lr, phi_scalar_lr, phi_embed_lr, phi_tied_embed_lr
