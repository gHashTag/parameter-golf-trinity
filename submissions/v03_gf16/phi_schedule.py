"""
φ-LR Schedule

LR(t) = α_φ · φ^(−t / τ)    where τ = T/(φ·27)

Number 27 = 3³ = (φ²+φ⁻²)³ from Trinity Identity.
τ — characteristic decay time in φ-units.

Physical image: exponential decay of coupling constant
under renormalization group flow.
"""

import math
from trinity_constants import ALPHA_PHI, PHI

def phi_lr_schedule(step: int, max_steps: int, base_lr: float = ALPHA_PHI) -> float:
    """
    φ-based learning rate schedule.
    
    LR starts at α_φ = φ^(-3)/2 = 0.118034 (eigenvalue of A_5 group).
    Decay follows φ^(-1) scaling over φ*27 steps.
    
    Args:
        step: Current training step
        max_steps: Maximum training steps
        base_lr: Base learning rate (default: α_φ)
    
    Returns:
        Learning rate for this step
    """
    # Characteristic time τ in φ-units
    # 27 = 3³ from Trinity Identity φ²+φ⁻²=3
    tau = max_steps / (PHI * 27)
    
    # φ^(-1) decay
    decay = PHI ** (-step / tau)
    
    return base_lr * decay

def phi_lr_with_warmup(step: int, warmup: int = 21, max_steps: int = 10000) -> float:
    """
    φ-LR schedule with Fibonacci warmup.
    
    Warmup: linear to α_φ over Fib(7)=21 steps.
    This matches the Fibonacci sequence of dimensions.
    
    Args:
        step: Current training step
        warmup: Warmup steps (default: Fib(7)=21)
        max_steps: Maximum training steps
        
    Returns:
        Learning rate for this step
    """
    if step < warmup:
        # Linear warmup to α_φ
        return ALPHA_PHI * (step / warmup)
    
    # φ-decay after warmup
    return phi_lr_schedule(step - warmup, max_steps - warmup)

def phi_lr_print_schedule(max_steps: int = 10000, steps: list = [0, 21, 500, 1000, 2000, 5000, 10000]):
    """
    Print φ-LR schedule for verification.
    """
    print("φ-LR Schedule:")
    print(f"  base_lr (α_φ) = {ALPHA_PHI:.6f}")
    print(f"  τ = {max_steps / (PHI * 27):.2f} steps")
    print()
    for s in steps:
        if s <= max_steps:
            lr = phi_lr_with_warmup(s, max_steps=max_steps)
            print(f"  step={s:6d} → lr={lr:.8f}")

if __name__ == '__main__':
    phi_lr_print_schedule()
