# Parameter Golf — Competitor Intelligence (Apr 19, 2026)

## Leaderboard Snapshot

| Rank | Score | Author | Key Techniques |
|------|-------|--------|----------------|
| 1 | 1.0810 | bigbag | SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal TTT |
| 2 | 1.0822 | aryanbhosale | SP8192 + parallel residuals + score-first TTT |
| 3 | 1.0828 | dexhunter | SP8192 + QK-Gain 5 + legal score-first TTT |
| 4 | 1.0835 | Robby Sneiderman | SP8192 + parallel residuals + Hessian-aware SDClip + progressive recurrence |
| 5 | 1.0856 | Kevin Clark | SP8192 + GPTQ embeddings + depth recurrence + SDClip + looped layers |

## Standard SOTA Stack

All top-5 submissions use:

### Quantization
- **Int6 QAT** - Quantization-aware training with STE (Straight-Through Estimator)
- **Per-row scaling** - Different scales for each weight matrix row
- **Tied embeddings** - Remain in FP16 to avoid quantization penalty

### Optimization
- **Muon optimizer** - Newton-Schulz momentum from OpenAI
- **Weight decay**: 0.01-0.05
- **Momentum warmup**: 0.92 → 0.99 over ~1500 steps
- **AdamW**: For embedding/scalar parameters only

### Architecture
- **MLP 3x expansion** - Hidden dim 1536 (vs 1024 baseline) - *single largest contributor*
- **SmearGate** - 512 params, bigram-level context at embedding
- **BigramHash** - 524K params, (prev*31+curr)%4096 hash table
- **GPTQ-lite** - Group-wise quantization calibration
- **GPTQ embeddings** - Quantized embedding lookup

### Training
- **SP8192** - 8192 batch size for efficiency
- **Depth recurrence / Parallel residuals** - Reuse activations across layers
- **Sliding Window Eval** - Stride=64 for efficient validation
- **Hessian-aware SDClip** - Spectral-based gradient clipping
- **SWA** - Stochastic Weight Averaging (last 50% of checkpoints)

### Attention
- **QK-Gain 5.0-5.25** - Scale key/query matrices
- **FlashAttention 3** - Optimized attention computation
- **8-head/4-KV GQA** - Grouped-query attention
- **Partial RoPE** - Rotary position encoding
- **XSA4** - Cross-attention with sparse access

### Compression
- **zstd-22** - ~5% better than zlib-9
- **lzma** - Alternative compressor (some use this)

### Regularization
- **EMA 0.997** - Exponential moving average
- **VRL 128** - Variational regularization layer
- **VE128** - Variational embedding
- **Legal TTT / Score-first TTT** - Test-time training strategies

## СЛАБЫЕ МЕСТА (Trinity Opportunities)

### 1. ❌ GF16 Quantization - **UNEXPLORED**
- No top-20 submission uses golden-ratio based encoding
- GF16 creates log-normal distribution (better for zstd-22)
- Expected gain: 0.005-0.010 BPB

### 2. ❌ BitNet b1.58 - **UNEXPLORED**
- Ternary weights {-1, 0, +1} = 1.58 bit/weight
- ~67M params fit in 16MB vs 21M for int6
- No submissions use ternary quantization

### 3. ❌ φ-based Inductive Biases - **UNEXPLORED**
- φ-attention weighting
- Fibonacci attention heads: [1,1,2,3,5,8,13] = 33
- Sacred bottleneck: hidden_dim = 377
- φ-based weight initialization

### 4. ❌ VSA Binding - **UNEXPLORED**
- Vector Symbolic Architecture for parameters
- Hyperdimensional encoding via zig-hdc

### 5. ❌ State Space Models - **UNEXPLORED**
- Mamba/RWKV instead of Transformer
- No submissions use SSM architecture

### 6. ❌ φ-based Pre-initialization - **UNEXPLORED**
- All use random Xavier/Glorot init
- φ-scale by layer depth could improve convergence

### 7. ❌ Knowledge Distillation - **RARELY USED**
- 10-minute distillation from teacher to student
- φ-based temperature scaling

## Threshold for New SOTA

- **Statistical significance**: +0.005 nats improvement (p < 0.01)
- **3-seed averaging**: Variance must be < 0.01
- **Training time**: < 10 minutes on 8×H100
- **Model size**: ≤ 16MB (code + compressed weights)

## Trinity Attack Plan

| Day | Branch | Technique | Target BPB |
|-----|--------|-----------|------------|
| 1 | v01_baseline | reproduce | 1.2244 |
| 2-3 | v01_sota | SOTA stack | 1.14 |
| 4-5 | v02_bitnet | ternary b1.58 | 1.12 |
| 6-7 | v03_gf16 | golden-float ⭐ | 1.11 |
| 8-9 | v04_hslm | φ + Fib heads | 1.105 |
| 10 | v05_final | ensemble | 1.10 |
| 11 | submit | 3 seeds + writeup | ⚡ |

## Key Files to Study

- `train_gpt.py` - Reference implementation
- `records/` - Official submission details
- Competition README - Full leaderboard and rules

---
*Last updated: 2026-04-19*
