# Parameter Golf - Baseline Analysis

## Target
**BPB**: 1.2244 (from competition leaderboard)

## Local Test Results
**Status**: Cannot run baseline locally - CUDA required

### Attempted
```bash
# 1. Setup venv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt  # ✅ Success

# 2. Run training
python train_gpt.py
# ❌ RuntimeError: CUDA is required (line 753)
```

### Train Script Analysis
- `train_gpt.py` explicitly checks `torch.cuda.is_available()`
- Raises `RuntimeError("CUDA is required")` if not present
- Uses `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
- Designed for 8×H100 training, not local CPU

## Next Steps
1. **RunPod Grant**: Submit application for 1000 H100 credits
   - URL: https://openai.com/index/parameter-golf/ → Infrastructure Grants
   - Angle: "Novel GF16 (golden-ratio float) quantization"

2. **Remote Training**: All training will be done on RunPod 8×H100
   - Baseline reproduction (verify 1.2244 BPB)
   - GF16 experiment (target 1.11 BPB)
   - BitNet ternary (target 1.12 BPB)
   - HSLM ensemble (target 1.10 BPB)

3. **Local Development**: Work on code structure without full training
   - GF16 quantization integration
   - Sacred geometry patterns
   - φ-based weight initialization

## Competition Context
- **Current SOTA**: 1.0810 BPB (bigbag)
- **Submission deadline**: April 30, 2026
- **Training limit**: < 10 minutes on 8×H100
- **Model limit**: ≤ 16MB (code + compressed weights)

## Files
- `train_gpt.py` - Main training script (CUDA required)
- `requirements.txt` - Dependencies (all installed)
- `.venv/` - Python virtual environment (torch 2.11.0)

---
*Created: 2026-04-19*
