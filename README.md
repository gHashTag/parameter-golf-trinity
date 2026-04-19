# Parameter Golf Trinity

Trinity Cognitive Stack's entry into OpenAI Parameter Golf.

## 🎯 GF16 Advantage
- 50% compression (16KB → 8KB for 64x64)
- Log-normal distribution matches neural weights
- ~0.00003 roundtrip error

## 📁 Structure
```
submissions/v03_gf16/  ← GF16 quantization
docs/COMPETITOR_INTEL.md
```

## 🏆 Current SOTA
1.0810 BPB — bigbag (SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT)

## 📅 Status
- ✅ GF16 prototype working
- 🔄 MLX smoke test in progress
- 📋 RunPod grant: manual submission required
