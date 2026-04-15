# Community Issues & Discourse Summary

## Source: GitHub Issues, Discussions, Reddit (March-April 2026)

---

## Critical Issues Filed

### 1. [#21517](https://github.com/ggml-org/llama.cpp/issues/21517) - Q8_0 4x Slower on Arc Pro B70

**Reporter:** PMZFX (April 6, 2026)  
**Status:** Closed - PR #21527 submitted

**Benchmark Data (Arc Pro B70, Qwen3.5-27B):**

| Quant | Token Gen (t/s) | BW Utilization |
|-------|-----------------|----------------|
| Q4_K_M | 20.56 | 53% |
| Q8_0 | 4.88 | 21% |

**Key Findings:**
- Q8_0 stuck on generic DMMV kernel (iter_stride=64)
- Q4_0 reorder kernel uses iter_stride=512 (8x more work)
- Driver updates don't help (IGC 2.28.4 → 2.30.1 unchanged Q8_0 perf)
- Both SYCL and Vulkan affected equally
- Dual GPU doesn't help - confirmed kernel-level issue

**Fix:** PR #21527 adds Q8_0 to reorder framework. Validation showed 3.1x speedup (4.88 → 15.24 t/s).

---

### 2. [#12318](https://github.com/intel/ipex-llm/issues/12318) - K-Quant Crash on Xe2 iGPU

**Reporter:** lhl (November 3, 2024)  
**Status:** Closed  
**Hardware:** Lunar Lake Arc 140V

```
Sub-group size 8 is not supported on the device
Exception at ggml-sycl.cpp:3164
```

**Reproduction:** Q4_K_M crashes, Q4_0 works fine.

**Workaround:** Use upstream llama.cpp SYCL backend (slower but stable).

---

### 3. [#20776](https://github.com/ggml-org/llama.cpp/issues/20776) - Arc 140T Misdetection

**Reporter:** diegokolling (March 19, 2026)  
**Status:** Open  
**Hardware:** Arrow Lake H, Arc 140T (48GB shared)

**Root Cause:**
- Driver reports `minSubgroupSize = 8`
- Code requires `minSubgroupSize == 16` for INTEL_XE2 classification
- Same driver on Arc 140V reports `minSubgroupSize = 16`

**Impact:** Cooperative matrix completely disabled despite hardware support.

---

## Key Discussions

### [#12570](https://github.com/ggml-org/llama.cpp/discussions/12570) - Arc Status for llama.cpp

**Date:** March 25-28, 2025  
**Participants:** ky438, Rbiessy (Codeplay), NeoZhangJianyu

**Key Quotes:**

> "tg should already be decent" - 0cc4m (llama.cpp collaborator)

> "There are huge performance gaps between k-quant and legacy quant. Some quantizations like IQ4_NL reach only 14% of memory bandwidth utilization." - Community report

> "For BMG, we don't promise to optimize it in time of the marketing." - NeoZhangJianyu

> "If you want to see the best performance on Intel GPU, please try OpenVINO." - NeoZhangJianyu

**Outcomes:**
- Acknowledged poor performance on k-quants
- Planned work on mul_mat_vec_q kernel optimization
- Discussion of DPAS instruction utilization
- Note that community contributors work on this in spare time

---

### [#12805](https://github.com/ggml-org/llama.cpp/discussions/12805) - A750 User Experience

**Date:** April 7-9, 2025  
**User:** codayon (Arch Linux, 8GB VRAM)

**Findings:**
- Ubuntu Vulkan binary worked on Arch Linux
- Q4_K_M slower than expected on 8GB card
- Q4_0 recommended for better performance
- IPEX-LLM provides better VRAM utilization
- Complexity of setup is barrier to entry

**Recommendations from community:**
- Use Qwen2.5-Coder-0.5B-Q8_0 for autocomplete (150+ t/s)
- Qwen2.5-Coder-7B-Q4_0 for chat
- Vulkan more stable than SYCL on Arch

---

## Reddit Discourse

### r/LocalLLaMA - "Intel Arc for LLMs?"

**Key Comments:**
- "Not a lot of kernels for arc so many of the quantized models will be out of reach" (u/shakhal1)
- Arc A770 with 16GB runs models up to 24B with 4-6bit quantization
- oneAPI less mature than CUDA - expect compatibility issues

### r/LocalLLaMA - "llama.cpp 3.1x Q8_0 speedup on Intel Arc GPUs"

**Key Details:**
- PR submitted by AI Agent + user collaboration
- Binary-patched Intel's closed-source IPEX-LLM to validate solution
- IPEX-LLM achieved 61% bandwidth - confirming problem is solvable in software

### r/IntelArc - "Intel ARC for local LLMs"

**User reports:**
- B580 setup issues (unsupported message)
- Even dual A770 (32GB) not enough for 30B at FP16
- No consumer Intel GPU has sufficient VRAM for large models

---

## GitHub Issue #19887 - A770 Inverse Quantization Anomaly

**On A770:** Q8_0 is faster than Q4/Q6  
**On B70:** Q8_0 is 4x slower than Q4

**This is a Xe2/Battlemage regression** - indicates:
- Xe1 optimizations work
- Xe2 memory architecture is different
- Kernel tuning needed for new architecture

---

## Performance Summary Table

Compiled from community benchmarks:

| GPU | Backend | Q4_0 tg | Q4_K_M tg | Q8_0 tg | Notes |
|-----|---------|---------|-----------|---------|-------|
| A770 (Xe1) | SYCL | ~40 t/s | ~25 t/s | ~30 t/s | Q8_0 works well |
| A770 (Xe1) | Vulkan | ~30 t/s | ~20 t/s | ~35 t/s | Good prompt processing |
| B580 (Xe2) | SYCL | ~45 t/s | ~20 t/s | ~8 t/s | Q8_0 broken |
| B580 (Xe2) | Vulkan | ~35 t/s | ~18 t/s | ~10 t/s | Better prompt perf |
| B70 (Xe2) | SYCL | ~35 t/s | ~20 t/s | ~5 t/s | Q8_0 very slow |
| 140V iGPU (Xe2) | SYCL | ~23 t/s | N/A (crash) | N/A | K-quants broken |

---

## Community Complaints Summary

1. **"30% of peak performance"** - Users see far below hardware potential
2. **"Instability with k-quants"** - Some formats crash, others work
3. **"Documentation chaos"** - Multiple docs, Ubuntu-focused, Arch struggles
4. **"IPEX-LLM is too slow but stable, llama.cpp is fast but broken"** - No perfect option
5. **"Driver updates don't fix issues"** - Confirms software stack problem
6. **"No Intel official contribution"** - Community maintains in spare time

---

*Last Updated: April 2026*