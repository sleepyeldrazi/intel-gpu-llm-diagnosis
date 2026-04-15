# Intel Arc GPU LLM Inference: Comprehensive Research Overview

**Author:** GLM Agent  
**Date:** April 15, 2026  
**Phase:** Research & Data Preparation (No driver/framework modifications)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Intel GPU Software Stack](#the-intel-gpu-software-stack)
3. [Identified Problems](#identified-problems)
4. [Performance Landscape](#performance-landscape)
5. [Root Cause Analysis](#root-cause-analysis)
6. [Speculations & Hypotheses](#speculations--hypotheses)
7. [Suggested Solutions](#suggested-solutions)
8. [Repo Map & Key Files](#repo-map--key-files)
9. [References](#references)

---

## Executive Summary

Intel Arc GPUs (both discrete Alchemist/Xe1 and Battlemage/Xe2, and integrated Lunar Lake/Arrow Lake) suffer from **severely degraded LLM inference performance** compared to their theoretical hardware capabilities. Users consistently report achieving only **21-40% of theoretical memory bandwidth** on quantized models, versus **80-95% on equivalent NVIDIA/AMD hardware**. The issues are multifaceted, spanning kernel optimization gaps, software stack fragmentation, driver/kernel version incompatibilities, and a fundamental underinvestment in open-source kernel development for Intel GPU architectures.

The Battlemage (B580/B570/B70) generation introduced a **new class of regression**: quantization types that worked well on Xe1 (Alchemist) perform catastrophically on Xe2, with Q8_0 being **4-5x slower** than Q4_K_M despite only having 1.7x more data. This was traced to a kernel dispatch path issue (now partially fixed by PR #21527).

---

## The Intel GPU Software Stack

### Layer Architecture

```
┌─────────────────────────────────────────────────┐
│  User-facing: Ollama, LM Studio, llama.cpp      │
├─────────────────────────────────────────────────┤
│  Framework: IPEX-LLM (archived Jan 2026)         │
│             vLLM (intel/vllm docker)              │
│             OpenVINO                              │
│             PyTorch + IPEX                        │
├─────────────────────────────────────────────────┤
│  Backend: SYCL (llama.cpp)                       │
│           Vulkan (llama.cpp, cross-vendor)        │
│           Level Zero (low-level compute)          │
├─────────────────────────────────────────────────┤
│  Compiler: DPC++ (intel/llvm sycl branch)         │
│            IGC (Intel Graphics Compiler)           │
├─────────────────────────────────────────────────┤
│  Runtime: compute-runtime (NEO)                   │
│           Level Zero Loader                       │
│           oneDNN (BLAS/GEMM)                      │
├─────────────────────────────────────────────────┤
│  Kernel Driver: i915 / xe (Linux)                 │
│  Firmware: linux-firmware                         │
├─────────────────────────────────────────────────┤
│  Hardware: Xe1 (Alchemist: A380/A750/A770)        │
│            Xe2 (Battlemage: B570/B580/B70)        │
│            Xe2 iGPU (Lunar Lake 140V, Arrow Lake) │
└─────────────────────────────────────────────────┘
```

### The Fragmentation Problem

Intel's software stack for GPU inference has **at least five semi-independent efforts** that don't fully interoperate:

| Stack | Maintainer | Status | Backend | Optimized? |
|-------|-----------|--------|---------|-----------|
| llama.cpp SYCL | Community (NeoZhangJianyu, Rbiessy/Codeplay) | Active | SYCL/Level Zero | Only Q4_0 fully |
| llama.cpp Vulkan | 0cc4m (community) | Active | Vulkan | Improving, behind CUDA |
| IPEX-LLM | Intel (analytics team) | **Archived Jan 2026** | SYCL + proprietary | Best perf, dying |
| OpenVINO | Intel (openvino team) | Active | Own runtime | Different model format |
| vLLM XPU | Intel (vllm fork) | Active | PyTorch/IPEX | Server-focused |

**Key complaint from Hacker News user lhl:**
> "PyTorch requires its own support kit separate from the oneAPI Toolkit (and runs slightly different versions of everything), the vLLM xpu support doesn't work — both source and the docker failed to build/run for me. The IPEX-LLM whisper support is completely borked."

### Version Compatibility Nightmare

| Component | IPEX-LLM requires | Latest available | Conflict? |
|-----------|-------------------|-----------------|-----------|
| oneAPI Base Toolkit | 2024.2.1 | 2025.3+ | **Yes** |
| PyTorch | 2.5.x | 2.8+ | **Yes** |
| compute-runtime | 25.x | 26.x | **Yes** |
| Linux kernel | 6.5-6.17 | 6.18+ | **Yes (6.18 breaks)** |
| IGC | 2.27+ | 2.30+ | Minor |
| DPC++ Compiler | 2024.2 | 2025.3 | **ABI changes** |

**Critical finding**: Linux kernel 6.18 completely breaks compute-runtime's memory management (`bindless_heaps_helper.cpp` abort). Users must stay on 6.17 or older. This is tracked in compute-runtime#875.

**Another critical finding**: IPEX-LLM shifted oneAPI dependency from 2024.2.1 to 2025.0.1 starting from ipex-llm[cpp]==2.2.0b20250207, but the archived repo won't get further updates. Users with older IPEX-LLM versions are stuck on old oneAPI.

---

## Identified Problems

### Problem 1: SYCL Kernel Dispatch — Quantization Type Inequality (CRITICAL)

**Status**: Partially fixed (Q8_0), open for most other types  
**Impact**: 2-5x performance degradation on non-Q4_0 quantizations

The llama.cpp SYCL backend has three kernel paths for quantized matrix-vector multiplication:
1. **DMMV** (Dequantize-Mul-Mat-Vec) — generic, slow
2. **MMVQ** (Mul-Mat-Vec-Q) — optimized, uses reorder
3. **SYCL native matmul** — fallback

Only Q4_0 and Q8_0 (after PR #21527) have full reorder support. All other quantization types fall through to slower paths:

| Format | DMMV Reorder | MMVQ Reorder | SYCL Matmul Reorder | Effective BW |
|--------|:------------:|:------------:|:-------------------:|:------------:|
| Q4_0 | ✅ | ✅ | ✅ | 57% |
| Q8_0 | ✅ (PR #21527) | ✅ (PR #21527) | ✅ | 66% |
| Q4_K | ❌ | ✅ | ✅* | 53% |
| Q5_K | ❌ | ❌ | ❌ | ~39% |
| Q6_K | ❌ | ✅ | ✅* | ~48% |
| IQ4_NL | ❌ | ❌ | ✅ | 14% |
| Q4_1/Q5_0/Q5_1 | ❌ | ❌ | ✅ | ~30-44% |

*\* Only when `g_ggml_sycl_prioritize_dmmv` is NOT set*

The root cause is in the dispatch logic (`ggml-sycl.cpp` lines 3269-3340):

```cpp
// Only Q4_0 and Q8_0 supported in DMMV reorder
inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;  // Q4_K, Q5_K, Q6_K all fall through
    }
}
```

### Problem 2: Xe2/Battlemage Regression — Q8_0 Catastrophically Slow (CRITICAL, mostly fixed)

**Status**: PR #21527 submitted, 3.1x speedup validated  
**Impact**: Q8_0 ran at 21% bandwidth on Xe2 vs 53-64% for Q4_K_M

On Arc A770 (Xe1), Q8_0 was **faster** than Q4/Q6. On Arc B70/B580 (Xe2), Q8_0 was **4-5x slower**. The root cause:

- **Generic DMMV path**: `iter_stride = 2 * GGML_SYCL_DMMV_X = 64` → processes 2 values per thread per iteration
- **Reorder DMMV path** (Q4_0): `iter_stride = 8 * 2 * GGML_SYCL_DMMV_X = 512` → processes 16 values per thread per iteration

Q8_0 was stuck on the generic path because it wasn't in `ggml_sycl_supports_reorder_dmmv()`. This was confirmed not a driver issue (IGC 2.28.4 → 2.30.1 showed no change) and not a backend issue (both SYCL and Vulkan equally affected).

PR #21527 added Q8_0 to the reorder framework, achieving 66% bandwidth utilization (up from 21%).

### Problem 3: K-Quantization Crashes on Xe2 iGPU (CRITICAL)

**Status**: Workaround exists (use upstream llama.cpp SYCL instead of IPEX-LLM)  
**Impact**: Q4_K_M, Q5_K, Q6_K crash on Arc 140V/140T

```
Sub-group size 8 is not supported on the device
Exception at ggml-sycl.cpp:3164
```

This error occurs in IPEX-LLM's bundled llama.cpp but not in upstream. IPEX-LLM's llama.cpp is based on an August 2024 snapshot and hasn't received the fixes. Since the project was archived in January 2026, this will likely never be fixed in IPEX-LLM.

### Problem 4: Arc 140T Misdetection — Coopmat Disabled (HIGH)

**Status**: Open issue  
**Impact**: Cooperative matrix operations completely disabled on valid Xe2 hardware

The Vulkan backend classifies GPUs by `minSubgroupSize`:
- `minSubgroupSize == 16` → classified as `INTEL_XE2` → coopmat enabled
- Everything else → classified as `OTHER` → no coopmat

Arrow Lake H (Arc 140T) reports `minSubgroupSize = 8` despite having Xe2 architecture and full cooperative matrix support. This appears to be a driver-level reporting bug.

### Problem 5: Linux Kernel Version Fragility (HIGH)

**Status**: Active issue (compute-runtime#875)  
**Impact**: Complete failure of all GPU compute on kernel 6.18

| Kernel Version | Works? | Notes |
|:--------------:|:------:|-------|
| ≤ 6.6.25 | ✅ | Stable baseline |
| 6.6.26 - 6.8 | ⚠️ | Some CCS fence timeout issues |
| 6.9 - 6.17 | ✅ | Working range |
| 6.18+ | ❌ | `bindless_heaps_helper.cpp` abort, all Level Zero fails |

Additionally, Linux firmware updates have broken Intel GPU compute in the past. The `linux-firmware` package version `20240409.1addd7dc` introduced fence timeouts that hung llama.cpp, requiring downgrades.

### Problem 6: IPEX-LLM Abandonment (HIGH)

**Status**: Archived January 2026  
**Impact**: Best-performing Intel GPU inference stack no longer maintained

IPEX-LLM consistently outperformed upstream llama.cpp SYCL by **50-80%** on token generation (e.g., 24.35 t/s vs 13.51 t/s on Arc 140V with Llama-2-7B Q4_0). This performance gap came from:

1. **Closed-source optimized kernels** not shared with upstream
2. **oneDNN GEMM integration** for prompt processing
3. **syclcompat library** for platform-specific tuning
4. **Proprietary quantization optimizations**

With IPEX-LLM archived, these optimizations are frozen. The llama.cpp community must independently re-derive all of this work.

### Problem 7: Vulkan vs SYCL Performance Inconsistency (MEDIUM)

**Status**: Improving, but still inconsistent  
**Impact**: Users must test both backends for each model/generation

Historically:
- SYCL: Better prompt processing (5-6x faster than Vulkan)
- Vulkan: Better token generation (up to 50% faster than SYCL in late 2024)
- Recent SYCL improvements have narrowed the gap

The two backends have completely separate kernel implementations with different optimization strategies, and neither consistently wins across all quantization types and hardware generations.

### Problem 8: vLLM XPU Quantization Limitations (MEDIUM)

**Status**: Active development  
**Impact**: Only FP16, Dynamic FP8, MXFP4 validated on XPU

vLLM's Intel XPU support does not support many quantization formats that work on CUDA:
- **GPTQ**: Limited, torchao AWQ models crash (hard-requires CUDA)
- **AWQ**: Occupies more memory than model size on XPU
- **Marlin/Machete kernels**: CUDA-only
- **GGUF**: Not supported in vLLM at all

The intel/llm-scaler-vllm docker images lag behind mainline vLLM (currently at 0.14.0-b8.1 vs 0.16+ mainline).

### Problem 9: Missing DPAS/XMX Utilization for Quantized Inference (HIGH)

**Status**: In early investigation  
**Impact**: Intel's key hardware advantage (XMX tensor cores) goes unused for quantized matmul

Intel Xe/Xe2 GPUs have XMX (Xe Matrix eXtensions) units capable of DPAS (Dot Product and Accumulate Systolic) operations. The Arc A770 has 4096-bit XMX units; B580 has similar. However:

- **SYCL backend**: Uses `joint_matrix` extension only for FP16/BF16 GEMM, not for quantized formats
- **Vulkan backend**: DP4A instruction support added, but not yet wired to matmul path
- **K-quants**: No DPAS path at all — rely entirely on scalar DP4A or software emulation
- The community contributors (Rbiessy, NeoZhangJianyu) have noted that kernels are memory-bound *before* they can benefit from DPAS, so the memory access patterns must be fixed first

Quote from Rbiessy (Codeplay):
> "I say potentially [using the matrix engine] because currently the kernel is memory bound in the configurations we have tried. If we're still not able to improve that for some reason using HMX won't help."

### Problem 10: Understaffed Open-Source Development (SYSTEMIC)

**Status**: Ongoing  
**Impact**: All other problems stem from this root cause

The SYCL backend in llama.cpp is primarily maintained by:
- **NeoZhangJianyu**: Independent contributor, spare time
- **Rbiessy (Codeplay/Samsung)**: Part of Codeplay team (acquired by Samsung), contributing to SYCL optimization
- **0cc4m**: Vulkan backend development
- **qnixsynapse**: Testing and CI

Quote from NeoZhangJianyu:
> "We are private contributors to maintain the SYCL backend on Intel GPU. You shouldn't complain so much, since we spend our spare time in past year to maintain it and make it work. Yes, it works, instead of work perfect. For BMG, we don't promise to optimize it in time of the marketing."

Intel itself does not officially contribute to the llama.cpp SYCL backend. Their focus is on IPEX-LLM (now archived), OpenVINO, and vLLM XPU.

---

## Performance Landscape

### Benchmarks: llama.cpp SYCL vs Vulkan vs IPEX-LLM

Compiled from community reports (llm-tracker.info, GitHub issues, Reddit):

#### Arc 140V iGPU (Lunar Lake, Xe2) — Llama-2-7B Q4_0

| Backend | pp512 (t/s) | tg128 (t/s) | MBW Efficiency |
|---------|:-----------:|:-----------:|:--------------:|
| CPU (4 P-cores) | 25.05 | 11.59 | 30% |
| Vulkan | 44.65 | 5.54 | 14% |
| SYCL FP32 | 180.77 | 14.39 | 38% |
| SYCL FP16 | 526.38 | 13.51 | 35% |
| **IPEX-LLM** | **708.15** | **24.35** | **64%** |

Theoretical max tg: 136.5 GB/s ÷ 3.56 GB = ~38.3 t/s

#### Arc A770 (Alchemist, Xe1) — Various Models

| Model | Quant | SYCL pp512 | SYCL tg128 | Vulkan tg128 |
|-------|-------|:----------:|:----------:|:------------:|
| Llama 7B | Q4_0 | ~500 | ~40 | ~30 |
| Llama 7B | Q6_K | ~700 | ~22 | ~21 |
| Llama 13B | Q5_K | ~400 | ~16 | ~8 |
| Llama 30B | Q2_K | ~160 | ~8.5 | ~5 |

#### Arc B580 (Battlemage, Xe2) — Llama-3.1-8B

| Quant | SYCL tg (t/s) | Expected (456 GB/s BW) | Efficiency |
|-------|:------------:|:---------------------:|:----------:|
| Q4_K_M | 25-30 | ~38 | 66-79% |
| Q8_0 (pre-fix) | ~8 | ~22 | 36% |
| Q8_0 (post-fix) | ~18 | ~22 | 82% |

#### Arc Pro B70 (Battlemage, Xe2) — Qwen3.5-27B (comprehensive sweep)

| Quant | Size (GiB) | tg128 (t/s) | Effective BW | % of 608 GB/s |
|-------|:----------:|:-----------:|:------------:|:-------------:|
| Q4_0 | 14.63 | 23.67 | 346 GB/s | **57%** |
| Q4_K_M | 15.58 | 20.56 | 321 GB/s | **53%** |
| Q6_K | 20.90 | 13.83 | 289 GB/s | **48%** |
| Q5_K_M | 18.25 | 13.78 | 252 GB/s | **41%** |
| IQ4_NL | 14.60 | 5.85 | 85 GB/s | **14%** |
| Q8_0 (pre-fix) | 26.62 | 4.88 | 130 GB/s | **21%** |
| Q8_0 (post-fix) | — | 15.24 | 402 GB/s | **66%** |

### Comparison vs NVIDIA/AMD

| GPU | Price | VRAM | BW (GB/s) | 8B Q4_K_M tg | BW Efficiency |
|-----|:-----:|:----:|:---------:|:------------:|:------------:|
| Arc B580 | $249 | 12GB | 456 | ~25-30 t/s | ~66-79% |
| RTX 4060 | $299 | 8GB | 272 | ~35 t/s | ~92% |
| RTX 4060 Ti 16GB | $499 | 16GB | 288 | ~40 t/s | ~90% |
| RTX 3060 12GB (used) | $170 | 12GB | 360 | ~35 t/s | ~88% |

Intel Arc delivers **20-30% less inference speed** than NVIDIA at similar price points, despite having competitive raw bandwidth. The gap is entirely in software efficiency.

---

## Root Cause Analysis

### Layer 1: Kernel Code — Missing Reorder Implementations

The **reorder optimization** (originally added for Q4_0 in PR #12035 by NeoZhangJianyu) separates quantized data from metadata (scales/zero-points) in GPU memory, enabling coalesced memory access patterns. This is the single biggest performance differentiator:

- **Without reorder**: iter_stride=64, 2 values per thread per iteration → poor memory coalescing
- **With reorder**: iter_stride=512, 16 values per thread per iteration → near-optimal memory access

Only Q4_0 had this optimization until PR #21527 added Q8_0. All K-quant formats (Q4_K, Q5_K, Q6_K) are missing DMMV reorder implementations.

**Files involved:**
- `llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` (dispatch logic)
- `llama.cpp/ggml/src/ggml-sycl/dmmv.cpp` (DMMV kernels)
- `llama.cpp/ggml/src/ggml-sycl/mmvq.cpp` (MMVQ kernels)

### Layer 2: Architecture Blindness — No Xe1 vs Xe2 Differentiation

The SYCL backend treats all Intel GPUs identically. There's no runtime adaptation for:
- Different L2 cache sizes (Xe1: 16MB, Xe2: larger)
- Different optimal block sizes (Xe1: 64, Xe2: 128)
- Different prefetch depths
- Different vector widths

This explains why Xe2-specific regressions occur: optimizations tuned for Xe1 can be counterproductive on Xe2's different memory hierarchy.

### Layer 3: Driver/Runtime — Kernel Version Coupling

Intel's compute-runtime has tight coupling with the Linux kernel's i915/xe driver. Changes in kernel memory management (e.g., SVM, CCS enablement, bindless heaps) break the userspace stack. The kernel 6.18 regression is the latest in a series:
- Kernel 6.6.26+: Fence timeouts (CCS changes)
- Kernel 6.8+: Various hangs
- Kernel 6.18: Complete memory allocation failure

### Layer 4: Organizational — Intel's Strategic Confusion

Intel has multiple teams working on overlapping GPU inference stacks without coordination:
1. **Intel Analytics** (China): IPEX-LLM → archived
2. **Intel OpenVINO team**: OpenVINO inference runtime
3. **Intel vLLM team**: intel/vllm docker fork
4. **Intel PyTorch team**: intel-extension-for-pytorch
5. **Intel compiler team**: DPC++/SYCL (llvm sycl branch)
6. **Intel compute-runtime team**: Level Zero driver
7. **Codeplay** (Samsung subsidiary): SYCL backend contributions to llama.cpp
8. **Community volunteers**: Most of the actual llama.cpp SYCL work

These teams use different oneAPI versions, different PyTorch versions, and target different benchmarks. There's no unified "make Intel GPUs fast for LLM inference" strategy.

---

## Speculations & Hypotheses

### H1: IPEX-LLM's Performance Secret Was Kernel Specialization, Not Magic

IPEX-LLM's 50-80% advantage over upstream llama.cpp SYCL likely comes from:
1. **Hand-tuned DPAS kernels** for specific quantization formats
2. **oneDNN integration** for prompt processing GEMM (not available in llama.cpp)
3. **Memory layout optimizations** that upstream hasn't replicated

The fact that PR #21527 achieved 66% bandwidth for Q8_0 (close to IPEX-LLM's ~61% on Arc 140V) suggests that the reorder approach can close much of the gap, but the closed-source DPAS kernels remain unrecoverable.

### H2: The Xe2 Memory Subsystem Has Different Optimal Access Patterns

The fact that Q8_0 was *faster* on Xe1 but *catastrophically slower* on Xe2 suggests fundamental architectural differences in:
- L2 cache behavior (Xe2 may have different cache line policies)
- Memory controller scheduling (Xe2 GDDR6 vs Xe1 GDDR6 may have different timing)
- EU thread scheduling (Xe2 may have different SIMT behavior)

Without Intel publishing detailed Xe2 microarchitecture documentation, kernel developers are flying blind.

### H3: The Driver Stack Will Remain Fragile

Intel's ongoing transition from i915 to xe kernel driver, combined with the compute-runtime's tight kernel coupling, suggests that:
- New kernel versions will periodically break GPU compute
- Docker images will be pinned to specific kernel versions
- Users on rolling-release distros (Arch, Fedora) will be most affected

### H4: OpenVINO Is Intel's Real Strategy, But It Doesn't Help llama.cpp

Intel seems to be positioning OpenVINO as their primary inference runtime. OpenVINO has:
- Its own model format (not GGUF)
- Its own quantization pipeline (not GPTQ/AWQ/GGUF)
- Better performance out of the box

But the local LLM community overwhelmingly uses GGUF/llama.cpp, not OpenVINO. Intel's strategy doesn't align with user demand.

### H5: The Community Could Fix Most Issues With Proper Resources

The reorder optimization framework is extensible by design. With focused effort, the remaining quantization types could be optimized:
- Q4_K DMMV reorder: ~2-3 weeks of focused work
- Q5_K, Q6_K reorder: ~4-6 weeks each
- Xe2-specific tuning: ~2-4 weeks with profiling tools
- DPAS integration: ~2-3 months for quantized formats

The bottleneck is not technical complexity but developer time.

---

## Suggested Solutions

### Priority 1: Critical Fixes (1-2 weeks)

| Task | Effort | Impact | Status |
|------|--------|--------|--------|
| Merge PR #21527 (Q8_0 reorder) | Done | 3.1x Q8_0 speedup | Pending review |
| Implement Q4_K DMMV reorder | Medium | 40% speedup for Q4_K | Not started |
| Fix Arc 140T minSubgroupSize detection | Small | Enables coopmat | Not started |
| Document kernel version compatibility | Small | Prevents user frustration | Not started |

### Priority 2: Quantization Coverage (1-3 months)

| Task | Effort | Impact |
|------|--------|--------|
| Add Q5_K to reorder framework | Medium | ~2x speedup |
| Add Q6_K to reorder framework | Medium | ~1.5x speedup |
| Fix IQ4_NL kernel (14% → 50%+ BW) | Hard | ~3.5x speedup |
| Increase DMMV iter_stride for non-reorder types | Small | 20-30% improvement |

### Priority 3: Architecture Optimization (3-6 months)

| Task | Effort | Impact |
|------|--------|--------|
| Implement Xe2-specific kernel variants | Hard | Architecture-appropriate tuning |
| Enable DPAS for quantized matmul via prefetch optimization | Hard | Could reach 80%+ BW |
| Complete FlashAttention for SYCL | Medium | 2-3x prompt processing improvement |
| Add runtime GPU architecture detection (Xe1 vs Xe2) | Medium | Auto-tuning |

### Priority 4: Ecosystem Fixes (ongoing)

| Task | Effort | Impact |
|------|--------|--------|
| Create unified benchmark suite for Intel GPUs | Medium | Reproducible perf tracking |
| Test matrix: kernel × compute-runtime × oneAPI | Large | Prevent version combo failures |
| Coordinate with Intel for official contributions | Political | Sustainable maintenance |
| Investigate OpenVINO's optimized kernels for porting | Medium | Leverage existing Intel work |
| Add vLLM XPU support for GPTQ/AWQ | Large | Production quantized serving |

### Priority 5: Strategic Recommendations

1. **Intel should officially contribute to llama.cpp SYCL backend** — this is where the users are
2. **Open-source IPEX-LLM's optimized kernels** before they're permanently lost
3. **Decouple compute-runtime from kernel version** — validate on LTS kernels, add version negotiation
4. **Create a "one command" setup** for Intel GPU LLM inference (like `pip install torch` for CUDA)
5. **Publish Xe2 microarchitecture details** to enable community kernel optimization

---

## Repo Map & Key Files

```
repos/
├── llama.cpp/                          # Main inference engine
│   ├── ggml/src/ggml-sycl/
│   │   ├── ggml-sycl.cpp               # Dispatch logic (lines 3269-3660)
│   │   ├── dmmv.cpp                    # DMMV kernels (iter_stride issue)
│   │   ├── mmvq.cpp                    # MMVQ reorder kernels
│   │   ├── dequantize.hpp              # Dequantization functions
│   │   └── vecdotq.hpp                 # Vector dot product implementations
│   ├── ggml/src/ggml-vulkan/
│   │   └── ggml-vulkan.cpp             # Vulkan backend (140T misdetection)
│   └── docs/backend/SYCL.md            # SYCL backend documentation
│
├── compute-runtime/                    # Level Zero + OpenCL driver
│   ├── shared/source/helpers/
│   │   └── bindless_heaps_helper.cpp   # Kernel 6.18 crash point
│   └── level_zero/                     # Level Zero implementation
│
├── intel-graphics-compiler/            # IGC - GPU shader/compute compiler
│   └── documentation/visa/instructions/
│       └── DPAS.md                     # DPAS instruction documentation
│
├── intel-extension-for-pytorch/        # IPEX - PyTorch GPU extension
│
├── ipex-llm/                           # IPEX-LLM (archived Jan 2026)
│   ├── docs/mddocs/Quickstart/         # Installation guides
│   └── [closed-source optimized kernels]
│
├── vllm/                               # vLLM mainline
├── vllm-xpu-kernels/                   # Intel XPU-specific vLLM kernels
│
├── oneDNN/                             # Intel BLAS/GEMM library
├── openvino/                           # Intel's inference runtime
├── llvm/                               # DPC++ compiler (SYCL branch)
│   └── sycl/                           # SYCL runtime implementation
│
├── level-zero/                         # Level Zero loader + headers
└── sycl-tla/                           # SYCL Templates for Linear Algebra
```

---

## References

### Critical GitHub Issues
| # | Repo | Title | Severity |
|---|------|-------|----------|
| [#21517](https://github.com/ggml-org/llama.cpp/issues/21517) | llama.cpp | Q8_0 4x slower on Arc Pro B70 | Critical |
| [#21527](https://github.com/ggml-org/llama.cpp/pull/21527) | llama.cpp | Q8_0 reorder fix (3.1x speedup) | Critical (fix) |
| [#12570](https://github.com/ggml-org/llama.cpp/discussions/12570) | llama.cpp | Current status of Intel Arc GPUs | High (discussion) |
| [#5277](https://github.com/ggml-org/llama.cpp/discussions/5277) | llama.cpp | SYCL Long Term Features & Issues | High (tracking) |
| [#12318](https://github.com/intel/ipex-llm/issues/12318) | ipex-llm | K-quant crashes on Xe2 iGPU | Critical |
| [#12991](https://github.com/intel/ipex-llm/issues/12991) | ipex-llm | Vulkan faster than SYCL on B580 | Medium |
| [#875](https://github.com/intel/compute-runtime/issues/875) | compute-runtime | Kernel 6.18 breaks Level Zero | Critical |
| [#788](https://github.com/intel/compute-runtime/issues/788) | compute-runtime | sycl-ls fails on B580 | High |

### Community Resources
- [llm-tracker.info Intel GPU guide](https://llm-tracker.info/howto/Intel-GPUs) — Comprehensive setup/performance data
- [CraftRigs B580 LLM review](https://craftrigs.com/reviews/intel-arc-b580-local-llm-performance/) — Honest benchmark assessment
- [Hacker News discussion](https://news.ycombinator.com/item?id=42500245) — Software stack critiques
- [Phoronix Vulkan benchmarks](https://www.phoronix.com/review/llama-cpp-vulkan-eoy2025) — Cross-vendor Vulkan comparison
- [vLLM XPU docs](https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/) — Supported models matrix

### Key People
- **NeoZhangJianyu**: Primary SYCL backend maintainer (spare-time contributor)
- **Rbiessy (Codeplay)**: Working on mul_mat_vec_q kernel optimization, prefetch, DPAS
- **0cc4m**: Vulkan backend developer for Intel GPUs
- **PMZFX**: Filed #21517, submitted PR #21527 (Q8_0 reorder fix)
- **lhl (llm-tracker.info)**: Comprehensive benchmarking and documentation

---

*This document is part of Phase 1: Research & Data Preparation*  
*No driver/framework modifications were made during this phase*  
*Companion documents: `research/community_issues/issues_and_discourse_minimax.md`, `research/kernels/kernel_analysis_minimax.md`*
