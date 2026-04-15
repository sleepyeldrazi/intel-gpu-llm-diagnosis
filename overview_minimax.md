# Intel Arc GPU LLM Inference Research Overview

**Date:** April 2026  
**Scope:** Intel Arc (Alchemist Xe1, Battlemage Xe2) GPUs for LLM inference, with focus on quantized model performance issues

---

## 1. Executive Summary

Intel Arc GPUs suffer from **severe performance underperformance** on quantized LLM inference compared to theoretical hardware capabilities. Community benchmarks reveal that token generation often achieves only **21-40% of theoretical memory bandwidth utilization**, while NVIDIA RTX and AMD GPUs typically achieve 80-95%. The root causes are multi-layered: missing kernel optimizations, quantization-specific bottlenecks, architecture detection bugs, and an immature software stack.

---

## 2. Hardware & Software Landscape

### 2.1 Supported Intel GPUs

| GPU | Architecture | Memory Bandwidth | Status |
|-----|-------------|------------------|--------|
| Arc A770 (16GB) | Xe1/Alchemist | 512 GB/s | Active support |
| Arc A750 (8GB) | Xe1/Alchemist | 448 GB/s | Active support |
| Arc B580 | Xe2/Battlemage | 456 GB/s | Partial support (driver issues) |
| Arc B70 Pro | Xe2/Battlemage | 608 GB/s | Active, but regressed |
| Arc 140T (iGPU) | Xe2/Arrow Lake H | Unified | **Broken** - misdetected |
| Arc 140V (iGPU) | Xe2/Lunar Lake | Unified | Working |

### 2.2 Software Stack

```
┌─────────────────────────────────────────────────────────┐
│                   User Applications                      │
│        (Ollama, vLLM, llama.cpp CLI, etc.)              │
├─────────────────────────────────────────────────────────┤
│                 Inference Frameworks                    │
│   IPEX-LLM (PyTorch) │ vLLM (Intel Port) │ llama.cpp    │
├─────────────────────────────────────────────────────────┤
│                Backend Implementations                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ SYCL     │  │ Vulkan   │  │ OpenVINO (IPEX-LLM)   │  │
│  │ backend  │  │ backend  │  │                       │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Intel Software Components                   │
│  oneAPI DPC++ │ IGC Compiler │ Level Zero │ oneDNN       │
├─────────────────────────────────────────────────────────┤
│                    GPU Hardware                          │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Key Repositories

- **llama.cpp**: Main inference engine with SYCL and Vulkan backends for Intel GPUs
  - Location: `repos/llama.cpp/`
  - Key paths: `ggml/src/ggml-sycl/`, `ggml/src/ggml-vulkan/`

- **IPEX-LLM**: Intel's optimized PyTorch extension (archived Jan 2026)
  - Location: `repos/ipex-llm/`
  - Note: Archive status raises concerns about future maintenance

- **vLLM Intel**: Production-grade serving with Arc Pro B-series support
  - Reference: https://blog.vllm.ai/2025/11/11/intel-arc-pro-b.html

---

## 3. Critical Problems Identified

### 3.1 Q8_0 Quantization Catastrophic Performance (Issue #21517)

**Severity:** Critical - **4-5x slower than expected**

The Arc Pro B70 (Xe2) achieves only **21-24% of theoretical memory bandwidth** on Q8_0 quantized models, compared to **53-64%** for Q4_K_M:

| Quantization | Size (GiB) | Token Gen (t/s) | BW Utilization |
|-------------|-----------|-----------------|----------------|
| Q4_K_M      | 15.58     | 20.56           | 53%            |
| Q8_0        | 26.62     | 4.88            | 21%            |

**Root Cause:**
- Q8_0 is stuck on **DMMV kernel path** (generic dequantize-mul-mat-vec)
- iter_stride = 64 → processes only 2 values per thread per iteration
- Q4_0 reorder kernel uses iter_stride = 512 → 16 values per iteration (8x more)
- Q8_0's 34-byte block structure is non-power-of-2, causing cache line misalignment

**Fix Status:** PR #21527 submitted (Apr 2026) - adds Q8_0 to reorder framework, achieves **3.1x speedup**

### 3.2 K-Quantization Crashes on Xe2 iGPU (Issue #12318)

**Severity:** Critical - crashes on Lunar Lake Arc 140V

```
Sub-group size 8 is not supported on the device
Exception at ggml_sycl.cpp:3164
```

**Root Cause:** IPEX-LLM's SYCL backend hardcodes sub-group size assumptions that don't hold on Xe2 architecture.

**Note:** Upstream llama.cpp works but with ~2x lower performance than IPEX-LLM.

### 3.3 Architecture Misdetection on Arc 140T (Issue #20776)

**Severity:** High - Cooperative Matrix completely disabled

The Arc 140T (Arrow Lake H, Xe2) reports `matrix cores: none` because:
- Driver reports `minSubgroupSize = 8`
- Code requires `minSubgroupSize == 16` to classify as `INTEL_XE2`
- Same driver branch on Arc 140V reports `minSubgroupSize = 16`

**Impact:** All DPAS/matrix unit optimizations skipped despite hardware support.

### 3.4 Missing/Imcomplete Kernel Support Matrix

| Quantization | Reorder DMMV | Reorder MMVQ | SYCL | Vulkan |
|-------------|--------------|--------------|------|--------|
| Q4_0        | ✅            | ✅            | Fast | Fast   |
| Q4_K_M      | ❌           | ✅            | Medium | Medium |
| Q5_K_M      | ❌           | ❌            | **Slow** | Medium |
| Q6_K        | ❌           | ❌            | **Slow** | Medium |
| Q8_0        | ✅ (fixed)   | ✅ (fixed)   | Was Broken | Was Broken |
| IQ4_NL      | ❌           | ❌            | **14% BW** | **Broken** |
| IQ4_XS      | ❌           | ❌            | Slow | Medium |

### 3.5 Xe2/Battlemage Regression

On Arc A770 (Xe1), Q8_0 is actually **faster** than Q4. On Arc B70 (Xe2), Q8_0 is **4-5x slower**. This regression indicates the kernel optimizations work on Xe1 but fail on Xe2's different memory architecture.

---

## 4. Code Analysis: Misaligned Components

### 4.1 llama.cpp SYCL Backend

**Location:** `repos/llama.cpp/ggml/src/ggml-sycl/`

**Key Files:**
| File | Purpose | Issue |
|------|---------|-------|
| `ggml-sycl.cpp` | Main dispatch logic | Routing to wrong kernels for Xe2 |
| `dmmv.cpp` | Generic dequantize mat-vec | Inefficient iter_stride for Q8_0 |
| `mmvq.cpp` | Optimized mat-vec quants | Missing Q4_K DMMV reorder support |
| `vecdotq.hpp` | Vector dot products | Suboptimal memory coalescing |

**Dispatch Logic Problem:**
```
Lines ~3269-3292: ggml_sycl_supports_reorder_* functions
- Q4_K supports reorder for MMVQ but NOT DMMV
- This forces Q4_K through slower generic DMMV path
- Q5_K, Q6_K have NO reorder support at all
```

### 4.2 llama.cpp Vulkan Backend

**Location:** `repos/llama.cpp/ggml/src/ggml-vulkan/`

**Issues:**
- DP4A/DPAS support incomplete
- Subgroup size detection relies on driver-reported values (unreliable for Arrow Lake H)
- Memory access patterns not optimized for Xe2 cache hierarchy

### 4.3 IPEX-LLM (Archived)

**Location:** `repos/ipex-llm/`

**Status:** Archive as of January 28, 2026 - maintainability concerns

**Key Issues:**
- Closed-source optimized kernels not merged upstream
- Lagging behind llama.cpp main development
- Version fragmentation (ollama integration, vLLM integration, standalone C++)

---

## 5. Root Cause Hypotheses

### 5.1 Primary Hypothesis: Memory Access Pattern Mismatch

The Intel GPU memory hierarchy (L3 cache, SLM) has different optimal access patterns than NVIDIA/AMD. Current kernels:
- Use generic dequantization that doesn't account for Xe2's larger L2 cache
- Process too few elements per thread, leaving EU utilization low
- Don't leverage prefetch mechanisms that work on CUDA/ROCm

**Evidence:** Q4_0 reorder achieves 56% bandwidth, but Q8_0 DMMV achieves only 21%. The difference is purely in kernel design, not hardware capability.

### 5.2 Secondary Hypothesis: Compiler/Driver Inefficiencies

Intel's IGC (Intel Graphics Compiler):
- May not be generating optimal SIMD instructions for quantization kernels
- Register allocation for mixed precision (fp16 scales + int8 data) may be suboptimal
- Loop unrolling and vectorization may not be aggressive enough

**Evidence:** Driver updates (IGC 2.28.4 → 2.30.1) showed no improvement for Q8_0, confirming issue is in llama.cpp kernels, not compiler.

### 5.3 Tertiary Hypothesis: Architecture-Specific Tuning Missing

Xe1 (Alchemist) and Xe2 (Battlemage) have:
- Different L2 cache sizes (16MB vs larger on B70)
- Different memory latency characteristics
- Different SIMD width preferences

Current code has **no architecture-aware tuning** - same kernels run on all Intel GPUs.

### 5.4 Ecosystem Fragmentation

```
IPEX-LLM: Closed-source optimizations, lagging updates
llama.cpp SYCL: Community maintained, gaps in coverage  
llama.cpp Vulkan: Good prompt processing, poor token gen
vLLM Intel: Production-grade but limited model support
```

No unified optimization effort across all quantization formats.

---

## 6. Proposed Solutions

### 6.1 Immediate Actions (Existing PRs)

1. **PR #21527** - Q8_0 Reorder Support (3.1x speedup for Q8_0)
   - Status: Submitted, needs testing on A770/A750
   - Impact: Major for users needing FP16-equivalent quality

2. **Issue #20776 Fix** - Add device ID fallback for Arc 140T
   ```cpp
   // In get_device_architecture(), add:
   if (props.deviceID == 0x7D51 || props.deviceID == 0x7D45) {
       return vk_device_architecture::INTEL_XE2;
   }
   ```

### 6.2 Short-term Optimizations (1-3 months)

1. **Extend Reorder Framework to K-Quants**
   - Implement `reorder_qw_q4_k()` equivalent for DMMV path
   - Add Q5_K, Q6_K to reorder support list
   - Target: 2-3x speedup for these formats

2. **Increase DMMV iter_stride**
   - Current: 64 elements per iteration
   - Target: 512 (match Q4_0 reorder)
   - Estimate: 50-100% speedup for formats stuck on DMMV

3. **Architecture-Aware Kernel Selection**
   - Detect Xe1 vs Xe2 at runtime
   - Select optimal kernel variants per architecture
   - Enable Xe2-specific optimizations (larger prefetch, different block sizes)

### 6.3 Medium-term Improvements (3-6 months)

1. **DPAS/Matrix Engine Utilization**
   - Intel Xe2 has matrix units (DPAS instructions)
   - Current utilization: 0% for quantized formats
   - Target: Use DPAS for Q4_K, Q8_0 via direct instruction insertion
   - Reference: Intel IGC DPAS documentation

2. **FlashAttention Implementation**
   - Currently disabled or partially working on Intel
   - Critical for long context models
   - Enable proper co-operative matrix usage

3. **Host-Side Kernel Submission Optimization**
   - Reduce submission overhead
   - Use SYCL Graphs to batch operations
   - Especially important for iGPU scenarios

### 6.4 Long-term Recommendations

1. **Consolidate IPEX-LLM Optimizations**
   - Negotiate with Intel to open-source closed-source kernels
   - Merge proven optimizations into llama.cpp mainline
   - Establish maintenance commitment (archive status is concerning)

2. **Comprehensive Quantization Coverage**
   - All K-quantizations need reorder support
   - IQ (Invisible Quantization) formats need optimization
   - AWQ format support for production deployments

3. **Unified Benchmark Suite**
   - Create Intel-specific benchmark covering all quantization formats
   - Track regression detection
   - Profile with Intel VTune for systematic optimization

---

## 7. Testing & Validation Plan

### 7.1 Benchmark Scenarios

```bash
# Token generation bandwidth test
./llama-bench -m <model>.Q8_0.gguf -ngl 99 -pp 512 -tg 128

# Expected on B70 (608 GB/s):
# Q4_K_M: ~18-22 t/s (baseline)
# Q8_0: ~15 t/s (after PR #21527) vs 4.88 t/s (before)
# Target: 35-40 t/s (60-65% bandwidth utilization)
```

### 7.2 Hardware Test Matrix

| GPU | Architecture | Priority | Test Focus |
|-----|-------------|----------|------------|
| Arc A770 | Xe1 | Medium | Regression check |
| Arc B580 | Xe2 | High | Primary Xe2 target |
| Arc Pro B70 | Xe2 | High | High-end Xe2 |
| Arc 140T iGPU | Xe2 | High | Detection fix validation |
| Arc 140V iGPU | Xe2 | Medium | Baseline comparison |

### 7.3 Quantization Test Matrix

| Quant | Priority | Current State | Target |
|-------|----------|---------------|--------|
| Q4_0 | Low | Optimized | Maintain |
| Q4_K_M | High | Medium | 2x speedup |
| Q5_K_M | High | Slow | 3x speedup |
| Q6_K | High | Slow | 3x speedup |
| Q8_0 | Critical | Fixed (PR pending) | Validate fix |
| IQ4_NL | Medium | Broken | Investigate root cause |

---

## 8. References & Links

### GitHub Issues

- [#21517](https://github.com/ggml-org/llama.cpp/issues/21517) - Q8_0 4x slower on B70
- [#12318](https://github.com/intel/ipex-llm/issues/12318) - K-quants crash Xe2 iGPU
- [#20776](https://github.com/ggml-org/llama.cpp/issues/20776) - Arc 140T misdetection
- [#12570](https://github.com/ggml-org/llama.cpp/discussions/12570) - Arc status discussion
- [#12805](https://github.com/ggml-org/llama.cpp/discussions/12805) - A750 user experiences
- [#19887](https://github.com/ggml-org/llama.cpp/issues/19887) - A770 inverse quant anomaly

### Documentation

- [llama.cpp SYCL Backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [IPEX-LLM Quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md)
- [vLLM Intel Arc Pro](https://blog.vllm.ai/2025/11/11/intel-arc-pro-b.html)
- [Intel DPAS Instructions](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md)

### Key Files in repos/

```
repos/llama.cpp/
├── ggml/src/ggml-sycl/
│   ├── ggml-sycl.cpp          # Dispatch logic (~line 3258-3648)
│   ├── dmmv.cpp               # Generic DMMV kernels
│   ├── mmvq.cpp               # Optimized MMVQ kernels
│   ├── vecdotq.hpp            # Vector dot products
│   └── quantize.hpp           # Quantization routines
├── ggml/src/ggml-vulkan/
│   └── ggml-vulkan.cpp        # Vulkan backend (~line 343 detection)
└── docs/backend/SYCL.md       # SYCL setup guide

repos/ipex-llm/
├── docs/mddocs/               # IPEX-LLM documentation
├── docker/llm/inference-cpp/  # C++ inference container
└── README.md                   # Project overview (archived notice)
```

---

## 9. Open Questions

1. **IPEX-LLM Future:** Intel archived the main ipex-llm repo in Jan 2026. Is there a replacement/maintained fork?

2. **Driver Release Cadence:** How frequently does Intel update GPU drivers? Does this impact reproducibility?

3. **Architecture-Specific Guidance:** Are there Intel-published optimization guides for Xe2?

4. **vLLM vs llama.cpp:** For production serving, which stack should be prioritized?

5. **Quantization Format Priority:** Given user needs, should we prioritize K-quants (quality) or legacy quants (speed)?

---

*Document Version: 1.0*  
*Research Sources: GitHub issues, llama.cpp discussions, Reddit, Intel developer articles*