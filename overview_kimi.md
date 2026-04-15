# Intel Arc GPU LLM Inference: Driver & Software Stack Research Overview

**Date:** 2026-04-15  
**Scope:** Research-only phase. No code or driver modifications were made. This document collects online community discourse, bug reports, and architectural analysis to identify why Intel Arc GPUs underperform on quantized LLM inference and where the relevant software stacks are misaligned.

---

## 1. Executive Summary

Intel Arc GPUs (Alchemist / Battlemage) are mechanically capable LLM inference cards on paper—large VRAM pools (up to 24 GB), wide memory buses (456–608 GB/s), and dedicated XMX matrix engines. In practice, community reports consistently describe **severe performance cliffs on quantized models**, **backend-specific kernel inefficiencies**, and **driver/runtime instability** that prevent Arc from reaching the bandwidth utilization seen on NVIDIA/AMD hardware.

The core observation across Reddit, GitHub Issues, and Intel documentation is that **token generation (TG) on quantized GGUF models often achieves only 20–40% of theoretical memory bandwidth**, while prompt processing (PP) can swing wildly depending on which backend (SYCL, Vulkan, OpenVINO, IPEX-LLM) is used. The problem is **not a single driver bug**; it is a misalignment between:

- **Kernel implementations** in `llama.cpp`’s SYCL/Vulkan backends that were optimized for NVIDIA/AMD data layouts and not fully ported for Intel Xe architecture.
- **The Intel graphics runtime stack** (Compute Runtime / Level Zero / IGC) which exposes the hardware but relies on user-space kernels to extract performance.
- **Rapid ecosystem churn**: IPEX-LLM was archived in Jan 2026, PyTorch XPU support moved upstream, and vLLM is mid-migration from IPEX to a new `vllm-xpu-kernels` backend, leaving users with fragmented, often conflicting setup instructions.

---

## 2. The Driver & Runtime Stack

For LLM inference on Intel Arc, the following layers are involved:

| Layer | Project / Driver | Role |
|-------|------------------|------|
| **Kernel driver** | `i915` / `xe` (Linux), Intel Graphics Driver (Windows) | Base GPU scheduling, memory management |
| **Compute Runtime** | `intel/compute-runtime` (NEO) | OpenCL + Level Zero driver; exposes SYCL devices |
| **Graphics Compiler** | Intel Graphics Compiler (IGC) | JIT-compiles SYCL kernels to Xe ISA |
| **Math libs** | oneMKL + oneDNN | GEMM/SDPA backends for PyTorch/SYCL |
| **Mesa Vulkan** | `anv` (Intel Vulkan driver) | Backend for `llama.cpp` Vulkan path |
| **Framework integrations** | `ipex-llm` (archived), `intel-extension-for-pytorch`, `vllm-xpu-kernels`, upstream `vllm` | User-facing inference stacks |

### Key Repositories Pulled

```
repos/
├── llama.cpp                      # SYCL & Vulkan backends, GGUF quantization kernels
├── ipex-llm                       # Intel’s former integration layer (archived Jan 2026)
├── intel-extension-for-pytorch    # PyTorch XPU extension (also archived / deprecated)
├── compute-runtime                # Intel Level Zero / OpenCL driver (NEO)
├── oneDNN                         # Intel deep-learning primitive library
├── vllm                           # vLLM mainline (XPU backend in flux)
└── vllm-xpu-kernels               # New dedicated Intel kernel repo for vLLM
```

---

## 3. Problems Identified in Community Discourse

### 3.1 Quantized Model Performance Is Disproportionately Bad

The most repeated complaint is that **quantized models run far slower than they should** given their reduced size.

**Example: `llama.cpp` SYCL backend on Arc Pro B70 (Xe2 / Battlemage)**  
*(GitHub Issue #21517, Reddit r/LocalLLAMA)*

| Quant | Model Size | TG (t/s) | Effective BW | % of Peak |
|-------|------------|----------|--------------|-----------|
| Q4_K_M | 15.6 GiB | 20.56 | 321 GB/s | 53% |
| Q8_0 | 26.6 GiB | 4.88 | 130 GB/s | **21%** |

**Critical finding:** Q8_0 is **4× slower** than Q4_K_M despite moving only **1.7× more bytes**. This rules out pure memory-bandwidth limits and points to **kernel-level inefficiency**.

The same issue affects both SYCL and Vulkan backends equally, and persists when splitting across two GPUs with abundant free VRAM. Updating the Compute Runtime / IGC had **no effect** on Q8_0 token-generation speed, confirming the bottleneck is in the inference-framework kernels, not the compiler or driver.

#### Inverse anomaly on older Arc A770 (Xe1 / Alchemist)
On Vulkan, some users report the *opposite*: Q8_0 **outperforms** Q4/Q6 in prompt processing (Issue #19887: ~600 t/s for Q8_0 vs ~200 t/s for Q6_K). This suggests the quantization-kernel imbalance is **architecture-specific**, not universal.

### 3.2 Missing or Partial "Reorder" Optimizations

`llama.cpp`’s SYCL backend introduced a **reorder optimization** (PR #12035, Feb 2025) that separates quantized weights from their scale factors so the GPU can load them with coalesced memory access. This optimization was implemented **only for Q4_0** and later extended to Q4_K and Q6_K. **Q8_0 was never added** to the reorder/MMVQ fast path.

Consequences:
- Q8_0 falls back to the generic **DMMV** kernel with `iter_stride = 64` (2 values per thread per iteration).
- The reorder path for Q4_0 uses `iter_stride = 512` (16 values per thread per iteration).
- A forced DMMV path for Q4_K_M drops its TG speed by ~40%, but forcing MMVQ on Q8_0 does **not** recover performance—both paths are slow for Q8_0 on Xe2.

Community speculation: the 34-byte `block_q8_0` layout is not a power-of-two, making it harder to vectorize efficiently on Intel’s EU/SIMD width without explicit data-layout rewriting.

### 3.3 XMX Matrix Engines Are Underutilized for Token Generation

Intel Arc has **Xe Matrix Extensions (XMX)**—analogous to NVIDIA Tensor Cores. They are used automatically by oneMKL/oneDNN for **FP16/BF16 GEMM**, which is why prompt processing (compute-bound) sees large speedups when `-DGGML_SYCL_F16=ON` is enabled (reported ~2.4× improvement: 302 → 725 t/s).

However, **token generation uses DMMV/MMVQ**, which are memory-bandwidth-bound dequantize-and-dot kernels. The SYCL backend does **not** currently route these small quantized matrix-vector operations through XMX/DPAS instructions. Developers (Codeplay/Intel contributors in Discussion #12570) note they are investigating DPAS directly, but consider the kernel "memory bound" in current configurations, so adding matrix engines may not help until memory-access patterns are fixed first.

### 3.4 Vulkan Backend Regressions and Driver Sensitivity

The Vulkan path in `llama.cpp` is maintained independently and sees recurring Intel-specific regressions:

- **Mesa version sensitivity**: Newer Mesa versions have reportedly **slowed TG on Intel** (Discussion #10879).
- **Performance degradation between builds**: A drop from 53 → 42 t/s on A770 was bisected to changes between `b7189` and `b7209` (Issue #17628).
- **Cooperative matrix TDRs**: Windows drivers (101.8509/101.8531) cause GPU timeouts when `VK_KHR_cooperative_matrix` is enabled with `llama.cpp` Vulkan (Issue #20554).
- **BMG (Battlemage) support lag**: B580 GPUs require very recent Mesa + kernel combinations; users often see fallback-to-CPU behavior or "unsupported device" warnings (Discussion #12570).

### 3.5 IPEX-LLM Deprecation and Ecosystem Fragmentation

`intel/ipex-llm` was **archived on January 28, 2026**. It was the primary documented way to run Ollama/llama.cpp on Intel Arc via Docker. Since archiving:

- Open issues (prompt-processing slowdowns, container GPU visibility, B580 model-load failures) are frozen.
- Users are left choosing between:
  1. **Upstream `llama.cpp` SYCL** (manual oneAPI setup, variable quant performance).
  2. **OpenVINO** (Intel-recommended, but not GGUF-native).
  3. **vLLM + IPEX** (deprecated; vLLM is migrating to `vllm-xpu-kernels`).
  4. **Vulkan `llama.cpp`** (easier setup, but lower peak performance and regressions).

### 3.6 Compute Runtime / Level Zero Issues

The `intel/compute-runtime` repository contains long-standing issues that affect LLM workloads:

- **4 GB allocation limit** on Arc A770 16 GB (`CL_DEVICE_MAX_MEM_ALLOC_SIZE`) — Issue #627. Large single buffers must be split or use host paging.
- **Incorrect free-memory reporting** — Issue #750. Runtime reports `free_memory == global_mem_size` even when VRAM is in use, confusing memory managers in `llama.cpp` and vLLM.
- **BAR / SVM allocation failures on Arrow Lake** — Issue #890. OEM laptops with fixed 256 MB BAR hang or fail to allocate >4 GB, verified on both Linux and Windows.
- **iGPU + dGPU conflicts**: SYCL initialization can fail or select the wrong device when both an Intel iGPU and Arc dGPU are present (Issues #13775, #9106).

### 3.7 vLLM on Intel XPU Is Still Immature

vLLM upstream added Intel XPU support, but user reports highlight:

- **B-series crashes during model inspection** (Issue #27408): `SIGABRT` inside `drm_neo.cpp` on Battlemage.
- **Dual-GPU scaling breaks TG**: Two A770s in tensor/pipeline parallel double throughput but **halve text-generation speed** to 3–4 t/s (Issue #12190).
- **AWQ/INT4 pre-quantized models fail** because the torchao codepath is CUDA-only (Issue #269 in `intel/llm-scaler`).
- **IPEX deprecation**: vLLM release notes explicitly "deprecated IPEX for XPU, switched to vllm-xpu-kernels" (v0.11.x+), but the new kernel repo is still catching up on feature parity.

---

## 4. Speculations on Root Causes

Based on the collected discourse, the following hypotheses best explain the observed gaps:

### A. Kernel Data Layouts Are CUDA-Centric
`llama.cpp`’s quantized kernels (DMMV, MMVQ) were originally written and tuned for NVIDIA warp sizes, shared-memory banking, and coalesced-load patterns. The Intel Xe architecture has **different SIMD widths, cache-line behavior, and scatter-gather characteristics**. The "reorder" fix for Q4_0 proves that **rewriting the data layout specifically for Intel** yields large gains, but this work was not systematically extended to all quant types.

### B. SYCL Is a Thick Abstraction with Opaque Performance Characteristics
Developers (both community and Intel-affiliated) note that the SYCL stack (DPC++ compiler → IGC → Level Zero) works, but makes it **difficult to reason about whether generated ISA actually uses XMX/DPAS or falls back to scalar ALU paths**. The IGC JIT compiler does auto-vectorization, but for non-power-of-two block sizes (e.g., Q8_0’s 34 bytes), the generated code may serialize loads and leave EUs idle.

### C. The Driver Stack Correctly Exposes Hardware, But Does Not Hide Its Quirks
Intel Compute Runtime is functionally correct—models load, kernels execute, and results are valid—but it does **not** provide the same transparent memory-management or profiling feedback that CUDA/Rocm drivers offer. Issues like the 4 GB `MAX_MEM_ALLOC_SIZE`, wrong free-memory reporting, and iGPU+dGPU device-selection bugs force **application-layer workarounds** that are inconsistently implemented across frameworks.

### D. Rapid Corporate Strategy Shifts Create Maintenance Debt
The shift from **BigDL-LLM → IPEX-LLM → archived IPEX-LLM**, and the parallel shift from **IPEX → native PyTorch XPU → vllm-xpu-kernels**, means optimizations and bug fixes were repeatedly abandoned mid-stream. Community contributors (e.g., Codeplay, private maintainers in Discussion #12570) are left carrying the SYCL backend in `llama.cpp` with limited resources.

---

## 5. Potential Solutions & Lines of Investigation

**No code changes were made in this phase.** The following are research-backed proposals for a subsequent implementation phase.

### 5.1 Complete the Reorder Optimization for Q8_0 and K-Quants in `ggml-sycl`
- Implement `dequantize_block_q8_0_reorder` and add Q8_0 to `ggml_sycl_supports_reorder_mmvq()`.
- Increase `iter_stride` for Q8_0 DMMV to match the 8× factor used in Q4_0 reorder kernels.
- **Expected impact:** Close the 4× TG gap between Q8_0 and Q4_K_M on Xe2 GPUs.

### 5.2 Profile SYCL Kernels with Intel-Specific Tools
- Use **Intel VTune** or **ze_tracer** on the slow Q8_0 DMMV kernel to measure:
  - EU utilization
  - L3 cache miss rates
  - Memory-latency hiding efficiency
- **Goal:** Determine whether the bottleneck is load coalescing, register pressure, or insufficient occupancy.

### 5.3 Implement DPAS/XMX Kernels for Quantized Matrix-Vector Multiplication
- Target **DPAS/DPASW** instructions directly (or via lightweight SYCL extensions) for INT8/INT4 dot products in MMVQ.
- oneDNN/oneMKL already use XMX for FP16 GEMM; the gap is in the **custom quantized DMMV/MMVQ kernels** inside `llama.cpp`.

### 5.4 Audit and Fix Intel Compute Runtime Memory Reporting
- Patch `compute-runtime` to return accurate `free_memory` via Level Zero (`zesMemoryGetState`) so that `llama.cpp` and vLLM can make correct offloading decisions without the `ZES_ENABLE_SYSMAN=1` workaround.
- Investigate the 4 GB `CL_DEVICE_MAX_MEM_ALLOC_SIZE` cap on 16 GB cards.

### 5.5 Stabilize the Vulkan Backend for Xe2
- Add BMG/B580 device IDs and tune pipeline layouts in `ggml-vulkan`.
- Bisect and revert or adapt the Mesa change that caused TG regression on Intel Arc.

### 5.6 Align vLLM XPU Quantization Roadmap
- Ensure `vllm-xpu-kernels` supports the same GGUF/AWQ/GPTQ paths that CUDA does, or at minimum documents unsupported formats clearly.
- Fix the B-series model-inspection crash (`drm_neo.cpp` abort) before broader Battlemage deployment.

---

## 6. Key Data Sources & References

| Source | Significance |
|--------|--------------|
| `ggml-org/llama.cpp#21517` | Rigorous Q8_0 bandwidth analysis on Arc Pro B70; proves kernel inefficiency |
| `ggml-org/llama.cpp#12035` | Original Q4_0 reorder optimization; template for fixing Q8_0 |
| `ggml-org/llama.cpp#12570` | Maintainer discussion on SYCL vs Vulkan, XMX, DPAS, and Codeplay roadmap |
| `ggml-org/llama.cpp#19887` | Inverse quant anomaly on A770 Vulkan (Q8_0 faster than Q4/Q6) |
| `intel/ipex-llm#12632` | Prompt-processing slowdown on Arc A380 in Docker (frozen, repo archived) |
| `intel/ipex-llm#12994` | B580 model load failures with Level Zero |
| `intel/compute-runtime#627` | 4 GB allocation limit on Arc A770 16 GB |
| `intel/compute-runtime#890` | Arrow Lake >4 GB allocation failures / system hangs |
| `vllm-project/vllm#27408` | B-series SIGABRT during model inspection |
| `vllm-project/vllm#33214` | Migration from IPEX to `vllm-xpu-kernels` |
| Reddit r/LocalLLAMA (2025) | User-reported bandwidth math, “3.1× Q8_0 speedup” thread |
| Reddit r/IntelArc | Benchmark comparisons: Vulkan vs SYCL, Ollama setup guides |

---

## 7. Conclusion

Intel Arc GPUs are **not fundamentally broken** for LLM inference, but they are **victims of incomplete kernel optimization and ecosystem churn**. The hardware has competitive bandwidth and matrix engines, yet the dominant open-source inference path (`llama.cpp` SYCL) only optimized Q4_0’s data layout, leaving Q8_0 and several K-quants on a slow generic fallback. Simultaneously, Intel’s own integration layer (`ipex-llm`) was archived while downstream projects (vLLM, PyTorch) are mid-transition to new backends.

The highest-leverage fixes, in order, appear to be:
1. **Kernel-side:** Extend the reorder/MMVD/MMVQ optimizations to Q8_0 and remaining quants in `ggml-sycl`.
2. **Driver-side:** Fix memory reporting and large-buffer allocation limits in Compute Runtime.
3. **Ecosystem-side:** Stabilize `vllm-xpu-kernels` and document a single, maintained path for Arc users.

All relevant repositories have been pulled locally for the next phase of analysis.
