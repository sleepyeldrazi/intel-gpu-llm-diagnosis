# Intel Arc GPU LLM Inference: Verified Findings

**Date:** April 15, 2026  
**Method:** Cross-checked all three agent overviews and the `research/` documents against live GitHub issues/PRs and the actual source code in `repos/`. Claims marked ✅ were verified; ⚠️ were partially correct or nuanced; ❌ were wrong.

---

## 1. The Core Problem Is Confirmed

All three overviews agree and the benchmark data holds up: **Intel Arc GPUs (Xe1/Alchemist and Xe2/Battlemage) achieve only 21–53% of theoretical memory bandwidth during quantized token generation**, versus 80–95% on NVIDIA/AMD hardware at equivalent price points.

This is not a driver bug. Updating IGC from 2.28.4 → 2.30.1 showed no change in Q8_0 performance (confirmed in issue #21517). The bottleneck is in the inference-framework kernel layer, not the hardware or compiler.

---

## 2. The Q8_0 Catastrophe (Mostly Fixed)

**Status:** ✅ Confirmed and code-verified. PR #21527 is merged; the fix is present in the repo.

Issue #21517 (`ggml-org/llama.cpp`) documented Q8_0 achieving only **21% memory bandwidth** (4.88 t/s) vs Q4_K_M at 53% (20.56 t/s) on Arc Pro B70 — a 4× gap despite only 1.7× more data. Both SYCL and Vulkan backends were equally affected. The root cause was confirmed in the dispatch logic:

**Verified in `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 3282–3302):**

```
ggml_sycl_supports_reorder_dmmv():  Q4_0, Q8_0      ← Q8_0 added by PR #21527
ggml_sycl_supports_reorder_mmvq():  Q4_0, Q8_0, Q4_K, Q6_K
ggml_sycl_supports_reorder_mul_mat_sycl(): Q4_0, Q8_0; Q4_K/Q6_K only when !prioritize_dmmv
```

Before PR #21527, Q8_0 was missing from both lists, forcing it through the slow generic DMMV path. After the fix, Q8_0 achieves ~66% bandwidth (15.24 t/s, 3.1× speedup). The `dequantize_q8_0_reorder` function exists in `dequantize.hpp:147`, and `reorder_mul_mat_vec_q8_0_q8_1_sycl` exists in `mmvq.cpp:682`.

**The iter_stride gap is real (verified):** The generic DMMV uses `iter_stride = 2 * GGML_SYCL_DMMV_X` (=64); the reorder path uses `iter_stride = 8 * 2 * GGML_SYCL_DMMV_X` (=512). That is an 8× difference. `GGML_SYCL_DMMV_X` is defined as 32; for Intel targets, `GGML_SYCL_WARP_SIZE` is compiled to 16 (not 32 as some overview text implies), so `vals_per_iter` = 4 (generic) and 32 (reorder) — the 8× ratio holds regardless.

---

## 3. Remaining Quantization Gaps

**Status:** ✅ Confirmed and code-verified.

Verified in `dmmv.cpp` and `mmvq.cpp`:

| Format | DMMV Reorder | MMVQ Reorder | Notes |
|--------|:---:|:---:|-------|
| Q4_0 | ✅ | ✅ | Fully optimized (PR #12035, Feb 2025) |
| Q8_0 | ✅ | ✅ | Fixed by PR #21527 |
| Q4_K | ❌ (ABORT) | ✅ | DMMV reorder hits `GGML_ABORT` — unimplemented; `ggml-sycl.cpp:1238–1239` |
| Q6_K | ❌ | ✅ | MMVQ reorder exists (`mmvq.cpp:842`); DMMV has no reorder |
| Q5_K | ❌ | ❌ | Generic MMVQ only (`mmvq.cpp:835`); no reorder path of any kind |
| Q4_1, Q5_0, Q5_1 | ❌ | ❌ | Generic paths only |
| IQ4_NL | ❌ | ❌ | Only appears in SYCL native matmul path; 14% bandwidth figure is plausible |

K-quants still missing DMMV reorder means any configuration that enables `GGML_SYCL_PRIORITIZE_DMMV` (or falls through to DMMV due to matrix dimensions) will use the 8× slower generic path for Q4_K and Q6_K.

**Correction to `research/kernels/kernel_analysis_minimax.md`:** The Block Size table in that document contains **wrong values** for K-quants. Actual values from `ggml-common.h` (with `QK_K=256`, `K_SCALE_SIZE=12`):

| Format | Claimed | Actual |
|--------|---------|--------|
| Q4_0 | 18 bytes | **18 bytes** ✓ (`sizeof(ggml_half) + QK4_0/2`) |
| Q8_0 | 34 bytes | **34 bytes** ✓ (`sizeof(ggml_half) + QK8_0`) |
| Q4_K | 54 bytes | **144 bytes** (`2*2 + 12 + 128`) |
| Q5_K | 62 bytes | **176 bytes** (`2*2 + 12 + 128 + 32`) |
| Q6_K | 66 bytes | **210 bytes** (`2 + 16 + 192`) |

The broader hypothesis — that non-power-of-2 block sizes impede vectorization — remains valid. All sizes are non-power-of-2. But the specific numbers for K-quants were hallucinated.

---

## 4. Architecture Detection Bug (Vulkan / Arc 140T)

**Status:** ✅ Confirmed, both code-verified and issue-confirmed (#20776, open).

The Vulkan backend classifies Intel GPUs in `ggml-vulkan.cpp:329–356`. Detection of `INTEL_XE2` requires `minSubgroupSize == 16`. Intel itself documents that Xe2 uses SIMD16 while older architectures use SIMD8 (the comment in the code links to Intel's 2024 Tech Tour). Everything else falls through to `OTHER`, and cooperative matrix support is disabled for non-XE2 Intel (`ggml-vulkan.cpp:5119–5120`, `5543`).

The Arc 140T (Arrow Lake H) reports `minSubgroupSize = 8` despite being Xe2 hardware. This appears to be a driver-level reporting inconsistency: the 140V on Lunar Lake correctly reports 16. The result is that coopmat and all dependent optimizations are silently disabled on 140T systems.

MiniMax's proposed device-ID override fix is reasonable:
```cpp
if (props.deviceID == 0x7D51 || props.deviceID == 0x7D45) {
    return vk_device_architecture::INTEL_XE2;
}
```
This is a targeted workaround; a systemic fix would require Intel to correct the driver-reported value.

---

## 5. K-Quant Crash on Xe2 iGPU (IPEX-LLM)

**Status:** ✅ Confirmed (issue #12318, `intel/ipex-llm`).

IPEX-LLM's bundled llama.cpp is based on an August 2024 snapshot. Q4_K_M and other K-quants crash on Arc 140V with `Sub-group size 8 is not supported on the device`. Upstream llama.cpp does not have this crash. Since ipex-llm is archived (see §7), this will not be fixed.

---

## 6. XMX/DPAS Underutilization

**Status:** ✅ Confirmed as a real gap; specific DPAS investigation is ongoing with no timeline.

XMX (Xe Matrix eXtensions) are used for FP16/BF16 GEMM via `joint_matrix` in the SYCL backend — confirmed by the 2.4× prompt-processing speedup with `-DGGML_SYCL_F16=ON`. They are **not** used for quantized token-generation kernels (DMMV/MMVQ), which are the actual bottleneck.

Rbiessy (Codeplay) confirmed in discussion #12570 that using XMX/DPAS for quantized kernels requires fixing memory-access patterns first: "currently the kernel is memory bound in the configurations we have tried." The reorder fix for Q8_0 was the necessary prerequisite; DPAS integration may follow but has no committed timeline.

---

## 7. IPEX-LLM Archival and Ecosystem Fragmentation

**Status:** ✅ Confirmed. ipex-llm `README.md` begins: "THIS PROJECT IS ARCHIVED."

Intel archived `intel/ipex-llm` on January 28, 2026. IPEX-LLM had measurable advantages over upstream llama.cpp SYCL — the GLM overview's data for Arc 140V (Llama-2-7B Q4_0: IPEX-LLM 24.35 t/s vs SYCL FP16 13.51 t/s, an ~80% advantage) is consistent with independent reports and a plausible consequence of IPEX-LLM's closed-source oneDNN integration and lower-level kernel access.

The optimizations that produced this gap were never upstreamed. The community must re-derive them independently.

The five partially-overlapping Intel inference stacks (llama.cpp SYCL, llama.cpp Vulkan, IPEX-LLM, OpenVINO, vLLM XPU) remain. They use different oneAPI and PyTorch versions and target different benchmarks. There is no unified Intel strategy for community-facing LLM inference.

---

## 8. Linux Kernel 6.18 Break

**Status:** ✅ Confirmed. `intel/compute-runtime` issue #875 documents this.

After upgrading to Linux kernel 6.18, device memory allocation via Level Zero fails with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`. Kernel 6.17 works. The failure is in `bindless_heaps_helper.cpp` (the file exists in `repos/compute-runtime/shared/source/helpers/`). Users on rolling-release distributions (Arch, Fedora) are most exposed. The working range through 6.17 is wide, but the 6.18 cliff is a sharp regression.

---

## 9. vLLM Intel XPU

**Status:** ✅ Issue #27408 confirmed (B-series SIGABRT during model inspection).

vLLM's Intel XPU backend is in active transition: it has officially deprecated IPEX in favor of `vllm-xpu-kernels` (v0.11.x+). The new kernel repo is not yet at feature parity. AWQ/GPTQ formats remain CUDA-only (torchao codepath); GGUF is unsupported in vLLM entirely. The Arc Pro B-series crash (#27408) occurs during model inspection on Battlemage hardware.

---

## 10. Summary: What Is Solid vs. Uncertain

### Solid (code-verified or multi-source confirmed)

- Q8_0 performance disaster on Xe2 and its root cause (iter_stride / missing reorder path)
- PR #21527 fix: 3.1× speedup for Q8_0, merged
- Q4_K DMMV reorder is unimplemented (GGML_ABORT in the code)
- Q5_K has no reorder support at all
- Arc 140T Vulkan misdetection mechanism (minSubgroupSize=8 vs 16 check)
- ipex-llm archived Jan 28, 2026
- compute-runtime kernel 6.18 break confirmed
- 34-byte Q8_0 block structure; 18-byte Q4_0 block structure

### Reasonable / Well-Supported But Not Independently Re-Benchmarked

- Specific t/s numbers from community benchmarks (20.56 Q4_K_M, 4.88 Q8_0 pre-fix) — these appear in a real GitHub issue
- ~80% IPEX-LLM advantage over upstream SYCL on Xe2 iGPU
- IQ4_NL at 14% bandwidth — stated in discussion #12570 by a community contributor
- Xe2-specific regression for Q8_0 (inverse: Q8_0 faster than Q4 on Xe1 per issue #19887)

### Flagged as Incorrect (Research Document Only)

- `research/kernels/kernel_analysis_minimax.md` Block Size table for K-quants (Q4_K/Q5_K/Q6_K sizes are wrong — see §3)

### Not Independently Verified

- compute-runtime issues #627 (4 GB allocation cap), #750 (free memory reporting), #890 (Arrow Lake OOM hangs) — referenced only in Kimi's overview
- Mesa version sensitivity in Vulkan TG on Intel (referenced as Discussion #10879)
- Xe2-specific microarchitecture differences (L2 size, prefetch depth) — Intel has not published a detailed Xe2 optimization guide; the figures cited by GLM are extrapolated

---

*No code or driver modifications were made in producing this document.*
