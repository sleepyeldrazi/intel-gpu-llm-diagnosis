# Why SYCL Is Slow on Intel Arc — And How to Fix It

**Date:** April 2026
**Source:** Hands-on debugging sessions with Arc A770 + RX 580, llama.cpp source analysis
**Hardware:** Intel Arc A770 16GB (xe driver), AMD RX 580 4GB (Vulkan/RADV)

---

## 1. The Core Problem

A $40 used RX 580 running Vulkan (20.3 t/s) beats a $350 Arc A770 running SYCL (18.0 t/s) on the same MoE model with `--cpu-moe`. The Arc A770 has 512 GB/s memory bandwidth vs the RX 580's ~256 GB/s. The hardware is not the problem. The SYCL backend is.

---

## 2. Why the SYCL Backend Is Slow (6 Root Causes)

### 2.1 Memory Transfers Double-Buffered Through Host

The SYCL backend copies `mmap → host buffer → device` as a workaround for PVC/Arc bugs. That's an extra memcpy on every model load and tensor transfer. The Vulkan backend doesn't do this.

### 2.2 SYCL Graph Execution Disabled by Default

Even with `GGML_SYCL_GRAPH=ON` at CMake time, the runtime **still defaults to graphs disabled** (`GGML_SYCL_DISABLE_GRAPH=1`). Without graphs, async memory operations are also disabled. Every kernel launch becomes a synchronous round-trip.

```bash
# Force enable at runtime (untested for stability on Arc):
export GGML_SYCL_DISABLE_GRAPH=0
```

### 2.3 Blocking `.wait()` Calls Everywhere

The SYCL backend has blocking synchronization after nearly every operation. No overlapping of compute and data transfer. Each `.wait()` goes through the OS scheduler — a full context switch with **10-100μs overhead per call**. With `.wait()` after damn near every memory operation, it's death by a thousand synchronizations.

### 2.4 Warp Size Mismatch (CUDA Port Artifacts)

Arc A770 uses subgroup size 16, but the SYCL backend was ported from CUDA (warp size 32) via Intel's DPCT tool. The translation artifacts are visible:

- `__ldg()` optimizations replaced with direct dereferences
- Local memory barriers used instead of more efficient fence operations
- Hardware-specific tuning paths are all stubs:

```cpp
#define VER_GEN9 700         // todo for hardware optimize.
#define VER_GEN12 1000000    // todo for hardware optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.
```

### 2.5 oneDNN Integration Is Half-Baked

Batched matmul requires contiguous tensors, and known failing configs are just `TODO`'d out.

### 2.6 MoE Is the Worst Case for SYCL

Lots of small expert dispatches instead of fat matmuls. Each one hits the kernel launch overhead + synchronization penalty. This is why `--cpu-moe` is faster — keeping experts on CPU avoids the per-expert SYCL dispatch overhead entirely.

---

## 3. Vulkan vs SYCL: Submission Architecture

This is the architectural difference that explains most of the performance gap.

### Vulkan: Smart Batching

The Vulkan backend **records 100+ operations into a command buffer before submitting to the GPU**:

```
CPU records ops → batch of ~100 nodes → ONE queue submit → GPU processes batch
CPU records NEXT batch while GPU is still working on the previous one
```

It tracks matmul bytes and dynamically adjusts batch size (~100MB per submit, scaled to model size). The synchronization uses fence-based spin-waiting with `_mm_pause()` — microsecond-level latency.

### SYCL: Sequential Submission

The SYCL backend has two modes:

1. **Graph mode** — records everything, finalizes, submits the whole graph at once. Sounds good in theory, but the SYCL graph API is immature and driver optimization is poor.
2. **Fallback mode** (the default) — loops through nodes one at a time, calling `ggml_sycl_compute_forward()` per node, with `.wait()` after each.

### The Comparison

| Aspect | Vulkan | SYCL |
|--------|--------|------|
| Submission | ~100 ops batched per submit | 1 op per submit (or monolithic graph) |
| Sync latency | `_mm_pause()` spin (~1μs) | OS `.wait()` (~10-100μs) |
| Memory copy | Async, pinned, pipelined | Sync, double-buffered, blocking |
| Subgroup tuning | 64-wide locked for GCN4 | 16-wide for Arc, generic |
| Shaders | Hand-written GLSL→SPIR-V | DPCT-translated CUDA→SYCL |

---

## 4. Arc A770 Vulkan Is Also Crippled

The Vulkan backend isn't a free win either. Arc A770 (Alchemist/Xe1) is classified as architecture `OTHER`:

### Cooperative Matrix Explicitly Disabled

```cpp
case VK_VENDOR_ID_INTEL:
    // Only allowing Xe2 GPU at the moment since Xe2 GPU can gain significant
    // performance boost, while some older hardware (ex. Arc A770) has
    // performance regressions
    return arch == vk_device_architecture::INTEL_XE2;
```

Xe2 (Battlemage, B580 etc.) gets `INTEL_XE2` with full optimization and cooperative matrix. Everything older, including the A770, gets `OTHER` — no coopmat, no custom warptiles.

### Subgroup Collectives Disabled for ALL Intel GPUs

```cpp
// Subgroup collectives explicitly disabled for all Intel GPUs
// Reference: PR #14316
```

### MMVQ Restricted on Linux

For Intel on Linux:
- Q4_0 and Q5_1: **MMVQ disabled entirely** (performance regression on A770)
- Other types: only allowed if `k >= 2048`

### Arc A770 Vulkan Compute: Broken Entirely

On our test system, Vulkan compute on Arc A770 produces `DeviceLostError` GPU hangs — both on i915 and xe drivers. This means SYCL/Level Zero is the **only** working compute path. We can't even use the Vulkan backend that has the better submission architecture.

---

## 5. DMMV/MMVQ Kernel Dispatch Issues

### The Dispatch Chain

`ggml-sycl.cpp` lines 3543-3599 check conditions in sequence:

1. F16 permuted tensors (KQ/KQV patterns)
2. F16 non-contiguous (KQV single-batch)
3. F16 multi-batch
4. Dequantize mul mat vec (**DMMV**)
5. Mul mat vec quantized (**MMVQ**) with optional reordering
6. Mul mat quantized (**MMQ**)
7. Fallback generic multiplication

### Reorder Dispatch Conflict

```cpp
// Dispatch becomes obscure with the reorder: MMVQ when reorder optimization
// is enabled takes precedence over DMMV
if (!g_ggml_sycl_prioritize_dmmv &&
    ((should_reorder_tensor(ctx, dst) &&
      ggml_sycl_supports_reorder_mmvq(src0->type)))) {
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
}
```

The reorder optimization can force the slower MMVQ path instead of DMMV, regardless of actual performance on the hardware.

### Suboptimal Defaults

```cpp
#define GGML_SYCL_DMMV_X 32    // threads per iteration — conservative
#define GGML_SYCL_MMV_Y 1      // batch size — minimal
```

These are not tuned for Arc A770. Nobody has profiled what the optimal values are.

### Quantization Reorder Support Matrix

| Format | DMMV Reorder | MMVQ Reorder | Status |
|--------|:---:|:---:|--------|
| Q4_0 | ✅ | ✅ | Fully optimized (iter_stride=512) |
| Q8_0 | ✅* | ✅* | *Fixed by PR #21527 (3.1x speedup) |
| Q4_K | ❌ | ✅ | DMMV falls to generic path (iter_stride=64) |
| Q5_K | ❌ | ❌ | No reorder at all |
| Q6_K | ❌ | ✅ | DMMV falls to generic path |
| IQ4_XS | ❌ | ❌ | Not optimized |
| IQ4_NL | ❌ | ❌ | 14% bandwidth utilization |

The generic DMMV path processes 2 values/iteration (iter_stride=64). The reorder path processes 16 values/iteration (iter_stride=512). That's an **8x difference** in work per thread.

---

## 6. xe Driver Memory Query Issue

`ext_intel_free_memory` is not in the device aspect list — the xe kernel driver doesn't expose the sysman memory interface that Level Zero needs. `ZES_ENABLE_SYSMAN=1` only works with the i915 driver, not xe.

When the free memory query fails, the allocator sets `free_memory = total_memory`, causing it to be too optimistic about what fits in VRAM. This contributes to OOM issues with `-ngl 99` on models near the 16GB boundary.

---

## 7. The `--cpu-dense` Experiment (Failed)

We added a `--cpu-dense` flag (inverse of `--cpu-moe`) to keep dense layers on CPU and experts on GPU:

```cpp
// common/common.h
const char * const LLM_DENSE_REGEX = "^(?!.*\\.ffn_(up|down|gate|gate_up)_(ch|)exps).*$";
```

**Result:** 9.0 t/s — much worse than `--cpu-moe` (18.0 t/s).

**Why:** With `--cpu-dense`, the CPU is in the critical execution path doing synchronous round-trips per layer. With `--cpu-moe`, the GPU drives the graph linearly and just fetches expert weight data from CPU RAM, which pipelines better. The MoE dispatch overhead is lower on CPU (no kernel launch penalty) while the dense path benefits from GPU parallelism.

---

## 8. P2P Transfer Gap

llama.cpp doesn't implement peer-to-peer GPU transfers. Cross-device tensor movement always copies `GPU → CPU RAM → GPU` — two PCIe hops. This kills any cross-GPU tensor splitting strategy:

- SYCL+Vulkan cross-GPU: **data corruption, garbage output** (5.5 t/s)
- Vulkan+Vulkan cross-GPU: **DeviceLostError crash**

The bandwidth bottleneck is the same whether using `--cpu-moe` or GPU-GPU split: PCIe 3.0/4.0 round trip through CPU RAM per layer.

---

## 9. Improvement Roadmap

### Priority 1: Reduce Synchronization Overhead (Biggest Win)

1. **Enable SYCL graph execution by default** on Arc GPUs (if `ext_oneapi_limited_graph` supported)
2. **Remove unnecessary `.wait()` calls** — use event-based dependencies instead of blocking sync
3. **Enable async memory operations** independently of graph support
4. **Implement batched submission** similar to Vulkan's ~100-ops-per-submit approach

### Priority 2: Tune Kernel Parameters for Arc

1. **Profile and tune `GGML_SYCL_DMMV_X` and `GGML_SYCL_MMV_Y`** specifically for Arc A770
2. **Implement hardware-specific code paths** (the VER_GEN13 stubs that are all `// todo`)
3. **Fix subgroup size handling** — Arc uses 16, code assumes 32 in many places

### Priority 3: Extend Reorder Optimization

1. Add Q4_K DMMV reorder (currently has MMVQ reorder but not DMMV)
2. Add Q5_K to reorder framework (no reorder at all currently)
3. Add Q6_K DMMV reorder (currently has MMVQ only)
4. Increase generic DMMV iter_stride from 64 toward 512

### Priority 4: Fix Memory Path

1. **Eliminate host-buffer double-copy** on Arc if driver bugs are resolved
2. **Fix xe driver memory query** so `free_memory` reports correctly
3. **Implement P2P transfers** for cross-GPU splitting (long-term)

### Priority 5: Vulkan Backend for Arc A770

1. **Enable cooperative matrix on Xe1** (currently disabled due to "performance regressions" — needs profiling to see if the regressions still exist)
2. **Re-enable subgroup collectives** for Intel GPUs
3. **Fix Vulkan compute on Arc** — the DeviceLostError issue blocks any Vulkan usage
4. **Remove MMVQ restrictions** for Q4_0/Q5_1 on Intel Linux (or at least benchmark to validate they're still needed)

---

## 10. Key Takeaways

1. **The SYCL submission model is the #1 bottleneck** — 1-op-at-a-time with OS-level `.wait()` vs Vulkan's batched submission with spin-wait. This alone likely explains 30-50% of the performance gap.

2. **Arc A770 is crippled in BOTH backends** — SYCL has the submission problem, Vulkan has coopmat/collectives disabled AND compute is broken entirely. There's no good path right now.

3. **The code is a DPCT port with TODO stubs** — it was mechanically translated from CUDA, not written for Intel hardware. The hardware-specific tuning was never done.

4. **MoE models expose SYCL's weakness the most** — many small kernel dispatches hit the synchronization overhead hardest. `--cpu-moe` is a workaround, not a solution.

5. **Intel needs to actually invest in this** — the backend is maintained by spare-time volunteers. The DPCT translation stubs, the disabled optimizations, the double-buffered memory path — these all need engineering time that the community doesn't have.
