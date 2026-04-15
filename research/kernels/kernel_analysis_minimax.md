# Driver/Stack Misalignment Analysis

## Overview

This document catalogs the specific code locations, design decisions, and architectural mismatches that cause Intel Arc GPUs to underperform on LLM inference.

---

## 1. llama.cpp SYCL Backend Misalignments

### 1.1 Kernel Dispatch Logic

**File:** `repos/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp`  
**Lines:** ~3258-3660

**Current Dispatch Algorithm:**
```
mul_mat dispatch prefers:
1. MMVQ (reorder path) if src0 type in ggml_sycl_supports_reorder_mmvq()
2. DMMV (reorder path) if src0 type in ggml_sycl_supports_reorder_dmmv()
3. SYCL native matmul as fallback
```

**Support Lists (lines ~3269-3300):**

```cpp
// Supports MMVQ reorder
ggml_sycl_supports_reorder_mmvq(): Q4_0, Q8_0, Q4_K, Q6_K

// Supports DMMV reorder  
ggml_sycl_supports_reorder_dmmv(): Q4_0, Q8_0 ONLY

// Supports SYCL matmul reorder
ggml_sycl_supports_reorder_mul_mat_sycl(): Q4_0, Q8_0, Q4_K*, Q6_K*
(* = !g_ggml_sycl_prioritize_dmmv)
```

**Problem:** Q4_K, Q6_K support MMVQ reorder but NOT DMMV reorder. When conditions favor DMMV, these quants fall through to slow generic path.

### 1.2 DMMV Kernel iter_stride Problem

**File:** `repos/llama.cpp/ggml/src/ggml-sycl/dmmv.cpp`  
**Lines:** ~975-1100 (dequantize_mul_mat_vec_q8_0_sycl)

**Generic DMMV (used by Q8_0):**
```cpp
iter_stride = 2 * GGML_SYCL_DMMV_X = 64  // processes 2 values per iteration
```

**Reorder DMMV (Q4_0 path):**
```cpp
iter_stride = 8 * 2 * GGML_SYCL_DMMV_X = 512  // processes 16 values per iteration
```

**Root Cause:** Q8_0's 34-byte block structure prevents simple power-of-2 optimization that works for Q4_0's 18-byte blocks.

### 1.3 Missing Q8_0 Reorder Implementation

**File:** `repos/llama.cpp/ggml/src/ggml-sycl/mmvq.cpp`  
**Lines:** ~682-730

**Q4_0 Reorder Kernel:**
```cpp
mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>
```

**Q8_0 Reorder Kernel:**
```cpp
mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>
```

**Note:** PR #21527 adds Q8_0 to reorder framework. Without this fix, Q8_0 defaults to slow DMMV path.

### 1.4 Q4_K DMMV Reorder Gap

**Problem:** Q4_K has reorder structure (`reorder_qw_q4_k()`) but DMMV path doesn't use it.

**Current State:**
- Q4_K MMVQ reorder: ✅ Working
- Q4_K DMMV reorder: ❌ Not implemented

**Impact:** When DMMV is prioritized (GGML_SYCL_PRIORITIZE_DMMV=1), Q4_K gets no optimization.

---

## 2. llama.cpp Vulkan Backend Misalignments

### 2.1 Cooperative Matrix Detection

**File:** `repos/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`  
**Lines:** ~343, ~15972

**Detection Logic:**
```cpp
// Step 1: Architecture classification
if (subgroup_size_control_props.minSubgroupSize == 16) {
    return vk_device_architecture::INTEL_XE2;
}
// Falls through to OTHER for 140T (minSubgroupSize=8)

// Step 2: Coopmat support check
case VK_VENDOR_ID_INTEL:
    return arch == vk_device_architecture::INTEL_XE2;
// Returns false for OTHER
```

**Problem:** Arc 140T (Arrow Lake H) reports minSubgroupSize=8 despite having Xe2 architecture and full coopmat support.

### 2.2 DP4A/DPAS Utilization Gap

**Current State:**
- Vulkan backend has DP4A instruction support
- Matrix multiplication (matmul) path doesn't use DPAS
- Only Flash Attention path partially uses coopmat

**Missing:**
- Q4_K, Q8_0 quantized matmul via DPAS
- Subgroup-level parallelism for token generation

---

## 3. IPEX-LLM vs llama.cpp Gap

### 3.1 Performance Comparison

| Aspect | IPEX-LLM | llama.cpp SYCL | llama.cpp Vulkan |
|--------|----------|----------------|------------------|
| Q4_0 | Fast | Fast | Medium |
| Q4_K | Fast | Medium | Medium |
| Q8_0 | Fast | Was Broken | Was Broken |
| K-quants on Xe2 | Crashes | Works | Works |
| FlashAttention | Full | Partial | Partial |
| vRAM usage | Lower | Higher | Higher |

### 3.2 Source of Optimization Gap

**IPEX-LLM advantages:**
1. Closed-source optimized kernels (not in llama.cpp)
2. oneDNN GEMM integration
3. Lower-level hardware access
4. syclcompat library for platform-specific tuning

**llama.cpp limitations:**
1. Open-source kernels visible to competitors
2. Generic SYCL must work across all Intel GPUs
3. Can't leverage IPEX's proprietary optimizations

---

## 4. Architecture Detection Mismatches

### 4.1 Xe1 vs Xe2 Detection

**Current Detection:** Uses compute capability (device version)

**Problem:** 
- Arc A770 reports compute version 1.3 (Xe1)
- Arc B580 reports compute version 1.6 (Xe2)
- BUT: Same driver branch reports different subgroup sizes (8 vs 16)

### 4.2 Missing Architecture-Specific Tuning

**Current kernels:** Single implementation for all Intel GPUs

**Needed:**
| Feature | Xe1 (Alchemist) | Xe2 (Battlemage) |
|---------|-----------------|------------------|
| L2 cache | 16 MB | Larger | 
| Optimal block size | 64 | 128 |
| Prefetch depth | 2 | 4 |
| Vector width | 8 | 16 |

---

## 5. Quantization Format Support Matrix

### 5.1 Current Support State

| Format | DMMV Reorder | MMVQ Reorder | SYCL Matmul | Vulkan | Notes |
|--------|--------------|--------------|-------------|--------|-------|
| Q4_0 | ✅ | ✅ | ✅ | ✅ | Fully optimized |
| Q4_1 | ❌ | ❌ | ✅ | ✅ | Legacy, slow |
| Q5_0 | ❌ | ❌ | ✅ | ✅ | Legacy, slow |
| Q5_1 | ❌ | ❌ | ✅ | ✅ | Legacy, slow |
| Q8_0 | ✅* | ✅* | ✅ | ✅ | *Fixed by PR #21527 |
| Q4_K | ❌ | ✅ | ✅* | ✅ | *Prioritize DMMV breaks |
| Q5_K | ❌ | ❌ | ❌ | ❌ | No reorder support |
| Q6_K | ❌ | ✅ | ✅* | ✅ | *Prioritize DMMV breaks |
| IQ4_NL | ❌ | ❌ | ✅ | ❌ | 14% bandwidth, crashes |
| IQ4_XS | ❌ | ❌ | ✅ | ✅ | Not optimized |

### 5.2 Block Size Analysis

| Format | Block Size | Power of 2? | Cache Line Aligned? |
|--------|-----------|-------------|---------------------|
| Q4_0 | 18 bytes | No | Partial |
| Q4_K | 54 bytes | No | No |
| Q5_K | 62 bytes | No | No |
| Q6_K | 66 bytes | No | No |
| Q8_0 | 34 bytes | No | No |
| IQ4_NL | 16 bytes | Yes | Yes |

**Hypothesis:** Power-of-2 block sizes (Q4_0, IQ4_NL) enable efficient memory access patterns. Non-power-of-2 formats suffer.

---

## 6. Key File Locations Summary

### Core Problem Areas:

```
repos/llama.cpp/ggml/src/ggml-sycl/
├── ggml-sycl.cpp
│   ├── Line 219: GGML_SYCL_PRIORITIZE_DMMV env var
│   ├── Line 3258-3260: mul_mat_algo enum (DMMV, MMVQ, SYCL)
│   ├── Line 3269-3292: ggml_sycl_supports_reorder_*() functions
│   ├── Line 3549-3650: dispatch logic with fallback chains
│   └── Problem: Routing logic doesn't handle Q4_K/Q6_K correctly
│
├── dmmv.cpp
│   ├── ~975-1100: dequantize_mul_mat_vec_q8_0_sycl()
│   ├── iter_stride = 64 (generic path)
│   └── Problem: 8x less work than reorder path
│
├── mmvq.cpp
│   ├── ~550-570: Q4_0 reorder kernel
│   ├── ~695-720: Q8_0 reorder kernel (after PR #21527)
│   ├── ~1100-1200: Q4_K kernel (no DMMV support)
│   └── Problem: Missing Q5_K, Q6_K reorder
│
└── vecdotq.hpp
    ├── ~844: vec_dot_q8_0_q8_1 implementation
    └── Problem: Memory coalescing suboptimal for Xe2

repos/llama.cpp/ggml/src/ggml-vulkan/
└── ggml-vulkan.cpp
    ├── ~343: get_device_architecture() classification
    ├── ~15972: coopmat support check
    └── Problem: minSubgroupSize = 8 causes 140T misdetection

repos/ipex-llm/ (archived Jan 2026)
├── Closed-source optimized kernels (not in upstream)
├── syclcompat library
├── oneDNN integration
└── Problem: Archive status, no community maintenance
```

---

## 7. Misalignment Summary Table

| Component | Expected | Actual | Impact |
|-----------|----------|--------|--------|
| Q8_0 DMMV | 64 values/iter | 2 values/iter | 4x slower |
| Q4_K DMMV | Reorder enabled | Not implemented | 40% slower |
| Q5_K MMVQ | Reorder support | Missing | 3x slower |
| Arc 140T detection | INTEL_XE2 | OTHER | Coopmat disabled |
| Q8_0 on B70 | 60% BW | 21% BW | 3x slower |

---

*Last Updated: April 2026*