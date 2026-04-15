# Empirical Findings: Arc A770 + RX 580 on llama.cpp

**Date:** April 2026
**Hardware:** AMD Ryzen 7 5800X, 32GB DDR4, Intel Arc A770 16GB, AMD RX 580 4GB
**OS:** CachyOS (Arch-based), kernel 6.19.10-1-cachyos
**Model:** Qwen3.5-35B-A3B (MoE, 35B total / 3B active) — UD-IQ4_XS (17.5 GB)

---

## 1. Hardware & Driver Setup

### Intel Arc A770

- **PCIe**: 0b:00.0 (DG2, device 56a0)
- **Driver**: `xe` kernel module — **not i915**
  - i915 causes GPU hangs on compute workloads (ecode `12:1:85def5fb`)
  - Switched via modprobe: `options i915 force_probe=!56a0` + `options xe force_probe=56a0`
- **Compute path**: SYCL/Level Zero is the **only reliable compute path**
  - Vulkan compute: **BROKEN** — `DeviceLostError` on all compute workloads, even on xe driver
  - Vulkan display: works fine (desktop environment runs on this GPU)
- **oneAPI**: basekit 2025.3.1-6, DPC++ 2025.3.2
  - 2025.0.4 is too old (missing `sycl/ext/oneapi/work_group_static.hpp`)
  - `source /opt/intel/oneapi/setvars.sh` required before cmake AND at runtime (SYCL backend crashes without it)

### AMD RX 580

- **PCIe**: 04:00.0 (Polaris10)
- **Driver**: amdgpu + RADV (Mesa Vulkan)
- **Vulkan**: works perfectly out of the box (needs `vulkan-radeon` package)
- **OpenCL**: via rusticl (ROCm broken on kernel 6.19.x)

### Multi-GPU OpenCL

```
Platform #0: Intel(R) OpenCL Graphics → Arc A770
Platform #1: rusticl → Mesa Arc A770 (DG2) + AMD RX 580 (polaris10)
```

DRI devices: `card0` = RX 580 (04:00.0), `card1` = Arc A770 (0b:00.0)
Render nodes: `renderD128` = Arc A770, `renderD129` = RX 580

---

## 2. llama.cpp Build

Built with oneAPI SYCL + Vulkan support:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DGGML_VULKAN=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release -DVulkan_INCLUDE_DIR=/usr/include \
  -DGGML_AVX2=ON -DGGML_AVX=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_SSE42=ON
cmake --build build --config Release -j$(nproc)
```

**Critical**: `icx`/`icpx` does NOT auto-detect SIMD with `GGML_NATIVE` — you must explicitly set AVX2/FMA/F16C/SSE42 flags.

---

## 3. Performance Benchmarks

### Token generation (t/s) — Qwen3.5-35B-A3B UD-IQ4_XS

| Config | Device | Prompt t/s | Gen t/s | Notes |
|--------|--------|------------|---------|-------|
| `-ngl 38 --flash-attn on` | SYCL0 (Arc A770) | 18.8 | **21.3** | Best speed, fragile — crashes if ngl too high |
| `--cpu-moe -ngl 99` | Vulkan1 (RX 580) | 42.6 | **20.3** | Best reliable config |
| `--cpu-moe -ngl 99` | SYCL0 (Arc A770) | 36.0 | 18.0 | Intel SYCL slower than AMD Vulkan |
| `--cpu-moe -ngl 99` (no AVX2) | SYCL0 (Arc A770) | 19.5 | 17.5 | AVX2 only helps prompt, not gen |
| `-ngl 0 -ot ".*_exps=SYCL0"` | SYCL0 | 15.5 | 10.3 | CPU-driven graph = slow |
| `--cpu-dense -ngl 99` | SYCL0 (Arc A770) | 27.9 | 9.0 | Inverse of cpu-moe, bad |
| Cross-GPU SYCL+Vulkan | Mixed | — | 5.5 | Garbage output, data corruption |
| Cross-GPU Vulkan+Vulkan | Mixed | — | — | DeviceLostError crash |

### Key observations

1. **Generation is memory bandwidth bound** — DDR4 ~50 GB/s is the wall for MoE experts on CPU, not GPU compute.
2. **AVX2 gives 3x prompt speedup** (19 → 57 t/s) but **zero gen improvement** — confirms bandwidth bottleneck.
3. **Q4_K_XL vs IQ4_XS = identical gen speed** — same bandwidth, different compute, same result.
4. **Q8 KV cache (`-ctk q8_0 -ctv q8_0`) = no speed difference** on RX 580.
5. **RX 580 ($40 used) beats Arc A770 ($350) on SYCL with --cpu-moe** — Intel's software stack is the bottleneck, not the hardware.
6. **Cross-GPU tensor splitting between different backends is broken** — corruption or crashes.

---

## 4. What Works, What Doesn't

### Working configurations

| Feature | Status |
|---------|--------|
| Arc A770 SYCL/Level Zero compute | Works |
| Arc A770 partial layer offload (`-ngl 38`) | Works |
| RX 580 Vulkan compute | Works perfectly |
| `--cpu-moe` with GPU for dense layers | Works, best approach for MoE |
| Flash attention on SYCL (no --cpu-moe) | Works |
| Multi-GPU OpenCL enumeration | Works |

### Broken

| Feature | Failure mode |
|---------|-------------|
| Arc A770 Vulkan compute | `DeviceLostError` GPU hangs (both i915 and xe) |
| i915 driver on Arc A770 | GPU hangs on compute (ecode `12:1:85def5fb`) |
| Flash attention on SYCL + `--cpu-moe` | Crashes on 2nd prompt (`fattn-tile.hpp:1255 fatal error`) |
| Cross-GPU SYCL+Vulkan tensor split | Data corruption, garbage output |
| Cross-GPU Vulkan+Vulkan | `DeviceLostError` crash |
| `-ngl 99` on SYCL without --cpu-moe | OOM (model 16.3GB > 15.5GB usable VRAM) |
| SYCL without setvars.sh | Crashes at startup |
| ROCm on kernel 6.19.x | Broken, use rusticl instead |

---

## 5. Qwen3.5 Model Quirks

- **`--reasoning off` required** — otherwise generates infinite empty thinking tokens (500MB+ of newlines)
- **`--flash-attn off` needed** when using `--cpu-moe` on SYCL (crashes on multi-turn)
- **`--flash-attn on` works** on SYCL with partial layer offload (`-ngl 38`)

---

## 6. Recommended Configs

### Reliable daily driver (RX 580 Vulkan)

```bash
ZES_ENABLE_SYSMAN=1 ./llama.cpp/build/bin/llama-cli \
  -m Qwen3.5-35B-A3B-UD-IQ4_XS.gguf \
  -ngl 99 --device Vulkan1 --cpu-moe \
  -c 2048 --reasoning off
```

20.3 t/s generation. Just works.

### Maximum speed (Arc A770 SYCL, fragile)

```bash
source /opt/intel/oneapi/setvars.sh && ZES_ENABLE_SYSMAN=1 \
  ./llama.cpp/build/bin/llama-cli \
  -m Qwen3.5-35B-A3B-UD-IQ4_XS.gguf \
  -ngl 38 --device SYCL0 \
  -c 4000 --reasoning off --flash-attn on
```

21.3 t/s generation. Crashes if you push ngl too high.

---

## 7. Conclusions

The Arc A770 is not fundamentally slow hardware — 512 GB/s memory bandwidth should destroy a $40 RX 580 on bandwidth-bound workloads. The fact that it doesn't tells you everything about Intel's software stack maturity:

- **SYCL/Level Zero is the only working compute path** on Linux. Vulkan compute is broken.
- **The xe driver is mandatory** — i915 GPU hangs on compute workloads.
- **oneAPI version matters enormously** — too old and the SYCL backend won't compile.
- **For MoE models, `--cpu-moe` is the only sane strategy** — it keeps the massive expert tensors on CPU (bandwidth-bound anyway) and uses GPU for the smaller dense layers.
- **Intel needs to fix Vulkan compute on Arc** — this is the biggest single blocker for mainstream llama.cpp users who don't want to deal with oneAPI.
