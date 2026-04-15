# Intel Arc GPU — LLM Inference Diagnosis

Research into why Intel Arc GPUs (Alchemist / Xe1 and Battlemage / Xe2) severely underperform on quantized LLM inference, often achieving only **21–40% of theoretical memory bandwidth** during token generation — compared to 80–95% on equivalent NVIDIA and AMD hardware.

## Key Result: +19% Q4_0 Token Generation

Through a 3-model council (GLM-5.1, Minimax-M2.7, Kimi k2p5) analyzing llama.cpp SYCL kernel performance on an **Intel Arc A770 16GB**, we identified that Q4_0 token generation was **dp4a-compute-bound** (not memory-bandwidth-bound as previously assumed) and achieved a **+19% improvement** (29.4 → 35.96 t/s) by tuning the `vdr_mmvq` parameter from 2 → 4.

| Config | Q4_0 tg128 | Q4_0 BW% | Q8_0 tg128 | Q8_0 BW% | Q4_K_M tg128 |
|--------|-----------|----------|-----------|----------|-------------|
| Baseline (HEAD) | 29.4 | 29% | 28.6 | 29% | — |
| **+Phase 5 (vdr_mmvq)** | **35.96** | **35%** | **30.82** | **32%** | 25.32 |

See [`repos/patch/README.md`](repos/patch/README.md) for full benchmark methodology and results.

## The Problem

Intel Arc GPUs look great on paper for LLM inference: ample VRAM, wide memory buses, dedicated XMX matrix engines. In practice, community benchmarks consistently show:

- **Q8_0 quantized models running 4–5× slower** than Q4_K_M despite only moving 1.7× more data
- Token generation achieving only **21% of peak bandwidth** on some quantization types
- Wildly inconsistent performance across SYCL, Vulkan, OpenVINO, and IPEX-LLM backends
- Architecture-specific regressions on Xe2 (Battlemage) that don't exist on Xe1 (Alchemist)

The root causes are multi-layered: missing kernel optimizations in `llama.cpp`, a fragmented Intel software stack (five semi-independent efforts that don't interoperate), quantization-specific dispatch path bugs, and an overall underinvestment in open-source kernel development for Intel GPU architectures.

## Empirical Findings

- **[Empirical Findings](empirical_findings.md)** — Real-world benchmarks and configurations from an Arc A770 + RX 580 system running llama.cpp with Qwen3.5-35B-A3B MoE. Includes driver setup (xe vs i915), SYCL/Vulkan status, performance tables, and working/broken configuration matrix.
- **[SYCL Optimization Analysis](sycl_optimization_analysis.md)** — Deep-dive into why the SYCL backend is slow: 6 root causes (double-buffered memory, disabled graph execution, blocking `.wait()` calls, DPCT translation artifacts), Vulkan vs SYCL submission architecture comparison, kernel dispatch issues, and a prioritized improvement roadmap.

## Root Cause Analysis

**[`logs/root-cause-analysis-20260415.md`](logs/root-cause-analysis-20260415.md)** — Corrected root cause for Q4_0 underperformance. Previous analysis blamed SYCL submission model overhead; empirical profiling proved this **wrong**. The real bottleneck:

1. Q4_0 nibble packing requires **2 dp4a operations per byte** (low + high nibbles), while Q8_0 needs only 1 dp4a per byte
2. Both formats hit the same dp4a throughput ceiling → same ~30 t/s despite Q8_0 reading 1.76× more data
3. The SYCL queue naturally batches async submissions (CPU submits 1077 ops in 7.5ms vs 32ms GPU execution) — the GPU is never starved
4. XMX/DPAS matrix units are **not used** for quantized kernels — only integer dp4a through the EU datapath

## Patch Development (Council)

A 3-model council (GLM-5.1, Minimax-M2.7, Kimi k2p5) developed and cross-reviewed patches through a phased system:

- **[`repos/patch/README.md`](repos/patch/README.md)** — Patch phases, benchmark results, and status
- **[`logs/workplan.md`](logs/workplan.md)** — Council structure, testing protocol, and guidelines
- **[`logs/decisions.md`](logs/decisions.md)** — All council decisions with rationale

### Patch Phases Summary

| Phase | Change | Result |
|-------|--------|--------|
| 1 — SYCL Graph Default | Enable graph by default | ⚠️ **Crashes on MoE** (`async_malloc` failure). Original disabled default was correct. |
| 2 — Kernel Tuning | Fix VER_GEN thresholds, DMMV tuning | ✅ Neutral on 9B dense |
| 3 — Vulkan Arc 140T | Xe2 device-ID override | ⏳ Not tested (missing spirv-headers) |
| 4 — Host-Buffer Copy | Remove blanket Linux double-copy | ✅ Neutral on 9B dense |
| **5 — Q4_0 vdr_mmvq** | **vdr_mmvq 2→4 for Q4_0 reorder** | **✅ +19% Q4_0 tg128** |

Further improvements require DPAS/XMX integration or algorithmic changes to the nibble dot-product. See the [Next Steps](#next-steps) section below.

## Summary of Findings

**[`overview.md`](overview.md)** — Cross-verified synthesis of all three agent overviews. Every major claim was checked against live GitHub issues/PRs and the actual source code in `repos/`. Includes confirmed findings, one correction to a research document (K-quant block sizes), and a clear breakdown of what is solid vs. uncertain.

## Overviews

Each overview was independently produced by a different LLM, analyzing community issues, kernel source code, driver stacks, and benchmark data:

- **[Kimi's Overview](overview_kimi.md)** — Focuses on driver/runtime stack mapping, quantization kernel inefficiencies (DMMV vs. MMVQ paths), and the missing reorder optimization for Q8_0.
- **[GLM's Overview](overview_glm.md)** — Broadest scope: full stack architecture diagram, version compatibility matrix, fragmentation analysis across five Intel inference stacks, and the Battlemage regression class.
- **[MiniMax's Overview](overview_minimax.md)** — Hardware landscape, per-GPU status table, critical issue triage (Q8_0 catastrophe, iGPU misdetection), and kernel-level root cause analysis.

## Research

Supporting deep-dives in [`research/`](research/):

- [`research/kernels/kernel_analysis_minimax.md`](research/kernels/kernel_analysis_minimax.md) — Detailed kernel dispatch path analysis
- [`research/community_issues/issues_and_discourse_minimax.md`](research/community_issues/issues_and_discourse_minimax.md) — Curated community issue reports and discourse

## Council Deliberations

Internal council analysis files (in `logs/`, gitignored):

- `logs/M-sync-overhead-*.md` — Agent-M SYCL submission analysis
- `logs/K-kernel-tuning-*.md` — Agent-K kernel tuning analysis
- `logs/M-review-K-*.md`, `logs/K-review-M-*.md` — Cross-reviews
- `logs/benchmark-research.md` — Benchmark methodology research

## Repo Map

The `repos/` directory contains source clones of the relevant Intel GPU and LLM inference projects for offline analysis (not tracked in this repository):

| Repository | Purpose |
|---|---|
| `llama.cpp` | SYCL & Vulkan backends, GGUF quantization kernels |
| `ipex-llm` | Intel's former PyTorch integration layer (archived Jan 2026) |
| `intel-extension-for-pytorch` | PyTorch XPU extension (deprecated) |
| `compute-runtime` | Intel Level Zero / OpenCL driver (NEO) |
| `intel-graphics-compiler` | JIT compiler (SYCL → Xe ISA) |
| `oneDNN` | Deep-learning primitive library |
| `vllm` | vLLM mainline (XPU backend in flux) |
| `vllm-xpu-kernels` | Dedicated Intel kernel repo for vLLM |
| `level-zero` | Level Zero loader and headers |
| `llvm` | DPC++ / SYCL compiler toolchain |
| `openvino` | Intel's inference optimizer/runtime |
| `sycl-tla` | SYCL abstraction layer |

## Next Steps

- **Phase 6+ (Deferred):** Q4_K / Q6_K DMMV reorder, Q5_K reorder, DPAS/XMX integration for quantized kernels
- **Upstream contributions:** Patches prepared against llama.cpp HEAD, ready for submission
- **Key blocker for further gains:** DPAS/XMX integration requires substantial kernel rewrites

## License

This research documentation is released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Referenced repositories carry their own licenses.
