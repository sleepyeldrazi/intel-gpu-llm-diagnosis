# Intel Arc GPU — LLM Inference Diagnosis

Research into why Intel Arc GPUs (Alchemist / Xe1 and Battlemage / Xe2) severely underperform on quantized LLM inference, often achieving only **21–40% of theoretical memory bandwidth** during token generation — compared to 80–95% on equivalent NVIDIA and AMD hardware.

## The Problem

Intel Arc GPUs look great on paper for LLM inference: ample VRAM, wide memory buses, dedicated XMX matrix engines. In practice, community benchmarks consistently show:

- **Q8_0 quantized models running 4–5× slower** than Q4_K_M despite only moving 1.7× more data
- Token generation achieving only **21% of peak bandwidth** on some quantization types
- Wildly inconsistent performance across SYCL, Vulkan, OpenVINO, and IPEX-LLM backends
- Architecture-specific regressions on Xe2 (Battlemage) that don't exist on Xe1 (Alchemist)

The root causes are multi-layered: missing kernel optimizations in `llama.cpp`, a fragmented Intel software stack (five semi-independent efforts that don't interoperate), quantization-specific dispatch path bugs, and an overall underinvestment in open-source kernel development for Intel GPU architectures.

## Empirical Findings

- **[Empirical Findings](empirical_findings.md)** — Real-world benchmarks and configurations from an Arc A770 + RX 580 system running llama.cpp with Qwen3.5-35B-A3B MoE. Includes driver setup (xe vs i915), SYCL/Vulkan status, performance tables, and working/broken configuration matrix.

## Overviews

Each overview was independently produced by a different LLM, analyzing community issues, kernel source code, driver stacks, and benchmark data:

- **[Kimi's Overview](overview_kimi.md)** — Focuses on driver/runtime stack mapping, quantization kernel inefficiencies (DMMV vs. MMVQ paths), and the missing reorder optimization for Q8_0.
- **[GLM's Overview](overview_glm.md)** — Broadest scope: full stack architecture diagram, version compatibility matrix, fragmentation analysis across five Intel inference stacks, and the Battlemage regression class.
- **[MiniMax's Overview](overview_minimax.md)** — Hardware landscape, per-GPU status table, critical issue triage (Q8_0 catastrophe, iGPU misdetection), and kernel-level root cause analysis.

## Research

Supporting deep-dives in [`research/`](research/):

- [`research/kernels/kernel_analysis_minimax.md`](research/kernels/kernel_analysis_minimax.md) — Detailed kernel dispatch path analysis
- [`research/community_issues/issues_and_discourse_minimax.md`](research/community_issues/issues_and_discourse_minimax.md) — Curated community issue reports and discourse

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

## License

This research documentation is released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). Referenced repositories carry their own licenses.
