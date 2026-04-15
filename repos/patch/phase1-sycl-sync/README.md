# Phase 1 — SYCL Synchronization

## 0001-enable-sycl-graph-by-default.patch

Changes `GGML_SYCL_DISABLE_GRAPH` default from 1 (disabled) to 0 (enabled).

### What it does
- Enables SYCL graph execution for single-GPU dense LLM inference
- Enables async memory operations (tied to graph support in upstream code)
- Eliminates 8 blocking `.wait()` calls in reorder functions (Q4_0, Q8_0, Q4_K, Q6_K)

### What it does NOT affect
- MoE models (MUL_MAT_ID) — `check_graph_compatibility()` auto-disables graphs
- CONCAT operations — auto-disabled
- Multi-GPU setups — always disabled
- Users can override: `GGML_SYCL_DISABLE_GRAPH=1`

### Expected impact
10-30% token generation speedup on single-GPU dense LLM inference.

### Testing checklist
- [ ] Build succeeds with `-DGGML_SYCL=ON`
- [ ] `GGML_SYCL_DEBUG=1` shows "SYCL-GRAPH" messages for dense models
- [ ] Dense model inference produces correct output
- [ ] MoE model falls back gracefully (logs "disabling SYCL graphs")
- [ ] `GGML_SYCL_DISABLE_GRAPH=1` restores old behavior
