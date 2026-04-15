# Phase 4 — Host-Buffer Double-Copy Fix

**Depends on:** Phase 1 and 2 (should be applied and tested first)

## 0001-remove-blanket-host-buffer-copy.patch

Removes the blanket Linux host-buffer double-copy workaround in `set_tensor`.

### Problem
`ggml_backend_sycl_buffer_set_tensor` on Linux does:
```
malloc(host_buf) → memcpy(host_buf, data) → memcpy(device, host_buf) → free(host_buf)
```

This was a workaround for a PVC (Ponte Vecchio) bug where `mmap()`-backed host
pointers caused issues with direct device copies. The `#ifndef _WIN32` guard
penalized ALL Linux Intel GPUs — including Arc A770, A750, Meteor Lake iGPUs —
with an unnecessary extra `malloc/memcpy/free` on every `set_tensor` call.

### Fix
- Replaces the `#ifndef _WIN32` compile-time guard with a runtime check
- New env var `GGML_SYCL_MMAP_WORKAROUND` defaults to 0 (disabled)
- PVC users who need the workaround: `GGML_SYCL_MMAP_WORKAROUND=1`
- The `else` branch now does the direct device copy for all platforms

### Impact
- Eliminates one `malloc + memcpy + free` per tensor during model loading
- On Arc A770 with a 17GB model (~1M tensors): saves ~17GB of host-side copying
- No effect on Windows (already used the direct path)

### Testing checklist
- [ ] Build succeeds
- [ ] Model loads correctly
- [ ] Inference produces correct output
- [ ] `GGML_SYCL_MMAP_WORKAROUND=1` restores old behavior
