# Phase 2 — SYCL Kernel Tuning

**Depends on:** Phase 1 (should be applied and tested first)

## 0001-fix-ver-gen-thresholds.patch

Fixes VER_GEN12 (1,000,000 → 1,200) and VER_GEN13 (1,001,030 → 1,300).

The original VER_GEN12 value was an unreachable placeholder that caused all Intel
Arc GPUs (cc≈1255 for A770) to fall through to the NVIDIA Ampere tuning path in
all MMQ kernels. After this patch, Intel discrete GPUs use the VER_GEN12 path.

## 0002-tune-dmmv-xy-for-arc.patch

Changes presets.hpp: DMMV_X 32→64, MMV_Y 1→2.

Doubles the data processed per thread in DMMV kernels and doubles rows per
work-group. All common model widths (4096-14336) are divisible by 64.

## 0003-tune-dmmv-xy-common-hpp.patch

Same changes as 0002 but in common.hpp (duplicate definitions).

### Expected impact
5-15% additional improvement on top of Phase 1.

### ⚠️ Needs Benchmarking
DMMV_X=64 and MMV_Y=2 were chosen analytically, not empirically. If MMV_Y=2
causes register spills (check with `GGML_SYCL_DEBUG=1`), revert 0002+0003 and
try DMMV_X=64 with MMV_Y=1 only.

### Testing checklist
- [ ] Build succeeds
- [ ] Unit tests pass
- [ ] Dense model inference produces correct output
- [ ] No assertion failures (`ncols % GGML_SYCL_DMMV_X == 0`)
- [ ] Benchmark comparison vs Phase 1 only
