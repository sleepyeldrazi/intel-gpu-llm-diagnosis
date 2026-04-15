# Phase 3 — Vulkan Intel Fixes

**Depends on:** Phase 1 and 2 (should be applied and tested first)

## 0001-arc-140t-xe2-override.patch

Adds device-ID override for Intel Arc 140T (Arrow Lake H) to force INTEL_XE2
classification in the Vulkan backend.

### Problem
Arc 140T reports minSubgroupSize=8 instead of 16. The Vulkan backend uses
minSubgroupSize to detect Xe2. When misreported, the 140T is classified as
OTHER, disabling cooperative matrix and all dependent optimizations.

### Fix
Checks for Arrow Lake H device IDs (0x7D51, 0x7D45) before the minSubgroupSize
check and returns INTEL_XE2 directly.

### Applies to
Both the EXT and KHR code paths in ggml-vulkan.cpp.

### Impact
Only affects Arrow Lake H (Arc 140T) systems. No effect on other hardware.

### Testing checklist
- [ ] Build succeeds with Vulkan support
- [ ] Arc 140T: device classified as INTEL_XE2
- [ ] Arc 140T: cooperative matrix shaders used for matmul
- [ ] Other Intel GPUs: no change in behavior
- [ ] Non-Intel GPUs: no change in behavior
