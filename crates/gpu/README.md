# turbospark-gpu

Metal device context (`MetalContext`), MSL pipeline cache, command pass encoders (`PassEncoder`, `CommittedPass`), per-kernel dispatches, KV cache manager (`KvCacheManager`), and resident weight Metal buffer wrappers (`ResidentGpuWeights`).

Downstream workspace crates import this package via the `gpu` alias:

```toml
[dependencies]
gpu = { package = "turbospark-gpu", path = "../gpu" }
```

## Platform Requirements

- **macOS only**: Source files are platform-gated with `#[cfg(target_os = "macos")]`. On non-macOS platforms, this crate compiles to an empty module.
- Requires a Metal-capable Apple Silicon or AMD device plus Xcode Metal toolchain (`xcrun -sdk macosx metal`) for testing.

## Key Modules

- `context.rs`: `MetalContext` managing `MTLDevice`, `MTLCommandQueue`, and address-keyed pipeline caching.
- `kv_cache.rs`: `KvCacheManager` managing persistent per-layer Metal KV buffers.
- `attention_decode.rs`: Split-KV decode attention dispatch (`chunks_for`, up to 16 splits).
- `moe_decode.rs`: Dispatches MoE router GEMV, phase-1 GEMVs, host router wait, and phase-2 down reduction.
- `rms_norm.rs`: RMSNorm dispatches (`rmsnorm_no_scale`, `rms_norm_bf16w`, per-head norm variants).
- `rope.rs`: Rotary positional embedding dispatches (`rope_proportional_neox`, `rope_neox_subdim`).
- `gdn.rs`: Eight gated-DeltaNet dispatches plus `GdnShape` structural preconditions.
- `dequant_int4_gemv.rs` & `dequant_int8_gemv.rs`: INT4/INT8 GEMV SIMD dispatches.
- `resident_metal.rs`: `ResidentGpuWeights` zero-copy `MTLBuffer` wrapping around mapped slices.
- `shaders/`: Metal Shading Language (MSL) source files containing **98 compute kernels** across 12 quantization formats, attention primitives, GDN linear attention, and vision processing. See [`docs/KERNELS.md`](../../docs/KERNELS.md) for the complete reference.

## Development & Test Commands

```sh
# Run tests for turbospark-gpu (macOS only)
cargo test -p turbospark-gpu
```

## Crate Gotchas

1. **Address-Keyed Pipeline Cache**: `MetalContext::pipeline` keys its function and pipeline caches on shader string memory address (`&'static str`), NOT string contents. Callers MUST pass identical `include_str!` static constants.
2. **Autorelease Pool Wrapping**: Metal command buffer and encoder allocations produce autoreleased objects. Fast loop iterations must be wrapped in `gpu::autorelease_pool`.
3. **MoE Phase 2 Reduction**: `moe_phase2_down_reduce_k8` reduces all 8 slots unconditionally. Unused slots require a 0.0 routing weight, a valid blob pointer, and a finite activation row.
