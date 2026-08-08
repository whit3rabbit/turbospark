# mrefrust-gpu

Metal device context (`MetalContext`), pipeline cache, command pass encoders (`PassEncoder`, `CommittedPass`), per-kernel dispatches, KV cache manager (`KvCacheManager`), and resident weight Metal buffer wrappers (`ResidentGpuWeights`).

## Platform Requirements

- **macOS only**: All source files are gated with `#[cfg(target_os = "macos")]`. On non-macOS platforms, this crate compiles to an empty module.
- Requires a Metal-capable device and Xcode's `metal` toolchain (`xcrun -sdk macosx metal`) to compile and execute tests.

## Directory & File Structure

```
crates/gpu/
+-- Cargo.toml                      # Crate manifest
+-- src/
|   +-- lib.rs                      # Library root
|   +-- context.rs                  # MetalContext, PassEncoder, CommittedPass
|   +-- kv_cache.rs                 # KvCacheManager managing per-layer Metal KV buffers
|   +-- attention_decode.rs         # Split-KV decode attention dispatch
|   +-- moe_decode.rs               # MoE router, phase 1 GEMV, phase 2 down-reduce dispatches
|   +-- rms_norm.rs                 # RMSNorm dispatches (no-scale, BF16, per-head)
|   +-- rope.rs                     # RoPE positional embedding dispatch
|   +-- dequant_int4_gemv.rs        # INT4 SIMD GEMV dispatches (resident & streamed)
|   +-- dequant_int8_gemv.rs        # INT8 SIMD GEMV dispatches (resident & streamed)
|   +-- dequant_q8_0_gemv.rs        # GGUF Q8_0 SIMD GEMV dispatch (port-local, Phase G)
|   +-- resident_metal.rs           # ResidentGpuWeights mmap zero-copy MTLBuffer wrapper
|   +-- dispatch_profile.rs         # Intra-command-buffer dispatch profiler
|   +-- utility.rs                  # Elementwise helper dispatches (scalar mul, softcap)
|   +-- logit_softmax.rs            # Softcap and logit softmax helpers
|   +-- bytes.rs                    # Metal buffer byte alignment utilities
|   +-- gdn.rs                      # Gated-DeltaNet kernel dispatches (8 kernels)
|   +-- gdn_state.rs                # GDN recurrent state buffers (Qwen flow's)
|   +-- dsv4_state.rs               # DSV4 Metal buffer allocation (unwired)
|   +-- prefill_scratch.rs          # Chunked prefill scratch buffer layout (undispatched)
|   \-- shaders/                    # MSL source, vendored from Swift except where marked port-local
|       +-- attention.metal         # Decode attention Metal shader source
|       +-- dequant_int4.metal      # INT4 dequantization GEMV shader source
|       +-- dequant_int8.metal      # INT8 dequantization GEMV shader source
|       +-- dequant_q8_0.metal      # GGUF Q8_0 dequantization GEMV shader source
|       +-- gdn.metal               # Gated-DeltaNet (Qwen 3.6 linear attention) shader source
|       +-- logit.metal             # Logit softcap and softmax shader source
|       +-- moe.metal               # MoE router GEMV and phase 1/2 shader source
|       +-- rmsnorm.metal           # RMSNorm shader source
|       +-- rope.metal              # RoPE shader source
|       \-- utility.metal           # Elementwise utility shader source
\-- tests/                          # Metal numerical parity & allocation unit tests
    +-- attention_chunk_bench.rs
    +-- attention_decode_parity.rs
    +-- attention_swa.rs
    +-- dequant_int4_gemv_parity.rs
    +-- dequant_int8_gemv_parity.rs
    +-- dequant_q8_0_gemv_parity.rs
    +-- dispatch_profile.rs
    +-- dsv4_state.rs
    +-- gdn_parity.rs
    +-- gdn_state.rs
    +-- gemma4_real_kernels_parity.rs
    +-- kv_cache.rs
    +-- logit_softmax_parity.rs
    +-- moe_decode.rs
    +-- prefill_scratch.rs
    +-- resident_metal.rs
    +-- rms_norm_parity.rs
    +-- rope_parity.rs
    +-- scaled_norm_and_embed.rs
    \-- utility_and_pass.rs
```

## Key Modules

- `context.rs`: `MetalContext` managing `MTLDevice`, `MTLCommandQueue`, and pipeline state caching.
- `kv_cache.rs`: `KvCacheManager` managing persistent per-layer Metal KV buffers (linear or sliding-window ring).
- `attention_decode.rs`: Split-KV decode attention dispatch (`chunks_for`, up to 16 splits).
- `moe_decode.rs`: Dispatches MoE router GEMV, phase-1 GEMVs, host router wait, and phase-2 down reduction.
- `rms_norm.rs`: RMSNorm dispatches (`rmsnorm_no_scale`, `rms_norm_bf16w`, per-head norm variants).
- `rope.rs`: Rotary positional embedding dispatches (`rope_proportional_neox`, `rope_neox_subdim`).
- `gdn.rs`: The eight gated-DeltaNet dispatches plus `GdnShape` and its structural preconditions.
- `dequant_int4_gemv.rs` & `dequant_int8_gemv.rs`: INT4/INT8 GEMV SIMD dispatches.
- `dequant_q8_0_gemv.rs`: the GGUF Q8_0 GEMV dispatch (ROADMAP Phase G Stage 2). PORT-LOCAL, not vendored -- the Swift engine has no GGUF intake, so its only contract is `mrefrust_compute::dequant_q8_0_gemv`. Shorter than the INT8 sibling because a Q8_0 row is ONE byte run: the scale lives inside each 34-byte block, so there are no scale or bias planes to bind and no group size to agree on. 32 lanes over a 32-element block, one weight per lane.
- `resident_metal.rs`: `ResidentGpuWeights` zero-copy `MTLBuffer` wrapping around `mmap` slices.

## Development & Test Commands

```sh
# Run tests for mrefrust-gpu (macOS only)
cargo test -p mrefrust-gpu
```

## Crate Gotchas

1. **Pipeline Cache Keying on Address**: `MetalContext::pipeline` keys its function and pipeline caches on shader string memory ADDRESS (`&'static str`), NOT string contents. Callers MUST pass identical `include_str!` static constants.
2. **Autorelease Pool Wrapping**: Metal command buffer and compute encoder creations return autoreleased objects. Repeated encode loops MUST be wrapped in `gpu::autorelease_pool`.
3. **MoE Phase 2 Down Reduction**: `moe_phase2_down_reduce_k8` reduces all 8 slots unconditionally. Unused slots must have a 0.0 routing weight, a valid blob pointer, and a finite activation row.
4. **`gdn.rs`'s shader source is a CONCATENATION** of `dequant_int4.metal` and `gdn.metal`, in that order: the fused input projection calls `dequant_int4_gemv_simd_body`, a `static inline` in the former, which resolves because the Swift build concatenates every module into one library. `concat!` of two `include_str!`s is one `&'static str` with one stable address, which is what Gotcha 1's address-keyed cache needs. Never pass `dequant_int4.metal`'s own constant to a GDN dispatch (it would miss the cache, not misbehave), and expect the INT4 kernels to be compiled twice in this process.
5. **`gdn_qk_norm` / `gdn_gated_norm` need EXACTLY 128 threads per threadgroup** -- both reduce four SIMD partials with a hardcoded loop. Fewer sums uninitialized slots, more drops work. `NORM_THREADS` pins it. The delta kernels are likewise fixed at `(32, 4)` threads, which is where `GdnShape::validate`'s `Dk % 32 == 0` and `Dk / 32 <= 8` come from.
6. **KV Cache Ring Specialization**: `KvCacheManager`'s `fp16_ring_enabled` mode specializes `FC_ATTN_RING_CAP` into Metal pipelines. Ring capacity values MUST be included in the pipeline cache constants key.
