# turbospark-gpu

Metal compute device context (`MetalContext`), MSL pipeline state caching, command pass encoders (`PassEncoder`, `CommittedPass`), per-kernel GPU dispatches, persistent KV cache management (`KvCacheManager`), and zero-copy resident weight buffer wrappers (`ResidentGpuWeights`).

Downstream workspace crates import this package via the `gpu` alias:

```toml
[dependencies]
gpu = { package = "turbospark-gpu", path = "../gpu" }
```

## Purpose & Role

`turbospark-gpu` encapsulates all Apple Silicon Metal GPU interaction for the inference engine. It compiles and caches Metal Shading Language (MSL) compute pipelines, binds device buffers, encodes compute command dispatches, and manages GPU-resident memory buffers (such as the persistent KV cache and zero-copy mmap weight allocations).

## Platform Requirements

- **macOS only**: All source code is platform-gated with `#[cfg(target_os = "macos")]`. On non-macOS platforms, this crate compiles to an empty module.
- Requires a Metal-capable Apple Silicon device (M1/M2/M3/M4) and the Xcode Metal toolchain (`xcrun -sdk macosx metal`) for compiling shaders and running tests.

## Key Modules

- `context/`: `MetalContext` managing `MTLDevice`, `MTLCommandQueue`, static pointer-keyed pipeline caching, and command pass wrappers (`PassEncoder`, `CommittedPass`).
- `kv_cache.rs`, `kv_cache_mem.rs`, `kv_cache_rewind.rs`: `KvCacheManager` managing persistent per-layer Metal KV buffers with sequence rollback/rewind support.
- `kv_quantize.rs` & `kv_quant_tables.rs`: TurboQuant KV cache quantization dispatches.
- `attention_decode.rs`: Split-KV causal decode attention dispatch (`chunks_for`, up to 16 splits).
- `attention_indexed.rs` & `attention_tq.rs`: Indexed block attention and TurboQuant quantized attention dispatches.
- `moe_decode.rs`: Mixture-of-experts dispatches: router GEMV, phase 1 expert GEMVs, host router wait, and phase 2 down reduction.
- `moe_gguf/`: GGUF format MoE expert GEMV and reduction dispatches.
- `moe_prefill_batch.rs` & `moe_prefill_batch_gguf.rs`: Batched prefill MoE dispatches.
- `rms_norm.rs`: RMSNorm dispatches (`rmsnorm_no_scale`, `rmsnorm_bf16w`, headwise norm).
- `rope.rs`: Rotary positional embedding dispatches (`rope_proportional_neox`, `rope_neox_subdim`, 2D spatial RoPE, YaRN).
- `gdn.rs`, `gdn_shape.rs`, `gdn_state.rs`: Gated-DeltaNet linear attention dispatches (conv, SiLU, delta recurrence, gated norm) and persistent state buffers.
- `dequant_int4_gemv.rs`, `dequant_int8_gemv.rs`, `dequant_int4_batch.rs`: INT4/INT8 affine GEMV and batch GEMM dispatches.
- `dequant_1bit_gemv.rs`, `dequant_1bit_gemm_batch.rs`, `dequant_2bit_gemv.rs`, `dequant_2bit_gemm_batch.rs`: 1-bit and 2-bit affine GEMV and speculative verification batch GEMM dispatches.
- `dequant_iq_gemv.rs` & `dequant_q2_k_gemv.rs`: GGUF importance-quantized (IQ) codebook and Q2_K dispatches.
- `dequant_q4_k_gemv.rs`, `dequant_q5_k_gemv.rs`, `dequant_q6_k_gemv.rs`, `dequant_q8_0_gemv.rs`: Standard GGUF k-quant dispatches.
- `dflash_conv.rs`: DFlash block drafter causal convolution dispatches.
- `qsa_indexer.rs` & `qsa_indexer_state.rs`: QSA sparse attention indexer dispatches.
- `ple.rs`: Per-layer n-gram embedding table dispatches.
- `minimax_router.rs`: MiniMax FP32 sigmoid router dispatches.
- `hyper_connection.rs`: Multi-stream residual connection dispatches.
- `resident_metal.rs`: `ResidentGpuWeights` zero-copy `MTLBuffer` creation over memory-mapped weight slices (`newBufferWithBytesNoCopy`).
- `shaders/`: Metal Shading Language source files containing compute kernels across 34 `.metal` files.

## Development & Test Commands

```sh
# Run all GPU unit and parity tests (macOS only)
cargo test -p turbospark-gpu

# Run a single parity test suite with debug output
cargo test -p turbospark-gpu --test rms_norm_parity -- --nocapture
```

## Tests

This crate contains 58 integration test suites in `tests/` comparing GPU kernel outputs against CPU reference calculations in `crates/compute`:
- Quantization parity: `dequant_1bit_gemv_parity.rs`, `dequant_1bit_gemm_parity.rs`, `dequant_2bit_gemv_parity.rs`, `dequant_2bit_gemm_parity.rs`, `dequant_int4_gemv_parity.rs`, `dequant_int4_gemm_parity.rs`, `dequant_int8_gemv_parity.rs`, `dequant_iq_gemv_parity.rs`, `dequant_q4_k_gemv_parity.rs`, `dequant_q5_k_gemv_parity.rs`, `dequant_q6_k_gemv_parity.rs`, `dequant_q8_0_gemv_parity.rs`.
- Attention and KV cache: `attention_decode_parity.rs`, `attention_indexed_parity.rs`, `attention_tq_parity.rs`, `attention_sinks.rs`, `attention_swa.rs`, `kv_cache.rs`, `kv_cache_quant.rs`, `kv_quantize_parity.rs`.
- Normalization and activations: `rms_norm_parity.rs`, `rms_norm_grouped_parity.rs`, `scaled_norm_and_embed.rs`, `unary_activation_parity.rs`, `logit_softmax_parity.rs`.
- Linear attention and MoE: `gdn_parity.rs`, `gdn_conv_dilated_parity.rs`, `gdn_gated_norm_sigmoid_parity.rs`, `moe_decode.rs`, `moe_gguf_parity.rs`, `moe_prefill_batch_parity.rs`, `moe_prefill_batch_gguf_parity.rs`.
- Vision and architecture components: `vision_parity.rs`, `vision_block_parity.rs`, `rope_parity.rs`, `rope_mrope_parity.rs`, `rope_yarn_parity.rs`, `hyper_connection_parity.rs`, `qsa_indexer_parity.rs`, `ple_gate_parity.rs`, `minimax_router.rs`.

## Crate Gotchas

1. **Address-Keyed Pipeline Cache**: `MetalContext::pipeline` keys its function and pipeline caches on static shader string memory address (`&'static str`), NOT string contents. Callers must pass identical `include_str!` static constants.
2. **Autorelease Pool Wrapping**: Metal command buffer and encoder allocations create autoreleased objects (~6 KiB per buffer). Tight decode loops must wrap dispatches inside `gpu::autorelease_pool` to avoid memory accumulation.
3. **MoE Phase 2 Reduction Top-K**: `moe_phase2_down_reduce` dynamically dispatches up to runtime `top_k` slots (supporting architectures like `qwen4_exp` with `top_k=10`), while standard GGUF MoE kernels require exactly matching slot buffers.
