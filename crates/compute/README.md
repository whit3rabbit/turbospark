# turbospark-compute

CPU reference mathematical kernels and compute strategy marker type (`ComputeStrategy`). These reference implementations serve as the numerical ground truth against which `crates/gpu` Metal kernels and real hardware dispatches are validated for parity.

Downstream workspace crates import this package via the `compute` alias:

```toml
[dependencies]
compute = { package = "turbospark-compute", path = "../compute" }
```

## Purpose & Role

`turbospark-compute` implements portable, unvectorized or reference-vectorized CPU mathematical kernels for all supported model layers, quantization formats, and attention mechanisms. Parity test suites across `crates/gpu` compare GPU buffer outputs directly against the functions defined in this crate.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- No device-specific or platform-locked assembly.

## Key Modules

- `attention.rs`: CPU reference causal attention, sliding window attention, and sink token attention (`causal_attention_with_sinks`).
- `encoder.rs`: FP32 reference for BERT and XLM-RoBERTa encoder architectures (BGE, Snowflake Arctic Embed): embedding lookup, Post-LN encoder block, CLS pooling, and cosine similarity.
- `gdn.rs`: `GdnReference`, reference implementation of Qwen 3.6 gated-DeltaNet linear attention chain (causal conv + SiLU, per-head q/k RMSNorm, FP32 delta recurrence, gated output norm).
- `gating.rs`: Activation gating operations (`sigmoid_gate_mul`, `sigmoid_scalar_mul`, `split_q_gate`).
- `hyper_connection.rs`: Multi-residual connection mixing and scatter operations used in deep model architectures.
- `kv_quant.rs`: TurboQuant KV-cache codec reference (per-row norm, Randomized Hadamard Transform, Lloyd-Max codebook quantization).
- `kv_quant_attention.rs`: CPU reference causal attention over TurboQuant-compressed K/V cache rows.
- `moe.rs`: Reference mixture-of-experts feed-forward network execution (`run_ffn`).
- `ple.rs`: Per-layer n-gram embedding (PLE) table lookup, dequantization, and dilated depthwise causal convolution.
- `qsa_indexer.rs`: QSA block indexer: block pooling, scoring, and top-k block selection.
- `quant.rs`: INT4 and INT8 affine block quantization, dequantization, and GEMV arithmetic.
- `quant_1bit.rs` & `quant_2bit.rs`: 1-bit and 2-bit (ternary) affine quantization formats matching MLX reference conventions.
- `quant_gguf/`: GGUF quantization block decoders (Q8_0, Q4_K, Q5_K, Q6_K) and Pearson correlation metrics.
- `quant_gguf_iq.rs`: GGUF importance-quantized codebook reference (IQ4_NL, IQ4_XS, IQ3_XXS).
- `quant_gguf_iq_tables.rs` & `quant_gguf_iq_lowbit_tables.rs`: Precomputed IQ codebook lookup tables.
- `quant_gguf_mxfp4.rs`: GGUF MXFP4 microscopic floating-point quantization reference.
- `rms_norm.rs`: Reference Root Mean Square Normalization with optional learned weight scaling.
- `rope.rs`: Rotary positional embeddings (RoPE), proportional NeoX RoPE, and YaRN frequency table generation.
- `sampling.rs`: Host-side logit shaping and logit soft-capping utilities.
- `steering.rs`: Directional steering operations on residual streams (`Ablate`, `Add`, `Clamp`, `Renorm`).
- `tolerance.rs`: NaN-sticky relative error metrics (`RelError`, `bounded_rel_error`, `worst()`) and tolerance comparisons.
- `vision.rs`: Vision tower reference kernels (LayerNorm, QuickGELU, NewGELU, 2D RoPE, bidirectional attention).
- `wht.rs`: Fast Walsh-Hadamard Transform reference.

## Development & Test Commands

```sh
# Run all tests for turbospark-compute
cargo test -p turbospark-compute
```

## Tests

- `tests/kernels.rs`: Validates attention, RMSNorm, RoPE, and standard activation math.
- `tests/kv_quant.rs`: Verifies TurboQuant compression error bounds and Hadamard transforms.
- `tests/quant_1bit.rs` & `tests/quant_2bit.rs`: Validates 1-bit and 2-bit dequantization and GEMV math.
- `tests/quant_gguf.rs`, `tests/quant_gguf_iq.rs`, `tests/quant_gguf_mxfp4.rs`: Validates GGUF and IQ codebook unpacking against reference fixtures.
- `tests/encoder_reference.rs`: Validates Post-LN transformer encoder math and CLS pooling.
- `tests/vision_reference.rs`: Validates vision patch embedding and 2D spatial RoPE.
- `tests/smoke.rs`: Quick sanity checks across foundational math routines.

## Crate Gotchas

1. **BF16 Conversion NaN Quieting**: `f32_to_bf16` explicitly quiets signaling NaNs rather than allowing round-to-nearest additions to overflow the mantissa into the exponent.
2. **GDN FP16 Rounding Points**: `gdn.rs` rounds to FP16 at four precise points (`conv_out`, normed q/k slices, conv tail rows, output `y`) to reproduce hardware numerics while maintaining FP32 state recurrence.
3. **NaN-Sticky Tolerance Checks**: `tolerance.rs` uses a custom `worst()` fold instead of `f32::max` so that NaN values cannot be masked by `f32::max(NaN, x) == x`.
4. **Precision Over Speed**: Algorithms in this crate prioritize reference correctness over vectorization.
