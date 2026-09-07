# turbospark-compute

CPU reference kernels and compute strategy marker type (`ComputeStrategy`). These reference implementations serve as the numerical ground truth against which `crates/gpu` Metal kernels are validated.

Downstream workspace crates import this package via the `compute` alias:

```toml
[dependencies]
compute = { package = "turbospark-compute", path = "../compute" }
```

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Key Modules

- `attention.rs`: CPU reference causal attention algorithm, plus sink-attention support (`causal_attention_with_sinks`).
- `encoder.rs`: FP32 reference for BERT/XLM-RoBERTa-style encoder models (BGE, Snowflake Arctic Embed): embeddings lookup, a Post-LN encoder block, CLS pooling, and cosine similarity.
- `gdn.rs`: `GdnReference`, the reference model of Qwen 3.6 gated-DeltaNet linear attention chain (conv + SiLU, per-head q/k norm, FP32 delta recurrence, gated output norm).
- `hyper_connection.rs`: `qwen4_exp`'s hyper-connection mix and scatter (port-local).
- `gating.rs`: Qwen 3.6 gating references (`sigmoid_gate_mul`, `sigmoid_scalar_mul`, `split_q_gate`).
- `kv_quant.rs`: TurboQuant KV-cache codec reference (per-row norm, Randomized Hadamard Transform, Lloyd-Max codebook quantization), ported from mlx-vlm.
- `kv_quant_attention.rs`: CPU reference for causal attention over TurboQuant-quantized K/V rows.
- `moe.rs`: CPU reference MoE FFN implementation (`run_ffn`) used to bridge gated FFN activations.
- `ple.rs`: `qwen4_exp`'s PLE (per-layer n-gram embedding) gate, n-gram table dequantization, and the dilated depthwise causal conv step (port-local).
- `qsa_indexer.rs`: `qwen4_exp`'s QSA block indexer: block pooling, scoring, and top-k block selection (port-local).
- `quant.rs`: INT4 and INT8 affine quantization, dequantization, and GEMV math.
- `quant_1bit.rs` / `quant_2bit.rs`: the affine 1-bit and 2-bit (ternary) reference formats, measured against real MLX-quantized checkpoints.
- `quant_gguf/`: GGUF block-quant reference (Q8_0, Q4_K, Q5_K, Q6_K) plus Pearson correlation.
- `quant_gguf_iq.rs`: GGUF IQ-codebook reference (IQ4_NL, IQ4_XS, IQ3_XXS).
- `quant_gguf_iq_tables.rs`: generated IQ codebooks, dumped from libggml.
- `quant_gguf_mxfp4.rs`: GGUF MXFP4 reference (`gpt-oss`).
- `rms_norm.rs`: CPU RMSNorm reference calculation.
- `rope.rs`: CPU rotary positional embedding calculation, plus YaRN frequency-table construction.
- `sampling.rs`: host-side sampling helper logic (logit soft-cap softmax).
- `steering.rs`: directional steering of a residual stream row (ablate/add/clamp/renorm), port-local.
- `tolerance.rs`: Relative error metric (`RelError`) tolerance table and comparison utilities.
- `vision.rs`: FP32 reference kernels for the `qwen3_5` vision tower (LayerNorm, both GELUs, 2-D RoPE, bidirectional attention), port-local.
- `wht.rs`: Walsh-Hadamard Transform reference implementation.

## Development & Test Commands

```sh
# Run tests for turbospark-compute
cargo test -p turbospark-compute
```

## Crate Gotchas

1. **BF16 vs FP16**: `crates/compute` uses hand-rolled bit-shift helpers for BF16 (`bf16_to_f32`/`f32_to_bf16`) by leveraging BF16 as the upper 16 bits of FP32. FP16 (binary16) storage elsewhere uses the `half` crate.
2. **GDN FP16 Rounding Points**: `gdn.rs` rounds to FP16 at four explicit points (`conv_out`, normed q/k slices, raw conv tail rows, and output `y`) to match hardware behavior while keeping recurrence FP32.
3. **Numerical Ground Truth**: Algorithms in this crate prioritize mathematical reference exactness over maximum CPU vectorization.
