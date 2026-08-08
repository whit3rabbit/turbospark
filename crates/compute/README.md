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

- `attention.rs`: CPU reference causal attention algorithm.
- `gdn.rs`: `GdnReference`, the reference model of Qwen 3.6 gated-DeltaNet linear attention chain (conv + SiLU, per-head q/k norm, FP32 delta recurrence, gated output norm).
- `gating.rs`: Qwen 3.6 gating references (`sigmoid_gate_mul`, `sigmoid_scalar_mul`, `split_q_gate`).
- `moe.rs`: CPU reference MoE FFN implementation (`run_ffn`) used to bridge gated FFN activations.
- `quant.rs`: INT4 and INT8 affine quantization, dequantization, and GEMV math.
- `rms_norm.rs`: CPU RMSNorm reference calculation.
- `rope.rs`: CPU rotary positional embedding calculation.
- `tolerance.rs`: Relative error metric (`RelError`) tolerance table and comparison utilities.
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
