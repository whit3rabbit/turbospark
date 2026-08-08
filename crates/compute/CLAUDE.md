# turbospark-compute

CPU reference kernels and compute strategy marker type (`ComputeStrategy`). These reference implementations serve as numerical ground truth against which `crates/gpu` Metal kernels are validated.

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/compute/
+-- Cargo.toml          # Crate manifest
+-- src/
|   +-- lib.rs          # Library root and ComputeStrategy marker declaration
|   +-- attention.rs    # CPU causal attention reference implementation
|   +-- gdn.rs          # Gated-DeltaNet (Qwen 3.6 linear attention) reference chain
|   +-- gating.rs       # Qwen 3.6 gating references (sigmoid gate/scalar, q/gate split)
|   +-- moe.rs          # CPU MoE FFN reference and gated activation bridge
|   +-- quant.rs        # INT4 and INT8 affine quantization and GEMV reference
|   +-- quant_gguf.rs   # GGUF block quant reference (Q8_0, Q4_K) + Pearson correlation
|   +-- rms_norm.rs     # CPU RMSNorm reference calculation
|   +-- rope.rs         # CPU rotary positional embedding calculation
|   +-- sampling.rs     # Host-side sampling helper logic
|   +-- tolerance.rs    # RelError tolerance table & numerical comparison utilities
|   \-- wht.rs          # Walsh-Hadamard Transform reference implementation
\-- tests/
    +-- kernels.rs      # Kernel numerical validation unit tests
    +-- quant_gguf.rs   # Q8_0 and Q4_K round trip, sign, zero block, GEMV, correlation
    \-- smoke.rs        # Basic compute smoke test
```

## Key Modules

- `attention.rs`: Reference CPU causal attention.
- `gdn.rs`: `GdnReference`, the straight-line model of Qwen 3.6's gated-DeltaNet chain (conv + SiLU, per-head q/k norm with folded delta scales, the FP32 delta recurrence, the gated output norm) that `crates/gpu/tests/gdn_parity.rs` checks the eight `gdn.metal` kernels against.
- `gating.rs`: `sigmoid_gate_mul`, `sigmoid_scalar_mul`, `split_q_gate`.
- `moe.rs`: CPU reference MoE FFN logic (`run_ffn`) used to bridge gated FFN activations.
- `quant.rs`: INT4 and INT8 affine (MLX-layout) quantization/dequantization and GEMV math used by repackers and CPU fallbacks.
- `quant_gguf.rs`: the GGUF block-quant reference (Q8_0 and Q4_K). A separate module from `quant.rs` because the layout is not a variant of the affine one: interleaved rather than planar, and Q8_0 is signed and symmetric with no bias. Q4_K is a second shape again, not a wider Q8_0: a 256-element superblock with two f16 super-scales and eight 32-element sub-blocks whose 6-bit scales and 6-bit mins are packed into 12 bytes, split across two bytes for sub-blocks 4..8, reconstructing as `w = d*sc*q - dmin*m` with UNSIGNED quants. Q6_K is a third shape again: 256 elements with one f16 super-scale and SIXTEEN signed int8 sub-block scales stored as plain bytes, and an element's six bits split across two runs (four low bits in `ql`, two high bits in `qh`) with a fixed bias of 32 subtracted rather than a per-sub-block min. Also carries `pearson`, which is what settles `FUSED_GATE_FIRST` over in `crates/repack`.
- `rms_norm.rs`: Reference RMSNorm implementation.
- `rope.rs`: Reference RoPE implementations.
- `tolerance.rs`: Relative error metric (`RelError`) used in parity testing across CPU and GPU pipelines.

## Development & Test Commands

```sh
# Run tests for turbospark-compute
cargo test -p turbospark-compute
```

## Crate Gotchas

1. **BF16 vs FP16**: `crates/compute` contains hand-rolled bit-shift helpers for BF16 (`bf16_to_f32`/`f32_to_bf16`), exploiting the fact that BF16 is the upper 16 bits of an FP32 word. FP16 (binary16) storage elsewhere in the workspace uses the `half` crate. Do not mix up the two or hand-roll FP16 logic.
2. **`gdn.rs`'s FP16 rounding points are part of the contract.** The GPU kernels store `conv_out`, the normed q/k slices, the raw conv tail rows, and `y` as `half`; the reference rounds at exactly those four places and nowhere else (the recurrence and the gated norm stay FP32). Moving one makes the parity test's tolerance meaningless. It rounds through `foundation::LogitValue` (`half::f16` by its public path) rather than adding a `half` dependency to this crate -- Gotcha 3 in AGENTS.md forbids hand-rolling binary16.
3. **Ground Truth Baseline**: Kernels in this crate are designed for exactness and reference correctness rather than maximum CPU throughput. Metal kernels in `crates/gpu` validate parity against these routines.
