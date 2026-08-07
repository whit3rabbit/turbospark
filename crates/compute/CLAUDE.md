# mrefrust-compute

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
|   +-- moe.rs          # CPU MoE FFN reference and gated activation bridge
|   +-- quant.rs        # INT4 and INT8 affine quantization and GEMV reference
|   +-- rms_norm.rs     # CPU RMSNorm reference calculation
|   +-- rope.rs         # CPU rotary positional embedding calculation
|   +-- sampling.rs     # Host-side sampling helper logic
|   +-- tolerance.rs    # RelError tolerance table & numerical comparison utilities
|   \-- wht.rs          # Walsh-Hadamard Transform reference implementation
\-- tests/
    +-- kernels.rs      # Kernel numerical validation unit tests
    \-- smoke.rs        # Basic compute smoke test
```

## Key Modules

- `attention.rs`: Reference CPU causal attention.
- `moe.rs`: CPU reference MoE FFN logic (`run_ffn`) used to bridge gated FFN activations.
- `quant.rs`: INT4 and INT8 quantization/dequantization and GEMV math used by repackers and CPU fallbacks.
- `rms_norm.rs`: Reference RMSNorm implementation.
- `rope.rs`: Reference RoPE implementations.
- `tolerance.rs`: Relative error metric (`RelError`) used in parity testing across CPU and GPU pipelines.

## Development & Test Commands

```sh
# Run tests for mrefrust-compute
cargo test -p mrefrust-compute
```

## Crate Gotchas

1. **BF16 vs FP16**: `crates/compute` contains hand-rolled bit-shift helpers for BF16 (`bf16_to_f32`/`f32_to_bf16`), exploiting the fact that BF16 is the upper 16 bits of an FP32 word. FP16 (binary16) storage elsewhere in the workspace uses the `half` crate. Do not mix up the two or hand-roll FP16 logic.
2. **Ground Truth Baseline**: Kernels in this crate are designed for exactness and reference correctness rather than maximum CPU throughput. Metal kernels in `crates/gpu` validate parity against these routines.
