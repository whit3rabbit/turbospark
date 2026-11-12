---
description: "Portable CPU reference kernels and quantized tensor helpers: normalization, RoPE, attention, MoE, steering, and every quantization format the engine reads."
icon: microchip
---

# compute

`crates/compute` (package `turbospark-compute`) holds the portable,
`#![forbid(unsafe_code)]` numerical primitives of the engine: FP32 reference
kernels for RMSNorm, RoPE, attention, MoE feed-forward, and directional
steering, plus quantize/dequantize and GEMV helpers for every weight format
the engine reads. Its defining role is to be the numerical ground truth that
Metal kernels are validated against, and to run identically on every target:
cross-compilation for a non-macOS host is part of the crate's contract.

## What it does

| Area | Modules | Contents |
| --- | --- | --- |
| Normalization and position | `rms_norm`, `rope`, `wht` | `rms_norm` (plain, centered, grouped), RoPE paired/NeoX/mRoPE variants, YaRN frequencies, Walsh-Hadamard transform |
| Attention | `attention`, `mla`, `kv_quant`, `kv_quant_attention` | Causal multi-head and multi-query attention (with optional sinks), MLA absorbed decode, TurboQuant KV codec and attention over quantized rows |
| Weight quantization | `quant`, `quant_1bit`, `quant_2bit`, `quant_gguf`, `quant_gguf_iq`, `quant_gguf_mxfp4` | Affine INT4/INT8, 1-bit and 2-bit formats, GGUF Q2_0 through Q8_0 and the IQ and MXFP4 families, each with dequantize, GEMV, and embedding-lookup entry points |
| MoE and gating | `moe`, `gating` | Routed-expert FFN (`run_ffn`, `apply_streamed_routed`), activation and gate-multiply helpers |
| Steering | `steering` | The CPU reference for the four-mode residual edit (`steer_in_place`) |
| Tolerance | `tolerance` | `rel_error`, `max_abs_diff`, `bounded_rel_error`, and the `Tolerance` bars used by parity tests |
| Reference towers | `encoder`, `vision`, `whisper`, `gdn`, `ple`, `qsa_indexer`, `hyper_connection` | FP32 references for BERT/XLM-R encoders, the vision tower, Whisper, GDN linear attention, and the qwen4_exp blocks |
| Sampling | `sampling` | Softmax with logit soft-capping |

## Where it sits

`turbospark-compute` depends only on `turbospark-core` (aliased
`foundation`). Its consumers are `repack` (intake-time dequantization and
repacking), `runtime` (CPU paths and reference math), `image` and `ffi`, and
`gpu`, which keeps the crate as the CPU reference its Metal dispatches are
parity-tested against (the TurboQuant tables it builds at model open are the
one real runtime use there).

## Public API tour

The crate root in `crates/compute/src/lib.rs` re-exports the whole surface;
these are the entry points you will reach for most often:

| Item | Path | Meaning |
| --- | --- | --- |
| `rms_norm(&[f32], &[f32], eps)` | `src/rms_norm.rs` | FP32 RMSNorm over one row |
| `rope_paired`, `rope_neox` | `src/rope.rs` | Paired and half-rotation RoPE over one row |
| `causal_attention`, `causal_attention_with_sinks`, `indexed_attention` | `src/attention.rs` | Causal attention reference kernels |
| `quantize_int4_affine` / `dequantize_int4_affine` | `src/quant.rs` | Affine 4-bit row codec (`Int4AffineRow`, `GROUP_SIZE = 64`, BF16 scales and biases) |
| `dequant_int4_gemv`, `embed_lookup_int4` | `src/quant.rs` | Quantized GEMV and embedding-table lookup |
| `dequantize_q4_0` ... `dequantize_q8_0`, `dequant_*_gemv` | `src/quant_gguf.rs` | GGUF K-quant and legacy formats, with `*_BLOCK_BYTES` / `*_BLOCK_ELEMS` layout constants |
| `dequantize_iq4_xs`, `dequant_iq3_xxs_gemv`, ... | `src/quant_gguf_iq.rs` | GGUF IQ formats, backed by generated codebooks in `quant_gguf_iq_lowbit_tables.rs` |
| `quantize_row` / `dequantize_row`, `QuantizedRow` | `src/kv_quant.rs` | TurboQuant KV row codec (randomized Hadamard rotation plus Lloyd-Max codebooks) |
| `run_ffn`, `apply_streamed_routed`, `gelu_tanh` | `src/moe.rs` | MoE feed-forward and streamed routed-expert application |
| `steer_in_place`, `inv_norm`, `direction_coefficient` | `src/steering.rs` | The four-mode steering edit shared with the Metal kernel |
| `rel_error`, `bounded_rel_error`, `max_abs_diff`, `Tolerance` | `src/tolerance.rs` | Parity-test error measurement and the per-kernel tolerance bars |
| `bf16_to_f32`, `f32_to_bf16` | `src/quant.rs` | BF16 conversion through the existing bit-shift implementation |

There are no Cargo feature flags: the whole surface compiles on every target.

## Examples

### Quantize a row to INT4 affine and run a GEMV

Adapted from the codec in `crates/compute/src/quant.rs`. A row's length must
be a multiple of `GROUP_SIZE` (64); scales and biases round through BF16
before quantization so decode reproduces the stored code exactly.

```rust
use turbospark_compute::{
    dequant_int4_gemv, dequantize_int4_affine, quantize_int4_affine, Int4AffineRow,
};
use turbospark_compute::quant::GROUP_SIZE;

let row: Vec<f32> = (0..GROUP_SIZE).map(|i| i as f32 * 0.25 - 8.0).collect();
let packed: Int4AffineRow = quantize_int4_affine(&row);

let restored = dequantize_int4_affine(&packed, row.len());
assert!(restored.len() == row.len());

// One quantized matrix-vector product: one row per output element, with
// `n` naming the shared row length.
let weights: Vec<Int4AffineRow> = vec![packed];
let x = vec![1.0f32; GROUP_SIZE];
let y = dequant_int4_gemv(&weights, &x, x.len());
assert_eq!(y.len(), 1);
```

### Check a kernel against a hand-computed value

From `crates/compute/tests/kernels.rs`, which is the pattern every Metal
parity test follows: run the CPU reference, then score it with the
`tolerance` helpers.

```rust
use turbospark_compute::{bounded_rel_error, rms_norm, Tolerance};

let x = [3.0f32, 4.0];
let w = [1.0f32, 1.0];
let out = rms_norm(&x, &w, 0.0);
let inv_rms = 1.0 / ((25.0f32 / 2.0).sqrt());
assert!((out[0] - 3.0 * inv_rms).abs() < 1e-6);
assert!((out[1] - 4.0 * inv_rms).abs() < 1e-6);

// Against another implementation, compare with the matching tolerance bar.
let err = bounded_rel_error(&out, &out, 1e-6);
assert!(err <= Tolerance::IDENTITY);
```

`Tolerance` exposes the sized bars used across the workspace:
`IDENTITY` (1e-5), `FP16_REDUCTION` (5e-3), `FP16_CHAINED_REDUCTION` (1e-2),
`QUANT_INT4`, and `QUANT_INT8`.

### Dequantize one GGUF Q4_0 block

From `crates/compute/tests/quant_gguf_q4_0.rs`. The block layout constants
travel with the codec so callers never hard-code byte counts.

```rust
use turbospark_compute::{dequantize_q4_0, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};

let mut block = vec![0u8; Q4_0_BLOCK_BYTES];
block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
block[2] = 0x90;
block[3] = 0xf0;

let values = dequantize_q4_0(&block, Q4_0_BLOCK_ELEMS);
assert_eq!(values[0], -4.0);
assert_eq!(values[16], 0.5);
assert_eq!(values[17], 3.5);
```

### Apply a steering edit to a residual row

`steer_in_place` is the CPU spelling of the same four-mode edit the Metal
kernel performs; the mode enum comes from `turbospark-core` so both sides
share one definition.

```rust
use foundation::SteeringMode;
use turbospark_compute::{direction_coefficient, inv_norm, steer_in_place};

let d = [1.0f32, 0.0];
let mut x = [3.0f32, 4.0];
let inv = inv_norm(&d);
let c_hat = direction_coefficient(&x, &d) * inv;
x[0] -= c_hat * d[0]; // what Ablate computes, one component at a time

let mut y = vec![3.0f32, 4.0];
steer_in_place(&mut y, &d, SteeringMode::Ablate, 1.0, inv, 0.0, 0.0);
```

(The hand unrolled line is illustrative; the real parity contract is that
`steer_in_place` and the GPU kernel agree, checked by a dedicated test.)

## Gotchas

{% hint style="warning" %}
Preserve quantization block layout and scale semantics exactly: low nibble =
even index for INT4, BF16-rounded scales, GGUF block byte orders. The crate's
`AGENTS.md` is explicit that a self-consistency test (quantize then
dequantize) is not an independent numerical oracle, so a layout change needs
fixture assertions, and subtle packing rules should be mutation-checked.
{% endhint %}

Two more rules from the crate docs: FP16 always uses the `half` crate and
BF16 conversion follows the existing bit-shift implementation (do not
introduce a second representation), and the crate must stay portable, so
keep macOS-only APIs out of it.

## Related pages

- [core](core.md)
- [Extreme Quantization](../advanced/extreme-quantization.md)
- [The .gturbo Format](../models/gturbo-format.md)
- [Testing and Verification](../developer/testing-and-verification.md)
