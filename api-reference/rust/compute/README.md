---
description: "CPU reference math kernels and the ComputeStrategy marker in turbospark-compute, used as numerical ground truth."
icon: box
---
# turbospark-compute

turbospark-compute holds CPU reference implementations of the math behind TurboSpark's model layers. Most functions take `f32` slices and return a `Vec<f32>`. Reach for it when you need a trusted answer to compare against, such as checking a GPU kernel for parity, or when you need the same math on a machine without Metal. Do not reach for it for speed. The crate has no model loader, tokenizer, sampler loop, or device dispatch.

The crate is portable and forbids `unsafe` code. Its kernels favor correctness over vectorization, so treat them as a reference, not a fast path.

## Add the crate

The Cargo package is `turbospark-compute`. Rust code imports it as `turbospark_compute`. Inside the TurboSpark workspace, other crates depend on it under the short alias `compute`. That alias is internal to the workspace, so your own `Cargo.toml` uses the package name.

Add the package from crates.io once the version you need is published. Until then, use a path or git dependency from a checkout:

```toml
[dependencies]
turbospark-compute = { path = "../turbospark/crates/compute" }
```

The minimum supported Rust version is 1.82, set by the workspace `rust-version`.

## Smallest working example

The program below compiles and runs as written against a checkout of this repository. The output shown after it comes from that run.

```rust
use turbospark_compute::quant::GROUP_SIZE;
use turbospark_compute::{
    dequantize_int4_affine, max_abs_diff, quantize_int4_affine, rms_norm, ComputeStrategy,
};

fn main() {
    let _strategy = ComputeStrategy::new();

    // RMSNorm: for [3, 4] the RMS is sqrt(12.5), so each output is x / RMS.
    let x = [3.0f32, 4.0];
    let weight = [1.0f32, 1.0];
    let y = rms_norm(&x, &weight, 1e-6);
    println!("rms_norm: {y:?}");

    // INT4 affine round trip. The row length must be a multiple of GROUP_SIZE.
    let row: Vec<f32> = (0..2 * GROUP_SIZE).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let packed = quantize_int4_affine(&row);
    let back = dequantize_int4_affine(&packed, row.len());
    let err = max_abs_diff(&back, &row);
    println!("int4 max abs error: {err}");
    assert!(err < 0.25, "err = {err}");
}
```

Output:

```text
rms_norm: [0.8485281, 1.1313708]
int4 max abs error: 0.20703125
```

The 0.25 bound in the assertion only shows the pattern. Derive your own bound from the quantization step for real tests.

## What is in the crate

Most kernels are re-exported at the crate root. The groups below cover the main ones. The [docs.rs module list](https://docs.rs/turbospark-compute/latest/turbospark_compute/) has the full set, including model-family pieces such as the Gated-DeltaNet reference (`GdnReference`), Whisper, and the vision tower.

* **Normalization and transforms:** `rms_norm`, `wht`
* **Positional encoding:** `rope_neox`, `rope_paired`, `rope_mrope_interleaved`, `yarn_frequencies`
* **Attention:** `causal_attention`, `causal_attention_with_sinks`, `indexed_attention`, `causal_attention_tq`
* **Quantization:** affine INT4 and INT8 (`quantize_int4_affine`, `dequantize_int4_affine`, `dequant_int4_gemv`, and the INT8 equivalents), plus the 1-bit, 2-bit, GGUF, and KV-cache codecs
* **Feed-forward and sampling:** `run_ffn`, `gelu_tanh`, `logit_softcap_softmax`
* **Parity helpers:** `rel_error`, `max_abs_diff`, `bounded_rel_error`, and the `Tolerance` constants

The `ComputeStrategy` type is covered in the next section because it does less than its name suggests.

## ComputeStrategy is a marker

`ComputeStrategy` is a struct with one private field, so callers cannot build it with a struct literal. It has no methods beyond `new()`, and it derives `Debug`, `Clone`, `Copy`, `Default`, `PartialEq`, and `Eq`. It does not select or dispatch a kernel. The project's roadmap still lists the dispatch trait behind it as open work, so do not build against it as an extension point.

## Parity checks

The `crates/gpu` parity tests import these functions as the CPU reference. The pattern is to compute the reference on the CPU, compute the same result on the device, and compare the two with `rel_error` or `max_abs_diff`. Then check the error against a `Tolerance` constant or a bound you derive.

| Constant | Value | Intended use |
|---|---|---|
| `Tolerance::IDENTITY` | `1e-5` | FP32 identity or pass-through |
| `Tolerance::FP16_REDUCTION` | `5e-3` | One FP16 reduction, such as RMSNorm or a single GEMV |
| `Tolerance::FP16_CHAINED_REDUCTION` | `1e-2` | Chained reductions and fused blocks |
| `Tolerance::QUANT_INT4` | `1.5e-3` | INT4 comparisons. Callers usually derive a tighter bound. |
| `Tolerance::QUANT_INT8` | `1e-3` | INT8 comparisons |

The constants are starting points. The source says quantized comparisons should use a bound derived from the format, such as the scale.

{% hint style="warning" %}
NaN passes some checks by accident. `rel_error` and `max_abs_diff` return NaN when any input is NaN. Written as `err < tol`, the check fails on NaN, which is what you want. A check written as `!(err > tol)` passes NaN.
{% endhint %}

## Errors and panics

No public function returns `Result` or `Option`. Bad input panics through `assert!`. The common cases are:

* A length mismatch between slices in `rel_error`, `max_abs_diff`, `bounded_rel_error`, or `rms_norm`.
* A row length that is not a multiple of `quant::GROUP_SIZE` (64) in the INT4 and INT8 affine functions.
* An empty `weight_rows` slice in `dequant_int4_gemv`.
* A length that is not a non-zero power of two in `wht`.

Validate lengths before you call these functions. Treat a panic as a bug in your input, not as an error to handle.

## What the crate is not for

* **Speed:** Use the GPU crate or the runtime for throughput. Measured numbers live in [docs/BENCHMARKS.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/BENCHMARKS.md), not in this page.
* **A model:** The weights are slices you supply. The crate does not load checkpoints.
* **A dispatch layer:** See the `ComputeStrategy` section above.

## Platform

The crate forbids `unsafe` code. Every `cfg` attribute in its source gates test-only code, and the source has no target-specific branches.

The Linux cross-check (`cargo check --target x86_64-unknown-linux-gnu`) is not re-run in this audit, so its result is unverified here. The test suite has not been run on Linux, and other targets have not been checked.

## Where to go next

* [docs.rs reference for turbospark-compute](https://docs.rs/turbospark-compute/latest/turbospark_compute/) for every public item.
* [GPU kernels](../gpu/README.md), which are validated against this crate.
* [Crate README in the repository](https://github.com/whit3rabbit/turbospark/blob/main/crates/compute/README.md) and the [testing rules](https://github.com/whit3rabbit/turbospark/blob/main/docs/TESTING.md) for contributors.
