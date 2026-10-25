---
description: "The Apple Core ML runner: compile and load .mlpackage and .mlmodelc models, marshal dense tensors, and run stateless or stateful (MLState) predictions, with a portable stub off macOS."
icon: apple-whole
---

# coreml

`crates/coreml` (package `turbospark-coreml`) is the generic Apple Core ML runner. It loads `.mlmodelc` (compiling a `.mlpackage` or `.mlmodel` into a content-keyed cache first), checks inputs against the model description, marshals dense row-major tensors to and from `MLMultiArray`, and runs stateless or stateful (`MLState`) predictions. The crate knows nothing about any model: the Core ML pipelines (Parakeet, EmbeddingGemma2, Unlimited OCR) live in `crates/runtime/src/coreml/`, because they need the tokenizer and audio front ends this crate must not depend on. On targets other than macOS every entry point returns `CoreMlError::UnsupportedPlatform`, which keeps the crate in the portable cross-target check.

## What it does

| Responsibility | Where |
|---|---|
| Compile and load | `CoreMlModel::open` and `compile_if_needed`. The compile-cache key covers `Manifest.json`, every file's path, size, and mtime, and the OS build string, so a changed package or OS recompiles. |
| Describe and validate | `inputs()`, `outputs()`, and `states()` return `FeatureSpec`s from the model description. The pure `check_input` / `check_inputs` refuse wrong dtype, wrong shape, unknown names, missing required inputs, and duplicate names before Core ML is touched. |
| Marshal tensors | `Tensor`, `TensorData`, and `DType` (f32, f16, i32 only). Copies go through the array's reported strides with a bounds check, because the Neural Engine can return padded rows. |
| Predict | `predict`, serialized by an internal `Mutex` and run in its own autoreleasepool per call, and `predict_with_state` for `MLState` models (macOS 15). |
| Plan | `compute_plan` reads `MLComputePlan` (macOS 14.4) into per-operation placements; `ComputePlan::summary` groups op count and cost weight per function and preferred device. |

## Where it sits

Depends on:

- `half` (f16 to f32 conversion) and `sha2` (compile-cache key).
- The objc2 family (`objc2`, `block2`, `objc2-foundation`, `objc2-core-ml`), compiled only for macOS. No other TurboSpark crate, so the runner never drags tokenizer or audio dependencies into a cross-target build.

Used by:

- [runtime](runtime.md): a macOS-gated dependency. The model pipelines built on this runner live in `crates/runtime/src/coreml/` and are documented under [Core ML models](../models/coreml.md).
- The catalog: availability checks (`os_version`) for Core ML embedding, OCR, and audio profiles.

## Public API tour

### Loading and configuration

- `CoreMlModel::open(path, cache_root, &LoadOptions)`: loads a compiled model, compiling a package into `cache_root` first. `CoreMlModel::load` requires an already-compiled `.mlmodelc`.
- `LoadOptions`: `compute_units` plus an optional `function` for multifunction packages (`Capability::MultiFunction`).
- `ComputeUnits`: `CpuOnly`, `CpuAndGpu`, `CpuAndNeuralEngine`, or `All` (default), with `parse` and `as_str`.
- `compile_if_needed(path, cache_root)`: returns the compiled path; a compiled input passes straight through, and a repeated call returns the same entry.
- `supports(Capability)` and `os_version()`: gate APIs newer than macOS 13 before calling them. `Capability::required_macos()` gives the floor: `State` and `MultiFunction` need 15.0, `AsyncCompile` and `ComputePlan` need 14.4.

### Tensors and features

- `Tensor::f32(shape, data)`, `Tensor::i32(shape, data)`, `Tensor::f16_from_f32(name, shape, data)` (refuses a non-finite conversion result), and `Tensor::zeros`; readers `shape()`, `dtype()`, `as_f32()`, `as_i32()`, and `to_f32_vec()`.
- `FeatureSpec` / `FeatureKind`: a `MultiArray` (dtype and shape are `Option` because flexible shapes are left to Core ML to validate), a `State` buffer, or `Other(raw MLFeatureType)` for feature types this crate does not marshal.
- `check_input(spec, tensor)` and `check_inputs(specs, inputs)`: pure validation, testable without a model.

### Prediction and state

- `predict(&[(&str, &Tensor)]) -> HashMap<String, Tensor>` over output names.
- `new_state()`, `predict_with_state(&mut state, inputs)`, `state.read(name)`, and `state.copy_from(&other)`. A `CoreMlState` records which model instance created it and is refused by any other.
- `inputs()`, `outputs()`, `states()`, `path()`, and `function()` on the loaded model.

### Compute plan

- `compute_plan(path, opts) -> ComputePlan` with `ops: Vec<OpPlacement>`: per operation, the program function, operator, output name, preferred and supported `ComputeDevice`s, and Core ML's estimated cost share.
- `ComputePlan::summary() -> Vec<DeviceShare>`: op count and summed cost per function and preferred device, sorted.
- Use the plan, not timing, to tell whether an export reaches the Neural Engine.

### Errors

`CoreMlError` variants: `UnsupportedPlatform`, `OsTooOld { feature, required }`, `Load`, `Compile`, `Shape`, `DType`, `MissingFeature`, `NonFinite`, `Prediction`, and `Io`.

## Examples

The examples need a model on disk. `crates/coreml/testdata/` holds tiny generated packages (`tools/gen_coreml_fixtures.py`, digests in `testdata/manifest.json`) that the tests compile and run; substitute your own `.mlpackage` or `.mlmodelc`.

### Describe and predict

Adapted from `examples/coreml_describe.rs` and `tests/fixtures.rs`. The first open compiles the package into the cache directory; describe loads on the CPU so reading a description never triggers a Neural Engine compile.

```rust
use std::path::Path;

use turbospark_coreml::{ComputeUnits, CoreMlModel, LoadOptions, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let package = Path::new("add_f32.mlpackage");
    let cache = package.parent().unwrap().join(".compiled");

    let model = CoreMlModel::open(package, &cache, &LoadOptions::new(ComputeUnits::CpuOnly))?;
    for spec in model.inputs() {
        println!("input {} {:?}", spec.name, spec.kind);
    }

    let a = Tensor::f32(vec![2, 3], vec![1., 2., 3., 4., 5., 6.])?;
    let b = Tensor::f32(vec![2, 3], vec![10., 20., 30., 40., 50., 60.])?;
    let out = model.predict(&[("a", &a), ("b", &b)])?;
    println!("sum: {:?}", out["sum"].to_f32_vec());
    Ok(())
}
```

Wrong dtypes, wrong shapes, and missing inputs are refused by the input check before Core ML runs, so the failure names the feature.

### Carry state across predictions

Adapted from the `stateful_accumulator_carries_state` test in `tests/fixtures.rs`:

```rust
use std::path::Path;

use turbospark_coreml::{
    supports, Capability, ComputeUnits, CoreMlModel, LoadOptions, Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // MLState arrived in macOS 15. Gate the call; on an older OS the
    // model open itself reports CoreMlError::OsTooOld.
    assert!(supports(Capability::State));

    let model = CoreMlModel::open(
        Path::new("accumulator.mlpackage"),
        Path::new(".compiled"),
        &LoadOptions::new(ComputeUnits::CpuOnly),
    )?;

    let x = Tensor::f16_from_f32("x", vec![1, 4], &[1., 2., 3., 4.])?;
    let mut state = model.new_state()?;
    let first = model.predict_with_state(&mut state, &[("x", &x)])?;
    let second = model.predict_with_state(&mut state, &[("x", &x)])?;
    println!(
        "{:?} then {:?}",
        first["y"].to_f32_vec(),
        second["y"].to_f32_vec()
    );
    Ok(())
}
```

Decode loops use this shape: one `CoreMlState` carried across `predict_with_state` calls, with `copy_from` to hand the buffers to a fresh model instance.

### Ask where Core ML plans to run

`examples/coreml_plan.rs` prints the `MLComputePlan` for the configured compute units:

```sh
cargo run -p turbospark-coreml --example coreml_plan -- Model.mlpackage --units ane
cargo run -p turbospark-coreml --example coreml_describe -- Model.mlpackage
```

The same read from Rust:

```rust
use std::path::Path;

use turbospark_coreml::{compile_if_needed, compute_plan, ComputeUnits, LoadOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compiled = compile_if_needed(Path::new("Model.mlpackage"), Path::new(".compiled"))?;
    let plan = compute_plan(&compiled, &LoadOptions::new(ComputeUnits::All))?;
    for row in plan.summary() {
        println!("{} {:?} {} ops", row.function, row.device, row.ops);
    }
    Ok(())
}
```

## Gotchas

{% hint style="warning" %}
objc2 resolves classes and selectors at runtime, so an API newer than the OS raises instead of failing to link. Check `supports(Capability::...)` before anything newer than macOS 13 and expect `CoreMlError::OsTooOld`. NSExceptions are deliberately not caught (objc2's `exception` feature adds a fourth personality routine to the app link, which compact unwind cannot encode), so make every call safe up front: validate inputs against the description with `check_inputs`, and do not feed a dtype this crate does not marshal (f32, f16, i32 only). Read outputs through the tensor's reported strides; the Neural Engine can return padded rows.
{% endhint %}

{% hint style="warning" %}
Pass a compile-cache directory inside the model's install, never a new `~/.turbospark` path: pruning only removes `<stem>-<16 hex>.mlmodelc` entries in the root the caller passed. A `CoreMlState` belongs to the model instance that created it and is refused by any other; use `copy_from` to continue from its buffers on a new instance. Asking for the Neural Engine can trigger its first compile, which takes minutes for large models; `coreml_describe` loads on the CPU to avoid that.
{% endhint %}

## Related pages

- [Core ML models](../models/coreml.md): the pipelines built on this runner
- [runtime](runtime.md): where the model pipelines live
- [Model families](../models/model-families.md)
- [Crate overview](overview.md)
- [Testing and verification](../developer/testing-and-verification.md)
