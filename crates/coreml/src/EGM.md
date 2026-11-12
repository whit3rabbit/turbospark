# turbospark-coreml (src)

## Overview
Safe Rust wrapper around Apple Core ML: load or compile a model, describe features, run (stateful) predictions, and query the compute plan. A `sys` module implements it on macOS and a `stub` module elsewhere: fallible calls return `UnsupportedPlatform`, while `supports` is false, `os_version` is `(0, 0, 0)`, and `inputs`, `outputs`, `states` and `path` are empty [stub.rs:supports], [stub.rs:CoreMlModel].

## Interface
- Model and state: [sys.rs:CoreMlModel] (`load`, `open`, `inputs`, `outputs`, `states`, `predict`, `new_state`, `predict_with_state`), [sys.rs:CoreMlState] (`specs`, `copy_from`, `read`); `CoreMlModel::path` and `function` are also public, and `Tensor` has accessors (`shape`, `dtype`, `data`, `into_data`, `as_f32`, `as_f16`, `as_i32`, `to_f32_vec`) [tensor.rs:Tensor]; identical stub API in [stub.rs:CoreMlModel].
- Free functions: [sys.rs:compile_if_needed], [sys.rs:compute_plan], [sys.rs:os_version], [sys.rs:supports].
- Feature `ane-prefill` (macOS only): [sys.rs:SharedF16Plane] (`new`, `base_ptr`, `alloc_size`, `rows`, `cols`, `stride`) and `predict_shared`.
- Options and plan types: [lib.rs:ComputeUnits], [lib.rs:Capability], [lib.rs:LoadOptions], [lib.rs:FeatureKind], [lib.rs:FeatureSpec], [lib.rs:ComputeDevice], [lib.rs:OpPlacement], [lib.rs:DeviceShare], [lib.rs:ComputePlan], [lib.rs:check_input], [lib.rs:check_inputs].
- Tensors: [tensor.rs:Tensor], [tensor.rs:TensorData], [tensor.rs:DType], [tensor.rs:element_count] (constructors `f32`, `i32`, `f16_from_f32`, `zeros`).
- Compile cache: [cache.rs:cache_key], [cache.rs:compiled_path_for].
- Errors: [error.rs:CoreMlError] (UnsupportedPlatform, OsTooOld, Load, Compile, Shape, ...), [error.rs:Result].

## Invariants
- Platform split is by cfg: `sys` only on macOS, `stub` otherwise, with the same exported names, except `SharedF16Plane`, which is re-exported only from `sys` under `ane-prefill` [lib.rs:sys] / [lib.rs:stub].
- Every predict input must be declared exactly once (duplicate names are refused), every required input supplied, and each tensor must match its spec [lib.rs:check_inputs].
- `predict_with_state` refuses a `CoreMlState` made by a different `CoreMlModel` instance (process-unique owner id) [sys.rs:CoreMlModel::predict_with_state].
- `compute_plan` takes only a compiled `.mlmodelc` directory and refuses networks that are not ML programs or pipelines [sys.rs:compute_plan], [sys.rs:walk_structure].
- `predict_shared` needs every input and output bound and refuses a result Core ML did not write into the backing [sys.rs:predict_shared].
- Strided reads are bounds-checked with `strided_extent` (negative strides refused) before copying; ANE padding must not appear in outputs [tensor.rs:Tensor].
- The compiled-model cache key hashes `Manifest.json` bytes, then each regular file's relative path, size and mtime (weight bytes are not read; symlinks skipped on purpose), then the OS build string, so a changed package or OS update recompiles. A single-file `.mlmodel` is hashed in full [cache.rs:cache_key], [sys.rs:os_tag].
- Cold compile and publish run under one process-wide `static` Mutex with a second cache check inside the lock, because Core ML may hand the same temp dir to simultaneous compilers; prediction stays per-model [sys.rs:serialized_cold_compile].
- `CoreMlState::copy_from` needs every buffer to match in name, shape, strides and dtype; the source may come from another model of the same package [sys.rs:CoreMlState].
- Capability gates: `MultiFunction` and `State` need macOS 15.0, `AsyncCompile` and `ComputePlan` 14.4 [lib.rs:Capability::required_macos].
- Before macOS 15, `CoreMlModel::load` (compiled `.mlmodelc` directories only; `open` takes packages) reports an empty `states()` without error, and `new_state` returns `OsTooOld` [sys.rs:CoreMlModel].
- Objective-C calls are wrapped in `unsafe` blocks with SAFETY comments [sys.rs:CoreMlModel].

## Side Effects
- Compiling a `.mlpackage` writes a compiled model into the cache directory (copying across volumes when the rename fails) plus a `.source` marker per entry, and prunes older entries and `.staging-*` dirs over one hour old [sys.rs:compile_if_needed], [sys.rs:prune_stale], [cache.rs:compiled_path_for].
- Loading with the Neural Engine can trigger a first compile (minutes for large models) [sys.rs:CoreMlModel].
- Allocates IOSurface-backed shared buffers (bytes per row aligned via IOSurfaceAlignProperty) for the ANE split under the `ane-prefill` feature [sys.rs:SharedF16Plane].

## Verification
- Run: `cargo test -p turbospark-coreml`
