# turbospark-coreml

Generic Apple Core ML runner: load `.mlmodelc` (compiling `.mlpackage` or
`.mlmodel` into a content-keyed cache first), check inputs against the model
description, marshal dense row-major tensors to and from `MLMultiArray`, and
run stateless or stateful (`MLState`) predictions.

Model pipelines (Parakeet, EmbeddingGemma2, Unlimited OCR) are NOT here.
They live in `crates/runtime/src/coreml/`, because they need the tokenizer
(onig_sys), which would drop this crate from the portable cross-target check.

## Rules

- All unsafe code is in `src/sys.rs`. `lib.rs` is `#![deny(unsafe_code)]`.
  Every unsafe block names its invariant in a `SAFETY:` comment.
- NSExceptions are not caught, and must not be. objc2's `exception`
  feature links an Objective-C `@try` helper whose `__objc_personality_v0`
  is a fourth personality routine in the app link (two Rust static libraries
  plus C++ already), and `ld` fails with "Too many personality routines for
  compact unwind". Make every call safe up front instead: check inputs
  against the model description and gate newer APIs behind `supports`.
- Any API newer than macOS 13 (the Swift package's floor) is gated by
  `supports(Capability::...)` first. objc2 resolves selectors at runtime, so
  `setFunctionName:` or `newState` on macOS 14 raises instead of failing to
  link. Callers get `CoreMlError::OsTooOld`.
- Every prediction runs in its own `autoreleasepool`. Decode loops call
  `predict` thousands of times. Without the pool, the autoreleased providers
  and arrays accumulate for the life of the thread.
- Read and write multi-arrays through their reported strides, and bounds-check
  the buffer against `strided_extent` before copying. The Neural Engine can
  return padded rows.
- Refuse dtypes this crate does not marshal (anything but f32, f16, i32)
  rather than converting. f32 to f16 conversion refuses non-finite results.
- `CoreMlModel` serializes predictions with an internal `Mutex`. A
  `CoreMlState` records which model instance created it and is refused by any
  other.
- The compile cache key covers `Manifest.json`, every file's path, size and
  mtime, and the OS build string. Pruning only removes
  `<stem>-<16 hex>.mlmodelc` entries in the cache root the caller passed.
  Callers pass a directory inside the model's install, never a new
  `~/.turbospark` path.

## Tests

- `cargo test -p turbospark-coreml`: unit tests for the stride, cache-key
  and input-check logic, plus `tests/fixtures.rs`, which compiles and runs
  the tiny packages in `testdata/` (no downloads).
- Regenerate fixtures with `tools/gen_coreml_fixtures.py` (the command is in
  its docstring). `testdata/manifest.json` records the generator versions and
  per-package digests.
- The crate is in the cross-target check list in
  `.claude/docs/verification.md`. Keep the non-macOS stub compiling.

## Diagnostics

- `compute_plan` reads `MLComputePlan` (macOS 14.4, gated): the device Core
  ML plans for each ML program op. It walks the plan inside Core ML's
  completion handler and sends back plain Rust, so no Objective-C object
  crosses threads. `examples/coreml_plan.rs` prints it. Use it, not timing,
  to tell whether an export reaches the Neural Engine.
