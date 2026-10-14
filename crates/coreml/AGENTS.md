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
- Any Objective-C call that can raise goes through `guarded`
  (`objc2::exception::catch`). An uncaught NSException aborts the process.
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
