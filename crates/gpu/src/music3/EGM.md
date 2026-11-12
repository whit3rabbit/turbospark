# gpu music3

## Overview
Music 3 sequence encoding: several operations (projections, norms, attention) recorded into one command buffer with a single readback, using the same kernel plans as the synchronous forms in the parent module so outputs are bit-identical.

## Interface
- [sequence.rs:Music3Sequence], [sequence.rs:Music3Slot], [sequence.rs:Music3Table] (from `table`), [sequence.rs:Music3Buffer] (`len`, `is_empty`, `write`, `read`).
- [sequence.rs:Music3Sequence] operations: `upload`, `bind`, `copy`, `rope`, `view`, `linear`, `convolution`, `snake`, `rms_norm`, `layer_norm`, `attention`, `add_round_bf16`, `swiglu_bf16`, `partial_rope_bf16`, `gated_silu_bf16`, `add_f32`, `partial_rope_f32`, `gated_silu_f32`, `finish`.
- Constructors on the device wrapper: `table`, `buffer`, `sequence` [sequence.rs].

## Invariants
- Intermediates stay on the device and each result is bit-identical to the per-operation result, because plans (kernel choice and parameters) are shared with the parent module [sequence.rs] (module doc).
- Buffer `write` and `read` return `GpuError` on out-of-range access [sequence.rs:Music3Buffer].

## Side Effects
- Allocates Metal buffers and commits one command buffer per sequence [sequence.rs:Music3Sequence].

## Verification
- Run: `cargo test -p turbospark-gpu`
