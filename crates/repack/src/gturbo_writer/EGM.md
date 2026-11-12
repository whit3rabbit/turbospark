# Gturbo Install Writer

## Overview
Assembles a byte-exact `.gturbo` install directory: `packed_experts/layer_NN.bin` blobs plus `layout.json`, `model_weights.bin` (resident index and tensor region), optional `packed_vision/` and `hadamard.bin`, and a `manifest.json` carrying per-file size and SHA-256. Batch functions take everything in memory; `StreamingGturboWriter` writes each layer to disk as it arrives so large checkpoints never hold all expert blobs at once.

## Interface
- `StreamingGturboWriter` [streaming.rs:StreamingGturboWriter]: `new`, `set_quant` (required for production-shape manifests, which the loader rejects without it), `set_hadamard`, `write_layer`, `adopt_layer` (resume), `finish_streaming`, `finish`
- `write_gturbo_install` [layers.rs:write_gturbo_install]: full install with an empty resident index around raw tensor bytes
- `write_gturbo_install_with_resident_index_and_experts` [layers.rs]: caller-supplied complete `model_weights.bin` plus expert layers (streamed MoE)
- `write_gturbo_install_with_resident_index` [layers.rs]: real resident index, zero expert layers
- `write_packed_vision` [layers.rs:write_packed_vision]: `packed_vision/{blobs.bin,layout.json}`, vision tower as one layer of `blocks.experts.len()` blocks (not checked against `arch.vision.depth`)
- `LayerBlobs`, `ExpertBlob`, `SubTensor` [types.rs]: in-memory blob model
- `WriterError` [types.rs:WriterError]: Io, ExpertOversized, WrongExpertCount
- `build_layer_file`, `build_manifest_json`, `build_empty_resident_index` [layers.rs, manifest.rs]: crate-private builders
- Subdirectories: none

## Invariants
- Each layer must hold exactly `experts_per_layer` experts, else `WrongExpertCount` [layers.rs:build_layer_file]
- No expert may exceed the manifest's `expert_stride` ceiling (`ExpertOversized`); a layer is padded to its own page-rounded stride clamped to that ceiling, so mixed installs do not pad to the maximum [layers.rs:build_layer_file]
- `manifest.json` is written last because it hashes every listed file from disk; `write_packed_vision` and `hadamard.bin` must exist first or a missing-file io error results [manifest.rs:build_manifest_json, layers.rs:write_packed_vision]
- Vision files are listed only when `arch.vision.is_active()`, hadamard only when set, not by probing the filesystem [manifest.rs:build_manifest_json]
- `adopt_layer` refuses an existing layer file whose size differs from what the walk would write [streaming.rs:StreamingGturboWriter::adopt_layer]
- `finish_streaming` flushes `model_weights.bin` before the manifest hashes it; files are hashed streamed, not read whole [streaming.rs:finish_streaming, manifest.rs]
- Only `StreamingGturboWriter` can set manifest `quant` or add `hadamard.bin`: the batch writers pass `hadamard_present=false` and leave `quant` null, so they cannot produce a production-shape manifest [layers.rs:write_gturbo_install_impl, manifest.rs:build_manifest_json]
- Family-extension arch fields are written unconditionally so non-Gemma installs validate [manifest.rs:build_manifest_json]

## Side Effects
- Filesystem: creates `packed_experts/`, writes layer files, `layout.json`, `model_weights.bin`, `manifest.json` [streaming.rs, layers.rs]
- Filesystem: reads written files back to compute SHA-256 [manifest.rs:build_manifest_json]
- No env vars, network, or global state

## Verification
- Run: `cargo test -p turbospark-repack`
