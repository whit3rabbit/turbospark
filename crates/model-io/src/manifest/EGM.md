# manifest (model-io)

## Overview
Loads and validates `manifest.json` of a `.gturbo` model install, checking version, required files, flags, quantization records and the arch block against the resolved `ArchConfig`.

## Interface
- `load(dir, expecting, max_bytes)`, `validate`, `peek_family`, `known_flags` [manifest/mod.rs:load]
- `MANIFEST_VERSION_MAJOR`, `MANIFEST_VERSION_MINOR`, `REQUIRED_FILES`, `DEFAULT_MAX_BYTES` [manifest/mod.rs]
- `Manifest`, `ManifestArch`, `ManifestFileEntry`, `ManifestQuant`, `ManifestQuantSlot`, `ManifestProducer`, `ManifestHadamard`, `ManifestHadamardSigns` [manifest/types.rs:Manifest]
- `EXECUTABLE_GGUF_TYPES`; `validate_quant` is `pub(crate)` [manifest/quant.rs:EXECUTABLE_GGUF_TYPES]

## Invariants
- A missing `manifest.json` is `PartialInstall`; oversize is `IndexCorrupt` (cap 4 MiB by default) [manifest/mod.rs:load]
- Any minor up to `MANIFEST_VERSION_MINOR` loads, a newer one is refused by name [manifest/mod.rs:MANIFEST_VERSION_MINOR]
- Flags outside `streamingPresent`, `turboQuantKV`, `aneSharedExpert` are an error; `turboQuantKV: true` is itself refused with `IndexCorrupt` (not a flag this port reads) [manifest/mod.rs:known_flags] [manifest/mod.rs:validate]
- A manifest whose `expert_stride` is not a multiple of 4096 is `ExpertStrideNotPageAligned`; 4096 is a hard-coded floor, not the OS page size [manifest/mod.rs:page_size_bytes]
- `capability == "image-generation"` is refused with `UnsupportedCapability` [manifest/mod.rs:load]
- `quant` is required when the arch matches a shipped baseline [manifest/mod.rs:is_production_arch]
- Per-layer `packed_experts/layer_NN.bin` (or `layer_N.bin`) files are required, plus `packed_vision/layout.json` and `blobs.bin` when the arch has a vision tower [manifest/mod.rs:validate]
- An optional `hadamard` section is validated and requires `hadamard.bin` [manifest/mod.rs:validate_hadamard]
- A missing `arch.family` resolves to Gemma4 in `peek_family` rather than erroring [manifest/mod.rs:peek_family]
- `model_weights.bin` and `packed_experts/layout.json` are required [manifest/mod.rs:REQUIRED_FILES]

## Side Effects
- Reads `manifest.json` from the install directory with a size-capped read [manifest/mod.rs:load]

## Verification
- Run: `cargo test -p turbospark-model-io`
