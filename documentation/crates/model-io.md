---
description: "The .gturbo install reader: manifest and architecture validation, packed-expert layout, resident index and mmap'd weights, SHA-256 verification, and the trusted install receipt."
icon: box-open
---

# model-io

`crates/model-io` (package `turbospark-model-io`, library
`turbospark_model_io`) owns the on-disk contract of a `.gturbo` model
installation: the `manifest.json` schema and its structural validation
against a resolved `ArchConfig`, the packed-expert layout, the resident
tensor index and its mmap'd weight bytes, SHA-256 file verification, and
the trusted install receipt. It also carries the capacity policy math
(context windows, KV byte counts, expert residency, load guardrails) that
decides whether an install can be opened within a memory budget. Every
consumer of an install, from `runtime` to `cli` to `ffi`, opens one by
calling this crate's loaders first.

## What it does

| Area | Modules | Contents |
| --- | --- | --- |
| Manifest | `src/manifest/` | `load_manifest`, `validate_manifest`, `peek_family`: decode `manifest.json` and validate it structurally against an expected `ArchConfig` |
| Architectures | `src/arch_config/`, `src/arch_baselines/` | `ArchConfig` and `ModelFamily`, per-family baseline constructors (`gemma4_26b_a4b`, `qwen_gdn_moe_35b_a3b`, ...), and `known_architecture` |
| Resident weights | `src/resident_index.rs`, `src/resident_buffer.rs` | The fixed-width index of `model_weights.bin` (24-byte header, 72-byte entries) and an mmap'd byte window over the tensor region |
| Packed experts | `src/packed_experts_layout.rs` | `packed_experts/layout.json`: per-layer, per-expert sub-tensor offsets; the same schema is reused for `packed_vision/` |
| Integrity | `src/sha256.rs`, `src/install_receipt.rs` | Streaming SHA-256 hashing and verification, plus the `verified-install.json` receipt and its binding checks |
| Capacity policy | `src/context_policy.rs`, `src/expert_cache_policy.rs`, `src/load_guard.rs`, `src/kv_quant.rs`, `src/cgroup.rs`, `src/disk_space.rs` | Context caps and KV byte math, expert cache sizing, load guardrails, TurboQuant KV row sizing, cgroup and free-disk probes |
| Other install formats | `src/ngram_table.rs`, `src/vision_sidecar.rs`, `src/steering_set.rs`, `src/speech_receipt.rs`, `src/whisper_config.rs`, `src/encoder_config.rs` | PLE n-gram tables, vision-tower sidecars, steering sets, speech receipts, Whisper and encoder configs |
| Intake reader | `src/safetensors.rs` | `SafetensorsFile`, an mmap'd `.safetensors` reader used on the model-intake side |

## Where it sits

`turbospark-model-io` depends only on `turbospark-core` (aliased
`foundation`, which supplies the interchange `TokenId` type) plus `serde`,
`sha2`, `memmap2`, and `libc`. It sits underneath almost everything:
`repack` writes exactly the files these loaders read, `runtime`, `gpu`, and
`streaming` consume the layouts at open and decode time, and `catalog`,
`cli`, `server`, `ffi`, `bench`, `audio`, and `image` all validate installs
through it. There are no Cargo feature flags; the crate is part of the
portable cross-target check except for the deliberate `mmap` boundary.

`unsafe` is confined to the documented I/O boundary: the `mmap` calls in
`resident_buffer.rs` and `safetensors.rs` and the `statvfs` call in
`disk_space.rs`. Everything else is safe Rust.

## Public API tour

The crate root in `crates/model-io/src/lib.rs` re-exports the surface.
The entry points you will reach for most often:

| Item | Path | Meaning |
| --- | --- | --- |
| `ArchConfig`, `ModelFamily` | `src/arch_config/` | The resolved architecture description every loader validates against; `ModelFamily` selects the decode flow downstream |
| `known_architecture(family)`, `all_known_architectures()` | `src/arch_baselines/mod.rs` | The baseline `ArchConfig` for a family, and the full known table |
| `peek_family(dir, max_bytes)` | `src/manifest/mod.rs` | Read just the family string from a manifest |
| `load_manifest(dir, &arch, max_bytes)`, `validate_manifest(&m, &arch)` | `src/manifest/mod.rs` | Decode and structurally validate `manifest.json` |
| `Manifest`, `ManifestArch`, `ManifestQuant`, `ManifestFileEntry` | `src/manifest/types.rs` | The serde types behind the manifest (camelCase on disk) |
| `load_packed_experts_layout(dir, max_bytes)`, `PackedExpertsLayout::expert(layer, expert)` | `src/packed_experts_layout.rs` | Per-layer, per-expert blob addressing |
| `load_resident_index(path)`, `ResidentIndex`, `ResidentIndexEntry` | `src/resident_index.rs` | Named tensor entries of `model_weights.bin`; `HEADER_BYTES` (24) and `ENTRY_BYTES` (72) travel with the reader |
| `ResidentBuffer::map(path, offset, size)`, `.data()`, `.mapped_bytes()` | `src/resident_buffer.rs` | The mmap'd resident region as byte slices |
| `SafetensorsFile::open(path)`, `.contains_tensor`, `.load_as_f32` | `src/safetensors.rs` | mmap'd safetensors inspection |
| `hash_file`, `hash_data`, `verify_file` | `src/sha256.rs` | Streaming hashing and checksum verification |
| `load_install_receipt`, `validate_install_receipt_manifest_binding`, `VerifiedInstallReceipt`, `ModelIntegrityPolicy` | `src/install_receipt.rs` | The trusted install receipt (`FullSha256` or `SizeCheckTrustedReceipt` policy) |
| `kv_bytes_for_context`, `resolve_max_context`, `largest_context_within`, `session_pool_bytes` | `src/context_policy.rs` | KV byte math, context caps, and refusal types (`ContextTooLarge`, `ContextRefused`) |
| `ExpertCacheSlots`, `mapped_residency_fits`, `ExpertResidency` | `src/expert_cache_policy.rs` | How many expert slots fit and whether mapped residency is preferred |
| `LoadGuard`, `LoadPolicy`, `GuardBudget` | `src/load_guard.rs` | Memory guardrails applied at model open |
| `KvQuant`, `kv_layer_strides`, `tq_row_bytes` | `src/kv_quant.rs` | TurboQuant KV row sizing per layer |
| `probe_cgroup_memory`, `available_bytes` | `src/cgroup.rs`, `src/disk_space.rs` | Environment limits an install path must respect |
| `ModelError` | `src/error.rs` | One error enum with a named variant per refusal (`PartialInstall`, `ArchMismatch`, `ChecksumMismatch`, `IndexCorrupt`, ...) |

Byte caps are explicit everywhere: `DEFAULT_MAX_BYTES` (4 MiB) caps a
manifest read and `PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES` (64 MiB) caps a
layout read, both defending against hostile metadata sizes before parse.

## Examples

### Peek the family, then load and validate the manifest

Adapted from `crates/model-io/tests/manifest.rs` and the round trip in
`crates/repack/tests/gturbo_writer.rs`. Family dispatch happens on the
peeked `ModelFamily`; the manifest load then fails with a named `ModelError`
unless magic, version, flags, arch fields, the quant block, the page-aligned
expert stride, and the required files all agree with the expected baseline.

```rust
use turbospark_model_io::{
    known_architecture, load_manifest, peek_family, DEFAULT_MAX_BYTES,
};

let dir = std::path::Path::new(
    "/Users/you/.turbospark/models/text/example.gturbo",
);

let family = peek_family(dir, DEFAULT_MAX_BYTES).unwrap();
let arch = known_architecture(family);
let manifest = load_manifest(dir, &arch, DEFAULT_MAX_BYTES).unwrap();
assert_eq!(manifest.magic, "GTURBO");
```

### Read the resident index and map the resident weights

Adapted from `crates/model-io/tests/resident_index.rs` and
`tests/resident_buffer.rs`. The index is read first; the buffer is mapped
over `[index_size, index_size + resident_size)` of the same file, and
`data()` presents that region starting at logical offset 0. A truncated
file fails at `map` with `ModelError::TensorSizeMismatch` instead of
SIGBUS on first touch.

```rust
use turbospark_model_io::{load_resident_index, ResidentBuffer};

let weights_path = dir.join("model_weights.bin");
let index = load_resident_index(&weights_path).unwrap();

let buffer = ResidentBuffer::map(
    &weights_path,
    index.header.index_size,
    index.header.resident_size,
).unwrap();
let resident: &[u8] = buffer.data();

// Entry offsets are absolute file offsets at or after `index_size`.
let entry = &index.entries["model.embed_tokens.weight"];
let start = (entry.file_offset - index.header.index_size) as usize;
let bytes = &resident[start..start + entry.size_bytes as usize];
```

### Address one packed expert blob

Adapted from the read-back half of `crates/repack/tests/gturbo_writer.rs`,
which loads a written install through this exact loader. Sub-tensor roles
are writer-defined strings such as `gate`, `gate_scales`, and `gate_biases`;
GGUF installs carry lowercased ggml dtype names per sub-tensor.

```rust
use turbospark_model_io::{
    load_packed_experts_layout, PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
};

let layout = load_packed_experts_layout(
    dir,
    PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
)
.unwrap();
let blob = layout.expert(1, 0); // layer 1, routed expert 0
let gate = &blob.sub_tensors["gate"];
println!(
    "gate: {} bytes at +{}, dtype {}",
    gate.size, gate.offset, gate.dtype
);
```

## Gotchas

{% hint style="warning" %}
`PackedExpertsLayout::expert_stride` is the model-wide MAXIMUM stride, the
one `manifest.json` declares. A mixed-quantization install writes a
different, smaller stride per layer, so size and address a specific layer's
blob with `LayerLayout::expert_stride`, never with the top-level value.
{% endhint %}

{% hint style="warning" %}
Manifest validation is intentionally strict, and the refusals are load
bearing. Unknown `flags` keys are errors; `turboQuantKV: true` is refused
outright because TurboQuant KV quantization is an open-time choice
(`--kv-bits`), not a manifest property; and a manifest whose capability is
`image-generation` is rejected by this reader rather than misread as a text
install.
{% endhint %}

Keep the crate's own distinction in mind when reasoning about memory:
resident allocation is not on-disk size. Routed expert tables may stream
from `packed_experts/`, while resident core weights, KV state, and cache
slots determine the working set the capacity-policy modules compute.

## Related pages

- [The .gturbo Format](../models/gturbo-format.md)
- [Model Families](../models/model-families.md)
- [streaming](streaming.md)
- [Wiring New Models](../developer/wiring-new-models.md)
