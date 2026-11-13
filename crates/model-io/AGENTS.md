# turbospark-model-io

Checkpoint manifests, tensor metadata, mappings, and resident buffers.

## Read first

- [Detailed module guide](../../.claude/docs/modules/model-io.md)
- [GTurbo format](../../docs/GTURBO.md)
- [Model family table](../../docs/MODEL_FAMILY.md)
- [Flint status](../../docs/FLINT.md)

## Rules

- Keep `unsafe` restricted to the documented narrow boundary: the mmap calls
  in resident buffers and safetensors intake, plus the single `statvfs`
  free-space call in `disk_space`.
- Decode family validation is structural. Resolve the family before applying
  family defaults, and reject missing or mismatched manifest fields. One
  known exception: a manifest with no `arch.family` is read as Gemma4
  (`peek_family`, `validate_arch`) because installs written before the field
  existed are Gemma installs, and `tests/manifest.rs` pins it. Rejecting it
  would orphan those installs, so that is an open decision, not a settled rule.
- Keep on-disk names, canonical family strings, ArchConfig fields, and
  manifest serialization synchronized.
- The index layout constants (`HEADER_BYTES`, `ENTRY_BYTES`, `TRAILER_BYTES`,
  `INDEX_MAGIC`, `INDEX_FORMAT_VERSION`) and the manifest version constants
  live here and repack imports them. Bump `MANIFEST_VERSION_MINOR` for a
  layout addition an older reader can skip, and the major (with
  `INDEX_FORMAT_VERSION`) for one an older reader would misparse. Both
  readers refuse a newer value by name instead of guessing.
- Distinguish resident allocation from on-disk size. MoE expert tables may
  stream, while slots, KV, and resident core determine the working set.
- A new family or checkpoint is not complete until model-io, runtime, FFI,
  Swift, catalog, and documentation agree.

## Checks

```sh
cargo test -p turbospark-model-io
cargo check --target x86_64-unknown-linux-gnu -p turbospark-model-io
```
