# turbospark-model-io

Checkpoint manifests, tensor metadata, mappings, and resident buffers.

## Read first

- [Detailed module guide](../../.claude/docs/modules/model-io.md)
- [GTurbo format](../../docs/GTURBO.md)
- [Model family table](../../docs/MODEL_FAMILY.md)

## Rules

- Keep `unsafe` restricted to the documented narrow boundary: the mmap calls
  in resident buffers and safetensors intake, plus the single `statvfs`
  free-space call in `disk_space`.
- Decode family validation is structural. Resolve the family before applying
  family defaults, and reject missing or mismatched manifest fields.
- Keep on-disk names, canonical family strings, ArchConfig fields, and
  manifest serialization synchronized.
- Distinguish resident allocation from on-disk size. MoE expert tables may
  stream, while slots, KV, and resident core determine the working set.
- A new family or checkpoint is not complete until model-io, runtime, FFI,
  Swift, catalog, and documentation agree.

## Checks

```sh
cargo test -p turbospark-model-io
cargo check --target x86_64-unknown-linux-gnu -p turbospark-model-io
```
