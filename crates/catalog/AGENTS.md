# turbospark-catalog

This is the catalog boundary. It turns published artifacts into validated
local installs, aliases, and model-store entries.

Keep it exact. A catalog row controls what users can download, open, and
measure, so a convenient approximation is still a broken row.

## Read first

- [Detailed module guide](../../.claude/docs/modules/catalog.md)
- [Model catalog contract](../../docs/MODELS.md)
- [GTurbo format](../../docs/GTURBO.md)
- [New model checklist](../../docs/NEW_MODEL.md)
- [Flint status](../../docs/FLINT.md)

## Rules

- This crate is safe Rust and must keep `#![forbid(unsafe_code)]`.
- Keep catalog aliases, persisted family strings, probe results, and install
  metadata in sync. A row is not support until its artifact and runtime
  contract agree.
- Probe headers and file lists before downloading weights. A network rot guard
  must validate the published file list, byte count, and GGUF headers without
  downloading the payload.
- Keep `download_bytes` as the fingerprint for main-pinned artifacts.
- Row status is evidence: `verified` has a frozen quality-gate or
  memory-oracle row, `runs` is installed and generated here without one, and
  `caveat` runs with a stated disqualification. Never upgrade a status without
  the evidence behind it.
- A repo can ship sharded safetensors with no `model.safetensors.index.json`.
  Shard discovery then reads the repository file listing: a consolidated
  `model.safetensors` wins, a complete `-of-N` set is accepted in order, and a
  partial `-of-N` set is refused, because a model with missing layers still
  opens.
- The flint intake keeps its own arms in `probe/safetensors.rs` and
  `stream.rs`: the config parses through `repack::parse_flint_config`, the
  INT8 embedding comes from `repack::parse_flint_quantization` (it lives in
  `mlx_export`, not `quantization`), and each shard header is canonicalized
  before the shared walk. Do not fold these into the generic safetensors path.
- An existing directory argument wins over an alias. Preserve that CLI
  compatibility.

## Checks

```sh
cargo test -p turbospark-catalog
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture
cargo test -p turbospark-catalog --test audio_catalog_network --release -- --ignored --nocapture
cargo test -p turbospark-catalog --test music_catalog_network --release -- --ignored --nocapture
```

Run the network target after changing a catalog row. Record durable facts in
`docs/MODELS.md`, not in this file.
