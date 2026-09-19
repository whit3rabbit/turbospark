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

## Rules

- This crate is safe Rust and must keep `#![forbid(unsafe_code)]`.
- Keep catalog aliases, persisted family strings, probe results, and install
  metadata in sync. A row is not support until its artifact and runtime
  contract agree.
- Probe headers and file lists before downloading weights. A network rot guard
  must validate the published file list, byte count, and GGUF headers without
  downloading the payload.
- Keep `download_bytes` as the fingerprint for main-pinned artifacts.
- An existing directory argument wins over an alias. Preserve that CLI
  compatibility.

## Checks

```sh
cargo test -p turbospark-catalog
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture
```

Run the network target after changing a catalog row. Record durable facts in
`docs/MODELS.md), not in this file.
