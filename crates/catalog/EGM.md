# turbospark-catalog Blueprint

## System Overview
The curated model catalog, the header-only Hugging Face probe, and the install
driver that turns either into a `.gturbo` directory (package
`turbospark-catalog`, see `Cargo.toml`). It also holds the per-modality pinned
catalogs, the install store and libraries, and the memory-fit recommender. It
is portable and forbids unsafe code [src/lib.rs:forbid(unsafe_code)].

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - catalog, store, hub, install, and modality modules plus the crate's public re-exports.
- src/probe: [./src/probe/EGM.md](./src/probe/EGM.md) - header-only GGUF and safetensors compatibility probe.
- src/recommend: [./src/recommend/EGM.md](./src/recommend/EGM.md) - fit arithmetic, ranking, and Hugging Face discovery.
- src/library: [./src/library/EGM.md](./src/library/EGM.md) - external-drive model libraries, move, delete, import.
- examples: [./examples/EGM.md](./examples/EGM.md) - `audio_pin` tool for pinning audio catalog rows.

## Global Invariants
- Workspace dependencies are core, model-io, repack, tokenizer, audio, and coreml (macOS version check only), plus serde, serde_json, reqwest, and sha2; nothing here decodes or dispatches Metal [Cargo.toml, src/lib.rs].
- Catalog rows name a repository and revision that were actually run; the probe covers everything else [src/lib.rs: module docs].
- Drive and Hub content is untrusted and validated before use [src/library/marker.rs, src/hub_validation.rs].

## Global Side Effects
- Hugging Face network traffic and disk writes under the store root and library drives [src/hf.rs, src/store.rs, src/library/transfer.rs].

## System Verification
- Run: `cargo test -p turbospark-catalog`
