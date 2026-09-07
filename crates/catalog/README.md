# turbospark-catalog

The curated model catalog, the header-only Hugging Face probe, and the install driver that turns either into a `.gturbo` directory. Backs the `turbospark-model` binary in `crates/cli`.

User-facing documentation: [`docs/MODELS.md`](../../docs/MODELS.md).

Downstream workspace crates import this package via the `catalog` alias:

```toml
[dependencies]
catalog = { package = "turbospark-catalog", path = "../catalog" }
```

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Key Modules

- `entry.rs`: `CatalogEntry`, one curated row (`Source`, `Sidecars`, `SourceKind`, `Status`, `MtpSource`, `Measured`). `CatalogEntry::validate` enforces the structural invariants at load.
- `catalog.rs`: `Catalog`, the embedded table (`models.json`, via `include_str!`) merged with an optional `$TURBOSPARK_HOME/models.json` override by alias.
- `hf.rs`: `Client`, KB-scale Hugging Face endpoints only (file listing, small-file GET, `content_length`). Never downloads weights; those stream through `repack::HttpRangeSource`.
- `auth.rs`: Hugging Face token resolution (explicit, env, store, cache file) and validation against `whoami-v2`.
- `probe/`: reads a repository's header and decides whether it would run here -- architecture, block types or affine width, expert stride, tokenizer sidecars -- at the cost of KB and seconds, never a download. `gguf.rs` and `safetensors.rs` each own one format's gates; `mod.rs` is the dispatcher and the sidecar/chat-template check.
- `recommend/`: what this machine should run. `fit.rs` is the memory arithmetic (reusing `model_io`'s sizing policies); `rank.rs` orders candidates (tiering adapted from shoehorn, see `NOTICE`); `discover.rs` surveys popular Hugging Face repos through `probe/`.
- `vision.rs`: finds the one installed vision-tower sidecar pairing with a given text family and hidden size.
- `install.rs`: the walk driver -- probe, fetch and verify tokenizer sidecars, stream the weights, read the install back, record it -- written once instead of once per checkpoint family.
- `stream.rs`: GGUF and MLX streaming and shard-writing helpers `install.rs` drives.
- `store.rs`: `Store`, the `~/.turbospark` layout, `installed.json`, and alias/path resolution (`resolve_model_arg`).

## Development & Test Commands

```sh
# Offline tests: the table, the store, and every probe gate.
cargo test -p turbospark-catalog

# The rot guard. A file list and a HEAD per row, ~26 s for the whole table,
# downloads nothing.
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture
```

## Usage

The binary lives in `crates/cli` (`turbospark-model`):

```sh
# List the curated catalog.
cargo run -p turbospark-cli --bin turbospark-model -- list

# Header-only probe of an arbitrary Hugging Face repo: architecture, block
# types, expert stride, tokenizer sidecars. Reads KB, never a download.
cargo run -p turbospark-cli --bin turbospark-model -- probe owner/name

# What this machine should run, ranked by fit.
cargo run --release -p turbospark-cli --bin turbospark-model -- recommend --context 8192

# Install a curated row into ~/.turbospark/models.
cargo run --release -p turbospark-cli --bin turbospark-model -- pull tinyllama
```

## Crate Gotchas

- Sidecars are fetched and verified (loaded, template rendered) BEFORE any weight byte moves -- the reason `install()` exists rather than a helper that streams first, the way every hand-written `crates/repack/tests/*_network.rs` file does.
- `Store::resolve` prefers an existing directory over an alias; a string that happens to collide with an alias still runs the path on disk.

See `CLAUDE.md` in this directory for the full gotcha list.
