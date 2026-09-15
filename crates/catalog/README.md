# turbospark-catalog

Curated model catalog, header-only Hugging Face repository probe, hardware-aware model recommendation, multi-file GGUF source resolution, and stream-install driver that turns remote checkpoints into `.gturbo` installations. Backs the `turbospark-model` binary in `crates/cli`.

Detailed documentation: [`docs/MODELS.md`](../../docs/MODELS.md).

Downstream workspace crates import this package via the `catalog` alias:

```toml
[dependencies]
catalog = { package = "turbospark-catalog", path = "../catalog" }
```

## Purpose & Role

`turbospark-catalog` provides discovery, inspection, and installation of LLM checkpoints. It allows users to browse a curated list of tested models, evaluate whether an arbitrary Hugging Face repository can run on their Apple Silicon hardware in seconds by reading only file headers (KB scale), and stream-download and repack model weights layer by layer.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Network interactions use safe HTTP abstractions (`reqwest`).

## Key Modules

- `entry.rs`: `CatalogEntry` representing one curated model row (`Source`, `Sidecars`, `SourceKind`, `Status`, `MtpSource`, `Measured`). Enforces structural invariants at load time.
- `catalog.rs`: `Catalog` managing the embedded table (`models.json`, via `include_str!`) merged with an optional `$TURBOSPARK_HOME/models.json` local override table.
- `hf.rs`: Lightweight Hugging Face API client handling file listings, small config JSON fetches, and content-length queries without downloading model weights.
- `auth.rs`: Hugging Face authentication token resolution (CLI argument, environment variable, keychain store, cache file) and validation against `whoami-v2`.
- `probe/`: Fast, header-only checkpoint inspector evaluating architecture compatibility, quantization formats, and memory headroom:
  - `gguf.rs`: GGUF header parser and block type compatibility checker.
  - `safetensors.rs`: Safetensors JSON header parser and tensor shape validator.
  - `mod.rs`: Dispatcher evaluating tokenizer sidecars, chat templates, and context limits.
- `recommend/`: Hardware-tailored model ranking and recommendations:
  - `fit.rs`: Memory requirement calculation (reusing `model_io` context and expert cache sizing policies).
  - `rank.rs`: Candidate tiering and ordering based on system memory and hardware configuration.
  - `discover.rs`: Surveys trending Hugging Face repositories through `probe/`.
- `vision.rs`: Matches and resolves vision tower sidecars for multimodal models.
- `gguf_source.rs`: Handles split multi-file GGUF shard discovery and URL construction.
- `install.rs`: Walk driver orchestrating probe validation, tokenizer sidecar downloads, weight streaming, install receipt creation, and stale sidecar pruning.
- `stream.rs`: GGUF and MLX streaming and shard-writing helpers driven by `install.rs`.
- `store.rs`: `Store` managing `~/.turbospark` layout, `installed.json`, and alias/path resolution (`resolve_model_arg`).

## Development & Test Commands

```sh
# Run offline catalog, store, and probe tests
cargo test -p turbospark-catalog

# Run network rot-guard test (verifies all curated catalog URLs still exist; ~26s, downloads no weights)
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture
```

## Tests

- `tests/catalog.rs`: Validates catalog JSON parsing, alias uniqueness, and entry validation.
- `tests/probe.rs`: Tests header-only probe logic against synthetic GGUF and Safetensors headers.
- `tests/store.rs`: Verifies local model installation bookkeeping, path lookups, and deletion.
- `tests/catalog_network.rs`: Opt-in rot guard verifying that remote Hugging Face repos and filenames in `models.json` remain active.

## Usage

Driven from the `turbospark-model` CLI binary:

```sh
# List the curated catalog
cargo run -p turbospark-cli --bin turbospark-model -- list

# Header-only probe of a remote Hugging Face repository (reads KB, never downloads weights)
cargo run -p turbospark-cli --bin turbospark-model -- probe Qwen/Qwen3-30B-A3B-GGUF

# Recommend models that fit local hardware RAM
cargo run --release -p turbospark-cli --bin turbospark-model -- recommend --context 8192

# Install a curated model into ~/.turbospark/models
cargo run --release -p turbospark-cli --bin turbospark-model -- pull tinyllama
```

## Crate Gotchas

1. **Header-Only Probe**: `probe` reads only the first few kilobytes of remote checkpoint files to extract metadata, tensor headers, and block quantization types. It exits with a non-zero code if any layer lacks an optimized Metal kernel.
2. **Layer-by-Layer Streaming**: `pull` streams weights in chunks directly into repacked files. A 30 GB source repository never lands on disk in its original format, drastically reducing scratch disk requirements.
3. **Mandatory Sidecar Verification**: Installation requires required tokenizer sidecars (`tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json` for vision models). Any stale temporary sidecar files are explicitly pruned before writing the install receipt.
