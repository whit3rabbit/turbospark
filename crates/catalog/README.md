# turbospark-catalog

Curated model catalog, header-only Hugging Face repository probe, hardware-aware model recommendation, multi-file GGUF source resolution, and stream-install driver that turns remote checkpoints into `.gturbo` installations. Backs the `turbospark-model` binary in `crates/cli`.

Detailed documentation: [`docs/MODELS.md`](../../docs/MODELS.md).

Downstream workspace crates import this package via the `catalog` alias:

```toml
[dependencies]
catalog = { package = "turbospark-catalog", path = "../catalog" }
```

## Purpose & Role

`turbospark-catalog` provides discovery, inspection, and installation of model checkpoints (text, speech, audio, image, OCR, and embedding). It allows users to browse a curated list of tested models, evaluate whether an arbitrary Hugging Face repository can run on their Apple Silicon hardware in seconds by reading only file headers (KB scale), and stream-download and repack model weights layer by layer.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Network interactions use safe HTTP abstractions (`reqwest`).

## Key Modules

- `entry.rs`: `CatalogEntry` representing one curated model row (`EntryKind`, `Source`, `Sidecars`, `SourceKind`, `Status`, `MtpSource`, `Measured`). Enforces structural invariants at load time.
- `catalog.rs`: `Catalog` managing the embedded table (`models.json`, via `include_str!`) merged with an optional `$TURBOSPARK_HOME/models.json` local override table.
- `hf.rs`: Lightweight Hugging Face API client handling file listings, small config JSON fetches, and content-length queries without downloading model weights.
- `auth.rs`: Hugging Face authentication token resolution (CLI argument, environment variable, store file, cache file) and validation against `whoami-v2`.
- `hub.rs`: `HubClient` search and trending over Hugging Face, composed with the bundled catalog, plus GGUF variant enumeration and pinned-source planning.
- `hub_transfer.rs`: All-or-nothing transport for modality-owned pinned source sets.
- `hub_validation.rs`: Validation of untrusted Hugging Face catalog metadata (owns no HTTP of its own).
- `probe/`: Fast, header-only checkpoint inspector evaluating architecture compatibility, quantization formats, and memory headroom:
  - `mod.rs`: Dispatcher (GGUF or safetensors) evaluating tokenizer sidecars, chat templates, and context limits.
  - `types.rs`: Probe verdict and result types.
  - `gguf.rs`: GGUF header parser and block type compatibility checker.
  - `safetensors.rs`: Safetensors config parser and quantization-shape validator, with a per-family config parse (flint included).
- `recommend/`: Hardware-tailored model ranking and recommendations:
  - `fit.rs`: Memory requirement calculation (reusing `model_io` context and expert cache sizing policies).
  - `rank.rs`: Candidate tiering and ordering based on system memory and hardware configuration.
  - `discover.rs`: Surveys trending Hugging Face repositories through `probe/`.
- `vision.rs`: Matches and resolves vision tower sidecars for multimodal models.
- `image.rs`: Curated Diffusers image sources (`image_models.json`) and their download pipeline.
- `speech.rs`: Bundled speech model catalog (`speech_models.json`), `SpeechCatalogEntry`, speech directory probing (`SpeechProbeReport`), and install verification.
- `audio_catalog.rs`: Task-aware pinned audio profiles (`audio_assets.json`, `audio_families.json`) and verified managed installs for speech models.
- `music.rs`: Curated, pinned MiniMax Music 3 profiles (`music_models.json`) and their install path.
- `embedding.rs`: Pinned embedding-model profiles (`embedding_models.json`) and their install path.
- `ocr.rs`: Pinned OCR-model profiles (`ocr_models.json`) and their install path.
- `pinned_dir.rs`: Install machinery shared by pinned directory profiles (embedding and OCR): SHA-256 pinned files, verified transfer, rename publication, and receipt.
- `drama3_install.rs`: Drama 3 audio installs by conversion from the upstream MLX checkpoint rather than file copy.
- `gguf_source.rs`: Handles split multi-file GGUF shard discovery and URL construction.
- `quant.rs`: Filename-derived GGUF quantization labels and logical file-set grouping.
- `install.rs`: Walk driver orchestrating probe validation, tokenizer sidecar downloads, weight streaming, install receipt creation, and stale sidecar pruning.
- `stream.rs`: GGUF and MLX streaming and shard-writing helpers driven by `install.rs`, including shard discovery for repos that ship no `model.safetensors.index.json` and the flint safetensors intake (`write_flint_install_streamed`).
- `store.rs`: `Store` managing `~/.turbospark` layout, `installed.json`, and alias/path resolution (`resolve_model_arg`).

## Development & Test Commands

```sh
# Run offline catalog, store, and probe tests
cargo test -p turbospark-catalog

# Run network rot-guard test (verifies all curated catalog URLs still exist; ~26s, downloads no weights)
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture

# Opt-in rot guards for the audio, music, and pinned embedding/OCR profile pins
cargo test -p turbospark-catalog --test audio_catalog_network --release -- --ignored --nocapture
cargo test -p turbospark-catalog --test music_catalog_network --release -- --ignored --nocapture
cargo test -p turbospark-catalog --test pinned_dir_catalog_network --release -- --ignored --nocapture
```

## Tests

- `tests/catalog.rs`: Validates catalog JSON parsing, the embedded table's structural invariants, and entry validation.
- `tests/probe.rs`: Tests header-only probe logic against synthetic GGUF and Safetensors headers.
- `tests/store.rs`: Verifies local model installation bookkeeping, path lookups, and deletion.
- `tests/hub_*.rs`: Offline fixture tests for hub search composition, GGUF variants, resumable transfers, metadata validation, and source identity.
- `tests/quant_taxonomy.rs`: Quantization label taxonomy and file-set grouping.
- `tests/catalog_network.rs`: Opt-in rot guard verifying that remote Hugging Face repos and filenames in `models.json` remain active.
- `tests/audio_catalog_network.rs`, `tests/music_catalog_network.rs`, `tests/pinned_dir_catalog_network.rs`: Opt-in rot guards verifying the pinned audio, music, and embedding/OCR profile sources (LFS metadata or byte hashes; no checkpoint payloads).

## Usage

Driven from the `turbospark-model` CLI binary:

```sh
# List the curated catalog
cargo run -p turbospark-cli --bin turbospark-model -- list

# List bundled speech models (STT installs)
cargo run -p turbospark-cli --bin turbospark-model -- list-speech

# Header-only probe of a remote Hugging Face repository (reads KB, never downloads weights)
cargo run -p turbospark-cli --bin turbospark-model -- probe Qwen/Qwen3-30B-A3B-GGUF

# Recommend models that fit local hardware RAM
cargo run --release -p turbospark-cli --bin turbospark-model -- recommend --context 8192

# Install a curated text model into ~/.turbospark/models/text
cargo run --release -p turbospark-cli --bin turbospark-model -- pull tinyllama
```

## Crate Gotchas

1. **Header-Only Probe**: `probe` reads only the first few kilobytes of remote checkpoint files to extract metadata, tensor headers, and block quantization types. It exits with a non-zero code when a gate refuses the checkpoint, for example a block type that has neither a Metal kernel nor a transcode path.
2. **Layer-by-Layer Streaming and Resume**: `pull` streams weights into repacked files without requiring a complete source checkpoint first. For immutable revisions it also retains SHA-256 checked range chunks until the install verifies, so temporary disk can approach download size plus install size. Failed and cancelled pulls reuse those chunks; floating revisions do not.
3. **Mandatory Sidecar Verification**: Installation requires the sidecars each row needs (`tokenizer.json` for model rows, `preprocessor_config.json` for vision-tower rows; `tokenizer_config.json` is optional). Any stale temporary sidecar files are explicitly pruned before writing the install receipt.
