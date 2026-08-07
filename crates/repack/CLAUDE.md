# mrefrust-repack

Safetensors header parsing, ranged HTTP/in-memory downloads (`RangeSource`), INT4/INT8 quantization repack, `.gturbo` directory installation assembly (`gturbo_writer.rs`), synthetic install generation (`synthetic_model.rs`, `synthetic_real.rs`), Hugging Face Llama repacker (`hf_checkpoint.rs`), and Gemma 4 checkpoint repacker (`gemma4_checkpoint.rs`).

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/repack/
+-- Cargo.toml                      # Crate manifest
+-- src/
|   +-- lib.rs                      # Library root
|   +-- safetensors_header.rs       # Pure safetensors JSON header parser
|   +-- ranged_download.rs          # RangeSource trait and ranged HTTP downloader
|   +-- repack.rs                   # Quantization repack algorithms (FP32/BF16 to INT4/INT8)
|   +-- gturbo_writer.rs            # Writes .gturbo directory tree and manifest/layout JSON
|   +-- resident_writer.rs          # Writes model_weights.bin resident tensor blob and index
|   +-- synthetic_model.rs          # Synthetic model generator (build_synthetic_gemma4_install)
|   +-- synthetic_real.rs           # Real-named synthetic generator (build_synthetic_gemma4_real_install)
|   +-- gemma4_checkpoint.rs        # Gemma 4 mlx-community checkpoint converter & streamer
|   +-- hf_checkpoint.rs            # Hugging Face Llama checkpoint converter
|   +-- install_verifier.rs         # Validates repacked install directory structure & receipt
|   \-- manifest_peek.rs            # Pre-fetches remote checkpoint manifests without full download
\-- tests/
    +-- gemma4_checkpoint.rs        # Gemma 4 repack pipeline unit tests
    +-- gemma4_checkpoint_network.rs# Real Gemma 4 checkpoint download integration test (ignored)
    +-- gturbo_writer.rs            # .gturbo layout writer unit tests
    +-- hf_checkpoint.rs            # HF Llama converter unit tests
    +-- hf_checkpoint_network.rs    # Real HF checkpoint download integration test (ignored)
    +-- install_verifier.rs         # Install verifier unit tests
    +-- repack.rs                   # Quantization repack unit tests
    +-- safetensors_header.rs       # Safetensors header parsing unit tests
    \-- synthetic_model.rs          # Synthetic install builder unit tests
```

## Key Modules

- `safetensors_header.rs`: Pure safetensors header parser.
- `ranged_download.rs`: Ranged HTTP download engine (`RangeSource`).
- `repack.rs`: Quantization repack pipelines converting FP32/BF16 weights to INT4/INT8.
- `gturbo_writer.rs`: `.gturbo` directory layout and binary file writer.
- `resident_writer.rs`: `model_weights.bin` resident tensor index writer.
- `synthetic_model.rs`: Synthetic dense and MoE model generators (`build_synthetic_gemma4_install`).
- `synthetic_real.rs`: Synthetic Gemma 4 real-named model generator (`build_synthetic_gemma4_real_install`).
- `gemma4_checkpoint.rs`: Gemma 4 checkpoint repacker and streamed expert pipeline builder.
- `hf_checkpoint.rs`: Hugging Face checkpoint layout converter.
- `install_verifier.rs`: Verifies checksums and directory layout of repacked installs.
- `manifest_peek.rs`: Inspects remote checkpoint manifests without full downloads.

## Development & Test Commands

```sh
# Run fast unit tests for mrefrust-repack
cargo test -p mrefrust-repack

# Run network checkpoint integration tests (ignored by default, downloads large files)
cargo test -p mrefrust-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
cargo test -p mrefrust-repack --test hf_checkpoint_network --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Synthetic Model Weight Meaning**: Synthetic models built by `build_synthetic_gemma4_install` use deterministic pseudo-random numbers rather than trained weights. Generated text on synthetic installs is structurally valid but semantically gibberish (and short generations may yield empty strings).
2. **Synthetic Model Uniform Routing**: Synthetic MoE routers feature near-uniform routing logits. Expert slot permutation bugs cannot be caught by testing synthetic models alone; routing assignments must be validated structurally.
