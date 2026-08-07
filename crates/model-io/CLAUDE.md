# mrefrust-model-io

Model installation layout, `manifest.json` parsing and architecture validation (`ArchConfig`), packed expert layout metadata (`PackedExpertsLayout`), resident tensor index reader (`ResidentIndex`), memory-mapped resident weight buffer (`ResidentBuffer`), SHA-256 verification (`sha256.rs`), and install receipt validation (`InstallReceipt`).

## Safety

- Contains `unsafe` code specifically restricted to memory-mapping (`mmap`) inside `resident_buffer.rs`.

## Directory & File Structure

```
crates/model-io/
+-- Cargo.toml                  # Crate manifest
+-- src/
|   +-- lib.rs                  # Library root re-exporting model-io API
|   +-- manifest.rs             # Decodes and validates manifest.json
|   +-- arch_config.rs          # ArchConfig struct and field resolution
|   +-- arch_baselines.rs       # Canonical baselines (Gemma 4, Qwen 3.6, DeepSeek-V4)
|   +-- arch_validation.rs      # Structural validation rules for architecture configs
|   +-- packed_experts_layout.rs# Decodes packed_experts/layout.json for streamed MoE
|   +-- resident_index.rs       # Reads tensor index entries from model_weights.bin
|   +-- resident_buffer.rs      # Zero-copy mmap wrapper (ResidentBuffer)
|   +-- sha256.rs               # Streaming SHA-256 checksum verifier
|   +-- install_receipt.rs      # Parses and validates .gturbo install receipts
|   \-- error.rs                # ModelIoError enum definition
\-- tests/
    +-- arch_config.rs          # Architecture config resolution unit tests
    +-- install_receipt.rs      # Install receipt parsing unit tests
    +-- manifest.rs             # Manifest JSON decoding & baseline validation tests
    +-- packed_experts_layout.rs# Layout decoding unit tests
    +-- resident_index.rs       # Binary resident index parsing tests
    \-- sha256.rs               # SHA-256 verification unit tests
```

## Key Modules

- `manifest.rs`: Decodes and validates `manifest.json`.
- `arch_config.rs`: Architecture configuration structs and field resolution.
- `arch_baselines.rs`: Baseline specifications for Gemma 4, Qwen 3.6, and DeepSeek-V4-Flash.
- `arch_validation.rs`: Structural validation of architecture configs.
- `packed_experts_layout.rs`: Decodes `packed_experts/layout.json` for streamed MoE layouts.
- `resident_index.rs`: Reads and parses tensor metadata entries from `model_weights.bin`.
- `resident_buffer.rs`: Zero-copy `mmap` wrapper (`ResidentBuffer`) for mapped model weights.
- `sha256.rs`: Streaming SHA-256 checksum calculator for installation integrity.
- `install_receipt.rs`: Parses and verifies `.gturbo` install receipts.

## Development & Test Commands

```sh
# Run tests for mrefrust-model-io
cargo test -p mrefrust-model-io
```

## Crate Gotchas

1. **Resident Memory Pinning**: While clean file-backed `mmap` pages are normally unpinned in host OS memory, wrapping `ResidentBuffer` into Metal buffers via `newBufferWithBytesNoCopy` pins the mapped virtual memory range. Resident weights count directly against process physical memory footprint (`phys_footprint`).
