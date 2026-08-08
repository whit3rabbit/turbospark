# turbospark-model-io

Model installation layout, `manifest.json` parsing and architecture validation (`ArchConfig`), packed expert layout metadata (`PackedExpertsLayout`), resident tensor index reader (`ResidentIndex`), memory-mapped resident weight buffer (`ResidentBuffer`), SHA-256 verification (`sha256.rs`), and install receipt validation (`InstallReceipt`).

Downstream workspace crates import this package via the `model_io` alias:

```toml
[dependencies]
model_io = { package = "turbospark-model-io", path = "../model-io" }
```

## Safety

- Contains `unsafe` code restricted specifically to memory-mapping (`mmap`) operations inside `resident_buffer.rs`.

## Key Modules

- `manifest.rs`: Decodes and validates `manifest.json`.
- `arch_config.rs`: Architecture configuration structs and field resolution.
- `arch_baselines.rs`: Baseline specifications for Gemma 4, Qwen 3.6, and DeepSeek-V4-Flash.
- `arch_validation.rs`: Structural validation of architecture configurations.
- `packed_experts_layout.rs`: Decodes `packed_experts/layout.json` for streamed MoE layouts.
- `resident_index.rs`: Reads and parses tensor metadata entries from `model_weights.bin`.
- `resident_buffer.rs`: Zero-copy `mmap` wrapper (`ResidentBuffer`) for mapped model weights.
- `sha256.rs`: Streaming SHA-256 checksum calculator for installation integrity.
- `install_receipt.rs`: Parses and verifies `.gturbo` install receipts.

## Development & Test Commands

```sh
# Run tests for turbospark-model-io
cargo test -p turbospark-model-io
```

## Crate Gotchas

1. **Resident Memory Pinning**: While clean file-backed `mmap` pages are normally unpinned in host OS memory, wrapping `ResidentBuffer` into Metal buffers via `newBufferWithBytesNoCopy` pins the mapped virtual memory range. Resident weights count directly against process physical memory footprint (`phys_footprint`).
