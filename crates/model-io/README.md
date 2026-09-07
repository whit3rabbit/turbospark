# turbospark-model-io

Model installation layout, `manifest.json` parsing and architecture validation (`ArchConfig`), packed expert layout metadata (`PackedExpertsLayout`), resident tensor index reader (`ResidentIndex`), memory-mapped resident weight buffer (`ResidentBuffer`), SHA-256 verification (`sha256.rs`), and install receipt validation (`VerifiedInstallReceipt`).

Downstream workspace crates import this package via the `model_io` alias:

```toml
[dependencies]
model_io = { package = "turbospark-model-io", path = "../model-io" }
```

## Safety

- Contains `unsafe` code restricted to memory-mapping (`mmap`) operations: `resident_buffer.rs` and `safetensors.rs`.

## Key Modules

- `manifest/`: Decodes and validates `manifest.json`, including quantization-scheme validation (`validate_quant`).
- `arch_config/`: Architecture configuration structs (`ArchConfig`, `ModelFamily`) and field resolution.
- `arch_baselines/`: Baseline specifications, one per `ModelFamily` (Gemma 4, Qwen 3.5/3.6/3.8, DeepSeek-V4-Flash, Llama/Mixtral, Qwen3-MoE, gpt-oss, Muse Glimmer, qwen4_exp).
- `arch_validation.rs`: Structural validation of architecture configurations.
- `context_policy.rs`: Context-window sizing (`MaxContext`, KV-cache byte estimation).
- `expert_cache_policy.rs`: Expert cache slot sizing (`ExpertCacheSlots`).
- `load_guard.rs`: Memory guardrail tiers (`LoadGuard`, `LoadPolicy`) bounding what a session may commit.
- `encoder_config.rs`: Config schema for BERT/XLM-RoBERTa encoder models.
- `kv_quant.rs`: TurboQuant KV-cache quantization policy (`--kv-bits` eligibility and row byte costs).
- `packed_experts_layout.rs`: Decodes `packed_experts/layout.json` for streamed MoE layouts.
- `ngram_hash.rs` / `ngram_table.rs`: `qwen4_exp`'s hashed n-gram PLE hash and on-disk table reader.
- `resident_index.rs`: Reads and parses tensor metadata entries from `model_weights.bin`.
- `resident_buffer.rs`: Zero-copy `mmap` wrapper (`ResidentBuffer`) for mapped model weights.
- `safetensors.rs`: Memory-mapped reader (`SafetensorsFile`) for local `.safetensors` files.
- `steering_set.rs`: Portable per-layer steering vectors (`SteeringSet`, `LayerDirection`).
- `vision_sidecar.rs`: The vision-tower sidecar install format (`SidecarRecord`).
- `sha256.rs`: Streaming SHA-256 checksum calculator for installation integrity.
- `install_receipt.rs`: Parses and verifies `.gturbo` install receipts.

## Development & Test Commands

```sh
# Run tests for turbospark-model-io
cargo test -p turbospark-model-io
```

## Crate Gotchas

1. **Resident Memory Pinning**: While clean file-backed `mmap` pages are normally unpinned in host OS memory, wrapping `ResidentBuffer` into Metal buffers via `newBufferWithBytesNoCopy` pins the mapped virtual memory range. Resident weights count directly against process physical memory footprint (`phys_footprint`).
