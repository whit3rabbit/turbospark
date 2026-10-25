# turbospark-model-io

Model installation layout, `manifest.json` parsing and architecture validation (`ArchConfig`), packed expert layout metadata (`PackedExpertsLayout`), resident tensor index reader (`ResidentIndex`), memory-mapped resident weight buffer (`ResidentBuffer`), SHA-256 verification (`sha256.rs`), and install receipt validation (`VerifiedInstallReceipt`).

Downstream workspace crates import this package via the `model_io` alias:

```toml
[dependencies]
model_io = { package = "turbospark-model-io", path = "../model-io" }
```

## Purpose & Role

`turbospark-model-io` defines the on-disk file formats, serialization structures, and validation rules for `.gturbo` model directories. It validates model architectural configurations against known baselines, calculates memory footprint requirements, manages memory-mapped resident weight files, and ensures cryptographic install integrity before weights are loaded into the runtime.

## Safety

- Contains `unsafe` code strictly isolated to the documented narrow boundary: memory mapping (`mmap`) in `resident_buffer.rs` and `safetensors.rs`, plus the single `statvfs` free-space call in `disk_space.rs`.
- All other modules enforce safe Rust.

## Key Modules

- `manifest/`: Decodes and validates `manifest.json`, ensuring supported quantization schemes (`validate_quant`) and required tensor presence.
- `arch_config/`: Architecture configuration structures (`ArchConfig`, `ModelFamily`) and field resolution logic.
- `arch_baselines/`: Per-family baseline specifications defining tensor layouts, hyper-parameters, and layer conventions:
  - `gemma.rs`: Gemma 4 family.
  - `qwen.rs`: Qwen 3.5, 3.6, 3.8 GDN hybrids, Qwen 2.5, Qwen 3 dense, Qwen3-MoE, Qwen3-VL, and `qwen4_exp`.
  - `deepseek.rs`: DeepSeek-V4-Flash.
  - `deepseek2.rs`: DeepSeek-V2-Lite latent-attention MoE (the `deepseek2` architecture).
  - `flint.rs`: Recursive Flint (`flint_40b_a2_5b`, `flint_layer_mask`): a 40B-A2.5B MoE with two dense lead layers, 256 routed experts under a top-6 INT8 sigmoid router, one ungated shared expert, learned attention sinks, and a 35 sliding / 14 full NoPE layer mask.
  - `llama.rs`: Llama 3, Mixtral, and related Dense/MoE variants.
  - `gpt_oss.rs`: GPT-OSS-20B (Harmony chat template) architecture.
  - `muse_glimmer.rs`: Muse Glimmer architecture.
  - `spark.rs`: Spark-X2.5-4B architecture.
  - `minimax.rs`: MiniMax-M2 split-GGUF and sigmoid routing architecture.
- `arch_validation.rs`: Structural validation of parsed configurations against family constraints.
- `context_policy.rs`: Context-window bounding (`MaxContext`) and KV cache memory estimation.
- `expert_cache_policy.rs`: MoE expert cache slot sizing algorithms (`ExpertCacheSlots`).
- `load_guard.rs`: Multi-tier memory guardrails (`LoadGuard`, `LoadPolicy`) bounding session memory commitments.
- `disk_space.rs`: Free-disk-space check (`available_bytes`) for the install path via one `statvfs` call.
- `encoder_config.rs`: Configuration schema and validation for BERT/XLM-RoBERTa encoder models.
- `kv_quant.rs`: TurboQuant KV cache quantization policy (`--kv-bits` parameter validation and memory byte savings).
- `packed_experts_layout.rs`: Decodes `packed_experts/layout.json`, supporting multi-dtype companion bias shapes for MoE streaming.
- `ngram_hash.rs` & `ngram_table.rs`: Hashed n-gram PLE hash and disk reader for sparse n-gram embeddings.
- `resident_index.rs`: Reads and parses tensor metadata entries from `model_weights.bin`.
- `resident_buffer.rs`: Zero-copy `mmap` wrapper (`ResidentBuffer`) for memory-mapped model weights.
- `safetensors.rs`: Memory-mapped reader (`SafetensorsFile`) for direct inspection of `.safetensors` headers and slices.
- `steering_set.rs`: Portable per-layer steering vector serialization and loading (`SteeringSet`, `LayerDirection`).
- `speech_family.rs`: `SpeechFamily` enumeration for audio transcription architectures (`whisper`, `qwen3_asr`, `parakeet_tdt`).
- `speech_receipt.rs`: `SpeechInstallReceipt` and schema validation for speech model installs (`speech-receipt.json`).
- `whisper_config.rs`: OpenAI Whisper configuration parser (`WhisperConfig`) and special token resolver (`WhisperSpecialTokens`).
- `vision_sidecar.rs`: Vision tower sidecar installation format and verification (`SidecarRecord`).
- `sha256.rs`: Streaming cryptographic SHA-256 verification of installed files.
- `install_receipt.rs`: Parses and validates `.gturbo` install receipts.
- `trunk_role.rs`: Headless-trunk install marker (`trunk_role.json`, `TrunkRoleRecord`) that refuses a chat open on a headless `.gturbo` trunk.

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-model-io
cargo test -p turbospark-model-io
```

## Tests

- `tests/arch_config.rs`: Validates architecture configuration resolution across all supported model families.
- `tests/manifest.rs`: Tests manifest JSON serialization, parsing, and quantization format checks.
- `tests/packed_experts_layout.rs`: Verifies MoE expert layout parsing, blob byte offsets, and companion bias shapes.
- `tests/resident_index.rs`: Tests binary tensor index parsing and lookup.
- `tests/resident_buffer.rs` & `tests/safetensors.rs`: Tests memory-mapped buffer loading and slice slicing.
- `tests/vision_sidecar.rs`: Verifies vision tower sidecar metadata verification.
- `tests/sha256.rs` & `tests/install_receipt.rs`: Tests cryptographic install receipt generation and validation.

## Crate Gotchas

1. **Resident Memory Pinning**: While clean file-backed `mmap` pages are normally purgeable under macOS virtual memory pressure, wrapping `ResidentBuffer` into Metal buffers via `newBufferWithBytesNoCopy` pins the mapped virtual memory range. Resident weights count directly against the process physical footprint (`phys_footprint`).
2. **Strict Manifest Schema Validation**: Unknown or unrecognized quantization identifiers fail validation immediately during manifest load to avoid launching corrupt or unsupported kernels.
