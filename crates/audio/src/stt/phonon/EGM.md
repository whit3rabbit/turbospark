# Phonon-1

## Overview
Phonon-1 English speech-to-text, a Qwen3-ASR backbone (frontend, audio tower, decoder, tokenizer, greedy loop reused from `stt::qwen3_asr`) whose decoder linears ship as packed five-value codes. `config.rs` parses the strict `packed_manifest.json`, `packed.rs` unpacks and fuses the codes, `materialize.rs` converts the whole shard to a temporary F32 safetensors file, and `mod.rs` loads that file and runs transcription.

## Interface
- `PhononProfile`, `PHONON_1` (pinned repo and revision) [mod.rs:PhononProfile] [mod.rs:PHONON_1]
- `Phonon` with `load`, `profile`, `config`, `transcribe`, `transcribe_with_options` [mod.rs:Phonon]
- `StageWitness`, `FirstLogits`, `TranscribeStages` (test stage capture) [mod.rs:TranscribeStages]
- `PhononConfig`, `PhononManifest` (re-exported) and `ManifestModule`, `AudioLinear`, `EmbeddingSpec`, `ShardSpec` [config.rs:PhononConfig] [config.rs:PhononManifest]
- Format constants `CONFIG_SCHEMA`, `MANIFEST_FORMAT_PREFIX`, `MANIFEST_STATUS`, `DECODER_CODES_FORMAT`, `SLIM_METADATA_FORMAT`, `HYBRID_FORMAT`, `PLANE_BITS` [config.rs:CONFIG_SCHEMA]
- `packed::GROUP_SIZE`, `packed_bytes_per_row`, `unpack_codes`, `slim_metadata`, `materialize_weight`, `PlaneMetadata` [packed.rs:unpack_codes] [packed.rs:materialize_weight]
- Parent-module-only (`pub(super)`) `materialize::{convert_checkpoint, open_shard, write_safetensors, temporary_path, load_models}` [materialize.rs:convert_checkpoint]

## Invariants
- Manifest format must start with `sttg1a-`, status `PASS`, group size 128, 2-bit planes, only `broadcast-scales-v1` metadata and `ten-base5-per-24bit-v1` codes are accepted [config.rs:PhononManifest::from_json]
- Manifest modules must equal the 7 linears per decoder layer exactly [config.rs:PhononManifest::validate_module_coverage]
- Un-materialized archives (no `packed_manifest.json`, or `.tar.zst`/`.bps` shard) are refused with `Unsupported` [config.rs:PhononConfig::load] [materialize.rs:open_shard]
- `model_type` is `qwen3_asr`, embeddings are tied, audio output dim equals decoder hidden size, token ids nonnegative i32, languages are fixed to English [config.rs:PhononConfig::load]
- Only a single-shard manifest loads [mod.rs:Phonon::load]
- Shard SHA-256 must match the manifest before conversion [materialize.rs:open_shard]
- Every shard tensor must be classified (packed module, hybrid embedding, audio linear, plain norm) or conversion fails [materialize.rs:convert_checkpoint]
- Quint5 unpack requires `in_features % 16 == 0` and exact byte length, fusing requires `in_features % 128 == 0` [packed.rs:unpack_codes] [packed.rs:materialize_weight]
- Fused weight is `base + residual` planes, with biases equal to negated scales [packed.rs:slim_metadata] [packed.rs:materialize_weight]
- Tokenizer ids for `<|audio_pad|>`, `<|audio_start|>`, `<|audio_end|>` must match config [mod.rs:Phonon::open_with_config]
- `transcribe_impl` rejects `max_tokens == 0`, empty or non-finite audio, and unsupported languages [mod.rs:Phonon::transcribe_impl]

## Side Effects
- Reads `config.json`, `packed_manifest.json`, the shard, `tokenizer.json`, hashes the whole shard [config.rs:PhononConfig::load] [materialize.rs:sha256_file]
- Writes a full F32 safetensors file to `std::env::temp_dir()` (`create_new`, pid plus nanosecond nonce), deleted after load on success and failure [materialize.rs:temporary_path] [mod.rs:Phonon::load]
- Memory-maps the shard and temp file via `SafetensorsFile::open` [materialize.rs:load_models]
- CPU-only, no threads, GPU, env vars, or global statics outside tests [mod.rs:Phonon]

## Verification
- Run: `cargo test -p turbospark-audio stt::phonon`
