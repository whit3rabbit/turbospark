# Mega-ASR

## Overview
Mega-ASR is a routed robust transcription family built on the shared Qwen3-ASR audio tower, text decoder, tokenizer loader, and prompt builder [mod.rs:MegaAsr]. This module adds strict config and profile detection [config.rs:MegaConfig], an audio-quality router that classifies a clip as clean or degraded [router.rs:AudioQualityRouter], and a LoRA adapter table [lora.rs:LoraAdapter]. Only the pre-merged always-on-robust profile is pinned and runnable end to end.

## Interface
- `MegaAsr` with `load`, `profile`, `config`, `route`, `transcribe`, `transcribe_with_options` [mod.rs:MegaAsr]
- `MegaAsrProfile`, `MEGA_ASR_8BIT` (pinned repo and revision) [mod.rs:MEGA_ASR_8BIT]
- `StageWitness`, `FirstLogits`, `TranscribeStages` (parity capture types) [mod.rs:TranscribeStages]
- `MegaConfig`, `ProfileKind`, `RouterSettings`, `DEFAULT_ROUTER_WEIGHTS`, `DEFAULT_LORA_WEIGHTS` [config.rs:MegaConfig]
- `AudioQualityRouter` with `from_tensors`, `from_safetensors`, `logmel`, `forward`, `route`, `d_model` [router.rs:AudioQualityRouter]
- `RouterStages`, `RouteDecision` [router.rs:RouterStages]
- `LoraAdapter` with `load_factors`, `load_adapter`, `module`, `module_names`, `module_count`, `merge_into`; `LoraModule::materialize_delta` [lora.rs:LoraAdapter]
- Re-exports of the config, lora, and router types at module root [mod.rs:pub use]

## Invariants
- Pinned checkpoint is `mlx-community/Mega-ASR-8bit` at revision `b9c3c7020f94944205df7f7b5d5d1ce96678d74f` [mod.rs:MEGA_ASR_8BIT]
- Profile rule: `model_type "mega_asr"` is Dynamic; `"qwen3_asr"` is accepted only when `merge_metadata.json` exists, else `Unsupported`; any other type is `BadConfig` [config.rs:MegaConfig::detect_profile]
- The PreMergedRobust profile must not declare `router_config`, `router_weights`, or `lora_weights` [config.rs:MegaConfig::from_json]
- Token embeddings must be tied, otherwise `Unsupported` [config.rs:MegaConfig::from_json]
- Router `nhead` must be 4 and `n_mels` 80, `d_model` divisible by heads [config.rs:RouterSettings::from_json]
- Router geometry is inferred from tensor shapes and a disagreeing `router_config` is refused [router.rs:AudioQualityRouter::from_tensors]
- Router frontend steps beyond the positional table length are an `Input` error [router.rs:AudioQualityRouter::forward]
- `use_lora` is `degraded_prob >= 0.5` [router.rs:AudioQualityRouter::route]
- Dynamic profile fails to load without router weights on disk (no untrained-router fallback) [mod.rs:MegaAsr::load]
- Tokenizer ids for `<|audio_pad|>`, `<|audio_start|>`, `<|audio_end|>` must match config.json [mod.rs:MegaAsr::load]
- `route` on a distribution with no router returns `Unsupported` [mod.rs:MegaAsr::route]
- A degraded route with a loaded LoRA table is refused: runtime LoRA over the reused backbone is not wired [mod.rs:MegaAsr::transcribe_impl]
- Input must be non-empty and finite, `max_tokens > 0`, and language must match a supported language case-insensitively [mod.rs:MegaAsr::transcribe_impl]
- LoRA scaling is 1.0 for the factors format and `alpha / rank` for the adapter format; rank mismatches between config, A, and B are refused [lora.rs:LoraAdapter::load_factors] [lora.rs:LoraAdapter::load_adapter]
- `merge_into` requires the weight length to equal the `[output, input]` delta length [lora.rs:LoraAdapter::merge_into]

## Side Effects
- Reads `config.json`, `model.safetensors`, tokenizer files, optional `merge_metadata.json`, and router/LoRA safetensors from the model directory [mod.rs:MegaAsr::load] [mod.rs:read_json]
- Opens safetensors via `SafetensorsFile::open` and eagerly materializes router and LoRA tensors as f32 vectors [router.rs:AudioQualityRouter::from_safetensors] [lora.rs:collect_factors]
- Reads `adapter_config.json` and `adapter_model.safetensors` for the adapter format [lora.rs:LoraAdapter::load_adapter]
- Checkpoint-gated tests read env var `TURBOSPARK_MEGA_ASR_MODEL_DIR` (test code only) [mod.rs:tests]
- No threads, global statics, or GPU use in this module (CPU f32 paths)

## Verification
- Run: `cargo test -p turbospark-audio stt::mega_asr`
