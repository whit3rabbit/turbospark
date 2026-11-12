# Spark-TTS
## Overview
Port of mlx-audio `tts/models/spark` (Spark-TTS-0.5B): a qwen2 backbone generates `<|bicodec_semantic_i|>` tokens over a task-qualified prompt; ids map back to codes and the BiCodec decodes semantic + global codes to a waveform. Batch 1, string-level prompts, caller-owned tokenization. Voice-clone extraction from audio is not ported (caller supplies codes).
## Interface
- `REFERENCE_COMMIT`, `TASK_TOKEN_MAP`, `GENDER_MAP`, `LEVELS_MAP`, `PITCH_SPEED_LABELS`, `SEMANTIC_TOKENS`, `GLOBAL_TOKENS` [mod.rs:TASK_TOKEN_MAP]
- `semantic_token`, `global_token` [mod.rs:semantic_token]
- `AudioTokenMaps` with `from_lookup`, `from_ids`, `extract_semantic`, `extract_global` [mod.rs:AudioTokenMaps]
- `ModelConfig` with `from_json` [mod.rs:ModelConfig]
- `GenerateParams` with `validate`, `StopReason`, `Generated` [mod.rs:GenerateParams]
- `Model` with `open`, `load`, `process_prompt`, `process_prompt_control`, `generate` [mod.rs:Model]
- `decode_audio` [mod.rs:decode_audio]
## Invariants
- `from_ids` requires the table lengths to equal the caller-supplied `semantic_codes` and `global_codes`; the 8192 and 4096 constants are not enforced there, only by `from_lookup` [mod.rs:AudioTokenMaps::from_ids]
- `validate`: max_tokens > 0, top_k >= 0, temperature >= 0, top_p in [0,1], repetition_penalty >= 0 [mod.rs:GenerateParams::validate]
- `generate` validates params first and rejects an empty prompt [mod.rs:Model::generate]
- Prefill feeds all prompt tokens but the last; the last token starts the first sampled step [mod.rs:Model::generate]
- Unknown gender/level names are an error [mod.rs:map_value]
## Side Effects
- Reads `config.json` and safetensors from the checkpoint dir [mod.rs:Model::open]
- RNG: seeded `rng::KeySequence` for sampling [mod.rs:Model::generate]
- Env vars, threads, logging: None
## Verification
- Run: `cargo test -p turbospark-audio tts::spark`
