# Sesame CSM TTS
## Overview
Port of mlx-audio `tts/models/sesame` (CSM-1B / MisoTTS-style). A llama backbone over summed per-codebook audio embeddings plus text embedding predicts codebook 0; a smaller depth decoder predicts codebooks 1..K-1 via the `audio_head` tensor. Frames decode through the 24 kHz Mimi codec. Batch 1, non-streaming, tokenization is the caller's job.
## Interface
- `MIMI_REPO`, `TOKENIZER_REPO`, `DEFAULT_SAMPLE_RATE`, `MS_PER_FRAME`, `MAX_SEQ_LEN`, `CSM_1B_GH_WATERMARK` [mod.rs:MIMI_REPO]
- `LlamaFlavor`, `llama_flavor` (llama-1B/8B/100M/300M presets) [mod.rs:llama_flavor]
- `ModelConfig` with `from_json` [mod.rs:ModelConfig]
- `GenerateParams`, `StopReason`, `Generated` [mod.rs:GenerateParams]
- `Prompt` with `concat` [mod.rs:Prompt]
- `text_prompt_string` builds `"[{speaker}]{text}"` [mod.rs:text_prompt_string]
- `Model` with `open`, `load`, `generate`, `generate_frame`, `generate_frames`, `tokenize_text_segment`, `tokenize_audio`, `tokenize_segment`, `reset_caches` [mod.rs:Model]
- `frames_to_codes`, `encode_audio`, `watermark` [mod.rs:frames_to_codes]
## Invariants
- Unknown flavor strings are a BadConfig error [mod.rs:llama_flavor]
- `audio_num_codebooks` must be at least 1 [mod.rs:ModelConfig::from_json]
- Prompt sections must share one width; empty section list is an error [mod.rs:Prompt::concat]
- Prompt width must equal `audio_num_codebooks + 1`; empty prompt rejected [mod.rs:Model::generate_frames]
- Prompt rows must stay below `MAX_SEQ_LEN - max_audio_frames` [mod.rs:Model::generate_frames]
- Caches are reset at the start of every generate [mod.rs:Model::generate_frames]
- Loop stops on an all-zero frame or the frame budget [mod.rs:Model::generate_frames]
- `watermark` always returns Unsupported (silentcipher not ported); output is unwatermarked [mod.rs:watermark]
- `encode_audio` rejects an empty waveform [mod.rs:encode_audio]
## Side Effects
- Reads `config.json` and safetensors weights from the checkpoint dir [mod.rs:Model::open]
- RNG: seeded `rng::KeySequence` drives categorical draws only [mod.rs:Model::generate_frames]
- Env vars, threads, logging: None
## Verification
- Run: `cargo test -p turbospark-audio tts::sesame`
