# MOSS-Music

## Overview
Lyrics transcription for the pinned MOSS-Music 8B Thinking 4-bit checkpoint. The module holds a whisper-family audio encoder (3 stride-2 Conv2d, sinusoidal positions, pre-norm layers), quantized gated-MLP adapter and deepstack mergers, and reuses the shared `qwen3_asr` decoder and tokenizer loader. A single greedy `transcribe` call builds a chat prompt with time-marker digits spliced into the audio span, injects audio embeddings, and strips `<think>` blocks from the output.

## Interface
- `MossMusicProfile`, `MOSS_MUSIC_8B_THINKING_4BIT`: pinned repository and revision [mod.rs:MossMusicProfile] [mod.rs:MOSS_MUSIC_8B_THINKING_4BIT]
- `TRANSCRIPTION_PROMPT`: pinned lyrics prompt [mod.rs:TRANSCRIPTION_PROMPT]
- `MossMusicConfig`, `MossMusicConfig::from_json`: parsed audio, text, and quantization config [mod.rs:MossMusicConfig]
- `compute_features`: 128-band Slaney log-mel frontend, returns `[frames, 128]` rows-major plus frame count [mod.rs:compute_features]
- `MossMusic::load`, `MossMusic::profile`, `MossMusic::config`, `MossMusic::transcribe` [mod.rs:MossMusic]

## Invariants
- Only the untied-head distribution loads: `tie_word_embeddings` must be `false`, else `Unsupported` [mod.rs:MossMusicConfig::from_json]
- `audio_config`, `language_config`, and `quantization` blocks are required, sizes must be positive integers [mod.rs:MossMusicConfig::from_json] [mod.rs:positive]
- Revision is pinned to `b14123a419b6254b92d9e55b8a5b6f7285e05c70` [mod.rs:MOSS_MUSIC_8B_THINKING_4BIT]
- Audio-side linears are never quantized and must have shape `[output, input]` [mod.rs:Linear::load_plain], conv kernels accept MLX `[out,3,3,in]` or `[out,in,3,3]` only [mod.rs:load_plain_conv]
- Adapter and deepstack mergers are quantized with the config scheme and must yield `input*output` values [mod.rs:GatedMlp::load]
- Each digit must encode to exactly one token, else `Unsupported` [mod.rs:digit_token_ids]
- Placeholder count must equal encoder output steps before injection, else `Input` error [mod.rs:MossMusic::transcribe]
- Every deepstack layer index must be reached by the encoder [mod.rs:MossMusicEncoder::forward]
- Time marker every 2 s at 12.5 audio tokens/s, special ids (audio 151654, start 151669, end 151670, EOS 151645) are hardcoded [mod.rs:audio_placeholder_ids] [mod.rs:AUDIO_TOKEN_ID]
- Greedy decoding is capped at 256 tokens and stops before emitting EOS [mod.rs:MAX_GENERATED_TOKENS] [mod.rs:MossMusic::transcribe]
- Empty audio is rejected, no padding mask, one chunk covers the clip [mod.rs:compute_features] [mod.rs:MossMusicEncoder::forward]
- Logits are taken by argmax directly, no softmax here [mod.rs:MossMusic::transcribe]

## Side Effects
- Reads `config.json` and opens `model.safetensors` from the supplied model dir, tokenizer loaded via `load_tokenizer` [mod.rs:MossMusic::load]
- Dequantizes weights to f32 in memory (large resident footprint, the ignored test notes about 40 GiB) [mod.rs:Linear::load_plain] [mod.rs:GatedMlp::load]
- `strip_thinking` leaks one heap string per call when a closed `<think>` block is removed (`Box::leak`) [mod.rs:strip_thinking]
- No env vars, threads, GPU use, or global statics in non-test code, the gated test reads `TURBOSPARK_MOSS_MUSIC_MODEL_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::moss_music`
