# Voxtral Mini 3B STT

## Overview
Port of mlx-audio Voxtral Mini 3B speech understanding: a Whisper-large-v3 style audio tower (two k3 convs, learned positions, GELU transformer), a 4-frame merge plus two-layer projector, and a Mistral text decoder reused from the Qwen3-ASR module. The caller supplies the fixed prompt token ids; audio embeddings overwrite the audio-token positions, then greedy decoding runs on CPU in f32.

## Interface
- `Voxtral` with `load`, `profile`, `config`, `transcribe(samples, prompt_ids)` [mod.rs:Voxtral]
- `VoxtralProfile` and pinned `VOXTRAL_MINI_3B_BF16` [mod.rs:VOXTRAL_MINI_3B_BF16]
- `VoxtralConfig` with `from_json` [mod.rs:VoxtralConfig]
- `compute_features(samples)` returning `(mel, frames)` [mod.rs:compute_features]

## Invariants
- Pinned checkpoint is `mlx-community/Voxtral-Mini-3B-2507-bf16` at revision `ab69f21c8e2c52b05335bf282898e141c8c42477` [mod.rs:VOXTRAL_MINI_3B_BF16]
- Config requires `audio_config` and `text_config` with positive dims, finite positive rms_norm_eps and rope_theta, and an integer `audio_token_id`; qk_norm and tied embeddings are fixed false, rotary_dim equals head_dim [mod.rs:VoxtralConfig::from_json]
- Scheme is bf16 passthrough (`bits: 16, group_size: 1`) [mod.rs:VoxtralConfig::from_json]
- Weights load from two fixed shards `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`; the audio tower must live in shard one [mod.rs:Voxtral::load]
- Frontend zero-pads to 480000 samples and refuses input over 30 s with `SpeechError::Input` (it used to truncate silently), so frames are always exactly 3000; callers must chunk longer audio [mod.rs:compute_features] [mod.rs:TARGET_SAMPLES]
- Frontend: 400-point FFT, hop 160, Hann, Slaney 128 mels, trailing STFT frame dropped, peak-8 floor, `(x+4)/4` [mod.rs:compute_features]
- `audio_tower.embed_positions.weight` must be `[1500, d_model]` (TOWER_POSITION_ROWS); forward errors if it is shorter than the conv output length [mod.rs:AudioTower::validate_positions] [mod.rs:AudioTower::forward]
- Conv weights accept `[out, 3, in]` (MLX, transposed at load) or `[out, in, 3]`; other shapes error [mod.rs:AudioTower::load_conv]
- Linear loads check `[output, input]` shape; projectors must carry no bias [mod.rs:Linear::load] [mod.rs:Voxtral::load]
- Tower output steps must divide by 4 (FRAMES_PER_MERGE) and the prompt must contain exactly `groups` audio-token ids, else `SpeechError::Input` [mod.rs:Voxtral::transcribe]
- Decoding is greedy, capped at 128 tokens, stops on token id <= 2; no max_tokens parameter [mod.rs:Voxtral::transcribe]
- `Voxtral::transcribe` refuses empty and non-finite audio with `SpeechError::Input` [mod.rs:validate_samples]; the public `compute_features` itself does not check (callers using it directly must)
- Audio tower attention applies no mask and no RoPE; query is pre-scaled [mod.rs:AudioAttention::forward]

## Side Effects
- Reads `config.json`, opens both safetensors shards, and loads the Tekken tokenizer from the model directory [mod.rs:Voxtral::load]
- Dequantizes or converts all weights to f32 Vecs at load (large resident memory) [mod.rs:Linear::load]
- Builds a Slaney mel filterbank on every `compute_features` call (no cache) [mod.rs:compute_features]
- No env vars, threads, GPU use, or statics in non-test code. Ignored test reads `TURBOSPARK_VOXTRAL_MODEL_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::voxtral`
- Checkpoint parity (ignored): `TURBOSPARK_VOXTRAL_MODEL_DIR=<snapshot> cargo test -p turbospark-audio stt::voxtral -- --ignored`
