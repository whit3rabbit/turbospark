# GLM-ASR-Nano STT

## Overview
Offline ASR port of mlx-audio GLM-ASR-Nano: a 32-layer Whisper-style audio encoder (two conv layers, rotary attention, GELU FFN), a 4x frame-merge MLP adapter, and a Llama decoder reused from the Qwen3-ASR module. The audio features replace placeholder rows inside a chat-style prompt, then greedy decoding runs until a stop token. All math runs on CPU in f32 after dequantizing 4-bit weights.

## Interface
- `GlmAsr` with `load`, `profile`, `transcribe`, `transcribe_with_max_tokens` [mod.rs:GlmAsr]
- `GlmAsrProfile` and pinned `GLM_ASR_NANO_2512` [mod.rs:GLM_ASR_NANO_2512]
- No other public items; `Config`, `AudioEncoder`, `WhisperEncoder`, `log_mel` are private [mod.rs:Config] [mod.rs:log_mel]

## Invariants
- Pinned checkpoint is `mlx-community/GLM-ASR-Nano-2512-4bit` at revision `35553fa5bebfcc3ece3ce7d47b98827cb0ac9eef` [mod.rs:GLM_ASR_NANO_2512]
- Config requires model_type glmasr, adapter_type mlp, mlp_adapter_act gelu, use_rope true [mod.rs:Config::from_json]
- Quantization must be affine, 4-bit, group 64; merge_factor 4 and max_whisper_length 1500 [mod.rs:Config::from_json]
- The Llama text config is hardcoded (vocab 59264, hidden 2048, 28 layers, 16 heads, 4 KV heads, head_dim 128) because the pinned config.json omits it [mod.rs:Config::from_json]
- Loaded decoder hidden size and vocab must match the hardcoded text config [mod.rs:GlmAsr::load]
- Tokenizer must contain `<|endoftext|>`, `<|user|>`, `<|observation|>` (stop ids), `<|begin_of_audio|>`, `<|end_of_audio|>`, `<|assistant|>` [mod.rs:GlmAsr::load]
- Audio must be non-empty, finite, at most 30 s (480000 samples); inputs under 1 s are zero-padded to 1 s [mod.rs:encode_audio]
- `max_tokens` must be positive; default is 128 [mod.rs:transcribe_with_max_tokens] [mod.rs:DEFAULT_MAX_TOKENS]
- Encoder row count must be within `[merge_factor, 1500]`; merged rows are capped at 1500 / 4 [mod.rs:encode_audio]
- Frontend: 400-point FFT, hop 160, 128 Slaney mels, last centered STFT frame dropped, then global peak-8 log clamp and `(x+4)/4` scaling [mod.rs:log_mel]
- Audio rows are written into the embedded prompt after the `<|user|>\n<|begin_of_audio|>` prefix; suffix is fixed text ending in `<|assistant|>\n` [mod.rs:decode_audio]
- Conv1d weights convert from MLX `[out, kernel, in]` to `[out, in, kernel]` at load [mod.rs:Conv1d::load]
- Audio attention applies interleaved RoPE to the first 32 dims only, base 10000, no mask [mod.rs:apply_rope_traditional] [mod.rs:WhisperAttention::forward]
- Token ids must fit i32 [mod.rs:decode_audio]

## Side Effects
- Reads `config.json`, `tokenizer.json`, and opens `model.safetensors` in the model directory [mod.rs:GlmAsr::load]
- Uses a cached mel projector from the shared mel module (process-wide cache) [mod.rs:log_mel]
- Dequantizes all linears to f32 at load. No env vars, threads, or GPU use in non-test code. Ignored test reads `TURBOSPARK_GLMASR_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::glmasr`
- Checkpoint parity (ignored): `TURBOSPARK_GLMASR_DIR=<snapshot> cargo test -p turbospark-audio stt::glmasr -- --ignored`
