# VibeVoice-ASR

## Overview
Offline single-window speech-to-text for the pinned VibeVoice-ASR-Streaming 1.5B checkpoint. Two causal convolutional tokenizer encoders (acoustic and semantic) feed per-side connectors whose summed output is spliced into a Qwen2 chat prompt, then a CPU f32 greedy decode runs until an EOS token. Weights are read from raw safetensors tensors with shape checks rather than an MLX module tree. Streaming, diarization controls, and hotwords are not implemented (absent from the API, not refused at runtime).

## Interface
- `VibeVoiceAsr` with `load`, `profile`, `config`, `resample_input`, `encode_speech`, `build_prompt`, `transcribe` [mod.rs:VibeVoiceAsr]
- `VibeVoiceAsrProfile`, `VIBEVOICE_ASR_STREAMING_1_5B` [mod.rs:VIBEVOICE_ASR_STREAMING_1_5B]
- `VibeVoiceAsrConfig` (`from_json`, `from_preprocessor`), `TokenizerSideConfig`, `DecoderConfig` [mod.rs:VibeVoiceAsrConfig]
- `SpeechEncoding`, `StepRecord`, `GreedyRun` [mod.rs:SpeechEncoding]
- `SpeakerSegment`, `parse_speaker_segments` [mod.rs:parse_speaker_segments]
- `resample_poly::{firwin_kaiser_lowpass, mlx_audio_polyphase_fir, resample_poly_edge, resample_mono_polyphase}` (crate-private module) [resample_poly.rs:resample_mono_polyphase]
- `tokenizer_encoder::{TokenizerEncoder, SConv1d, load_sharded, load_linear_sharded, load_rms_norm_sharded}` (pub(crate)) [tokenizer_encoder.rs:TokenizerEncoder]
- `language_model::Qwen2` with `load`, `embed`, `tied_logits`, `prefill`, `step` (pub(crate)) [language_model.rs:Qwen2]

## Invariants
- Pinned repo and revision constants are immutable: `microsoft/VibeVoice-ASR-Streaming-1.5B` at `4262d23d8a539a6530cf64fbd0b1751ef9a30853` [mod.rs:VIBEVOICE_ASR_STREAMING_1_5B]
- Config refuses anything but model_type `vibevoice`, architecture `VibeVoiceForASRStreamingTraining`, `use_semantic_feature=true`, causal encoders with `disable_last_norm=true`, RMSNorm, constant padding, vae_dim 64 (acoustic) and 128 (semantic) [mod.rs:VibeVoiceAsrConfig::from_json]
- Decoder must be qwen2/silu with `tie_word_embeddings=true`; `rope_scaling` and sliding window are errors; head geometry must divide evenly [mod.rs:VibeVoiceAsrConfig::from_json]
- `normalize_audio` must be false, and preprocessor sample rate and compress ratio must equal config.json values [mod.rs:from_preprocessor, VibeVoiceAsr::load]
- Tensors ending `.scales` or `.biases` (MLX quantized) are refused at load [mod.rs:VibeVoiceAsr::load]
- Every tensor load checks the exact shape and errors if missing [tokenizer_encoder.rs:load_sharded]
- Encoder strides are the declared ratios reversed; stage i blocks consume downsample i's output [tokenizer_encoder.rs:TokenizerEncoder::load, TokenizerEncoder::forward]
- Acoustic and semantic frame counts must match, and empty audio and non-finite samples are errors [mod.rs:encode_speech, resample_input]
- The prompt must contain exactly one `<|box_start|>` placeholder per frame; special tokens are resolved at load [mod.rs:build_prompt, VibeVoiceAsr::load]
- Greedy decode stops on EOS ids 151643 or 151645 and is capped at 8192 tokens; logits are returned raw (argmax, no sampler) [mod.rs:EOS_TOKEN_IDS, MAX_NEW_TOKENS, greedy_from_embeddings]
- `embed` rejects ids outside the vocab (token ids are signed 32-bit) [language_model.rs:Qwen2::embed]
- The `lm_head` tensor is never read; the embedding matrix is the head [language_model.rs:Qwen2::tied_logits]
- Cached prefill plus steps must match the cache-free forward bitwise [language_model.rs:Qwen2::prefill, Qwen2::step]
- Resampling uses f64 design and accumulation and rounds through f32; `pad` is grown so `pad*up` divisible by `down`; zero rates are an error [resample_poly.rs:resample_poly_edge, resample_mono_polyphase]
- Resampler constants (64 zeros, rolloff 0.9475937167399596, beta 14.769656459379492) are pinned [resample_poly.rs:NUM_ZEROS, ROLLOFF, BETA]

## Side Effects
- Reads `config.json`, `preprocessor_config.json`, safetensors shards, and tokenizer files from the model dir; never downloads [mod.rs:VibeVoiceAsr::load]
- Loads all weights eagerly into f32 `Vec`s on the CPU; no GPU or Metal use [tokenizer_encoder.rs:load_sharded]
- Decode allocates a growing per-layer, per-KV-head key/value cache [language_model.rs:LayerCache]
- None for global statics, threads, or env vars in non-test code

## Verification
- Run: `cargo test -p turbospark-audio stt::vibevoice_asr`
- Ignored checkpoint gate: set `TURBOSPARK_VIBEVOICE_ASR_MODEL_DIR` and add `-- --ignored` [mod.rs:tests]
