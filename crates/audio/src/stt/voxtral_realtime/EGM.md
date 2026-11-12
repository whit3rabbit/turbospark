# Voxtral Realtime STT (offline buffered path)

## Overview
Voxtral Mini 4B Realtime speech-to-text, ported from mlx-audio 0.5.7. It runs the offline buffered branch only: silence padding, Slaney log-mel, a 32-layer causal sliding-window audio encoder with a 4x downsample adapter, a 26-layer Mistral-style GQA decoder with adaptive RMSNorm and tied embeddings, greedy decoding, and a decode-only Tekken tokenizer. The checkpoint profile, quantization (4-bit affine, group 64), and geometry are pinned; any other value is refused at config parse.

## Interface
- `VoxtralRealtime` with `load`, `profile`, `config`, `prepare`, `transcribe` [mod.rs:VoxtralRealtime]
- `VoxtralRealtimeProfile` and `VOXTRAL_MINI_4B_REALTIME_4BIT` [mod.rs:VoxtralRealtimeProfile, mod.rs:VOXTRAL_MINI_4B_REALTIME_4BIT]
- `VoxtralRealtimeConfig` with `from_json` [mod.rs:VoxtralRealtimeConfig]
- `PreparedAudio` (padded, mel, adapter output, frame counts, n_delay) [mod.rs:PreparedAudio]
- `EncoderGeometry` [encoder.rs:EncoderGeometry]
- `DecoderGeometry` [decoder.rs:DecoderGeometry]
- crate-internal: `pad_audio_streaming`, `num_audio_tokens`, `num_delay_tokens`, `prompt_ids`, `mel_spectrogram`, `RingCache`, `RopeTables` [mod.rs]; `AudioEncoder`, `load_linear` [encoder.rs]; `Decoder`, `DecodeCache`, `compute_time_embedding` [decoder.rs]; `TekkenTokenizer`, `decode_base64` [tokenizer.rs]

## Invariants
- Profile pin: repository `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`, revision `fdebf7b2af834a1db4b8a3c99ab7480b333adf9e` [mod.rs:VOXTRAL_MINI_4B_REALTIME_4BIT]
- Config refuses `model_type` other than `voxtral_realtime`, a missing `quantization`, any mode other than `affine`, and anything other than 4-bit group 64 [mod.rs:VoxtralRealtimeConfig::from_json]
- Geometry keys may be absent (pinned default) but a present, different value is an error [mod.rs:pinned_int, mod.rs:pinned_float]
- Non-causal encoder, untied embeddings, or a disabled ada RMSNorm conditioning are refused [mod.rs:VoxtralRealtimeConfig::from_json]
- Transcription delay is fixed at 480 ms (6 tokens), left pad 32 tokens, BOS 1, EOS 2, STREAMING_PAD 32, generation stops once more than 4096 tokens are recorded (the check runs after the push, so up to 4097 entries including EOS) [mod.rs:TRANSCRIPTION_DELAY_MS, mod.rs:N_LEFT_PAD_TOKENS, mod.rs:BOS_TOKEN_ID, mod.rs:MAX_NEW_TOKENS]
- Non-finite samples are rejected in `prepare` and in the mel stage [mod.rs:VoxtralRealtime::prepare, mod.rs:mel_spectrogram]
- `transcribe` errors if the adapter yields fewer rows than the prompt length [mod.rs:VoxtralRealtime::transcribe]
- `precompute_ada_scales` must run before `prefill` or `step`, which index `ada_scales` directly; `load` does this [decoder.rs:Decoder::precompute_ada_scales, mod.rs:VoxtralRealtime::load]
- `encode_full` is valid only for seq <= sliding_window (750) and errors otherwise; `prepare` picks `encode_chunked` above that [encoder.rs:AudioEncoder::encode_full, mod.rs:VoxtralRealtime::prepare]
- `RingCache::append` does not trim; `trim_to_capacity` must run after the chunk's queries so same-chunk keys stay visible [mod.rs:RingCache::append, mod.rs:RingCache::trim_to_capacity]
- Linears must match expected shape and bias presence (encoder wq, wv, wo, FFN w2 biased; wk, w1, w3 not; decoder none) [encoder.rs:load_linear, encoder.rs:EncoderAttention::load, decoder.rs:DecoderAttention::load]
- Conv weights are shape-checked as MLX `[out, kernel, in]` and transposed to `[out, in, kernel]` [encoder.rs:CausalConv::load]
- Decoder embedding refuses ids outside `0..vocab_size`; ids are `i32` [decoder.rs:Decoder::embed]
- Tekken: ids below `n_special` or in `special_ids` or out of range yield empty bytes; decode is lossy UTF-8 [tokenizer.rs:TekkenTokenizer::token_bytes, tokenizer.rs:TekkenTokenizer::decode]
- Greedy argmax over logits; sampler-side normalization is not applied [mod.rs:VoxtralRealtime::transcribe]

## Side Effects
- Reads `config.json`, `tekken.json`, and safetensors shards (index `weight_map` order, else `model.safetensors`, else `weights.safetensors`) from a caller-supplied directory; never downloads [mod.rs:load_config, mod.rs:open_checkpoint, tokenizer.rs:TekkenTokenizer::from_model_path]
- Loads and dequantizes all weights to f32 in host memory (CPU path, no Metal) [encoder.rs:load_linear, decoder.rs:Decoder::load]
- `mel_spectrogram` rebuilds the filterbank and DFT twiddle tables on every call [mod.rs:mel_spectrogram]
- No threads, global statics, or env vars in non-test code; the ignored gate test reads `TURBOSPARK_VOXTRAL_REALTIME_MODEL_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::voxtral_realtime`
- Real checkpoint gate (ignored): `TURBOSPARK_VOXTRAL_REALTIME_MODEL_DIR=<dir> cargo test -p turbospark-audio stt::voxtral_realtime -- --ignored`
