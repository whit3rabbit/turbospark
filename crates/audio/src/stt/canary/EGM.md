# Canary-1B-v2 STT

## Overview
Multilingual ASR and translation port of the mlx-audio Canary reference. The encoder reuses the shared FastConformer and NeMo log-mel frontend; this module adds the config parser with refusals, the 8-layer pre-norm transformer decoder with a fixed sinusoidal table and KV cache, an embedded SentencePiece vocabulary parsed from `config.json`, and a greedy decode loop. Everything runs on CPU in f32 after dequantizing 8-bit weights.

## Interface
- `Canary` with `load`, `profile`, `config`, `tokenizer`, `transcribe`, `transcribe_with_options`, `transcribe_impl` [mod.rs:Canary]
- `CanaryProfile`, `CANARY_1B_V2_Q8` pinned checkpoint reference [mod.rs:CANARY_1B_V2_Q8]
- `StageWitness`, `FirstLogits`, `TranscribeStages` parity-capture types [mod.rs:TranscribeStages]
- `CanaryConfig::from_json`, `DecoderConfig` [config.rs:CanaryConfig]
- `CanaryTokenizer` with `from_config`, `from_proto`, `vocab_size`, `piece`, `token_to_id`, `eos_id`, `build_prompt_tokens`, `decode` [tokenizer.rs:CanaryTokenizer]
- `CanaryCache` (re-export of decoder `Cache`) [mod.rs:CanaryCache]
- `decoder::CanaryDecoder` and `fixed_positional_encoding` are `pub` inside a private module [decoder.rs:CanaryDecoder] [decoder.rs:fixed_positional_encoding]

## Invariants
- Pinned checkpoint is repo `Mediform/canary-1b-v2-mlx-q8` at revision `0b6b32ee10f30c89e3ead7249bb636445e3019ee` [mod.rs:CANARY_1B_V2_Q8]
- Config refuses anything but 16 kHz, hann window, per_feature normalization [config.rs:from_json]
- Config refuses non-dw_striding, non-rel_pos, causal-downsampling encoders, subsampling_factor != 8, even conv kernel, feat_in != mel bins [config.rs:from_json]
- Config refuses enc_output_dim != d_model (no projection tensor is loaded) [config.rs:from_json]
- Decoder config requires pre_ln=true, pre_ln_final_layer_norm=true, learn_positional_encodings=false, relu activation, hidden_size equal to encoder d_model [config.rs:from_json]
- Only 8-bit, group size 64 quantization is accepted [config.rs:from_json]
- Frontend lengths (n_fft, window, stride) are bounded before the Hann window is allocated; n_fft must be even and >= window [config.rs:from_json]
- Tokenizer vocab size must equal config vocab_size and fit i32; classifier width must match [mod.rs:Canary::load]
- Audio must be non-empty and finite, else `SpeechError::Input` [mod.rs:transcribe_impl]
- Prompt order is startofcontext, startoftranscript, emo:undefined, src, tgt, pnc/nopnc, noitn, notimestamp, nodiarize; missing tokens (e.g. unknown language) error [tokenizer.rs:build_prompt_tokens]
- `max_tokens = 0` still yields the first token; loop runs `max_tokens - 1` more steps; EOS is not emitted [mod.rs:transcribe_impl]
- Decoder forward errors on empty input, negative or out-of-vocab ids, and `start_pos + t > max_sequence_length` [decoder.rs:CanaryDecoder::forward]
- Batch size is one; cross-attention keys beyond `encoder_len` get the -1e9 mask; `new_cache` requires `encoder_output.len() == encoder_len * d_model` [decoder.rs:new_cache] [decoder.rs:MASK]
- Decode drops control pieces, refuses unknown-type ids and other piece types, maps U+2581 to space and trims [tokenizer.rs:decode]
- Linear layers require a bias tensor; token embedding tensor must exist [decoder.rs:Linear::load] [decoder.rs:CanaryDecoder::load]
- EOS falls back to id 0 if `<|endoftext|>` is absent [tokenizer.rs:eos_id]

## Side Effects
- Reads `config.json` and opens `model.safetensors` from the model directory [mod.rs:Canary::load]
- Dequantizes all linears and the embedding to f32 Vecs at load (memory heavy, CPU only) [decoder.rs:Linear::load]
- Allocates a `[max_sequence_length, d_model]` positional table per model [mod.rs:Canary::load]
- No threads, env vars, GPU use or global statics in non-test code. Ignored test reads `TURBOSPARK_CANARY_MODEL_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::canary`
- Checkpoint parity (ignored): `TURBOSPARK_CANARY_MODEL_DIR=<snapshot> cargo test -p turbospark-audio stt::canary -- --ignored`
