# Qwen2-Audio Offline Transcription

## Overview
Offline single-window transcription for the pinned Qwen2-Audio-7B-Instruct 4-bit checkpoint. A whisper-family log-mel frontend feeds a conv plus transformer audio tower and a linear projector (encoder.rs), whose rows are spliced into a chat prompt and decoded greedily by a Qwen2 decoder (language_model.rs). mod.rs owns config pinning, the frontend, tokenizer setup, and the two-phase load (tower first, decoder later) that keeps resident memory bounded.

## Interface
- `Qwen2AudioProfile`, `QWEN2_AUDIO_7B_INSTRUCT_4BIT` pinned repo and revision [mod.rs:QWEN2_AUDIO_7B_INSTRUCT_4BIT]
- `Qwen2AudioConfig`, `AudioEncoderConfig`, `TextDecoderConfig`, `Qwen2AudioConfig::from_json` [mod.rs:Qwen2AudioConfig]
- `DEFAULT_USER_PROMPT`, `build_prompt_string` [mod.rs:build_prompt_string]
- `AudioFeatures` (input_features, tower_output, projected, frames, hidden) [mod.rs:AudioFeatures]
- `Qwen2AudioAudioTower` with `load`, `profile`, `config`, `encode` [mod.rs:Qwen2AudioAudioTower]
- `Qwen2AudioLanguageModel` with `load`, `build_prompt`, `splice`, `generate_from_embeddings`, `decode_generated`, `transcribe_embeddings` [mod.rs:Qwen2AudioLanguageModel]
- `Qwen2Audio` with `load`, `transcribe` (combined path) [mod.rs:Qwen2Audio]
- crate-internal: `extract_features`, `mel_filterbank` [mod.rs:extract_features], `AudioTower`, `Projector` [encoder.rs:AudioTower], `LanguageModel`, `DecodeCache` [language_model.rs:LanguageModel]

## Invariants
- Config keys present must equal pinned values (d_model 1280, 32 encoder layers, 20 heads, ffn 5120, 128 mels, 1500 positions; text 4096/11008/32/32/32, vocab 156032); absent keys take the pinned default [mod.rs:pinned_int].
- Only 4-bit group-64 quantization, untied lm_head, attention_bias true, no rope_scaling, no sliding window, silu/gelu activations, and `<|AUDIO|>` id 151646 are accepted; others return errors [mod.rs:Qwen2AudioConfig::from_json].
- Frontend is fixed: audio padded or truncated to 480000 samples, 3001 mel frames, 750 audio tokens; non-finite samples are rejected [mod.rs:extract_features].
- Tower output is band-major `[num_mels, frames]` in, `[750, d_model]` out; conv length is even so pair pooling is exact [encoder.rs:AudioTower::forward].
- Conv kernels are stored `[out, kernel, in]` and transposed once at load; shapes are checked per tensor [encoder.rs:TowerConv::load, encoder.rs:load_sharded].
- Prompt placeholder count must equal the encoded frame count, else `SpeechError::Input` [mod.rs:Qwen2AudioLanguageModel::build_prompt].
- Splice writes feature rows at audio positions and embeds id 0 where the placeholder stood [mod.rs:Qwen2AudioLanguageModel::splice].
- Token ids are signed 32-bit; embed refuses ids outside `[0, vocab)` [language_model.rs:LanguageModel::embed].
- Decode is greedy argmax over logits with no sampling normalization, stops at the tokenizer EOS or 4096 tokens [mod.rs:generate_from_embeddings, mod.rs:MAX_NEW_TOKENS].
- Decode steps must use the cache position (`cache.layers[0].len`) for RoPE; cached prefill plus steps matches the full forward bit for bit [language_model.rs:LanguageModel::step].
- Tokenizer: added tokens from `tokenizer_config.json` are added in id order and each must retain its id; class must be Qwen2Tokenizer [mod.rs:load_tokenizer].
- Shard order follows sorted `weight_map` values, else `weights.safetensors`, else `model.safetensors` [mod.rs:open_checkpoint].

## Side Effects
- Reads `config.json`, `tokenizer.json`, `tokenizer_config.json`, safetensors shards from the given model dir; never downloads [mod.rs:load_config, mod.rs:load_tokenizer, mod.rs:open_checkpoint].
- Dequantizes the decoder to f32 at load (roughly 31 GB resident per the doc comment), so load it after dropping the tower [mod.rs:Qwen2AudioLanguageModel::load, language_model.rs:load_linear].
- Builds 400x201 f64 DFT twiddle tables per `extract_features` call; no global statics or env vars [mod.rs:dft_tables].
- No threads, no GPU or Metal use (CPU f32 only) [language_model.rs:LanguageModel].

## Verification
- Run: `cargo test -p turbospark-audio stt::qwen2_audio`
