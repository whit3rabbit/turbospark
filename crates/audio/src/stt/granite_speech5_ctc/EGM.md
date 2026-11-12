# Granite Speech 5.0 TurboCTC

## Overview
Encoder-only CTC speech-to-text for the pinned IBM Granite Speech 5.0 TurboCTC 470M checkpoint. `mod.rs` validates the checkpoint config, loads a safetensors file and tokenizer, and exposes `transcribe` over mono 16 kHz f32 PCM. `frontend.rs` builds stacked log-mel plus delta features, and `encoder.rs` runs a CPU f32 Conformer with two stride-2 subsample layers, midpoint self-conditioning and a greedy CTC collapse.

## Interface
- `GraniteSpeech5Profile`, `GRANITE_SPEECH5_TURBOCTC` (repository and revision pin) [mod.rs:GraniteSpeech5Profile] [mod.rs:GRANITE_SPEECH5_TURBOCTC]
- `GraniteSpeech5Config` and `GraniteSpeech5Config::from_json` [mod.rs:GraniteSpeech5Config] [mod.rs:from_json]
- `GraniteSpeech5Ctc::load(model_dir)`, `config()`, `transcribe(samples)` [mod.rs:GraniteSpeech5Ctc]
- Crate-private: `frontend::compute_features`, `encoder::Encoder::{load, forward}` [frontend.rs:compute_features] [encoder.rs:Encoder]

## Invariants
- The checkpoint is pinned to repo `ibm-granite/granite-speech-5.0-470m-turboctc` at a 40-char commit [mod.rs:GRANITE_SPEECH5_TURBOCTC].
- Config must have `model_type` `granite_speech5_ctc`, architecture `GraniteSpeech5ForCTC`, and `tie_word_embeddings` true, else `BadConfig` [mod.rs:from_json].
- Only `num_mel_bins` 80 or 128, `subsample_layers == [0, 1]`, `pad_token_id == 0`, odd conv kernel, and `hidden_size == heads * head_dim` are accepted [mod.rs:from_json].
- Encoder-level `vocab_size` must equal the top-level one, non-silu activation, nonzero dropout, or missing attention bias returns `Unsupported` [mod.rs:from_json].
- Features: input must be non-empty and finite, frames are rounded up to a multiple of 2, and padded buffer is capped at 2^27 samples [frontend.rs:compute_features].
- Feature row layout is `[logmel, delta, logmel, delta]` per frame pair, width `4 * mel_bins`, the log-mel floor is peak minus 8 [frontend.rs:compute_features] [frontend.rs:LOGMEL_FLOOR_DB].
- The mel filterbank is built in f64 HTK "precise" mode with 512-point FFT, hop 160, Hann window 400 [frontend.rs:precise_htk_filterbank] [frontend.rs:N_FFT].
- Every tensor is shape-checked against its expected shape before f32 decode, mismatch or absence gives `SpeechError::Tensor` [encoder.rs:load_tensor].
- Subsample layers halve rows with `rows / 2` and average adjacent pairs for the residual, so odd row counts drop the last row [encoder.rs:EncoderBlock::forward].
- Attention runs in blocks of `context_size` with zero-padded rows and a relative table of `2 * max_position_embeddings + 1` rows, clamped distance [encoder.rs:Attention::forward] [encoder.rs:Attention::load].
- Self-conditioning is applied after layer `len / 2` (index + 1 == midpoint) using softmax over `output` logits then `output_mid` [encoder.rs:Encoder::forward] [encoder.rs:apply_midpoint_conditioning].
- Logits are computed in batches of 8 rows and collapsed with `CtcCollapse` using `pad_token_id` as blank, never materializing full logits [encoder.rs:collapse_final_logits] [encoder.rs:LOGIT_ROW_BATCH].
- Token ids must fit `u32` for the tokenizer, decode output is trimmed [mod.rs:transcribe].

## Side Effects
- Reads `config.json`, opens `model.safetensors` via `SafetensorsFile::open`, and loads `tokenizer.json` [mod.rs:GraniteSpeech5Ctc::load].
- Copies all weights into owned `Vec<f32>` at load time [encoder.rs:load_tensor].
- No threads, env vars, global statics, or GPU use, inference is single-threaded CPU [encoder.rs:Encoder::forward].

## Verification
- Run: `cargo test -p turbospark-audio stt::granite_speech5_ctc`
