# Nemotron 3.5 ASR

## Overview
Offline and cache-aware-streaming causal FastConformer encoder with a language-prompt kernel and greedy RNN-T decode (LSTM predictor plus joint). `NemotronAsr` is the offline entry point; the `encoder` and `rnnt` submodules (crate-private) are also reused by Multitalker Parakeet for its streaming path. Config parsing validates the single supported graph variant. [mod.rs:NemotronAsr] [mod.rs:NemotronAsrConfig] [encoder.rs:NemotronEncoder] [rnnt.rs:greedy_resume]

## Interface
- `NemotronAsrProfile`, `NEMOTRON_3_5_ASR` pinned checkpoint [mod.rs:NEMOTRON_3_5_ASR]
- `NemotronAsrConfig` with `from_json`, `prompt_index`; `from_json_prompt_free` is `pub(crate)` for Multitalker [mod.rs:NemotronAsrConfig]
- `NemotronAsr` with `open`, `config`, `transcribe`, `transcribe_with_language` [mod.rs:NemotronAsr]
- `StreamEncoderWeights`, `StreamLayerWeights` (pub structs, borrowed weights for device executors, re-exported by multitalker) [encoder.rs:StreamEncoderWeights]
- `pub(crate)`: `NemotronEncoder` (`load`, `encode`, `pre_encode`, `stream_geometry`, `new_stream_cache`, `stream_layers`, `stream_weights`), `EncoderStreamCache`, `StreamGeometry` [encoder.rs:NemotronEncoder]
- `pub(crate)`: `NemotronPredictor`, `Joint`, `LstmLayer`, `lstm_stack`, `GreedyStreamState`, `greedy_tokens`, `greedy_resume`, `sigmoid` [rnnt.rs:greedy_resume]

## Invariants
- Pinned checkpoint: `mlx-community/nemotron-3.5-asr-streaming-0.6b` at `e550040c0478027ed679b2b6b0d055502c103663`; `from_json` requires `model_type == "nemotron_asr"`. [mod.rs:NEMOTRON_3_5_ASR] [mod.rs:NemotronAsrConfig::from_json]
- Only sample_rate 16000, hann window, `normalize == "NA"`, layer_norm conv, causal conv context, rel_pos attention, chunked_limited style, causal downsampling, no bias, no xscaling are accepted; else `Unsupported`. [mod.rs:NemotronAsrConfig::parse]
- Config cross-checks: `feat_in == features`, `d_model % n_heads == 0`, power-of-two subsampling, odd conv kernel, default attention context in `att_context_size`, default language in prompt dictionary, vocab sizes consistent, `joint_hidden == pred_hidden`, relu joint. [mod.rs:NemotronAsrConfig::parse]
- `transcribe` rejects other sample rates and empty or non-finite PCM; offline mel uses `StftPaddingMode::Reflect`. [mod.rs:NemotronAsr::transcribe_with_language]
- Unknown language falls back to the default language prompt, then index 0. [mod.rs:NemotronAsrConfig::prompt_index]
- Special pieces (`<unk>`, `<pad>`, `<s>`, `</s>`, `<xx-YY>` language tags) are dropped from output text. [mod.rs:is_special_piece] [mod.rs:NemotronAsr::greedy_decode]
- Weight tensors must exist with exact shape and F16/BF16/F32 dtype; attention, FF and conv layers are bias-free and a present bias is an error. [encoder.rs:load_tensor] [encoder.rs:load_bias_free_linear] [encoder.rs:ConformerConvolution::load]
- Streaming geometry requires left context nonzero and a whole multiple of `right + 1`. [encoder.rs:NemotronEncoder::stream_geometry]
- `stream_layers` errors when `rows == 0`, `rows > chunk`, `valid > rows`, or input length mismatches; rows at or past `valid` yield zero attention rows and are zeroed before the depthwise conv. [encoder.rs:NemotronEncoder::stream_layers] [encoder.rs:RelPosAttention::forward_cached] [encoder.rs:ConformerConvolution::forward_cached]
- Cache `attn_len` grows by `rows` capped at `attn_frames`. [encoder.rs:NemotronEncoder::stream_layers]
- Valid length follows `floor(L/2)+1` per subsampling stage minus `drop`; `drop > rows` errors. [encoder.rs:NemotronEncoder::pre_encode]
- Blank is `blank_id == vocabulary.len()`; the predictor embedding has `vocabulary + 1` rows; a token above `vocabulary` errors. [rnnt.rs:NemotronPredictor::step] [rnnt.rs:NemotronPredictor::load]
- After a blank the cached prediction is reused; `max_symbols` forces a frame advance. Split-chunk resume must equal single pass. [rnnt.rs:greedy_resume] [rnnt.rs:GreedyStreamState]
- Producers return logits; argmax is taken in the decoder, no softmax. [rnnt.rs:Joint::logits] [mod.rs:argmax]

## Side Effects
- Reads `config.json` and maps `model.safetensors` in `open`. [mod.rs:NemotronAsr::open]
- Per-call allocation of FFT/mel plans through `nemo_log_mel_spectrogram_with_padding`; no GPU use in this module, CPU `ops` only. [mod.rs:NemotronAsr::transcribe_with_language]
- No env vars, threads, caches or global statics.

## Verification
- Run: `cargo test -p turbospark-audio stt::nemotron_asr`
