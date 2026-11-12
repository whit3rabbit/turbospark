# Wav2Vec2 Base CTC

## Overview
Wav2Vec2 base (group-norm feature extractor, post-norm encoder) with its own CTC head, transcribing mono 16 kHz audio by greedy CTC. `backbone.rs` and `ctc.rs` hold helpers shared with the `mms` family (positional conv, attention, vocab parsing, CTC decoding and collapse). The module also exposes a headless `SemanticTower` used by the Higgs tokenizer path.

## Interface
- `Wav2Vec` with `load`, `profile`, `config`, `transcribe` [mod.rs:Wav2Vec]
- `Wav2VecProfile`, `WAV2VEC2_BASE_960H` [mod.rs:WAV2VEC2_BASE_960H]
- `Wav2VecConfig` with `from_json` (pub) and `from_semantic_dict` (pub(crate)) [mod.rs:Wav2VecConfig]
- `SemanticTower` with `load`, `config`, `forward_hidden_mean` (pub(crate)) [mod.rs:SemanticTower]
- `backbone::{positive, positive_array, channels_first_to_rows, rows_to_channels_first, PositionalConv, Attention, add, load_tensor_alias, parse_vocab, decode_ctc}` (pub(crate), module is pub(super)) [backbone.rs:decode_ctc]
- `ctc::CtcCollapse` with `new`, `push`, `finish` [ctc.rs:CtcCollapse]

## Invariants
- Pinned repo and revision: `facebook/wav2vec2-base-960h` at `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` [mod.rs:WAV2VEC2_BASE_960H]
- `from_json` requires model_type `wav2vec2`, architecture `Wav2Vec2ForCTC`, `feat_extract_norm=group`, `do_stable_layer_norm=false`, `hidden_act=gelu`, and no `adapter_attn_dim` (those belong to mms) [mod.rs:Wav2VecConfig::from_json]
- Geometry check: hidden divisible by heads and pos-conv groups; conv_dim, stride, kernel equal length and non-empty [mod.rs:Wav2VecConfig::parse_fields]
- `from_semantic_dict` accepts model_type `wav2vec2` or `hubert` with the same norm and activation constraints [mod.rs:Wav2VecConfig::from_semantic_dict]
- Input must be non-empty and finite; normalization is zero-mean unit-variance with epsilon 1e-7 inside the sqrt [mod.rs:Wav2Vec::logits]
- Audio shorter than a conv kernel returns an input error [mod.rs:GroupNormConv::forward, PlainConv::forward]
- Encoder order: positional conv add, then `encoder_norm`, then layers, no final norm; each layer is attention, residual, norm, FFN, residual, final norm [mod.rs:Wav2Vec2::forward, EncoderLayer::forward]
- `forward_hidden_mean` averages the normed encoder input plus every layer output [mod.rs:Wav2Vec2::forward_hidden_mean]
- Positional conv weight-norm gain length must be 1, kernel, width, or width*kernel, accepting both `weight_g/weight_v` and `parametrizations.weight.original0/1` names; even kernels drop the extra frame [backbone.rs:PositionalConv::load, PositionalConv::forward, load_tensor_alias]
- `vocab.json` ids must be unique, in range, and cover every id [backbone.rs:parse_vocab]
- `decode_ctc` errors on logit or vocab size mismatch; blank is id 0, `|` becomes a space [backbone.rs:decode_ctc]
- `CtcCollapse` advances `previous` on every frame including blanks, so a repeat split by a blank is kept [ctc.rs:CtcCollapse::push]
- Tensors load via shape-checked `load_tensor`; `lm_head` is loaded with bias [mod.rs:Wav2Vec::load]

## Side Effects
- Reads `config.json`, `model.safetensors`, and `vocab.json` from the model dir; never downloads [mod.rs:Wav2Vec::load]
- Loads all weights into f32 `Vec`s; CPU only, no GPU or Metal [mod.rs:Wav2Vec2::load]
- None for threads, global statics, or env vars in non-test code (tests read `TURBOSPARK_WAV2VEC_MODEL_DIR`) [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::wav2vec`
