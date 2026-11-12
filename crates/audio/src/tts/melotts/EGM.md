# MeloTTS
## Overview
Port of mlx-audio `melotts` (pinned commit): a VITS2-family TTS with a window-4 relative-attention transformer encoder, deterministic duration predictor, transformer coupling flow, and HiFi-GAN decoder with speaker conditioning, producing 44.1 kHz audio. Includes the 12-layer BERT used for ja_bert features and the symbol and phone-id helpers. Batch size 1, non-streaming.

## Interface
- `Model` (open, resolve_speaker, infer, config) [mod.rs:Model]
- `ModelConfig` (from_json, sample_rate) [mod.rs:ModelConfig]
- `BertDims` [mod.rs:BertDims]
- `BertModel` (load, extract_features, pooled, phone_features) [bert.rs:BertModel]
- `REFERENCE_COMMIT`, `REFERENCE_BERT_DIM`, `SAMPLE_RATE` [mod.rs:SAMPLE_RATE]
- `default_symbols`, `cleaned_text_to_sequence`, `add_blank_interleave`, `distribute_phone` [text.rs:cleaned_text_to_sequence]
- `text_normalize`, `number_to_words`, `refine_ph`, `post_replace_ph` [text.rs:text_normalize]

## Invariants
- `infer` refuses `sdp_ratio != 0` (stochastic duration predictor not ported) [mod.rs:Model::infer]
- Tone and language id slices must match the phone count [mod.rs:Model::infer]
- Speaker id must index within `emb_g` [mod.rs:Model::infer]
- Flow is exactly four coupling+flip pairs, coupling layers at even tree indices; reverse pass flips before each coupling [mod.rs:Model::open] [mod.rs:flow_reverse]
- Speaker resolution order: voice, lang code, "EN-Default", then 0 [mod.rs:resolve_speaker]
- Training-only `enc_q.*` and `sdp.*` tensors are ignored [mod.rs:Model::open]
- Unknown phones map to `UNK`; unknown language falls back to id 0 and tone start 0 [text.rs:cleaned_text_to_sequence]
- BERT features come from the third-to-last hidden state, per-layer LayerNorm eps 1e-5 [bert.rs:LAYER_NORM_EPS], embedding LayerNorm eps from config `layer_norm_eps` (default 1e-12) [bert.rs:embedding_eps] [mod.rs:BertDims] [bert.rs:extract_features]

## Side Effects
- Reads `config.json` and `model.safetensors` from the model dir [mod.rs:Model::open]
- Noise uses the crate-local MLX-compatible threefry key sequence seeded by `seed` argument [mod.rs:Model::infer]
- No env vars, threads, or logging

## Verification
- Run: `cargo test -p turbospark-audio tts::melotts`
