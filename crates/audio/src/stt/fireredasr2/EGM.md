# FireRedASR2-AED Speech Recognition

## Overview
Portable CPU port of the FireRedASR2-AED attention encoder-decoder from mlx-audio 0.5.7. `FireRedAsr2` loads config, CMVN, dictionary and a safetensors checkpoint from a local directory, then runs Kaldi FBANK with CMVN, a 16-block Conformer encoder with relative-position attention, and a 16-layer Transformer decoder with beam search [mod.rs:FireRedAsr2]. Only the pinned 80-bin, width-1280 profile is accepted [config.rs:FireRedAsr2Config::from_json].

## Interface
- `FireRedAsr2` with `open`, `profile`, `config`, `transcribe`, `transcribe_with_options` [mod.rs:FireRedAsr2]
- `FireRedAsr2Options` (beam_size 3, max_len 0, softmax_smoothing 1.25, length_penalty 0.6, eos_penalty 1.0 by default) [mod.rs:FireRedAsr2Options]
- `FireRedAsr2Profile`, `FIREREDASR2_AED` [mod.rs:FIREREDASR2_AED]
- `FireRedAsr2Config` (re-export) with `from_json` [mod.rs:FireRedAsr2Config, config.rs:FireRedAsr2Config]

## Invariants
- Config must have `model_type == "fireredasr2"`, idim 80, odim 8667, d_model 1280, 16 encoder and decoder layers, 20 heads, kernel 33, position limits >= 5000, otherwise `Unsupported` [config.rs:FireRedAsr2Config::from_json]
- Special token ids must be < vocab_size, model_dim must divide by both head counts and the encoder kernel must be odd [config.rs:FireRedAsr2Config::from_json]
- `dict.txt` entry count must equal `config.vocab_size`, an empty line maps to a single space token [mod.rs:FireRedAsr2::open, mod.rs:parse_dictionary]
- Pinned snapshot is `mlx-community/FireRedASR2-AED-mlx` at revision f3212eac... [mod.rs:FIREREDASR2_AED]
- CMVN needs exactly 80 finite `means` and `istd` values [frontend.rs:Frontend::from_json]
- Audio must be non-empty and finite, a peak above 1.0 is divided by 32768 before the shared FBANK (25 ms/10 ms, 80 mel, 16 kHz), zero complete frames is an error [frontend.rs:Frontend::extract]
- Encoder input length must equal frames * 80, two stride-2 3x3 convs follow 6 zero-padded frames [encoder.rs:Encoder::forward_layers]
- Checkpoint tensor names are the raw PyTorch keys: `conv1`/`conv2`/`net_N` are remapped to `conv.0`/`conv.2`/`net.N`, every tensor shape is checked on load [encoder.rs:raw_tensor_name, encoder.rs:tensor, decoder.rs:tensor]
- Attention only computes the T relative positions the MLX `_rel_shift` selects (index T-1+key-query), test pins bitwise equality with the full-matrix form [encoder.rs:RelativeAttention::forward, encoder.rs:relative_index]
- `decode` rejects beam_size 0 or > vocab, non-finite or non-positive smoothing, negative length_penalty, non-finite eos_penalty, and max_len 0 or above the decoder position limit [decoder.rs:Decoder::decode]
- `max_len == 0` means decode for the encoder output length [mod.rs:FireRedAsr2::transcribe_with_options]
- Beam 0 starts at score 0 and others at -1e10, candidate ties break by score, then parent, then token, finished beams drop their KV cache [decoder.rs:Decoder::decode]
- Self-attention KV cache is preallocated to max_len and cached logits must match full-prefix recompute bitwise [decoder.rs:ProjectedKeys, decoder.rs:Decoder::beam_logits]
- `decoder.tgt_word_prj` falls back to the tied `decoder.tgt_word_emb.weight` when absent [decoder.rs:load_linear]
- Output is trimmed and lowercased with `<space>` and U+2581 as spaces and `<blank>`/`<sil>` removed [decoder.rs:Decoder::decode]

## Side Effects
- Reads `config.json`, `cmvn.json`, `dict.txt`, `model.safetensors` from the model dir, never downloads [mod.rs:FireRedAsr2::open]
- Opens the checkpoint via `SafetensorsFile` and copies every tensor into f32 `Vec`s [encoder.rs:tensor, decoder.rs:tensor]
- No threads, GPU/Metal use, global statics or env vars in production code, ignored tests read `TURBOSPARK_FIREREDASR2_DIR` [mod.rs:tests]

## Verification
- Run: `cargo test -p turbospark-audio stt::fireredasr2`
- Pinned-checkpoint parity tests are `#[ignore]` and need `TURBOSPARK_FIREREDASR2_DIR` [mod.rs:pinned_checkpoint_matches_mlx_stages_and_transcript]
