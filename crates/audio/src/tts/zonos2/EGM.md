# Zonos v2 TTS

## Overview
Rust port of mlx-audio `zonos2` (Zyphra ZONOS2) at a pinned mlx-audio commit. A custom backbone (summed per-column embeddings, gated attention, fused-residual RMS blocks, dense or router-MoE feed-forward) runs over a 9+1 column delayed-audio frame layout. Waveforms decode through the Descript Audio Codec port [mod.rs:module doc]. Single segment, batch 1, non-streaming. Speaker embeddings are caller supplied.

## Interface
- `Model` with `open`, `load`, `load_config`, `compute_logits`, `build_prompt_rows`, `generate`, `decode_audio` [mod.rs:Model]
- `ModelConfig` with `from_json`, `is_moe_layer`, `num_experts_per_tok`, `frame_width`, `audio_vocab_size` [mod.rs:ModelConfig]
- `TtsSamplingParams` (defaults: temp 1.15, top_k 106, top_p 0.0, min_p 0.18, max_tokens 1024, rep penalty 1.2 over window 50 and 8 codebooks) [mod.rs:TtsSamplingParams]
- `GenerationState` with `new`, `finished`, `append` [mod.rs:GenerationState]
- `sample_frame` [mod.rs:sample_frame]
- `text_to_byte_ids` [mod.rs:text_to_byte_ids]
- `speaking_rate_token_id`, `quality_token_id`, `speaker_background_token_id`, `accurate_mode_token_id` [mod.rs:speaking_rate_token_id]
- `shear`, `shear_up`, `silence_prompt_tokens`, `SILENCE_TOKENS_0_2S` [mod.rs:shear] [mod.rs:SILENCE_TOKENS_0_2S]
- Constants `REFERENCE_COMMIT`, `PAD_ID`, `UNK_ID`, `BOS_ID`, `EOS_ID`, `LEGACY_SYMBOL_VOCAB_SIZE`, `BYTE_TEXT_VOCAB_SIZE` [mod.rs:REFERENCE_COMMIT]
- No child directories.

## Invariants
- `text_vocab` must be at least 448 (byte ids) and large enough for the conditioning buckets [mod.rs:ModelConfig::validate] [mod.rs:conditioning_base].
- Text ids are BOS, utf-8 bytes + 192, EOS [mod.rs:text_to_byte_ids].
- Bucket indices beyond their bucket count return `Input` errors [mod.rs:speaking_rate_token_id].
- `shear` and `shear_up` require a non-empty rectangular table; `shear_up` inverts `shear` delay [mod.rs:shear] [mod.rs:shear_up].
- Frame layout width is `frame_width`; input frames must be a multiple of it [mod.rs:Model::forward_hidden].
- `sample_frame` asserts `logits.len() == codebooks * vocab` [mod.rs:sample_frame].
- Generation ends only after the EOS countdown (`n_codebooks + 1` frames after the first EOA) unless `ignore_eos` [mod.rs:GenerationState::append] [mod.rs:GenerationState::finished].
- `generate` takes `&self`: KV caches are created per call, so it is re-entrant [mod.rs:Model::generate].
- Speaker conditioning prepends a slot at row 0 only when an embedding is supplied; the speaker is injected on the prefill only [mod.rs:Model::build_prompt_rows] [mod.rs:Model::generate].
- Decode clamps codes to `[0, codebook_size - 1]`, truncates to the EOS frame, and trims audio to `frames * hop_length` [mod.rs:Model::decode_audio].

## Side Effects
- Reads `config.json` and safetensors shards from the checkpoint dir [mod.rs:Model::open].
- RNG: a fresh threefry key sequence per step, seeded by two rounds of splitmix64 over `(seed, step)` [mod.rs:step_seed] [mod.rs:Model::generate].
- No env vars, threads, or logging found.

## Verification
- Run: `cargo test -p turbospark-audio tts::zonos2`
