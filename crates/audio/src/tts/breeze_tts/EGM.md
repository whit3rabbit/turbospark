# Breeze TTS 2
## Overview
Port of `mlx_audio/tts/models/breeze_tts` (BreezeBlue): a qwen3-style backbone over summed per-codebook audio embeddings predicts codebook zero, a llama depth decoder fills the remaining codebooks, and a T5Gemma2 text encoder embeds the prompt. Generation stops at codebook frames; waveform decode (Qwen3-TTS speech tokenizer) has no Rust port here [mod.rs:Model::generate].

## Interface
- `Model` (open, load, generate, prompt_embeddings, embed_audio_frame, sample, num_codebooks) [mod.rs:Model]
- `ModelConfig`, `TextEncoderConfig`, `DepthDecoderConfig` [mod.rs:ModelConfig]
- `GenerateParams` (validate, use_cfg), `Generated`, `StopReason`, `ReferencePrompt` [mod.rs:GenerateParams]
- `TextEncoder` (forward), `DepthDecoder` (next_logits) [mod.rs:TextEncoder] [mod.rs:DepthDecoder]
- `mask_reserved_codec_logits`, `apply_repetition_penalty` [mod.rs:Model::mask_reserved_codec_logits]
- `DEFAULT_SPEAKER`, `DEFAULT_SAMPLE_RATE` [mod.rs:DEFAULT_SPEAKER]

## Invariants
- Params validated first: max_tokens > 0, top_k >= 0, temperature >= 0, top_p in [0,1], repetition_penalty > 0, cfg_scale finite [mod.rs:GenerateParams::validate]
- CFG runs only when scale is set and != 1.0 [mod.rs:GenerateParams::use_cfg] and an uncond prompt is given [mod.rs:Model::generate]
- CFG combine is `uncond + scale * (cond - uncond)`, widths must match [mod.rs:combine_cfg]
- Codec ids in `[codec_vocab_size, vocab_size)` are masked to -inf [mod.rs:Model::mask_reserved_codec_logits]
- Sampling the id `vocab_size` is EOS and stops generation [mod.rs:Model::generate]
- Temperature 0 never draws and is deterministic [mod.rs:Model::generate]
- Prompts are pre-tokenized ids, caller-owned [mod.rs:Model::prompt_embeddings]

## Side Effects
- Reads config.json and weights from the checkpoint dir [mod.rs:Model::open]
- Seeded draws via crate threefry `rng::KeySequence` [mod.rs:Model::generate]
- No env vars, threads, or logging seen

## Verification
- Run: `cargo test -p turbospark-audio tts::breeze_tts`
