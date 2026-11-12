# Voxtral TTS

## Overview
Rust port of mlx-audio `voxtral_tts` (Voxtral-4B-TTS-2603) at a pinned mlx-audio commit. A Mistral-geometry LM backbone (shared lm engine) consumes caller-tokenized prompt ids. Each step an acoustic head predicts one frame: a semantic code by masked argmax plus acoustic FSQ codes by an Euler flow-matching head with classifier-free guidance. A codec decoder turns the code grid into 24 kHz audio [mod.rs:module doc]. Single batch, non-streaming, no tokenizer.

## Interface
- `Model` with `open`, `sample_rate`, `num_codebooks`, `voices`, `build_prompt_embeddings`, `generate`, `decode_codes`, `decode_codes_traced` [mod.rs:Model]
- `ModelConfig` with `from_json` (accepts direct HF and Mistral `params.json` layouts) [mod.rs:ModelConfig]
- `GenerateParams` (only `max_tokens`, default 4096) [mod.rs:GenerateParams]
- `Generated` (frames, codes, stopped, samples, sample_rate) [mod.rs:Generated]
- `VoiceEmbedding` (rows, values) [mod.rs:VoiceEmbedding]
- `FlowNoise` trait with `normal` [mod.rs:FlowNoise]
- `SeededFlowNoise` with `new` [mod.rs:SeededFlowNoise]
- `StageTrace` (crate-visible type alias) [mod.rs:StageTrace]
- No child directories.

## Invariants
- `generate` rejects `max_tokens == 0` and empty `prompt_ids` [mod.rs:Model::generate].
- A named voice that is not found is an error, never a silent unconditioned run [mod.rs:Model::voice] [mod.rs:Model::generate].
- A voice file must hold an `embedding` tensor; the reference first-tensor fallback is not ported [mod.rs:Model::voice].
- `[AUDIO]` positions take voice rows by running index, clipped to the last row [mod.rs:Model::build_prompt_embeddings].
- Semantic code <= 1 ends generation; acoustic codes carry a +2 special-token offset [mod.rs:AcousticHead::decode_one_frame] [mod.rs:Model::generate].
- Each codebook has its own offset in one shared embedding table; out-of-range codes error [mod.rs:Model::generate] [mod.rs:Model::codebook_row_width].
- Flow head: Euler over linspace(0,1) with `steps - 1` iterations and CFG blend `alpha*cond + (1-alpha)*uncond`; uncond conditioning is a zero vector [mod.rs:AcousticHead::decode_one_frame].
- FSQ rounding is ties-to-even to match `mx.round` [mod.rs:round_ties_even].
- No audio frames generated is an error [mod.rs:Model::generate].
- Decoder stage strides, kernels, and transformer lengths must share a stage count and be nonempty [mod.rs:TokenizerDecoder::load].
- Generation is deterministic given the noise; temperature/top_k/top_p are not accepted [mod.rs:GenerateParams].
- LM KV caches reset at the start of each `generate` [mod.rs:Model::generate].

## Side Effects
- Reads `config.json`, safetensors shards, and lists `voice_embedding/` under the checkpoint dir at open [mod.rs:Model::open] [mod.rs:fs_entries].
- Lazily reads a voice safetensors file on first use and caches it in the model [mod.rs:Model::voice].
- Remaps raw Mistral tensor names when `tok_embeddings.weight` is present [mod.rs:Model::open] [mod.rs:mistral_name_map].
- RNG: crate-local threefry via `SeededFlowNoise` [mod.rs:SeededFlowNoise].
- No env vars, threads, or logging found.

## Verification
- Run: `cargo test -p turbospark-audio tts::voxtral_tts`
