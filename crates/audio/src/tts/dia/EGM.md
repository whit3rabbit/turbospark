# Dia Dialogue TTS
## Overview
Nari Labs Dia 1.6B dialogue TTS, port of mlx-audio `dia` pinned at one commit [mod.rs:REFERENCE_COMMIT]. A byte-level text encoder conditions a nine-channel audio-code decoder via precomputed cross attention, with CFG and a channel delay pattern. Codes decode through a caller-loaded Descript Audio Codec (`Dac`) [mod.rs:Model::generate].

## Interface
- `Model` (open, load, sample_rate, split_turns, prepare_text, generate) [mod.rs:Model]
- `GenerateParams` [mod.rs:GenerateParams]
- `Generated` [mod.rs:Generated]
- `SampleSource` trait, `Greedy`, `Sampled::new(seed, top_p, top_k)` [mod.rs:SampleSource] [mod.rs:Greedy] [mod.rs:Sampled]
- `DiaConfig` (from_json), `DataConfig`, `ModelConfig` re-exports [config.rs:DiaConfig] [config.rs:DataConfig] [config.rs:ModelConfig]
- `REFERENCE_COMMIT`, `EXTRA_STEPS_AFTER_EOS` (30) [mod.rs:REFERENCE_COMMIT] [mod.rs:EXTRA_STEPS_AFTER_EOS]
- `Act` activation enum [layers.rs:Act]
- Crate-private: `apply_audio_delay`, `revert_audio_delay`, `codebook_to_audio`, `audio_to_codebook` [audio.rs:apply_audio_delay]; `Encoder`, `Decoder` [model.rs:Encoder] [model.rs:Decoder]

## Invariants
- delay_pattern length equals channels, checked at config parse and at load [config.rs:DataConfig::from_json] [mod.rs:Model::validate]
- kv_heads is positive and divides gqa_query_heads [mod.rs:Model::validate]
- Only float32 `training.dtype` is accepted [config.rs:DiaConfig::from_json]
- Logits keep codes 0..=1024; PAD and BOS are masked [mod.rs:KEEP_VOCAB]
- Self-cache capacity is prompt_frames + max_tokens + 1, overflow-checked; max_tokens must be positive [mod.rs:self_cache_capacity] [mod.rs:Model::generate_segment]
- KV store asserts on prefill or append past capacity [layers.rs:KvStore::prefill] [layers.rs:KvStore::update]
- Negative embedding ids wrap from the end; ids >= rows yield an empty row (zero contribution in the decoder sums, matches pinned MLX). The encoder copies the row into a fixed-width slice, so an out-of-range source byte id panics there [model.rs:embed_lookup]
- Masked attention uses additive -1e9 [mod.rs:MASK_DENIED]

## Side Effects
- Reads config.json and weights (first safetensors shard holds `encoder.*`/`decoder.*`) [mod.rs:Model::open] [mod.rs:Model::load]
- RNG: threefry key sequence from the caller seed in `Sampled` [mod.rs:Sampled::new]
- No env vars, threads, or logging [mod.rs:Model]

## Verification
- Run: `cargo test -p turbospark-audio tts::dia`
