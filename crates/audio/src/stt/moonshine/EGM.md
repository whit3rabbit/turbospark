# Moonshine

## Overview
Moonshine is a pure CPU f32 port of Useful Sensors' lightweight English ASR: a three-conv front end, a bidirectional encoder with partial interleaved RoPE, and a causal decoder with cross-attention and a gated SiLU MLP, decoded greedily [mod.rs:Moonshine]. A small decode-only tokenizer reads `tokenizer.json` [mod.rs:Tokenizer]. Config is parsed strictly with reference defaults [mod.rs:MoonshineConfig].

## Interface
- `MoonshineConfig` with `from_json` [mod.rs:MoonshineConfig]
- `Moonshine` with `open`, `encode`, `transcribe` [mod.rs:Moonshine]
- `Tokenizer` with `load`, `decode` [mod.rs:Tokenizer]

## Invariants
- Required integer config fields (vocab, hidden, intermediate, layer counts, head and kv-head counts) must be present and fit usize, otherwise `BadConfig` [mod.rs:MoonshineConfig::from_json]
- `decoder_start_token_id` and `eos_token_id`, when present, must fit u32; defaults are 1 and 2 [mod.rs:MoonshineConfig::from_json]
- Heads must divide `hidden_size`, kv heads must divide heads, `partial_rotary_factor` in (0, 1], `rope_theta` finite positive, token ids below `vocab_size`, tensor sizes checked against allocation overflow [mod.rs:MoonshineConfig::validate]
- `encode` requires at least 895 finite samples (127/64, 7/3, 3/2 unpadded conv stack) [mod.rs:Moonshine::encode]
- Rotary dims are `head_dim * partial_rotary_factor` rounded down to even [mod.rs:Moonshine::encode] [mod.rs:Moonshine::decode_step]
- Tensor names accept both `model.`-prefixed and stripped forms; `proj_out.weight` is optional and otherwise the tied embedding is the output head [mod.rs:Moonshine::open] [mod.rs:Moonshine::transcribe]
- Conv weights are consumed in PyTorch `[out, in, kernel]` layout with no conversion [mod.rs:Moonshine::frontend]
- The decoder keeps one self-attention KV buffer per layer; sharing one corrupts later layers. Cross-attention K/V are projected once and cached [mod.rs:DecodeCache] [mod.rs:Moonshine::decode_step]
- `final_layernorm` runs before the MLP in each decoder layer [mod.rs:Moonshine::decode_step]
- Greedy decode stops at `eos_token_id` or after 448 steps [mod.rs:Moonshine::transcribe]
- Tokenizer load fails when no `model.vocab` or added-token entries exist; decode skips special ids and resolves `<0xNN>` byte fallback [mod.rs:Tokenizer::load] [mod.rs:Tokenizer::decode]

## Side Effects
- Reads `config.json`, `model.safetensors`, and `tokenizer.json` from the model directory via `SafetensorsFile::open` and eagerly loads all weights as f32 [mod.rs:Moonshine::open] [mod.rs:Tokenizer::load]
- No env vars, threads, global statics, or GPU use here. The Metal path is a separate sibling [crates/runtime/src/moonshine.rs:MoonshineMetalEngine]

## Verification
- Run: `cargo test -p turbospark-audio stt::moonshine`
