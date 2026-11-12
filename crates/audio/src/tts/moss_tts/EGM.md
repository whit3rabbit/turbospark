# MOSS-TTS
## Overview
Port of mlx-audio `moss_tts` (pinned commit) for OpenMOSS MOSS-TTS, MOSS-TTSD and MOSS-TTS-Local: a Qwen3 backbone over `n_vq + 1` channels. Two variants: the delay pattern (`moss_tts_delay`) and the v1.5 local transformer (GPT-2 local stack). Decodes through `MossAudioTokenizer`. Batch size 1, non-streaming; prompt ids are caller-encoded.

## Interface
- `Model` (open, load, load_v15_local, generate, generate_delay_pattern_ids, generate_v15_local_ids, decode_generated_audio, sample_rate, variant) [mod.rs:Model]
- `ModelConfig`, `LanguageConfig`, `Variant` [mod.rs:ModelConfig] [mod.rs:Variant]
- `GenerationParams`, `DelayParams`, `V15Params`, `Generated` [mod.rs:GenerationParams]
- `apply_delay_pattern`, `apply_de_delay_pattern`, `normalize_audio_codes` [mod.rs:apply_delay_pattern]
- `Role`, `ContentPiece`, `DelayMessage`, `DelayMode`, `ChatTemplateIds` (prompt builders) [mod.rs:DelayMessage]
- `V15TemplateIds`, `V15PromptMode`, `LOCAL_V15_*` template consts, `normalize_template_value` [mod.rs:V15PromptMode]
- `DEFAULT_AUDIO_TOKENIZER_REPO`, `DEFAULT_AUDIO_TOKENIZER_V2_REPO`, `AUDIO_PLACEHOLDER` [mod.rs:DEFAULT_AUDIO_TOKENIZER_REPO]
- Internal: `LocalGpt2` [gpt2.rs:LocalGpt2]; sampling `apply_repetition_penalty`, `apply_top_k`, `apply_top_p`, `sample_token` [sampling.rs:sample_token] (module `sampling` is private)

## Invariants
- Legacy local-transformer checkpoints are refused with `Unsupported` at load [mod.rs:Model::open]
- `Model::load` refuses the v1.5 variant (needs two weight views) [mod.rs:Model::load]
- Prompt must be non-empty and every row exactly `n_vq + 1` wide [mod.rs:generate_delay_pattern_ids] [mod.rs:generate_v15_local_ids]
- Params must match the loaded variant, else `Input` error [mod.rs:Model::generate]
- Temperature 0 means greedy for that stream [mod.rs:DelayParams]
- Sampling order: repetition penalty, top-k, then top-p, then categorical; greedy never draws [sampling.rs:sample_token]
- Delay sentinel `INT64_MAX` marks "delay not engaged" [mod.rs:INT64_MAX]
- GPT-2 local stack requires `n_embd % n_head == 0` [gpt2.rs:LocalGpt2::load]

## Side Effects
- Reads `config.json` and safetensors weights from the checkpoint dir [mod.rs:Model::open]
- Sampling uses the crate-local threefry `KeySequence` seeded by `seed` [mod.rs:generate_delay_pattern_ids]
- No env vars, threads, or logging

## Verification
- Run: `cargo test -p turbospark-audio tts::moss_tts`
