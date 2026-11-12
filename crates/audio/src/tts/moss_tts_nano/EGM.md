# MOSS-TTS-Nano

## Overview
Port of the MOSS-TTS-Nano text-to-speech model: a GPT-2 global transformer plus a local GPT-2 transformer that emits n_vq audio codebook tokens per frame. Prompt construction, the generation loop and family-specific sampling are here; waveform decoding goes through the MOSS audio tokenizer codec.

## Interface
- `Model` with `open`, `load`, `generate`, `generate_audio_token_ids`, `build_inference_input_ids`, `resolve_nq` [mod.rs:Model]
- `ModelConfig`, `Gpt2Config` with `from_json` [mod.rs:ModelConfig] [mod.rs:Gpt2Config]
- `GenerateParams`, `Generated`, `StopReason`, `PromptMode`, `TemplateIds` [mod.rs:GenerateParams]
- Prompt helpers `build_user_prompt_prefix`, `build_user_prompt_after_reference`, `build_assistant_prompt_prefix`, `build_prompt_token_ids`, `lightweight_normalize_text`, `frames_to_codes` [mod.rs:build_prompt_token_ids]
- Template and token consts `DEFAULT_AUDIO_TOKENIZER_REPO`, `DEFAULT_PAD_TOKEN_ID`, `USER_TEMPLATE_SUFFIX`, etc. [mod.rs:DEFAULT_AUDIO_TOKENIZER_REPO]
- Sampling (private module `sampling`): `apply_repetition_penalty`, `apply_top_k`, `apply_top_p`, `sample_next_token`, `sample_assistant_text_token` [sampling.rs:sample_next_token]

## Invariants
- A zero head count or unsupported config is refused at load, not at first forward [mod.rs:Gpt2Config::from_json]
- Prompt rows must be n_vq + 1 wide and non-empty [mod.rs:Model::generate_audio_token_ids]
- `do_sample=false` is deterministic; `seed` only feeds the draws [mod.rs:Model::generate]
- Non-positive temperature is refused when sampling [sampling.rs:sample_next_token]
- Out-of-range token ids are refused in embedding lookup [mod.rs:lookup_row]

## Side Effects
- Reads `config.json` and safetensors weights from the checkpoint dir [mod.rs:Model::open]
- RNG: threefry `KeySequence` seeded from the caller [mod.rs:Model::generate_audio_token_ids]
- Env vars, threads, logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::moss_tts_nano`
