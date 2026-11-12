# OmniVoice

## Overview
Port of the k2-fsa OmniVoice masked-diffusion TTS: a Qwen3 backbone with fused per-codebook audio embeddings and heads, decoded by iterative unmasking with classifier-free guidance. The pipeline wraps the model with a text tokenizer and the Higgs audio tokenizer codec. The private `duration` module (`duration.rs`) holds the rule-based `estimate_duration`, which scales the reference clip length by relative character weights, with a boost for very short estimates [duration.rs:estimate_duration].

## Interface
- `Model` with `open`, `open_with_config`, `forward`, `prepare_inference_inputs`, `iterative_unmask`, `generate`, `with_backend`, `set_backend`, `sample_rate`, `tokens_per_sec`, `target_len_for` [mod.rs:Model]
- `OmniVoiceConfig` with `from_json` [mod.rs:OmniVoiceConfig]
- Text helpers `combine_text`, `build_style_text`, `build_wrapped_text` [mod.rs:combine_text]
- `time_steps` (pub), `cfg_log_probs` (pub(crate)) [mod.rs:time_steps]
- `pipeline` module: `OmniVoicePipeline` (`open`, `encode_reference`, `encode_prompt`, `estimate_target_len`, `generate`), `GenerateParams` [pipeline.rs:OmniVoicePipeline]

## Invariants
- `num_steps` of 0 is refused (would return an all-mask target) [mod.rs:Model::iterative_unmask]
- `target_len` may not exceed the sequence length (target is the tail block) [mod.rs:Model::iterative_unmask]
- `input_ids` must cover len x num_audio_codebook entries [mod.rs:Model::forward]
- Config must carry `llm_config.hidden_size` [mod.rs:Model::open_with_config]
- Non-verbal tags are tokenized atomically [pipeline.rs:NONVERBAL_TAGS]

## Side Effects
- Reads `config.json`, `model.safetensors` [mod.rs:Model::open] [mod.rs:Model::open_with_config]
- Pipeline also reads `tokenizer.json` and the `audio_tokenizer/` dir [pipeline.rs:OmniVoicePipeline::open]
- RNG: threefry uniform draws keyed by `seed + step` when no recorded draws are supplied [mod.rs:Model::iterative_unmask]
- Optional compute backend held via `Rc` [mod.rs:Model::set_backend]
- Env vars, threads, logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::omnivoice`
