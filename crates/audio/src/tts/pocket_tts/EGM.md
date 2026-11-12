# Pocket TTS

## Overview
Port of Kyutai Pocket TTS: a streaming transformer flow-LM over a latent space with an AdaLN MLP flow net and Lagrangian self-distillation (LSD) decode. It produces continuous latents (not waveforms); the Mimi side is supplied by callers.

## Interface
- `Model` with `open`, `load`, `start_state`, `start_state_capture`, `generate_latents`, `generate_latents_with_noise`, `generate_latents_with_trace` [mod.rs:Model]
- `ModelConfig`, `FlowLMConfig`, `FlowConfig`, `TransformerConfig`, `MimiConfig`, `QuantizerConfig`, `LookupTableConfig` [mod.rs:ModelConfig]
- `GenerateParams`, `Generated`, `StopReason` [mod.rs:GenerateParams]
- `FlowLm`, `LutConditioner`, `StreamingTransformer`, `TimestepEmbedder`, `SimpleMlpAdaln` [mod.rs:FlowLm]
- `lsd_decode`, `rms_norm_ddof1`, `prepare_text_prompt` [mod.rs:lsd_decode]
- Consts `REFERENCE_COMMIT`, `DEFAULT_TEMPERATURE`, `DEFAULT_LSD_DECODE_STEPS`, `DEFAULT_EOS_THRESHOLD`, `DEFAULT_AUDIO_PROMPT` [mod.rs:DEFAULT_TEMPERATURE]

## Invariants
- Zero heads rejected at config parse (head_dim = d_model / heads) [mod.rs:ModelConfig::from_json]
- `lsd_decode_steps` must be > 0 for generation [mod.rs:Model::generate_latents_inner]
- Conditioning length must be a multiple of d_model [mod.rs:Model::start_state_capture]
- Noise draw width must equal the latent dim [mod.rs:Model::generate_latents_inner]
- Step 0 input is the bos embedding in place of the reference NaN seed [mod.rs:Model::generate_latents_inner]
- Empty text prompt refused [mod.rs:prepare_text_prompt]

## Side Effects
- Reads `config.json` and flow-LM weights only (not the Mimi file) [mod.rs:Model::open]
- Allocates KV caches per generation [mod.rs:FlowLm::make_cache]
- Noise is drawn through a caller-supplied closure [mod.rs:Model::generate_latents_inner]
- Env vars, threads, logging: None

## Verification
- Run: `cargo test -p turbospark-audio tts::pocket_tts`
