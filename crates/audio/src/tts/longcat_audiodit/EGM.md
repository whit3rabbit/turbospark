# LongCat-AudioDiT TTS
## Overview
Port of meituan-longcat LongCat-AudioDiT from mlx-audio: conditional flow matching over a latent space. A frozen UMT5 text encoder conditions a CrossDiT transformer; an Euler ODE integrates from noise with classifier-free guidance (or APG) against a zero-text null branch, and a WAV-VAE decodes the latent to 24 kHz audio. Scope is batch 1, non-streaming, with caller-owned tokenization [mod.rs:Model].

## Interface
- `Model::open`, `Model::load`, `Model::sample_rate`, `Model::encode_prompt_audio`, `Model::generate`; pub field `config` [mod.rs:Model]
- `GenerateParams` (steps 16, cfg_strength 4.0, guidance_method "cfg" or "apg") and `Generated` [mod.rs:GenerateParams] [mod.rs:Generated]
- `ModelConfig::from_json`, `VaeConfig`, `TextEncoderConfig` [mod.rs:ModelConfig] [mod.rs:VaeConfig] [mod.rs:TextEncoderConfig]
- `Noise` trait and `SeededNoise::new(seed)` [mod.rs:Noise] [mod.rs:SeededNoise]
- `normalize_text`, `approx_duration`, `prompt_text` [mod.rs:normalize_text] [mod.rs:approx_duration] [mod.rs:prompt_text]
- `REFERENCE_COMMIT`, `EN_DUR_PER_CHAR`, `ZH_DUR_PER_CHAR` [mod.rs:REFERENCE_COMMIT]
- Private submodules: `AudioDiTTransformer` [dit.rs:AudioDiTTransformer], `Umt5Encoder` [text_encoder.rs:Umt5Encoder], `AudioDiTVae` [vae.rs:AudioDiTVae]

## Invariants
- `steps` must be at least 2, otherwise raw noise would be returned as audio [mod.rs:check_steps]
- A voice prompt must be shorter than the max-frame cap (`max_wav_duration * sr / hop`) [mod.rs:check_prompt_fits]
- Total duration is clamped to the frame cap; zero frames is an error [mod.rs:Model::generate]
- `vae_config.c_mults` must be nonempty and `strides` the same length [mod.rs:VaeConfig::validate]
- Prompt audio is padded to a hop multiple plus 3 hops, and 3 tail frames are dropped [mod.rs:Model::encode_prompt_audio]
- Noise draw order: VAE draw for a voice prompt first, then initial latent noise; prompt frames are re-interpolated from the original noise each step [mod.rs:Noise] [mod.rs:Model::generate]
- Guidance is skipped when `cfg_strength < 1e-5`; APG guides only the non-prompt slice with momentum -0.3 [mod.rs:Model::generate]
- Mask length must match input ids; token ids must be nonnegative and inside the table [text_encoder.rs:check_mask_len] [text_encoder.rs:Umt5Encoder::forward]
- Tensors must exist with expected shapes at load [dit.rs] [vae.rs] [text_encoder.rs] (each file's tensor loader returns `SpeechError::Tensor`)

## Side Effects
- Reads `config.json` and safetensors from the checkpoint directory; tensors under `text_encoder.`, `transformer.`, `vae.` [mod.rs:Model::open] [mod.rs:Model::load]
- `SeededNoise` draws from the crate-local threefry RNG [mod.rs:SeededNoise]
- `generate` takes `&mut self` [mod.rs:Model::generate]
- No environment variables are read and nothing is printed. No threads.

## Verification
- Run: `cargo test -p turbospark-audio tts::longcat_audiodit`
