# Echo-TTS
## Overview
Echo-TTS is a flow-matching DiT over Fish S1 DAC continuous latents with PCA compression and independent text/speaker classifier-free guidance. It ports `mlx_audio/tts/models/echo_tts/` at mlx-audio e1b19b90, single batch, non-streaming, portable f32.

## Interface
- `Model` (`open`, `open_with_config`, `set_codec`, `waveform_ready`, `generate_latents`, `generate`) [mod.rs:Model]
- `Generated`, `GenerateControls::from_config` [mod.rs:Generated] [mod.rs:GenerateControls]
- `ModelConfig::from_json`, `EchoDiTConfig`, `SamplerConfig` [config.rs:ModelConfig] [config.rs:EchoDiTConfig] [config.rs:SamplerConfig]
- `EchoDiT` (`get_kv_cache_text`, `get_kv_cache_speaker`, `get_kv_cache_latent`, `forward`), `KvCache`, `RotaryCache` [dit.rs:EchoDiT] [dit.rs:KvCache] [dit.rs:RotaryCache]
- `precompute_freqs_cis`, `timestep_embedding` [dit.rs:precompute_freqs_cis] [dit.rs:timestep_embedding]
- `GaussianNoise`, `SeededGaussian`, `SamplerParams`, `t_schedule`, `temporal_score_rescale` [sampler.rs:GaussianNoise] [sampler.rs:SeededGaussian] [sampler.rs:t_schedule]
- `sample_euler_cfg_independent_guidances`, `sample_blockwise_euler_cfg_independent_guidances`, `DEFAULT_TRUNCATION_FACTOR` [sampler.rs:sample_euler_cfg_independent_guidances] [sampler.rs:DEFAULT_TRUNCATION_FACTOR]
- `PcaState::open`, `ae_encode`, `ae_decode`, `find_flattening_point`, `crop_audio_to_flattening_point`, `get_speaker_latent_and_mask` [audio.rs:PcaState] [audio.rs:ae_decode]
- `normalize_text_prompt`, `tokenizer_encode`, `TextBatch`, `get_text_input_ids_and_mask` [text.rs:get_text_input_ids_and_mask]

## Invariants
- `generate` refuses to run unless the Fish S1 DAC and PCA state are installed [mod.rs:Model::generate]
- Blockwise generation is refused when `delete_blockwise_modules` is set [mod.rs:Model::generate_latents]
- `model_size` must divide by `num_heads` (and the text/speaker sizes likewise) at config load [config.rs:EchoDiTConfig::valid]
- Initial noise is drawn through the injected `GaussianNoise` in reference order, scaled by the truncation factor (default 0.96) [sampler.rs:sample_euler_cfg_independent_guidances] [sampler.rs:DEFAULT_TRUNCATION_FACTOR]
- Guidance KV caches are tripled (text, speaker) via `concat_kv` for the independent-CFG batch [sampler.rs:sample_euler_cfg_independent_guidances]
- Masked attention positions use a -1e9 fill [dit.rs:MASK_VALUE]
- Without a speaker latent a zero one-patch latent with an all-false mask is used [mod.rs:Model::generate_latents]

## Side Effects
- Reads `config.json`, `model.safetensors` (prefix `model.`) and the PCA file named by `pca_filename` when it exists [mod.rs:Model::open_with_config] [audio.rs:PcaState::open]
- RNG: `SeededGaussian` uses the crate threefry key from `music::minimax_music3::rng` [sampler.rs:SeededGaussian]
- No env vars, threads or logging found in non-test code.

## Verification
- Run: `cargo test -p turbospark-audio tts::echo_tts`
