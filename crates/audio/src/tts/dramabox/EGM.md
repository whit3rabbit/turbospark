# DramaBox TTS
## Overview
DramaBox (LTX-2.3 audio-only DiT, flow matching), port of mlx-audio `dramabox` zero-reference path, batch 1, f32. Runs a guided Euler loop over the LTX2 sigma schedule on patchified mel latents. The pipeline ends at mel latents: AudioVAE decoder and vocoder are refused, and Gemma-3 hidden states are caller-supplied [mod.rs:decode_waveform_refusal] [mod.rs:Model::encode_context].

## Interface
- `Model` (open, sample_rate, tokens_for_duration, encode_context, generate) [mod.rs:Model]
- `GenerateParams` (defaults cfg 2.5, stg 1.5, stg_block 29, steps 30) [mod.rs:GenerateParams]
- `Generated` {latents, channels, frames, mel_bins, sample_rate} [mod.rs:Generated]
- `DiffusionNoise` trait, `SeededNoise::new(seed)` [mod.rs:DiffusionNoise] [mod.rs:SeededNoise]
- `ModelConfig` (from_json), `TransformerConfig`, `AudioConfig`, `InferenceDefaults` [mod.rs:ModelConfig] [mod.rs:TransformerConfig] [mod.rs:AudioConfig] [mod.rs:InferenceDefaults]
- `estimate_nonverbal_duration`, `estimate_speech_duration`, `resolve_generation_duration`, `aligned_frame_count` [mod.rs:estimate_speech_duration] [mod.rs:resolve_generation_duration] [mod.rs:aligned_frame_count]
- `decode_waveform_refusal` [mod.rs:decode_waveform_refusal]
- Crate-private: `TextConditioner`, `ltx2_sigmas`, `euler_step`, `TransformerBlock`, `load_sharded` [conditioning.rs:TextConditioner] [diffusion.rs:euler_step] [dit.rs:TransformerBlock] [sharded.rs:load_sharded]

## Invariants
- Waveform decode is unsupported and returns `SpeechError::Unsupported` [mod.rs:decode_waveform_refusal]
- audio_in_channels must equal vae_channels * mel_bins, checked after denoising at unpatchify rather than up front [mod.rs:Model::generate]
- Sigma 0 is refused in velocity conversion [diffusion.rs:to_velocity]
- Plain guidance (cfg 1, stg 0, modality 1) runs only the conditional pass; STG pass skips self-attention of `stg_block` [mod.rs:Model::generate] [dit.rs:TransformerBlock::forward]
- Connector heads * head_dim must equal the cross dim; token count must divide by register count; hidden states must cover tokens * hidden * layers [conditioning.rs:TextConditioner::load] [conditioning.rs:Embeddings1DConnector] [conditioning.rs:TextConditioner::forward]
- RoPE dims must match max_pos and inner_dim must divide by 2*dims [dit.rs:split_freqs_cis]
- Tensors load from the first shard holding them with an exact shape check [sharded.rs:load_sharded]

## Side Effects
- Reads config.json and sharded safetensors weights [mod.rs:Model::open]
- RNG: threefry key sequence from the seed in `SeededNoise` [mod.rs:SeededNoise::new]
- No env vars, threads, or logging [mod.rs:Model]

## Verification
- Run: `cargo test -p turbospark-audio tts::dramabox`
