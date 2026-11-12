# Chatterbox Turbo TTS
## Overview
Distilled voice-clone TTS ported from mlx-audio `chatterbox_turbo` at a pinned commit [mod.rs:REFERENCE_COMMIT]. A GPT-2 based T3 generates S3 speech tokens, a meanflow S3Gen flow (two Euler steps, no classifier-free guidance) and HiFT vocoder render 24 kHz audio, with a Kaldi-fbank CAMPPlus speaker encoder and an LSTM voice encoder. It is single-batch and non-streaming, tokenization and resampling are caller-owned. It reuses the sibling `chatterbox` conv/linear layers and `prompt_mel` [mod.rs:Model::prepare_conditionals].

## Interface
- `Model` (open, load, with_s3_tokenizer, prepare_text_tokens, prepare_conditionals, generate) [mod.rs:Model]
- `ModelConfig` (from_json, Default) [mod.rs:ModelConfig]
- `Conditionals`, `GenCond`, `Generated`, `GenerateParams` [mod.rs:Conditionals] [mod.rs:GenCond] [mod.rs:Generated] [mod.rs:GenerateParams]
- `punc_norm` [mod.rs:punc_norm]
- consts `REFERENCE_COMMIT`, `S3_SR`, `S3GEN_SR`, `SPEECH_VOCAB_SIZE`, `S3GEN_SIL` [mod.rs:S3GEN_SIL] [mod.rs:SPEECH_VOCAB_SIZE]
- T3: `T3`, `T3Config`, `T3Cond`, `T3GenParams`, `gpt2_medium_config`, `apply_repetition_penalty`, `apply_top_k_threshold`, `apply_top_p_shift`, `inference_turbo` [t3.rs:T3] [t3.rs:T3GenParams] [t3.rs:gpt2_medium_config] [t3.rs:apply_top_k_threshold]
- Flow: `S3GenFlow` (load, n_timesteps, speaker_embed, infer), `FlowConfig`, `FlowNoise` [flow.rs:S3GenFlow] [flow.rs:FlowConfig] [flow.rs:FlowNoise]
- Vocoder: `Hift`, `HiftConfig`, `SineDraws` [hift.rs:Hift] [hift.rs:HiftConfig] [hift.rs:SineDraws]
- Front ends: `kaldi_fbank`, `mean_normalized_fbank` [fbank.rs:kaldi_fbank] [fbank.rs:mean_normalized_fbank], `CampPlus`, `CampPlusConfig` [campplus.rs:CampPlus], `VoiceEncoder`, `VoiceEncConfig` [voice_encoder.rs:VoiceEncoder]
- crate-private: `UpsampleConformerEncoder` [conformer.rs:UpsampleConformerEncoder]
- Child dirs: none.

## Invariants
- A checkpoint dir needs `model.safetensors`, otherwise BadConfig [mod.rs:checkpoint_file].
- Text ids pass to the T3 verbatim, with no start/stop wrapping [mod.rs:Model::prepare_text_tokens].
- Weight prefixes: `t3.`, `s3gen`, `s3gen.mel2wav`, `ve.` [mod.rs:Model::load].
- `prepare_conditionals` requires the S3 tokenizer, else Input error [mod.rs:Model::prepare_conditionals].
- The 16 kHz S3Gen band is trimmed to `dec_cond_len * s3_sr / s3gen_sr` samples [mod.rs:Model::prepare_conditionals].
- Speech tokens are all ids below 6561 in stream order, followed by three 4299 silence tokens, so the list is never empty [mod.rs:Model::sanitize_speech_tokens].
- `inference_turbo` rejects an empty text prompt and drops a trailing stop token [t3.rs:T3::inference_turbo].
- `temperature == 0` is greedy argmax and never draws from the RNG [t3.rs:T3::sample].
- `T3::load` requires `encoder_type == "voice_encoder"` [t3.rs:T3::load].
- Flow errors unless encoder frames equal prompt frames plus `tokens * upsample_stride` [flow.rs:S3GenFlow::infer], meanflow defaults to 2 timesteps [flow.rs:FlowConfig].
- HiFT with no `SineDraws` uses zero phase and zero noise [hift.rs:Hift::sine_gen], output gets the sibling's 20 ms fade [mod.rs:Model::generate].

## Side Effects
- Reads `config.json`, `model.safetensors`, and optional `s3_tokenizer.safetensors` plus `s3_tokenizer_config.json` [mod.rs:Model::open].
- Reads `config.json` and `model.safetensors` from a separate S3 tokenizer dir [mod.rs:Model::with_s3_tokenizer].
- RNG: crate-local threefry `KeySequence` seeded from `GenerateParams::seed` feeds T3 draws and, when `flow_noise` is `None`, the flow start noise [mod.rs:Model::generate] [flow.rs:S3GenFlow::infer].
- Env vars, threads, logging: None found in this directory's non-test files.

## Verification
- Run: `cargo test -p turbospark-audio tts::chatterbox_turbo`
