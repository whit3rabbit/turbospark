# Chatterbox TTS
## Overview
Voice-clone TTS over the S3 speech-token space, ported from mlx-audio `chatterbox` at a pinned commit [mod.rs:REFERENCE_COMMIT]. It chains a Llama-based T3 speech-token generator with classifier-free guidance, an S3Gen flow plus HiFT vocoder producing 24 kHz audio, a CAMPPlus speaker encoder, and an LSTM voice encoder. It is single-batch and non-streaming, text tokenization and resampling are caller-owned.

## Interface
- `Model` (open, load, with_s3_tokenizer, prepare_text_tokens, prepare_conditionals, generate) [mod.rs:Model]
- `ModelConfig` (from_json, Default) [mod.rs:ModelConfig]
- `Conditionals`, `GenCond`, `Generated`, `GenerateParams`, `GenerateError` [mod.rs:Conditionals] [mod.rs:GenCond] [mod.rs:Generated] [mod.rs:GenerateParams] [mod.rs:GenerateError]
- `punc_norm`, `drop_invalid_tokens` [mod.rs:punc_norm] [mod.rs:drop_invalid_tokens]
- consts `REFERENCE_COMMIT`, `S3_SR`, `S3GEN_SR`, `SPEECH_VOCAB_SIZE` [mod.rs:S3_SR] [mod.rs:S3GEN_SR] [mod.rs:SPEECH_VOCAB_SIZE]
- T3: `T3`, `T3Config`, `T3Cond`, `T3GenParams`, `StopReason`, `llama_520m_config` [t3.rs:T3] [t3.rs:T3Config] [t3.rs:T3Cond] [t3.rs:T3GenParams] [t3.rs:StopReason] [t3.rs:llama_520m_config]
- Flow: `ChatterboxFlow` (load, infer, up_rate, n_timesteps), `FlowConfig` [flow.rs:ChatterboxFlow] [flow.rs:FlowConfig]
- Vocoder: `Hift` (load, predict_f0, infer), `HiftConfig`, `SineDraws` [hift.rs:Hift] [hift.rs:HiftConfig] [hift.rs:SineDraws]
- Speaker encoders: `CampPlus`, `CampPlusConfig`, `kaldi_fbank` [campplus.rs:CampPlus] [campplus.rs:kaldi_fbank], `VoiceEncoder`, `VoiceEncConfig`, `melspectrogram`, `trim_silence` [voice_encoder.rs:VoiceEncoder] [voice_encoder.rs:melspectrogram] [voice_encoder.rs:trim_silence]
- crate-private helpers: `SaLinear`, `SaConv1d`, `SaConvTr1d`, `SaLayerNorm`, `prompt_mel` [s3gen.rs:prompt_mel] (reused by `chatterbox_turbo`), `UpsampleConformerEncoder` [conformer.rs:UpsampleConformerEncoder]
- Child dirs: none.

## Invariants
- A checkpoint dir needs `model.safetensors`, otherwise BadConfig [mod.rs:checkpoint_file].
- Submodule weight prefixes are `t3.`, `flow`, `mel2wav`, `speaker_encoder.`, `ve.` [mod.rs:Model::load].
- Text prompt is `[start_text_token] ids [stop_text_token]` [mod.rs:Model::prepare_text_tokens].
- `prepare_conditionals` requires the S3 tokenizer, else Input error [mod.rs:Model::prepare_conditionals].
- Prompt mel frames are kept at twice the prompt token count, the longer side is truncated [mod.rs:Model::prepare_conditionals].
- Speech tokens are the span between the first 6561 and first 6562, ids at or above 6561 are dropped, an empty result is an Input error [mod.rs:drop_invalid_tokens] [mod.rs:Model::generate].
- T3 requires `encoder_type == "voice_encoder"` and a non-empty text prompt [t3.rs:T3CondEnc::load] [t3.rs:T3::inference].
- Flow rejects a speaker embedding whose length differs from `spk_embed_dim`, and a `prompt_feat` whose length is not a multiple of the mel width [flow.rs:ChatterboxFlow::infer].
- HiFT requires phase length 9 and noise length `9 * t_wav` when draws are given, `None` means zero phase [hift.rs:Hift::infer].
- Output gets the first 480 samples zeroed and the next 480 faded in [hift.rs:trim_fade] [mod.rs:Model::generate].
- Hardcoded vocoder geometry (upsample rates 8,5,3, iSTFT n_fft 16, hop 4, 480 samples per mel frame) [hift.rs:UPSAMPLE_RATES] [hift.rs:ISTFT_N_FFT] [hift.rs:UPSCALE].

## Side Effects
- Reads `config.json`, `model.safetensors`, and optional `s3_tokenizer.safetensors` plus `s3_tokenizer_config.json` from the checkpoint dir [mod.rs:Model::open].
- Reads `config.json` and `model.safetensors` from a separate S3 tokenizer dir [mod.rs:Model::with_s3_tokenizer].
- RNG: T3 categorical draws use the crate-local MLX-compatible threefry `KeySequence` seeded from `GenerateParams::seed` [mod.rs:Model::generate].
- Env var `TURBOSPARK_DEBUG_CONV`, when set, makes conv forward print to stderr [s3gen.rs:SaConv1d::forward].
- Threads: None found.

## Verification
- Run: `cargo test -p turbospark-audio tts::chatterbox`
