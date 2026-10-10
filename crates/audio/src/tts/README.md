# Text-to-Speech (TTS) Models

This directory contains Text-to-Speech (TTS) synthesis model implementations organized by model family.

The [central inventory](../../MODELS.md#tts) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[BailingMM](bailingmm/README.md)**: Ant Group multimodal speech language model for streaming speech synthesis.
- **[Breeze-TTS](breeze_tts/README.md)**: MediaTek Research Mandarin and English autoregressive speech synthesis model.
- **[Higgs Audio](higgs_audio/README.md)**: Discrete audio tokenizer acoustic generator with streaming RVQ heads.
- **[Higgs Audio v3](higgs_audio_v3/README.md)**: Next-generation Higgs Audio architecture with improved prosody modeling.
- **[Index-TTS](indextts/README.md)**: Bilibili zero-shot voice cloning autoregressive text-to-speech model.
- **[KittenTTS](kitten_tts/README.md)**: Lightweight StyleTTS 2 architecture with an ALBERT text encoder and iSTFTNet vocoder.
- **[Kokoro (82M)](kokoro/README.md)**: 82-million parameter StyleTTS 2 architecture with PLBert text encoder, AdaLayerNorm duration/prosody predictors, and high-fidelity 24 kHz iSTFTNet vocoder.
- **[KugelAudio](kugelaudio/README.md)**: Hybrid autoregressive Qwen2.5 backbone with VibeVoice diffusion head under SDE-DPM-Solver++.
- **[Llama-TTS](llama/README.md)**: Llama-based autoregressive acoustic model paired with SNAC neural codec.
- **[MegaTTS3](megatts3/README.md)**: ByteDance zero-shot voice-cloning TTS (latent diffusion transformer over WavVAE latents); staged port, the whole pipeline is ported (`MegaTts3`) and gated against the reference on real weights.
- **[MOSS-TTS](moss_tts/README.md)**: OpenLMLab dual-stream speech synthesis model with delay-pattern AR backbone over MOSS audio tokens.
- **[MOSS-TTS Nano](moss_tts_nano/README.md)**: Compact low-latency MOSS-TTS variant optimized for on-device inference.
- **[OuteTTS](outetts/README.md)**: Autoregressive transformer modeling audio tokens with WavTokenizer and prompt conditioning.
- **[Qwen3-TTS](qwen3/README.md)**: Alibaba Qwen3 speech synthesis model integrating 12 Hz discrete audio tokens.
- **[Rumik-OSS](rumik_oss/README.md)**: Open-source speech synthesis model utilizing causal masked prediction.
- **[Sesame (CSM / MisoTTS)](sesame/README.md)**: Continuous speech model with Multi-codebook interleaved streaming architecture.
- **[Soprano](soprano/README.md)**: Fast multi-speaker acoustic model with explicit pitch and energy predictors.
- **[Spark-TTS](spark/README.md)**: Unified voice cloning and generation architecture with single-codebook BiCodec tokens.
- **[Zonos-v0.2](zonos2/README.md)**: Zyphra autoregressive acoustic model operating over 9-codebook DAC neural codec tokens.

## Scope

Each model family owns its directory containing:
- Text normalization and phoneme intake interfaces
- Acoustic, prosody, and duration models
- Neural vocoder synthesis pipelines
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
