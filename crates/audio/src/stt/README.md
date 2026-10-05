# Speech-to-Text (STT) Models

This directory contains Speech-to-Text (STT) and speech-text alignment model implementations organized by model family.

The [central inventory](../../MODELS.md#stt) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[Whisper & Distil-Whisper](whisper/README.md)**: OpenAI Whisper encoder-decoder architecture with 80/128-band Slaney log-mel frontend and autoregressive greedy decoding.
- **[Moonshine](moonshine/README.md)**: Useful Sensors Moonshine encoder-decoder architecture optimized for resource-constrained edge transcription.
- **[Parakeet TDT](parakeet/README.md)**: FastConformer transducer with Token-and-Duration Transducer (TDT) decoding, supporting v2, v3, and Redux ternary quantization.
- **[SenseVoice Small](sensevoice/README.md)**: 50-block SANM CTC recognizer with language, emotion, and event tags.
- **[FireRedASR2-AED](fireredasr2/README.md)**: 16-block Conformer encoder and Transformer beam-search decoder for English ASR.
- **[Fun-ASR-Nano-2512](fun_asr_nano/README.md)**: SANM speech encoder and audio adaptor with a tied Qwen3 0.6B decoder.
- **[GLM-ASR-Nano-2512](glmasr/README.md)**: Whisper encoder with RoPE, merge-four audio adaptor, and affine 4-bit Llama decoder.
- **[Granite Speech 5.0 TurboCTC](granite_speech5_ctc/README.md)**: IBM Granite Speech encoder-only CTC model with precise HTK delta-feature frontend.
- **[MMS (Massively Multilingual Speech)](mms/README.md)**: Meta MMS Wav2Vec2/adapter acoustic model supporting 1,000+ languages.
- **[Qwen3-ASR](qwen3_asr/README.md)**: Qwen3 ASR speech-to-text architecture with grouped-query attention and RoPE.
- **[Mega-ASR](mega_asr/README.md)**: routed robust ASR over the shared Qwen3-ASR backbone: an audio-quality router switches between the base decode path and a LoRA-adapted robust path, with an always-on-robust pre-merged 8-bit profile.
- **[Qwen3-ForcedAligner](qwen3_forced_aligner/README.md)**: CTC-based forced alignment producing word- and character-level timestamps for English and Chinese.
- **[Nemotron ASR](nemotron_asr/README.md)**: NVIDIA Nemotron 3.5 ASR architecture with FastConformer encoder and RNN-T decoder.
- **[Canary](canary/README.md)**: NVIDIA Canary-1B-v2 multilingual ASR and translation with a FastConformer encoder, cross-attention transformer decoder, and prompt-driven language/punctuation control.

## Layout and Conventions

Each model family owns its subfolder containing:
- Architecture and decoder modules
- Pinned profile configs and weight loaders
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
- Test suites asserting numeric parity against pinned MLX / Python reference golden tensors
