# Speech-to-Text (STT) Models

This directory contains Speech-to-Text (STT) and speech-text alignment model implementations organized by model family.

The [central inventory](../../MODELS.md#stt) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[Whisper & Distil-Whisper](whisper/README.md)**: OpenAI Whisper encoder-decoder architecture with 80/128-band Slaney log-mel frontend and autoregressive greedy decoding.
- **[Moonshine](moonshine/README.md)**: Useful Sensors Moonshine encoder-decoder architecture optimized for resource-constrained edge transcription.
- **[Parakeet TDT](parakeet/README.md)**: FastConformer transducer with Token-and-Duration Transducer (TDT) decoding, supporting v2, v3, and Redux ternary quantization.
- **[Granite Speech 5.0 TurboCTC](granite_speech5_ctc/README.md)**: IBM Granite Speech encoder-only CTC model with precise HTK delta-feature frontend.
- **[MMS (Massively Multilingual Speech)](mms/README.md)**: Meta MMS Wav2Vec2/adapter acoustic model supporting 1,000+ languages.
- **[Qwen3-ASR](qwen3_asr/README.md)**: Qwen3 ASR speech-to-text architecture with grouped-query attention and RoPE.
- **[Qwen3-ForcedAligner](qwen3_forced_aligner/README.md)**: CTC-based forced alignment producing word- and character-level timestamps for English and Chinese.
- **[Nemotron ASR](nemotron_asr/README.md)**: NVIDIA Nemotron 3.5 ASR architecture with FastConformer encoder and RNN-T decoder.

## Layout and Conventions

Each model family owns its subfolder containing:
- Architecture and decoder modules
- Pinned profile configs and weight loaders
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
- Test suites asserting numeric parity against pinned MLX / Python reference golden tensors
