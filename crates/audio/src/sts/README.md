# Speech-to-Speech (STS) and Speech Enhancement Models

This directory contains Speech-to-Speech (STS), voice conversion, and neural speech enhancement model implementations.

The [central inventory](../../MODELS.md#sts) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[DeepFilterNet (DFN)](deepfilternet/README.md)**: Full-band 48 kHz speech enhancement and noise suppression framework combining ERB spectrogram gain masking and deep complex linear filtering.

## Scope

Each model family owns its directory containing:
- STFT domain feature extraction (ERB filterbanks, Vorbis windowing)
- Encoder and mask/filtering decoders
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
