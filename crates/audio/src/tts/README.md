# Text-to-Speech (TTS) Models

This directory contains Text-to-Speech (TTS) synthesis model implementations organized by model family.

The [central inventory](../../MODELS.md#tts) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[Kokoro (82M)](kokoro/README.md)**: 82-million parameter StyleTTS 2 architecture with PLBert text encoder, AdaLayerNorm duration/prosody predictors, and high-fidelity 24 kHz iSTFTNet vocoder.

## Scope

Each model family owns its directory containing:
- Text normalization and phoneme intake interfaces
- Acoustic, prosody, and duration models
- Neural vocoder synthesis pipelines
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
