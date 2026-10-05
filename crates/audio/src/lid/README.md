# Spoken Language Identification (LID) Models

This directory contains spoken language identification (LID) model families.

The [central inventory](../../MODELS.md#lid) tracks upstream source families, checkpoint profiles, and verification status.

## Target Families

- **ECAPA-TDNN**: Time Delay Neural Network with squeeze-and-excitation for robust utterance-level speaker and language identification.
- **Wav2Vec2 LID**: Pre-trained self-supervised acoustic representations fine-tuned on multilingual language identification benchmarks.

## Scope

Each model family owns its directory containing:
- Feature extraction and acoustic frame processing
- Classification head and language probability mapping
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
