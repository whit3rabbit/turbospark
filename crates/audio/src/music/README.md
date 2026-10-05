# Music Generation Models

This directory contains neural music and audio generation model families.

The [central inventory](../../MODELS.md#music) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[MiniMax Music 0.5 (`minimax_music3`)](minimax_music3/README.md)**: Hierarchical autoregressive music generation with a flow-matching Diffusion Transformer (DiT) latent decoder and a 44.1 kHz stereo vocoder.

## Scope

Each model family owns its directory containing:
- Model architecture and sampling routines
- Configuration and weight loaders
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
- Parity tests against golden reference tensors
