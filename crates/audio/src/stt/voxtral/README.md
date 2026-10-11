# Voxtral Mini 3B

This module ports `mlx_audio/stt/models/voxtral/` (voxtral.py, config.py) from mlx-audio 0.5.7 at source commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Architecture Overview

Voxtral Mini 3B is Mistral's speech understanding model combining a Whisper-large-v3 audio tower with a Mistral text backbone:

1. **Frontend**: 16 kHz mono audio with 128-band log-mel spectrogram (N_FFT 400, hop 160).
2. **Audio Tower**: Whisper-large-v3 encoder (32 Transformer blocks, 1280 hidden dimension, 20 heads).
3. **Audio Projector**: 4-frame window pooling projector mapping acoustic representations into text hidden dimensions (3072).
4. **Text Decoder**: Mistral transformer backbone (24 layers, 3072 hidden dimension, 32 attention heads, 8 KV heads, intermediate size 9216).
5. **Prompt**: Tekken byte-level BPE tokenizer with pinned English transcription prompt tokens.

## Pinned Profile

- **Profile**: `VOXTRAL_MINI_3B_BF16`
- **Repository**: `mlx-community/Voxtral-Mini-3B-2507-bf16`
- **Immutable Revision**: `ab69f21c8e2c52b05335bf282898e141c8c42477`
- **Input**: 16 kHz mono f32 PCM up to 30 seconds (480,000 samples)
- **Precision**: BF16 / F32

## Verification

The Rust implementation matches the reference MLX output on the shared 16 kHz audio clip:
- Audio tower features match within 9e-3 relative tolerance vs BF16 reference.
- Emitted token IDs and transcript are byte-identical.
