# KittenTTS TTS family

Port of `mlx_audio/tts/models/kitten_tts/` at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/kitten_tts)
(mlx 0.32.3). StyleTTS 2-family architecture with an ALBERT text encoder, duration
and prosody predictors, and an iSTFTNet decoder.

## Overview

Data flow for one segment (batch size 1, non-streaming, matching reference):
1. Phoneme IDs -> ALBERT (`KittenAlbert`)
2. Duration predictor (BLSTMs + AdaLayerNorm) -> duration alignment
3. F0/noise prosody predictor -> text encoder
4. iSTFTNet decoder with a harmonic sine source -> 24 kHz audio

## Architecture & Specifications

- **Sample Rate**: 24,000 Hz
- **Text Encoder**: ALBERT backbone (`KittenAlbert`)
- **Duration & Prosody**: Bidirectional LSTMs with AdaLayerNorm conditioning
- **Decoder**: iSTFTNet vocoder with SineGen harmonic generator and complex iSTFT synthesis
- **Weight Layout**: Safetensors checkpoints follow MLX parameter tree conventions (`bert.*`, `predictor.*`, `decoder.*`, etc.). Convolution weights are converted from `[out, kernel, in]` to the layouts required by `ops`. Weight normalization folds at load time.

## Verification

Fixture generator: `crates/audio/tools/gen_kitten_tts_fixtures.py` produces stage-level and end-to-end parity traces against the pinned MLX reference.
