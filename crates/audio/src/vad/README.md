# Voice Activity Detection (VAD) and Diarization Models

This directory contains Voice Activity Detection (VAD) and multi-speaker diarization models.

The [central inventory](../../MODELS.md#vad) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[Silero VAD](silero_vad/README.md)**: Lightweight, low-latency speech/non-speech detector with dual 16 kHz / 8 kHz learned-STFT branches and streaming LSTM.
- **[Sortformer Diarization](sortformer/README.md)**: NVIDIA NeMo FastConformer multi-speaker diarization tracking up to 4 speakers with overlap detection.
- **[Nemotron 3 Diarization](nemotron_diarization/README.md)**: High-resolution streaming diarization for up to 8 concurrent speakers at 10 ms granularity with subpixel convolution.

## Scope

Each model family owns its directory containing:
- Feature extraction and streaming state cache buffers
- Model backbone, speaker heads, and threshold post-processing
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
