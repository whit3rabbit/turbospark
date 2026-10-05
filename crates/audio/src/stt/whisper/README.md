# STT / Whisper Model Family

OpenAI Whisper and Distil-Whisper automatic speech recognition models.

- Reference: `mlx_audio/stt/models/whisper/` (mlx-audio v0.5.7)
- Upstream repo: [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)

## Architecture

- **Audio Frontend**: Slaney log-mel filterbank (80 or 128 mel bins), 400 FFT, 160 hop at 16 kHz. Provided by [`turbospark_audio::models::whisper`].
- **Audio Encoder**: 2x 1D convolutions (kernel size 3, stride 1 and stride 2) mapping 3000 mel frames to 1500 audio feature states, followed by sinusoidal/learned position embeddings and standard transformer encoder layers.
- **Text Decoder**: Autoregressive transformer decoder with causal self-attention, cross-attention attending over the audio encoder features, and feed-forward MLP layers.
- **Decoding**: Greedy or beam search decoding starting with standard prompt tokens (`<|startoftranscript|>`, language token, task token, `<|notimestamps|>`).

## Profiles

| Profile | Parameters | Mel Bins | Hidden Dim (`d_model`) | Enc Layers | Dec Layers | Heads |
|---|---|---|---|---|---|---|
| `whisper-tiny` | 39M | 80 | 384 | 4 | 4 | 6 |
| `whisper-base` | 74M | 80 | 512 | 6 | 6 | 8 |
| `whisper-small` | 244M | 80 | 768 | 12 | 12 | 12 |
| `whisper-medium` | 769M | 80 | 1024 | 24 | 24 | 16 |
| `whisper-large-v3` | 1550M | 128 | 1280 | 32 | 32 | 20 |
| `distil-large-v3` | 756M | 128 | 1280 | 32 | 2 | 20 |
