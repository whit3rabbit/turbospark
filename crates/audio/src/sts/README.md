# Speech-to-Speech (STS) and Speech Enhancement Models

This directory contains Speech-to-Speech (STS), voice conversion, and neural speech enhancement model implementations.

The [central inventory](../../MODELS.md#sts) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[DeepFilterNet (DFN)](deepfilternet/README.md)**: Full-band 48 kHz speech enhancement and noise suppression framework combining ERB spectrogram gain masking and deep complex linear filtering.
- **[Mel-Band RoFormer](mel_roformer/README.md)**: Stereo 44.1 kHz music/vocal source separation with a binarized Slaney mel band split, dual-axis RoFormer transformers, and complex mask resynthesis.
- **[DialogueSidon](dialogue_sidon/README.md)**: Two-speaker separation and speech restoration at 24 kHz; a w2v-BERT-style encoder conditions an adaLN diffusion transformer sampled with DPM-Solver++ into a DAC decoder.
- **[SAM-Audio](sam_audio/README.md)**: Text-guided source separation; a DACVAE codec plus T5 text encoder feed a DiT whose velocity field is integrated with a midpoint ODE over codebook-space features.
- **[MossFormer2 SE 48K](mossformer2_se/README.md)**: 48 kHz speech enhancement; a MossFormer MaskNet predicts a complex STFT mask from Kaldi fbank features with FLASH (ReLU^2 group + linear attention) blocks interleaved with gated FSMN blocks.

## Scope

Each model family owns its directory containing:
- STFT domain feature extraction (ERB filterbanks, Vorbis windowing)
- Encoder and mask/filtering decoders
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
