# Neural Audio Codec and Vocoder Models

This directory contains neural audio codecs, vocoders, and discrete audio tokenizers.

The [central inventory](../../MODELS.md#codec) tracks upstream source families, checkpoint profiles, and verification status.

## Model Families

- **[BigVGAN](bigvgan/README.md)**: Universal neural vocoder using periodic activations (Snake) for high-fidelity speech and audio synthesis.
- **[DAC-VAE](dacvae/README.md)**: Variational autoencoder variant of the Descript Audio Codec for continuous audio representation.
- **[Descript Audio Codec (DAC)](descript/README.md)**: High-fidelity discrete multi-rate acoustic neural codec using residual vector quantization (RVQ) and Snake activations.
- **[ECAPA-TDNN](ecapa_tdnn/README.md)**: Time Delay Neural Network with squeeze-and-excitation feature extractor for speaker and acoustic embedding.
- **[EnCodec](encodec/README.md)**: High-fidelity multi-bandwidth neural audio codec with residual vector quantization and LSTM sequence modeling.
- **[Fish S1 DAC](fish_s1_dac/README.md)**: Fish Audio discrete autoencoder with downsampled residual vector quantization and interleaved rotary transformer layers.
- **[Higgs Audio](higgs_audio/README.md)**: High-compression neural audio representation and discrete tokenizer.
- **[Mimi](mimi/README.md)**: Kyutai low-latency multi-scale neural audio codec with causal Conv1d and projected transformer backbones.
- **[MiMo Audio Tokenizer](mimo_audio_tokenizer/README.md)**: Discrete audio tokenizer with residual vector quantization, causal transposed convolution, and an iSTFT synthesis vocoder head.
- **[MOSS Audio Tokenizer](moss_audio_tokenizer/README.md)**: OpenLMLab discrete audio tokenizer featuring causal pointwise convolutions and residual LFQ quantizers.
- **[Nemotron VoiceChat](nemotron_voicechat/README.md)**: NVIDIA speech synthesis vocoder with complex iSTFT projection heads.
- **[S3](s3/README.md)**: Continuous speech representation with sliding-window causal convolutions and streaming downsampling.
- **[SNAC](snac/README.md)**: Multi-scale neural audio codec with hierarchically downsampled codebook rates for real-time speech and audio generation.
- **[Vocos](vocos/README.md)**: Fourier-based neural vocoder synthesizing waveforms via inverse STFT from learned magnitude and phase projections.

## Scope and Structure

Each model family owns its subfolder containing:
- Encoder and quantizer implementations (residual vector quantization, lookup codebooks, causal convolutions)
- Decoder and vocoder synthesis layers (transposed convolutions, complex iSTFT overlap-add)
- Pinned profile configs and weight loaders
- Test suites asserting numeric parity against pinned MLX and Python golden reference tensors
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
