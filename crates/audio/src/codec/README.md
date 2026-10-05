# Neural Audio Codec and Vocoder Models

This directory contains neural audio codecs, vocoders, and discrete audio tokenizers.

The [central inventory](../../MODELS.md#codec) tracks upstream source families, checkpoint profiles, and verification status.

## Target Families

- **EnCodec / DAC (Descript Audio Codec)**: High-fidelity discrete multi-rate acoustic neural codecs using residual vector quantization (RVQ).
- **Mimi / SNAC**: Low-latency multi-scale neural audio codecs designed for real-time speech and audio generation.
- **Higgs Audio**: High-compression neural audio representation.

## Scope

Each model family owns its subfolder containing:
- Encoder and quantizer implementations (residual vector quantization, codebooks)
- Decoder / vocoder synthesis layers
- Family `README.md` documenting upstream references, URLs, benchmarks, and model parameters
