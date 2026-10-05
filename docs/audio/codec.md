# Neural Audio Codecs & Vocoders

Neural audio codecs compress raw audio waveforms into discrete tokens or continuous embeddings, and decode tokens back into audio waveforms.

## Target Families & Upstream References

### 1. DAC (Descript Audio Codec)
- **Reference**: High-fidelity universal audio compression with residual vector quantization (RVQ).
- **Sample Rates**: 16 kHz, 24 kHz, 44.1 kHz.

### 2. EnCodec
- **Reference**: Meta AI EnCodec multi-bandwidth neural audio codec.
- **Bitrates**: 1.5 kbps, 3.0 kbps, 6.0 kbps, 12.0 kbps, 24.0 kbps.

### 3. Mimi / SNAC
- **Reference**: Low-latency multi-scale acoustic tokenizers for conversational speech models (e.g. Moshi).

### 4. In-Crate Directory
- [`crates/audio/src/codec/`](../../crates/audio/src/codec/)
- Documentation: [`crates/audio/src/codec/README.md`](../../crates/audio/src/codec/README.md)
