# Speech-to-Speech (STS) and Speech Enhancement Models

Speech-to-Speech (STS) and Enhancement models remove acoustic noise, enhance speech clarity, separate sources, or perform voice conversion.

Runnable today (runtime tasks `enhancement` and `separation`, the app's Clean Up page, `turbospark enhance`): Inworld Voice Isolator v1, MossFormer2 SE 48K, and Mel-Band RoFormer vocals. DeepFilterNet below is source only, with no runtime session. The product surfaces are in [Audio workspace](../AUDIO_WORKSPACE.md).

## Implemented Model Families

### DeepFilterNet (DFN 1 / 2 / 3)

- **In-Crate Directory**: [`crates/audio/src/sts/deepfilternet/`](../../crates/audio/src/sts/deepfilternet/)
- **Documentation**: [`crates/audio/src/sts/deepfilternet/README.md`](../../crates/audio/src/sts/deepfilternet/README.md)
- **Upstream URLs**:
  - Official Repository: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
  - MLX Conversion: [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx)
  - Research Papers: Schroeter et al. (ICASSP 2022, IWAENC 2022, 2023)
- **Architecture**:
  - Sample Rate: Full-band 48 kHz audio.
  - Spectral Analysis: 960 FFT (20 ms window) and 480 hop (10 ms) with Vorbis windowing.
  - Multi-scale Processing: ERB-scale filterbank masking for higher frequencies, deep complex linear filtering for lower frequencies (`nb_df`).
  - Backbone: Multi-layer Convolutional + Gated Recurrent Unit (GRU) encoder.
- **Benchmarks**:
  - Latency: 10 ms algorithmic frame delay with lookahead compensation.
  - Verification: Waveform correlation 0.964 on noisy benchmark audio.
- **Status**: no runtime session, catalog row, or app control, and no unit tests in the module ([MODELS.md](../../crates/audio/MODELS.md)).

### Inworld Voice Isolator v1

- **Documentation**: [`crates/audio/src/sts/voice_isolator/README.md`](../../crates/audio/src/sts/voice_isolator/README.md), Core ML streaming step in [`COREML.md`](../../crates/audio/src/sts/voice_isolator/COREML.md).
- **Upstream**: [inworld/voice-isolator](https://huggingface.co/inworld/voice-isolator) (Apache-2.0); reference code [inworld-ai/voice-isolator](https://github.com/inworld-ai/voice-isolator).
- **Catalog**: `voice-isolator-v1` (offline) and `voice-isolator-v1-realtime` (causal, streams with 63 ms algorithmic latency).
- **Runtime**: task `enhancement`, family `voice_isolator`, 48 kHz mono; Metal by default (`TURBOSPARK_VOICE_ISOLATOR_DEVICE=cpu` forces the CPU) with CPU fallback. Live streams: `ts_audio_enhance_stream_*`.

### MossFormer2 SE 48K

- **Documentation**: [`crates/audio/src/sts/mossformer2_se/README.md`](../../crates/audio/src/sts/mossformer2_se/README.md).
- **Catalog**: `mossformer2-se-48k-fp16` (`starkdmi/MossFormer2_SE_48K_MLX`).
- **Runtime**: task `enhancement`, family `mossformer2_se`, 48 kHz mono, portable CPU.

### Mel-Band RoFormer (vocals)

- **Documentation**: [`crates/audio/src/sts/mel_roformer/README.md`](../../crates/audio/src/sts/mel_roformer/README.md).
- **Catalog**: `mel-roformer-vocals-v1` (`mlx-community/mel-roformer-zfturbo-vocals-v1-mlx`, MIT, training data undisclosed).
- **Runtime**: task `separation`, family `mel_roformer` (config `model_type` `mel_band_roformer`), 44.1 kHz interleaved stereo vocals, portable CPU, 8 s chunks with 50% overlap. The instrumental is input minus vocals, computed by the host.

Quality (PESQ, DNSMOS, SDR) is not measured for any of the runnable families; only reference parity is.
