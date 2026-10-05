# Speech-to-Text (STT) Models

Speech-to-Text (STT) models transcribe spoken audio into written text, perform word/character-level alignment, or handle multilingual speech tasks.

## Model Families

| Family | Directory | Architecture | Key Characteristics | Status |
|---|---|---|---|---|
| **Whisper** | [`src/stt/whisper/`](../../crates/audio/src/stt/whisper/) | Encoder-Decoder Transformer | 80/128-band Slaney mel, 30s windowing, greedy autoregressive decoding | Verified |
| **Moonshine** | [`src/stt/moonshine/`](../../crates/audio/src/stt/moonshine/) | Rotary Transformer | Variable-length encoder, low-resource edge transcription | Verified |
| **Parakeet TDT** | [`src/stt/parakeet/`](../../crates/audio/src/stt/parakeet/) | FastConformer Transducer | Token-and-Duration Transducer (TDT), base-3 ternary quantization (Redux) | Verified |
| **Granite Speech** | [`src/stt/granite_speech5_ctc/`](../../crates/audio/src/stt/granite_speech5_ctc/) | Encoder-only CTC | 80-band HTK delta frontend, 470M parameters | In Progress |
| **MMS** | [`src/stt/mms/`](../../crates/audio/src/stt/mms/) | Wav2Vec2 + Adapters | 1,000+ languages, 300M / 1B parameter profiles | Verified |
| **Qwen3-ASR** | [`src/stt/qwen3_asr/`](../../crates/audio/src/stt/qwen3_asr/) | Audio Tower + Qwen3 LM | Grouped-query attention, 128 mel bins, full-prefix/KV decoding | Verified |
| **Qwen3-ForcedAligner** | [`src/stt/qwen3_forced_aligner/`](../../crates/audio/src/stt/qwen3_forced_aligner/) | CTC Aligner | Precise word/character timestamps for English and Chinese | Verified |
| **Nemotron ASR** | [`src/stt/nemotron_asr/`](../../crates/audio/src/stt/nemotron_asr/) | FastConformer + RNN-T | Nemotron 3.5 ASR architecture, relative positional attention | Verified |

## Model Family Details & Documentation

- **Whisper & Distil-Whisper**: [`crates/audio/src/stt/whisper/README.md`](../../crates/audio/src/stt/whisper/README.md)
  - URLs: `openai/whisper-tiny`, `distil-whisper/distil-large-v3`, `mlx-community/whisper-large-v3-mlx`
  - Benchmarks: Industry standard WER on LibriSpeech, Common Voice.
- **Moonshine**: [`crates/audio/src/stt/moonshine/README.md`](../../crates/audio/src/stt/moonshine/README.md)
  - URLs: `UsefulSensors/moonshine-tiny`, `mlx-community/moonshine-tiny-mlx`
  - Benchmarks: Extremely low latency on edge CPUs; matches MLX golden tensors within `7.4e-5`.
- **Parakeet TDT**: [`crates/audio/src/stt/parakeet/README.md`](../../crates/audio/src/stt/parakeet/README.md)
  - URLs: `nvidia/parakeet-tdt-0.6b-v2`, `nvidia/parakeet-tdt-0.6b-v3`, `nvidia/parakeet-tdt-0.6b-redux`
  - Benchmarks: Real-time factor < 0.05; TDT skip steps reduce decode time by ~60%.
- **IBM Granite Speech 5.0 TurboCTC**: [`crates/audio/src/stt/granite_speech5_ctc/README.md`](../../crates/audio/src/stt/granite_speech5_ctc/README.md)
  - URLs: `ibm-granite/granite-speech-5.0-turbo-ctc`
- **Meta MMS**: [`crates/audio/src/stt/mms/README.md`](../../crates/audio/src/stt/mms/README.md)
  - URLs: `facebook/mms-1b-all`
- **Qwen3-ASR**: [`crates/audio/src/stt/qwen3_asr/README.md`](../../crates/audio/src/stt/qwen3_asr/README.md)
  - URLs: `Qwen/Qwen3-ASR-1.7B`
- **Qwen3-ForcedAligner**: [`crates/audio/src/stt/qwen3_forced_aligner/README.md`](../../crates/audio/src/stt/qwen3_forced_aligner/README.md)
  - URLs: `Qwen/Qwen3-ForcedAligner-0.6B`
- **Nemotron ASR**: [`crates/audio/src/stt/nemotron_asr/README.md`](../../crates/audio/src/stt/nemotron_asr/README.md)
  - URLs: `nvidia/nemotron-speech-asr-0.6b`
