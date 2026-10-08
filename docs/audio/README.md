# TurboSpark Audio Architecture

All audio functionality in TurboSpark is consolidated into a single unified crate: **`crates/audio`** (`turbospark-audio`), organized by role modality.

## Role Organization

```text
crates/audio/
├── src/
│   ├── conversion.rs        # Channel moves, mixdown, planar interchange, mono resampling
│   ├── dsp.rs               # Windowing, RMS, peak, normalize
│   ├── error.rs             # AudioError and SpeechError diagnostic variants
│   ├── fft.rs               # Radix-2 and Bluestein real FFT
│   ├── mel.rs               # Slaney and HTK mel filterbanks
│   ├── nemo_mel.rs          # NeMo pre-emphasis and centered mel frontend
│   ├── ops.rs               # Pure Rust neural tensor operations (linear, norms, attention, conv)
│   ├── quant.rs             # MLX groupwise affine dequantization (2/3/4/6/8-bit)
│   ├── resample.rs          # Linear and polyphase sinc-Hann resamplers
│   ├── stft.rs              # STFT and iSTFT with window-sum-of-squares normalization
│   ├── wav.rs               # Strict RIFF/WAVE reader and writer
│   ├── waveform.rs          # Interleaved f32 PCM buffer abstraction
│   ├── whisper.rs           # Whisper log-mel DSP frontend (80/128 bands, 30s windowing)
│   │
│   ├── music/               # Music Generation
│   │   ├── minimax_music3/  # MiniMax Music 3 (AR + Flow DiT + 44.1 kHz Vocoder)
│   │   └── README.md
│   │
│   ├── stt/                 # Speech-to-Text & Alignment
│   │   ├── whisper/         # OpenAI Whisper & Distil-Whisper
│   │   ├── moonshine/       # Useful Sensors Moonshine
│   │   ├── parakeet/        # NeMo Parakeet TDT (v2, v3, Redux ternary)
│   │   ├── granite_speech5_ctc/ # IBM Granite Speech 5.0 TurboCTC
│   │   ├── mms/             # Meta MMS Wav2Vec2 multilingual CTC
│   │   ├── qwen3_asr/       # Qwen3 ASR
│   │   ├── qwen3_forced_aligner/ # Qwen3 Forced Aligner
│   │   ├── nemotron_asr/    # NeMo Nemotron 3.5 ASR (RNN-T)
│   │   └── README.md
│   │
│   ├── tts/                 # Text-to-Speech
│   │   ├── kokoro/          # Kokoro 82M (StyleTTS2 + iSTFTNet)
│   │   └── README.md
│   │
│   ├── vad/                 # Voice Activity Detection & Diarization
│   │   ├── silero_vad/      # Silero VAD (16 kHz & 8 kHz streaming LSTM)
│   │   ├── sortformer/      # Sortformer multi-speaker diarization (4 speakers)
│   │   ├── nemotron_diarization/ # Nemotron 3 Diarization (8 speakers, 10 ms)
│   │   └── README.md
│   │
│   ├── sts/                 # Speech-to-Speech & Audio Enhancement
│   │   ├── deepfilternet/   # DeepFilterNet 1/2/3 (48 kHz ERB + deep filtering)
│   │   └── README.md
│   │
│   ├── codec/               # Audio Codecs & Vocoders
│   │   └── README.md        # EnCodec, DAC, Mimi, SNAC planning
│   │
│   └── lid/                 # Spoken Language Identification
│       └── README.md        # ECAPA-TDNN, Wav2Vec2 LID planning
```

## Role Documentation

- [Music Generation Guide](music.md)
- [Speech-to-Text (STT) Guide](stt.md)
- [Text-to-Speech (TTS) Guide](tts.md)
- [Voice Activity Detection & Diarization Guide](vad.md)
- [Speech-to-Speech & Enhancement Guide](sts.md)
- [Neural Codecs Guide](codec.md)
- [Language Identification Guide](lid.md)

## In-Crate Model Folders

Each model and model family has its own folder under `crates/audio/src/<role>/<family>/` containing full source code and a detailed `README.md` documenting:
- Description and architecture
- Hugging Face and upstream reference URLs
- Verified benchmarks, latency, and resource footprint
- Sample rates, parameters, and input/output contracts
