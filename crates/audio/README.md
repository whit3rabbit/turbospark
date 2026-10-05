# turbospark-audio

Unified portable audio crate for TurboSpark: DSP primitives, shared neural tensor operations, and audio model implementations organized by role (music, STT, TTS, VAD, STS, codec, LID).

## Architecture Overview

### 1. Signal Processing Primitives
- **`waveform`**: Interleaved f32 audio buffer abstraction with sample rate and channel count.
- **`wav`**: Strict RIFF/WAVE reader (PCM 8/16/24/32, IEEE float 32, extensible format) and writer.
- **`conversion`**: Mixdown, channel duplication, planar interchange, and mono-resample pipelines.
- **`resample`**: Linear and torchaudio-compatible sinc-Hann polyphase resamplers.
- **`fft` / `stft`**: Radix-2 / Bluestein real FFT, PyTorch-convention STFT and iSTFT with window-sum-of-squares normalization.
- **`mel`**: HTK and Slaney mel filterbank construction and power log-mel spectrograms.
- **`nemo_mel`**: NeMo-compatible pre-emphasis and centered Slaney power-mel frontend (Parakeet, Nemotron).
- **`whisper`**: OpenAI Whisper log-mel filterbank (80/128 bands) and 30-second windowing.
- **`dsp`**: Windowing functions (Hann), peak, RMS, gain, and peak normalization.

### 2. Neural Operations & Quantization
- **`ops`**: Pure Rust f32 neural tensor primitives (linear projection, RMSNorm, LayerNorm, SiLU, GELU, RoPE tables/rotations, attention SDPA, Conv1D, Conv2D, ConvTranspose1D).
- **`quant`**: MLX groupwise affine dequantization (2, 3, 4, 6, 8-bit) for U32-packed safetensors checkpoints.

### 3. Models Organized by Role
- **`music`**: Music generation model families (e.g. [`minimax_music3`](src/music/minimax_music3/README.md)).
- **`stt`**: Speech-to-Text and alignment models ([`whisper`](src/stt/whisper/README.md), [`moonshine`](src/stt/moonshine/README.md), [`parakeet`](src/stt/parakeet/README.md), [`granite_speech5_ctc`](src/stt/granite_speech5_ctc/README.md), [`mms`](src/stt/mms/README.md), [`qwen3_asr`](src/stt/qwen3_asr/README.md), [`qwen3_forced_aligner`](src/stt/qwen3_forced_aligner/README.md), [`nemotron_asr`](src/stt/nemotron_asr/README.md)).
- **`tts`**: Text-to-Speech synthesis models ([`kokoro`](src/tts/kokoro/README.md)).
- **`vad`**: Voice activity detection and diarization ([`silero_vad`](src/vad/silero_vad/README.md), [`sortformer`](src/vad/sortformer/README.md), [`nemotron_diarization`](src/vad/nemotron_diarization/README.md)).
- **`sts`**: Speech-to-speech and enhancement models ([`deepfilternet`](src/sts/deepfilternet/README.md)).
- **`codec`**: Learned neural codecs and vocoders ([`codec`](src/codec/README.md)).
- **`lid`**: Spoken language identification ([`lid`](src/lid/README.md)).

Each model family owns a dedicated folder containing its architecture, configuration, weight loader, and a detailed `README.md` covering background, URLs, benchmarks, and model metadata.

## Verification & Portability

The crate is portable plain f32 Rust, free of macOS- or Metal-specific dependencies, building and testing on any target architecture.

```sh
cargo test -p turbospark-audio
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
```
