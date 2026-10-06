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
- **`music`**: Music generation model families ([`minimax_music3`](src/music/minimax_music3/README.md)).
- **`stt`**: Speech-to-Text and alignment models ([`canary`](src/stt/canary/README.md), [`fireredasr2`](src/stt/fireredasr2/README.md), [`fun_asr_nano`](src/stt/fun_asr_nano/README.md), [`glmasr`](src/stt/glmasr/README.md), [`granite_speech`](src/stt/granite_speech/README.md), [`granite_speech5_ctc`](src/stt/granite_speech5_ctc/README.md), [`higgs_audio_3`](src/stt/higgs_audio_3/README.md), [`mega_asr`](src/stt/mega_asr/README.md), [`mms`](src/stt/mms/README.md), [`moonshine`](src/stt/moonshine/README.md), [`moss_transcribe_diarize`](src/stt/moss_transcribe_diarize/README.md), [`nemotron_asr`](src/stt/nemotron_asr/README.md), [`parakeet`](src/stt/parakeet/README.md), [`phonon`](src/stt/phonon/README.md), [`qwen3_asr`](src/stt/qwen3_asr/README.md), [`qwen3_forced_aligner`](src/stt/qwen3_forced_aligner/README.md), [`sensevoice`](src/stt/sensevoice/README.md), [`whisper`](src/stt/whisper/README.md)).
- **`tts`**: Text-to-Speech synthesis models ([`kokoro`](src/tts/kokoro/README.md)).
- **`vad`**: Voice activity detection and diarization ([`silero_vad`](src/vad/silero_vad/README.md), [`sortformer`](src/vad/sortformer/README.md), [`nemotron_diarization`](src/vad/nemotron_diarization/README.md)).
- **`sts`**: Speech-to-speech and enhancement models ([`deepfilternet`](src/sts/deepfilternet/README.md)).
- **`codec`**: Learned neural codecs and vocoders ([`bigvgan`](src/codec/bigvgan/README.md), [`dacvae`](src/codec/dacvae/README.md), [`descript`](src/codec/descript/README.md), [`ecapa_tdnn`](src/codec/ecapa_tdnn/README.md), [`encodec`](src/codec/encodec/README.md), [`fish_s1_dac`](src/codec/fish_s1_dac/README.md), [`higgs_audio`](src/codec/higgs_audio/README.md), [`mimi`](src/codec/mimi/README.md), [`mimo_audio_tokenizer`](src/codec/mimo_audio_tokenizer/README.md), [`moss_audio_tokenizer`](src/codec/moss_audio_tokenizer/README.md), [`nemotron_voicechat`](src/codec/nemotron_voicechat/README.md), [`s3`](src/codec/s3/README.md), [`snac`](src/codec/snac/README.md), [`vocos`](src/codec/vocos/README.md)).
- **`lid`**: Spoken language identification ([`lid`](src/lid/README.md)).

Each model family owns a dedicated folder containing its architecture, configuration, weight loader, and a detailed `README.md` covering background, URLs, benchmarks, and model metadata.

## Verification & Portability

The crate is portable plain f32 Rust, free of macOS- or Metal-specific dependencies, building and testing on any target architecture.

```sh
cargo test -p turbospark-audio
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
```
