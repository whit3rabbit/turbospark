# Audio Architecture and Layer Scope

`crates/audio` (`turbospark-audio`) is TurboSpark's unified portable audio crate. It combines signal processing primitives (waveform, containers, resampling, FFT, STFT, mel spectrograms), shared neural tensor operations, and audio model implementations organized by role modality (`music`, `stt`, `tts`, `vad`, `sts`, `codec`, `lid`).

For comprehensive model and role documentation, see [TurboSpark Audio Architecture](audio/README.md).

For the macOS product workflows, see [Audio workspace](AUDIO_WORKSPACE.md).
The [capability audit](AUDIO_WORKSPACE_CAPABILITIES.md) distinguishes portable model
source from runnable native integrations and qualification.

## Module Map

### 1. Signal Processing Primitives
| Module | Ports / Basis | Notes |
|---|---|---|
| `waveform` | `WavData` | Interleaved f32; `samples[frame * channels + channel]` |
| `wav` | `wav_reader.cpp`, `wav_writer.cpp` | PCM u8/i16/i24/i32 + IEEE f32; extensible; 1 GiB intake cap |
| `conversion` | `conversion.cpp` | Float32-only channel moves, mixdown with f32/f64 accumulation, planar interchange |
| `resample` | `resampling.cpp` | Linear + torchaudio-compatible sinc-Hann polyphase |
| `fft` | `fft.cpp` wrapper | Radix-2/Bluestein real FFT for even sizes |
| `stft` | `stft_graph.cpp` | torch.stft conventions: reflect center padding, window-sum-of-squares inverse |
| `mel` | `mel_spectrogram_frontend.cpp` | HTK and Slaney scales; power spectrogram; `log10(max(x, floor))` |
| `nemo_mel` | mlx-audio Parakeet/Nemotron | NeMo pre-emphasis, centered STFT, Slaney power mel, natural log |
| `whisper` | OpenAI Whisper frontend | 16 kHz, FFT 400, hop 160, 80/128 Slaney bands, 30s windowing |
| `dsp` | `waveform_ops.cpp` | Periodic Hann, peak, RMS, gain, peak normalization |

### 2. Neural Operations and Quantization
| Module | Description | Notes |
|---|---|---|
| `ops` | Shared f32 tensor operations | Linear, RMSNorm, LayerNorm, SiLU, GELU, RoPE, SDPA attention, 1D/2D convolutions |
| `quant` | MLX affine dequantization | Fast 2, 3, 4, 6, and 8-bit groupwise dequantization for packed safetensors |

### 3. Models Organized by Role
| Role | Modality | Key Families & Documentation |
|---|---|---|
| `music` | Music Generation | [MiniMax Music 3](audio/music.md) (`minimax_music3`) |
| `stt` | Speech-to-Text & Alignment | [Whisper, Moonshine, Parakeet, Granite, MMS, Qwen3 ASR, Nemotron ASR](audio/stt.md) |
| `tts` | Text-to-Speech | [Kokoro 82M](audio/tts.md) |
| `vad` | VAD & Diarization | [Silero VAD, Sortformer, Nemotron Diarization](audio/vad.md) |
| `sts` | Speech Enhancement | [DeepFilterNet 1/2/3](audio/sts.md) |
| `codec` | Neural Codecs | [DAC, EnCodec, Mimi, SNAC](audio/codec.md) |
| `lid` | Language Identification | [ECAPA-TDNN, Wav2Vec2 LID](audio/lid.md) |

## Layout Conventions

- Waveforms are interleaved. Planar buffers exist only inside `conversion`.
- File I/O uses standard library file readers and writers with bounded buffer checks.
- Every shape disagreement returns descriptive `AudioError` or `SpeechError` variants.

## Verification

```sh
cargo test -p turbospark-audio
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
```

Real Metal offload and hardware performance gates are documented in the runtime and model gates references.
