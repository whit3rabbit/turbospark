---
description: "Unified portable audio crate: DSP primitives (waveform, WAV I/O, resampling, FFT, STFT, mel spectrograms), neural audio ops, and models organized by role (music, STT, TTS, VAD, STS, codec, LID)."
icon: music
---

# audio

`crates/audio` (package `turbospark-audio`) is TurboSpark's unified portable audio crate and the largest crate in the workspace at roughly 230k lines of Rust. It combines the signal processing layer ported from the C++ `audio.cpp` utility layer (waveform, WAV containers, resampling, FFT, STFT, mel frontends), shared f32 neural tensor operations, and the audio model families ported from the pinned mlx-audio 0.5.7 snapshot, organized by role. The crate is `#![forbid(unsafe_code)]`, builds without Metal, MLX, or macOS APIs, and is consumed by the runtime, server, CLI, catalog, FFI, and embeddings crates.

## What it does

| Layer | Contents |
|---|---|
| DSP primitives | The `Waveform` value type (interleaved f32 plus sample rate and channel count), a strict RIFF/WAVE reader and writer (PCM u8/i16/i24/i32 and IEEE f32), channel conversion and mono mixdown, linear and torchaudio-compatible sinc-Hann polyphase resampling, radix-2 real FFT, STFT and inverse with window-sum-of-squares normalization, HTK and Slaney mel filterbanks, and NeMo and Whisper log-mel frontends |
| Neural ops | Portable f32 kernels (linear, norms, RoPE, attention pieces, convolutions), MLX groupwise affine dequantization (2/3/4/6/8-bit), shared layer structs, and the dense LM backbones the TTS families import |
| Models by role | 94 family directories under seven role modules. Each family directory is self-contained: implementation, configuration, weight loader, and a README with checkpoint URLs, upstream references, and verified stages |

Waveforms are interleaved: `samples[frame * channels + channel]`. Planar buffers exist only inside `conversion`. Unsupported geometry or corrupt checkpoints return descriptive `AudioError` or `SpeechError` variants that name the input, so a reader of the message alone can tell what failed.

## Where it sits

Depends on:

- [core](core.md) (shared primitives and `TokenId`), [model-io](model-io.md) (checkpoint and manifest loading), [tokenizer](tokenizer.md)
- `serde`/`serde_json` and the Hugging Face `tokenizers` crate for configuration and text frontends

Used by:

- [runtime](runtime.md): sessions, state, and Metal offload over these model implementations
- [server](server.md) and [cli](cli.md): audio capabilities on the user-facing surfaces
- [catalog](catalog.md): pinned profile metadata and local installation validation
- [ffi](ffi.md): qualified C ABI capabilities consumed by the Swift app
- [embeddings](embeddings.md): reuses the quantization scheme and shared layer loading
- [gpu](gpu.md): a macOS dev-dependency only, for parity checks against the portable reference

## Public API tour

### DSP primitives

- `src/waveform.rs`: the `Waveform` value type every DSP module consumes and produces. Build one with `Waveform::new(sample_rate, channels, samples)`, which validates the layout.
- `src/wav.rs`: `read_wav_f32`, `read_wav_f32_bytes`, `write_wav_f32`, `write_wav_i16`. Intake is capped at `MAX_WAV_DATA_BYTES` (1 GiB) to refuse hostile multi-GiB files.
- `src/conversion.rs`: mixdown, mono duplication, planar interchange, channel extraction, and `to_mono_resampled`, the pipeline the model frontends call.
- `src/resample.rs`: `resample_mono_linear` and `resample_mono_sinc_hann` with `SincHannOptions`.
- `src/fft.rs`: radix-2 FFT with a real-FFT wrapper (`RealFftPlan`, `real_fft_forward`, `real_fft_inverse`).
- `src/stft.rs`: `stft`, `istft`, and variants following `torch.stft` conventions: reflect center padding, window-sum-of-squares inverse normalization.
- `src/mel.rs`: `mel_filterbank`, `mel_spectrogram`, `log_mel_spectrogram`, with HTK and Slaney scales selected by `MelScale`.
- `src/nemo_mel.rs`: NeMo pre-emphasis and centered Slaney power-mel. Parakeet and Nemotron keep separate padding and normalization contracts.
- `src/whisper.rs`: the Whisper log-mel frontend: 16 kHz, FFT 400, hop 160, 80 or 128 Slaney bands, 30-second windowing.
- `src/dsp.rs`: periodic Hann windows, peak, RMS, gain, and peak normalization.

### Neural tensor ops and quantization

- `src/ops.rs`: shared f32 kernels. Tensors are row-major slices, linear weights use the HF `[out, in]` layout, and matmul and conv split output rows across threads with `std::thread::scope`, keeping the crate dependency-free.
- `src/quant.rs`: MLX groupwise affine dequantization for U32-packed safetensors. A model module must refuse a quantization scheme it has not verified rather than silently mis-decode.
- `src/nn.rs`: shared `Linear` and `LayerNorm` structs so families stop carrying private copies of the same two layers.
- `src/lm/`: the dense LM backbones the TTS families import (llama, qwen2, qwen3, gpt2, cohere2, bailing_moe, falcon_h1 with a Mamba2 selective scan), including KV caches and RoPE constructions.
- `src/backend.rs`: the safe compute-backend contract (`ComputeBackend`, `DeviceWeight`, `WeightEncoding`, `DType`) shared with Metal offload; model sequencing stays portable.
- `src/models/`: a facade re-exporting the role modules plus flat paths (`models::whisper`, `models::kokoro`, `models::deepfilternet`, `models::silero_vad`) kept for compatibility.

### Role modules

| Role | Path | Families | Representative names |
|---|---|---|---|
| Music generation | `src/music/` | 1 | `minimax_music3` (flow-matching DiT) |
| Speech-to-text | `src/stt/` | 29 | `whisper`, `parakeet`, `moonshine`, `qwen3_asr`, `canary`, `voxtral`, `granite_speech5_ctc`, `sensevoice`, `mms` |
| Text-to-speech | `src/tts/` | 39 | `kokoro`, `megatts3`, `bark`, `dia`, `omnivoice`, `chatterbox`, `confucius4`, `zonos2` |
| VAD and diarization | `src/vad/` | 3 | `silero_vad`, `sortformer`, `nemotron_diarization` |
| Speech-to-speech and enhancement | `src/sts/` | 5 | `deepfilternet`, `mel_roformer`, `mossformer2_se`, `dialogue_sidon`, `sam_audio` |
| Codecs and vocoders | `src/codec/` | 17 | `encodec`, `descript` (DAC), `mimi`, `snac`, `bigvgan`, `vocos`, `wavvae`, `s3`, `ecapa_tdnn` |
| Language identification | `src/lid/` | 0 ported | Reserved facade aligned with the mlx-audio `lid` layout; the ECAPA-TDNN backbone is ported under `codec/ecapa_tdnn` |

Two structural notes. Composite TTS families compose the shared codecs instead of re-implementing them: `confucius4` over `bigvgan`, `dia` over `descript` DAC, `echo_tts` over `fish_s1_dac`, `omnivoice` over `higgs_audio`, and `megatts3` over `wavvae`. And zero-shot voice cloning (`megatts3`) requires explicit caller consent (`consent: true`) because the community WavVAE encoder provenance is unverified. The full catalog with port status per family lives in `crates/audio/MODELS.md` and the user-facing pages linked at the end of this document.

## Examples

The crate ships runnable programs under `crates/audio/examples/`. Run one with:

```sh
cargo run -p turbospark-audio --example granite5_transcribe -- MODEL_DIR AUDIO.wav
```

### Load a WAV, resample to mono, and compute log-mel

The shape every speech frontend starts from, adapted from the cross-module pipeline test `tests/dsp_pipeline.rs`:

```rust
use turbospark_audio::conversion::{to_mono_resampled, MonoResampleStrategy};
use turbospark_audio::mel::{log_mel_spectrogram, MelSpectrogramOptions};
use turbospark_audio::resample::SincHannOptions;
use turbospark_audio::{read_wav_f32, write_wav_f32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Any WAV on disk becomes interleaved f32 plus rate and channel count.
    let wave = read_wav_f32(std::path::Path::new("input.wav"))?;

    // Mixdown to mono and resample to the 16 kHz the speech frontends want.
    let mono = to_mono_resampled(
        &wave,
        16_000,
        &MonoResampleStrategy::SincHann(SincHannOptions::default()),
    )?;

    // Log-mel spectrogram with the default frontend (80 mel bands).
    let log_mel = log_mel_spectrogram(&mono, &MelSpectrogramOptions::default(), 1e-10)?;
    println!(
        "{} mono samples -> {} frames x {} bands",
        mono.len(),
        log_mel.len(),
        log_mel.first().map(|frame| frame.len()).unwrap_or(0)
    );

    // The writer returns RIFF bytes; write_wav_i16 is the PCM sibling.
    std::fs::write("copy.wav", write_wav_f32(&wave))?;
    Ok(())
}
```

### Transcribe with Granite Speech 5 CTC

Adapted from `examples/granite5_transcribe.rs`. Each family exposes its own loader and task entry point; this one takes a checkpoint directory and returns text:

```rust
use std::path::PathBuf;

use turbospark_audio::models::stt::granite_speech5_ctc::GraniteSpeech5Ctc;
use turbospark_audio::read_wav_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let audio_path = PathBuf::from(args.next().expect("audio.wav"));

    let waveform = read_wav_f32(&audio_path)?;
    if waveform.sample_rate != 16_000 || waveform.channels != 1 {
        return Err(format!(
            "expected mono 16 kHz WAV, got {} Hz and {} channels",
            waveform.sample_rate, waveform.channels
        )
        .into());
    }

    let model = GraniteSpeech5Ctc::load(&model_dir)?;
    println!("{}", model.transcribe(&waveform.samples)?);
    Ok(())
}
```

### Enhance speech with DeepFilterNet

Adapted from `examples/dfn_enhance.rs`, which reads a 48 kHz WAV and writes the enhanced samples as raw little-endian f32:

```rust
use std::path::PathBuf;

use turbospark_audio::sts::DeepFilterNet;
use turbospark_audio::read_wav_f32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().expect("model dir"));
    let wav = args.next().expect("48 kHz wav");
    let out = args.next().expect("output path");

    let samples = read_wav_f32(std::path::Path::new(&wav))?;
    if samples.sample_rate != 48_000 {
        return Err("expected 48 kHz input".into());
    }

    let model = DeepFilterNet::open(&model_dir)?;
    let enhanced = model.enhance(&samples.samples)?;
    println!("{} samples in, {} samples out", samples.samples.len(), enhanced.len());

    let bytes: Vec<u8> = enhanced.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out, bytes)?;
    Ok(())
}
```

## Testing and fixtures

Every model module ships op-level parity tests against golden tensors produced by the pinned Python reference, plus an end-to-end run for checkpoints small enough to install locally. Fixtures live in `crates/audio/testdata/` (about 340 MB, excluded from the published crate through `Cargo.toml`) alongside `tests/fixtures/`; generator scripts such as `tests/reference/generate_whisper.py` live with the tests. Tests must not download assets, and regeneration commands and digests are recorded per family separately from the tests.

Cross-module integration tests hold the DSP layer together: `tests/dsp_pipeline.rs` recomputes the mel reference through a direct DFT instead of the crate's FFT, so a packing or split bug cannot cancel itself across modules, and `tests/wav_roundtrip.rs` checks that writer output reads back identical.

Gate stages (implementation, fixture parity, checkpoint parity, task quality, runtime, catalog, FFI, Swift) are recorded separately per family. Compilation and parity alone do not establish quality or device performance.

```sh
cargo test -p turbospark-audio
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
cargo fmt -p turbospark-audio --check
cargo clippy -p turbospark-audio --tests
```

## Gotchas

{% hint style="warning" %}
Numeric contracts are load-bearing and change only with new evidence from the reference. The FFT is unscaled forward with a 1/(n/2) inverse. Slaney area normalization is `2 / (upper_hz - lower_hz)`, never a mel-coordinate width. Whisper drops the final centered STFT frame before its global log-mel peak and clamp. Parakeet's mel uses constant signal padding plus feature normalization while Nemotron's uses reflect padding with no normalization. Always compare against the pinned mlx-audio commit, not a floating branch.
{% endhint %}

{% hint style="warning" %}
The crate must stay portable: plain f32 arithmetic, no Metal, MLX, or macOS APIs. Keep the cross-target check green (`cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio`) because Metal offload lives in the runtime, not here. Model families must refuse unverified quantization schemes and unsupported geometry with explicit errors rather than guessing.
{% endhint %}

## Related pages

- [Audio overview](../audio/overview.md)
- [Speech to text](../audio/speech-to-text.md)
- [Text to speech](../audio/text-to-speech.md)
- [Music generation](../audio/music-generation.md)
- [Testing and verification](../developer/testing-and-verification.md)
