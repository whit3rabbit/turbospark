---
description: "Rust reference for turbospark-audio DSP primitives: Waveform, WAV reading and writing, mono conversion, resampling, STFT, and mel features."
icon: box
---
# DSP primitives

Use these functions when you need the same preprocessing the speech models expect, or when you need strict WAV reading and writing without a media library. The shared value is `Waveform`, which holds interleaved f32 samples. Read a file with `read_wav_f32`, convert it with `to_mono_resampled`, then compute the feature your model needs.

The fallible functions return `Result` with a descriptive error. The WAV writers return plain bytes because they cannot fail on a valid `Waveform`. The [README example](README.md#smallest-working-example) runs this whole path, so start there if you want working code first.

## Waveform

`Waveform` is the value type every DSP function accepts. Its fields are public: `sample_rate: u32`, `channels: u16`, and `samples: Vec<f32>`. The samples are interleaved, so frame `f` and channel `c` sit at `samples[f * channels + c]`.

| Item | Signature | Notes |
|---|---|---|
| `Waveform::new` | `(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Result<Waveform, AudioError>` | Rejects rates above 768000, channel counts outside 1 to 256, and sample counts that are not whole frames. |
| `Waveform::mono` | `(sample_rate: u32, samples: Vec<f32>) -> Result<Waveform, AudioError>` | One channel. |
| `Waveform::silence` | `(sample_rate: u32, channels: u16, frames: usize) -> Result<Waveform, AudioError>` | Zero-filled. |
| `frame_count` | `(&self) -> usize` | Samples divided by channels. |
| `duration_seconds` | `(&self) -> f64` | Frames divided by sample rate. |

Use `Waveform::new` for audio you build yourself, so a bad layout fails early. Use `read_wav_f32` for files. The type is documented at [docs.rs](https://docs.rs/turbospark-audio/latest/turbospark_audio/struct.Waveform.html).

## WAV reading and writing

`read_wav_f32` reads a file, and `read_wav_f32_bytes` reads a byte slice. Both accept PCM at 8, 16, 24, and 32 bits, IEEE float at 32 bits, and `WAVE_FORMAT_EXTENSIBLE` wrappers. Integer samples scale into the range negative one to one, and 8-bit PCM is unsigned around a midpoint of 128.

The reader refuses a data chunk larger than `MAX_WAV_DATA_BYTES`, which is 1 GiB. It returns `AudioError::BufferTooLarge` in that case. Other malformed input returns `NotWav`, `MalformedWav`, `UnsupportedWavFormat`, or `UnsupportedBitsPerSample`, depending on which check failed.

| Item | Signature |
|---|---|
| `read_wav_f32` | `(path: &Path) -> Result<Waveform, AudioError>` |
| `read_wav_f32_bytes` | `(bytes: &[u8]) -> Result<Waveform, AudioError>` |
| `write_wav_f32` | `(wave: &Waveform) -> Vec<u8>`, IEEE float, 32 bits |
| `write_wav_i16` | `(wave: &Waveform) -> Vec<u8>`, PCM, 16 bits |

`write_wav_f32` writes the samples unchanged. `write_wav_i16` multiplies by 32768, rounds to nearest, and clamps to the 16-bit range, so negative full scale maps to -32768 and positive full scale clips to 32767. There is no 24-bit writer.

## Mono conversion and resampling

`to_mono_resampled` mixes a waveform down to mono and resamples it in one call. It returns `Vec<f32>`, not a `Waveform`, because speech frontends want a plain sample buffer. The second argument is the target rate, and the third is a `MonoResampleStrategy`.

`MonoResampleStrategy::Linear` is the default and uses linear interpolation. It does not filter before it decimates, so it can alias when you downsample speech. `MonoResampleStrategy::SincHann` takes a `SincHannOptions` value, and its defaults match the torchaudio-compatible resampler: rolloff 0.99, a lowpass filter width of 6, and f32 kernel and accumulation. Use `SincHann` for speech.

| Item | Signature |
|---|---|
| `to_mono_resampled` | `(wave: &Waveform, target_rate: u32, strategy: &MonoResampleStrategy) -> Result<Vec<f32>, AudioError>` |
| `resample_mono_linear` | `(input: &[f32], source_rate: u32, target_rate: u32) -> Result<Vec<f32>, AudioError>` |
| `resample_mono_sinc_hann` | `(input: &[f32], source_rate: u32, target_rate: u32, options: &SincHannOptions) -> Result<Vec<f32>, AudioError>` |
| `mixdown_interleaved_to_mono_average` | `(samples: &[f32], channel_count: usize, accumulation: MonoMixAccumulation) -> Result<Vec<f32>, AudioError>` |

The `conversion` module also exports the planar helpers `deinterleave_to_planar`, `interleave_planar`, `extract_interleaved_channel`, and `duplicate_mono_to_interleaved`. Planar layout exists only for these moves. Your own code should stay interleaved.

## STFT and mel features

`stft` returns one row of `ComplexF32` per frame. `istft` reverses it, using window-sum-of-squares normalization, and needs the original length. `mel_spectrogram` projects the power spectrum onto a mel filterbank. `log_mel_spectrogram` also takes the log of each value, clamped at a floor you pass in.

The defaults in `MelSpectrogramOptions::default()` describe a 16 kHz frontend: 80 mel bands, a 512-point FFT, a 400-sample Hann window, hop 160, centered frames, the Slaney scale, a range from 0 to 8000 Hz, and power 2.0. The Whisper frontend uses a floor of `1e-10`, and the example uses the same value.

| Item | Signature |
|---|---|
| `stft` | `(samples: &[f32], options: &StftOptions) -> Result<Vec<Vec<ComplexF32>>, AudioError>` |
| `istft` | `(spectra: &[Vec<ComplexF32>], options: &StftOptions, original_len: usize) -> Result<Vec<f32>, AudioError>` |
| `mel_spectrogram` | `(samples: &[f32], options: &MelSpectrogramOptions) -> Result<Vec<Vec<f32>>, AudioError>` |
| `log_mel_spectrogram` | `(samples: &[f32], options: &MelSpectrogramOptions, floor: f32) -> Result<Vec<Vec<f32>>, AudioError>` |

Model families have their own frontends, and the generic call is wrong for them. Whisper uses `whisper::whisper_log_mel` and `whisper::whisper_log_mel_window`, which match the OpenAI frontend. Parakeet and Nemotron use `nemo_mel`, which has its own padding and normalization rules. The wrong frontend still returns a correctly shaped matrix, so the mistake is silent. Check the family README before you choose one.

## Errors on this page

- `ShapeMismatch` means the interleaved sample count is not a whole number of frames, or a buffer has the wrong length for the call.
- `InvalidParameter` names a bad option, such as a rolloff outside zero to one, a zero filter width, or a non-positive power or floor.
- `InvalidWavLayout` covers a zero channel count, a channel count above 256, or a sample rate above 768000.
- `BufferTooLarge` means a computed output or a WAV data chunk exceeds the crate's cap.

All of these are `AudioError` values, documented at [docs.rs](https://docs.rs/turbospark-audio/latest/turbospark_audio/enum.AudioError.html). Read the variant's fields before you retry, because each one names the value that failed.

## Source

The implementation lives in `crates/audio/src`. The [`waveform.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/src/waveform.rs), [`wav.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/src/wav.rs), [`resample.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/src/resample.rs), [`stft.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/src/stft.rs), and [`mel.rs`](https://github.com/whit3rabbit/turbospark/blob/main/crates/audio/src/mel.rs) files define every signature above. The [audio architecture notes](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO.md) list the numeric conventions these functions follow.
