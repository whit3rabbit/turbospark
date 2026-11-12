# turbospark-audio source root

## Overview
Library root of `turbospark-audio`: portable DSP primitives (waveform, WAV I/O, resampling, FFT, STFT, mel frontends), shared neural ops and MLX affine dequantization, and audio models grouped by role. Models live in the child directories below; the files here are the shared layer they all build on.

## Interface
- Waveform and I/O: `Waveform` [waveform.rs], `read_wav_f32`, `read_wav_f32_bytes`, `write_wav_f32`, `write_wav_i16` [wav.rs].
- Conversion and resampling: `mixdown_interleaved_to_mono_average`, `duplicate_mono_to_interleaved`, `deinterleave_to_planar`, `interleave_planar`, `extract_interleaved_channel`, `to_mono_resampled`, `MonoResampleStrategy` [conversion.rs]; `resample_mono_linear`, `resample_mono_sinc_hann`, `SincHannOptions` [resample.rs].
- Spectral: `RealFftPlan`, `real_fft_forward`, `real_fft_inverse`, `ComplexF32` [fft.rs]; `stft`, `istft`, `stft_with_modes`, `StftOptions`, `StftPaddingMode`, `StftWindowPlacement` [stft.rs]; `mel_filterbank`, `mel_spectrogram`, `log_mel_spectrogram`, `MelFilterbank` [mel.rs]; `nemo_log_mel_spectrogram`, `nemo_log_mel_spectrogram_with_padding` [nemo_mel.rs]; `whisper_log_mel`, `whisper_log_mel_window` [whisper.rs]; `hann_window`, `peak`, `rms`, `normalize_peak`, `scale_by_gain` [dsp.rs].
- Neural ops: `matmul`, `linear`, `layernorm`, `rmsnorm`, `sdpa`, `conv1d`, `conv_transpose1d`, `conv2d`, `rope_neox`, `rope_interleaved` [ops.rs]; `Linear`, `LayerNorm`, `RmsNorm`, `load_tensor`, `open_shards` [nn.rs]; `load_quantized`, `dequantize_rows`, `is_quantized`, `QuantScheme` [quant.rs].
- Errors: `AudioError`, `SpeechError`, `AudioModelError` (alias of `SpeechError`) [error.rs].
- Backend re-exports from the MiniMax Music 3 module [backend.rs].
- Child contracts:
  - [codec](./codec/EGM.md): codecs, vocoders, neural tokenizers.
  - [lid](./lid/EGM.md): language identification placeholder.
  - [lm](./lm/EGM.md): shared f32 language-model ports.
  - [models](./models/EGM.md): re-export facade by model role.
  - [music](./music/EGM.md): music generation.
  - [ops](./ops/EGM.md): test-only parity reference kernels.
  - [sts](./sts/EGM.md): speech enhancement and separation.
  - [stt](./stt/EGM.md): speech-to-text families.
  - [tts](./tts/EGM.md): text-to-speech families.
  - [vad](./vad/EGM.md): voice activity detection and diarization.

## Invariants
- `#![forbid(unsafe_code)]` [lib.rs].
- `extern crate self as turbospark_audio` keeps absolute crate paths valid inside the crate [lib.rs].
- WAV reads reject data chunks over `MAX_WAV_DATA_BYTES` (1 GiB) [wav.rs:MAX_WAV_DATA_BYTES].
- `AudioModelError` is the same type as `SpeechError` [error.rs:AudioModelError].

## Side Effects
- `read_wav_f32` reads a file from disk [wav.rs:read_wav_f32]; `load_tensor`, `open_shards` and `load_quantized` read safetensors files [nn.rs, quant.rs].
- `RealFftPlan::cached` [fft.rs], `mel_projector_cached` [mel.rs] and the sinc resampler [resample.rs] keep process-wide `TableCache` tables (`PLANS`, `BANKS`, `KERNELS`) that live in process-wide statics, so they outlive any one call. Each cache is bounded: oversized tables are marked not cacheable and a full cache is cleared [dsp.rs:TableCache::get_or_build].
- `ops::in_pool` runs work on a rayon worker pool [ops.rs:in_pool].

## Verification
- Run: `cargo test -p turbospark-audio`
