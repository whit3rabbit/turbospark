# Mel-Band RoFormer

## Overview
Mel-Band RoFormer source separation model.

## Interface
- `MelRoFormerConfig::from_json`
- `MelRoFormer::open`
- `MelRoFormer::load`
- `MelRoFormer::forward` (one chunk)
- `MelRoFormer::separate`, `MelRoFormer::separate_controlled`: planar stereo of any length, ZFTurbo `demix` overlap-add (`chunk_size` windows every `chunk_size / num_overlap` samples); the controlled form calls `progress(done, total)` between chunks and stops with `SpeechError::Input` on false

## Invariants
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor, Unsupported) at load or call time [mod.rs:load]. A filterbank the DSP layer rejects (for example a sample rate above 768 kHz, which `validate` allows) returns `BadConfig` from `BandGeometry::build` instead of panicking [mod.rs:BandGeometry::build].
- The iSTFT drops the imaginary part of the DC and Nyquist bins, as torch.istft does [mod.rs].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and scans the directory with `std::fs::read_dir` for weights in `open` [mod.rs:MelRoFormer::open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio sts::mel_roformer`
