# MossFormer2 SE

## Overview
MossFormer2 speech enhancement with Kaldi fbank front end, mask net, and iSTFT; chunked or whole-signal.

## Interface
- `MossFormer2SeConfig::from_json`
- `MossFormer2Se::open`
- `MossFormer2Se::load`
- `MossFormer2Se::enhance`
- `MossFormer2Se::enhance_with_mode`
- `MossFormer2Se::process_chunk`
- `MossFormer2Se::kaldi_fbank`
- `MossFormer2Se::kaldi_deltas`
- `MossFormer2Se::stft_frames`
- `MossFormer2Se::istft`
- `MossFormerMaskNet::forward`

## Invariants
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `model.safetensors` and an optional `config.json` (reference defaults when absent) from the model directory in `open` [mod.rs:open]; no writes, network, or env access.

## Verification
- Run: `cargo test -p turbospark-audio sts::mossformer2_se`
