# DialogueSidon

## Overview
Sidon dialogue speech restoration: feature extraction, encoder conditioning, diffusion head with DPM solver, and DAC latent decode.

## Interface
- `SidonConfig::from_json`
- `DialogueSidon::open`
- `DialogueSidon::load`
- `DialogueSidon::extract_features`
- `DialogueSidon::normalize_chunk`
- `DialogueSidon::conditioning`
- `DialogueSidon::sample_latents`
- `DialogueSidon::decode_latents`
- `SidonEncoder::forward`
- `DiffusionHead::forward`
- `DpmSolver::new`
- `DpmSolver::step`

## Invariants
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input, Tensor, Unsupported) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.
- Config `sample_rate` (default 24000) is parsed into `SidonConfig` but never read or validated by the model code [mod.rs:SidonConfig::from_json].

## Side Effects
- Reads `model.safetensors` and the standalone `dac_config.json` + `dac.safetensors` tree in `open` [mod.rs:DialogueSidon::open]; no writes.

## Verification
- Run: `cargo test -p turbospark-audio sts::dialogue_sidon`
