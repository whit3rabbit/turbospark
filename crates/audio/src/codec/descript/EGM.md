# Descript Audio Codec (DAC)

## Overview
Descript RVQ audio codec: encode to codebook indices, decode from codes or latents, plus loudness-normalized compress/decompress.

## Interface
- `DacConfig::from_json`
- `DacConfig::hop_length`
- `Dac::open`
- `Dac::load`
- `Dac::padded_len`
- `Dac::encode`
- `Dac::encode_partial`
- `Dac::decode_latents`
- `Dac::decode_codes`
- `Dac::compress`
- `Dac::decompress`

## Invariants
- [mod.rs:REFERENCE_COMMIT] pins the mlx-audio commit the port and its parity fixtures are normative against; re-pin only with regenerated fixtures.
- [mod.rs:Dac::decode_codes] validates code-book count and index range, returning `SpeechError::Input`.
- Bad config or tensor shapes surface as `SpeechError` (BadConfig, Input) at load or call time, not as panics [mod.rs:load].
- Inference is CPU f32 over host `Vec<f32>` buffers; no GPU state is held here.

## Side Effects
- Reads `config.json` and the safetensors weights from the model directory in `open` [mod.rs:open]; no writes or network. The `real_checkpoint_gate` test reads `TURBOSPARK_DAC_MODEL` and is skipped when unset [tests.rs:real_checkpoint_gate].

## Verification
- Run: `cargo test -p turbospark-audio codec::descript`
