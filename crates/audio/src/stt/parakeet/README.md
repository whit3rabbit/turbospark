# Parakeet TDT

This module follows `mlx_audio/stt/models/parakeet/` from mlx-audio 0.5.7
at commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. It implements the
NeMo log-mel frontend, FastConformer encoder, recurrent predictor, joint
token/duration head, and greedy TDT decode used by the MLX TDT v2/v3
checkpoints.

`ParakeetTdt::open` reads a local `config.json` and `model.safetensors`.
`transcribe` accepts finite mono PCM at the configured 16 kHz sample rate.
`decode` returns the same transcript grouped into sentences with per-token
waveform timestamps on the upstream grid (subsampling factor times hop
length over the sample rate per encoder frame), mirroring upstream
`ParakeetTDT.decode` through the shared `stt::nemo` alignment module.
The end-to-end example also accepts WAV input and resamples to 16 kHz.

| Profile | Hugging Face repository | Pinned revision | Rust loader |
|---|---|---|---|
| TDT v2 | `mlx-community/parakeet-tdt-0.6b-v2` | `8ae155301e23d820d82aa60d24817c900e69e487` | Supported |
| TDT v3 | `mlx-community/parakeet-tdt-0.6b-v3` | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | Supported |
| Redux | `moondream/parakeet-redux` | `2bf128600aac4b16946f7ed8372e56117fe5e23b` | Supported with temporary F32 expansion |
| Fluid Speech Pro | `FluidVoice-Labs/fluid-speech-pro-mlx` | `aeb1d4e43487586b15d86c0a1dd4511e1cf6b7a0` | Supported: fp32 or 8-bit file, verbatim and clean styles |

```sh
cargo run -p turbospark-audio --example parakeet_transcribe -- <model-dir> <audio.wav>
```

The upstream MLX Python smoke checks passed for all three pinned profiles.
Rust v2, v3, and Redux also transcribed the same generated 16 kHz smoke WAV as
the pinned MLX reference: "The quick brown fox jumps over the lazy dog." Redux
expands ternary tensors into a temporary F32 safetensors file during load and
deletes it once the model weights are resident. The local checkpoint downloads
were removed after the runs.

Stage fixtures under `testdata/parakeet/` pin the v2 frontend, encoder, and
decode loop against the reference in float32. Regenerate them with
`python3 scripts/generate_parakeet_fixtures.py` (needs the pinned checkpoint
and the reference virtualenv); the manifest records shapes and SHA-256
digests. The checkpoint-gated tests run with TURBOSPEECH_PARAKEET_MODEL
(default `~/models/parakeet-tdt-0.6b-v2`) and skip cleanly without weights.
Measured stage parity on the fixture clip: mel 8.2e-5, encoder 7.5e-6,
predictor states at the 1e-7 level, joint logits 1.2e-4 to 2.7e-4, and an
identical 21-step decode table. Rust also transcribes a second generated
clip ("Pack my box with five dozen liquor jugs.") identically to the float32
Python reference. These witnesses do not establish transcription quality,
performance, or product integration.

## Fluid Speech Pro

`fluid.rs` follows the publisher's `fluid_speech_mlx.py` (parakeet-mlx 0.7.0),
not mlx-audio. A `fluid_speech` block in `config.json` routes `open` there, so
the extra block can never be dropped silently. Differences from v3:

- FrameCleaner at `encoder.pre_encode.fv_insert`, between the subsampling
  projection and the first Conformer layer: LayerNorm, linear, depthwise
  kernel-15 conv over time, exact GELU, linear, residual. It looks 7 frames
  ahead, so the model is offline only.
- NeMo framing: one trailing padded mel frame, statistics over complete hop
  frames, masked conv subsampling (`FastConformer::encode_mel_masked`), the
  block on every subsampled frame, then truncation to the valid length.
- Clean style (`TranscriptStyle::Clean`, `--clean` in the example):
  `clean-mode.safetensors` adds `alpha / rank * B @ A` to every encoder linear
  and pointwise conv, and folds `decoder_input` into the first LSTM bias and
  `joint_hidden` into `joint.pred.bias`. Unused adapter modules are refused.
- `model.safetensors` (fp32) is preferred; otherwise `model.8bit.safetensors`
  is dequantized to f32 at load after its 8-bit, group-64 layout is checked.

Fixtures under `testdata/parakeet_fluid/` come from
`python -I scripts/generate_fluid_speech_fixtures.py --reference-dir <dir with
the pinned fluid_speech_mlx.py> --audio <clip>` on the 8-bit weights in
float32. The clip is macOS `say` speech with fillers, trimmed to 488 valid
frames so the padded frame adds a subsampled frame and masking is exercised.
Measured on 2026-11-27: mel 2.6e-4, subsampling and FrameCleaner 1e-5 relative
to the tensor peak, encoder 1.6e-6 (verbatim) and 3.7e-7 (clean), identical
token sequences in both styles. Gated tests use TURBOSPEECH_FLUID_SPEECH_MODEL
(default `~/models/fluid-speech-pro-mlx`). Mutations (no masking, no block,
no low-rank add, no bias folds, truncation off by one) each fail them. The fp32
file was not run. These witnesses do not establish WER, speed or memory.

The runtime runs this port as family `fluid_speech_pro`
(`crates/runtime/src/fluid_speech.rs`): 15 s windows with the Core ML runner's
overlap merge, one resident style at a time (a style change reopens the
model), and the request field `transcript_style`. Peak footprint on one 5 s
clip (M4 Max, 8-bit install): 2.57 GB verbatim, 2.68 GB clean.

The Glade bundle (`coder543/fluid-speech-pro-glade`, Apple Core AI `.aimodel`)
is not supported: it needs macOS 27 or iOS 27 and the unpublished Glade
runtime. Its card's numbers are the converter's, not this repo's.
