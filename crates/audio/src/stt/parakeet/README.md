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
