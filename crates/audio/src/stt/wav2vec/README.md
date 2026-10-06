# Wav2Vec2 base CTC

This module ports the standalone Wav2Vec2 family from
`mlx_audio/stt/models/wav2vec/` (wav2vec.py, feature_extractor.py) in
mlx-audio 0.5.7 at source commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

The upstream module is a bare Wav2Vec2 backbone: its `sanitize` drops the
`lm_head`, the class exposes no `generate` or CTC decode, and the mlx-audio
STT loader does not auto-route `model_type "wav2vec2"` checkpoints (they are
the base architecture that MMS builds on). This family port keeps the
checkpoint's own CTC head so the standalone distribution is transcribable,
matching transformers `Wav2Vec2ForCTC` inference.

The architecture is the classic post-norm Wav2Vec2 variant: group-norm first
feature convolution (per-channel statistics over the temporal dimension,
pytorch-compatible epsilon placement) followed by six un-normalized GELU
convolutions, per-timestep layer norm plus projection, a grouped
weight-normalized positional convolution, one encoder layer norm applied
before the layer stack, and twelve post-norm transformer layers (attention on
the un-normalized input, residual, layer norm, feed forward, residual, final
layer norm). The stable-layer-norm variant lives in the `mms` family port.

## Pinned profile

- Hugging Face repository: `facebook/wav2vec2-base-960h`
- Immutable revision: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`
- Input: mono f32 PCM at 16 kHz
- Base architecture: 12-layer post-norm encoder, hidden 768, 12 heads,
  group-norm conv frontend, CTC vocabulary of 32 pieces
- Rust profile: `WAV2VEC2_BASE_960H`

Artifact sizes are bytes; SHA-256 digests were computed from the downloaded
snapshot at the pinned revision.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 377607901 | `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a` |
| `config.json` | 1596 | `d3ec255c063d9f95057b553b19c20135b259875834a4fe9deb218a6be25b4cf3` |
| `feature_extractor_config.json` | 158 | `d3de0c797bf9b65f90bc65c30cb7b303ebeda341f6fc80af33628c4b26b95632` |
| `preprocessor_config.json` | 159 | `b225d617c025463b9e157e06afea8b90dc7078fc70b013c533328423e0486b4a` |
| `tokenizer_config.json` | 163 | `dc790594f5bc351a4311c6624f40acd95850d4aaf2a5cb3c656c9b610720b608` |
| `special_tokens_map.json` | 85 | `bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675` |
| `vocab.json` | 291 | `19727f8944fe6459fc3f240ae2c198395b740f6a029bd23e06656266b83bcf64` |

## Inference contract

- The waveform is zero-mean unit-variance normalized with the epsilon inside
  the square root, `(x - mean) / sqrt(var + 1e-7)`, matching the upstream
  `feature_extractor.py` convention (the `mms` port normalizes with the
  epsilon outside the square root; the two variants are pinned by their own
  fixture tests and are not interchangeable).
- Conv weights load in the PyTorch `[out, in_per_group, kernel]` layout the
  official checkpoint ships, which is also the layout `ops::conv1d`
  consumes; the weight-norm positional convolution recomputes
  `g * v / ||v||` exactly like the reference sanitize.
- Greedy CTC decoding collapses adjacent repeats, drops blank token 0,
  concatenates vocabulary pieces, and converts `|` to a space. The upper-case
  letter vocabulary has no case information and the model is a small 1992
  base checkpoint, so recognition imperfections on hard clips are expected;
  the parity contract is Rust == Python reference on the same weights.
- Config parsing refuses stable-layer-norm encoders, adapter checkpoints,
  and non-GELU activations with reasons pointing at the `mms` family.

## Reference and parity

`generate_wav2vec_fixture.py` runs the pinned checkpoint through the
reference MLX classes in eval mode (MLX `nn.Dropout` defaults to training
mode, and an earlier fixture generated without `model.eval()` both corrupted
the stage tensors and degraded the transcript) and records the normalized
waveform, conv0 group-norm output, feature extractor, feature projection,
encoder layer inputs, final hidden state, CTC logits, greedy token ids, and
transcript into `crates/audio/testdata/wav2vec_reference.json`.

Regenerate on an Apple Silicon host with the mlx-audio reference environment:

```sh
~/.venv-mlxaudio/bin/python crates/audio/scripts/generate_wav2vec_fixture.py \
  --model-dir ~/models/wav2vec2-base-960h \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/wav2vec_reference.json \
  --revision 22aad52d435eb6dbaf354bdad9b0da84ce7d6156
TURBOSPARK_WAV2VEC_MODEL_DIR=~/models/wav2vec2-base-960h \
  cargo test --release -p turbospark-audio --lib stt::wav2vec -- \
  --ignored --nocapture --test-threads=1
```

The fixture is checked into `testdata/wav2vec_reference.json` (force-added;
the workspace `.gitignore` covers testdata directories); tests never
download model files.

- **Python reference transcript (pinned checkpoint, greedy):**
  `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG`
- **Rust checkpoint transcript:** identical, byte for byte, including the
  full greedy token id sequence.
- **Stage parity:** conv0 group norm, feature extractor, feature projection,
  every recorded encoder layer input, and the final hidden state match their
  fixture spots within `2.0e-4` absolute. CTC logits compare within `5.0e-3`
  (measured worst `2.2e-3`): the 768-length CTC dot product amplifies the
  hidden-state spot tolerance, while the transcript and token ids stay exact.

## Verification status

Both the Python reference and the Rust release-mode checkpoint run were
executed on this machine against the pinned revision. One short English clip
does not qualify recognition quality, other wav2vec2 checkpoints, or long
audio; performance, memory, runtime, catalog, FFI, and Swift gates remain
open and are separate work. This family does not change the `mms` module.

The `facebook/wav2vec2-base-960h` repository does not mark `model.safetensors`
as an official distribution format, but the pinned revision's safetensors
file is the repository's canonical weight artifact at that revision and the
digest above pins it immutably.
