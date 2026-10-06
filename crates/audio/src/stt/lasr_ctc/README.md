# LASR CTC / MedASR

This module ports the standalone LASR CTC family from
`mlx_audio/stt/models/lasr_ctc/` (lasr.py, config.py) in mlx-audio 0.5.7 at
source commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`. The upstream target
checkpoint is Google MedASR, a medical-domain automatic speech recognition
model.

## Overview

LASR CTC is an encoder-only CTC transcriber:

- A dense-conv-conv-dense subsampling front end (128 log-mel bins in, hidden
  512, stride-2 convolutions with kernel 5, so a 277-frame clip becomes 67
  frames).
- Seventeen encoder blocks, each: a SiLU feed forward with a scaled residual
  pair (`1.5 * residual + 0.5 * output`), RoPE multi-head self-attention
  (8 heads, head dim 64, plain residual), a Conformer-style convolution
  module (pointwise to 1024 channels, GLU, grouped depthwise kernel-32
  convolution with asymmetric padding left 15 / right 16, batch norm,
  SiLU, pointwise back to 512) with the scaled residual pair
  (`2.0 * residual + 1.0 * output`), a second SiLU feed forward with the
  same scaled residual pair, and a final layer norm.
- A linear CTC head over a 512-piece sentencepiece vocabulary.
- Bias-free layer norms (eps 1e-6); attention and convolution-module
  projections are bias-free; the subsampling stack and the CTC head carry
  biases.
- The frontend is the transformers `LASRFeatureExtractor` contract: unfold
  framing (window 400, hop 160, no centering), symmetric float64 Hann
  window, 512-point rfft, power spectrum, a lingvo-style kaldi-mel slope
  filterbank (257 bins, 125 to 7500 Hz, DC bin excluded), clamp at 1e-5,
  natural log, then a float32 cast. The Rust port runs this pipeline in
  float64 exactly like the reference.

## The upstream load path is broken

This family has no working upstream end-to-end path in mlx-audio 0.5.7.
Both failures were reproduced against the pinned conversion:

1. The stock `load()` path calls `LasrForCTC.sanitize`, which transposes
   every 3-axis conv weight to MLX `(out, kernel, in)` layout. The public
   MLX MedASR conversion already stores conv weights in that layout, so the
   stock sanitize double-transposes them; MLX `load_weights` accepts the
   mismatched arrays without complaint and the model silently decodes
   garbage.
2. The upstream `LasrForCTC` defines no `generate` method, so the mlx-audio
   STT generate entry point (`generate_transcription`) raises
   `AttributeError: 'LasrForCTC' object has no attribute 'generate'`, and
   the model's own `decode` returns an empty text string.

The verified path, which both the Python reference and this Rust port use:

- Read the raw safetensors tensors in their stored MLX layout directly. No
  conv weight is transposed. Only the CTC head's stored `(out, in, 1)`
  Conv1d weight is squeezed to the `(out, in)` linear layout (the upstream
  sanitize's one correct special case for this conversion).
- The feature frontend comes from the transformers `LASRProcessor`
  (`LASRFeatureExtractor`), which is the only working frontend for this
  family; the Rust port implements its exact float64 pipeline.
- Greedy CTC decoding follows the transformers `LasrForCTC.generate` and
  `LasrTokenizer._decode` contract: per-frame argmax, collapse consecutive
  repeats, drop the blank (pad) id, drop special tokens
  (`skip_special_tokens=True`), join the pieces with the sentencepiece
  metaspace marker turned into a space, and strip the one leading space the
  metaspace decoder prepends. Note the exact order: the collapse runs
  before the blank drop, so a blank between two identical tokens does not
  merge them.

## Pinned profile

- Hugging Face repository: `drankush-ai/medasr-mlx-fp32`
- Immutable revision: `3b967580b5176144bc633fac60420d1b122dfba8`
- Base model: `google/medasr` (official repository is access gated, HTTP 403
  in the verification environment; it remains unverified here)
- Input: mono f32 PCM at 16 kHz
- Architecture: 17 encoder blocks, hidden 512, 8 heads, intermediate 2048,
  128 log-mel bins, CTC vocabulary of 512 sentencepiece pieces
- Precision: fp32 throughout (351 tensors, all F32); no dequantization
- Rust profile: `MEDASR_MLX_FP32`

Artifact sizes are bytes; SHA-256 digests were computed from the downloaded
snapshot at the pinned revision. The ten files total 421,484,269 bytes.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 421170592 | `a16ea872d22fe7b71a5da99cbe1289dc0d81e71d149076e4e558b369dd6c0a83` |
| `config.json` | 1137 | `f964513aa89375d027292f1c47e1729bd05ec7852d4c6c1f68720b05b39ea33a` |
| `preprocessor_config.json` | 284 | `c1b0d9110224c2a196b06088803b958c0ee2df56234a85f01dcd38f27c8514f3` |
| `processor_config.json` | 371 | `6b11a5e6b9e6782cba8ed8fc66eebe0aa6e3427b13032282cc7d68107e5a1320` |
| `tokenizer.json` | 51454 | `6d74a85891527e9427cc68f27dfa4b4a90b279ff565aeddf58e90ec88246d84a` |
| `tokenizer_config.json` | 5120 | `939fbf4fd276db0fff0cb7f89e7e29e1b1835c099c587b51450feba023f061d8` |
| `added_tokens.json` | 2409 | `fc7d7435d19f28c8358be8abd65490b565c9944fedaaac313b2552b24e86eafc` |
| `spiece.model` | 246526 | `ef70658038e59086177bb16bdaf165e1bd9667905959fa1d538925a6489836e4` |
| `README.md` | 4857 | `49c4ff4e30f16a8b51245b9f31fcd8519bc9b37615b9acaf83d0da52a838c7af` |
| `.gitattributes` | 1519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |

## Inference contract

- Waveform in, greedy CTC transcript out: `LasrCtc::transcribe(samples)`.
  `greedy_token_ids` exposes the collapsed id sequence, `logits` the raw
  `[steps, vocab_size]` CTC logits.
- The frontend frames `(samples - 400) / 160 + 1` timesteps; shorter input
  is refused, non-finite samples are refused.
- Conv weights load in the stored MLX `(out, kernel, in / groups)` layout,
  consumed over channels-last rows; linears load in the HF `[out, in]`
  layout; a non-F32 dtype or unexpected shape is refused (this pin is fp32
  and the port verifies no other layout or quantization scheme).
- Batch normalization runs in inference mode with the checkpoint's running
  statistics and epsilon 1e-5 (the MLX and transformers default; the config
  does not carry it).
- Layer norms are weight-only: the pinned checkpoint ships no norm bias
  tensors, the transformers reference builds `bias=False` norms, and the
  MLX reference's affine biases stay at their zero init after the raw load,
  so the three agree.
- Config parsing refuses non-`lasr_ctc` model types, non-SiLU activations,
  biased attention or convolution projections, grouped-query attention, and
  non-default RoPE, because none of those variants were verified.
- Text decoding needs the id-to-piece table, which the port materializes
  from the checkpoint's own `tokenizer.json` (a Unigram piece list) at load;
  special tokens come from the same file's added-token flags.

## Reference and parity

`generate_lasr_ctc_fixture.py` runs the pinned checkpoint through the
reference MLX classes in eval mode with the raw-weights load described
above, and records the log-mel features, the kaldi-mel filterbank matrix,
the RoPE tables, the subsampler output, encoder layers 0, 8, and 16, the
final hidden state, the CTC logits, the per-frame argmax ids, the collapsed
greedy ids, and the transcript into
`crates/audio/testdata/lasr_ctc_reference.json`.

Regenerate on an Apple Silicon host with the mlx-audio reference
environment:

```sh
/Users/whit3rabbit/Documents/GitHub/.venv-mlxaudio/bin/python \
  crates/audio/scripts/generate_lasr_ctc_fixture.py \
  --model-dir ~/models/medasr-mlx-fp32 \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/lasr_ctc_reference.json \
  --revision 3b967580b5176144bc633fac60420d1b122dfba8
TURBOSPARK_LASR_CTC_MODEL_DIR=~/models/medasr-mlx-fp32 \
  cargo test --release -p turbospark-audio --lib stt::lasr_ctc -- \
  --ignored --nocapture --test-threads=1
```

The fixture is checked into `testdata/lasr_ctc_reference.json` (force-added;
the workspace `.gitignore` covers testdata directories); tests never
download model files.

- **Python reference transcript (pinned checkpoint, greedy):**
  `quick brown fx jumps over the laazy dog.` on the shared smoke WAV. The
  two recognition slips (`fx` for fox, `laazy`) are the model's own output,
  identical under the reference and the Rust port; they are preserved as
  evidence, not corrected.
- **Rust checkpoint transcript:** identical, byte for byte, including the
  full 67-id per-frame argmax sequence and the 23-id collapsed sequence.
- **Stage parity (release run, measured worst spot differences):**
  log-mel features `0.0e0` (the float64 frontend matches the reference
  exactly), encoder layers 0/8/16 `6.3e-5`, encoder final state `4.3e-7`,
  CTC logits `1.5e-5`, all against a `2.0e-4` absolute gate. The subsampler
  output reaches spot magnitudes near `1.1e5` (the conv stack produces very
  large pre-norm activations), so its stage gate adds a `1.0e-5` relative
  term; measured worst there is `3.9e-2` on a `1.1e5`-magnitude spot, a
  relative difference of `3.4e-7`. The transcript and token ids stay exact.

Always-on tests (no checkpoint required) pin the config refusals, the RoPE
table convention including its half-duplication, the kaldi-mel filterbank
matrix, the log-mel frontend against the fixture, the CTC decode rules, the
fixture transcript from the recorded argmax ids, and the fixture
provenance.

## Verification status

Both the Python reference and the Rust release-mode checkpoint run were
executed on this machine against the pinned revision of the fp32 MLX
conversion. One short English clip does not qualify recognition quality,
medical-domain accuracy, other checkpoints, or long audio; the official
`google/medasr` checkpoint stays gated and unverified, so this profile is
pinned as the best available candidate, not as the canonical model.
Performance, memory, runtime, catalog, FFI, and Swift gates remain open and
are separate work.
