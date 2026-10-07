# Granite Speech 4.1 2B NAR

This module ports `mlx_audio/stt/models/granite_speech_nar/`
(granite_speech_nar.py, config.py, encoder.py, projector.py, editor.py,
decoding.py) from mlx-audio 0.5.7 at source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

Single-pass non-autoregressive ASR:

1. A 16-block Conformer encoder (context-blocked multi-head self-attention
   with Shaw relative-position embeddings over a 1025-entry table, GLU conv
   module with an inference BatchNorm, half-weight feed-forwards, post-norm)
   runs over 160-dim paired log-mel features; block 8 applies the
   self-conditioning projection, and a posterior-weighted window pool
   (weight `1 - blank_probability`, window 4) feeds the BPE CTC head.
2. The initial hypothesis is the argmax over the BPE logits collapsed by
   greedy CTC (dedup adjacent repeats, then drop blanks).
3. The projector normalizes four encoder hidden states (layers 4, 8, 12,
   and the final state), fuses them, windows them in blocks of 15, mean-pools
   groups of 5 as the query init, and runs a two-layer Q-Former cross
   stack with learned query and window-position embeddings to produce one
   2048-wide audio token per 5 frames.
4. Audio embeddings are pre-divided by the editor's
   `embedding_multiplier = 12` (the editor re-multiplies every input; the
   text embeddings are not pre-divided). The hypothesis is interleaved with
   blank insertion slots (odd positions, minimum length 8) and concatenated
   after the audio tokens.
5. The 40-layer bidirectional Granite editor (RMSNorm, GQA 16/4, custom
   `attention_multiplier = 1/128` scale instead of `1/sqrt(head_dim)`,
   half-rotation RoPE, `residual_multiplier = 0.22`, tied LM head,
   `logits_scaling = 8`) scores the text tail; a second CTC collapse of the
   tail argmax produces the final token ids, decoded with the checkpoint
   tokenizer (special tokens skipped).

The frontend follows the reference `_compute_features`: reflect-padded
centered 512-point STFT (hop 160), periodic Hann window of 400 samples
centered in the frame, an f64 HTK 80-band filterbank (the reference
`precise=True` path), log10 with a 1e-10 clamp, a global peak-relative
floor of 8 dB, /4 + 1 scaling, and truncation to
`2 * (n_samples // 320)` frames before the log.

## Pinned profile

- Hugging Face repository: `mlx-community/granite-speech-4.1-2b-nar-mlx`
- Immutable revision: `6acb7892068dd30227f20aba6eb7c4b0ae5c7e7c`
- Input: mono f32 PCM at 16 kHz
- Encoder: 16 blocks, hidden 1024, 8 heads, head dim 128, context 200
- Editor: 40 layers, hidden 2048, GQA 16/4, tied embeddings, vocab 100352
- Rust profile: `GRANITE_SPEECH_4_1_2B_NAR`

Artifact sizes are bytes; SHA-256 digests were computed from the downloaded
snapshot at the pinned revision.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 4509376040 | `f2ec4215196eaa20c7b816c0526b3012dfb9b0078e545c9ecdb09c601a6c4992` |
| `config.json` | 2553 | `bd6da3a78d176c3b786ce2c97772b463d75be6735b9d360b36caa4f8e9236eba` |
| `tokenizer.json` | 7153802 | `64c10a88b2495872bd7da5a885861a1757d9c23590c40fd378546ae176d280f6` |
| `tokenizer_config.json` | 398 | `58105c6db9e2746eb3006726399c8668d515b8efbfc53005a27c35a3085a48e9` |
| `special_tokens_map.json` | 579 | `c08676c49fd7969a3130f72be6d4bf34da66aa484a6e21dffe359893a1bd5f2e` |
| `generation_config.json` | 38 | `af7cd9bbb214dea4f6a37756c2c133d002f6325ed4f497efc991f02b4d486dad` |
| `preprocessor_config.json` | 289 | `e12be1e9d4ec5c459741328f28d9d8c00c3c688e8fe4230a6ec28df5470db8b3` |
| `processor_config.json` | 153 | `74d21364b507dcbe465420152d48edd9276f94a84d22a78d0da371e716a79374` |
| `chat_template.jinja` | 6418 | `9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce` |

## Deliberate divergences from the reference

- The reference computes the whole stack in bfloat16; this port computes in
  dequantized f32. Stage gates below are set from the measured drift, while
  the primary contract is exact hypothesis, final token ids, and transcript.
- The reference materializes weights through mlx-audio's quantize helper for
  editor weights; the pinned distribution stores plain (bf16-rounded)
  safetensors and this port loads them directly.
- The official `ibm-granite` NAR checkpoint ships PyTorch-layout Conv1d
  weights and fails upstream inference; this port reads the pinned MLX
  conversion's stored layouts and refuses unexpected shapes.

## Reference and parity

`generate_granite_speech_nar_fixture.py` runs the pinned checkpoint through
the reference classes in eval mode and records input features, BPE logits,
the initial hypothesis, projector hidden states, audio embeddings, editor
text ids, editor logits, final token ids, and the transcript into
`crates/audio/testdata/granite_speech_nar_reference.json` (force-added; the
workspace `.gitignore` covers testdata directories). Tests never download
model files.

Regenerate on an Apple Silicon host with the mlx-audio reference
environment:

```sh
~/.venv-mlxaudio/bin/python crates/audio/scripts/generate_granite_speech_nar_fixture.py \
  --model-dir ~/models/granite-speech-4.1-2b-nar-mlx \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/granite_speech_nar_reference.json \
  --revision 6acb7892068dd30227f20aba6eb7c4b0ae5c7e7c
TURBOSPARK_GRANITE_SPEECH_NAR_MODEL_DIR=~/models/granite-speech-4.1-2b-nar-mlx \
  cargo test --release -p turbospark-audio --lib stt::granite_speech_nar -- \
  --ignored --nocapture --test-threads=1
```

- **Python reference transcript (pinned checkpoint):**
  `the quick brown fox jumps over the lazy dog.` (10 final tokens; the
  initial BPE hypothesis is identical to the final ids on this clip)
- **Rust checkpoint run:** identical transcript and final token ids; the
  gated test asserts both plus the editor text ids.
- **Stage parity (f32 port vs bf16 reference, measured worst spot):** input
  features within `2.0e-4` (both sides f32), projector input hidden state
  `9.4e-3` (gate `2.0e-2`), BPE logits `9.2e-2` (gate `3.0e-1` on ~20-magnitude
  logits), audio embeddings `1.9e-1` (gate `1.0`), editor tail logits `2.5`
  (gate `1.0e1` on ~40-magnitude logits over a 100352-wide vocabulary).
  The argmax decisions that feed the two CTC collapses are exact.

## Verification status

Both the Python reference and the Rust release-mode checkpoint run were
executed on this machine against the pinned revision (release wall time
about 28 s including load). One short English clip does not qualify
recognition quality; editing-pass behavior on degraded hypotheses, longer
audio, and the official IBM checkpoint layout remain unverified.
Performance, memory, runtime, catalog, FFI, and Swift gates remain separate
work.
