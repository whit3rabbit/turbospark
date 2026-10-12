# TADA (Text-Audio Dual Alignment TTS)

Port of `mlx_audio/tts/models/tada/` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/tada)
(MLX 0.32.3). Spec task 75.

## Architecture

A Llama backbone (the shared [lm engine](../../lm/README.md) llama
flavor: no biases, explicit head_dim, tied head, standard rope) driven
entirely by summed embeddings: token ids through the embed table,
acoustic features through a biased linear, a 2-row acoustic-mask
embedding, and two duration ids through `time_start_embed` /
`time_end_embed` tables. Per step the last hidden state conditions a
flow-matching prediction head (the VibeVoice head layout: sinusoidal
timestep embedding, AdaLN SwiGLU blocks, affine-free final norm) that
predicts a velocity over `[acoustic_dim, 2 * num_time_bits]` latents;
Euler integration walks a logSNR-spaced grid (sigmoid of a linspace
5 to -5, endpoints pinned to 0 and 1) with the two-half classifier-free
guidance (the acoustic first `acoustic_dim` channels and the duration
bits guided at separately scheduled scales, cosine/linear/constant
schedules). The duration halves decode through a gray code
(MSB-first +/-1 bits, xor unwind) into the next inter-frame gaps.

Generation ends at the de-normalized acoustic features
(`x * acoustic_std + acoustic_mean`) and the per-frame durations after
the zero-shot skip (`max(0, prefix_len - num_transition_steps) +
num_transition_steps - 1`), plus the trailing duration repeat. The
waveform decode (DAC-style conv stack under a segment-masked local
attention) and the reference-audio encoder/aligner live in the
SEPARATE `HumeAI/tada-codec` download (the aligner additionally needs
torch + transformers wav2vec2), so this port ends at the codes with a
documented refusal (`decode_waveform_refusal`), exactly like the
breeze_tts/pocket_tts patterns.

Scope: batch 1, caller-owned text tokenization (`normalize_text` is
ported; the caller encodes the normalized text plus the header words
with the checkpoint tokenizer and resolves the special ids;
[`build_prompt`] assembles the chat prefix, including the
`<|eot_id|> * shift_acoustic` suffix and the prefix length the skip
math needs). Noise comes from a [`DiffusionNoise`] provider (one
standard-normal draw per step, scaled by the noise temperature). The
negative CFG branch runs as a second single-batch forward with its own
cache (the reference doubles the batch axis; batch rows are
independent). The reference text sampler is UNREACHABLE from
`generate` with its defaults (`num_extra_steps` has no CLI surface and
logits are never computed), so the port implements free continuation
greedily over the pad-suppressed, sign-aware repetition-penalized,
temperature/top-k/top-p filtered logits (shared
[tts::sampling](../sampling.rs) processors) and documents the
divergence. The reference `speed_up_factor` two-pass mode re-rolls the
noise draws and its duration scaling only flows through the
prompt-audio path, so it is not ported. Voice cloning
(`encode_reference`, the DP aligner, the prompt-audio prefill) needs
the codec and torch and is not ported.

## Verification

Fixture protocol identical to the codec families:
`tools/gen_tada_fixtures.py` builds seeded tiny instances of the real
reference modules (2-layer llama, hidden 32, 4 query heads, 2 kv
heads, head_dim 8, vocab 200, tied head; acoustic_dim 8,
num_time_classes 16, shift_acoustic 2; 2-layer diffusion head over
16-dim latents; 4 flow-matching steps; a character tokenizer standing
in for Llama-3.2) and records, per case in
`testdata/tts_tada/<case>/` (config.json + model.safetensors +
traces.json, digests in manifest.json):

- `generate/`: the zero-shot run with acoustic cfg 1.6 (cosine
  schedule over the logSNR grid); the recorded `mx.random.normal`
  draws (one per step), the de-normalized skip-trimmed features, and
  the decoded durations.
- `nocfg/`: the cfg-1.0 twin (single-row forwards).

The tiny tada-codec decoder rides the config (the reference
constructs it) but is never run. The Rust tests reload each case
through the same `Model::open` path a real checkpoint takes and replay
the recorded draws; the generator enforces the Python-side replay
caveat before freezing (fresh weights plus the recorded stream
reproduce the features; the reference has no mx.compile or
hash-dependent iteration on this path).

Measured parity on the frozen fixtures (relative to the reference
magnitude): 3.1e-7 both cases against a 2e-4 relative gate (over 600x
margin). `TADA_MEASURE=1` re-prints the diffs. Unit tests pin the
gray-code round trip and bit order, the prompt-builder layout, the
waveform refusal, and the config defaults.

Real-checkpoint gate NOT run: the main TADA checkpoint and
`HumeAI/tada-codec` were not downloaded in this porting round. Open
items: a real load-and-generate pass over the codes, the tada-codec
decoder port decision, voice quality, runtime/FFI/Swift integration.

## Gotchas found during verification

- The guidance split point is the acoustic half of the LATENT
  (`config.acoustic_dim`), not half of it: the duration bits after it
  follow their own cfg scale. Splitting at `latent / 2` corrupts the
  last acoustic channels and first duration bits.
- The CFG pair is two BATCH rows in the reference. Feeding both rows
  as a two-token sequence through a single-batch engine lets pos and
  neg attend each other and silently diverges (the nocfg twin stays
  green, so only the cfg fixture catches it).
- `mx.linspace(5.0, -5.0, steps + 1)` then `sigmoid(-x / 2)` never
  actually reaches 0 or 1; the reference explicitly pins the first and
  last grid points, and the Euler integration uses `t_curr` (not the
  interval midpoint) for the cfg schedules.
- The tied head means logits are `hidden @ embed_tokens.T`; there is
  no separate lm_head tensor (the reference sanitize drops it).
- The zero-shot skip counts the structural prefix WITHOUT the BOS
  (matching PyTorch), which is one less than the ids the builder
  emits before the text.
