# Dia family (Nari Labs Dia 1.6B)

Port of `mlx_audio/tts/models/dia/` (dia.py, config.py, layers.py,
audio.py) at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/dia)
(mlx 0.32.3). A two-stack sequence-to-sequence dialogue TTS. The text
encoder consumes the prompt as raw utf-8 bytes with the `[S1]`/`[S2]`
speaker marks replaced by the bytes 0x01/0x02, right-padded (or
truncated) to `text_length`, with a JAX-segment-style attention mask
where positions agree to attend only to positions of the same pad
status. The decoder sums one embedding table per audio channel,
predicts one code per channel per step through grouped-query self
attention and precomputed cross attention, and is run batch 2: a
zero-text unconditional row travels alongside the conditional row and
the logits combine as `cond + cfg_scale * (cond - uncond)`.

Per-step sampling masks the CFG logits past code 1024 (a hardcoded
keep width of 1025: the codebook plus EOS; PAD 1025 and BOS 1026 are
masked) and either takes the per-channel argmax (temperature 0) or
applies the reference `make_sampler` chain - top-p on the raw CFG
logits, then top-k (`cfg_filter_top_k`, default 35) - and draws. The
channel-delay pattern (`[0, 8, 9, ..., 15]`) force-BOSes each channel's
prediction until its delay elapses. Channel-0 EOS starts a fixed
30-step countdown; during the countdown the delayed channels receive
EOS/PAD bookkeeping added on top of the freshly sampled codes at
position `step + 1` (which can push ids past the vocabulary - see the
deviations). The sliced code grid drops its first frame and the final
30 frames, clamps codes outside `[0, 1023]` to 0, and decodes through
the Descript Audio Codec 44 kHz checkpoint.

The DAC is a separate downloadable model
(`mlx-community/descript-audio-codec-44khz`), already ported in
[codec/descript](../../codec/descript/README.md); `Model::generate`
takes a caller-loaded `Dac`. Optional voice prompts run the DAC encoder
over the reference audio, apply the forward delay pattern, and prefill
the decoder caches before the loop.

Everything is portable single-run f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: the generate parameters, the `SampleSource` trait with the
  `Greedy` and `Sampled` implementations, segment splitting
  (`split_turns`, the `generate` segment loop), text preparation, the
  encoder and cross masks, and the decode loop (delay forcing, CFG
  combine, keep mask, EOS countdown, code slicing).
- `config.rs`: the `DiaConfig` section parser with the reference
  defaults and the round-up-to-128 length rule.
- `layers.rs`: `DenseGeneral` kernels flattened and transposed into
  crate linears, the fused-gate MLP, the timescale rotary embedding
  (`min * (max/min)^(2d/dim)`, sinusoid `position / timescale`), and
  the `[2, heads, max_len, head_dim]` KV stores.
- `model.rs`: the encoder and decoder stacks, per-channel embedding
  lookup with the MLX out-of-range semantics, and the cross-attention
  precompute.
- `audio.rs`: the delay/revert index math, `audio_to_codebook`, and
  `codebook_to_audio` with the invalid-code clamp.
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch 2 (the reference's unconditional/conditional pair), one segment
  at a time, non-streaming. The reference `generate` yields per-segment
  results; the port returns them all.
- The reference `generate` never forwards `temperature`/`top_p` to
  `_generate` (they stay at the `_generate` defaults 1.3/0.95); the
  port exposes them through `GenerateParams` and honors them.
- Text tokenization is the reference byte scheme (no external
  tokenizer); `ref_text` is trimmed and prepended with a space exactly
  like the reference.
- The decode-time cross mask uses the reference's fixed query pad flags
  (generated rows are never PAD). The prefill path computes per-position
  pad flags generally.
- The reference allocates the self-attention caches with
  `max_tokens` rows and asserts when a reference-audio prompt prefill
  plus the decode budget exceeds it; the port sizes the caches from the
  actual prompt length plus budget, so a voice prompt that would trip
  the reference assert runs instead.
- `training.dtype` other than `float32` is a config error: the
  reference would compute activations in that dtype and a portable f32
  port cannot reproduce it.
- Out-of-vocabulary embedding ids produce zero rows and negative ids
  wrap Python-style (the observed pinned-MLX gather semantics). This
  matters for the post-EOS countdown: the adds push ids past 1028, and
  the pinned MLX gather reads PAST THE BUFFER there (in practice
  process-dependent memory), so the reference's countdown-phase
  predictions are not reproducible bit for bit. The port defines the
  zero row; the fixtures pin the clean prefix and replay the countdown
  choices (see Verification).
- An EOS at the very start of a segment leaves fewer than 30 frames
  after the tail trim; the reference crashes inside the quantizer
  (a conv over a zero-length grid). The port refuses with a descriptive
  error instead.

## Verification

Fixture protocol identical to the other families:
`tools/gen_dia_fixtures.py` runs the true reference modules driven from
`../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_dia_fixtures.py
```

Fixtures live in `testdata/tts_dia/` with SHA-256 digests in
`manifest.json`. Seeded tiny stacks: a 2-layer DiaModel (encoder hidden
32 with 4 heads; decoder hidden 32 with 8 GQA query heads over 2 kv
heads and 4 cross heads; vocab 1028; nine channels; text_length 128)
and a tiny nine-codebook DAC (codebook 1024x4, hop 8, 44.1 kHz); all
weights round-trip through f16 so the traces come from exactly the
values the Rust fixture loads. Cases:

- `text/` the byte tokenizer (S1/S2 replacement, padding, mask) and
  `_split_turns` on a three-pair script (pairwise merge).
- `delay/` `apply_audio_delay` / `revert_audio_delay` and
  `codebook_to_audio` with invalid codes through the tiny DAC.
- `generate/` greedy end to end with a scripted EOS at step 20: the
  countdown runs to its break at step 49, the adds corrupt the late
  grid, and 18 frames decode (144 samples).
- `budget/` natural argmax throughout with EOS suppressed, budget stop
  at 45 steps, 13 frames decode.
- `midstop/` EOS at step 30 with the budget stop at 45: no add fires,
  pinning the exact detection/countdown bookkeeping.
- `voice/` reference-audio prompt: DAC-encoded delayed prompt codes,
  the prefill, and the countdown break (21 frames decode).
- `sampling/` the step-0 CFG logits through the real sampler chain at
  the reference defaults (temperature 1.3, top-p 0.95, top-k 35): the
  filtered support is exact, the drawn token is a support-membership
  check (the draw itself is crate-local threefry).

Every e2e case asserts the Rust per-channel argmax against the
recorded reference natural selection for the clean prefix
(`natural_steps` in the traces: before the first post-EOS add), then
replays the recorded predictions so the sliced code grid is compared
exactly over the countdown region too, where the reference embeds
out-of-vocabulary ids with undefined content. Measured when the gates
were frozen: code grids exact, token counts exact, waveforms within the
2e-4 absolute gate (the tiny DAC decode dominates), filtered sampler
logits within 1e-3 on the finite support.

`cargo test -p turbospark-audio --lib tts::dia` runs the family gate.

## Open items

- No real-checkpoint gate: an end-to-end run against
  `mlx-community/Dia-1.6B` (22-layer stacks, hidden 2048/2304, real
  tokenizer-free byte prompt, 44 kHz waveform sanity against the Python
  family) has not been executed; the fixtures are seeded tiny
  instances.
- Sampled (temperature > 0) decoding is verified only at the filter
  level; the categorical draw uses the crate-local threefry sequence
  and does not reproduce an MLX draw (same policy as the shared
  sampler).
