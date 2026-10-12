# LongCat-AudioDiT family (meituan-longcat LongCat-AudioDiT)

Port of `mlx_audio/tts/models/longcat_audiodit/` (longcat_audiodit.py,
config.py, dit.py, text_encoder.py, vae.py) at mlx-audio 0.5.7, pinned
to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/longcat_audiodit)
(mlx 0.32.3). Conditional flow matching over a latent space at 24 kHz:

- a frozen UMT5 text encoder (`text_encoder.py`): the T5 v1.1 encoder
  structure with a relative-attention-bias table in every block (the
  UMT5 signature), RMS-style layer norms, gated exact-GELU feed
  forwards, no score scaling, and the `text_norm_feat` /
  `text_add_embed` post-processing (affine-free layer norm of the
  final hidden state added to the normed initial embedding);
- a CrossDiT transformer (`dit.py`): Qwen2-style rotary embeddings
  (base 100000, half-split rotation) over both the latent and text
  axes, global AdaLN conditioning (`t + masked text mean` through one
  MLP, per-block scale/shift residuals), self attention plus cross
  attention with qk RMS norms, a ConvNeXtV2 text-processing stack
  (depthwise conv channels-last, GRN bottleneck), latent-condition
  embedders (the input and zero latent concatenation), and the long
  skip - including the reference's mid-stack add at the `repa_dit_layer`
  (the generate path passes `return_ith_layer` and its `x += x_clone`
  fires twice);
- the Euler ODE (`longcat_audiodit.py`): `linspace(0, 1, steps)`
  integration of the velocity field from pure noise, with
  classifier-free guidance (`pred + (pred - null) * cfg_strength`
  against a zero-text null branch) or APG (momentum-smoothed difference
  projected onto the conditional prediction, eta 0.5), the heuristic
  duration estimator (CJK 0.21 s and alphabetic 0.082 s per character,
  capped, stretched by the prompt-duration ratio for voice prompts),
  and the prompt-frame interpolation `prompt_noise * (1 - t) +
  latent_cond * t`;
- the WAV-VAE codec (`vae.py`): a channels-last conv autoencoder with
  SnakeBeta activations, dilated residual units, pixel
  shuffle/unshuffle shortcuts (the decoder shortcut adds to the
  POST-unit value, the encoder's averaging shortcut to the head conv),
  and the reparameterization draw `normal * (softplus(scale) + 1e-4) /
  0.71` in the encoder.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: the config sections with the reference defaults
  (`VaeConfig`, `TextEncoderConfig`, `ModelConfig`), the text helpers
  (`normalize_text`, `approx_duration`, `prompt_text`), the `Noise`
  provider trait with `SeededNoise`, the APG momentum buffer and
  update, and the generate flow (encode text, prompt-audio encode,
  duration estimate, masks, Euler loop with cfg/APG, VAE decode).
- `text_encoder.rs`: the UMT5 encoder with the per-block relative
  bias and the bidirectional bucket function.
- `dit.rs`: the transformer (embedders, ConvNeXtV2, attention, AdaLN,
  long skip with the repa add, final AdaLayerNormZero head).
- `vae.rs`: the WAV-VAE encoder and decoder (channels-last conv and
  conv transpose with the MLX weight layouts `[out, kernel, in]`,
  SnakeBeta, residual units, pixel shuffle shortcuts, the mean/scale
  split with the reparameterization draw).
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch 1 and non-streaming; the streaming chunk decoder
  (`_stream_decode`, cosine-crossfaded overlapping VAE windows) is out
  of scope.
- Text tokenization is caller-owned: the reference downloads
  `google/umt5-base` with transformers inside generate. The caller
  tokenizes [prompt_text]; `generate` takes `input_ids` and
  `attention_mask`.
- Noise comes from the [`Noise`] provider: fixture tests replay the
  recorded reference `mx.random.normal` draws (the voice prompt's VAE
  reparameterization draw first, then the initial latent noise);
  standalone runs seed the crate-local threefry RNG, which does not
  reproduce an MLX draw.
- `voice`/`speed`/`lang_code` are unused in the reference and dropped;
  `repa_dit_layer` is honored only for its long-skip side effect (the
  recorded hidden state is never read by generate).
- The reference generate reseeds `mx.random.seed(seed)` internally;
  the port has no global RNG, so draw order is the contract instead.
- `sigma` and the training-side fields are parsed but unused in the
  decode path, matching the reference.

## Verification

Fixture protocol identical to the other families:
`tools/gen_longcat_audiodit_fixtures.py` runs the true reference
modules driven from `../mlx-audio` at the pinned commit (with the
transformers tokenizer patched to a local character tokenizer):

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_longcat_audiodit_fixtures.py
```

Fixtures live in `testdata/tts_longcat_audiodit/` with SHA-256 digests
in `manifest.json`. Seeded tiny stacks: a 2-layer UMT5 encoder
(d_model 32, 2 heads, d_ff 64, vocab 96), a 2-layer DiT (dim 32, 4
heads, ff 128, global AdaLN, qk norms, text conv, long skip, repa
layer 1 so the mid-stack add is exercised, latent 16), and a tiny
WAV-VAE (channels 16, c_mults [2, 4], strides [2, 4], hop 8, 8 kHz);
all weights round-trip through f16 so the traces come from exactly the
values the Rust fixture loads. Cases:

- `text/` `normalize_text` (smart quotes, whitespace collapse) and
  `_approx_duration` (capped and uncapped) on sample inputs.
- `dit/` the transformer forward on fixed inputs with a masked text
  tail (exercising the embedder zeroing and the masked text mean) and
  the repa long-skip add.
- `generate/` end to end without a prompt: cfg 4.0, 4 Euler steps, the
  recorded latent draw, 400 decoded samples.
- `voice/` reference-audio prompt: the padded VAE encode (recorded
  draw, prompt latent parity), the prompt-frame interpolation, and the
  waveform.
- `apg/` the generate twin under `guidance_method="apg"`.

Measured when the gates were frozen: the DiT forward within 2e-4
absolute, prompt latents within 2e-4 relative (the tiny VAE rides in
the hundreds), and waveforms within 2e-4 relative of the reference
magnitudes (the unbounded conv decoder scales with the fixture's
random weights); draw counts exact.

`cargo test -p turbospark-audio --lib tts::longcat_audiodit` runs the
family gate.

## Open items

- No real-checkpoint gate: an end-to-end run against
  `mlx-community/LongCat-AudioDiT-1B-bf16` (24-layer 1536-wide DiT,
  12-layer UMT5 with the 256k vocab, full 2048-hop WAV-VAE, real
  tokenizer, 24 kHz waveform sanity against the Python family) has not
  been executed; the fixtures are seeded tiny instances.
- Sampled runs (standalone `SeededNoise`) draw different noise than
  mlx; the fixture parity replays the recorded draws.
