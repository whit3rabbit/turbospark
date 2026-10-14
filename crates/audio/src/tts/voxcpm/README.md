# VoxCPM (1.5)

Port of `mlx_audio/tts/models/voxcpm/` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/voxcpm)
(MLX 0.32.3). Spec task 77 (`voxcpm`, alias `voxcpm1.5`).

## Architecture

Tokenizer-free in the codebook sense: there are no audio tokens. Four
MiniCPM backbones (`minicpm.rs`, a family-local port) over one shared
math:

- `base_lm`: the text LM with the embed table (`embed * scale_emb`,
  12). LongRoPE factor tables swap on the `original_max_position_
  embeddings` threshold, and `use_mup` rescales both residual adds by
  `scale_depth / sqrt(layers)` (off in this checkpoint family).
- `residual_lm`: vocabulary-free twin over the fused LM state.
- `feat_encoder` (`encoder.rs`): the CLS patch encoder. A special
  token (a saved random-init parameter) is prepended to every
  `[patch x feat_dim]` window and each row runs through a
  bidirectional pass; position 0 of every row is the patch embedding.
  The reference `(B*T, P+1, H)` batch is folded into one sequence with
  a blocked mask and per-block rope positions, which is bit-equivalent
  at single batch.
- `feat_decoder` (`dit.rs`): the DiT denoiser inside the
  `UnifiedCFM` Euler flow solver. Hidden = `[mu + t_emb + dt_emb |
  cond_proj | x_proj]` with full attention; the solver uses the sway
  t-span (`linspace(1, 0, n+1)` plus the cosine adjustment), the
  CFG zero-star trick (`st* = <cond, uncond> / (||uncond||^2 + 1e-8)`
  with `max(1, int(len*0.04))` zero-derivative warmup steps), and one
  recorded `mx.random.normal` draw per generate step. The reference
  doubles the CFG batch; the rows are independent sequences so the
  port runs them one at a time.

A two-linear stop head (`stop_head(silu(stop_proj(lm_hidden)))`)
ends the run at class 1 only after step 5 (`i > 5`); the sampled
latent at the stop step IS kept and decoded. Voice cloning splices
VAE-encoded reference patches plus ref-text embeddings into the prefix
(`fsq(enc) * audio_mask + enc * text_mask`, residual input adds
`audio_mask * feat_embed`). The concatenated patches decode through
the causal-convolutional AudioVAE (`audio_vae.rs`): Snake activations,
residual units at dilations 1/3/9, depthwise residual convs
(`depthwise=True` is hardcoded in the reference constructor),
left-padded causal convs (the constructor `padding` is the symmetric
amount; the forward left-pads twice that), right-trimmed transposed
convs, weight-norm folded at load (`weight_g`/`weight_v` pairs fuse
exactly like the reference sanitize; fused `weight` tensors are also
accepted), and a tanh output. The noise block is never constructed
(`use_noise_block=False` is hardcoded); `noise.` tensors under a
decoder block are refused.

Scope: single batch, non-streaming (the reference yields one result),
caller-owned text tokenization (encode the bare text; the
`audio_start_token` 101 is appended here). Flow noise comes from a
[`FlowNoise`] provider - recorded-draw replay for fixtures, the
crate-local threefry RNG (seeded) for standalone runs, which
reproduces the recorded MLX draws bit-for-bit (asserted by a fixture
test). The `audiovae.pth` PyTorch fallback load is not ported; weights
are expected in the safetensors layout the reference sanitize
produces.

## Verification

Fixture protocol identical to the codec families:
`tools/gen_voxcpm2_fixtures.py`'s sibling
`tools/gen_voxcpm_fixtures.py` builds seeded tiny instances of the
real reference modules (2-layer backbones at hidden 32, 4 query / 2
kv heads, vocab 128, a 2-block AudioVAE at 24 kHz with hop 8, a
word-id mock tokenizer, per-case stop-head biasing) and records, per
case in `testdata/tts_voxcpm/<case>/` (config.json + model.safetensors
+ traces.json, digests in manifest.json):

- `generate/`: zero-shot, stop head fires at step 6 (7 patches).
- `length/`: zero-shot, stop silenced, runs to max_tokens.
- `voice_clone/`: reference audio + ref text (the concatenated
  `ref_text + text` encode includes a merged boundary word, which the
  mock tokenizer maps through a stable fallback id).
- `longrope/`: the 21-position prompt crosses
  `original_max_position_embeddings` = 16, flipping the factor table
  inside the same run.

Traces carry the prompt stages, per-step latents/stop flags,
euler t-spans, the first six raw estimator calls, the AudioVAE decode
stages, the full draw stream, and the waveform. The generator
enforces the recording caveat before freezing: run 2 (a fresh
`model.generate`) and run 3 (a second replay) must reproduce run 1's
waveform with `np.array_equal` and the identical draw stream.

The Rust tests reload each case through `Model::open` and replay the
draws. Measured parity (relative to the reference magnitude):
prompt embeds exact, `enc_outputs` 4.4e-7, `lm_hidden` 1.0e-7,
`residual_hidden` 1.3e-6, the reference-audio patch encode 7.4e-7,
all against a 2e-4 gate. The end-to-end loop is gated separately:
the sinusoidal time embedding evaluates sin/cos at angles up to 1000
radians, where MLX's f32 argument reduction differs from Rust's
correctly-rounded libm by up to 5e-5, and each sampled latent feeds
the next conditioning step, so diffs grow roughly 3.5x per step
(measured on the frozen fixtures: step-0 latent 2.9e-6 relative
rising monotonically to 1.5e-2 on the final 8-step patch, waveform
1.1e-2). The latents/waveform gate is frozen at 2e-2 relative - above
the measured worst case (1.35x margin on that worst step; the
per-step latents, all exact within 2e-3 relative until the last
steps, are the sharper regression signal). `VOXCPM_MEASURE=1`
re-prints every diff.

Real-checkpoint gate NOT run: no `mlx-community/VoxCPM1.5` checkpoint
was downloaded in this porting round. Open items: a real
load-and-decode pass (config fields, tensor names, the `.pth`
audio-VAE path, quantized variants), voice quality, and
runtime/FFI/Swift integration.

## Gotchas found during verification

- The reference rotates rope pairs per HEAD BLOCK: the q tensor is
  `[row, heads * dim]` and every head owns the pairs `(d, d + dim/2)`
  inside its own `dim` slice. Rotating only the first head's pairs
  (position 0 is angle 0, so head 0 row 0 still matches) corrupts
  every other row while row 0 of the prefill stays exact.
- The MiniCPM cache is `(B, L, H_kv, D)` in the reference and the
  stored keys are the ROPED ones. Appending a `[heads, new, dim]`
  block flat onto the head-major `[heads, total, dim]` buffer lands
  the new rows inside the wrong head's region (the same gotcha the
  lm README records); the shared `lm::cache::KvCache` owns the
  per-head insert.
- MLX conv weights are stored `[out, K, in/groups]` and conv-transpose
  weights `[out_ch, K, in_ch]` (the fixture pins it: decoder block 0
  maps `in 32 -> out 16` with `K 8` and stores `[16, 8, 32]`); both
  rotate to the PyTorch layouts the codec kernels take at load.
- The stop head may report class 1 from the very first step (the
  fixture biases it that way); the reference simply ignores it until
  `i > 5`, so the parity target is the recorded flag sequence, not a
  0-then-1 ramp.
- MLX's `mx.round` (used by the FSQ quantizer) and Rust `f32::round`
  agree (half away from zero); the tanh feeding it is accurate to
  ulps on both sides and no level flip was observed on the fixtures.
