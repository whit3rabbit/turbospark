# Irodori TTS

Port of `mlx_audio/tts/models/irodori_tts/` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/irodori_tts)
(MLX 0.32.3). Spec task 73 (Japanese flow-matching TTS, repo family
`mlx-community/Irodori-TTS-*`; upstream `Aratako/Irodori-TTS`).

## Architecture

A Rectified Flow DiT over continuous DACVAE latents at 48 kHz
(`Semantic-DACVAE-Japanese-32dim`, latent_dim 32, hop 1920):

- **DiT backbone**: gated joint attention over latent self-tokens,
  text context, and speaker/caption context. Each branch carries its
  own key/value projections (`wk_text`/`wv_text`, `wk_speaker`/...);
  the self q/k take q/k RMSNorm over `(heads, head_dim)`, and the
  reference `_apply_rotary_half` applies FULL head-dim interleaved RoPE
  to the FIRST HALF OF THE HEADS only (`y[..., :y.shape[-2] // 2, :]`
  slices the heads axis despite the "head dimensions" docstring).
  Attention output passes a sigmoid gate; both attention and SwiGLU MLP
  are conditioned through low-rank AdaLN (`up(down(silu(part))) +
  part` per third of the timestep embedding, tanh gate). The timestep
  module is a five-layer Sequential (Linear, SiLU, Linear, SiLU,
  Linear); MeanFlow checkpoints add a delta-conditioning module on the
  sampling interval `t - r`.
- **Text encoder**: the scratch bidirectional encoder (embedding +
  gated self-attention blocks with full per-head RoPE and mask zeroing
  after every block) for v1-v3, or the v4 PRETRAINED ModernBERT
  backbone (`modernbert.py`, bundled weights: fused `Wqkv`, rotate-half
  RoPE with per-layer-type theta, GeGLU MLP, first-layer Identity
  attn_norm, global/local window alternation with the diagonal always
  open, `finfo.min` additive masks) feeding separate text and caption
  projectors (`linear` or `residual_mlp`).
- **Speaker encoder**: `ReferenceLatentEncoder` over speaker-patched
  DACVAE latents (in_proj / 6.0, non-causal blocks).
- **Duration predictor** (v3+): token-sum architectures
  (`token_sum_adarn_zero_no_aux`, dual caption variant) with zero-init
  AdaRN-Zero modulation, softplus per-token frames, and the 14-dim
  feature row (`build_duration_features`, kana/kanji/ascii classes and
  the annotation-emoji allowlist). The pooled legacy architectures are
  refused.
- **Samplers**: `sample_euler_cfg` (per-condition CFG in independent
  and joint modes with the `cfg_min_t`/`cfg_max_t` window, sway or
  linear schedule from 0.999, optional truncation factor, temporal
  score rescaling, speaker-KV scaling with the one-time rollback at
  `speaker_kv_min_t`) and the MeanFlow distilled sampler (single
  conditional branch, `delta_t` conditioning). The reference batches
  the CFG bundles into one forward pass; the port runs them
  sequentially (rows are independent).
- **Codec**: the shared [DACVAE codec port](../../codec/dacvae/README.md)
  (encode for reference audio, decode for the waveform). The
  reference's chunked decode with crossfade is a memory optimization
  the port replaces with the full pass.

## Scope and deviations

- Single batch, non-streaming (the reference raises on `stream=True`).
- Caller-owned Japanese text normalization (including the NFKC pass)
  and tokenization: `generate_latents` takes the padded token ids and
  masks; an EMPTY caption must arrive fully masked (the reference
  zeroes the caption mask itself). `build_duration_features` takes the
  already-normalized text.
- `cfg_guidance_mode='alternating'` is refused (Unsupported); pooled
  duration architectures are refused; `delta_t` on an RF model and a
  missing `delta_t` on a MeanFlow model are input errors.
- The reference `forward_with_conditions` caption-only backward-compat
  routing (speaker slots carrying caption state) is collapsed: the port
  routes by the resolved configuration flags.
- The DACVAE checkpoint loads from the bundled `dacvae/` folder
  (`post_load_hook`'s local path); a hub download fallback is
  caller-owned. The codec is optional at load (the reference warns and
  leaves it None) and the decode path refuses without it.

## Verification

Fixture protocol identical to the codec families:
`tools/gen_irodori_tts_fixtures.py` builds seeded tiny instances of the
real reference modules (a 2-layer DiT at model_dim 32 with 4 heads, a
3-layer tiny ModernBERT backbone with linear/residual_mlp projectors, a
1-layer speaker encoder with speaker_patch_size 2, a 2-layer token-sum
duration predictor, and the tiny DACVAE from the codec fixtures bundled
as `dacvae/`) and records, per case in
`testdata/tts_irodori_tts/<case>/` (config.json + model.safetensors +
traces.json, digests in manifest.json):

- `duration/`: scratch encoder, speaker-only, the duration chain
  (features, conditions, `predict_duration_log_frames`, clamp).
- `generate_rf/`: pretrained ModernBERT, dual speaker+caption,
  independent CFG over the sway schedule with a `cfg_min_t` floor that
  skips the last step, 3 Euler steps, DACVAE decode, silence trim.
- `generate_joint/`: scratch encoder, speaker-only, joint CFG (equal
  scales) with speaker-KV scaling and its rollback at `speaker_kv_min_t`
  inside the run, manual seconds.
- `meanflow/`: MeanFlow distillation with delta conditioning, 3 steps.
- `codec/`: standalone reference-audio encode/decode through the family
  path.

The generator verifies every sampler replay against the reference
sampler entry point on the Python side. The reference batches the CFG
rows; the kernel-fusion drift accumulates over the Euler feedback
(1.5e-5 to 2e-5 on the unit-scale latents, pinned under 1e-4), so the
PINNED path is the sequential replay, which is also what the Rust port
runs; the MeanFlow case agrees bit-exactly.

The Rust tests reload each case through `Model::open` and replay the
draws. Measured parity: duration features exact, codec encode 8.9e-8
and decode 1.6e-7, latent gates 4.9e-5 (rf) / 5.1e-5 (joint) /
7.1e-5 (meanflow) against the 2e-4 step gate (three Euler accumulations
of the about 3e-6 cond-module SiLU/matmul ordering drift), waveforms
2.1e-7 against 1e-3 (the decode amplifies the latent delta through the
transposed convolutions). The threefry draw stream matches the recorded
MLX draws under 1e-6.

Bugs the fixtures caught during bring-up: the AdaLN branches apply SiLU
to the branch INPUT (`up(down(silu(part)))`, not `silu(down(part))`);
the half-RoPE slices the heads axis, not head dims; the scratch
encoder's RoPE runs per head over head_dim on the reshaped projection
(a partial rotary of the full width would double-count); the caption
context must join the no-CFG forward too; and the reference
`text_mlp_ratio`/`speaker_mlp_ratio` default to 2.6 (not None) before
the `mlp_ratio` fallback.

Open item: the real-checkpoint gate. The fixtures are seeded tiny
instances; the `mlx-community/Irodori-TTS-v4.1-Small-*` downloads have
not been run through this port.
