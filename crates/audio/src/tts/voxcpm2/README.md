# VoxCPM2 (2B, tokenizer-free, 48 kHz)

Port of `mlx_audio/tts/models/voxcpm2/` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/voxcpm2)
(MLX 0.32.3). Spec task 37. The voxcpm architecture with the V2
changes; the MiniCPM backbone, the flow solver, the FSQ quantizer, and
the causal-conv building blocks are shared with
[voxcpm](../voxcpm/README.md) (additive `kv_channels` / `no_rope`
there; `no_rope` skips the rope tensors and the rotation, `kv_channels`
overrides the head dim).

## V2 differences from voxcpm

- `use_mup` defaults to TRUE and the embedding scale inverts:
  `embed * scale_emb` when `use_mup` (voxcpm scales when NOT mup).
- The conditioning additions become concatenations: the residual input
  is `fusion_concat_proj([enc_outputs | audio_mask * feat_embed])`,
  each step is `fusion_concat_proj([lm_hidden | curr_embed])`, and the
  DiT conditioning is the 2-wide pair
  `[lm_to_dit(lm_hidden), res_to_dit(residual_hidden)]`.
- The DiT estimator (`dit.rs`, `VoxCPMLocDiTV2`) reshapes that pair
  into two mu TOKENS and prepends a separate delta-time token:
  `[mu_1 | mu_2 | t_comb | cond_proj | x_proj]`, slicing everything
  before the patches on the way out. `mean_mode` (the config's
  `dit_mean_mode`) feeds the live delta time instead of zero; both
  still route through the delta-time embedding tower.
- The prompt grammar gains special tokens: reference cloning wraps the
  encoded patches as `[103 | zero-token rows | 104]` with
  `[z1 | patches | z1]` features; continuation prompts append the
  left-padded prompt patches after the text rows; the combined mode
  concatenates both prefixes. `warmup_patches` generate-and-discard
  conditioning patches in non-continuation modes, and the stop head
  gates on `real_steps > min_tokens` (default 2) instead of the fixed
  `i > 5`.
- The AudioVAE (`audio_vae.rs`) encodes at 16 kHz (hop
  `prod(encoder_rates)`) and decodes at `out_sample_rate` (48 kHz)
  with per-block sample-rate conditioning: a `scale_bias`
  `SampleRateConditionLayer` applies `x * scale_embed[bucket] +
  bias_embed[bucket]` before every decoder block, the bucket being the
  count of `sr_bin_boundaries` the output rate reaches. Array inputs
  ride the `scipy.signal.resample` Fourier resample (16 kHz <- 48 kHz)
  before the patch padding. `use_noise_block` (fresh random draws per
  decode) is refused: no deterministic Rust counterpart.
- The streaming machinery only feeds the continuation prefix: the last
  `streaming_prefix_len - 1` prompt patches join the decode sequence
  and the equivalent front samples are trimmed from the waveform.

Scope: single batch, non-streaming, caller-owned text tokenization
(the reference `_tokenize` plus the multi-char Chinese split is
tokenizer-side; `(instruct)` voice design is a caller string concat).
Reference mode encodes `ref_text + text` as ONE string (the reference
concatenates before encoding; boundary words can merge), continuation
mode `prompt_text + text`.

## Verification

`tools/gen_voxcpm2_fixtures.py` builds seeded tiny instances of the
real reference modules (2-layer backbones at hidden 32, 4 query / 2 kv
heads, a 1-block AudioVAE V2 with latent 8 and 4 SR buckets, 16 kHz
encode / 48 kHz output, a word-id mock tokenizer with `tokenize` /
`convert_tokens_to_ids`) and records, per case in
`testdata/tts_voxcpm2/<case>/`:

- `zero_shot/`: stop head fires at `real_steps == min_tokens + 1`.
- `length/`: runs to max_tokens.
- `ref_clone/`: reference-audio prefix; traces also pin the scipy
  resample and the VAE patch encode stages.
- `continuation/`: prompt-audio suffix with the decode front trim.

The generator enforces the recording caveat before freezing: a fresh
`model.generate` and a second replay must reproduce the first replay's
waveform (`np.array_equal`) and the full draw stream.

The Rust tests reload each case through `Model::open` and replay the
draws. Measured parity (relative to the reference magnitude): the
scipy Fourier resample 1.4e-7, the VAE patch encode 7.0e-7, prompt
and backbone stages 9.4e-7 or below (2e-4 gate); the generate loop
stayed at or below 1.5e-5 on latents and 2.7e-6 on waveforms across
all four frozen cases - these carry the same transcendental floor as
voxcpm (see that README) and are gated at 2e-2 relative for margin
(`VOXCPM2_MEASURE=1` re-prints). A dedicated test also pins the
threefry RNG reproducing the recorded MLX normal stream from the same
seed.

Real-checkpoint gate NOT run: no `mlx-community/VoxCPM2-*` checkpoint
was downloaded in this porting round. Open items: a real
load-and-decode pass, voice quality, the `mean_mode` and
`cond_out_layer` checkpoint variants (refused), and runtime
integration.

## Gotchas found during verification

- The reference encodes `prompt_text + text` with NO separator; the
  fixture generator must replay that exact concatenation or the
  boundary word merges differently and the whole run diverges.
- `scipy.signal.resample` (the installed version) truncates the rfft
  to `min(num, n)/2 + 1` bins, DOUBLES the new-length Nyquist bin when
  downsampling to an even length, and divides by `n/num`. numpy's
  irfft ignores the Nyquist bin's imaginary part, but the shared
  crate real-IFFT plan reads it, so the port zeroes it explicitly.
- MLX conv-transpose weights store `[out_ch, K, in_ch]` (block 0 maps
  `in 32 -> out 16` at `K 8` as `[16, 8, 32]`), while the encoder's
  plain conv stores `[out, K, in]`; the loaders accept both that and
  the PyTorch layouts plus unfused `weight_g`/`weight_v` pairs.
- The recorded reference ids in clone mode BEGIN with the
  `[103 | zeros | 104]` prefix the model builds itself; the Rust tests
  strip it before passing the text ids (the port builds the prefix
  from the encoded audio).
