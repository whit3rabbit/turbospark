# DramaBox TTS (LTX-2.3 audio-only DiT flow matching)

Port of `mlx_audio/tts/models/dramabox/` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/dramabox)
(MLX 0.32.3). Spec task 69 (`dramabox`, alias `dramabox-tts`).

## Architecture

Latent flow matching over patchified audio-mel latents
(`[tokens, channels * mel_bins]`, 25 fps-aligned token counts):

- Text conditioning: the Gemma-3 text encoder is a SEPARATE downloadable
  model (`mlx-community/gemma-3-12b-it-8bit`), so text conditioning is
  caller-owned: the caller supplies the stacked Gemma hidden states
  `[tokens, hidden * layers]` (hidden dim OUTER, layer INNER per token)
  plus the attention mask, and the ported conditioner runs
  `FeatureExtractorV2` (per-token per-layer RMS over the hidden dim,
  mask zeroing, the `sqrt(cross_dim / hidden)` rescale, one biased
  projection) and `Embeddings1DConnector` (the register reorder that
  walks valid tokens to the front and refills the tail with tiled
  learnable registers, the rope'd gated-attention blocks with the split
  LTX rope, and the final plain RMSNorm).
- The DiT (`AudioOnlyLTXModel`): AdaLN-single timestep conditioning
  (PixArt sinusoidal projection with the flip to cos-first, per-token
  timesteps from the denoise mask times the sigma), split LTX rope over
  the causal patch-grid time bounds
  (`max(d * (k - 1) + 1, 0) * hop / sample_rate`, middle-index grid),
  gated attention (2 * sigmoid per head), cross attention to the text
  context with the prompt-AdaLN key/value modulation (one sigma token
  broadcast over every context token), and the AdaLN-zero scale/shift
  output head over an affine-free LayerNorm.
- The scheduler: the LTX2 sigma schedule (token-count-shifted logistic
  sigmas with the terminal stretch to 0.1), the x0-model Euler loop
  with the multi-modal guider (cfg + STG + modality terms, the
  `auto_rescale_for_cfg` std rescale), and the long-clip silence prior
  patch on the unpatchified latents.
- Duration estimation: the reference heuristic (quoted-speech
  extraction, length-tiered characters-per-second, punctuation pauses,
  the nonverbal keyword table, the contextual laugh modifiers, and the
  25 fps / 8-token frame alignment) is ported with whole-word literal
  matchers.

## Scope and refusal

Batch 1, f32, single prompt. The pipeline ENDS AT THE MEL LATENTS
(`[channels, frames, mel_bins]`, silence-prior patched):
`decode_waveform_refusal` refuses the AudioVAE decoder and vocoder
decode. UNLIKE the tada-codec case these two modules ship in the main
download's `dramabox-audio-components.safetensors`, so this is the one
open porting item of this family (an in-checkpoint conv/VAE stack,
roughly 1200 reference lines), not a separate-download refusal. The
reference-audio voice-clone path additionally needs the AudioVAE
encoder and resampling and is not ported. The bf16 working dtype of
the reference generate wrapper is f32 here (the modules are
dtype-generic; the fixture pins the f32 path).

Noise: one standard-normal draw over the patched latent
(`add_gaussian_noise`), from a [`DiffusionNoise`] provider.

## Verification

Fixture protocol identical to the codec families:
`tools/gen_dramabox_fixtures.py` builds seeded tiny instances of the
real reference modules (2-block DiT, heads 4, head_dim 8, inner 32,
patch 16 = 4 channels x 4 mels, cross dim 32; a 1-block connector with
heads 4, head_dim 8 and 4 learnable registers over an 8-token grid; 4
flow steps) and records, per case in
`testdata/tts_dramabox/<case>/` (config.json + model.safetensors +
traces.json, digests in manifest.json):

- `generate/`: the guided loop (cfg 2.0, stg 1.0 on block 1, auto
  rescale): every x0 call (cond, uncond-text, STG-perturbed per step),
  the sigma schedule, the conditioner intermediates, the AdaLN and rope
  tables, and the final silence-patched latents.
- `plain/`: the cfg-1.0/stg-0.0 single-pass twin.

The conditioner rides the fixture through the engineered Gemma hidden
states (the text encoder being a separate download), and the Rust test
rebuilds the contexts through the ported conditioner before the loop.
A Python-side replay check (fresh weights plus the recorded draw
stream) gates every freeze.

Measured parity on the frozen fixtures: latents match inside the 2e-4
relative gate; the per-stage probes used during bring-up pinned the
conditioner output, the sigma schedule, the AdaLN/rope tables, every
guided x0 call, and the output head (all at 1e-5 relative or below
once fixed). `DRAMABOX_MEASURE=1` re-prints the diffs. Unit tests pin
the duration estimator, the frame alignment, the config defaults, and
the waveform refusal.

Real-checkpoint gate NOT run: the DramaBox checkpoint was not
downloaded in this porting round. Open items: the AudioVAE decoder +
vocoder port (in-checkpoint, required for waveform output), a real
load-and-generate pass, the Gemma text-encoder caller contract on real
hidden states, voice quality, runtime/FFI/Swift integration.

## Gotchas found during verification

- The AdaLN coefficient is `6 + 3` with cross-attention AdaLN (9 rows
  in the scale/shift table), and the reference `mx.split` semantics
  make the head's shift the FIRST table row and the scale the SECOND,
  broadcast per token over the `[2, D]` table plus the embedded
  timestep. An interleaved `[token, row]` layout there silently
  corrupts the output head by exactly the table rows.
- The STG "all-perturbed" pass does NOT bypass the gated attention:
  `if all_perturbed: out = value` skips only the q/k/rope/attention
  computation; the 2 * sigmoid gate and the out projection still run.
  Returning the raw values early diverged the guided loop from step 0.
- The split LTX rope cuts the `[tokens, inner/2]` frequency row across
  the heads (`[T, heads, head_dim/2]`), it does not broadcast the same
  row to every head, and the connector's per-token frequencies ride
  the max_pos division of the token index as both grid bounds.
- The reference timestep embedding flips the concatenated `[sin, cos]`
  halves, so COS comes first (`flip_sin_to_cos = True`).
- The causal patch-grid time bound is `max(d * (k - 1) + 1, 0)`, not
  `k + 1 - d`: the arange multiplies by the downsample factor before
  the causal shift.
- `ltx2_sigmas` derives its shift from `math.prod(latent.shape[2:])`
  of the PATCHED latent, i.e. channels * mel_bins, not the token count.
- The FeatureExtractorV2 layer concat is hidden-OUTER per token
  (`[d * L + l]` after the stack on the last axis), and the rescale is
  `sqrt(cross_dim / gemma_hidden)`, not 1.
- `estimate_speech_duration` floors AFTER rounding to one decimal, and
  `resolve_generation_duration` multiplies the ALREADY-floored estimate
  by the duration multiplier before its own floor.
