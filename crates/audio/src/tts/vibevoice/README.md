# VibeVoice (streaming TTS)

Port of `mlx_audio/tts/models/vibevoice/` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/vibevoice)
(MLX 0.32.3). Spec task 76 (`vibevoice`, alias `vibevoice_streaming`).

## Architecture

A split Qwen2 decoder over the shared [lm engine](../../lm/README.md):

- `language_model`: the lower `num_hidden_layers - tts_backbone_num_hidden_layers`
  layers plus the token embed table, with NO final norm (the reference
  `use_norm=False`; the checkpoint carries no `norm` tensor there and no
  lower-half head is ever read, so the loader declares the head tied to
  skip it).
- `tts_language_model`: the upper `tts_backbone_num_hidden_layers` layers
  with the final norm. The checkpoint reuses a full embed table here but
  generation never reads it; `Backbone::load_without_embed_table` skips
  the dead vocabulary-sized buffer.
- `tts_input_types`: a 2-row type embedding, text = 1, speech = 0.

Generation (`Model::generate`, batch 1, non-streaming): text ids enter
the lower backbone in windows of 5 (`TTS_TEXT_WINDOW_SIZE`), get the
text type embedding, and advance the upper backbone; the negative CFG
branch runs the upper backbone on zeros-plus-type-embedding inputs in
parallel. Then up to 6 (`TTS_SPEECH_WINDOW_SIZE`) speech latents per
round: the last upper-backbone hidden states condition the diffusion
prediction head under the deterministic DPM-Solver++ scheduler
(v-prediction, cosine betas, cfg against the negative branch; the
guided pair stays in lockstep so the port steps one half of the
reference's doubled batch), the sampled latent feeds back through the
speech connector with the speech type embedding on both branches, and
the binary `tts_eos_classifier` over the last hidden state ends the run
at p > 0.5 (or `max_tokens`). The concatenated latents are unscaled by
the checkpoint's `speech_scaling_factor` / `speech_bias_factor` and
decoded by the convolutional VAE acoustic tokenizer (the same decoder
architecture the kugelaudio family reuses, ported family-locally).

Voice conditioning (`Model::load_voice`): loads an exported
`voices/<name>.safetensors` (upper-backbone hidden states plus
per-layer key/value rows for the positive and negative branches, the
Swift-style 4D `(B, kv, S, d)` storage the reference transposes on
load; batch-free 3D files are normalized too) and seeds the caches, so
generation starts from the voice prompt state exactly as the reference
does. The `lm_hidden` and `neg_lm_*` entries the format carries are
never read by the reference generation loop and are not loaded.

Scope: single batch, non-streaming, caller-owned text tokenization
(the reference encodes `text.strip() + "\n"` without special tokens;
encode that string with the checkpoint tokenizer and hand over the
ids). The multi-speaker dialogue wrapper is a caller loop over single
segments; the streaming chunks are a decode-schedule wrapper around the
same latents and are not ported. The reference encoder side of the
acoustic tokenizer does not ship for inference and is not ported.

## Verification

Fixture protocol identical to the codec families:
`tools/gen_vibevoice_fixtures.py` builds seeded tiny instances of the
real reference modules (2-layer split decoder, hidden 16, 4 query
heads, 2 kv heads, one lower and one upper layer; 2-stage acoustic
decoder, ratios [2, 2], vae_dim 8; 2-layer diffusion head, latent 8, 4
DDPM inference steps over a 20-step cosine schedule; a character-level
stand-in tokenizer; an engineered EOS classifier bias) and records, per
case in `testdata/tts_vibevoice/<case>/` (config.json +
model.safetensors + traces.json, digests in manifest.json):

- `generate/`: the full-budget loop; three text windows, eight
  diffusion samples with cfg 3.0, the recorded `mx.random.normal`
  draws (one initial latent noise per `sample_speech_tokens` call; the
  scheduler is deterministic and draws nothing), the unscaled latents,
  and the decoded waveform.
- `stop/`: the engineered-stop twin; the classifier ends the run after
  the first latent.
- `voice/`: the voice-conditioned twin; a voice cache built from a
  four-token probe prompt rides the reference `load_voice` layout
  (Swift-style 4D rows) and generation starts from it.

The Rust tests reload each case through the same `Model::open` path a
real checkpoint takes and replay the recorded draws. The generator
enforces the recording caveat on the Python side before freezing: a
fresh model rebuilt from the saved safetensors plus the recorded draw
stream must reproduce the recorded audio (the reference has no
mx.compile or hash-dependent iteration on this path; the save is an
f16-mantissa round trip and the replay recomputes in f32 over the same
bits).

Measured parity on the frozen fixtures (relative to the reference
magnitude; latents peak at 1.2e3, the waveform at 2.2e3): latents
1.1e-6 relative worst case, waveform 1.5e-6, both against a 2e-4
relative gate (over 100x margin). `VIBEVOICE_MEASURE=1` re-prints the
diffs. Scheduler unit tests pin the half-to-even timestep rounding,
the terminal (1, 0, +inf) cache row, the final-step x0 collapse, and
the cosine-schedule refusal.

Real-checkpoint gate NOT run: no VibeVoice checkpoint was downloaded
in this porting round. Open items: a real `mlx-community/VibeVoice-*`
load-and-decode pass (config fields, tensor names, scaling factors,
quantized variants), voice quality, runtime/FFI/Swift integration, and
the streaming decode schedule.

## Gotchas found during verification

- The lower backbone's final norm genuinely does not exist in the
  checkpoint (`use_norm=False`); loading it through the stock
  `Backbone::load` fails on the missing `norm` tensor, which is why
  the engine gained `load_without_final_norm` /
  `load_without_embed_table` (additive; all lm fixtures stay green).
- The reference `load_voice` transposes EVERY 4D tensor on load, so
  exported caches store keys AND values Swift-style
  `(B, kv, S, d)`; a 3D file is used as-is `(rows, kv, d)`. Storing
  either half in the model-native `(B, S, kv, d)` layout silently
  corrupts attention after the first seeded decode step.
- `mx.cumprod` over the f32 beta schedule is a sequential f32 chain;
  reproducing it in f64 drifts the cached alpha/sigma values. The
  scheduler keeps the betas, alphas, and cumulative product in f32 and
  only the per-inference-step recomputation in f64, exactly like the
  reference (`np.array(self.alpha_t.tolist())`).
- The scheduler steps the DOUBLED cfg batch in the reference, but both
  halves stay identical under elementwise math, so stepping one half
  is exact; `prev_x0` threads the previous x0 prediction for the
  second-order updates, and the final step (sigma = 0, lambda = inf)
  collapses to the x0 prediction.
- The EOS latent IS kept: the classifier check runs after the latent
  is appended and fed back, so a stop at step k decodes k latents, not
  k-1.
