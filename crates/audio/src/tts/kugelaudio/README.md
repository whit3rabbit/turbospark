# KugelAudio TTS family (kugelaudio/kugelaudio-open)

Port of `mlx_audio/tts/models/kugelaudio/` at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/kugelaudio)
(mlx 0.32.3). A hybrid AR + diffusion architecture derived from Microsoft
VibeVoice: a Qwen2.5 decoder (the shared [lm](../../lm/README.md) qwen2
flavor: biased q/k/v projections, GQA, non-traditional rope, final norm)
runs autoregressively over a four-token speech vocabulary - speech start
151652, speech end 151653, speech diffusion 151654, EOS 151643 - with a
greedy argmax under a keep-set constraint (`VALID_SPEECH_TOKENS`). Each
speech_diffusion token triggers the VibeVoice diffusion prediction head
under the SDE-DPM-Solver++ stochastic scheduler (cosine betas,
v-prediction, 2-order solver with first-order first/last steps, one
standard-normal noise draw per step), with classifier-free guidance
against a speech-start-only negative branch when `cfg_scale > 1.0`; the
sampled acoustic latent feeds back through the speech connector
(fc1 -> RMSNorm -> fc2) into the decoder stream. speech_end (or EOS)
stops the loop, with one extra latent when the diffusion logit sits
within `FINAL_LATENT_LOGIT_MARGIN` (5.0) of the winning end logit and at
least one latent exists. The concatenated latents are unscaled by the
checkpoint's speech scaling/bias factors, decoded by the convolutional
VAE acoustic tokenizer decoder, and peak-normalized to 0.95 when the
decode exceeds unit amplitude.

The acoustic decoder is the VibeVoice `TokenizerDecoder` (decoder only;
the checkpoints ship no encoder): stem conv, per-stage pre-norm blocks
(depthwise causal convolution mixer plus exact-GELU feed forward, each
with a per-channel layer scale), causal transposed-convolution upsamples
(the declared encoder ratios double as strides; the reversed encoder
depths as block counts), and a head conv. The converted checkpoints
store the MLX conv layouts (`[out, kernel, in]` for both conv and conv
transpose); the loader transposes into the PyTorch layouts
[`ops::conv1d`](../../ops) takes.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: the config (acoustic tokenizer, diffusion head, and the
  shared qwen2 decoder flavor, including the `acostic_vae_dim` typo
  handling), the prompt text builders (`prompt_text` /
  `format_speaker_text` / `text_section`), the speech connector, the
  AR + diffusion generate loop, `should_append_final_latent`, and the
  `DiffusionNoise` provider trait with `SeededNoise`.
- `acoustic_decoder.rs`: the conv VAE decoder.
- `diffusion_head.rs`: the AdaLN prediction head with the sinusoidal
  timestep embedder (cos-then-sin concat), loading the converted
  `mlp.layers.{i}` / `adaLN_modulation.layers.{i}` names with the raw
  numeric-index fallback; the AdaLN modulation applies SiLU to the
  conditioning before the linear (the reference `Sequential(SiLU,
  Linear)`).
- `scheduler.rs`: the SDE-DPM-Solver++ scheduler (cosine beta schedule,
  Python half-to-even timestep rounding, the cached alpha/sigma/lambda
  values with the terminal `1.0 / 0.0 / +inf` row, v-prediction x0,
  stochastic first/second-order updates whose coefficients compute in
  f64 and multiply f32 like the reference).
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch 1 and non-streaming (the reference generate is non-streaming;
  `verbose` progress prints are dropped).
- Caller-owned text tokenization: `_build_prompt_tokens` strings are
  exported (`prompt_text`, `format_speaker_text`, `text_section`); the
  caller encodes the joined prompt text with the checkpoint tokenizer
  and appends [`SPEECH_START_ID`].
- Diffusion noise comes from a [`DiffusionNoise`] provider: the fixture
  tests replay the recorded reference `mx.random.normal` draws; standalone
  runs can seed the crate-local threefry RNG (`SeededNoise`). Draw order
  per `sample_speech_tokens` call: the initial latent noise, then one
  draw per scheduler step.
- The checkpoint loads through the converted (model-tree) tensor names:
  the backbone under `language_model.` (probing the raw composite
  `model.` prefix too), `lm_head`, `acoustic_tokenizer.decoder.*`,
  `acoustic_connector.*`, `prediction_head.*`, and the top-level
  `speech_scaling_factor` / `speech_bias_factor` scalars (1.0 / 0.0
  defaults when absent). The reference sanitize skips the encoder and
  semantic stacks; the loader reads decoder-only tensors and would
  refuse a checkpoint missing any of them.
- `ddpm_batch_mul`, `fix_std`, `std_dist_type`, and the corpus
  normalization fields are encode- or training-side and unused in the
  decode-only path; the config parser keeps the fields the runtime
  reads.

## Verification

Fixture protocol identical to the other audio families:
`tools/gen_kugelaudio_fixtures.py` runs the true reference loop driven
from `../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_kugelaudio_fixtures.py
```

Fixtures live in `testdata/tts_kugelaudio/` with SHA-256 digests in
`manifest.json`:

- `generate/` (tiny stack: 2-layer qwen2 decoder hidden 16 with the
  real 151k speech ids, 2-stage acoustic decoder ratios [2, 2] vae_dim
  8 at 8 kHz, 2-layer diffusion head with 4 DDPM steps over a 20-step
  cosine schedule): the recorded prompt, four greedy speech-diffusion
  steps under classifier-free guidance (cfg 3.0), the 20 recorded noise
  draws, the unscaled latents, and the decoded, peak-normalized
  waveform. The model weights round-trip through f16 in the generator
  so the traces come from exactly the values the Rust fixture loads.
- `stop/` (engineered-stop twin): speech_end wins the first constrained
  argmax with no latents yet, so the run finishes with no audio; the
  token count pins the loop bookkeeping.

Measured parity when the gates were frozen: latents within 3.7e-6
relative (gate 1e-4 relative; the tiny fixture divides by a 0.03
scaling factor, so the latents ride in the hundreds and an absolute
gate would be meaningless) and the waveform within the same f32
accumulation noise under the 2e-4 relative gate; frames and token
counts exact.

`cargo test -p turbospark-audio --lib tts::kugelaudio` runs the family
gate.

## Open items

- No real-checkpoint gate: an end-to-end run against
  `kugelaudio/kugelaudio-0-open` (7B decoder, 3584-wide diffusion head,
  full 32-filter decoder stack, real tokenizer, 24 kHz waveform sanity
  against the Python family) has not been executed; the fixtures are
  seeded tiny instances.
- The acoustic tokenizer ENCODER is not ported (the kugelaudio generate
  path is decode-only; the checkpoints ship no encoder).
- Sampled (non-argmax) AR token choices do not exist in the reference
  loop (it is greedy by construction), so no sampling deviation applies;
  the diffusion noise, however, is only replay-exact against the
  recorded draws, and a seeded Rust run draws different noise than mlx.
