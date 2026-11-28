# ACE-Step 1.5 (`ace_step15`)

Second music family: flow-matching DiT over 25 Hz Oobleck VAE latents,
48 kHz stereo. The target is text2music plus audio-to-audio restyle
("cover", for example rap to classical) and repaint, which MiniMax
Music 3 cannot do.

Status: **phase 2 of 6 in progress, not runnable.** CPU f32: VAE,
Qwen3-Embedding text encoder, condition encoder, DiT and the turbo
text2music sampler match the reference on tiny fixtures and on the real
checkpoint, and so do the FSQ tokenizer, detokenizer and the cover
sampler (strength switch). Text2music runs end to end
(`pipeline.rs`: prompt assembly and tokenization, text encoder,
conditioning, sampling, tiled decode) and matches an upstream-path
reference waveform from the same noise, with or without upstream's DCW
sampler correction (`dcw.rs`). The runtime serves it: an audio music
session opened on an ACE-Step checkpoint root dispatches to
`crates/runtime/src/ace_step15.rs` (family `ace_step15`, Metal, or the
host when the caller allows a portable engine) with upstream's turbo
recipe (8 steps, shift 3, DCW). The catalog lists it as
`ace-step15-turbo` (9 pinned files with sizes and SHA-256, 6.33 GB,
`turbospark-model pull-audio ace-step15-turbo`), and `turbospark-music
generate` runs it (`--duration` required, `--language`, fixed 8 steps).
Not yet: Swift UI, request fields for bpm / key / time signature, source
audio for cover, and repaint. Metal: the condition encoder,
DiT, sampler and FSQ path run with projections resident on
`gpu::Music3Device` and pass the same real-checkpoint gate (test-local
backend in `crates/runtime/tests/ace_step15_metal.rs`), and so does the
VAE (device convolutions, tiled decode). The text encoder is still CPU
only. Nothing is
wired into runtime, FFI, catalog, CLI, server or Swift, and the
persisted family string is not registered anywhere yet.

## Pinned references

| What | Pin |
|---|---|
| Upstream code | `ACE-Step/ACE-Step-1.5` at `ca1e85fe9430179831e6bc6be790c332190a3866` (2026-08-29), MIT |
| Checkpoint | `ACE-Step/Ace-Step1.5` at `19671f406d603126926c1b7e2adc169acbcade22`, MIT (model card) |
| Model reference | the checkpoint's own `modeling_acestep_v15_turbo.py`, transformers 4.57.6, torch 2.15.0, CPU f32, SDPA |
| FSQ reference | vector-quantize-pytorch 1.27.20 `ResidualFSQ` (upstream `uv.lock`) |
| VAE reference | diffusers 0.37.1 `AutoencoderOobleck` (upstream `uv.lock`) |
| Second reference | upstream `acestep/models/mlx/` (MLX VAE and DiT ports) and `ServeurpersoCom/acestep.cpp` (MIT, GGML) |

The Apple Silicon fork `clockworksquirrel/ace-step-apple-silicon` is a
PyTorch MPS fork behind upstream (no XL, old parameter names). It is
read for MPS pitfalls only, not ported.

Checkpoint files used (sizes and SHA-256 from the Hugging Face LFS
metadata at the pinned revision, verified locally):

| File | Bytes | SHA-256 |
|---|---|---|
| `vae/diffusion_pytorch_model.safetensors` | 337431388 | `da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0` |
| `acestep-v15-turbo/model.safetensors` | 4787825604 | `3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7` |
| `Qwen3-Embedding-0.6B/model.safetensors` | 1191586416 | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |

Read the checkpoint LICENSE text at the pinned revision before any
catalog row ships.

## Architecture (from the pinned configs)

- VAE: Oobleck, strides `[2, 4, 4, 6, 10]` (hop 1920), 64 latent
  channels, 25 frames/s, weight-norm convs and Snake activations.
- DiT turbo: 24 layers, hidden 2048, GQA 16/8, head_dim 128, layers
  alternate sliding window 128 and full attention, RoPE theta 1e6,
  AdaLN timestep modulation, cross-attention to packed text, lyric and
  timbre states, input is a channel concat of context latents (source
  latents plus chunk mask) and noisy latents (192 channels, patch 2).
- Text: Qwen3-Embedding-0.6B. Lyric encoder (8 layers) and timbre
  encoder (4 layers) are inside the DiT checkpoint.
- Sampler: Euler ODE from t=1 to 0, shift 3 for turbo, 8 steps, no CFG
  on turbo (guidance forced to 1.0).

## Tasks and how they work upstream

- text2music: noise to latents from caption plus lyrics.
- cover: the source is VAE-encoded, then **round-tripped through the
  5 Hz FSQ tokenizer and detokenizer** (`prepare_condition`:
  `src_latents = detokenize(tokenize(src))` when `is_covers`), and that
  coarse version fills the context channels for the first
  `int(steps * audio_cover_strength)` steps, then the loop switches to
  the non-cover context (silence latents, and its own caption). With 8 turbo steps the strength
  has 9 distinct outcomes. Cover keeps vocals unless lyrics are
  `[Instrumental]`. Duration is locked to the source.
- flow-edit (`flow_edit_morph`, text2music task): paired source and
  target captions over the encoded source audio, a FlowEdit-style
  edit. Compare against cover in phase 4 before choosing the restyle
  default.
- repaint: a 0/1 mask over a time range in the context channels, a
  latent crossfade and a waveform splice.
- Base-model-only (deferred): extract stems, lego, complete.

## Phases

0. Pins, fixtures, this README. Done for the VAE.
1. VAE CPU f32 with parity gates. **Done** (`vae.rs`).
2. Text encoder, lyric and timbre encoders, DiT, turbo sampler: CPU
   parity **done** (`text.rs`, `transformer.rs`, `dit.rs`), FSQ
   tokenizer, detokenizer and cover sampling **done** (`tokenizer.rs`).
   Metal, first cut **done**: a `Loader` with a `ComputeBackend` makes
   every projection a resident bf16 device weight (tiled f32 GEMM,
   `tiled_for_f32`) and runs unmasked attention on the device. Sliding
   layers whose window is shorter than the sequence run banded on the
   host, because `music3_attention*` has no window parameter and its
   parameter block is shared with Music 3. VAE on Metal **done**:
   folded conv weights resident as F32 (`convolution_f32_grouped`),
   transposed convs split into `stride` kernel-2 phase convs so only
   real taps run, host Snake and interleave threaded, and upstream's
   overlap-discard tiled decode. Remaining: a windowed device attention
   in a new kernel (not by changing `music3.metal`), recording each DiT
   layer as one `Sequence`, an Oobleck Snake kernel so the VAE can stay
   on the device between convolutions, and faster F32 conv tiles.
3. Text2music end to end on Metal with the real checkpoint: parity,
   memory peak, timing, listening check. Only then can a catalog row
   become runnable.
4. Cover, flow-edit and repaint; runtime request fields; FFI header;
   CLI.
5. Catalog and Swift Music page modes (Create, Restyle, Repaint).
6. Server text2music with per-family limits.

Later: base checkpoint, the 5Hz LM planner (Qwen3), quantized DiT, XL.

## Text2music pipeline (`pipeline.rs`, `prompt.rs`)

What one request builds, from upstream's handler at the pinned commit:

- Caption prompt `SFT_GEN_PROMPT`: `# Instruction` (task instruction,
  colon appended if missing), `# Caption` (an SFT-formatted caption is
  unwrapped first), `# Metas` (`bpm`, `timesignature`, `keyscale`,
  `duration`, each `N/A` when empty or zero; duration is `int(seconds)`,
  default 30), then `<|endoftext|>`. Lyric prompt `# Languages` /
  `# Lyric` then `<|endoftext|>`.
- Tokenizer: the checkpoint `tokenizer.json`; its post-processor appends
  one more `<|endoftext|>`, and truncation (256 caption, 2048 lyric
  tokens) happens before that, so a cut prompt still ends with it.
- Latent frames `max(128, int(duration * 48000) // 1920)`; source
  latents `silence[:frames]`, chunk mask all ones, timbre reference
  `silence[:750]`; turbo shift 3 (upstream's default of 1.0 targets
  base/SFT; its docs say 3.0 for turbo).
- After decode, divide by the peak when it exceeds 1.
- `silence_latent.pt` is read by `silence.rs` without unpickling: the
  archive's entries are stored, `data.pkl` must equal the pinned
  166-byte record byte for byte (float32 `[1, 64, 15000]`, contiguous),
  `byteorder` must be `little`, and the 3,840,000-byte payload is copied
  and transposed. It equals the torch `weights_only` conversion bit for
  bit. It is not `VAE(silence)`: that differs by up to 0.027 and the
  shipped latent varies over time, so it cannot be recomputed. A
  `silence_latent.safetensors` from `tools/convert_ace_step15_silence.py`
  is accepted when no `.pt` is present.

DCW: upstream's loader copies its own `acestep/models/turbo` modeling
file over the checkpoint's whenever they differ
(`_sync_model_code_if_needed`), and that file applies DCW after every
turbo step by default (`double`, scaler 0.05, high scaler 0.02, Haar,
one level, zero padding, `denoised = x_before - v * t`, the final x0
step included). With DCW off it reproduces the checkpoint file exactly
(max difference 0.0, same noise). DCW is a large effect on this request:
latents differ by up to 5.1 against a 5.6 scale and the decoded peak
goes from 0.44 to 0.97. Callers pass `Dcw::TURBO` to match upstream's
default, `None` for the checkpoint's own sampler. If `pytorch_wavelets`
is missing upstream silently skips DCW; the reference generator refuses
that.

Remaining gap against upstream's handler: the initial noise is our seeded
generator, not torch's (same seed, different song).

## Runtime session (`crates/runtime/src/ace_step15.rs`)

- Detection: a directory with `acestep-v15-turbo/` (config and weights),
  `vae/` and `Qwen3-Embedding-0.6B/` (tokenizer). The silence latent must
  already be converted (`silence_latent.safetensors`).
- Request mapping (`native_audio::ace_step15_request`): `caption`,
  `lyrics`, `language` (default `en`, a short code), `duration_seconds`
  (required, (0, 600]), `seed`. `steps` must be unset or 8; `context_size`
  and `max_frames` are refused, as are PCM and the Drama 3 sampler fields.
- Progress stages: `running` (session), `text`, `condition`, `dit_step`
  0..8 of 8, `decode`. Output: interleaved stereo f32 at 48 kHz.
- Admission: `ace_step15::open_bytes` (DiT file + 2 x text encoder + 3 x
  VAE + 512 MiB) = 8.12 GiB against a measured 7.64 GiB open footprint
  on M4 Max; one 10 s song adds 2.45 GiB, inside the shared music run
  estimate (2 GiB + 64 MiB per second) with little margin. Measure a long
  request before widening the duration a host offers.
- Tests: `crates/runtime/tests/ace_step15_session.rs` (validation by
  default, a real session ignored).
- CLI, indicative: `turbospark-music generate` rendered 30 s in 19.2 s
  (RTF 0.64, load 2.8 s) on M4 Max at host load average 15. The music
  CLI now checks each family's own bounds after resolving the model; only
  model-agnostic checks (positive duration, steps at least 1) run while
  parsing.

## Pitfalls found

- Reference environment: upstream requires transformers `>=4.51,<4.58`.
  Under 5.x, `from_pretrained` leaves every `Qwen3RotaryEmbedding.inv_freq`
  at zero, so the reference silently runs without RoPE while models
  built directly (the tiny fixtures) are fine. The real-checkpoint gate
  failed at 0.3 relative until the venv matched upstream's `uv.lock`.
  `gen_ace_step15_dit_fixtures.py` now refuses both conditions.
- The DiT drops its own attention masks (`attention_mask = None`,
  `encoder_attention_mask = None` in `AceStepDiTModel.forward`), so
  cross-attention sees every packed encoder position. Batched padded
  requests therefore attend to padding upstream; the port takes one
  unpadded request.
- Sliding layers use a symmetric window `|i - j| <= 128` over patch
  indices (2 latent frames per patch, so about 10 s each way).
- `silence_latent.pt` is a torch pickle (`[1, 64, 15000]` f32). The
  runtime must never unpickle it: convert it once at install with a
  trusted tool and ship safetensors.
- Upstream truncates captions at 256 and lyrics at 2048 tokens silently;
  the port refuses over-length input so the caller chooses.

## Not ported

Gradio UI, the DJ chat (network keys), LoRA training, PMI scoring,
vLLM paths, `torch.compile` toggles, the fork's RAM tiers (our memory
admission decides), and the pipeline's chunked VAE windows until the
Metal VAE exists (chunking is a device decision there).

## Verification

```sh
cargo test -p turbospark-audio --lib music::ace_step15
```

Tiny fixtures (`crates/audio/testdata/ace_step15_vae/`, `ace_step15_dit/`):

```sh
../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_vae_fixtures.py
../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_dit_fixtures.py <ckpt>/acestep-v15-turbo
```

Real checkpoint (ignored tests, references outside the repository):

```sh
../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_vae_fixtures.py --real <ckpt>/vae <ckpt>/reference/vae
../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_dit_fixtures.py --real <ckpt>/acestep-v15-turbo <ckpt>/reference/dit
../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_text_fixtures.py <ckpt>/Qwen3-Embedding-0.6B <ckpt>/reference/text
TURBOSPARK_ACE_STEP15_VAE=<ckpt>/vae TURBOSPARK_ACE_STEP15_VAE_REF=<ckpt>/reference/vae \
TURBOSPARK_ACE_STEP15_DIT=<ckpt>/acestep-v15-turbo TURBOSPARK_ACE_STEP15_DIT_REF=<ckpt>/reference/dit \
TURBOSPARK_ACE_STEP15_TEXT=<ckpt>/Qwen3-Embedding-0.6B TURBOSPARK_ACE_STEP15_TEXT_REF=<ckpt>/reference/text \
  cargo test --release -p turbospark-audio --lib music::ace_step15 -- --ignored --test-threads=1
```

Measured on M4 Max (2026-11-27), worst relative error per stage:

| Gate | Tiny | Real checkpoint |
|---|---|---|
| VAE encode mean / decode | 2.7e-6 (sample) | 4.8e-6 / 2.8e-6 (2 s stereo) |
| Condition encoder | 3.1e-7 | 2.6e-7 |
| DiT, one step at t=0.75 | 2.5e-6 | 3.5e-6 (48 frames) |
| Turbo text2music, 8 steps | 3.5e-7 | 7.3e-6 (48 frames) |
| Text encoder caption hidden | n/a | 5.0e-6 (69 tokens) |
| FSQ tokens / indices | 5.4e-8 / 3 of 3 exact | 1.5e-8 / 10 of 10 exact |
| Detokenized cover hints | 4.4e-7 | 1.2e-6 (tokenize to hints chained: 1.3e-6) |
| Cover, 8 steps at strength 0.5 | 4.2e-7 | 6.1e-6 |

These are CPU f32 numbers on bf16 weights widened to f32. They say
nothing about bf16 compute, quality or speed. The real DiT gate holds
about 13 GB resident and takes about 4 minutes on the CPU.

Metal path (`cargo test -p turbospark-runtime --test ace_step15_metal`,
real gates with `--ignored`), same reference arrays, M4 Max:

| Gate | Tiny | Real checkpoint |
|---|---|---|
| Condition encoder | 3.6e-7 | 2.7e-7 |
| DiT, one step | 2.1e-6 | 2.7e-6 |
| Turbo text2music, 8 steps | 3.7e-7 | 5.8e-6 |
| FSQ tokens / indices | 5.4e-8 / exact | 1.5e-8 / exact |
| Cover hints / cover 8 steps | 3.6e-7 / 3.8e-7 | 1.0e-6 / 3.0e-6 |

Indicative timing, not a frozen row (one run, host load average 8 to 17
from other sessions, warm-up forward discarded): 30 s of audio (750
frames, 300 encoder tokens), DiT resident 2.93 GiB, one forward 1.39 s,
the 8-step turbo loop 11.1 s.

Where the DiT time goes (one forward at 30 s, same caveats): device
linears 0.76 s (270 calls), device attention 0.23 s (36 calls), host
0.41 s, of which about 0.26 s is banded attention for the 12 sliding
layers (measured by rerunning with every layer full). The DiT is
GEMM-bound; a windowed device attention would save under 20%.

VAE on Metal, same reference arrays: tiny worst 2.3e-6, real 2 s
stereo worst 5.2e-6 (encoder mean), decode 3.0e-6. Tiled decode at
chunk 512 / overlap 64 latent frames is bit-identical to the untiled
decode (the receptive field fits inside the overlap). 30 s of audio,
indicative: host decode about 30 s (10.1 s measured at 10 s), Metal
decode 11.2 s with direct transposed convs, 8.7 s with phase-split
transposed convs, 7.6 s once the phase interleave is threaded (device
convs 6.9 s of it, 116 calls). Metal encode of 30 s is about 6 s. The
remaining VAE time is the F32 tiled conv kernel itself (roughly 0.4
TFLOP/s on these shapes).

End to end (`real_*_text2music_matches_reference`, reference from
`tools/gen_ace_step15_e2e_reference.py`, 10 s, replayed noise):

| Path | Latents | Audio | Wall time, 10 s song |
|---|---|---|---|
| CPU | 1.3e-5 | 9.7e-6 | text 0.6 s, condition 1.6 s, sampling 28.6 s, decode 10.0 s |
| Metal | 1.1e-5 | 6.7e-6 | 6.3 s total: text 0.42, condition 0.41, sampling 3.3, decode 2.2 |
| CPU, DCW | 1.3e-5 | 1.1e-4 | sampling 11.1 s (quieter host), decode 10.9 s |
| Metal, DCW | 1.1e-5 | 9.6e-5 | 6.7 s total: sampling 3.6, decode 2.3 |

The DCW audio gap is ten times the plain one with the same latent gap
because the decoder is ten times more sensitive at that louder operating
point: perturbing the reference latents by about 1e-4 moves the
diffusers decode by 1.1e-4 for the DCW latents and 1.1e-5 for the plain
ones. DCW itself matches upstream's `DCWCorrector` over pytorch-wavelets
within 1.1e-7 (`testdata/ace_step15_dcw/`, 8 cases, all modes, odd and
even lengths); its cost is negligible next to the DiT.

Metal timing is the second run after a warm-up, host load average 11 to
16, not a frozen row. Prompt strings match upstream's helpers in all 10
fixture cases (`testdata/ace_step15_prompt/`), and token ids match the
upstream tokenizer call exactly, including a lyric truncated from 4010
to 2048 tokens.
