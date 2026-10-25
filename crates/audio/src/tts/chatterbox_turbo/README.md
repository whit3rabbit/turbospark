# Chatterbox Turbo (GPT-2 T3 + meanflow S3Gen voice-clone TTS)

Chatterbox Turbo renders text to a 24 kHz waveform conditioned on a
reference voice: the T3 GPT-2 generator produces S3 speech tokens from
text tokens with NO classifier-free guidance, and the S3Gen half (a
conformer encoder plus a meanflow U-Net token decoder and an NSF-iSTFT
vocoder) turns the tokens into audio conditioned on the reference
prompt (prompt tokens, a length-matched prompt mel, and a CAMPPlus
speaker embedding) in two Euler steps. An LSTM voice encoder produces
the 256-dim speaker embedding the T3 conditioning consumes. This is
the distilled turbo sibling of the chatterbox family
([`../chatterbox/`](../chatterbox/README.md)): same token space, same
S3Gen skeleton, a different backbone, a different flow solver, and a
different checkpoint key tree.

## Upstream references and URLs

- **Reference implementation**:
  [`mlx_audio/tts/models/chatterbox_turbo/`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/chatterbox_turbo)
  at mlx-audio `e1b19b9054bf163f5d812221a54fcc346f1890e9` (MLX
  0.32.3): `chatterbox_turbo.py`, `models/t3/` (the GPT-2 T3 generator,
  `gpt2.py` backbone), `models/s3gen/` (`s3gen.py` inline flow,
  `flow_matching.py` meanflow, `decoder.py` U-Net, `encoder.py`
  consolidated conformer, `hifigan.py`, `mel.py`, `xvector.py`),
  `models/voice_encoder/`, `models/s3tokenizer/` (the S3 tokenizer
  import).
- **Shared components**: the S3 speech tokenizer is the codec port
  ([`../../codec/s3/`](../../codec/s3/README.md)) and the T3 GPT-2
  backbone is the shared LM engine
  ([`../../lm/`](../../lm/README.md) `gpt2`); both are reused, not
  re-ported. The S3Gen mel front end helpers and the conv/linear layers
  come from the sibling port's `s3gen` module (identical math at this
  commit).
- **Upstream checkpoint**: `ResembleAI/chatterbox-turbo` (GPT-2 medium
  text table, 50276 tokens; separate `mlx-community/S3TokenizerV2`
  speech tokenizer checkpoint), plus the pre-computed `conds.safetensors`
  voice conditionals.

## Architecture

- **T3** (`t3.rs`): a GPT-2 medium backbone (24 layers, hidden 1024,
  16 heads, learned positions added inside the forward from the cache
  offset, tanh-approximation gelu) fed composite embeddings
  `[conditioning | text | start_speech_token]`. The conditioning
  encoder is a single speaker projection: turbo drops the sibling
  family's perceiver resampler and emotion scalar
  (`use_perceiver_resampler = False`, `emotion_adv = False`). The
  reference wraps the text ids verbatim (the sibling family wraps
  `[255] text [0]`; turbo does NOT). The backbone's own token table
  (`tfmr.wte`) and the T3 text head are never read by the forward, so
  the loader skips both (real installs save ~200 MB of dead weight);
  the learned position table loads from the family root through the
  additive `Gpt2Model::with_position_table_at_root`.
- **Generation loop**: no classifier-free guidance. Per step:
  speech-head logits (with bias) on the last position, the sign-aware
  repetition penalty over the UNIQUE generated ids (`np.unique` before
  the gather), the temperature divide, the top-k filter that keeps
  every logit at or above the k-th largest value (ties survive), and
  the descending top-p filter that masks where the sorted inclusive
  cumulative probability exceeds `top_p` shifted one right. The
  reference draws through `mx.random.categorical` on the still
  unnormalized filtered logits; the Rust draw uses the crate-local
  MLX-compatible threefry RNG, so a seeded Python stream replays
  exactly. `temperature == 0` is the explicit greedy mode: the plain
  post-penalty argmax (top-k and top-p never mask the maximum); the
  reference itself always samples. The stream ends at
  `stop_speech_token` (6562) or `max_gen_len` (the reference pops the
  trailing stop before returning).
- **Token crop** (`generate`): keep the ids strictly below 6561 in
  stream order (the whole stream, not a start/stop span) and append
  THREE 4299 silence tokens (`S3GEN_SIL`) - the sibling family crops
  between the markers and adds none.
- **S3Gen flow** (`flow.rs`, `conformer.rs`): tokens embed (vocab
  clipped), the conformer encoder doubles the frame rate (the turbo
  constructor hardcodes FOUR post-upsample conformer layers, no
  `num_up_blocks` knob), a linear projects to 80 mel dims, and a
  diffusers-style U-Net estimator (causal conv-LayerNorm-Mish resnet
  blocks with the time embedding broadcast channel-major, bidirectional
  self-attention transformer blocks with exact-erf GELU feed-forwards,
  one down stage and one up stage at the single reference width, each
  ending in a `CausalConv1d`) runs a 2-step Euler solve over the LINEAR
  schedule with the meanflow time mixing: the estimator consumes both
  the start time `t` and the end time `r` through one shared time MLP
  and a bias-free mixing matrix (`time_embed_mixer`). No CFG, no cosine
  warp (the distilled `_basic_euler` path). The output slice drops the
  prompt frames. The flow start state is spliced per channel: the
  prompt span of one normal draw plus the generation span of a second
  draw (`noised_mels`), both explicit [`flow::FlowNoise`] inputs.
- **HiFT vocoder** (`hift.rs`): an ELU conv stack predicts f0 per mel
  frame; the f0 is nearest-upsampled 480x and rendered through the
  neural source filter with the phase `(2*pi*cumsum(f0*(h+1)/sr)) mod
  2*pi` plus constant initial phases drawn for harmonics 1..8 ONLY
  (eight uniforms; harmonic 0 a structural zero - the sibling draws
  nine and masks the first). The source STFT is UNCENTERED (frames of
  the raw waveform, no reflect pad), so the fusion lengths run short
  and the reference's `min_len` clamp truncates the mel path at every
  fusion stage; the last upsample stage pads ONE ZERO on the left (the
  reference `mx.pad` call, despite its "reflection" comment); the final
  leaky ReLU takes the `nn.leaky_relu` default slope 0.01 (the upsample
  stages pass 0.1); the output head emits log-magnitude and phase
  halves for an iSTFT, clipped to +/-0.99, and a 20 ms trim fade zeroes
  the first 480 output samples and fades the next 480.
- **CAMPPlus** (`campplus.rs`, `fbank.rs`): the fbank front end is a
  float64 Kaldi pipeline (strict-edge mel triangles, snip-edges
  framing, per-frame DC removal before the 0.97 pre-emphasis, 512-point
  rfft, log floored at the f32 epsilon) followed by the per-bin mean
  removal over frames - it disagrees with the sibling family's f32
  librosa-style front end by up to 0.015 on the log-mel values, so it
  is ported separately. The network (FCM head with shortcut resblocks,
  strided TDNN, three dense CAM blocks 12/24/16 at dilations 1/2/2,
  transit convs, statistics pooling WITHOUT the sibling's +1e-5
  epsilon inside the sqrt, affine-free batchnorm dense head) matches
  the sibling port's math under the turbo key tree (`bn`/`bn1`/`bn2`/
  `out_bn`, `shortcut_conv`/`shortcut_bn`).
- **Voice encoder** (`voice_encoder.rs`): the sibling port's math
  (40-mel front end, energy trim, partial windows at `rate = 1.3`,
  three-layer LSTM, projection, ReLU, `norm + 1e-8` L2 normalization)
  under the turbo key tree (`lstm1`/`lstm2`/`lstm3` instead of
  `lstm.layers.N`; the `similarity_*` checkpoint arrays are unused at
  inference and not loaded).

## Scope and established choices

- Single batch, non-streaming. The streaming chunk paths
  (`stream_generate`, `inference_stream`, the HiFT `cache_source`) are
  never built by the reference non-streaming call and are not ported.
- Caller-owned text tokenization. The real checkpoint ships a Hugging
  Face tokenizer the portable crate does not embed. The reference's
  portable text normalization ([`punc_norm`], a DIFFERENT replacement
  table from the sibling: no three-dot or spaced-dash rules) is ported;
  the prompt layout is the caller ids verbatim and the sentence
  chunking (`max_tokens // 8 * 4` chars) stays caller-side.
- Caller-owned resampling and loudness: `Model::prepare_conditionals`
  takes the three reference wavebands directly (16 kHz full for the
  voice encoder and T3 prompt tokens up to `enc_cond_len` = 15 s, 24 kHz
  truncated to `dec_cond_len` = 10 s for the S3Gen mel, and the 16 kHz
  resample of that band, itself trimmed to the decoder length at the 16
  kHz rate, for its tokens). The `integrated_loudness` normalization is
  not portable and stays out.
- Deterministic filter math; recorded-randomness replay for the sampled
  paths. The T3 categorical draws use the crate-local MLX-compatible
  threefry RNG (one uniform per sample, the standard key evolution), so
  a fixture seed replays the Python stream exactly. The HiFT sine
  randomness (eight phases, the sine noise bed, and the discarded
  source-module noise bed) and the two flow start windows are explicit
  inputs ([`hift::SineDraws`], [`flow::FlowNoise`]); `None` draws from
  the crate RNG (uniforms bit-match; the normal draws differ by the
  erfinv implementation, documented in the crate RNG module - the flow
  start point shifts, the conditioning does not).
- Explicit greedy mode: `temperature == 0` takes the post-penalty
  argmax, fixture-verified end to end (the reference has no greedy
  path; the generator drives the reference forward methods with an
  argmax pick and self-checks the picks against the collected logits).

## Verification

Parity is established against pinned Python MLX fixtures generated by
`tools/gen_chatterbox_turbo_fixtures.py` (tiny seeded GPT-2 at a
2-layer hidden-32 backbone through the patched reference
`create_gpt2_config` and `GPT2_MEDIUM_CONFIG`; tiny flow geometry;
HiFT and CAMPPlus at the reference structure with shrunk widths; a tiny
seeded S3TokenizerV2; every case replayed on the Python side before
freezing). The flow-input seed is chosen by an attention-margin search
(the smallest top-2 attention score gap over every head, row, and Euler
step must clear 5e-4) so the crate's summation order cannot flip an
attention weight; the voice encoder and mel front ends are the
sibling-verified helpers. Fixtures live in `testdata/tts_chatterbox_turbo/`
with a sha256 manifest:

```sh
cargo test -p turbospark-audio --lib tts::chatterbox_turbo
```

Case coverage:

- `t3/`: the conditioning rows and the composite prompt embeds, the
  greedy generate loop end to end (ids exact, per-step argmax reproduced
  from the recorded head logits with the penalty math, margins pinned),
  and the sampled loop with the seeded categorical stream replayed
  exactly from the same seed (per-step head logits recorded by
  re-running the reference forward pass over the frozen stream).
- `flow/`: token + prompt -> mel through the meanflow solve (the
  2-step Euler, per-channel noise splice, prompt-span crop) with the
  recorded start windows.
- `hift/`: mel -> waveform through the vocoder with the recorded sine
  draws (f0 exact, waveform within 2e-4) plus the deterministic 20 ms
  fade applied on both sides.
- `end_to_end/`: greedy ids -> token crop -> silence tail -> flow ->
  vocoder -> fade on one realization.
- `campplus/`: the float64 Kaldi fbank (mean-removed) and the embedding
  through the full CAMPPlus structure.
- `sampling/traces.json`: the filter math on fixed logits - the
  unique-token repetition penalty, the temperature divide, the top-k
  threshold, the descending top-p shift mask, the categorical draw with
  a recorded uniform, and the greedy argmax.

Measured tolerances (the gates are the asserts; the numbers are the
observed maxima on the current fixtures): every id stream and the
speech-token crop are exact; the flow mel compares at 6e-7; the vocoder
f0 is exact and the waveform at 5.4e-8; the end-to-end waveform at
8e-8; the conditioning rows and prompt embeds at 1.7e-6; the Kaldi
fbank is bitwise equal and the CAMPPlus embedding at 6e-7.

Regeneration (deterministic, no network; run from the mlx-audio
checkout root so `mlx_audio` imports from the pinned source):

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_chatterbox_turbo_fixtures.py
```

## Open items

- Real-checkpoint gates are recorded open: the tiny-fixture parity
  above is not a hardware or real-model claim. The real-checkpoint
  smoke gate and a real-model end-to-end comparison against the Python
  reference remain to be run against `mlx-community/chatterbox-turbo`
  converted weights (which carry `t3.tfmr.wte` and the derived
  `pos_enc.pe`/`stft_window`/`trim_fade` buffers the Rust port skips or
  recomputes).
- The flow start windows: real installs draw fresh normals at call
  time. The crate RNG's uniforms bit-match MLX but the normal draws
  differ by the erfinv implementation, so real samples stay valid but
  are not bit-identical to a Python run (the same caveat the sibling
  family records for its CFM noise buffer).
- Text tokenization stays with the caller; the crate does not embed the
  checkpoint's BPE tokenizer, and `punc_norm` covers the portable
  cleanup only.
- The streamed variants (`stream_generate`, `inference_stream`,
  `inference` with `finalize=False`) are unported reference surface.

## Gotchas found during verification

- The turbo checkpoint keys differ from the sibling family's nearly
  everywhere: the conformer block lists keep MLX module-list indices
  dotted (`encoders.0.` vs the sibling's renamed `encoders_0.`), every
  conv routes through the `Conv1dPT` wrapper (one `.conv` deeper; the
  f0 predictor's list holds the wrapper directly), the decoder U-Net
  uses raw diffusers names (`resnet.mlp.0`, `block1.block.0.conv.conv`,
  `transformer_blocks.0`, `attn1.to_q`, `ff.net.0.proj`), CAMPPlus
  names its norms `bn`/`bn1`/`bn2`/`out_bn` with a
  `shortcut_conv`/`shortcut_bn` pair, and the voice encoder stacks
  `lstm1`/`lstm2`/`lstm3`. The forked modules exist for the key trees,
  not the math.
- `T3Config.n_channels` reads the module-level `GPT2_MEDIUM_CONFIG`
  dict, NOT the (patched) `create_gpt2_config()`; the fixture patches
  both or the conditioning encoder silently stays at the medium width.
- The turbo encoder hardcodes FOUR post-upsample conformer layers; a
  configurable count silently drops the last two (caught as a
  38.9-mel divergence at `up_encoders.2`).
- The CFM noise splice is per channel (`concat([z[:,:,:prompt_len],
  noised_mels], axis=2)` on the channel-major layout): concatenating
  the two channel-major windows flat produces channel blocks and
  scrambles the start state.
- The hift fixture must consume the flow realization the flow case
  froze (the recorded-draw run); a second run draws fresh normals and
  its mel - and therefore the recorded f0 and waveform - belongs to a
  different realization.
- The turbo StatsPool computes the plain population std
  (`mx.std`, no epsilon) where the sibling family adds +1e-5 inside the
  sqrt; the sibling port's epsilon moved the embedding by 1.5e-2.
- The meanflow estimator runs at batch one: no CFG duplication
  anywhere in the turbo pipeline.
