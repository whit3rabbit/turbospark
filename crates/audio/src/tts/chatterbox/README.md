# Chatterbox (T3 + S3Gen voice-clone TTS)

Chatterbox renders text to a 24 kHz waveform conditioned on a reference
voice: the T3 llama generator produces S3 speech tokens from text tokens
under classifier-free guidance, and the S3Gen half (flow-matching token
decoder plus an NSF-iSTFT vocoder) turns the tokens into audio
conditioned on the reference prompt (prompt tokens, a length-matched
prompt mel, and a CAMPPlus speaker embedding). An LSTM voice encoder
produces the 256-dim speaker embedding the T3 conditioning consumes.

## Upstream references and URLs

- **Reference implementation**:
  [`mlx_audio/tts/models/chatterbox/`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/chatterbox)
  at mlx-audio `e1b19b9054bf163f5d812221a54fcc346f1890e9` (v0.5.7,
  MLX 0.32.3): `chatterbox.py`, `config.py`, `t3/` (the T3 generator),
  `s3gen/` (flow + HiFT + CAMPPlus + mel), `voice_encoder/`,
  `s3tokenizer/` (the S3 tokenizer import), `tokenizer.py`.
- **Shared components**: the S3 speech tokenizer is the codec port
  ([`../../codec/s3/`](../../codec/s3/README.md)) and the T3 llama
  backbone is the shared LM engine ([`../../lm/`](../../lm/README.md));
  both are reused, not re-ported.
- **Upstream checkpoints**: `mlx-community/chatterbox-fp16` (English,
  704-token text table) and `chatterbox-multilingual-v3` (2454-token
  text table), plus the separate `mlx-community/S3TokenizerV2` speech
  tokenizer checkpoint.

## Architecture

- **T3** (`t3.rs`): a llama backbone (pinned 520M profile: 30 layers,
  hidden 1024, 16+16 heads, head dim 64, llama3 rope scaling) fed
  composite embeddings `[conditioning | text(+learned positions) |
  BOS(+pos 0) | BOS(+pos 0)]`. The conditioning encoder projects the
  256-dim speaker embedding to one row, resamples the prompt speech
  tokens to 32 rows through a perceiver (one shared 4-head attention
  block applied as cross- then self-attention), and projects the emotion
  scalar to one row (34 conditioning rows). Dedicated text and speech
  embedding tables and output heads; the llama head is never applied
  (the backbone loads without its token table and head).
- **Generation loop**: classifier-free guidance doubles the batch (the
  uncond row zeroes the text embedding before the position add; the
  workspace engine is single-batch, so the two rows run as two forwards
  over independent prompt caches - the same math, because attention
  never crosses the batch axis). Per step: speech-head logits on the
  last position, CFG combine `cond + w * (cond - uncond)` on the RAW
  logits, the sign-aware repetition penalty over every generated id
  (context size = the full generation cap; the gather precedes the
  scatter, so duplicate context tokens are idempotent), then the
  sampler. The reference samples on the unnormalized logits: top-p
  compares unnormalized exponential sums, min-p is shift invariant, and
  the categorical draw renormalizes internally; temperature 0 is the
  plain post-penalty argmax (`make_sampler(0, ...)` applies no filter).
  Stop at `stop_speech_token` (6562) or `max_new_tokens`.
- **Token crop** (`drop_invalid_tokens` + the `< 6561` filter): the
  stream strictly between the first start marker (6561) and the first
  stop marker (6562), specials dropped in order.
- **S3Gen flow** (`flow.rs`, `conformer.rs`): tokens embed (vocab
  clipped), a linear-input conformer encoder with an espnet
  relative-position attention (closed-form `rel_shift`), a pre-lookahead
  conv, and a repeat-and-conv stride-2 upsample doubles the frame rate;
  a linear projects to 80 mel dims; a matcha U-Net estimator (causal
  resnet blocks, self-attention transformer blocks with exact-erf GELU,
  one down stage and one up stage at the single reference width, each
  ending in a `CausalConv1d`) runs a 10-step Euler conditional flow
  match over the cosine schedule `1 - cos(t*pi/2)` with guidance rate
  0.7 (the uncond half zeroes mu, the projected speaker, and the
  conditioning mel). The output slice drops the prompt frames.
- **HiFT vocoder** (`hift.rs`): an ELU conv stack predicts f0 per mel
  frame; the f0 is nearest-upsampled 480x (8 * 5 * 3 * iSTFT hop 4) and
  rendered through the neural source filter with the reference's
  NON-interpolated phase path (`2*pi*cumsum(f0*(h+1)/sr) mod 1` plus one
  constant random initial phase per harmonic, harmonic 0 zeroed, and a
  voiced-gated noise bed). The mel upsampling path fuses the STFT-domain
  source through residual blocks; the output head emits log-magnitude
  and phase halves for an iSTFT, clipped to +/-0.99, and a 20 ms trim
  fade zeroes the first 480 output samples and fades the next 480.
- **CAMPPlus** (`campplus.rs`): the kaldi fbank front end (Povey window,
  DC removal, pre-emphasis, HTK mel, log at float epsilon, per-feature
  mean removal), the batchnorm FCM conv head, a strided TDNN, three
  dense CAM blocks (12/24/16 layers, dilations 1/2/2), transit convs,
  statistics pooling (mean + population std), and an affine-free
  batchnorm dense head producing the 192-dim speaker embedding.
- **Voice encoder** (`voice_encoder.rs`): a 40-mel front end (centered
  Hann STFT 400/160/400, magnitude squared, Slaney mel 0-8 kHz), an
  energy trim (`trim_top_db = 20`: RMS over 2048-sample frames at a
  512 hop), partial windows of 160 frames at step 77 (the
  `rate = 1.3` default), a three-layer LSTM (PyTorch i/f/g/o gate order
  over the combined bias), the final hidden state through a projection
  with ReLU, per-partial L2 norms, the partial mean, and a final L2
  norm.

## Scope and established choices

- Single batch, non-streaming. The reference `generate` is a
  single-yield generator with `finalize=True` everywhere; the streaming
  chunk paths (the flow `streaming` flag, the HiFT `cache_source`,
  the pre-lookahead context) are never built by the reference call and
  are not ported.
- Caller-owned text tokenization. The real checkpoints ship a Hugging
  Face tokenizer the portable crate does not embed. The portable text
  normalization ([`punc_norm`]) and the exact prompt layout
  (`Model::prepare_text_tokens`: `[255] text [0]`) are ported so a
  caller reproduces the reference ids from its own tokenizer; the
  `text_preprocessing` config flag selects the `capitalize` behavior
  (the v3 `NFKD,fullcase` convention does not capitalize).
- Caller-owned resampling. `Model::prepare_conditionals` takes the
  three reference wavebands directly (16 kHz full for the voice encoder
  and T3 prompt tokens, 24 kHz truncated to `dec_cond_len` for the S3Gen
  mel, and the 16 kHz resample of that band for its tokens) instead of
  running the polyphase resampler.
- Deterministic filter math; recorded-uniform replay for sampled paths.
  The T3 categorical draws use the crate-local MLX-compatible threefry
  RNG, and the fixture proves the replay: the seeded Python stream is
  redrawn exactly from the same seed (one uniform per step, the standard
  `mx.random` key evolution). The HiFT sine randomness (initial phase
  draw and sine noise bed) and the CFM noise window are explicit inputs
  ([`hift::SineDraws`]; the reference's third draw, the discarded
  source-module noise bed, is recorded only to keep the draw order
  honest).
- The CFM `rand_noise` is generated at reference init from the seed-0
  stream and real converted checkpoints do not carry it; the Rust loader
  accepts any checkpoint-provided buffer (whole mel rows) and refuses
  empty ones, so the fixture pins a short recorded window while real
  installs inject or accept the divergence documented below.
- The reference CFG prompt rides a batch of two through the transformer
  (`chatterbox.generate` duplicates the text row before
  `t3.inference`); the single-batch engine runs the two rows as two
  cache chains, which is exact for the attention math, and the fixture
  records the per-step decision margins (all >= 0.05 greedy, >= 1e-4
  sampled) so the f32 reordering cannot flip a choice.
- `chatterbox_turbo` is a separate spec task (67) and is not included.

## Verification

Parity is established against pinned Python MLX fixtures generated by
`tools/gen_chatterbox_fixtures.py` (seeded tiny T3 at a 2-layer
hidden-32 llama through the patched reference `LLAMA_CONFIGS` entry;
tiny flow geometry; HiFT and CAMPPlus at the reference structure with
shrunk widths; a tiny seeded S3TokenizerV2; every case replayed on the
Python side before freezing). Fixtures live in
`testdata/tts_chatterbox/` with a sha256 manifest:

```sh
cargo test -p turbospark-audio --lib tts::chatterbox
```

Case coverage:

- `t3/`: the conditioning-encoder rows, the greedy generate loop end to
  end (ids exact, per-step argmax reproduced from the recorded head
  logits with the CFG + penalty math, margins pinned), and the sampled
  loop with the seeded categorical stream replayed exactly from the
  same seed.
- `flow/`: token + prompt -> mel through the flow (the 10-step Euler
  solve, prompt-span crop); `hift/`: mel -> waveform through the vocoder
  with the recorded sine draws (f0 exact within 2e-3, waveform within
  2e-4) plus a zero-draw mutation check that pins the draws as
  load-bearing; `end_to_end/`: greedy ids -> token crop -> flow ->
  vocoder -> fade on one realization.
- `campplus/`: kaldi fbank (mean-removed, the embedding input) and the
  embedding through the full batchnorm CAMPPlus structure.
- `voice_encoder/`: trim + 40-mel + partial windows + stacked LSTM ->
  the L2-normalized speaker embedding.
- `conditioning/`: `prepare_conditionals` on the three wavebands
  through the tiny S3 tokenizer (mel + token alignment, T3 prompt
  tokens, prompt mel, CAMPPlus embedding, speaker embedding).
- `sampling/traces.json`: the filter math on fixed logits - CFG combine,
  the repetition penalty with duplicate context tokens, raw top-p and
  min-p filters, the categorical draw with a recorded uniform, and the
  greedy argmax.

Measured tolerances (the gates are the asserts above; the numbers are
the observed maxima on the current fixtures): every id stream and every
FSQ/S3Gen token stream is exact; the flow mel compares at 6.7e-6
absolute (mel values reach 8.9); the vocoder f0 is exact and the
waveform at 1.0e-7; the end-to-end waveform (greedy T3 -> flow ->
vocoder -> fade) at 1.1e-7; the conditioning rows, fbanks, and speaker
embeddings hold their 2e-4 asserts with margin. Stage-level bisection
traces (optional, not shipped): `debug_flow_stages` and
`debug_hift_stages` diff every pipeline stage against fresh Python
dumps from `tools/dbg_chatterbox_traces.py` and skip when the
`dbg_*.npy` scratch files are absent.

Real-checkpoint smoke gate (finiteness, shapes, config defaults; skipped
unless the directories exist):

```sh
TURBOSPARK_CHATTERBOX_MODEL_DIR=/models/chatterbox \
TURBOSPARK_S3_MODEL_DIR=/models/s3-tokenizer \
  cargo test -p turbospark-audio --lib tts::chatterbox::tests::real_checkpoint_gate -- --ignored
```

Regeneration (deterministic, no network; run from the mlx-audio checkout
root so `mlx_audio` imports from the pinned source):

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_chatterbox_fixtures.py
```

## Open items

- Real-checkpoint gates are recorded open: the tiny-fixture parity above
  is not a hardware or real-model claim. The real-checkpoint smoke gate
  and a real-model end-to-end comparison against the Python reference
  remain to be run against `mlx-community/chatterbox-fp16`.
- The seed-0 CFM noise buffer: the reference generates
  `(1, 80, 50 * 300)` normals at init; the Rust port replays a recorded
  window for fixtures but a real install must either carry the buffer in
  the checkpoint or accept the crate-RNG substitution (the threefry
  stream matches MLX's uniforms bit for bit; the normal draws differ by
  the erfinv implementation, documented in the crate RNG module). This
  shifts the flow start point, not the conditioning, so samples stay
  valid but are not bit-identical to a Python run.
- Multilingual v2/v3 checkpoints: the 2454-token text table and the
  language-tagged `MTLTokenizer` prompt path are config-plumbed (the
  text table width widens from the `multilingual` flag), but no
  multilingual fixture is generated; the portable `punc_norm` covers the
  language-independent cleanup only.
- Text tokenization stays with the caller; the crate does not embed the
  checkpoint's BPE tokenizer.

## Gotchas found during verification

- The T3 decode loop duplicates the next-token embedding to batch two
  whenever `cfg_weight > 0`, but the prompt is batch one when the caller
  passes batch-one text tokens (the real `chatterbox.generate` always
  duplicates first) - a batch mismatch the KV cache broadcast rejects.
  The port runs the CFG pair as two cache chains, so the mismatch cannot
  arise, but fixture recordings must replicate the reference batch
  doubling before `t3.inference`.
- The matcha U-Net's last down stage and last up stage replace the
  stride stages with `CausalConv1d` at the single reference width
  (`channels = [256]`); the causal conv needs the kernel-minus-one left
  pad, which the shared `SaConv1d` does not apply.
- The espnet positional tables (`pos_enc.pe`) are derived buffers: real
  checkpoints store the computed table, and the Rust port recomputes it;
  the fixture generator must not seed it, or the checkpoint carries
  random normals the Rust side never reads.
- The reference CFM runs the estimator at batch two (cond/uncond); the
  uncond half zeroes mu, the projected speaker embedding, and the
  conditioning mel - not the noise or the mask.
- `kaldi_fbank` fixtures store the mean-removed variant (that is the
  embedding input); comparing the raw fbank against it fails with an
  offset of the mean level.
- MLX `nn.LSTM` returns `(hidden, cell)` sequences and consumes PyTorch
  gate order (i, f, g, o quarters of the combined bias); the sanitize
  path passes torch weights straight through, which is what pins the
  order.
- The resnet time-embedding broadcast is channel-major: `h` is `[C, T]`
  between decoder stages, so `h[c * T + t] += mlp(c)` steps the frame
  index, not `h[t * C + c]`. A row-major broadcast passes every
  fixture-sized square case silently and scrambles the cond everywhere
  else; the bisection caught it as a block2 output divergence with
  block1 and the time MLP individually matching.
- The `decode` final leaky ReLU passes `negative_slope=self.lrelu_slope`
  explicitly (0.1), the same slope as the upsample stages - not the
  `nn.leaky_relu` default 0.01.
- Two more derived buffers follow the `pos_enc.pe` rule: the HiFT
  `stft_window` (periodic Hann) and the S3Gen `trim_fade` (raised
  cosine) are pure init-time formulas the Rust port recomputes; seeding
  them in the fixture generator desyncs the fixture from both the port
  and a real converted checkpoint.
- The HiFT resblock chain is high-gain: at the generator's shared 0.2
  weight scale the vocoder activations reach 1e10 before the exp/sin
  iSTFT head and the waveform pins at the 0.99 audio limit, which
  carries no parity information. `build_s3gen` re-seeds the vocoder at
  0.04 so the fixture waveform stays unsaturated (peak ~0.03).
