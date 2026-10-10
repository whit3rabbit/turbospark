# MeloTTS family (VITS2 transformer flow + BERT prosody + HiFi-GAN)

Port of `mlx_audio/tts/models/melotts/` (melotts.py, modules.py,
attentions.py, hifigan.py, bert.py, text.py) at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/melotts)
(mlx 0.32.3). A VITS2-relative architecture: phones -> embeddings (phone,
tone, language, the 1024-dim bert projection, the 768-dim ja_bert
projection) -> transformer text encoder (window-4 shared relative position
tables, post-norm, the speaker projection injected at layer index
`min(2, n_layers)`) -> deterministic duration predictor (two k-3 convs with
timestep LayerNorms and the speaker bias) -> monotonic alignment on
`ceil(exp(logw) / speed)` -> mean/log-variance alignment -> latent draw
`m + normal * exp(logs) * noise_scale` on the shared MLX key-sequence RNG ->
transformer coupling flow (four coupling layers with channel flips, reversed
for inference, mean-only so the log-scale is zero) -> HiFi-GAN decoder with
the speaker bias after `conv_pre` -> 44.1 kHz mono. The ja_bert features
come from a bert-base transformer whose third-to-last hidden state is
repeated per `word2ph` entry (`bert.rs`); the reference applies the
tokenizer attention mask additively (`-inf` on padded key columns) and the
embedding LayerNorm uses the config `layer_norm_eps` (1e-12) while the
per-layer norms use the `nn.LayerNorm` default (1e-5).

Everything is portable single-batch f32. No Metal, no macOS APIs.

## Layout

- `mod.rs`: `ModelConfig` (the reference dataclass fields and defaults, the
  `spk2id` table, the checkpoint symbol table, and a fixture-only `bert`
  sub-config), the MLX-order loaders (conv weights `[out, kernel, in]`
  transposed to `[out, in, kernel]`, transpose-conv to `[in, out, kernel]`),
  the `Conv1dPT` projections as per-timestep linears (`PointwiseLinear`),
  the window-4 `MultiHeadAttention` (heads share one relative table; the
  pad/slice `_get_relative_embeddings` and both relative/absolute index
  permutes are reproduced literally), the transformer `Encoder` (channel
  axis LayerNorms, speaker conditioning at `cond_layer_idx`), the
  `TextEncoder`, `DurationPredictor`, `TransformerCouplingLayer` (reverse,
  mean-only), the HiFi-GAN `Generator` (ResBlock1 stack averaged per
  upsample stage), the flip helper, and `Model` with `open`, `infer`,
  `resolve_speaker`, and `flow_reverse`. The training-only `enc_q` and
  `sdp` checkpoint tensors are ignored.
- `bert.rs`: the reference `BertModel` (embeddings with the additive
  attention mask, post-norm layers with erf GELU, `extract_features` =
  `all_hidden_states[-3]`, the pooled CLS output for traces, and
  `phone_features` = the `get_bert_features` word2ph repeat transposed to
  the channel-major `[hidden, phones]` layout the ja_bert projection
  consumes).
- `text.rs`: `text_normalize` (lowercase, the abbreviation table,
  comma-number stripping, decimal and integer expansion), the pure post-G2P
  mapping (`refine_ph`, `post_replace_ph`, `distribute_phone`,
  `cleaned_text_to_sequence`, `add_blank_interleave`), and the default
  symbol table.
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch size 1, non-streaming. The sequence masks are all-ones on the
  full-length batch-1 path; the `[T, T]` attention masks derived from them
  are materialized as ones and the `-1e4` fill never triggers. The
  reference streaming path (chunked decode with overlap) is not ported.
- G2P is caller-owned, like kitten_tts's espeak. The reference loads the
  bert-base-uncased wordpiece tokenizer, the CMU dict, and the g2p_en
  neural fallback from external assets this crate cannot embed; the intake
  is the raw ARPAbet phone list per word (what the reference g2p returns),
  and everything after it - `refine_ph`, `post_replace_ph`, the pad
  wrapping, `cleaned_text_to_sequence`, `add_blank_interleave`, and
  `text_normalize` - runs here.
- The 1024-dim bert features are zeros in the shipped `generate` path
  (`_prepare_inputs` passes `mx.zeros((1, 1024, n_phones))`), so the Rust
  `infer` takes the bert input as a slice the caller fills (zeros on the
  shipped path) and only the ja_bert features are computed by `bert.rs`.
  The ja_bert width comes from the loaded projection (768 in the shipped
  checkpoints; the fixture model rebuilds it at 16).
- `sdp_ratio` must be 0: the stochastic duration predictor is
  training/sampling-only in this port and the reference default is 0. The
  deterministic `DurationPredictor` logw feeds the alignment directly.
- The latent noise draw is one normal draw of `m_p.shape` in channel-major
  order on the shared MLX key-sequence RNG (`Rng::normal`), pinned per call
  by the seed argument; the reference seeds globally (`mx.random.seed`).
- Weight layouts: checkpoints carry the raw MLX parameter tree; the loader
  transposes conv weights at load. Weight norm folding does not apply (the
  melotts checkpoints carry none; the reference `sanitize` drops
  `weight_g`/`weight_v` pairs that do not occur in the shipped files).
- The fixture BERT is a tiny seeded stand-in (hidden 16, 2 layers, vocab
  100) with a `bert` config extension the reference hard-codes at bert-base
  dims; the Rust `BertDims` defaults match the reference and read the
  extension when present.

## Verification

Fixture protocol per the repo standard: `tools/gen_melotts_fixtures.py`
runs the true reference modules against a tiny seeded model (hidden 32, 2
encoder layers, 2 flow layers, gin 16, upsample 4x4x2, n_heads 2), driven
from `../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_melotts_fixtures.py
```

The G2P and the wordpiece tokenizer are pinned stand-ins (g2p_en/cmudict
are not installed in the fixture venv and the bert-base vocab is not
cached); the reference post-processing runs for real on the pinned phones.
Fixtures live in `testdata/tts_melotts/` with SHA-256 digests in
`manifest.json`:

- `frontend/traces.json`: the symbol table, `text_normalize` cases
  (abbreviations, comma numbers, decimals, integers, zero, big and
  negative), post-G2P mapping cases with ids and tones, and the add_blank
  interleave case.
- `bert/`: the tiny seeded BERT weights and per-case traces (input ids,
  attention mask including a padded tail, word2ph, per-layer hidden
  states, the pooled CLS vector, and the phone-level features).
- `stages/`: the tiny seeded model (`model.safetensors`, `config.json`)
  and per-stage traces of the assembled infer internals: the speaker row,
  the pre-encoder embedding sum, every encoder layer output, x, m_p,
  logs_p, the duration-predictor conv+norm stages, logw, w_ceil,
  y_lengths, the alignment, the aligned means, the latent draw, the latent
  after every reverse coupling step, the first step's internals (post-flip
  input, coupling h, predicted mean), z, and the audio.
- `generate/`: end-to-end scenarios through the reference
  `_prepare_inputs` + `infer` (two texts with ja_bert features from the
  tiny BERT, the 1024-dim bert input zeroed like the shipped path).

Measured parity when the gates were frozen: ids and strings are exact;
every stage checkpoint is at f32 reordering level (max 1.2e-7 across x,
logw, the flow latents, and the audio; BERT phone features 2.4e-7 and
pooled 4.5e-8). Gates: `FEATURE_TOLERANCE` 2e-4 on string-to-model stages,
`AUDIO_TOLERANCE` 5e-4 on waveforms.

`cargo test -p turbospark-audio --lib tts::melotts` runs the family gate.

## Open items

- No real-checkpoint gate: a MeloTTS end-to-end run with real weights, the
  real bert_weights.npz, and the real symbol table has not been executed;
  the fixtures are seeded tiny instances.
- The g2p intake (wordpiece tokenizer, CMU dict, g2p_en fallback) is not
  ported; callers must supply the ARPAbet phones per word and the word2ph
  alignment.
- The reference streaming decode (chunked z with 16-frame context and tail
  trim) is not ported.
- `text_normalize` covers the English path only; the reference multilingual
  branches (zh/jp/kr/es/fr front ends) are out of scope for this port.
