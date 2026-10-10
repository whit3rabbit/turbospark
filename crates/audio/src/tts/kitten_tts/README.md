# KittenTTS family (ALBERT + StyleTTS2 prosody + iSTFTNet)

Port of `mlx_audio/tts/models/kitten_tts/` (kitten_tts.py, modules.py,
istftnet.py, preprocess.py, quant.py) at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/kitten_tts)
(mlx 0.32.3). A StyleTTS2-relative architecture: phoneme ids -> KittenAlbert
(the reference's "bert", an ALBERT over the phoneme symbol table with the
ONNX tanh-GELU constants) -> duration predictor (BLSTMs + AdaLayerNorm,
sigmoid sum over the `max_dur` classes divided by speed, round-half-to-even,
min 1) -> alignment -> F0/noise prosody predictor (shared BLSTM + AdaIN res
blocks, one upsample stage doubling the curve) -> text encoder (weighted
convs + LayerNorm + BLSTM) -> iSTFTNet decoder with a harmonic sine source
(SineGen with 8 harmonics, voiced threshold 10, snake alphas per conv pair)
-> 24 kHz mono. The 256-dim voice style splits at a hard-coded 128
(decoder / predictor halves). The reference `generate` glue (chunking, the
20 ms crossfade, the RMS-based tail trim, the dynamic fade-out, the 200 ms
tail silence) is ported, including its quirk of compounding a voice speed
prior across chunks (the resolved speed feeds the next `_prepare_inputs`).
The ONNX heritage shows up as `activation_quant_modules`: a dynamic
per-tensor uint8 fake quantization (`DynamicQuantizeLinear` semantics,
round-half-to-even) applied at the named module sites.

Everything is portable single-batch f32 (the harmonic-spectrum STFT runs in
f64; see the determinism note). No Metal, no macOS APIs.

## Layout

- `mod.rs`: `ModelConfig` (the reference fields and defaults; the `plbert`
  and `istftnet` sub-configs; `decoder_out_dim None -> max_conv_dim`),
  `ActivationQuant` / `fake_quant_dynamic_u8`, the weight-normalized convs
  (checkpoint MLX order `[out, kernel, in]`, transposed at load, weight norm
  folded with the reference `+ 1e-7` epsilon), the bidirectional LSTM (gate
  order i/f/g/o; `Wx_forward` or PyTorch spellings), KittenAlbert,
  DurationEncoder, TextEncoder, ProsodyPredictor, KittenDecoder, the iSTFTNet
  generator (harmonic source, noise convs + snake resblocks, transpose-conv
  ups with the left reflection pad on the last stage, `exp` magnitude /
  `sin` phase, iSTFT with length `(frames - 1) * hop`), and `Model` with
  `prepare_segment`, `generate_segment(_with_durations)`, `synthesize(_with)`,
  `crossfade`, and `apply_tail`.
- `frontend.rs`: `TextCleaner` (the exact 175-symbol table with the Python
  dict later-wins duplicate rule; the apostrophe appears twice and lands on
  the later id), `basic_english_tokenize`, `ensure_punctuation`,
  `chunk_text`, and the `TextPreprocessor` default pipeline (contractions,
  IP addresses, leading decimals, currency, percentages, scientific
  notation, times, ordinals, units, scale suffixes, fractions, decades,
  phone numbers, ranges, model names, bare numbers, punctuation removal,
  lowercase, whitespace collapse) as left-to-right scanners with sequential
  `re.sub` semantics.
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch size 1, non-streaming. The per-call padding masks are no-ops on the
  full-length batch-1 path and are not materialized; the ALBERT additive
  attention mask and the pooled output (unused by the TTS path) are likewise
  dropped.
- espeak G2P is caller-owned. The reference phonemizes through `phonemizer`
  EspeakBackend, an external OS engine this crate cannot embed; the Rust
  intake is `Model::prepare_segment(phonemes, ...)` taking the phonemized
  string (with the espeak settings the reference uses: en-us, preserve
  punctuation, with stress). Everything after the phonemizer - tokenize,
  symbol ids, pad wrapping, the voice row by the CLEANED text length, speed
  priors and aliases - runs here, and the fixtures exercise the real espeak.
- Dropout (eval-time identity) is not materialized.
- `expand_roman_numerals`, `remove_hashtags`, `remove_mentions`,
  `remove_accents`, and `remove_stopwords` are default-off in the reference
  `TextPreprocessor` and are not ported; `normalize_unicode` (NFC) is
  skipped (a no-op on ASCII input, matching the soprano family documented
  deviation).
- Voice tables: checkpoints ship `voices.npz` (a zip of numpy arrays). This
  crate reads `voices.safetensors` (a `[rows, 256]` tensor named `voices` or
  `voice`) or accepts a table through `install_voice`; converting
  `voices.npz` is a packaging step, like the kokoro voice packs.
- MLX silently returns zero vectors for out-of-range embedding ids; the Rust
  loader refuses them (the fixture sets `n_token` to the symbol table
  maximum id plus one so every cleaner id fits).

## Waveform determinism (the phase-channel branch cut)

The generator noise branch consumes the STFT of the harmonic source as
`[magnitude; phase]` features, with the phase from a raw `arctan2`. That
channel is discontinuous at the arctan2 branch cut wherever the true
imaginary part sits below the FFT rounding error, and the reference FFT
(mlx on macOS: vDSP) rounds those near-zero imaginary parts differently from
this crate numpy-exact FFT (and would round them differently again on
Linux). Measured on the fixture model: the magnitude matches to 6e-7 and the
phase to 4.4e-6 away from the cut, but roughly 1.6 percent of spectrum
positions (the Nyquist bin imaginary zero sign, plus near-zero interior
bins) land on the other side, and the fixture waveforms decorrelate
(correlation 0.55-0.63) because the conv over the phase channel amplifies
each 2*pi jump. The port computes the spectrum with an f64 direct DFT
(preserving the sign of near-zero imaginary parts the way the reference
accurate FFT does, including the `+0.0` DC convention) and documents this as
an inherent cross-implementation limit of the family, not a port defect: the
same reference produces different waveforms per FFT backend.

Consequences for the gates: everything deterministic is pinned tightly
(string pipeline exact, model stages 2e-4, and the generator waveform from
the PINNED spectrum at 1.3e-4 against audio magnitudes up to 44), while the
raw text-to-waveform path is gated as a length plus correlation smoke only.

## Verification

Fixture protocol per the repo standard:
`tools/gen_kitten_tts_fixtures.py` runs the true reference modules against a
tiny seeded model (style_dim 128 because the reference hard-codes the split;
ALBERT hidden 24, 2 layers; hidden_dim 32; istftnet upsample 4x4, n_fft 32,
hop 4; n_token 178 so every cleaner id fits), driven from `../mlx-audio` at
the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_kitten_tts_fixtures.py
```

The espeak step requires the espeak-ng library (the generator locates it on
disk; no network). Fixtures live in `testdata/tts_kitten_tts/` with SHA-256
digests in `manifest.json`:

- `frontend/traces.json`: the symbol table, TextPreprocessor cases
  (numbers, floats, ordinals, currency, time, ranges, units, scale
  suffixes, scientific notation, fractions, decades, phone numbers, model
  names, HTML/URL/email, contractions, whitespace), chunk and tokenize
  cases, the espeak phonemized texts, cleaner id contracts, an STFT parity
  probe (deterministic signal, magnitude and phase), and crossfade/tail
  cases on pinned inputs.
- `stages/`: the tiny seeded weights (`model.safetensors`,
  `voices.safetensors`, `config.json`) and per-stage traces of the
  instrumented `Model.__call__` (bert_out, d_en, duration logits,
  durations, aligned features, F0/N, text encoder, asr, decoder blocks,
  harmonic source, STFT, and the generator stages from the pinned spectrum)
  for three scenarios (base, speed 1.4, different style row).
- `generate/`: end-to-end scenarios (two single-chunk texts through the
  reference `generate`, one through the `kv` voice alias with the configured
  speed prior) and a six-chunk crossfade scenario with the compounding
  speed prior, plus a fake-quant parity case.

Measured parity when the gates were frozen: the string pipeline and ids are
exact; model stages within 2e-4 (FEATURE_TOLERANCE); the generator waveform
from the pinned spectrum 1.3e-4 max absolute error against audio magnitudes
up to 44 (gate 1e-3); the raw e2e waveforms correlate 0.55-0.63 with the
reference (gate 0.45) for the branch-cut reason above.

`cargo test -p turbospark-audio --lib tts::kitten_tts` runs the family gate.

## Open items

- No real-checkpoint gate: a KittenTTS nano/micro/mini end-to-end run (real
  weights, real voice table, waveform sanity against the Python family on
  the same FFT backend) has not been executed; the fixtures are seeded tiny
  instances. The phase-channel branch cut means a cross-backend waveform
  comparison needs the pinned-spectrum protocol above rather than raw audio.
- `voices.npz` (zip/numpy) is not read natively; convert to safetensors or
  call `install_voice`.
- The default-off `TextPreprocessor` passes listed above are unported.
- The activation-quant sites are ported and unit-pinned
  (`fake_quant_dynamic_u8`), but no fixture case runs a fully quantized
  model end to end; the real nano checkpoints set `activation_quant_modules`
  for most of the network.
