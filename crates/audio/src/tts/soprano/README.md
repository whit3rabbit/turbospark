# Soprano family (qwen3 straight into a Vocos iSTFT decoder)

Port of `mlx_audio/tts/models/soprano/` (soprano.py, decoder.py,
text.py) at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/soprano)
(mlx 0.32.3). The family is unusual: the qwen3 backbone (shared
[crates/audio/src/lm](../../lm/README.md) port, untied lm head
supported) emits no audio tokens at all. The per-step post-norm hidden
states feed a decoder directly: a linear upsample (x4,
`align_corners=True`), a ConvNeXt backbone (`VocosBackbone` from the
mlx-audio vocos codec, input width = LM hidden size, depthwise kernel
3, input kernel 1 for 1.1 checkpoints), and an iSTFT head (linear to
n_fft + 2, exp/clamp magnitude, overlap-add divided by the summed
window, the reference `istft` default). 2048 samples per LM token at
32 kHz (`upscale * hop_length`); after generation the waveform is
trimmed to the last `tokens * token_size` samples. Prompt frames are
`[STOP][TEXT]{sentence}[START]` per sentence after the `text.py`
cleaning pipeline; the loop stops on the tokenizer pad id (falling back
to the `[STOP]` token id and the tokenizer eos) and never sees a
stopping token again: the break happens before the forward pass, so the
stopping token contributes no hidden state.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: `ModelConfig` (qwen3 args plus `sample_rate` 32000, nested
  `decoder_config`, and the `model_path` hook: pre-1.1 paths override
  the decoder to dim 512 / intermediate 1536 / input kernel 3),
  `preprocess_text` (clean, split after terminal punctuation, merge
  short pieces with the reference strip-on-concat semantics), the
  generate loop (`generate_from_prompt`, `generate_hidden`,
  `decode_hidden`), and `GenerateParams` with the reference defaults
  (temperature 0.3, top-p 0.95, max_tokens 512; no min-p, top-k, or
  logits processors in this family).
- `decoder.rs`: `DecoderConfig` (the reference dataclass defaults:
  8 layers, dim 768, intermediate 2304, hop 512, n_fft 2048, upscale 4,
  input kernel 1, dw kernel 3, token size 2048, receptive field 4) and
  `SopranoDecoder` (upsample + ConvNeXt + iSTFT head). The backbone
  math mirrors the [vocos codec](../../codec/vocos/README.md) port but
  lives here because the checkpoint names, input width, and the
  pre-backbone upsample are family specific and the codec module keeps
  its backbone private to its own loaders.
- `text.rs`: the `clean_text` pipeline (number expansion incl. the
  year-like 1000-2999 pronunciations, abbreviation tables, symbol
  expansions, punctuation dedup), each regex pass as a left-to-right
  scanner with sequential `re.sub` semantics.
- `tests.rs`: fixture parity and regressions.
- Shared sampler math (log-softmax, top-p, categorical draw) lives in
  [../sampling.rs](../sampling.rs).

## Checkpoint layout

The loader maps the raw checkpoint the way the reference `sanitize`
reads it: LM tensors under `model.*` (plus root `lm_head.weight` when
untied), decoder tensors under `decoder.decoder.*` / `decoder.head.*`
stored float32. MLX `nn.Conv1d` keeps its weight `[out, kernel, in]`
(channels last); the family loader transposes to the `[out, in, kernel]`
layout `ops::conv1d` wants (the same family-owns-the-transpose rule the
gpt2 backbone documents). Both halves load through `lm::Weights`: the
backbone with the auto-detected `model.` prefix, the decoder with an
empty prefix.

## Scope decisions and deviations from the reference

- Batch size 1, non-streaming single decode per sentence; the reference
  `stream_generate` yields the same hidden stack incrementally and its
  streaming intervals are out of scope. The hidden stack (prefill row
  plus one row per accepted token) is exposed via `generate_hidden`.
- Text tokenization and the stop ids are caller-owned (the reference
  loads a Hugging Face tokenizer in `post_load_hook`, encodes with
  `add_special_tokens=False`, and derives the stop id from the
  tokenizer pad token / `[STOP]` / eos).
- `convert_to_ascii` deviation: the reference NFKD-decomposes Unicode
  before dropping non-ASCII, which needs the `unicode-normalization`
  tables this crate does not carry. The port drops non-ASCII characters
  outright; callers that want the reference behavior NFKD-normalize
  before calling (accented Latin otherwise loses its letter instead of
  the diacritic). Everything after that pass (numbers, abbreviations,
  symbols, punctuation, whitespace) matches the reference byte for byte
  on the fixture cases.
- Python `re` word boundaries are matched over ASCII word characters;
  non-ASCII word chars in abbreviation boundaries behave slightly
  differently (moot after the ASCII filter in the normal pipeline).
- Sampled (temperature > 0) draws use the crate-local MLX-compatible
  threefry RNG; the filtered distribution is pinned by fixtures,
  temperature 0 never draws and is fully deterministic.

## Verification

Fixture protocol per the repo standard:
`tools/gen_soprano_fixtures.py` runs the true reference modules against
a tiny seeded model (qwen3, 2 layers, hidden 32, untied lm head,
vocab 256, weights F16; decoder dim 16, 2 ConvNeXt blocks, n_fft 32,
hop 8, upscale 4, so one token covers 32 samples), driven from
`../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_soprano_fixtures.py
```

Fixtures live in `testdata/tts_soprano/` with SHA-256 digests in
`manifest.json`:

- `text/`: clean_text cases (years incl. 2000/2005 and the 1234
  branch, ordinals, money with comma stripping and the three-part
  `1.2.3` path, K/M suffixes and `#7`, abbreviation tables, symbol
  expansions, punctuation dedup, whitespace, casing) and
  `_preprocess_text` pieces (split, short-piece merges on both sides,
  no-split, trailing whitespace, min_length 0).
- `decoder/`: the tiny decoder weights and its waveform on a pinned
  seeded hidden matrix (6 positions).
- `generate/`: one backbone, three greedy scenarios: the length run
  exhausts max_tokens 8 (its first and fourth draws become the stop ids
  of the other runs), a late stop after exactly three tokens, and an
  immediate stop that keeps only the prefill row (exercising the
  zero-token trim path). Generated ids, stop reasons, and waveforms are
  pinned.

Measured parity when the gates were frozen: decoder waveform 1.4e-7 max
absolute difference and end-to-end generate waveforms 6.6e-7 (gate
2e-6); the text pipeline, prompt pieces, and generated token ids are
exact. The backbone itself is pinned separately by the `lm` fixtures at
2e-4.

`cargo test -p turbospark-audio --lib tts::soprano` runs the family
gate.

## Open items

- No real-checkpoint gate: a Soprano-1.1-80M end-to-end run (real
  tokenizer, real decoder weights, waveform sanity against the Python
  family) has not been executed; the fixtures are seeded tiny
  instances.
- Streaming intervals and batched prompts are out of scope.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
- The decoder-only fixture exercises the input_kernel 1 backbone; the
  pre-1.1 input_kernel 3 geometry is config-tested (the conv transpose
  is kernel-agnostic).
