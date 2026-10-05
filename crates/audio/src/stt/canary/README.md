# Canary

This module ports `mlx_audio/stt/models/canary/` (canary.py, config.py,
decoder.py, tokenizer.py) from mlx-audio 0.5.7, source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

Canary-1B-v2 is NVIDIA's multilingual ASR and x->en / en->x translation
model. The upstream port composes two existing families plus one new
decoder:

- a FastConformer encoder (relative-position attention, 32 layers,
  d_model 1024, 8 heads, 4x feed-forward, 8x depthwise-striding
  subsampling, kernel 9, 128 mel bands) that is the Parakeet `Conformer`
  verbatim; and
- a small pre-norm transformer decoder (8 layers, hidden 1024, 8 heads,
  inner 4096, relu feed-forward) with a fixed sinusoidal positional table
  (`FixedPositionalEncoding`, max 1024 positions), cross-attention over
  the encoder states, and a 16384-piece SentencePiece vocabulary.

Transcription and translation are prompt-driven: the decoder input is
`<|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|src|>
<|tgt|> (<|pnc|>|<|nopnc|>) <|noitn|> <|notimestamp|> <|nodiarize|>`
and generation stops at `<|endoftext|>`.

The Rust port reuses `crate::vad::sortformer::FastConformer` for the
encoder forward (the reference imports the Parakeet Conformer the same
way) and `turbospark_audio::nemo_mel` for the NeMo log-mel frontend
(per-feature normalization, preemph 0.97, log guard 2^-24). The shared
encoder gained a `load_canary` loader next to `load_parakeet` in
`src/vad/sortformer/mod.rs`: the Canary conversion uses the same tensor
names as the Parakeet layout but quantizes every linear to 8-bit
groupwise affine and keeps the linear biases (`use_bias: true`,
`linear_pos` bias-free; `xscaling: false`). The decoder, prompt builder,
vocabulary parsing, and greedy loop live in this module
(`decoder.rs`, `tokenizer.rs`, `config.rs`).

## Pinned profile

| Field | Value |
|---|---|
| Profile constant | `CANARY_1B_V2_Q8` |
| Repository | `Mediform/canary-1b-v2-mlx-q8` |
| Revision | `0b6b32ee10f30c89e3ead7249bb636445e3019ee` |
| Input | mono f32 PCM at 16 kHz |
| Encoder | FastConformer, 32 layers, d_model 1024, 8 heads, ff 4096, conv kernel 9, subsampling conv channels 256 |
| Decoder | 8 layers, hidden 1024, 8 heads, inner 4096, relu, max 1024 positions |
| Vocabulary | 16384 SentencePiece pieces embedded in `config.json` at `tokenizer.model_base64` |
| Quantization | 8-bit groupwise affine, groups of 64 (`config.quantization.bits = 8`, MLX default group size; the port refuses other schemes) |
| Task verified | `source_lang=en, target_lang=en, use_pnc=true`, greedy |

Pinned artifact sizes are bytes; SHA-256 digests were computed from the
installed snapshot at the pinned revision.

| File | Size | SHA-256 |
|---|---:|---|
| `config.json` | 673039 | `fdecfe775789a06d2e2860de7f73dbb311b6f78f7694d1e2dff712aca36b56a5` |
| `model.safetensors` | 1136436574 | `f637b904eeb83d4327158b3df0a687e8ba8cdde2cde1576127f696dc36a84ba5` |

## Inference contract

- `Canary::load(model_dir)` parses `config.json`, refuses any graph
  switch outside the implemented layout (unsupported self-attention
  model, causal downsampling, `pre_ln = false`, non-relu feed-forward,
  `learn_positional_encodings = true`, an encoder output projection,
  quantization other than 8-bit groups of 64), reconstructs the tokenizer
  from the embedded protobuf, and dequantizes all linears through
  `crate::quant`.
- `transcribe(samples)` runs the reference defaults (en -> en, punctuation
  and capitalization on, greedy, `max_tokens = 200` like
  `Model.generate`). `transcribe_with_options` exposes source language,
  target language, `use_pnc`, and the token budget; unknown language
  codes fail the vocabulary lookup exactly like the reference's
  `token2id` KeyError.
- Decode: prefill the 9 prompt tokens with a causal mask and fixed
  positional rows, argmax the last logits row, then single-token steps
  that extend the self-attention KV cache; cross-attention keys and
  values over the encoder output are computed once (the reference's lazy
  cross cache). `max_tokens = 0` still yields the first sampled token,
  matching the reference loop shape.
- The encoder mask from the reference (`arange < enc_len`) is all-valid
  for the single-clip full-length path, so the port applies validity
  masking (`-1e9` additive) only for frames past `encoder_len`, which is
  a no-op here; the smoke fixture records `encoder_mask_all_valid: true`
  to pin that assumption.

### Deliberate divergences from the reference

1. **Compute dtype.** The reference casts the mel to bfloat16 and every
   quantized layer dequantizes into bfloat16, so the whole reference
   pipeline rounds to bf16 between ops. The port computes the
   dequantized f32 pipeline (crate rule: portable f32 CPU). Stage
   comparisons against the fixture therefore use relative gates; the
   greedy decode is token-exact anyway (see evidence).
2. **Positional tables.** The reference materializes the encoder
   relative-position table for `pos_emb_max_len = 5000` and grows it on
   demand; the shared port computes the `[2T - 1, d_model]` slice per
   sequence. Values agree to f32 rounding (fixture-pinned). The decoder
   table is fixed at `max_sequence_length = 1024` and out-of-range
   positions return an error instead of the reference's index panic.
3. **Weight names.** The pinned conversion stores the converter's
   MLX-native names (`transf_decoder.layers.N.first_sub_layer.linear_q`,
   `head.classifier`) rather than the sanitized module names; the port
   reads them directly, which is the same tensors the reference's
   `_sanitize_mlx_native` remap would feed the model.
4. **Decode rendering.** `sp.decode` is reproduced for the sequences the
   greedy loop produces: control pieces dropped, normal and user-defined
   pieces concatenated with U+2581 mapped to space, then the final strip
   matching the reference's `text.strip()`. Unknown-type ids refuse
   (sentencepiece raises); byte-fallback pieces do not exist in this
   vocabulary and refuse as unsupported.
5. **WAV input.** `mlx_audio.audio_io` decodes WAV through miniaudio,
   which quantizes float32 WAV input to 16 bits; the crate reader
   consumes the true float32 samples. The fixture generator reads the
   float32 data chunk directly (same mitigation as Mega-ASR) so both
   sides of every gate see bit-identical inputs; the stock
   `model.generate(mx.array(samples))` path produces the same transcript
   either way.

## Verification evidence

All fixture values come from
`crates/audio/scripts/generate_canary_fixture.py` run with the pinned
mlx-audio 0.5.7 checkout against the installed checkpoint and the shared
smoke clip `crates/audio/testdata/qwen3_forced_aligner_reference.wav`
(float32 WAV, 44715 samples at 16 kHz, SHA-256
`ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb`).
The fixture is `crates/audio/testdata/canary_reference.json` and embeds
the checkpoint's tokenizer protobuf plus mel/encoder/decoder stage
witnesses with provenance (repository, revision, source commit, audio
digest, per-file digests).

- **Python reference transcript (pinned checkpoint, greedy):**
  `The quick brown fox jumps over the lazy dog.`
  280 mel frames, 35 encoder frames, 9 prompt tokens
  `[7, 4, 16, 64, 64, 5, 9, 11, 13]`, 15 generated tokens
  `[1839, 1259, 2172, 4033, 2210, 2143, 16138, 10077, 2392, 2255, 1289,
  1273, 1470, 5391, 16073]` = `The qu ick bro wn fo x jum ps over the la
  zy dog .`
- **Rust checkpoint transcript:** identical, byte for byte, including
  the generated token id sequence and prompt token ids (gated test
  `pinned_checkpoint_matches_reference_stages_and_transcript`, release
  build, `TURBOSPARK_CANARY_MODEL_DIR` set).
- **Always-on fixture parity (no checkpoint needed):** log-mel on the
  smoke clip matches the reference within 2.0e-4 absolute, measured
  maximum 9.350e-6. Tokenizer parse (16384 pieces, special ids,
  selected piece strings), both prompts (`en->en pnc`,
  `de->fr nopnc` = `[7, 4, 16, 78, 71, 6, 9, 11, 13]`), vocab decode of
  the generated ids, and both positional tables match the fixture; the
  decoder table spots agree within 1.0e-6 and the encoder relative
  slice within 1.0e-5.
- **Config refusals (always-on):** 4-bit and non-64 group quantization,
  causal downsampling, non-rel_pos attention, gelu feed-forward,
  `pre_ln = false`, encoder output projection geometry, head class
  mismatch, and non-16 kHz or non-`per_feature` preprocessing all
  refuse.
- **Checkpoint stage parity (gated test):** mel 9.350e-6 max absolute
  (gate 2.0e-4); encoder output witness 6.686e-3 max absolute, 8.112e-3
  of the witness scale (gate 2.0e-2) - the bf16-vs-f32 gap of the
  reference encoder; prefill last hidden row 3.811e-2 max absolute,
  4.625e-3 of the row scale (gate 2.0e-2); reference top-8 prefill
  logits all survive inside this port's top-16 with a maximum value
  difference 9.919e-2, 4.859e-3 relative (gate 5.0e-2). Generated ids
  and the transcript are exact.
- Backend profile: CPU f32, release build. The gated test completed in
  3.7 s wall for load, dequantize, and the full transcription of the
  smoke clip in one process (2026-10-05, darwin arm64); that is a single
  exploratory reading, not a benchmark row.

Baseline for the crate at the start of this port: all `stt` tests green.
The canary tests add 9 always-on tests and 1 ignored gated test. The
full-crate `cargo test -p turbospark-audio --lib` run after the port
shows 289 passed / 1 failed / 23 ignored, the single failure being the
pre-existing concurrent-work `codec::bigvgan::tests::tiny_end_to_
end_matches_reference`, untouched by this port.

## Remaining gates

- **Task quality.** No WER, BLEU, or multilingual evaluation has been
  run. The 25-language claim of Canary-1B-v2 is not reproduced here;
  compilation and parity on one English smoke clip do not establish
  quality.
- **Translation path.** `transcribe_with_options` builds any prompt the
  vocabulary supports, but only the en->en task has decoded evidence;
  a de/fr decode parity run would need a suitable non-English clip.
- **Long-form audio.** Single-clip full-length encoding only; no
  chunking, streaming, or padded-batch path exists (the reference has
  none either).
- **Runtime, catalog, FFI, Swift.** No `SpeechFamily` entry, catalog
  row, install probe, `ts_stt_*` ABI dispatch, or Swift selection
  exists for this family yet.
- **Performance.** No benchmark rows; the CPU f32 path dequantizes
  about 1.1 GB of 8-bit weights per load.

## Local checks

```sh
hf download Mediform/canary-1b-v2-mlx-q8 \
  --revision 0b6b32ee10f30c89e3ead7249bb636445e3019ee \
  --local-dir ~/models/canary-1b-v2-q8

cargo test -p turbospark-audio --lib stt::canary

TURBOSPARK_CANARY_MODEL_DIR=~/models/canary-1b-v2-q8 \
  cargo test --release -p turbospark-audio --lib stt::canary -- \
  --ignored --nocapture --test-threads=1
```

The fixture generator (never part of tests):

```sh
~/.venv-mlxaudio/bin/python crates/audio/scripts/generate_canary_fixture.py \
  --model-dir ~/models/canary-1b-v2-q8 \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/canary_reference.json \
  --revision 0b6b32ee10f30c89e3ead7249bb636445e3019ee
```
