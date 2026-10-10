# OuteTTS family (two-codebook DAC)

Port of `mlx_audio/tts/models/outetts/` (outetts.py, prompt_processor.py,
tokens.py, audio_processor.py, dac_interface.py) at mlx-audio 0.5.7,
pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/outetts)
(mlx 0.32.3). The family wraps a llama, qwen2, or qwen3 backbone (shared
[crates/audio/src/lm](../../lm/README.md) ports, selected by the config
`model_type` exactly like the reference model map; the deployed
OuteTTS-1.0 checkpoints are llama) that emits interleaved `<|c1_i|>` /
`<|c2_i|>` audio tokens: the completion prompt frames
`{bos} {text_start} text {text_end} {audio_start}` plus a word-level
speaker block, the loop samples until the tokenizer end-of-text id or
`max_tokens`, and the extracted two codebooks decode through the 24 kHz
DAC (`mlx-community/dac-speech-24khz-1.5kbps`, two books of 1024
entries) followed by the BS.1770 loudness stage (`process_audio_array`:
pad to one 0.4 s block, normalize to -18 LUFS, peak limit to -1 dBFS).

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: the `SpecialTokens` strings, `text_normalizations`,
  `chunk_text`, `merge_speaker_text`, `create_codes`,
  `get_global_features`, the `Speaker` JSON parser, `AudioTokenMaps`
  (the reference `get_audio_token_map` / `extract_audio_from_tokens`),
  `ModelConfig` with the model-type dispatch, `GenerateParams` with the
  reference defaults (temperature 0.4, top-p 0.9, min-p 0.05, top-k 40,
  repetition penalty 1.1 over a 64-token context, max_tokens 1200), the
  generate loop, and `decode_codes_to_audio`.
- `loudness.rs`: `process_audio_array` and the BS.1770 helpers it calls
  (`integrated_loudness`, `normalize_loudness`, `normalize_peak`, the
  K-weighting biquads, the direct-form-II `lfilter`), all f64 like the
  numpy reference.
- `tests.rs`: fixture parity and regressions.
- Shared sampler math (logit bias, sign-aware repetition penalty,
  log-softmax, top-p, min-p, top-k, categorical draw) lives in
  [../sampling.rs](../sampling.rs); min-p was added for this family
  (its `make_sampler` call enables min-p 0.05 by default) and is shared.
  The sampler chain order is the reference `make_sampler` order:
  top-p, then min-p, then XTC (never enabled by any family, not
  ported), then top-k.

## Prompt construction is string-level

`Model.get_completion_prompt` produces the exact reference prompt
string; the caller tokenizes it with `add_special_tokens=False`. The
`<|c1_i|>` / `<|c2_i|>` token ids come from the caller through
`AudioTokenMaps::from_lookup` (the reference builds the same maps from
its Hugging Face tokenizer over `range(1025)`). The generate loop stops
on caller-supplied `stop_token_ids` (the reference stops on the
tokenizer's eos ids through `stream_generate`); the stopping token is
included in the returned ids, matching the reference loop appending the
final `stream_generate` response.

## Scope decisions and deviations from the reference

- Batch size 1, non-streaming single decode per segment; the reference
  `stream` / `streaming_interval` partial-yield path is out of scope.
- Text tokenization, the audio-token vocabulary lookup, and the stop
  ids are caller-owned (see above).
- Speaker creation from reference audio (`create_speaker_from_whisper`,
  `create_speaker_from_dict`) is out of scope: it needs a
  word-timestamped transcript (whisper). Speakers arrive as the
  reference speaker JSON (`Speaker::from_json`; the shipped
  `default_speaker.json` shape), which is what `get_completion_prompt`
  consumes.
- Multi-chunk DAC decode: the reference concatenates its 4096-code
  decode chunks along the size-1 channel axis, which only works while
  the decode is a single chunk (a second `[1, samples, 1]` chunk
  mismatches on the samples axis and `mx.concatenate` raises). The port
  concatenates along samples, which is the single-chunk path the
  reference takes and the clear intent beyond it.
- `create_codes` refuses a c2 book shorter than c1 (the reference
  indexes past the end), and `get_completion_prompt` refuses a speaker
  with no words (the reference `words[-1]` index error).
- Sampled (temperature > 0) draws use the crate-local MLX-compatible
  threefry RNG, so a fixed seed fixes the Rust sequence while the
  filtered distribution is pinned by fixtures; temperature 0 never
  draws and is fully deterministic.
- The word-codebook feature extraction (`Features.extract_audio_features`)
  is out of scope with speaker creation; the stored integer features
  flow through the prompt exactly.

## Verification

Fixture protocol per the repo standard:
`tools/gen_outetts_fixtures.py` runs the true reference modules against
a deterministic offline `PreTrainedTokenizerFast` (word-level
vocabulary over the special tokens; c1 ids 200+i, c2 ids 1300+i, the
mock end-of-text id 2), a seeded tiny two-book DAC (n_codebooks 2,
codebook 64, hop 8, 24 kHz), and a seeded tiny llama (2 layers, hidden
32, vocab covering the mock tokenizer, weights F16), driven from
`../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_outetts_fixtures.py
```

Fixtures live in `testdata/tts_outetts/` with SHA-256 digests in
`manifest.json`:

- `dac/`: the tiny codec, the reference c1/c2 token maps,
  `extract_audio_from_tokens` cases (interleaved, unknown-id skips,
  book-length truncation, empty), the chunked 4500-code decode
  (exercises the chunk seam), the loudness measurement, and the
  loudness-normalized waveforms (full and pad-path short inputs).
- `prompt/`: text_normalizations, chunk_text (grouping, oversized
  sentences, CJK split), merge_speaker_text (Latin, question, empty
  speaker, CJK), and the `create_codes` / `get_completion_prompt`
  strings on a pinned speaker with the last-word mutation.
- `generate/`: the pinned prompt ids and two greedy scenarios at
  penalty 5.0 (stop-token stop after audio tokens with the decoded
  waveform, and max_tokens stop after 10 draws).
- `sampling/`: the full family chain on fixed logits (logit bias +
  repetition penalty 1.1 over 64, log-softmax, top-p 0.9, min-p 0.05,
  top-k 40), a min-p case with `min_tokens_to_keep = 3` (the
  guaranteed-survivor path), and the penalty window with duplicates.

Measured parity when the gates were frozen: DAC decode plus loudness
chain 1.5e-7 max absolute difference and the greedy end-to-end waveform
3.6e-7 (gate 2e-6); the loudness measurement sits at 2.8e-14 (gate
1e-6); sampler stages and the generated token ids, extracted codebooks,
and prompt strings are exact.

`cargo test -p turbospark-audio --lib tts::outetts` runs the family
gate.

## Open items

- No real-checkpoint gate: an OuteTTS-1.0 checkpoint end-to-end run
  (real tokenizer, real DAC, waveform sanity against the Python family)
  has not been executed; the fixtures are seeded tiny instances.
- The qwen2 and qwen3 dispatches are config-tested but their
  family-level fixtures use the llama backbone (the deployed 1.0
  profile).
- Streaming intervals, batched prompts, and speaker creation from audio
  are out of scope.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
