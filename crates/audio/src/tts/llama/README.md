# Llama SNAC TTS family (Orpheus-style)

Port of `mlx_audio/tts/models/llama/llama.py` at mlx-audio 0.5.7,
pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/llama)
(mlx 0.32.3). The family is a dense llama backbone (shared
[crates/audio/src/lm](../../lm/README.md) port) that interleaves text
with discrete SNAC audio codes: after `[SOH] text [EOT EOH]` it emits
`[SOS]` plus flat 7-code groups (one SNAC L1 code, two L2 codes, four L3
codes per group, slots offset by 0/1/2/3/4/5/6 times 4096), stops at the
end-of-speech token, and the group stream decodes through the 24 kHz
SNAC codec. The deployed reference checkpoint is
`mlx-community/orpheus-3b-0.1-ft-bf16` over
`mlx-community/snac_24khz`.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: config (backbone args plus `tokenizer_name` and
  `sample_rate`), special-token constants, `codes_to_layers`,
  `decode_audio_from_codes`, `encode_audio_to_codes`,
  `prepare_input_ids`, `prepare_zeroprompt`, `parse_output`, the
  generate loop (`generate`, `generate_from_prompt`,
  `generate_tokens_from_prompt`), and `GenerateParams` with the
  reference defaults (temperature 0.6, top-p 0.8, top-k off, repetition
  penalty 1.3 over a 20-token context, max_tokens 1200).
- `tests.rs`: fixture parity and regressions.
- Shared sampler math (logit bias, sign-aware repetition penalty,
  log-softmax, top-p, top-k, categorical draw) lives in
  [../sampling.rs](../sampling.rs) and is shared with the qwen3 family.

## Special tokens

| Token | Id | Role |
|---|---|---|
| END_OF_TEXT | 128009 | ends the prompt text |
| START_OF_SPEECH | 128257 | `parse_output` crops after its last occurrence |
| END_OF_SPEECH | 128258 | stops the generate loop, removed by parse |
| START_OF_HUMAN | 128259 | opens the prompt |
| END_OF_HUMAN | 128260 | closes the prompt |
| START_OF_AUDIO | 128261 | zeroprompt wrapper |
| END_OF_AUDIO | 128262 | zeroprompt wrapper |
| AUDIO_OFFSET | 128266 | flat code c rides token `128266 + c` |

## Scope decisions and deviations from the reference

- Batch size 1 only (the reference padding/masking path for batches is
  not ported). Streaming decode intervals and `decode_audio_stream` are
  out of scope; a segment decodes once, after generation.
- Text tokenization is a caller concern: the reference loads a
  Hugging Face tokenizer from `tokenizer_name`; the portable crate does
  not embed one. `prepare_input_ids` takes pre-tokenized ids, and the
  reference `voice` argument (it prefixes `"{voice}: "` before
  tokenization) folds into those ids.
- `encode_audio_to_codes` is implemented over the Rust SNAC encoder
  (the Rust snac port exposes `encode`, so no refusal was needed).
- The generate decode uses the zero-noise codec backbone (`|| 0.0`
  noise source). The reference draws fresh MLX normals for the SNAC
  noise terms on every decode; callers wanting stochastic noise pass
  their own source to `decode_audio_from_codes`. Sampling draws use the
  crate-local MLX-compatible threefry RNG
  (`music::minimax_music3::rng`): the filtered distribution is pinned
  by fixtures, but a sampled (temperature > 0) Rust run draws different
  tokens than mlx at the same request. Temperature 0 is fully
  deterministic and never draws.
- min-p and XTC sampling are not ported: the reference family calls
  `make_sampler` with both disabled.
- The reference's `codes_to_layers` group count `(len + 1) / 7` indexes
  out of bounds on a remainder-of-6 tail; the port refuses by name.
- The reference's Python decode wraps negative layer codes silently
  through MLX negative indexing; the Rust codec refuses negative codes.
  Real checkpoints emit slot-consistent codes so this only matters for
  garbage input.

## Verification

Fixture protocol identical to the other audio families:
`tools/gen_llama_fixtures.py` runs the true reference module (its
network SNAC singleton patched out and replaced with a tiny seeded
noise-free codec), driven from `../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_llama_fixtures.py
```

Fixtures live in `testdata/tts_llama/` with SHA-256 digests in
`manifest.json`:

- `decode/` (tiny codec `model.safetensors` f32,
  sha256 4b5c8dfb09853b1c...): seven `parse_output` cases on fixed id
  sequences (last-SOS crop, second-SOS override, below-one-group trim,
  negative-code faithfulness, tail trim, end-of-speech removal, empty
  crop), the flat-group split, the codes-to-waveform decode, the encode
  inverse on a seeded wave, and the zeroprompt id layout.
- `generate/` (tiny llama, 2 layers, hidden 32, vocab 128307, weights
  F16 sha256 233a434137535afc...): two greedy scenarios
  (end-of-speech stop after 11 tokens, max_tokens stop after 10) driven
  by the reference `generate_step` with the family logits processors at
  penalty 5.0 (a stronger value than the 1.3 default so the tiny
  model's argmax leaves single-token traps; recorded per scenario).
  Generated ids and parsed codes must match exactly.
- `generate_eos/` (second seeded backbone, F16 sha256
  60da854b410cb3ed...): the first greedy draw is the end-of-speech
  token, so the parsed codes are exactly the prompt's slot-valid groups
  and the segment waveform decodes cleanly end to end.
- `sampling/` (traces only): logit bias, sign-aware repetition penalty
  (window truncation and duplicate tokens), log-softmax, top-p, and
  top-k on fixed tie-free logit vectors; filtered entries carry null
  sentinels for negative infinities.

Measured parity when the gates were frozen: waveform decode 1.1e-9 max
absolute difference (gate 1e-7); sampler stages bit-exact except the
log-softmax reduction at 9.6e-7 (gate 2e-6); generated token ids, parsed
codes, layer splits, encode codes, and zeroprompt ids exact. The
backbone itself is pinned separately by the `lm` fixtures at 2e-4.

`cargo test -p turbospark-audio --lib tts::llama` runs the family gate.

## Open items

- No real-checkpoint gate: an orpheus-3b end-to-end run (real
  tokenizer, 24 kHz SNAC with noise, waveform sanity against the Python
  family) has not been executed; the fixtures are seeded tiny
  instances. A greedily decoded real checkpoint needs the audio
  codebook ranges to be learned behavior, which no tiny seeded model
  can exercise.
- Streaming intervals, batched prompts, and the batch attention mask
  are out of scope.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
- Voice cloning (`prepare_zeroprompt`) is a parity-only layout: the
  reference itself warns cloning does not work reliably on Orpheus.
