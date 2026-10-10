# MOSS-TTS-Nano TTS family (OpenMOSS MOSS-TTS-Nano)

Port of `mlx_audio/tts/models/moss_tts_nano/` at mlx-audio 0.5.7,
pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/moss_tts_nano)
(mlx 0.32.3). A global GPT-2-style transformer consumes one embedding
row per time step - the text token embedding plus the masked sum of
per-channel VQ audio embeddings - and a one-layer local transformer
turns the global hidden state into, per frame, a continue/stop text
choice (the two candidate embedding rows `audio_assistant_slot` and
`audio_end` act as the text head; anything but the assistant slot stops
the loop) followed by one code per VQ channel through the per-channel
embedding heads, with a sign-aware repetition penalty over each
channel's full history. Frames decode through the MOSS audio tokenizer
(48 kHz deployed, 16 VQ channels over 1024-code books). The deployed
checkpoint is `mlx-community/MOSS-TTS-Nano-100M`.

The `gpt2.py` backbone here is NOT the shared
[crates/audio/src/lm](../../lm/README.md) gpt2: it adds rope positions
(interleaved pairs), a fused biased QKV with equal thirds, and
gelu_new MLPs, while the lm gpt2 has learned positions and no rope. The
family-local transformer reuses the shared attention math
(`lm::engine::Attention`), the plain traditional rope, the fused-linear
split, and the dense KV cache. The `moss_tts` delay-pattern family
shares none of this module; the families stay separate per the
inventory.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: config (`Gpt2Config` with the HF alias fields,
  `ModelConfig` with the reference `from_dict` defaults and
  `local_gpt2_config`), the family-local GPT-2 (`Gpt2`), the prompt
  assembly (`TemplateIds`, `build_user_prompt_prefix` and friends,
  `build_inference_input_ids`, `lightweight_normalize_text`), the
  generate loop (`generate_audio_token_ids`, `generate`), and
  `GenerateParams` with the reference defaults (do_sample on, text
  temperature 1.0 / top-k 50 / top-p 1.0, audio temperature 0.8 /
  top-k 25 / top-p 0.95, penalty 1.2, 375 frames).
- `sampling.rs`: the family sampler port (clamping repetition penalty
  over the full channel history, top-k BEFORE top-p on the
  temperature-scaled scores, masks mapped back onto the raw logits,
  `sample_assistant_text_token` over the two candidates with top-k
  capped at 2). Deliberately not the shared
  [../sampling.rs](../sampling.rs): that one refuses out-of-range
  context ids, windows the penalty, and orders top-p before top-k.
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch size 1 (the reference refuses batches too) and non-streaming
  (the reference has no streaming path). The `attention_mask` argument
  is dropped: `build_inference_input_ids` produces an all-true mask and
  the reference left-padding path only matters for batched input.
- Text tokenization and the token-budget sentence chunking of `text.py`
  are caller concerns (the checkpoint's sentencepiece `tokenizer.model`
  is not embedded). The `text.py` prompt template STRINGS are exported
  as constants; the caller encodes them and hands the ids to
  [`TemplateIds`] and the builder helpers. `lightweight_normalize_text`
  and the prompt id assembly are ported; `prepare_text_for_sentence_
  chunking` / `split_text_into_best_sentences` stay out of scope.
- The reference continuation mode encodes `prompt_text + text` as ONE
  string; the Rust composition concatenates id lists, so the caller
  performs the reference's string concatenation before encoding (the
  fixture tokenizer is character-level, where the two are identical).
- The audio codec is caller-loaded (`MossAudioTokenizer::open`);
  `generate` decodes the first `effective_nq` codebooks (audio-pad
  filler channels beyond `effective_nq` are not decoded).
- Sampling draws use the crate-local MLX-compatible threefry RNG
  (`music::minimax_music3::rng`): the filtered distribution is pinned
  by fixtures, but a sampled run draws different tokens than mlx at the
  same request. `do_sample=False` is fully deterministic and never
  draws.
- Out-of-range prompt audio codes are refused by name where the
  reference embedding lookup would raise an MLX indexing error.

## Verification

Fixture protocol identical to the other audio families:
`tools/gen_moss_tts_nano_fixtures.py` runs the true reference module
(deterministic character-level tokenizer, seeded tiny mono 4-book MOSS
audio tokenizer, greedy `do_sample=False` throughout), driven from
`../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_moss_tts_nano_fixtures.py
```

Fixtures live in `testdata/tts_moss_tts_nano/` with SHA-256 digests in
`manifest.json`:

- `decode/` (tiny 4-book mono MOSS audio tokenizer, 8 kHz, codebook 16,
  f32 weights): the reference encode codes for a seeded wave and the
  `decode_audio_codes` waveform of the generated frames.
- `generate/` (tiny MOSS-TTS-Nano: 2-layer global transformer, 1-layer
  local transformer, 4 VQ channels, n_embd 32, rope, weights f16): the
  composed 209-row voice-clone prompt, four greedy frames (the channel
  repetition penalty visibly shifts channel 0 between frames), and the
  decoded waveform. The continue/stop choice is pinned by engineering
  the two candidate `wte` rows along the first frame's local hidden
  (seed search keeps the whole budget filled).
- `generate_stop/` (engineered-stop twin): the first choice picks the
  audio-end candidate; both reference and port stop with no frames and
  no audio.
- `prompt/` (traces only): the encoded template pieces and the composed
  voice-clone and continuation (with and without reference codes)
  prompt rows from the reference `build_inference_input_ids`.
- `sampling/` (traces only): the clamping repetition penalty (negative
  and out-of-range context ids), the top-k/top-p masks mapped back onto
  raw scores with null sentinels, disabled-filter passthrough, and the
  greedy candidate choice in both directions.

Measured parity when the gates were frozen: waveform decode 6.9e-7 max
absolute difference (gate 2e-6); generated frames, prompt rows, sampler
scores, and candidate choices exact.

`cargo test -p turbospark-audio --lib tts::moss_tts_nano` runs the
family gate.

## Open items

- No real-checkpoint gate: a MOSS-TTS-Nano-100M end-to-end run (real
  sentencepiece tokenizer, 48 kHz codec, waveform sanity against the
  Python family) has not been executed; the fixtures are seeded tiny
  instances.
- The `use_kv_cache=false` replay path is ported but not fixture-covered
  (the cacheless global recompute); the cache path is the deployed one.
- Sampled (do_sample=True) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
