# MOSS-TTS TTS family (OpenMOSS MOSS-TTS / MOSS-TTSD / MOSS-TTS-Local)

Port of `mlx_audio/tts/models/moss_tts/` at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/moss_tts)
(mlx 0.32.3). A Qwen3 backbone consumes `n_vq + 1` interleaved channels
per step (column 0 is the text token, columns 1.. one code per RVQ
book) and the reference config selects one of two inference variants;
the `moss_tts_delay` and `moss_tts_local` aliases ride this module per
the inventory (the reference directories re-export the same model).

- Delay pattern (`model_type` `moss_tts_delay`; MOSS-TTS v1.0 / v1.5 and
  MOSS-TTSD-v1.0): the backbone heads are `lm_heads.0` (text, over the
  full vocab) plus one head per book (`lm_heads.1..`, last pad slot
  masked); `emb_ext.{book}` are `audio_vocab_size + 1` row embedding
  tables whose pad row embeds like any code. Generation
  (`generate_delay_pattern_ids`) runs the delay bookkeeping: forced
  assistant-delay-slot text while the delay depth is under `n_vq`, a
  forced audio_end when it reaches it, the keep/exclude logit sets
  inside and outside audio, the im_end warmup guard for the first
  `n_vq + 1` steps, per-book codes gated by the
  `audio_lengths`/`delayed_lengths` state with the sign-aware penalty
  over each book's full history, and the loop break on the im_end row.
  Book `b` of frame `f` lands in row `f + b`; the decode undoes the
  delay, drops all-pad rows, splits contiguous segments, and decodes
  each through the MOSS audio tokenizer with the first-segment trim
  ratio `start_length / first_codes_length`.
- v1.5 local transformer (`model_type` `moss_tts_local` or a
  `gpt2_config` block; MOSS-TTS-Local-Transformer-v1.5): the qwen3
  global transformer, separate `text_lm_head` / per-book
  `audio_lm_heads` (not tied to the embeddings), an optional binary
  `local_text_lm_head` for `local_text_head_mode: binary`, and the
  moss_tts_nano `gpt2.py` local transformer (rope positions, fused
  biased QKV, gelu_new) re-declared in `gpt2.rs` because the sibling
  module's loader is private. Generation (`generate_v15_local_ids`)
  mirrors the nano loop: the continue/stop choice gathers the two
  candidate text-head rows (assistant slot vs audio end; top-k capped
  at 2) and, while the assistant slot wins, the local transformer
  replays `[global hidden, previous book embedding, ...]` cacheless
  while each book head samples with the per-book generated-history
  penalty. Prompts are rows: all-audio-pad text rows and slot-token
  audio rows (`audio_user_slot` / `audio_assistant_slot`).

The legacy local-transformer checkpoints (the four
`additional_mlp_ffn_hidden_size` / `local_ffn_hidden_size` /
`local_hidden_size` / `local_num_layers` fields with no `gpt2_config`
and `model_type != moss_tts_local`) are refused at load with a named
`SpeechError::Unsupported`.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: the config (`ModelConfig.from_dict` defaults, the variant
  properties, `local_gpt2_config`), the delay processor in id space
  (`ChatTemplateIds`, `ContentPiece`, `build_delay_message_ids` /
  `build_delay_unified_codes` / `build_delay_prompt`,
  `apply_delay_pattern` / `apply_de_delay_pattern`,
  `normalize_audio_codes`), the v1.5 row builders
  (`V15TemplateIds`, `build_v15_inference_input_ids`, the
  `LOCAL_V15_*` template constants and the after-reference renderer),
  the two generate loops, `decode_generated_audio`, and the
  `DelayParams` / `V15Params` defaults.
- `gpt2.rs`: the family-local GPT-2 local transformer (the nano
  `gpt2.py` copy).
- `sampling.rs`: the `moss_tts/sampling.py` port (clamping penalty over
  the full history, top-k BEFORE top-p on caller-scaled scores, masks
  mapped back onto raw logits, `sample_token`). Deliberately not the
  shared [../sampling.rs](../sampling.rs): that one refuses
  out-of-range context ids, windows the penalty, and orders top-p
  before top-k.
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch size 1 (the reference refuses batches too) and non-streaming
  (the reference streams only the v1.5 local variant through a codec
  session; not ported). The `attention_mask` the processors produce is
  all-true for single conversations and dropped.
- Text tokenization and the `text.py` `normalize_tts_text` cleanup are
  caller concerns (the checkpoint tokenizer is not embedded). The
  processor is ported in id space: the caller splits each rendered
  message content on the `<|audio|>` placeholder and passes
  `ContentPiece` text ids / audio-code blocks; `build_delay_message_ids`
  expands the audio blocks with the role-appropriate slot tokens (user
  messages carry the user slot in both the generation and delay
  positions) and applies the chat template. The v1.5 template strings
  are exported as constants; the caller encodes them plus the rendered
  after-reference block.
- The v1.5 continuation concatenates ref text + target text as ONE
  string in the reference; the Rust composition takes the encoded
  result, so the caller performs the string concatenation before
  encoding (the fixture tokenizer is additive over composition).
- The audio codec is caller-loaded (`MossAudioTokenizer::open`); the
  config's `audio_tokenizer_repo` default resolves to
  `OpenMOSS-Team/MOSS-Audio-Tokenizer` (delay) or
  `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` (the v1.5 local type). The
  decode uses all `n_vq` books; out-of-range codes (pad codes in
  boundary frames) fail with a named error where the reference gather
  reads out of bounds silently.
- Checkpoint naming follows the converted mlx saves (the model
  parameter tree): the backbone under `language_model.` (delay) or
  `transformer.` (v1.5 local), `emb_ext.*` / `lm_heads.*` /
  `audio_embeddings.*` / `text_lm_head` / `audio_lm_heads.*` /
  `local_transformer.*` prefix-free; `Model::open` probes both that
  layout and the raw composite `model.` prefix. The qwen3 backbones in
  these checkpoints carry no `lm_head`, so the loader flips the config
  to the tied path (inert: only `forward_hidden` runs).
- Sampling draws use the crate-local MLX-compatible threefry RNG
  (`music::minimax_music3::rng`): the filtered distribution is pinned
  by fixtures, but a sampled run draws different tokens than mlx at the
  same request. Greedy runs (temperature 0 for the delay variant,
  `do_sample=False` for the v1.5 local) are fully deterministic and
  never draw. The repetition penalty applies before the greedy argmax,
  exactly like the reference `sample_token`.
- `use_static_local_kv_cache` is parsed but unused (the reference's
  static cache path serves the streaming loop; the ported loop uses the
  dynamic prompt cache).

## Verification

Fixture protocol identical to the other audio families:
`tools/gen_moss_tts_fixtures.py` runs the true reference modules
(deterministic character-level tokenizer with multi-char special
tokens, seeded tiny mono 4-book MOSS audio tokenizer, greedy sampling
throughout, engineered text-head rows so the delay bookkeeping and the
continue/stop choices are deterministic), driven from `../mlx-audio`
at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_moss_tts_fixtures.py
```

Fixtures live in `testdata/tts_moss_tts/` with SHA-256 digests in
`manifest.json`:

- `decode/` (tiny 4-book mono MOSS audio tokenizer, 8 kHz, codebook 16,
  f32 weights): the reference encode codes for a seeded wave and the
  decode waveform.
- `delay_generate/` (tiny delay model: 2-layer qwen3, 4 books, head
  width 17): the composed voice-clone prompt, six greedy steps pinned
  to the assistant-slot loop (audio_start, then gen slots with every
  book sampled and the full-history penalty), the reference decode
  trim (which consumes the whole first segment at this
  start_length-to-codes ratio, pinning the trim math) and the
  untrimmed segment decode.
- `delay_stop/` (engineered-stop twin): audio_start, four forced delay
  slots writing the code diagonal, the forced audio_end, and the im_end
  break; the decoded waveform pins the de-delay alignment.
- `delay_prompt/` (traces only): the template piece ids and the unified
  grids for generation with and without reference codes and the
  truncated continuation.
- `delay_pattern/` (traces only): the delay/de-delay round trip.
- `sampling/` (traces only): the clamping penalty, the top-k / top-p
  masks mapped back onto raw scores, disabled-filter passthrough, and
  the greedy penalty-shifted argmax.
- `v15_generate/` (tiny v1.5 local model: 2-layer qwen3 global, 1-layer
  gpt2 local, 4 books): the composed voice-clone prompt rows, four
  greedy frames with the per-book generated-history penalty, and the
  decoded waveform. The continue/stop choice is pinned by engineering
  the two candidate `text_lm_head` rows along the first frame's local
  hidden (seed search keeps the budget filled).
- `v15_stop/` (engineered-stop twin): the first choice picks the
  audio-end candidate; both reference and port stop with no frames and
  no audio.
- `v15_prompt/` (traces only): the voice-clone, text-only, and
  continuation row layouts.

Measured parity when the gates were frozen: waveform decode within
2e-6 max absolute difference (gate 2e-6, measured delta below the gate
on the v1.5 path); prompt rows, generated rows, sampler scores, and
greedy choices exact.

`cargo test -p turbospark-audio --lib tts::moss_tts` runs the family
gate.

## Open items

- No real-checkpoint gate: an end-to-end run against a deployed
  MOSS-TTS / MOSS-TTS-Local-Transformer-v1.5 checkpoint (real
  tokenizer, 24/48 kHz codec, waveform sanity against the Python
  family) has not been executed; the fixtures are seeded tiny
  instances.
- The legacy local-transformer checkpoints are refused, not ported
  (the `MosiTTSModel` embedding-list stack, the local qwen3-style
  transformer, and the per-channel MLP plumbing).
- Streaming (`_generate_v15_local_streaming_results`, codec session,
  chunk scheduling) is not ported; non-streaming only.
- The `use_kv_cache=false` replay path of the v1.5 loop is ported but
  not fixture-covered; the cache path is the deployed one.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
