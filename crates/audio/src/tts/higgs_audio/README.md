# Higgs Audio v2 TTS family (Boson AI)

Port of `mlx_audio/tts/models/higgs_audio/` at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/higgs_audio)
(mlx 0.32.3). The family is a Llama-3.2-style backbone whose every layer
is a dual-FFN block: text and audio positions share self-attention while
the input/post norms and the MLPs come in text and audio variants routed
by the `audio_out_mask` (`HiggsDualFFNDecoderLayer`). A separate audio
codebook embedding table sums per-codebook lookups, and a two-head
projector emits text logits plus `[K, codebook_size + 2]` audio logits
(`HiggsAudioDecoderProjector`).

Generation follows the reference state machine: the first frame is the
synthetic all-BOS AUDIO_INIT (prefill logits are never sampled), tail
codebooks are ramped in with BOS (`use_delay_pattern`), each step draws
one code per codebook (greedy through `sampling_warmup_frames`, then
temperature with optional top-k/top-p and the Gumbel-max draw), the
repetition-avoidance sampler (RAS) re-draws codebooks whose token
already fills its recent window, any codebook EOS starts a K-frame
ramp-out that forces EOS prefixes until it completes, and the raw
delayed frames revert through `revert_delay_pattern`, drop the boundary
BOS/EOS columns, clip into the codec range, and decode through the
Higgs audio codec with linear edge fades.

The codec itself is the already-ported
[crates/audio/src/codec/higgs_audio](../../codec/higgs_audio/README.md)
tokenizer; this family consumes `decode_tokens` over the aligned
codebook-major codes.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: config (`TextConfig` plus the audio geometry and generation
  flags, `HiggsAudioConfig.from_dict` semantics), the dual-FFN model
  (`forward_hidden`, `forward`), the delay-pattern helpers
  (`revert_delay_pattern`, `apply_delay_pattern`,
  `build_delay_pattern_mask`), `lookup_audio_embedding`, the sampler
  (`greedy_sample_audio`, `sample_audio` with the kth-value top-k and
  descending-cumsum top-p, Gumbel-max over crate-local threefry
  uniforms), RAS, the frame loop (`generate_frames`), the end-to-end
  `generate` with codec decode and fades (`apply_fades`), and the
  prompt builders (`smart_voice_prompt_string`,
  `voice_clone_prefix_string`, `voice_clone_middle_string`,
  `build_prompt_smart_voice`, `encode_reference`,
  `build_prompt_voice_clone`).
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch size 1, non-streaming: the streaming overlap-add path
  (`iter_overlap_add_pcm`, `generate_stream*`) is out of scope; a
  segment decodes once, after generation.
- Text tokenization is a caller concern: the reference loads the
  checkpoint's Hugging Face tokenizer and encodes the ChatML-style
  prompt strings with `add_special_tokens=False`. The exact strings are
  exposed as functions and the builders take pre-tokenized ids.
- Reference-audio ENCODE is out of scope: the codec `encode` hard-requires
  the wav2vec2 semantic tower, which the portable codec contract does not
  port. `encode_reference` takes precomputed `[K][T]` codes (for example
  from the Python codec) and performs the delay wrap, the summed
  codebook embedding, and the transcript prefix embedding.
- Weight layout: the reference module tree names (`embed_tokens`,
  `layers.*`, `norm`, `audio_decoder_proj.*`,
  `audio_codebook_embeddings`) load directly; a bosonai-style save that
  nests the backbone under `model.` is detected and followed. The real
  checkpoint's exact tensor naming has not been inspected (no network);
  the loader's probe covers both layouts.
- `use_audio_out_self_attention = true` and
  `audio_decoder_proj_num_layers != 0` are refused: the reference parses
  and ignores both flags (it never builds the extra audio attention
  layers or projector transformer), so a config enabling them describes
  an architecture the reference itself cannot load. The pinned v2 3B
  checkpoint runs with `false` / `0`.
- The sampled (temperature > 0) draw uses the crate-local threefry RNG
  in place of `mx.random.uniform`, so sampled runs are deterministic in
  Rust but not token-identical to mlx at the same seed. Greedy runs
  never draw. The reference's per-call `mx.random` state makes sampled
  parity impossible by construction.
- The Gumbel uniforms draw `K * (C+2)` values per step in row-major
  order (the reference draws the whole `[1, 1, K, C+2]` at once).
- The reference `generate` stop reason strings are preserved in the
  fixture traces (`eos-ramp-complete-at-frame-N`, `max-frames-N`); the
  port returns the `StopReason` enum.

## Verification

Fixture protocol identical to the other audio families:
`tools/gen_higgs_audio_fixtures.py` runs the true reference modules
(`HiggsAudioModel` at tiny seeded dims, the serve.py prompt steps, and a
tiny seeded codec standing in for the network tokenizer with the real
RVQ/fc2/acoustic-decoder stacks), driven from `../mlx-audio` at the
pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_higgs_audio_fixtures.py
```

Fixtures live in `testdata/tts_higgs_audio/` with SHA-256 digests in
`manifest.json` (backbone seed 1, stream-EOS head amplification 1.6,
stored F16 with traces recorded from the upcast values):

- `generate/` (tiny model, 2 dual-FFN layers, hidden 32, text vocab 96,
  3 codebooks of 16 + stream BOS/EOS, llama3 rope scaling; codec of 3
  RVQ books, latent 8): the voice-clone prompt build (prefix/middle ids,
  reference codes, embeddings, routing mask), a forward pin over those
  positions (text and audio logits with a mixed mask), and two greedy
  scenarios. The EOS scenario stops through a full ramp-out at raw frame
  8 (9 raw frames, 5 aligned); the max-frames scenario stops at the
  budget of 6. Both pin the raw delayed frames, the aligned codes, the
  stop reason, the tiny-codec waveform, and the faded waveform.
- `codec/` (config.json + model.safetensors in the
  `HiggsAudioTokenizer::open` layout): the decode side of the family
  gate.
- `delay/` (traces only): `revert_delay_pattern`,
  `apply_delay_pattern`, `build_delay_pattern_mask` on fixed grids, and
  `lookup_audio_embedding` on a fixed id block.
- `sampling/` (traces only): `greedy_sample_audio` on fixed logit
  blocks, the top-k kth-value mask with a tie case, and the top-p
  descending-cumsum mask; a temperature 0.8 Gumbel record is kept for
  reference only (not a gate).

Measured parity when the gates were frozen: forward text and audio
logits 8.9e-7 / 8.5e-7 (gate 2e-5), prompt and reference embeddings
exact (gate 1e-6), waveforms 2.2e-8 (gate 1e-6); delay math, generated
frames, aligned codes, and stop reasons exact. The backbone math itself
is pinned by the `lm` fixtures at 2e-4 and the codec by its own family
fixtures.

`cargo test -p turbospark-audio --lib tts::higgs_audio` runs the family
gate.

## Open items

- No real-checkpoint gate: a bosonai v2 generation 3B run (real
  tokenizer, the q8/q6 mlx-community profiles, waveform sanity against
  the Python family) has not been executed; the fixtures are seeded tiny
  instances. Greedy decoding of a real checkpoint needs the audio
  codebook ranges and stream BOS/EOS heads to be learned behavior, which
  no tiny seeded model can exercise.
- The bosonai checkpoint tensor naming (module tree vs `model.` prefix)
  is unverified against a real download; the loader probes both.
- Streaming overlap-add, batched prompts, and quantized-checkpoint
  loading (the `model_quant_predicate` protected audio head) are out of
  scope.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
