# Breeze TTS 2 family (BreezeBlue)

Port of `mlx_audio/tts/models/breeze_tts/` (breeze_tts.py, config.py) at
mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/breeze_tts)
(mlx 0.32.3). Despite the lm imports, the reference instantiates three
custom stacks over the shared backbone math: a qwen3-style autoregressive
backbone over a summed per-codebook audio embedding (`_Backbone` over
`_AudioEmbedding`, wrapper vocabulary, optional width projector) predicts
codebook zero through a wrapper lm head with one extra EOS class; a llama
depth decoder fills codebooks 1..K-1 with per-position output
projections (`_DepthDecoder` over `_DepthModel` with the placeholder
position, codebook-offset embeddings, and the `codebooks_head` slices);
and a bidirectional T5Gemma2 text encoder embeds the prompt text
(`_TextEncoder` with the scaled token table, EOI replacement, offset
RMSNorms, sliding/full layer pattern, and per-class RoPE bases).

Generation (`generate` for one non-streaming segment): prompt embeddings
flow through the backbone with a prompt cache; each frame samples
codebook zero from the wrapper head (reserved codec ids masked,
sign-aware repetition penalty over the generated first codes, top-p /
top-k via the shared `make_sampler` chain), fills the remaining
codebooks through the depth decoder (each step a full cacheless causal
pass over the placeholder prefix), feeds the frame back through the
audio embedding, and stops at the wrapper EOS class or the frame budget.
Classifier-free guidance runs a second instruct-free prompt when an
instruct is present and the scale is not 1.0, combining logits at both
the wrapper head and every depth step.

The bundled audio codec is the Qwen3-TTS speech tokenizer
(`qwen3_tts/speech_tokenizer.py`, 1449 lines, a separate upstream
component with no Rust port; see the MODELS.md `qwen3_tts` row). This
port therefore ends at the codebook frames; the waveform decode is an
open item tied to that codec port.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: the wrapper config (`ModelConfig.from_dict` with the
  `audio_num_codebooks`/`audio_vocab_size` alias precedence, the nested
  `backbone_config` resolution, the codec `codebook_size` gate, and the
  codec-config sample-rate chain), the audio embedding and qwen3
  backbone, the T5Gemma2 text encoder, the llama depth decoder, prompt
  construction (`prompt_embeddings` with the reference turn,
  codebook-EOS frame, and target text), the sampler
  (`mask_reserved_codec_logits`, `sample`, `apply_repetition_penalty`),
  `depth_tokens`, the generate loop with classifier-free guidance, and
  `GenerateParams` with the reference defaults (max_tokens 750,
  temperature 0.9, top-p 1.0, top-k 50, repetition penalty 1.0).
- `tests.rs`: fixture parity and regressions.

## Scope decisions and deviations from the reference

- Batch size 1, non-streaming; the streaming chunk path and the
  `_split_long_segment` text chunking are out of scope. Both operate on
  raw text before tokenization, and text tokenization is a caller
  concern here (the reference loads the bundled fast tokenizer in
  `post_load_hook`): the caller tokenizes `{speaker}`-prefixed targets
  (`[S0]text`, or `[S0]<ins_bos>instruct<ins_eos>text` with an
  instruct) and the `{speaker}{ref_text}` reference transcript, then
  passes the id sequences.
- Waveform decode is not ported: the bundled audio tokenizer is the
  Qwen3-TTS speech tokenizer, an unported separate component (task 35
  owns it upstream). `generate` returns the frames; the reference's
  empty-audio fallback (decoding a one-frame dummy when EOS precedes all
  frames) is likewise not reproducible and an immediate EOS yields empty
  frames.
- Reference-audio encode needs the codec encoder; with the codec
  unported, `ReferencePrompt` carries precomputed `[K][T]` codes (the
  layout `_encode_reference` produces after its transpose).
- Weight naming: the loader reads the checkpoint's module-tree names
  (`backbone_model.*`, `depth_decoder.*`, `text_encoder.*`, `lm_head`,
  `embed_text_tokens`, `text_encoder_proj`). The checkpoint ties the
  backbone audio wrapper table with the depth decoder table; the loader
  prefers the backbone name and falls back to the depth tensor (the
  reference sanitize always derives the backbone copy from the depth
  tensor). The official checkpoint also drops `codec_model.*`,
  `.initialized`, and `rotary_emb.inv_freq` keys at load; a Rust loader
  reads named tensors, so extra keys are naturally ignored.
- The `_sample` top-k clamp (`min(top_k, valid)`, disabled when the
  clamp reaches the whole width) matches the reference; temperature 0
  takes the argmax exactly like `make_sampler(0, ...)`. Sampled draws
  use the crate-local threefry RNG, so a sampled Rust run diverges
  token-by-token from mlx at the same request while the filtered
  distribution is pinned by fixtures.
- The repetition penalty applies over every generated frame's first
  code with deduplication and no context window (the reference
  `_apply_repetition_penalty`), not the lm generate-step windowed
  processor.

## Verification

Fixture protocol identical to the other audio families:
`tools/gen_breeze_tts_fixtures.py` runs the true reference Model at tiny
seeded dims (mock tokenizer and audio tokenizer), driven from
`../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_breeze_tts_fixtures.py
```

Fixtures live in `testdata/tts_breeze_tts/` with SHA-256 digests in
`manifest.json` (backbone seed 1, stored F16 with traces recorded from
the upcast values):

- `generate/` (tiny model: 2-layer qwen3 backbone at hidden 32 over 4
  codebooks of vocabulary 64 with an 8-wide audio embedding and width
  projector, 3-layer T5Gemma2 encoder at hidden 16 with pattern-3
  sliding/full layers and an EOI id in use, 2-layer llama depth decoder
  at hidden 16): the plain, instruct (cond + uncond), and voice-clone
  prompt-embedding builds, three greedy scenarios (EOS stop after 14
  frames, max-tokens stop at 6, classifier-free guidance at scale 2.0
  over 24 frames), the depth-head mapping logits for head indices 0..2,
  and a second `generate_eos/` model whose wrapper head is EOS-biased so
  the first draw stops with zero frames.
- `math/` (traces only): the reserved-logit mask (negative infinities as
  null sentinels), the `_sample` slice/log-softmax/argmax for both
  `allow_eos` values, the sign-aware repetition penalty with
  deduplication and out-of-range ids, the T5Gemma2 text encoder states
  (EOI replacement, sliding and full layers), and one audio-frame
  embedding through the projector.

Measured parity when the gates were frozen: prompt builds 2.8e-6 (gate
2e-5), text-encoder states 1.1e-5 (gate 1e-4), depth-head logits 9.5e-7
(gate 1e-4), sampler stages exact; prompt ids, generated frames, and
stop reasons exact. The backbone math itself is pinned by the `lm`
fixtures at 2e-4.

`cargo test -p turbospark-audio --lib tts::breeze_tts` runs the family
gate.

## Open items

- No real-checkpoint gate: a `mlx-community/Breeze-TTS-2-mlx` end-to-end
  run (real tokenizer, bundled Qwen3-TTS audio tokenizer, voice design /
  clone / direction modes, waveform sanity against the Python family)
  has not been executed; the fixtures are seeded tiny instances.
- The waveform decode requires the Qwen3-TTS speech tokenizer Rust port
  (the codec encoder AND decoder); until then the family ends at frames.
- The cfg path is pinned at guidance scale 2.0 and the unity-scale
  no-op; real checkpoint guidance quality is unqualified.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
