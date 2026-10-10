# Sesame CSM TTS family (CSM-1B / MisoTTS-style)

Port of `mlx_audio/tts/models/sesame/` at mlx-audio 0.5.7, pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/sesame)
(mlx 0.32.3). The CSM architecture is two llama transformers: a backbone
over summed per-codebook audio embeddings plus a text embedding predicts
codebook 0 through the `codebook0_head`; a smaller depth decoder,
conditioned on the projected backbone hidden state, predicts codebooks
1..K-1 through the `audio_head` tensor. One frame of `K =
audio_num_codebooks` codes decodes through the 24 kHz Mimi codec
(12.5 fps, one frame per 80 ms), the loop stops on an all-zero frame,
and the deployed checkpoints are `sesame/csm-1b` (backbone llama-1B
flavor, depth decoder llama-100M flavor) over
`kyutai/moshiko-pytorch-bf16` with the `unsloth/Llama-3.2-1B` text
tokenizer. `MisoTTS` and the aliases `csm` and `marvis` are the same
family per the MODELS.md inventory.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: config (the two llama backbone configurations with the
  explicit-fields-then-flavor fallback, the audio vocabulary geometry,
  and the `speaker_prefix_space`/`use_default_voice_prompt`/
  `voice_match` flags), the four `llama_flavor` presets,
  `text_prompt_string`, the prompt builders (`tokenize_text_segment`,
  `tokenize_audio`, `tokenize_segment`), `frames_to_codes`,
  `encode_audio`, `generate_frame`, the frame loop
  (`generate`, `generate_frames`), and `GenerateParams` with the
  reference defaults (`make_sampler(temp=0.9, top_k=50)`, 90 s frame
  budget).
- `tests.rs`: fixture parity and regressions.
- The backbone and decoder transformers live in this module (the
  reference `embed_tokens` is an `Identity` and no LM head exists), but
  their blocks reuse the shared `lm::engine` `Attention`/`Mlp` math and
  the `lm::rope`/`lm::cache` constructions; `lm/engine.rs` is unchanged.
- Shared sampler math (log-softmax-free raw-logit filters, top-p,
  min-p, top-k, categorical draw) lives in [../sampling.rs](../sampling.rs).

## The attention variant

`attention.py` replaces every backbone and decoder `self_attn` with a
custom `Attention` whose `Llama3ScaledRoPE` differs from the shared lm
rope path in one load-bearing way: it rotates ADJACENT feature pairs
(`x[2i], x[2i+1]`, the GPT-J/traditional convention) instead of the
split-half convention `mx.fast.rope` uses with `traditional=False`. The
frequency VALUES are the standard llama3 scaling (low 1, high 4, old
context 8192, factor from the config `rope_scaling` - the attention
module hardcodes the three non-factor defaults instead of reading them).
The family therefore builds each stack's rope with
`Rope::initialize(head_dim, rope_theta, traditional=true, llama3-scaling)`,
which reproduces `Llama3ScaledRoPE` exactly; everything else in the
attention (bias on all four projections including `o_proj`, rope before
the cache update, causal masking through the KV cache) matches the
shared engine path.

## Scope decisions and deviations from the reference

- Batch size 1 and non-streaming: `generate_result`'s streaming yields,
  `streaming_interval` bookkeeping, and the batch dimension are not
  ported. The prompt's `curr_pos` tensor is not carried: the reference
  computes it and never feeds it to the model (positions come from the
  cache offsets).
- Text tokenization is a caller concern: the reference loads the
  Llama-3.2 tokenizer from `unsloth/Llama-3.2-1B` (with the bos/eos
  template of `load_llama3_tokenizer`). [`text_prompt_string`] builds
  the exact `"[{speaker}]{text}"` (or `"[{speaker}] {text}"` under
  `speaker_prefix_space`) string the caller tokenizes; `BOS`/`EOS` ids
  belong in the caller's ids, as in the reference.
- The default voice prompt (`default_speaker_prompt` downloading
  `prompts/{voice}.wav` from `sesame/csm-1b`) and `prepare_prompt` audio
  file loading are out of scope (network I/O and audio decoding stay
  caller-side). `voice_match` prompt merging folds into the caller's
  prompt construction: with the flag on, the reference concatenates the
  context text with the prompt text before tokenization, which callers
  reproduce by building one text segment.
- The WATERMARK STAGE IS REFUSED. `watermarking.py` applies the
  silentcipher 44.1 kHz watermark, which downloads a separate model
  checkpoint outside the pinned mlx-audio tree. `Model` construction in
  the reference already tolerates a missing watermarker (`watermarker
  is None` skips the stage), and this port keeps that behavior:
  generated audio is returned unwatermarked, and [`watermark`] fails
  with a descriptive `Unsupported` error instead of porting blind.
- The decode is one full Mimi batch pass over the codebook-major codes.
  The reference decodes through the stateful `MimiStreamingDecoder` in
  62-frame chunks; the fixture gate pins the equality (63-frame
  two-chunk case at 2.7e-7), which is the codec's reset-state streaming
  contract.
- Sampling draws use the crate-local MLX-compatible threefry RNG
  (`music::minimax_music3::rng`): the filtered distribution is pinned by
  fixtures, but a sampled (temperature > 0) Rust run draws different
  tokens than mlx at the same request. Temperature 0 is fully
  deterministic and never draws. Note the family applies the sampler to
  RAW logits (the reference `generate_frame` never log-softmaxes), so
  top-p filtering sees unnormalized values exactly like the reference
  (including its numeric fragility on huge logits); the fixture vectors
  are moderate.
- An immediate all-zero frame produces empty frames, empty codes, and
  an empty waveform with `StopReason::AllZeroFrame`; the reference
  yields no `GenerationResult` in that case.
- The reference `codes`/`audio` metadata token ids on the config
  (`audio_eos_token_id`, `audio_token_id`, `bos_token_id`,
  `codebook_eos_token_id`, `codebook_pad_token_id`, `pad_token_id`,
  `num_codebooks`) are never read at inference and are not parsed.

## Verification

Fixture protocol identical to the other audio families:
`tools/gen_sesame_fixtures.py` runs the true reference module (its
network tokenizer, Mimi, and watermarker singletons patched for tiny
seeded instances; greedy `make_sampler(0)` throughout), driven from
`../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_sesame_fixtures.py
```

Fixtures live in `testdata/tts_sesame/` with SHA-256 digests in
`manifest.json`:

- `decode/` (tiny 4-book Mimi, quantizer_bins 16, 24 kHz, weights f32,
  sha256 in the manifest): the reference encode codes for a seeded wave,
  the `_tokenize_audio` frame layouts with and without the eos frame,
  and the 63-frame two-chunk streaming decode waveform.
- `generate/` (tiny CSM: 2-layer backbone, 1-layer depth decoder, 4
  codebooks, audio vocab 16, hidden 32, head_dim 16, Llama3-scaled
  interleaved rope, weights f16): the prompt text ids and frame layout,
  four greedy frames, and the waveform from the true `Model.generate`
  (reference chunked decode included).
- `generate_eos/` (second seeded backbone with zeroed `codebook0_head`
  and `audio_head`): every sampler argmax is code 0, the first frame is
  all zeros, and both the reference and the port stop with no frames and
  no audio.
- `sampling/` (traces only): greedy argmax, top-k 50, top-p 0.8, and
  the top-p 0.9 then top-k 50 chain on fixed raw-logit vectors, with
  null sentinels for negative infinities.
- `flavors/` (traces only): the four `create_llama_model_args` presets
  the flavor config fallback must reproduce.

Measured parity when the gates were frozen: chunked streaming decode
2.7e-7 and end-to-end generate waveform 2.7e-7 max absolute difference
(gate 2e-6); generated frames, prompt frame layouts, encode codes, and
sampler stages exact. The reference chunk boundary (62 frames) is
exercised by the second decode chunk.

`cargo test -p turbospark-audio --lib tts::sesame` runs the family gate.

## Open items

- No real-checkpoint gate: a csm-1b end-to-end run (real Llama-3
  tokenizer, real Mimi, waveform sanity against the Python family) has
  not been executed; the fixtures are seeded tiny instances. The
  all-zero eos convention and the codebook-head mapping are learned
  behavior no tiny seeded model can exercise beyond structure.
- Watermarking stays refused; see the deviation above.
- Streaming intervals, batched prompts, and the voice prompt downloads
  are out of scope.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
