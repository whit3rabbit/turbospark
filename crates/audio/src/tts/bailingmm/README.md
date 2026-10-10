# Ming Omni TTS: bailingmm MoE and dense (spec tasks 42 and 41)

Port of `mlx_audio/tts/models/bailingmm/bailingmm.py` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/bailingmm),
covering the sibling `dense/` directory's `Model` (spec task 41) as a
config variant of this module per the MODELS.md alias policy: the dense
variant is the same implementation with `_is_moe_llm_config` forced to
false (`model_type "dense"` in the checkpoint config), so one family
module serves both. The reference `ming_omni_tts` architecture:

- an LLM backbone read for hidden states only: the bailing MoE (shared
  [`crate::lm`](../../lm/README.md) port) or the Ming Qwen2 dense
  variant (`ming_qwen2.rs`), whose attention adds per-layer sliding
  windows over a dense cache (`use_sliding_window` / `sliding_window` /
  `max_window_layers` / `layer_types`);
- the AudioVAE (`audio_vae.rs`): a Ming Qwen2 encoder over 64-wide frame
  windows (with the optional patch aggregator and learned CLS row) and
  a Ming Qwen2 decoder with a streaming ISTFT head (`np.fft.irfft`,
  periodic Hann window, window-squared envelope normalization, and the
  audio/window buffer chain);
- the DiT flow-matching head (`dit.rs`): timestep/condition embedding,
  DiT blocks with the local interleaved `RotaryEmbedding`, the CFM
  solver (EPSS timestep tables, sway adjustment, Euler steps plus
  `sigma * sqrt(temperature) * sqrt(|dt|)` noise), and the CFG guide
  `pred + (pred - null) * cfg` with a doubled batch and zero condition;
- the DiT aggregator (`linear_proj_audio`) projecting each latent patch
  (and the voice-clone prompt latent) back into the LLM stream through
  a learned CLS token;
- the two-class stop head; the loop stops when
  `softmax(stop_head(z))[1] > 0.5` after step 3.

There are NO generated vocabulary tokens: each step reads the LLM's last
hidden row, flow-samples one latent patch, decodes it through the cached
decoder one step behind (the streaming upsample emits the previous
patch's audio, and the final chunk trims the pad), and feeds the
aggregated patch embedding back in.

## Layout

- `mod.rs`: config (`ModelConfig.from_json` reading `llm_config`/`
  text_config`, `audio_tokenizer_config`, `ditar_config`,
  `aggregator_config` exactly like the reference `from_dict`), the MoE
  detection (all six MoE keys present; forced false for the dense
  variant), prompt assembly (`_prepare_input_embed` over pre-tokenized
  segments with the `<audio>` splice), and the generate loop.
- `ming_qwen2.rs`: the Ming Qwen2 dense backbone (biased q/k/v, rope at
  the cache offset, per-layer sliding windows over a dense cache).
- `dit.rs`: the flow stack and `FlowNoise` (recorded replays and the
  crate threefry `SeededNoise`).
- `audio_vae.rs`: encoder, decoder, ISTFT head, streaming upsample.

## Fixture protocol

`tools/gen_bailingmm_fixtures.py` runs from the mlx-audio checkout with
the project venv, deterministic seeds, no network: the reference modules
run in full (bailingmm and dense), the checkpoint tokenizer is replaced
by a deterministic offline mock (fixed `<audio>` / `<audioPatch>` /
`<|endoftext|>` ids plus word ids), and every `mx.random.normal` draw is
recorded in reference draw order (per generate step: the flow noise,
then one solver draw per integration step; the voice-clone case prepends
the encoder's `[batch, latent, frames]` draw). Each case runs
`Model.sample` (per-step latents and stop decisions), replays the
reference decode loop body (`audio.decode` with the cache and stream
state, instrumenting the ISTFT head) and asserts the chunks equal
`Model.generate`'s waveform.

Cases in `testdata/tts_bailingmm/`:

- `moe_generate/`: 3-layer bailing MoE (4 experts top-2, shared expert,
  q/k norms, untied head), stop head re-biased to fire at step 4.
- `moe_length/`: stop head silenced; 12 steps (max) with the one-behind
  chunk pipeline and the dropped final tail.
- `dense_generate/`: the dense variant (`model_type "dense"`, tied head)
  with a sliding window on the decoder's second layer.
- `voice_clone/`: MoE with a reference-audio prompt latent (one
  alignment unit of audio, 30 latent rows, 15 patch tokens), covering
  `_pad_prompt_waveform`, `encode_latent` with the recorded noise, the
  `<audioPatch>` spliced input embeddings, and the flush-right
  `latent_history` init.

Every case records the assembled prompt ids and input embeddings, the
flow timestep/sway grids, the per-step `(z, latent_history, latent)`
triples, the first two raw DiT guide calls (full-sequence output, per
block inputs and outputs, cropped guide output), the per-step decoded
chunks, the ISTFT head streaming calls (input rows, projection rows,
output samples), the recorded noise draws, and the joined waveform.

## Measured parity (frozen gates in tests.rs)

| Trace | Measured max abs diff | Gate |
|---|---|---|
| Prompt input embeddings (MoE, dense) | 0 (exact ids) | 4e-6 |
| Voice-clone input embeddings | 1.4e-6 | 4e-6 |
| Voice-clone encoder stats | 3.8e-6 | latent gate 2e-5 |
| Voice-clone latent | 4.1e-6 | 2e-5 |
| Per-step latents (flow replay) | 3.6e-6 | 2e-5 |
| DiT guide stages (assembly, blocks, full) | 7.6e-5 at scale 269 | 2e-5 x scale |
| Guided crop + CFG arithmetic | 5.5e-6 | 2e-5 |
| Joined waveform (MoE, dense) | 1.1e-5 | 2e-5 |
| Voice-clone waveform | 1.2e-4 | 2e-4 |

Generated ids, chunk lengths, stop reasons, and step counts are exact.

## Deviations from the reference

- Batch 1. The reference generate loop is single-batch; the CFG guide
  doubles the batch inside the solver exactly like the reference.
- Non-streaming audio assembly: the per-step chunk state (cache,
  upsample, ISTFT buffers) is implemented exactly as the reference, but
  the chunks are joined once at the end instead of being yielded as
  `GenerationResult` chunks; `streaming_interval` is out of scope.
- Caller-owned text tokenization: [`MarkerIds`] carries the tokenized
  marker strings and special ids; `music_style_text` mirrors the
  Genre/Mood/Instrument/Theme/Duration header-skip rule on the raw text.
- `temperature 0.0` (the generate default) collapses the solver's noise
  shift to exact zeros; the noise source is still consulted in the
  reference draw order so recorded replays stay aligned. Standalone
  seeded runs use the crate threefry RNG (`SeededNoise`), whose draw
  sequence is crate-local by design (see `tts::sampling`).
- The dense variant's LLM head (`lm_head` for an untied dense save) is
  not loaded: the TTS path reads hidden states only. The bailing MoE
  path keeps the shared loader's head handling (its `lm_head` loads
  through the `model.` prefix or the nested `model.lm_head.` spelling).
- `audio.encoder.cls_embed` is a plain random array in the reference
  (never a saved parameter, so a checkpoint save drops it and a fresh
  load would use unseeded noise). The loader accepts its absence and
  refuses the patch path without it; the fixture pins it as a zero
  parameter so the traces carry exactly the value the Python run used.
- The reference's `RotaryEmbedding.inv_freq` (flow DiT and aggregator)
  IS a saved module attribute, so it loads from the checkpoint; when a
  save dropped it, the loader falls back to the classic
  `base ** (-2i / dim)` table. The fixture generator seeds every module
  attribute (including these frequencies and the ISTFT window), so the
  fixtures pin the seeded garbage rather than the classic tables - the
  Rust loads and uses exactly what the checkpoint carries, matching the
  real-checkpoint behavior.
- `np.fft.irfft` ignores the imaginary parts of the DC and Nyquist bins;
  the shared real-IFFT plan gets those zeros explicitly so arbitrary
  (non-Hermitian) spectra agree.

## Open items

- Real-checkpoint gate: `mlx-community/Ming-omni-tts-16.8B-A3B-bf16`
  (MoE) and `mlx-community/Ming-omni-tts-0.5B-bf16` (dense) are too
  large / not locally installed; task 42's bar is fixture-level parity
  and the real-checkpoint gate stays open. The checkpoint key mapping
  the loader expects (`model.` prefix over the MoE names, top-level
  audio stacks) follows the reference sanitize; a real-checkpoint run
  should confirm it.
- The reference `speed` parameter is a documented no-op upstream and is
  not carried here.
- Streaming generation (`stream` / `streaming_interval` in the
  reference generate signature) is rejected upstream too (deleted
  arguments); only the non-streaming result exists.
