# rumik-oss 1 (spec task 36)

Port of `mlx_audio/tts/models/rumik_oss/rumik_oss.py` at mlx-audio
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/rumik_oss)
(mlx 0.32.3): Cohere's Tiny Aya Fire `cohere2` decoder (shared
[`crate::lm`](../../lm/README.md) port) extended with a stop predictor
head and flattened Mimi codec tokens. Text conditioning and audio
generation share one autoregressive sequence; for every 80 ms frame the
model emits `num_quantizers` (8) codebook tokens in codebook order, the
round-robin assembler regroups them, and the frozen Mimi codec (24 kHz)
turns the frames back into audio.

Prompt format (the reference tokenizer prepends `[BOS]` itself):

```text
<text>Ira: <description="happy, Hindi accent, steady pace"> namaste<audio>
```

## Layout

- `mod.rs`: config (`ModelConfig` over the cohere2 args plus the audio
  vocabulary fields), the stop predictor head
  (`LayerNorm -> Linear -> GELU -> Linear`, MLX `nn.GELU` exact erf,
  LayerNorm eps 1e-5), the resolved `AudioVocab` (first/last unit id with
  the reference `__post_init__` derivation, `</audio>` id, quantizer
  count), the round-robin assembler, the audio logits processor, the
  prompt builders, and the greedy-or-sampled generate loop.
- `tests.rs`: fixture parity against `testdata/tts_rumik_oss/`.

## Fixture protocol

`tools/gen_rumik_oss_fixtures.py` runs from the mlx-audio checkout with
the project venv (`cd ../mlx-audio && ../.venv-mlxaudio/bin/python
../turbospark/crates/audio/tools/gen_rumik_oss_fixtures.py`),
deterministic seeds, no network: the reference module imports with its
network `Mimi.from_pretrained` patched out, a deterministic offline mock
tokenizer (BOS plus one word id per space token) stands in for the
CohereTokenizer, and a seeded tiny 8-book Mimi (quantizer_bins 64,
24 kHz, 12.5 fps) is injected as `model._mimi`. Backbones are seeded
tiny cohere2 instances (4 layers, hidden 32, window 4, three sliding
layers plus one global layer, tied head); the stop-predictor readout is
re-biased per scenario (always fires / never fires).

Cases (all through the reference code):

- `decode/`: the round-robin assembler step states, completed frames,
  and level-major codes on crafted id sequences at the reference-shaped
  `num_quantizers 8` (in-order, quantizer-jump resync, non-unit break,
  end-token stop, incomplete tail), the frameless-input error message,
  the `__post_init__` derivation at 2048 x 8 (first 24 -> last 16407),
  the `build_prompt`/`resolve_voice` cases, stop-predictor probabilities
  on fixed hidden rows, and the Mimi waveform through the reference
  `_decode_frames`.
- `generate_forced/` and `generate_length/`: end-to-end greedy runs
  (temperature 0) driven by the reference `generate_step` with the
  family audio logits processor: the forced scenario re-biases the stop
  predictor so `</audio>` is forced at the first eligible step
  (`min_tokens 16` -> 16 unit tokens then the stop); the length scenario
  silences the predictor and stops at `max_tokens 24` with a dropped
  partial tail. The reference sliced audio head is cross-checked
  against the full-head processor path in the generator; ids agree.
- `sampling/traces.json`: the reference `make_audio_logits_processor`
  masks on fixed full-vocabulary logits for the three step regimes
  (below min tokens, stop fires, stop silent - each branch on a model
  whose stop head actually takes it), the log-softmax normalization, the
  greedy argmax, and the `make_sampler` top-p (0.9) / top-k (30)
  filters. No RNG draws are recorded.

## Measured parity (frozen gates in tests.rs)

| Trace | Measured max abs diff | Gate |
|---|---|---|
| Mimi decode waveform (`decode`) | 8.9e-8 | 2e-6 |
| Forced generate waveform | 7.5e-8 | 2e-6 |
| Length generate waveform | 1.2e-7 | 2e-6 |
| Stop-predictor probability | 1.2e-7 | 1e-6 |
| Processor masks, log-softmax, top-p/top-k | 9.5e-7 | 2e-6 |

Generated ids, frames, level-major codes, prompt strings, prompt ids
(mock tokenizer), assembler states, and stop reasons are exact.

## Deviations from the reference

- Batch 1, non-streaming: the whole segment decodes to one waveform
  after generation. The reference `MimiStreamingDecoder` chunked path
  and `streaming_interval` are out of scope (the streaming decoder emits
  the same frames per chunk; composition is the only difference).
- Caller-owned text tokenization: the portable crate does not embed the
  stock CohereTokenizer. `build_prompt` produces the exact prompt string
  (voice resolution, description injection, trimming included) and the
  caller tokenizes it, prepending BOS the way the reference tokenizer
  does; `generate_from_prompt` takes the ids.
- The audio-head slice (`audio_head`, `make_sliced_sampler`,
  `make_sliced_logits_processor`) is not ported: it is a memory
  optimization with bit-identical logits over the legal token set (the
  reference docstring records the bit-exactness; the fixture generator
  asserts sliced and full-head ids agree on both generate scenarios).
  This port computes the full tied head and restricts through the audio
  logits processor instead.
- Sampled (temperature > 0) draws use the crate-local threefry RNG via
  `tts::sampling`, so the draw sequence differs from
  `mx.random.state`; the filtered distribution (processor masks,
  log-softmax, top-p, top-k, argmax) is pinned by fixtures, matching the
  llama-family protocol. Greedy (temperature 0) is fully deterministic
  and fixture-verified end to end.
- The reference `generate` raises when no complete frame was produced;
  the port returns the same error by name. `max_tokens 0` yields no
  tokens (the reference generate_step never enters its loop).
- The checkpoint stop-predictor tensors load under both spellings the
  two savers produce: PyTorch-numbered `stop_predictor.{0,1,3}.*` (what
  the reference sanitize rewrites) and MLX-side
  `stop_predictor.layers.{0,1,3}.*`.

## Open items

- Real-checkpoint gate: `rumik-ai/rumik-oss-1` (3B, quantized variants)
  has not been run; fixtures are seeded tiny instances. The real
  checkpoint pulls the kyutai Mimi weights by repo id, which the
  portable crate resolves through the caller-supplied codec.
- Streaming decode intervals and voice-clone-style zeroprompt inputs are
  reference behaviors this port does not exercise (rumik-oss has no
  zeroprompt path in the reference; streaming is the reference
  `stream=True` flag).
