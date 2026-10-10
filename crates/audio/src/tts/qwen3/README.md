# Qwen3 SNAC TTS family (VyvoTTS-style)

Port of `mlx_audio/tts/models/qwen3/qwen3.py` at mlx-audio 0.5.7,
pinned to
[`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/qwen3)
(mlx 0.32.3). The family is a dense qwen3 backbone (shared
[crates/audio/src/lm](../../lm/README.md) port) that interleaves text
with discrete SNAC audio codes: after
`[START_OF_HUMAN] text [END_OF_TEXT END_OF_HUMAN]` it emits
`[START_OF_SPEECH]` plus flat 7-code groups (one SNAC L1 code, two L2
codes, four L3 codes per group, slots offset by 0/1/2/3/4/5/6 times
4096) above `AUDIO_TOKENS_START`, stops at the end-of-speech token, and
the group stream decodes through the 24 kHz SNAC codec. The reference
pairs the checkpoint with `mlx-community/snac_24khz` and loads its
tokenizer from the model directory in `post_load_hook`. This is the
same protocol as the llama (Orpheus-style) family
[../llama/README.md](../llama/README.md) with qwen3 constants and the
qwen3 backbone; only family-specific details are repeated here.

Everything is portable single-batch f32: no Metal, no macOS APIs.

## Layout

- `mod.rs`: config (backbone args plus `tokenizer_name` and
  `sample_rate`), special-token constants, `codes_to_layers` (in the
  qwen3 reference the split lives inline in `decode_audio_from_codes`),
  `decode_audio_from_codes`, `encode_audio_to_codes`,
  `prepare_input_ids`, `prepare_zeroprompt`, `parse_output`, the
  generate loop (`generate`, `generate_from_prompt`,
  `generate_tokens_from_prompt`), and `GenerateParams` with the
  reference defaults (temperature 0.6, top-p 0.8, top-k off, repetition
  penalty 1.3 over a 20-token context, max_tokens 1200).
- `tests.rs`: fixture parity and regressions.
- Shared sampler math (logit bias, sign-aware repetition penalty,
  log-softmax, top-p, top-k, categorical draw) lives in
  [../sampling.rs](../sampling.rs) and is shared with the llama family.

## Special tokens

| Token | Id | Role |
|---|---|---|
| TOKENIZER_LENGTH | 151669 | anchor for the added tokens |
| START_OF_TEXT | 151643 | used by the zeroprompt transcript |
| END_OF_TEXT | 151645 | ends the prompt text |
| START_OF_SPEECH | 151670 | `parse_output` crops after its last occurrence |
| END_OF_SPEECH | 151671 | stops the generate loop, removed by parse |
| START_OF_HUMAN | 151672 | opens the prompt |
| END_OF_HUMAN | 151673 | closes the prompt |
| START_OF_AI | 151674 | zeroprompt wrapper |
| END_OF_AI | 151675 | zeroprompt wrapper |
| PAD_TOKEN | 151676 | batch padding in the reference |
| AUDIO_TOKENS_START | 151679 | flat code c rides token `151679 + c` |

## Scope decisions and deviations from the reference

Identical to the llama family (see that README): batch size 1,
non-streaming single decode per segment, caller-owned text
tokenization (the reference's `post_load_hook` tokenizer load is a
serving-layer concern), zero-noise codec decode by default with a
caller-supplied noise source on `decode_audio_from_codes`, crate-local
RNG for sampled draws (the filtered distribution is pinned; sampled
runs differ from mlx, temperature 0 never draws), no min-p/XTC, and a
named refusal where the reference `(len + 1) / 7` group count would
index out of bounds. The reference's Python decode wraps negative layer
codes silently through MLX negative indexing; the Rust codec refuses.

## Verification

Fixture protocol identical to the llama family:
`tools/gen_qwen3_fixtures.py` runs the true reference module (its
network SNAC singleton patched out and replaced with a tiny seeded
noise-free codec), driven from `../mlx-audio` at the pinned commit:

```sh
cd ../mlx-audio
../.venv-mlxaudio/bin/python \
    ../turbospark/crates/audio/tools/gen_qwen3_fixtures.py
```

Fixtures live in `testdata/tts_qwen3/` with SHA-256 digests in
`manifest.json`. The tiny codec under `decode/` is byte-identical to
the llama family's (sha256 4b5c8dfb09853b1c...). The backbones are
seeded tiny qwen3 models (2 layers, hidden 32, head_dim 8, q/k RMSNorm,
vocab 151720, weights F16, generate sha256 e7cf033f1bf3ab6a...) with
the same two greedy scenarios at penalty 5.0 (end-of-speech stop after
24 tokens, max_tokens stop after 10), the second end-of-speech-first
backbone over the slot-valid prompt codes (generate_eos, sha256
a42e644810affdd0...), and the same four fixed-logit sampler traces
(sha256 231337edd7b6298d..., byte-identical to the llama family's).

Measured parity when the gates were frozen: waveform decode 1.1e-9 max
absolute difference (gate 1e-7); sampler stages bit-exact except the
log-softmax reduction at 9.6e-7 (gate 2e-6); generated token ids,
parsed codes, layer splits, encode codes, and zeroprompt ids exact. The
backbone itself is pinned separately by the `lm` fixtures at 2e-4.

`cargo test -p turbospark-audio --lib tts::qwen3` runs the family gate.

## Open items

- No real-checkpoint gate: a VyvoTTS Qwen3 checkpoint end-to-end run
  (real tokenizer from the model directory, 24 kHz SNAC with noise,
  waveform sanity against the Python family) has not been executed; the
  fixtures are seeded tiny instances.
- Streaming intervals, batched prompts, and the batch padding path are
  out of scope.
- Sampled (temperature > 0) generation is not bit-reproducible against
  mlx by design; see the sampler deviation above.
- Voice cloning (`prepare_zeroprompt`) is a parity-only layout: the
  reference itself warns cloning does not work reliably on this model.
