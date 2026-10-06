# Granite Speech 1B

IBM Granite Speech ASR (with speech-translation prompting) ported from
`mlx_audio/stt/models/granite_speech/` and `mlx_audio/lm/models/granite.py`
at mlx-audio 0.5.7, commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

The audio path is a paired-mel frontend (centered 512-point STFT with a
400-sample periodic Hann window, power spectrum, 80-band f32 HTK mel
filterbank, global `log10` peak floor at max - 8 with `/4 + 1`
normalization, odd-frame drop, pair stacking to 160 dims) feeding a
16-layer Conformer encoder: depthwise/pointwise GLU convolutions with
BatchNorm, context-blocked multi-query attention over 200-frame blocks
with one shared relative-position embedding (`2 * 512 + 1` entries indexed
by clipped in-block distance), and midpoint output self-conditioning
(`out` to 348 logits, softmax, `out_mid` back to 1024). A BLIP-2 style
QFormer (2 layers, 16 heads, eps 1e-12 LayerNorms) compresses each window
of 15 encoder frames into 3 audio embeddings through a learned query, and
a linear reaches the 2048-dim text width.

The text backbone is the Granite llama variant: 40 layers, 16 query
heads over 4 KV heads, no q/k norms, no projection biases, and four
load-bearing config scalars: `embedding_multiplier` 12.0 (applied once to
the prompt embeddings including the injected audio embeddings),
`attention_multiplier` 0.0078125 (the softmax scale, replacing
`head_dim^-0.5`), `residual_multiplier` 0.22 (scales both branch outputs
per block), and `logits_scaling` 8.0 (divides the final logits). Decoding
is greedy with a per-layer KV cache, stopping at the tokenizer EOS id
(`<|end_of_text|>`, 100257) before emission. The pinned checkpoint ships a
193-byte `chat_template.jinja`; this port expands it statically as
`USER: {content}\n ASSISTANT:` (the template ignores
`add_generation_prompt`) and pins the expansion with tests. The default
ASR prompt is `can you transcribe the speech into a written format?`;
translation uses `Translate the speech to {Language}.` with the six
upstream language codes (en, fr, de, es, pt, ja); unknown codes pass
through verbatim like the reference.

The frontend differs from the crate's whisper frontend in three
load-bearing ways: the filterbank is HTK-scaled without Slaney area
normalization over a zero-padded 512-point transform (whisper uses a
400-point transform and Slaney normalization), the global peak is taken
over the whole spectrogram BEFORE any frame is dropped (whisper always
drops the final centered frame before its peak), and rows are stacked in
pairs to 160 dims (whisper keeps one 80/128-band row per frame).

## Reference source

- `granite_speech.py` (`Model`, `CTCEncoder`, `ConformerBlock`,
  `ConformerAttention`, `ConformerConvModule`, `DepthWiseConv1d`,
  `BatchNorm1d`, `EncoderProjector`, `QFormer*`, `_extract_features`,
  `_build_prompt`, `_build_inputs_embeds`, `generate`)
- `config.py` (`EncoderConfig`, `ProjectorConfig`, `TextConfig`,
  `ModelConfig`)
- `mlx_audio/lm/models/granite.py` (`Model`, `GraniteModel`,
  `TransformerBlock`, `Attention`, `MLP`)
- `mlx_audio/lm/generate.py` (`generate_step`, chunked prefill)
- `mlx_audio/dsp.py` (`hanning`, `stft`, `mel_filters` with the default
  `precise=False` f32 path)

The checkpoint stores torch-layout conv weights (`[out, in, kernel]`);
the reference sanitize transposes them to MLX layout at load. This port's
conv1d kernels already consume the torch layout, so weights load directly.

## Pinned profile

| Profile | Repository | Revision | Format |
|---|---|---|---|
| Granite Speech 1B | `ibm-granite/granite-4.0-1b-speech` | `bd87ab862416353633ea431fe49b1614003623c5` | BF16 safetensors (3 shards + index), tokenizer JSON |

Install:

```sh
hf download ibm-granite/granite-4.0-1b-speech \
  --revision bd87ab862416353633ea431fe49b1614003623c5 \
  --local-dir ~/models/granite-4.0-1b-speech
```

SHA-256 of every file in the pin (18 files, 4637933040 bytes total):

| File | Size | SHA-256 |
|---|---|---|
| .gitattributes | 1798 | `175131bcd4df86d78f25b574a9be978143bbc1cfa67e51dc1c06df7cf1619ca1` |
| README.md | 16325 | `29466fa9692e2feacbe6b688833299196ebcbbdb3eb7f901ee935739cba6120a` |
| added_tokens.json | 26 | `b4956ad3505e979f2f25cc2e2be2163d8e23730f854b6c5a14ed106f3dd90549` |
| chat_template.jinja | 193 | `e3219da52634178f68696475ea4518ade9d1b5c9a71b825f28312b73afc432c0` |
| config.json | 2324 | `28f340731fbd08cec8f89586682f73d9df4c23e6a7cfd7edc02ba12ae913261e` |
| merges.txt | 916646 | `b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502` |
| model-00001-of-00003.safetensors | 2143518808 | `2ed1c8a94a3ea0bebc3faa93490cc1b5543ed0d33e345c26f4b0ca878ad12f8e` |
| model-00002-of-00003.safetensors | 2143963456 | `cbc16c9712d1174fa77dcfba36b6bfa32961ec6c713d9494fce8d61b657809be` |
| model-00003-of-00003.safetensors | 339045512 | `1f9ea0edb2847633edcdf257c8d5d3cd8be751cab2cefb039452ec689fe7f6fc` |
| model.safetensors.index.json | 84396 | `dec67acdc06e74d87650a54a4b5fbb07bfcdb4ed9fd1b95edef8651f5d64599b` |
| multilingual_sample.wav | 1596240 | `91d243650809c1274141ec20ff23045315eaf27567694002ea3ef390048b7058` |
| preprocessor_config.json | 336 | `382a6f26300937721969b01909c96c2e519620b6781f7b45cb8b2e348a932d87` |
| processor_config.json | 80 | `13edc723022827cc3620ba2e2b4bcf52287b365d77e60a31227f0a4b7dea5f9c` |
| special_tokens_map.json | 579 | `c08676c49fd7969a3130f72be6d4bf34da66aa484a6e21dffe359893a1bd5f2e` |
| tokenizer.json | 7153607 | `43ca88fd0519c64ef93fa0a90cbc4e560fe485b5ba60348a86bc3c624f37918e` |
| tokenizer_config.json | 17882 | `de233c19cd9efa63738f7481f318fc2048060f751ac9fd7957b131c43226b403` |
| vocab.json | 1612704 | `8af71076de8b0b626eed0f4c984faf0a7c062479164b2a31308a948524d4f69c` |

The pin's real encoder geometry differs from the mlx-audio `config.py`
defaults and the loader reads config.json: 16 Conformer layers (default
10) and `output_dim` 348 (default 42). The checkpoint config declares
`text_config.dtype` float32 while the shards are bfloat16; the reference
computes the audio path and the LM in bfloat16, this port in f32 with
weights upcast at load. Quantized variants are refused: any
`quantization_config` or `.scales` tensor stops the load.

## Inference contract

- Input: mono 16 kHz PCM samples as `f32`, non-empty and finite.
- Frontend: `[rows, 160]` paired log-mel; the smoke clip gives 280 STFT
  frames, 140 encoder rows.
- Prompt: `USER: ` + one `<|audio|>` (id 100352) per audio embedding +
  prompt text + LF + ` ASSISTANT:`, tokenized piecewise. The pinned
  tokenizer (GPT2 BPE, no post-processor, no BOS) makes the piecewise
  encode equal the whole-string encode of the expanded template; the
  fixture pins both.
- Audio rows: `ceil(encoder_rows / window_size) * (window_size /
  downsample_rate)` = `ceil(rows / 15) * 3`.
- Decode: greedy argmax over scaled logits, bounded by `max_tokens`
  (tests use 256; the reference default is 4096), stopping at EOS 100257
  before emission; text decoded with `skip_special_tokens=True`.
- Streaming (`_stream_generate`) is not ported.

## Verification

Fixture regeneration (never part of tests; requires the local mlx-audio
clone at the pinned commit and the venv with jinja2):

```sh
/Users/whit3rabbit/Documents/GitHub/.venv-mlxaudio/bin/python \
  crates/audio/scripts/generate_granite_speech_fixture.py \
  --model-dir ~/models/granite-4.0-1b-speech \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/granite_speech_reference.json \
  --revision bd87ab862416353633ea431fe49b1614003623c5
```

Python reference on the smoke WAV with the pinned checkpoint: transcript
`the quick brown fox jumps over the lazy dog` (lowercase, no terminal
punctuation), matching the upstream mlx-audio smoke. 280 STFT frames, 140
encoder rows, 30 audio rows, 49 prompt tokens, 9 generated tokens, first
token 1820 (`the`), EOS 100257.

Always-on tests (`cargo test -p turbospark-audio --lib stt::granite_speech`)
pin the frontend against the fixture, the prompt structure, the chat
template expansion, the translation prompts, config refusals, and the
token-level fixture contract.

Checkpoint-gated test (release mode, single thread):

```sh
TURBOSPARK_GRANITE_SPEECH_MODEL_DIR=~/models/granite-4.0-1b-speech \
cargo test --release -p turbospark-audio --lib stt::granite_speech -- \
  --ignored --nocapture --test-threads=1
```

Evidence from that run (release, Apple Silicon CPU):

- Rust transcript: `the quick brown fox jumps over the lazy dog`, exact
  match with the Python reference and the upstream mlx-audio smoke; all 9
  greedy token ids match exactly (`1820, 4062, 14198, 39935, 35308, 927,
  279, 16053, 5679`), the 49 prompt ids match exactly, and the final
  decode decision is the same comfortable margin (Rust eos logit 22.608
  vs reference 22.500).
- Stage parity against the fixture (max abs diff, f32 port vs bf16
  reference): input_features 1.192e-6 (the frontend is f32 on both
  sides), encoder_output 6.509e-4, audio_embeddings 9.473e-5, prefill
  last hidden 2.215 (8.3e-3 relative on a 266-magnitude row), first-step
  top-8 logit max 1.674e-1 (0.6% relative), wall time about 16 s for
  load plus decode.

Two port-only divergences were required and are pinned by tests:

1. The fused `to_kv` projection splits into column halves
   (`k = kv[:, :1024]`, `v = kv[:, 1024:]`), not flat buffer halves; a
   flat split silently interleaves k and v from row 1 on.
2. The shipped `tokenizer.json` pre-tokenizer carries a newer pattern
   whose punctuation branch absorbs a trailing LF, so a raw tokenizers
   library encodes `format?\n` with `?` and LF merged (token 5380); the
   transformers reference repairs the pre-tokenizer to the canonical GPT2
   pattern at load and keeps them separate (`?` 30, LF 198). The port
   encodes the prompt text and the LF + ` ASSISTANT:` tail as separate
   pieces, which reproduces the reference ids with the un-repaired
   library.

## Remaining gates

- Streaming `_stream_generate` (the upstream non-blocking partial-results
  path) is not ported.
- Translation decoding is untested against the checkpoint (prompt builder
  is pinned; no reference run on a translated target yet).
- Broader task quality (WER on public sets), runtime integration,
  Metal offload, catalog/FFI/Swift surfaces, performance, and memory
  gates are all open. One-clip parity does not establish quality or
  product readiness.
