# MOSS-Transcribe-Diarize

Timestamped speech-to-text with speaker labels. The model transcribes audio
into segments marked `[start][Sxx] spoken content [end]`, for example:

```
[0.06][S01] The quick brown fox jumps over the lazy dog.[1.82]
```

## Overview

The architecture combines three parts:

- `MossWhisperEncoder` (`encoder.rs`): the HF whisper encoder layout with two
  GELU convolutions (stride 1 then stride 2, kernel 3, padding 1), learned
  position embeddings added on the time axis, 24 pre-norm attention/FFN
  blocks reusing the glmasr `WhisperEncoderLayer` math (bias-carrying
  projections, no RoPE), and a final layer norm. 80-band input, d_model 1024,
  16 heads, FFN 4096, 1500 source positions.
- `VqAdaptor` (`adaptor.rs`): Linear(4096, 1024) -> SiLU ->
  Linear(1024, 1024) -> LayerNorm. Each adaptor row concatenates
  `audio_merge_size = 4` consecutive encoder frames, so one audio token
  covers 160 (hop) x 2 (encoder stride) x 4 (merge) = 1280 samples, about
  12.5 tokens per second.
- MOSS backbone: the shared `qwen3_asr` decoder (Qwen3 0.6B geometry, 28
  layers, tied embeddings) loaded from the same safetensors shard through the
  `model.language_model.` key prefix. The backbone is affine 4-bit group-64
  quantized; the encoder and adaptor stay unquantized and their loader
  refuses any `.scales` tensor.

The generate path (`mod.rs`) pads the clip to one 30-second window, computes
the 80-band log-mel frontend from `crate::whisper`, encodes, time-merges and
adapts, splices the audio embeddings into the rendered chat-template prompt,
prefills once through the shared decoder, and decodes greedily until a Qwen
EOS id (`<|endoftext|>` 151643, `<|im_end|>` 151645). `parse_segments` ports
the reference `TRANSCRIPT_SEGMENT_RE`: segments with `end < start` or empty
text are dropped, and a transcript with no surviving segment falls back to
one unsegmented entry over the clip duration.

Time markers: with `enable_time_marker` on (the pinned default), the audio
placeholder span injects decimal marker tokens every
`time_marker_every_seconds = 5` seconds of audio, counting 62 placeholders
per marker interval (`int(12.5 x 5)`). The port reproduces the reference
arithmetic exactly, including its quirk that only placeholders count as
consumed, so a span can end with trailing placeholders after a marker.

## Upstream reference

- `mlx_audio/stt/models/moss_transcribe_diarize/moss_transcribe_diarize.py`
- `mlx_audio/stt/models/moss_transcribe_diarize/config.py`
- Dependencies reused by the reference:
  `mlx_audio/stt/models/glmasr/glmasr.py` (`WhisperEncoderLayer`),
  `mlx_audio/stt/models/qwen3_asr/qwen3_asr.py` (`TextModel`).
- mlx-audio 0.5.7, commit
  [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio).
  The Rust module header of every file names its reference file.

## Pinned checkpoint profile

Repository
[`vanch007/mlx-MOSS-Transcribe-Diarize-4bit`](https://huggingface.co/vanch007/mlx-MOSS-Transcribe-Diarize-4bit)
at revision `d42a296ee807e933ddd7588e2041dbbc84aff85d`. 15 files,
976,318,991 bytes total. Install locally; tests never download:

```sh
hf download vanch007/mlx-MOSS-Transcribe-Diarize-4bit \
  --revision d42a296ee807e933ddd7588e2041dbbc84aff85d \
  --local-dir ~/models/moss-transcribe-diarize-4bit
```

| File | Bytes | SHA-256 |
|---|---|---|
| model.safetensors | 960,434,705 | `0483be31b7adaa81ff6f94da3eb4c61a993e29b2fe2c3ba07827fbd00494a3c4` |
| tokenizer.json | 11,423,222 | `bcf03774334462d6e34b5005cb11120a62275f146ee2953e68731ecdbce84fbb` |
| vocab.json | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| merges.txt | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| chat_template.jinja | 4,762 | `8641466a16b184ebaf7c4903391e607cfd532ab937e81a64120628ab79d827f4` |
| config.json | 2,543 | `b96fbfcc5e921d0620a11cf84b03e545c9a72c35691664fc77f2533f8c6a1a5b` |
| added_tokens.json | 707 | `c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680` |
| special_tokens_map.json | 613 | `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` |
| tokenizer_config.json | 474 | `e1be11414f6b66d01085411aead0704662fd9515ed30d7c509e1474755c76144` |
| README.md | 656 | `9fd76c449895eb8c625c5999295beab3225084fd773cdb733146177e03126326` |
| mlx_conversion.json | 339 | `a4a1e2d7234a92172b1275edf78d21521e85161d15d03d51cd1c36907584fe3b` |
| preprocessor_config.json | 315 | `ba2e601484abc80f4cded977f9a4fd4a53175b7d35c2f2511f0cfc3a32ad2499` |
| processor_config.json | 292 | `a978c2dd54a65b576c3dae4b654fe9bcbac1184c6db2df0afb2c90fcdc872ae7` |
| generation_config.json | 107 | `e53a4b3ce4f944230cf1ca8fed0c42f4ff0d8c1443eaf98b5315d987334dd9e4` |
| .gitattributes | 1,570 | `34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930` |

Checkpoint facts verified at load: `quantization` is affine 4-bit group-64
with scope `text_backbone_only` and `excluded_prefixes`
`["model.whisper_encoder", "model.vq_adaptor"]`; `audio_token_id` 151671
matches the tokenizer's `<|audio_pad|>`; digits 0-9 encode to single token
ids 15-24; `chat_template.jinja` matches the pinned digest (the prompt
contract comes from rendering this exact template). Any other scheme,
geometry, or template is refused rather than mis-decoded.

## Inference contract

1. Audio: mono f32 at 16 kHz, at most one 30-second window (480,000 samples;
   the reference multi-chunk path is not part of this port). Zero-padded to
   exactly one window, then the shared 80-band whisper log-mel frontend
   (Slaney filterbank, log10 with 1e-10 floor, global peak minus 8 clamp,
   `(x + 4) / 4`, final centered frame dropped).
2. Audio token length: `(num_samples - 1) / 1280 + 1`; the encoder output is
   sliced to `4 x` that many frames before the time merge.
3. Prompt: the pinned chat template renders
   `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n{DEFAULT_PROMPT}<|im_end|>\n<|im_start|>assistant\n`;
   the placeholder is replaced by the audio span ids (placeholders plus
   decimal time markers), split and encoded with the checkpoint tokenizer.
4. Decode: tied-embedding logits, greedy argmax, stop on 151643/151645.
5. Output: text plus `parse_segments` results (`start`, `end`,
   `"[Sxx] text"`, `speaker_id`).

## Verification evidence

Fixture: `crates/audio/testdata/moss_transcribe_diarize_reference.json`
(schema `turbospark.moss_transcribe_diarize.reference/1`), generated by
`crates/audio/scripts/generate_moss_transcribe_fixture.py` on the shared
smoke clip `crates/audio/testdata/qwen3_forced_aligner_reference.wav`
(44,715 samples, SHA-256 `ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb`)
through the reference stack (mlx-audio 0.5.7 at the commit above, pinned
revision). Regenerate with:

```sh
cd ../mlx-audio && ../.venv-mlxaudio/bin/python \
  ../turbospark/crates/audio/scripts/generate_moss_transcribe_fixture.py \
  --model-dir ~/models/moss-transcribe-diarize-4bit \
  --audio ../turbospark/crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output ../turbospark/crates/audio/testdata/moss_transcribe_diarize_reference.json \
  --revision d42a296ee807e933ddd7588e2041dbbc84aff85d
```

Python reference transcript (venv mlx-audio, `Model.generate` on the raw
float32 samples):

```
[0.06][S01] The quick brown fox jumps over the lazy dog.[1.82]
```

Parsed segment: start 0.06, end 1.82, speaker `S01`. 112 prompt tokens (35
audio rows), 25 generated tokens.

Rust release checkpoint run (`cargo test --release -p turbospark-audio --lib
stt::moss_transcribe_diarize -- --ignored --nocapture
--test-threads=1` with `TURBOSPARK_MOSS_TRANSCRIBE_MODEL_DIR` set to the
pinned snapshot) reproduces the exact contract:

- prompt token ids: 112/112 exact match
- greedy token ids: 25/25 exact match
- transcript: `[0.06][S01] The quick brown fox jumps over the lazy dog.[1.82]`
  exact match, and the parsed segment equals the fixture segment
- digit token ids 15-24 exact match

Numeric stage witnesses (reference stores encoder, adaptor, and decoder in
bfloat16 and runs quantized backbone matmuls; this port computes f32, so
gates are relative while tokens stay exact):

| Stage | Max abs diff | Gate |
|---|---|---|
| input_features (log-mel) | 8.98e-4 | 3.0e-3 absolute (bf16 rounding bound) |
| encoder_output | 4.29e-2 | 6e-2 of row max (measured 3.1%) |
| audio_embeddings (adaptor) | 2.19e-2 | 6e-2 of row max (measured 1.1%) |
| prefill last hidden | 8.14e-1 | 6e-2 of row max (measured 1.12%) |
| first-step logits top-8 | 2.55e-1 | 5e-2 relative; top-8 ids survive in top-16 |

Always-on tests (no checkpoint) cover the frontend witness, the rendered
prompt contract against the fixture, the audio placeholder span structure,
the segment parser (fixture transcript plus inverted-range, empty-text, and
fallback cases), and the time-marker arithmetic. Config tests pin the
profile parse and refusals (model type, quant bits/scope/exclusions, mel
bins, activation, merge size, tie, sample rate, adaptor input dim, token id,
processor values).

Gates run on 2026-10-05 (macOS, arm64, CPU f32):

- `cargo test -p turbospark-audio --lib stt`: 114 passed, 15 ignored
  (baseline after phonon: 101 passed, 14 ignored; deltas are this family).
- `cargo fmt -p turbospark-audio` applied to this family's files and
  `rustfmt --check` clean on all of them.
- `cargo clippy -p turbospark-audio --tests`: no diagnostics mentioning
  `moss_transcribe`.

## Remaining gates

- Task quality beyond the smoke clip, multi-speaker diarization quality,
  long-audio chunking (the reference chunks every 30 s and concatenates
  adaptor features; this port refuses inputs over one window), runtime
  integration, catalog probes, FFI, and Swift surfaces are not started.
- `cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio` could
  not run on this machine: `onig_sys` (a transitive dependency of the
  `tokenizers` crate that predates this family) needs a linux cross-gcc
  which is absent. The port uses portable std + crate-internal APIs only.
- Metal offload, streaming, and the wider checkpoint zoo (other bit widths,
  untied heads, other merge sizes) are unverified and refused at load.
