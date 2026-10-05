# Mega-ASR

This module ports `mlx_audio/stt/models/mega_asr/` (mega_asr.py, router.py,
config.py, convert_lora.py, lora.py) from mlx-audio 0.5.7, source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

Mega-ASR is a robustness layer over the Qwen3-ASR-1.7B backbone (the same
family as `crate::stt::qwen3_asr`). A small audio-quality router classifies
each clip as clean or degraded:

- clean audio runs the unmodified Qwen3-ASR base path,
- degraded audio runs a LoRA-adapted robust path (dense deltas merged into the
  base linears at load, the reference `apply_deltas` semantics).

The port reuses the `qwen3_asr` audio tower, text decoder, 128-band log-mel
frontend, tokenizer loader, prompt construction, and greedy decode helpers
rather than duplicating them; this module adds the router
(`router.rs`), the LoRA adapter table with both upstream adapter formats
(`lora.rs`), and the family profile rules (`config.rs`).

Two distribution profiles exist:

- **Dynamic** (`model_type "mega_asr"`): the unmerged base plus
  `extras/router.safetensors` and `extras/lora.safetensors`. The router runs
  per clip; `use_lora` is the softmax argmax rule (degraded probability >=
  0.5).
- **Pre-merged robust** (pinned profile): the robustness LoRA was folded into
  the Qwen3-ASR weights and the result quantized, so the distribution is a
  plain `qwen3_asr` config plus `merge_metadata.json` (the profile
  discriminator: without that file, a `qwen3_asr` checkpoint is refused and
  belongs to the `qwen3_asr` family). There is no router and no LoRA file;
  every clip runs the robust path.

## Pinned profiles

Primary verified profile (always-on robust):

- Repository: `mlx-community/Mega-ASR-8bit`
- Revision: `b9c3c7020f94944205df7f7b5d5d1ce96678d74f`
- Profile constant: `MEGA_ASR_8BIT`
- Input: mono f32 PCM at 16 kHz
- Backbone: Qwen3-ASR-1.7B (24-layer audio tower, d_model 1024, 128 mel
  bands; 28-layer decoder, hidden 2048, 16 query heads, 8 KV heads, head_dim
  128, tied embeddings), affine 8-bit quantization in groups of 64

The pinned repo provides `vocab.json`, `merges.txt`, and
`tokenizer_config.json` but no `tokenizer.json`; the shared `qwen3_asr`
loader reconstructs the Qwen2 byte-level BPE and checks the audio token ids
against `config.json` (`audio_token_id` 151676, start 151669, end 151670).

Pinned artifact sizes are bytes; SHA-256 digests were computed from the
installed snapshot at the pinned revision.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 2463307541 | `55dbe2054579a134b56d83ff2e271618774c0e6569495a8a0bd61a68ed1ea91f` |
| `model.safetensors.index.json` | 78968 | `0a5d0ec11188602242ff81a9969883d0fdeb98cd5d85cd1413089d897c201af5` |
| `config.json` | 2394 | `606eb5494d978be3e65a9b538b859e6ba268b1a9712d8cea84293e60d2be3dfc` |
| `merge_metadata.json` | 219 | `0ec67ed8e805613b4b613247989d8551f7450c25dd15366fb33b36d573c3f55f` |
| `preprocessor_config.json` | 330 | `45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d` |
| `tokenizer_config.json` | 12487 | `4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c` |
| `vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `chat_template.json` | 1161 | `75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff` |
| `generation_config.json` | 128 | `167fcd0d46020ae6245f3177c47aa1bd419302866933785e1d4d4a07ccc50395` |
| `README.md` | 1948 | `08208bb582a574b31979687eb36d7b115db893feb1947a1f0d99013b26df40f8` |

Router and LoRA fixture source (the pinned 8-bit repository omits the
`extras/` directory by construction; the router is backbone independent):

- Repository: `mlx-community/Mega-ASR-bf16`
- Revision: `58b2aa4ed62343871f6dba78a4f71db015406152`
- `extras/router.safetensors` | 4691980 | `7a27e5d6cd23e58fa928ae37e8dc6f0ee3d1e9f1e4b15c4b85dbd5ea1c961c39`
- `extras/lora.safetensors` | 92504400 | `366a27472d5f98284115c623cf68620eb0c37d826ec175c498e7145148b1342b`
- Installed locally at `~/models/mega-asr-bf16-extras/` for fixture
  regeneration only; the runtime never reads them for the pinned profile.

## Inference contract

- `MegaAsr::load(model_dir)` detects the profile, validates the backbone
  config through the shared `Qwen3Config` parser (including the quantization
  scheme), loads the audio tower and decoder, reconstructs the tokenizer, and
  for the dynamic profile loads the trained router (refusing to pretend to
  route with an untrained one) and the LoRA factors when present.
- `route(samples)` returns the raw two-class logits, the degraded softmax
  probability, and the `use_lora` decision (argmax).
- `transcribe_with_options(samples, language, max_tokens)` mirrors the
  reference prompt exactly (empty system content, `<|audio_start|>` padded
  with `audio_rows` `<|audio_pad|>` tokens, `<|audio_end|>`, greedy decode,
  `<|im_end|>`/`<|endoftext|>` stops) through the shared
  `qwen3_asr::prompt_token_ids`. When the router routes degraded and no LoRA
  table exists, both reference decisions decode identically (its
  `apply_deltas` over an empty table is a no-op), so the port runs the base
  path. When a real LoRA table exists and the router routes degraded, the
  port refuses: see "Remaining gates".
- Hotwords/system prompts are not exposed, matching the `qwen3_asr` port.

### Deliberate divergences from the reference

1. **Untrained-router fallback.** With a dynamic-type config but no router
   file, the reference silently routes with a freshly initialized router; the
   port refuses to load such a distribution. The decision is meaningless
   noise and the decode is identical either way (empty delta table).
2. **Waveform decoding.** `mlx_audio.audio_io` decodes WAV input through
   miniaudio, which quantizes 32-bit float WAV input to 16 bits; the crate
   reader consumes the true float32 samples. The fixture generator reads the
   float32 data chunk directly so both sides of every parity gate see
   bit-identical inputs, and the transcript was re-verified through the stock
   reference audio path.
3. **`layers.` module alias.** The converted `lora.safetensors` lists the
   decoder modules twice, as `model.layers.N.*` and as `layers.N.*`; the
   reference resolves both to the same decoder linear (its `layers` property
   aliases `model.layers`) and sums both deltas. The port keeps the inventory
   verbatim and merges per module name, so a runtime caller reproduces the
   reference by merging every entry that resolves to a given linear.

## Verification evidence

All fixture values come from
`crates/audio/scripts/generate_mega_asr_fixture.py` run with the pinned
mlx-audio 0.5.7 checkout against the installed checkpoints and the shared
smoke clip `crates/audio/testdata/qwen3_forced_aligner_reference.wav`
(float32 WAV, 44715 samples at 16 kHz). The fixture is
`crates/audio/testdata/mega_asr_reference.json`.

- **Python reference transcript (pinned checkpoint, greedy):**
  `The quick brown fox jumps over the lazy dog.`
  36 audio rows, 13 generated tokens.
- **Rust checkpoint transcript:** identical, byte for byte, including the
  generated token id sequence and prompt token ids (gated test
  `pinned_checkpoint_matches_reference_stages_and_transcript`).
- **Router parity (always-on fixture test):** the full router forward runs in
  Rust from the fixture's embedded weights against two inputs:
  `smoke_clean` routes base (degraded probability 0.043880) and
  `smoke_degraded` routes LoRA (0.928008), matching the reference decisions.
  Gates: log-mel, conv frontend, transformer, and pooled spots within
  2.0e-4 absolute; logits within 2.0e-4; probabilities within 5.0e-5.
  Measured maximums: clean stage 2.027e-5, degraded stage 3.994e-6; logit
  diffs 1.192e-7 and 4.768e-7; probability diffs 3.7e-9 and 0.
- **LoRA parity (always-on fixture test):** the fixture pins the full 539
  module inventory (539 A/B factor pairs, scaling 1.0) plus three selected
  modules with embedded factors; Rust recomputes `scaling * (B @ A)` and
  matches spot values within 2.0e-4 and max/sum magnitudes within 1e-3
  absolute and 2.0e-3 relative. Measured: spot maximums 1.455e-11 to
  2.910e-11; delta max abs 0.001280 to 0.001652; delta sum abs 584.2 to
  981.2 across the three modules.
- **Checkpoint stage parity (gated test):** 128-band input features match the
  reference within 1.669e-6 absolute (gate 2.0e-4); audio tower embeddings
  within 1.276e-7 (gate 2.0e-4). The decoder stages compare against a
  reference that computes in bf16 end to end (bf16 scales, bf16 activations
  between ops) while this port runs the dequantized f32 pipeline, so the
  honest decoder gates are relative: the prefill last-hidden row differs by
  at most 2.965 absolute, 3.953e-2 of the row scale (gate 6.0e-2), and the
  reference top-8 logits all survive within this port's top-16 with values
  within 4.652e-1 absolute, 1.241e-2 relative (gate 5.0e-2). The greedy
  decode is unaffected: generated token ids and the transcript are exact.
- Backend profile: CPU f32, release build, `cargo test -p turbospark-audio
  --release` with `TURBOSPARK_MEGA_ASR_MODEL_DIR` set. The gated test
  completed in 42.2 s wall for load, dequantize, and the full transcription
  of the smoke clip in one process; that is a single exploratory reading,
  not a benchmark row.

Baseline for the crate at the start of this port: 223 passed, 21 ignored.
The mega_asr tests add 11 always-on tests (router forward, LoRA deltas and
loader round trip, config parsing and refusals) and 1 ignored gated test.

## Remaining gates

- **Quality.** No WER or robustness evaluation has been run. The upstream
  claim (large WER gains on noisy speech, clean speech unchanged) is not
  reproduced or refuted here; compilation and parity do not establish
  quality.
- **Dense-base dynamic runtime path.** The router and both LoRA loader
  formats are ported and fixture-verified, but a degraded-route decode with
  real deltas needs a dense (bf16) base distribution (`mlx-community/
  Mega-ASR-bf16`, about 3.4 GB); the shared `qwen3_asr` decoder keeps its
  weights private, so merging deltas into a live backbone requires either a
  materialized dense weight file at load or a wiring change in the shared
  module. Until then, `transcribe` refuses that combination with a precise
  error; the pinned pre-merged profile never reaches it.
- **Runtime, catalog, FFI, Swift.** No `SpeechFamily` entry, catalog row,
  install probe, `ts_stt_*` ABI dispatch, or Swift selection exists for this
  family yet.
- **Performance.** No benchmark rows; the CPU f32 path for the 1.7B backbone
  is expected to be slow (see the Qwen3-ASR notes for the 0.6B profile's
  phase costs).

## Local checks

```sh
hf download mlx-community/Mega-ASR-8bit \
  --revision b9c3c7020f94944205df7f7b5d5d1ce96678d74f \
  --local-dir ~/models/mega-asr-8bit

cargo test -p turbospark-audio --lib stt::mega_asr

TURBOSPARK_MEGA_ASR_MODEL_DIR=~/models/mega-asr-8bit \
  cargo test --release -p turbospark-audio --lib stt::mega_asr -- \
  --ignored --nocapture
```

The fixture generator (never part of tests):

```sh
~/.venv-mlxaudio/bin/python crates/audio/scripts/generate_mega_asr_fixture.py \
  --model-dir ~/models/mega-asr-8bit \
  --router-weights ~/models/mega-asr-bf16-extras/router.safetensors \
  --lora-weights ~/models/mega-asr-bf16-extras/lora.safetensors \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/mega_asr_reference.json \
  --revision b9c3c7020f94944205df7f7b5d5d1ce96678d74f \
  --extras-revision 58b2aa4ed62343871f6dba78a4f71db015406152
```
