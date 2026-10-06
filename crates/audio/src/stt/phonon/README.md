# Phonon-1

This module ports `mlx_audio/stt/models/phonon/` (phonon.py, packed.py,
transport.py, config.py) from mlx-audio 0.5.7, source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

Phonon-1 is Fermion Research's English speech-to-text family. The
architecture is Qwen3-ASR (the same family as `crate::stt::qwen3_asr`,
derived from Qwen/Qwen3-ASR-0.6B): a 128-band Whisper log-mel frontend, a
chunked audio tower, a chat prompt, and greedy autoregressive decoding.
Phonon's learned decoder replaces every dense linear with a "packed trit"
five-value layer:

- each weight is one of five learned values, stored as an exact base-5
  symbol;
- ten symbols pack into 24 bits (`quint5_q` uint8 rows of 3-byte
  little-endian groups; `packed_bytes_per_row = ((in + 9) / 10) * 3`);
- the five values are the sums of two native 2-bit affine planes (base
  `-alpha/0/+alpha` per row plus residual `-r/0/+r` per linear), so the
  reference evaluates each linear as two MLX quantized matmuls summed at the
  output;
- slim distribution metadata (`broadcast-scales-v1`) stores one bf16 base
  scale per output row (`base_alpha`) and one bf16 residual scale per linear
  (`residual_scale`), with biases equal to the negated scales.

The released distribution is a SHA-256-checked byte-plane tar+zstd archive
(`phonon-audio6.bps.tar.zst`). Upstream materializes it to a local directory
(transport.py `prepare_model_path`) and then loads. The port consumes an
already-materialized directory and refuses the un-materialized archive with
instructions; the tar+zstd byte-plane transport is deliberately not
implemented in Rust.

The log-mel frontend, audio tower, text decoder, tokenizer loader, prompt
construction, and greedy decode are reused from `crate::stt::qwen3_asr`.
This module adds `config.rs` (detection plus strict manifest validation),
`packed.rs` (quint5 unpack and two-plane materialization), and
`materialize.rs` (one temporary f32 safetensors handed to the shared
loaders, the Parakeet Redux pattern, because the shared decoder keeps its
weights private).

## Archive materialization contract

`Phonon::load` accepts a directory that carries `packed_manifest.json`
(format prefix `sttg1a-`), `config.json` (`model_type qwen3_asr`), the
manifest's weight shard, and the tokenizer assets. It refuses, with
materialization instructions, a directory whose `config.json` has
`config_schema "fermion.phonon/1"` but no manifest: that is the
un-materialized release repository.

Materialize once with the reference stack (kept out of the Rust build on
purpose; it needs `zstandard`):

```sh
~/.venv-mlxaudio/bin/python -c "from pathlib import Path; from mlx_audio.stt.models.phonon.transport import prepare_model_path; print(prepare_model_path(Path('$HOME/models/phonon-1')))"
```

The pinned archive verifies at
SHA-256 `214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2`
and materializes to a single 449,914,173-byte
`model-00001.safetensors` (SHA-256
`b0f77bd7146f40e3dfe8fa60052723d7756a0d69b56b5c141e0a4c3a61a1ac00`, both
digests re-checked by the Rust loader against the manifest). Keep both the
archive directory and the materialized directory under `~/models/`.

## Pinned profile

Primary verified profile:

- Repository: `FermionResearch/Phonon-1`
- Revision: `0428da04625c51b6f069a9829c7060e6b167b92a`
- Profile constant: `PHONON_1`
- Input: mono f32 PCM at 16 kHz; output: English text (the released
  checkpoints are English-only; the reference config override forces
  `support_languages = ["English"]`)
- Manifest format `sttg1a-armc2-head8audio6-v1`, status PASS, 196 packed
  decoder modules (28 layers x 7 linears), 440,401,920 decoder parameters
- Backbone geometry: audio tower d_model 896, 18 layers, ffn 3584, 128 mel
  bands, n_window 50, n_window_infer 800, output_dim 1024, downsample
  hidden 480; decoder hidden 1024, 28 layers, intermediate 3072, 16 query
  heads, 8 KV heads, head_dim 128, vocab 151,936, tied embeddings

Quantization layout (all values re-checked against the tensors at load):

| Component | Scheme |
|---|---|
| 196 decoder linears | quint5 codes (`ten-base5-per-24bit-v1`, 2.4 logical bits per weight) unpacking to two 2-bit affine planes, groups of 128 |
| Decoder metadata | slim (`broadcast-scales-v1`): per-row bf16 `base_alpha`, one bf16 `residual_scale`, biases `-scale` |
| Tied token embedding | MLX affine 8-bit, groups of 64, tied output head |
| 111 audio tower linears | MLX affine 6-bit, groups of 128 (all linears biased except `audio_tower.conv_out`) |

Pinned repository files (8 files, 415,110,101 bytes total). Sizes in bytes;
SHA-256 digests computed from the installed snapshot at the pinned revision.

| File | Size | SHA-256 |
|---|---:|---|
| `.gitattributes` | 509 | `8fe5525fa4ac25737cc64d15a66afa22133779efffcd21d8fef099f14d9b579f` |
| `LICENSE` | 11358 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `NOTICE` | 2705 | `354163498be2ed26c0ed637afca33fdfe47e88321be08da2e54065ba874224d4` |
| `README.md` | 2840 | `4050027cee8e2c64d9f325137d01566605f45f190753de5b6ab81a560e93704f` |
| `config.json` | 2511 | `7d9547bdeb4d4aba98cdf89b0f439cf69a2213f2c0dd1e9e6ac6bed02307a62a` |
| `package_release_bps.py` | 10503 | `c6268fac27baa89df9c940ee870679ccb1cb61d2935805cad575071fd2d898ce` |
| `phonon-audio6.bps.tar.zst` | 415077202 | `214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2` |
| `verify_install.py` | 2473 | `4078323b38e9fc729364b2382df42a604913139b90bccc1321707bb2ba5d987e` |

The tokenizer ships as `vocab.json`, `merges.txt`, and
`tokenizer_config.json` (no `tokenizer.json`); the shared `qwen3_asr`
loader rebuilds the Qwen2 byte-level BPE and checks the audio token ids
against `config.json` (`audio_token_id` 151676, start 151669, end 151670).

## Inference contract

- `Phonon::load(model_dir)` validates `config.json` through the shared
  Qwen3 audio/text parsers, validates `packed_manifest.json` (status PASS,
  groups of 128 with 2-bit planes, exact 196-module coverage of the decoder
  linears, code and metadata formats, hybrid embedding shape and audio
  linear schemes), SHA-256 checks the single shard against the manifest,
  unpacks every quint5 module into its two 2-bit planes (the exact integer
  semantics of the reference Metal kernel), fuses base + residual into one
  f32 weight through the slim metadata, and hands the converted tensors to
  the shared loaders through a temporary f32 safetensors file that is
  removed once the weights are resident. Every checkpoint tensor must be
  classified by the manifest or be an expected decoder norm; anything else
  fails the load. Multi-shard distributions are refused (the pinned
  release is single-shard).
- `transcribe(samples)` runs greedy decoding with automatic language
  detection in the prompt; `transcribe_with_options(samples, language,
  max_tokens)` accepts `"English"`. The prompt, stop tokens, and
  `<asr_text>` transcript extraction match the reference exactly through
  `qwen3_asr::prompt_token_ids` and `transcript_from_tokens`.

### Deliberate divergences from the reference

1. **No Rust transport.** The tar+zstd byte-plane archive is not unpacked in
   Rust; an un-materialized directory is refused with a precise
   `prepare_model_path` instruction. Adding zstd/tar dependencies is out of
   scope.
2. **Fused weight.** The reference computes two bf16-scale quantized
   matmuls per decoder linear and sums the outputs; this port materializes
   `base + residual` into one f32 weight and runs one dense linear. The
   difference is f32 summation order of the same real numbers, covered by
   the staged decoder gates; the greedy decode is token-exact.
3. **Shard digest at load.** Upstream verifies the archive (and each member)
   during materialization only; the port re-verifies the materialized
   shard's SHA-256 against the manifest at every load.
4. **Language refusal.** The reference would silently prompt with an
   unknown language string; the port, like the sibling `qwen3_asr` and
   `mega_asr` ports, refuses languages outside the profile's list.
5. **Verified layout only.** The reference `PackedTritLinear` also supports
   explicit per-group scales/biases and pre-unpacked plane words; the port
   accepts only the slim quint5 layout the release ships
   (`broadcast-scales-v1` + `ten-base5-per-24bit-v1`) and refuses the other
   manifest variants until they are verified with evidence, per the
   crate rule to refuse quant schemes it has not verified.
6. **bf16 vs f32 decode.** The reference decoder computes in bf16 (the
   quantized embedding emits bf16 and both quantized matmuls carry bf16
   scales); the port runs the materialized f32 pipeline. Staged gates
   absorb the drift.

## Verification evidence

All fixture values come from
`crates/audio/scripts/generate_phonon_fixture.py` run with the pinned
mlx-audio 0.5.7 checkout (`e1b19b9054bf163f5d812221a54fcc346f1890e9`)
against the materialized pinned checkpoint and the shared smoke clip
`crates/audio/testdata/qwen3_forced_aligner_reference.wav` (float32 WAV,
44715 samples at 16 kHz). The fixture is
`crates/audio/testdata/phonon_reference.json` and embeds the verbatim
materialized config and packed manifest.

- **Python reference transcript (pinned checkpoint, greedy, language
  auto-detect):** `The quick brown fox jumps over the lazy dog.`
  36 audio rows, prompt of 51 ids, 13 generated tokens
  (`11528, 6364, 151704, 785, 3974, 13876, 38835, 34208, 916, 279, 15678,
  5562, 13`; 151704 is the literal `<asr_text>` token the transcript
  helper strips). Reference wall time in the pinned venv: load 1.69 s,
  generate 0.28 s (exploratory reading, not a benchmark row).
- **quint5 goldens (always-on, exact):** for three representative modules
  (`model.layers.0.self_attn.q_proj`, `model.layers.14.mlp.gate_proj`,
  `model.layers.27.mlp.down_proj`, covering both packed row widths 309 and
  924), the Rust unpack of the transport bytes reproduces the reference
  base/residual plane words exactly, and the slim-metadata fusion
  reproduces the reference effective weight spots exactly (bit-identical
  f32, 63 spot values across rows 0, mid, last and columns spanning the
  group boundary).
- **Frontend parity (always-on):** the 128-band log-mel features over the
  smoke clip match the fixture within 2.0e-4 absolute (gate 2.0e-4;
  measured maximum printed by the test).
- **Checkpoint stage parity (gated test):** input features within 1.669e-6
  absolute (gate 2.0e-4); audio tower embeddings within 1.509e-7 (gate
  2.0e-4). The decoder stages compare against a reference that computes in
  bf16 end to end while this port runs the materialized f32 pipeline, so
  the honest decoder gates are relative: the prefill last-hidden row
  differs by at most 3.939e-1 absolute, 6.203e-3 of the row scale (gate
  6.0e-2), and the reference top-8 logits all survive within this port's
  top-16 with values within 1.168e-1 absolute, 3.927e-3 relative (gate
  5.0e-2). The greedy decode is unaffected: prompt token ids, generated
  token ids, and the transcript are exact.
- Backend profile: CPU f32, release build,
  `cargo test --release -p turbospark-audio --lib stt::phonon -- --ignored
  --nocapture --test-threads=1` with `TURBOSPARK_PHONON_MODEL_DIR` set to
  the materialized directory. The gated test completed in 19 to 24 s wall
  across three runs (shard digest, quint5 unpack and fusion, temporary file
  write, shared loader load, and the full transcription of the smoke clip
  in one process); those are single exploratory readings, not benchmark
  rows.

## Remaining gates

- **Quality.** No WER or task evaluation has been run. Compilation and
  stage parity do not establish quality; the upstream accuracy claims are
  not reproduced or refuted here.
- **stream_generate.** The reference exposes a head-elision streaming
  generator; the port implements the equivalent batched greedy decode. The
  arithmetic is the same (the elision only skips projecting unused prompt
  positions), but no separate streaming surface is wired.
- **Multi-shard distributions.** Refused at load; the pinned release is
  single-shard. Phonon-1 Big and Micro are not pinned or verified. The
  non-slim and non-quint5 packed layouts are refused at manifest parse
  (see divergences).
- **Runtime, catalog, FFI, Swift.** No `SpeechFamily` entry, catalog row,
  install probe, `ts_stt_*` ABI dispatch, or Swift selection exists for
  this family yet.
- **Performance.** No benchmark rows; the CPU f32 path materializes the
  decoder to f32 (about 1.7 GB) and the full model to about 3.1 GB of f32
  weights plus a temporary file of the same size at load.

## Local checks

```sh
hf download FermionResearch/Phonon-1 \
  --revision 0428da04625c51b6f069a9829c7060e6b167b92a \
  --local-dir ~/models/phonon-1

# materialize (once, reference stack)
MLX_AUDIO_CACHE_DIR=$HOME/models/mlx-audio-cache \
  ~/.venv-mlxaudio/bin/python -c "from pathlib import Path; from mlx_audio.stt.models.phonon.transport import prepare_model_path; print(prepare_model_path(Path('$HOME/models/phonon-1')))"

cargo test -p turbospark-audio --lib stt::phonon

TURBOSPARK_PHONON_MODEL_DIR=$HOME/models/mlx-audio-cache/phonon/214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2 \
  cargo test --release -p turbospark-audio --lib stt::phonon -- \
  --ignored --nocapture --test-threads=1
```

The fixture generator (never part of tests):

```sh
~/.venv-mlxaudio/bin/python crates/audio/scripts/generate_phonon_fixture.py \
  --model-dir ~/models/mlx-audio-cache/phonon/214c3b45aa57257013811a53f99a905848466ad4ab21f2b2f8368f7ac79427b2 \
  --archive ~/models/phonon-1/phonon-audio6.bps.tar.zst \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/phonon_reference.json \
  --revision 0428da04625c51b6f069a9829c7060e6b167b92a
```
