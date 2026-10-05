# Qwen3-ASR

This module ports `mlx_audio/stt/models/qwen3_asr/` from mlx-audio 0.5.7,
source commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Pinned profile

- Repository: `mlx-community/Qwen3-ASR-0.6B-8bit`
- Revision: `89e96d92ba34aca20b3e29fb10cc284097d1219f`
- Profile constant: `QWEN3_ASR_06B_8BIT`
- Input: mono f32 PCM at 16 kHz

The pinned repo provides `vocab.json`, `merges.txt`, and
`tokenizer_config.json`, but no `tokenizer.json`. The Rust loader reconstructs
the Qwen2 byte-level BPE from those same-revision files and checks the audio
token ids against `config.json`.

Pinned artifact sizes are bytes. SHA-256 values come from the immutable Hub
revision, with the large model weight digest verified from its LFS metadata.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 1006229426 | `b5bfe4abc1b4c6e58b633096682ec2b6297298add1527119936107d211adf0e8` |
| `config.json` | 7187 | `5d104a945fed08728ab010f12bf3ce5ab4d0794bba276d81bff5bd83ae9d2be0` |
| `preprocessor_config.json` | 330 | `45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d` |
| `tokenizer_config.json` | 12487 | `4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c` |
| `vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `chat_template.json` | 1161 | `75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff` |
| `generation_config.json` | 142 | `1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c` |

## Implementation and evidence

The port includes the 128-band Whisper feature frontend, three-layer audio
convolution stack, Qwen audio transformer, affine 8-bit text decoder, and
greedy transcription path. `qwen3_forced_aligner` uses the audio encoder as a
shared dependency.

The frontend matches the checked-in Transformers feature fixture in
`crates/audio/testdata/qwen3_asr_features.json`. The complete Rust audio tower
matched the selected outputs in
`crates/audio/testdata/qwen3_asr_encoder.json` exactly for the pinned model;
the release rerun at the pinned revision measured max and mean absolute
error 0.000000 on the selected rows.

The decoder computes the prompt once and extends per-layer self-attention K/V
for each generated token. A synthetic grouped-query test compares each step
against the retained full-prefix reference and catches incorrect KV head
mapping. At the pinned revision the incremental-cache path reproduced the
pinned MLX transcript `The quick brown fox jumps over the lazy dog.`
exactly (debug correctness run; all three ignored tests in 223.5 s), and a
release CPU phase profile measured, over three fresh processes after one
discarded warmup on a quiet local machine: features about 4 ms, audio
encoder 5.02 to 5.08 s for 36 encoded frames, prompt 0.3 ms, prefill 1.38 to
1.42 s, and decode 3.56 to 3.69 s for 13 tokens, with peak `phys_footprint`
between 3224947544 and 3225455448 bytes (about 3.0 GiB, dominated by the
f32 dequantized resident weights). The audio encoder is the CPU hotspot:
5.0 s for 36 frames against 0.81 s of prefill for a 13-frame clip. These
are single-clip exploratory readings, not frozen benchmark rows, not a
serving gate, and not a quality evaluation. The earlier 491.94 s debug
figure belonged to the removed full-prefix decoder and is not comparable.

The pinned checkpoint stores 8-bit affine weights in groups of 64 with BF16
`.scales` and `.biases` on every quantized linear (197 tensors: the text
decoder plus the tied embeddings; the audio tower is plain BF16). This is
the same row layout `crates/gpu` `dequant_int8_gemv` consumes (`N` packed
bytes plus `N/64` BF16 scale and bias bit patterns), so that kernel is
layout-compatible with this decoder without a dtype change.

## Opt-in Metal decoder

`turbospark-runtime` `qwen3_asr_metal` keeps the decoder weights resident in
their packed checkpoint form (U32 words plus BF16 companions in one buffer
per linear) and dequantizes inside every GEMV; per-head RMSNorm, NeoX RoPE,
the split-KV decode attention with grouped query heads, and one command pass
per prompt row and per generated token reuse existing `crates/gpu` kernels.
The mel frontend and audio tower stay on the CPU path above. Embedding rows
are dequantized on the host per token because a one-hot GEMV over the whole
table is the wrong shape for a lookup.

At the pinned revision on a Metal device, the packed decoder reproduced the
CPU transcript exactly, both for one greedy step on a one-second prefix and
for the full pinned sentence. A synthetic full-layer gate additionally
generates a complete miniature checkpoint in the test (every packed decoder
weight byte, a byte-level BPE tokenizer, and a BF16 audio tower) and
requires `Qwen3AsrMetalEngine` and the CPU reference to greedy decode the
same clip to identical text, so a layout or cache regression cannot hide
behind real weights that happen to work
(`synthetic_checkpoint_metal_matches_cpu_reference` in
`turbospark-runtime`).

Release phase profiles of the Metal-only
process (three fresh runs after one discarded warmup on the same quiet local
machine): weight upload 357 to 371 ms, features about 4 ms, CPU audio
encoder 5577 to 5670 ms, Metal prefill of 46 prompt rows 212 to 231 ms, and
Metal decode of 13 tokens 60 to 63 ms (about 4.6 ms per token including the
151936-row LM-head GEMV). Peak `phys_footprint` measured
2117798336 to 2118142400 bytes (about 1.97 GiB), against 3.0 GiB for the
CPU path, because the packed resident weights replace the f32 dequantized
copy. The CPU audio tower is now the dominant phase end to end. The one
bring-up bug this gate caught: the decode-attention `kv_start` is the
attended window start, not the write row, so full causal context needs
`kv_start = 0`. The Metal path is
opt-in through `Qwen3AsrMetalEngine`; nothing dispatches to it by default.

The warm interleaved fresh-process protocol from
[Batching and interleaved probes](../../../../../docs/BENCHMARKING.md#qwen3-asr-stt-backend-probe)
(`crates/audio/scripts/benchmark_qwen3_asr.py`, three runs per backend,
alternating order, one discarded warmup per process) recorded on 2026-10-05
on the same single clip: CPU median transcription 10159.63 ms
(10068.45 to 10262.23) at a median peak `phys_footprint` of 3080.4 MiB;
Metal median 5411.69 ms (5369.82 to 5502.44) at 2073.9 MiB. All six
fresh-process transcripts hashed identically. The Metal path still carries
the shared CPU audio tower (about 5.6 s of its 5.4 s), so the packed decoder
roughly halves the end-to-end clip instead of reaching its 0.28 s decoder
time; a Metal audio tower remains the open pre-wiring bottleneck. These are
single-clip exploratory readings, not frozen benchmark rows, not a serving
gate, and not a quality evaluation; broader quality checks and the
default-on decision remain open.

Long-form chunk splitting, quality evaluation, and a Metal audio tower
remain open. The family itself is wired end to end: `SpeechFamily::Qwen3Asr`
(wire string `qwen3_asr`) in `crates/model-io`, the pinned
`qwen3-asr-06b-8bit` catalog row (`mlx-8bit-bf16` format) in
`crates/catalog`, an install probe that validates the tokenizer assets and
every checkpoint tensor descriptor before a receipt is written, the
portable CPU runner `turbospark_runtime::Qwen3AsrRunner` (one clip-level
segment; Qwen3-ASR has no alignment output), and the `ts_stt_*` C ABI
dispatch on the installed family. The Metal decoder stays opt-in through
`Qwen3AsrMetalEngine`; the session surface runs the CPU path. Swift
selection waits on the app's speech subsystem.

## Local checks

Download the pinned revision and verify it against the digests above:

```sh
hf download mlx-community/Qwen3-ASR-0.6B-8bit \
  --revision 89e96d92ba34aca20b3e29fb10cc284097d1219f \
  --local-dir ~/models/qwen3-asr-0.6b-8bit
```

The two decoder tests use the checked-in 16 kHz reference clip
`crates/audio/testdata/qwen3_forced_aligner_reference.wav`, which speaks the
pinned transcript sentence.

```sh
cargo test -p turbospark-audio --lib stt::qwen3_asr
TURBOSPARK_QWEN3_ASR_DIR=~/models/qwen3-asr-0.6b-8bit \
TURBOSPARK_QWEN3_ASR_WAV=crates/audio/testdata/qwen3_forced_aligner_reference.wav \
TURBOSPARK_QWEN3_ASR_PROFILE=1 \
  cargo test --release -p turbospark-audio --lib stt::qwen3_asr -- --ignored --nocapture
```

`TURBOSPARK_QWEN3_ASR_PROFILE=1` prints one `qwen3_asr_cpu` line with
per-phase wall times. Run the binary under `/usr/bin/time -l` and read its
`peak memory footprint` line for the `phys_footprint` high-water mark.

The opt-in Metal decoder lives in `crates/runtime` and takes the same two
environment variables:

```sh
TURBOSPARK_QWEN3_ASR_DIR=~/models/qwen3-asr-0.6b-8bit \
TURBOSPARK_QWEN3_ASR_WAV=crates/audio/testdata/qwen3_forced_aligner_reference.wav \
TURBOSPARK_QWEN3_ASR_PROFILE=1 \
  cargo test --release -p turbospark-runtime --lib qwen3_asr_metal -- \
  --ignored --nocapture
```

`pinned_checkpoint_transcript_matches_cpu_reference` also loads the f32 CPU
reference model, so its process footprint is NOT the Metal serving
footprint; use `pinned_checkpoint_transcribes_metal_only` for that.


The fixture generators are
`crates/audio/scripts/generate_qwen3_asr_features_fixture.py` and
`crates/audio/scripts/generate_qwen3_asr_encoder_fixture.py`. The encoder
generator needs the pinned checkpoint path in `TURBOSPARK_QWEN3_ASR_DIR`.
