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
`crates/audio/testdata/qwen3_asr_encoder.json` exactly for the pinned model.
An earlier end-to-end Rust checkpoint run matched the pinned MLX transcript exactly:
`The quick brown fox jumps over the lazy dog.` The debug run took 491.94
seconds with the original full-prefix decoder; this is correctness evidence,
not a serving performance measurement for the new cache path.

The decoder now computes the prompt once and extends per-layer self-attention
K/V for each generated token. A synthetic grouped-query test compares each
step against the retained full-prefix reference and catches incorrect KV
head mapping. The pinned checkpoint transcript and timing need rerunning for
this change. Long-form chunk splitting, quality evaluation, packed 8-bit
Metal weights, runtime/catalog integration, FFI, and Swift remain open.

## Local checks

```sh
cargo test -p turbospark-audio --lib stt::qwen3_asr
TURBOSPARK_QWEN3_ASR_DIR=/path/to/pinned-model \
  cargo test -p turbospark-audio --lib \
  stt::qwen3_asr::encoder::tests::pinned_audio_tower_matches_mlx_fixture \
  -- --ignored --nocapture
```

The fixture generators are
`crates/audio/scripts/generate_qwen3_asr_features_fixture.py` and
`crates/audio/scripts/generate_qwen3_asr_encoder_fixture.py`. The encoder
generator needs the pinned checkpoint path in `TURBOSPARK_QWEN3_ASR_DIR`.
