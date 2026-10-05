# Qwen3-ForcedAligner

This module ports `mlx_audio/stt/models/qwen3_asr/qwen3_forced_aligner.py`
and reuses the shared Qwen3 audio/text implementation from
`mlx_audio/stt/models/qwen3_asr/qwen3_asr.py`. Reference version is
mlx-audio 0.5.7, source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Pinned profile

- Repository: `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`
- Revision: `0e1a68e91d815300c7c9754b2a7639378b23db15`
- Profile constant: `QWEN3_FORCED_ALIGNER_06B_8BIT`
- Input: mono f32 PCM at 16 kHz and a supplied transcript
- Output: per-word or per-character start/end times in seconds

The pinned model has 24 audio encoder layers, 28 text decoder layers, untied
token embeddings, and a 5000-class timestamp head at 80 ms resolution. Its
repo has no `tokenizer.json`; the loader reconstructs Qwen2 byte-level BPE
from the pinned vocab/merges and added-token table, including `<timestamp>`.

## Implementation and evidence

The port implements the audio feature path, shared Qwen3 audio encoder and
causal text decoder, timestamp projection, transcript tokenization, and the
reference timestamp repair. English space-delimited input and Chinese
character segmentation are supported. Japanese and Korean currently return
an explicit unsupported error because the Python reference relies on external
word-segmentation packages for those languages.

The checked-in `qwen3_forced_aligner.json` and
`qwen3_forced_aligner_reference.wav` were generated from this pinned MLX
checkpoint. Rust matched all nine words and start/end times within 1 ms. The
MLX spans run from `The` at 0.000-0.080 seconds to `dog` at 2.320-2.720
seconds. Fixture digests:

- `qwen3_forced_aligner.json`:
  `fb36285fc1eed0a6dc88a366d9cc08fb89adda8c974d7ad110c61c07a0ebac79`
- `qwen3_forced_aligner_reference.wav`:
  `ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb`

This is one smoke clip, not broader alignment quality evidence.

## Pinned artifact digests

Sizes are bytes; SHA-256 values were computed from files downloaded at the
revision above.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 1271924386 | `be19ef8ac4326d032e7673342930b14c2df30bd68c1632493b0f563e30829f91` |
| `config.json` | 6940 | `8dc126e1a4ac001acb02f3a9ca6277c8a20a25485fde293bc99eeab3aedc604d` |
| `preprocessor_config.json` | 330 | `45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d` |
| `tokenizer_config.json` | 12666 | `3ab80063f8511deb9566e6ad438d17b7a6277fcffd52d92854112f19d36bd81c` |
| `vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `chat_template.json` | 1161 | `75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff` |
| `generation_config.json` | 115 | `948d089b23bca1d214e768d59c4438365665f52ec6d33678f4062206b3fbbb8c` |

## Local checks

```sh
cargo test -p turbospark-audio --lib models::stt::qwen3_forced_aligner
TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR=/path/to/pinned-model \
  cargo test -p turbospark-audio --lib \
  models::stt::qwen3_forced_aligner::tests::pinned_checkpoint_matches_mlx_word_alignment_fixture \
  -- --ignored --nocapture
```

Regenerate the MLX fixture with
`crates/audio/scripts/generate_qwen3_forced_aligner_fixture.py`. Set
`TURBOSPARK_QWEN3_FORCED_ALIGNER_DIR` to the pinned model directory and
`TURBOSPARK_QWEN3_ASR_WAV` to a mono 16 kHz WAV.
