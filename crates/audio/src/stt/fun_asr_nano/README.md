# Fun-ASR-Nano-2512

This module ports the pinned mlx-audio inference path for Fun-ASR-Nano, using
its SANM speech encoder, audio adaptor, and tied Qwen3 0.6B decoder.

## Pinned profile

- Hugging Face repository: `mlx-community/Fun-ASR-Nano-2512`
- Immutable revision: `a7bc96fceaafce39ed6748e0c0fa9a9508b67f86`
- mlx-audio reference: 0.5.7 at `e1b19b9054bf163f5d812221a54fcc346f1890e9`
- Input: mono f32 PCM at 16 kHz
- Frontend: 80-bin Kaldi FBANK, 25 ms window, 10 ms hop, LFR 7x6
- Audio encoder: 50 SANM blocks and 20 prediction blocks, width 512
- Adaptor: two self-attention blocks projecting 512 to Qwen3 width 1024
- Text decoder: tied Qwen3 0.6B BF16 weights, 28 layers

Pinned artifact digests:

| Artifact | Size in bytes | SHA-256 |
|---|---:|---|
| `model.safetensors` | 1659734710 | `2e2ba5e2591f75f63b25ee5b53a895d3cd5afcc642c3db4482d3ee27b0a00449` |
| `config.json` | 1706 | `4df3d9ff5b222b7a0be5dcf3f815dba995f37c6fbefe387436072303670c0ad8` |
| `Qwen3-0.6B/tokenizer.json` | 11422654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `Qwen3-0.6B/tokenizer_config.json` | 9732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |

`FunAsrNano::load` reads a completed local snapshot only. It rejects different
frontend, encoder, adaptor, or tokenizer geometry. The Rust inference path is
portable CPU f32 math. BF16 Qwen weights are converted to f32 on load.

## Reference and parity

The pinned mlx-audio 0.5.7 profile loaded with `mlx_audio.stt.utils.load` and
returned `The quick brown fox jumps over the lazy dog.` from the checked-in
direct-float WAV. `generate_fun_asr_nano_fixture.py` records selected LFR,
encoder, and adaptor values from that same pinned MLX run. The checkpoint-gated
Rust release test compares those stages within 0.01 max absolute error and
matches the final transcript exactly. The reference WAV SHA-256 is
`ea38d350100b5d532fbc3e3517499dbd9915b9b0890d9e819bae384fa1384bcb`.

Regenerate the fixture on an Apple Silicon host with the mlx-audio reference
environment:

```sh
export TURBOSPARK_FUN_ASR_NANO_DIR="$(../mlx-audio/.venv/bin/python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("mlx-community/Fun-ASR-Nano-2512", revision="a7bc96fceaafce39ed6748e0c0fa9a9508b67f86", cache_dir="/tmp/turbospark-funasr-cache", allow_patterns=["config.json", "model.safetensors", "Qwen3-0.6B/*.json", "Qwen3-0.6B/*.txt"]))')"
../mlx-audio/.venv/bin/python crates/audio/scripts/generate_fun_asr_nano_fixture.py \
  --model-dir "$TURBOSPARK_FUN_ASR_NANO_DIR" \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/fun_asr_nano_reference.json
CARGO_TARGET_DIR=/tmp/turbospark-funasr-target CARGO_BUILD_JOBS=1 \
  TURBOSPARK_FUN_ASR_NANO_DIR="$TURBOSPARK_FUN_ASR_NANO_DIR" \
  cargo test --release --locked -p turbospark-audio \
  fun_asr_nano::tests::pinned_checkpoint_matches_mlx_transcript --lib \
  -- --ignored --nocapture --test-threads=1
python3 -c 'from pathlib import Path; import shutil; shutil.rmtree(Path("/tmp/turbospark-funasr-cache"), ignore_errors=False)'
```

The fixture is checked into `testdata/fun_asr_nano_reference.json`; tests do not
download model files. The pinned snapshot and waveform are removed after a
verification run. One short English clip does not qualify recognition quality,
other languages, long-form chunking, runtime integration, memory, throughput,
FFI, or Swift support.
