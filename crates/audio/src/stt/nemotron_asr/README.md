# Nemotron 3.5 ASR

This Rust port follows `mlx_audio/stt/models/nemotron_asr/` from mlx-audio
0.5.7 at commit `e1b19b9054bf163f5d812221a54fcc346f1890e9`.

Pinned checkpoint:

- Repository: `mlx-community/nemotron-3.5-asr-streaming-0.6b`
- Revision: `e550040c0478027ed679b2b6b0d055502c103663`

The library loads a local Hugging Face snapshot and transcribes mono PCM with
the reflect-padded NeMo frontend, causal FastConformer, language prompt, and
greedy RNN-T decoder. The current path encodes a complete recording with its
trained chunk-limited attention mask. It does not implement cache-aware online
streaming.

Example:

```sh
cargo run -p turbospark-audio --example nemotron_transcribe -- <model-dir> <audio.wav> [language]
```

One generated 16 kHz speech clip produced the same transcript in Rust and the
mlx-audio 0.5.7 Python reference. The first encoder block's attention output
and final encoder stage were also compared against MLX. This single-clip
evidence does not qualify broader recognition quality, streaming, performance,
memory, or runtime, catalog, FFI, and Swift integration.
