# Voxtral Realtime 4B

This module ports `mlx_audio/stt/models/voxtral_realtime/`
(voxtral_realtime.py, encoder.py, decoder.py, audio.py, config.py,
tokenizer.py) from mlx-audio 0.5.7 at source commit
`e1b19b9054bf163f5d812221a54fcc346f1890e9`.

## Overview

Offline buffered transcription path:

1. The 16 kHz waveform is aligned to 1280-sample streaming tokens and padded
   with left and right silence in token units (`_pad_audio_streaming`).
2. A 128-band log-mel spectrogram (whisper-family frontend with the
   `global_log_mel_max = 1.5` clamp).
3. A causal sliding-window encoder: 32 layers, 32 heads, head dim 64, RoPE
   theta 1e6 (interleaved), window 750, conv-style norm, 4x frame
   downsampling in the adapter.
4. The prompt is `[BOS]` plus streaming pad tokens; each decoder input is
   the audio embedding plus the token embedding at that position
   (`compute_time_embedding`), prefill runs once and greedy steps continue
   to EOS.
5. Tekken (byte-level BPE) decode skips special tokens and joins bytes.

The real-time chunked `StreamingSession` (delay/commit semantics in
streaming.py) is out of scope and refused with a reason; the offline path
shares the same encoder, decoder, and window semantics (the ring cache
evicts only what no future query can see, verified bit-exact against the
unwindowed causal path inside the window and against a cache-free windowed
reference across the boundary).

## Pinned profile

- Hugging Face repository: `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`
- Immutable revision: `fdebf7b2af834a1db4b8a3c99ab7480b333adf9e`
- Input: mono f32 PCM at 16 kHz
- Encoder: 32 layers, dim 1280, 32 heads, head dim 64, window 750
- Decoder: 26 layers, dim 3072, 32 query heads, 8 KV heads, head dim 128
- Rust profile: `VOXTRAL_MINI_4B_REALTIME_4BIT`

Artifact sizes are bytes; SHA-256 digests were computed from the downloaded
snapshot at the pinned revision.

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 3133798126 | `6f59b425d8a1ceb2de795454558be63937cf75b59f9c9bc77accd85aaf32af05` |
| `model.safetensors.index.json` | 118632 | `80f68b80cf4b1638d864d1504061a266f59e37a8d90d7b20f2e1f30c2d034c2e` |
| `config.json` | 1513 | `02060864a4f33df5e4944684fc17f3026af4011830cac4def6e9e025315b10c5` |
| `tekken.json` | 14910348 | `8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44` |
| `README.md` | 3135 | `022af0f470bbb2036d1d8fea62118d1d14f64b9cbb38415004544990e653f99e` |

## Reference and parity

`generate_voxtral_realtime_fixture.py` runs the pinned checkpoint through
the reference classes and records the padding, mel, encoder, prompt,
prefill hidden, first logits, per-step logits, generated token ids, token
bytes, and transcript into
`crates/audio/testdata/voxtral_realtime_reference.json` (force-added; the
workspace `.gitignore` covers testdata directories). Tests never download
model files.

Regenerate on an Apple Silicon host with the mlx-audio reference
environment:

```sh
~/.venv-mlxaudio/bin/python crates/audio/scripts/generate_voxtral_realtime_fixture.py \
  --model-dir ~/models/voxtral-mini-4b-realtime-4bit \
  --audio crates/audio/testdata/qwen3_forced_aligner_reference.wav \
  --output crates/audio/testdata/voxtral_realtime_reference.json \
  --revision fdebf7b2af834a1db4b8a3c99ab7480b333adf9e
TURBOSPARK_VOXTRAL_REALTIME_MODEL_DIR=~/models/voxtral-mini-4b-realtime-4bit \
  cargo test --release -p turbospark-audio --lib stt::voxtral_realtime -- \
  --ignored --nocapture --test-threads=1
```

- **Python reference transcript (pinned checkpoint):**
  `The quick brown fox jumps over the lazy dog.`
- **Rust checkpoint run:** identical transcript and generated token bytes;
  chunked-vs-full encoder outputs bit-identical (worst diff `0.0`), adapter
  spots `1.3e-6`, prefill hidden `6.1e-5`, per-step log-probability gates
  all under `1.1e-5` on the smoke clip.

## Verification status

Both the Python reference and the Rust release-mode checkpoint run were
executed on this machine against the pinned revision. Two port bugs were
found and fixed during verification, each caught by a unit test before the
checkpoint run: the causal encoder attention path skipped key rotation
(upstream rotates q and k with the same offsets), and the ring cache
trimmed to capacity when a chunk was appended, over-evicting keys that
in-chunk queries still needed (upstream answers query `p` before any later
key exists; trimming now happens after a chunk's attention).

The smoke clip is shorter than the 750-frame window, so the window-boundary
path is exercised by the synthetic self-test only; long-form audio, the
real-time streaming session, other languages, quality, performance, memory,
runtime, catalog, FFI, and Swift gates remain separate work.
