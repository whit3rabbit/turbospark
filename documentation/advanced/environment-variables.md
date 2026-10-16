---
description: Comprehensive reference of all TURBOSPARK_* environment variables.
icon: gear
---

# Environment Variables Reference

TurboSpark recognizes environment variables for runtime tuning, storage configuration, diagnostic profiling, and hardware controls. All TurboSpark-specific variables use the `TURBOSPARK_*` prefix.

## Storage and Paths

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_HOME` | All binaries | Root directory for installed models, catalogs, and temporary staging. | `~/.turbospark` |
| `TURBOSPARK_STATE_DIR` | Desktop App | Directory for application state, chat history, and encrypted user profiles. | `~/Library/Application Support/TurboSparkApp` |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | `turbospark-model` | Bearer token for accessing private or gated Hugging Face repositories. | unset |

---

## Server & Networking

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_API_KEY` | `turbospark-server` | Secret key required for API authorization (`Authorization: Bearer <key>`). | unset (unauthenticated on loopback) |

---

## Runtime & Memory Controls

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_EXPERT_SLOTS` | CLI, server, app | Number of slot-cache buffers allocated per layer for MoE streaming. | `16` |
| `TURBOSPARK_PREFILL_CHUNK` | CLI, server | Token chunk size for prompt evaluation (e.g. `128`, `256`, `512`). | unset (auto) |
| `TURBOSPARK_BATCHED_GEMV` | Qwen MoE runtime | Enables batched resident weight reads (Steel GEMM). Set `0` for per-token loop. | auto (enabled on INT4) |
| `TURBOSPARK_KV_QUANT` | Runtime, server | KV-cache quantization format: `fp16`, `int8`, or `int4`. | `fp16` |
| `TURBOSPARK_METAL_KERNEL_WARMUP` | Runtime | Set `1` to pre-compile and warm Metal pipelines at model open time. | `0` (off) |
| `TURBOSPARK_EXPERT_RESIDENCY` | Runtime | Set `mapped` to read routed experts directly from mmap rather than slot cache. | `streamed` |
| `TURBOSPARK_READ_QOS` | Streaming I/O | Set `utility` to drop background expert disk reads to efficiency cores. | `user-initiated` |

---

## Speculative Decoding

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_MTP_DRAFT` | Runtime | Multi-Token Prediction draft block depth (`0` to disable, `2` for block 2). | `auto` |
| `TURBOSPARK_SPEC_STATS` | CLI, server | Set `1` to log token acceptance rates and rollback counts to stderr. | unset |
| `TURBOSPARK_DFLASH_DRAFT` | Runtime | DFlash block-diffusion draft depth (`0` to disable). | `auto` |

---

## Audio Controls

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_WHISPER_DEVICE` | Audio runtime | Set `cpu` to force CPU reference execution for Whisper instead of Metal. | `metal` (auto) |
| `TURBOSPARK_WHISPER_PROFILE`| Audio runtime | Set `1` to print phase timing (mel, encode, decode) to stderr. | unset |

---

## Profiling and Diagnostics

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_PHASES` | `turbospark-check` | Set `1` to print forward-pass phase timing breakdowns to stderr. | unset |
| `TURBOSPARK_DISPATCH_PROFILE` | CLI, server | Set `1` to collect per-dispatch Metal kernel timing inside command buffers. | unset |
| `TURBOSPARK_ROUTER_HIST` | Runtime | File path to dump per-layer expert selection frequency histograms as JSON. | unset |
