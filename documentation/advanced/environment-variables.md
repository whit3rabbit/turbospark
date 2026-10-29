---
description: Comprehensive reference of all TURBOSPARK_* environment variables.
icon: gear
---

# Environment Variables Reference

TurboSpark recognizes environment variables for runtime tuning, storage configuration, diagnostic profiling, and hardware controls. All TurboSpark-specific variables use the `TURBOSPARK_*` prefix.

Most performance switches below are **on by default** and exist as kill switches: set the variable to `0` to restore the earlier code path for an A/B comparison or to work around a problem. Each is read once when the model opens, not per token.

## Storage and Paths

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_HOME` | All binaries | Root directory for installed models, catalogs, and temporary staging. | `~/.turbospark` |
| `TURBOSPARK_LAB_DIR` | `tslab` | Where the TurboSpark Lab evaluator writes and reads its result ledgers (`tslab serve` serves this directory). Checked before `TURBOSPARK_HOME`, so `$TURBOSPARK_HOME/lab` applies only when this is unset. See [TurboSpark Lab (tslab)](tslab.md). | `~/.turbospark/lab` |
| `TURBOSPARK_STATE_DIR` | Desktop App | Directory for application state, chat history, and encrypted user profiles. | `~/Library/Application Support/TurboSparkApp` |
| `TURBOSPARK_DISABLE_DOWNLOAD_CACHE` | `turbospark-model` | Set `1` to reduce peak disk use during pinned model pulls. Interrupted downloads must fetch their byte ranges again. | unset |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | `turbospark-model` | Bearer token for accessing private or gated Hugging Face repositories. | unset |

---

## Server & Networking

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_API_KEY` | `turbospark-server` | Secret key required for API authorization (`Authorization: Bearer <key>`). `--bind tailnet` refuses to start without one. | unset (unauthenticated on loopback) |

---

## Runtime & Memory Controls

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_PREFILL_CHUNK` | CLI, server | Token chunk size for prompt evaluation (e.g. `128`, `256`, `512`). | unset (auto) |
| `TURBOSPARK_METAL_KERNEL_WARMUP` | Runtime | Set `1` to pre-compile and warm Metal pipelines at model open time. | `0` (off) |
| `TURBOSPARK_EXPERT_RESIDENCY` | Runtime | Set `mapped` to read routed experts directly from mmap rather than the slot cache. A mapped residency that would exceed Metal's wired budget is refused with the reason; `auto` resolves down to `streamed`. | `streamed` |
| `TURBOSPARK_EXPERT_CACHE_DECAY` | Streaming I/O | A positive integer `N`: halve every routed-expert LFU count each `N` plans, so a workload change is not shadowed by stale favorites. Output is unchanged; only which experts stay resident moves. A value that is not a positive integer is an error, not a silent fallback. | unset (cumulative LFU) |
| `TURBOSPARK_READ_QOS` | Streaming I/O | Set `utility` to drop background expert disk reads to efficiency cores. | `user-initiated` |

{% hint style="info" %}
KV-cache quantization is not an environment variable. Pass `--kv-bits off|2|3|3.5|4` (TurboQuant) to `turbospark-check` or `turbospark-server`. The default is `off`, and an unsupported head dimension or family refuses the flag at open rather than falling back to FP16.
{% endhint %}

---

## Qwen Prefill and Decode Kernels

These switches select the kernel path for the dense and MoE Qwen families. Every arm produces the same bits unless the row says otherwise.

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_QWEN_FUSED_FFN` | Qwen dense and MoE decode | Fused INT4 FFN dispatches: gate+up+activation in one GEMV, and the down projection with its residual add or shared-expert gate folded into the store epilogue. Bit-identical to the per-tensor chain. Set `0` to restore the per-tensor dispatch chain. Non-INT4 tensors never take the fused branch. | on |
| `TURBOSPARK_QWEN_BATCHED_ROW_OPS` | Qwen batched prefill, speculative verify, MTP priming | Encode the small per-row operations (q/gate split, q/k norm, RoPE, output gate, FFN activation, residual adds and norms) as one dispatch over all rows. Bit-identical. Set `0` for the one-dispatch-per-row form. | on |
| `TURBOSPARK_QWEN_MOE_CHUNKED` | Qwen MoE prefill (`qwenGdnMoe`: Qwen 3.6, Ornith 35B, MiniCPM-V 4.7) | Chunked prefill: the attention and router half of up to 16 tokens runs before one wait per layer, then the routed experts token by token. Set `0` to fall back to sequential prefill without an error; this is also the reference arm for byte-identity checks. | on |
| `TURBOSPARK_BATCHED_GEMV` | Qwen MoE chunked prefill | On INT4-affine installs, run the GDN and attention projections and the shared expert as multi-row Steel GEMMs at a 64-row micro-batch. 1.4x to 1.6x faster prefill on a 2,940-token prompt, but top-8 routing near-ties can flip under the batched reduction order (mean final-position KL 3.9e-3 nats). Set `0` for the per-token arm, which still matches sequential prefill exactly. | auto (on for INT4) |

---

## ANE+Steel Prefill Split (dense Qwen)

The split sends part of each dense projection to the Neural Engine while Steel computes the rest. It needs the `ane-prefill` cargo feature, an Apple Silicon M4 or A18 class chip, and a programs directory written by `crates/coreml/tools/gen_ane_prefill_programs.py`. Missing or mismatched programs turn the split off for the runner with one warning, and prefill runs on Steel. A failed prediction or a non-finite ANE row redoes that micro-batch on Steel.

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_ANE_PREFILL` | Runtime (dense Qwen) | `1` or `on` enables the split. `break` keeps every command-buffer break and host handoff but runs every row on Steel, a diagnostic for the cost of the breaks alone. Anything else is off. | off |
| `TURBOSPARK_ANE_PREFILL_DIR` | Runtime | The programs directory. | unset |
| `TURBOSPARK_ANE_PREFILL_STATS` | Runtime | Set `1` to print one counter line per split micro-batch to stderr. | unset |

The desktop app exposes the same split as an **ANE+Steel prefill** picker in Engine Settings (Auto, On, Off). See [Hardware Acceleration](../models/hardware-acceleration.md).

---

## Speculative Decoding

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_MTP_DRAFT` | Runtime | Multi-Token Prediction draft block depth (`0` to disable, `2` for block 2). | `auto` |
| `TURBOSPARK_DFLASH_DRAFT` | Runtime | DFlash block-diffusion draft depth (`0` to disable). | `auto` |
| `TURBOSPARK_GDN_LAZY_TAPE` | Runtime (GDN families) | Lazy rank-1 verify tape and state fold: a verify on clean recurrent state records lazily instead of snapshotting with blits. Set `0` to snapshot on every verify. | on |
| `TURBOSPARK_SPEC_STATS` | CLI, server | Set `1` to log token acceptance rates and rollback counts to stderr. | unset |

The block size and the adaptive controller are command-line flags, not variables: `--speculative auto|off|<1-15>|adaptive[:MAX]`. See [Speculative Decoding](speculative-decoding.md).

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
