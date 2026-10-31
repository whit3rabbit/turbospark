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
| `TURBOSPARK_ROUTED_BATCH` | MoE prefill (Gemma 4, gpt-oss, Qwen MoE batched arm) | Set `1` to run each layer's routed experts as one batched dispatch pair per union-bounded sub-batch during chunked prefill. On Qwen MoE it needs the batched arm and refuses steering and the residual capture by name. | off |
| `TURBOSPARK_ROUTED_BATCH_OVERLAP` | Qwen MoE prefill | Set `1`, beside `TURBOSPARK_ROUTED_BATCH=1`, to read ahead: each sub-batch's experts are planned around the slots the in-flight sub-batch reads and streamed while it runs. Byte-identical (slot choice only). Pays only where the cache holds two 32-expert unions (64 slots and up), otherwise falls back to wait-then-plan. Prints `[engaged] routed_batch_overlap` under `TURBOSPARK_ENGAGEMENT=1`. | off |
| `TURBOSPARK_BATCHED_GEMV` | Qwen MoE chunked prefill | On INT4-affine installs, run the GDN and attention projections and the shared expert as multi-row Steel GEMMs at a 64-row micro-batch. 1.4x to 1.6x faster prefill on a 2,940-token prompt, but top-8 routing near-ties can flip under the batched reduction order (mean final-position KL 3.9e-3 nats). Set `0` for the per-token arm, which still matches sequential prefill exactly. | auto (on for INT4) |

---

## Decode Attention and Per-Family Kernels

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_ATTN_DECODE_KERNEL` | Decode attention (all families) | Forces the decode attention kernel process-wide. `gqa` runs the GQA-shared kernel, which serves a whole query group from one K/V read. `barrier` runs the older one-threadgroup-per-head kernel. Unset lets each family choose: dense Qwen GDN, LFM3 and Flint take GQA-shared (Flint: 1.49 to 0.40 ms per full layer at 512 positions in `tslab kernel --bench attn_decode`), every other family keeps barrier until its own quality gates run. Not bit-exact, KL-bounded. A forced value also changes the trunk's speculative verify. | unset (per family) |
| `TURBOSPARK_LFM3_CHUNKED_PREFILL` | LFM3 prefill | Set `0` to run LFM3's prefill chunks through the per-token loop instead of the M-row driver (Steel GEMMs from 32 rows, short-conv prefill, multi-row GQA attention). INT4-affine installs only: the GGUF Q8_0 install stays sequential either way. KL-bounded. | on |
| `TURBOSPARK_LFM3_FUSED_FFN` | LFM3 decode | Set `0` to run decode's FFN gate, up and SiLU as three dispatches instead of one fused INT4 GEMV. Bit-exact either way. INT4 installs only. | on |
| `TURBOSPARK_FLINT_CHUNKED_PREFILL` | Flint prefill | Set `1` to run Flint's chunked prefill as the layer-batched driver (one command buffer per layer, routed half pipelined across two banks). Byte-identical to the default sequential path. The speedup is not established. | off |

Each of these is a `tslab knobs` entry, so an A/B arm can set it by short name
(`attn_decode_kernel=barrier`, `lfm3_fused_ffn=0`). Since GQA-shared is the
default for the three families above, the A/B to confirm it is baseline
`default` against candidate `barrier:attn_decode_kernel=barrier`.

---

## ANE+Steel Prefill Split (dense Qwen)

The split sends part of each dense projection to the Neural Engine while Steel computes the rest. It needs the `ane-prefill` cargo feature, an Apple Silicon M4 or A18 class chip, and a programs directory written by `crates/coreml/tools/gen_ane_prefill_programs.py`. Missing or mismatched programs turn the split off for the runner with one warning, and prefill runs on Steel. A failed prediction or a non-finite ANE row redoes that micro-batch on Steel.

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_ANE_PREFILL` | Runtime (dense Qwen) | `1`, `on` or `split` enables the split. `auto` enables it only on M4+ / A18+ and keeps Steel otherwise. `break` keeps every command-buffer break and host handoff but runs every row on Steel, a diagnostic for the cost of the breaks alone. Anything else is off. Steps aside while an MTP head is open, and while a DFlash2 drafter is open unless `TURBOSPARK_ANE_PREFILL_DFLASH=1`. | off |
| `TURBOSPARK_ANE_PREFILL_DIR` | Runtime | The programs directory (`manifest.json` plus Core ML packages). Unset falls back to the first of `~/.turbospark/models/ane-prefill/qwen38`, `~/.turbospark/ane-prefill/qwen38` and `~/models/ane-prefill/qwen38` that holds a manifest. | unset |
| `TURBOSPARK_ANE_PREFILL_DFLASH` | Runtime | Set `1` to let the split run with a DFlash2 drafter open. The drafter's aux capture widens from 64 to 2048 rows (about 100 MiB at Qwen3.8-27B's shape) and priming moves to 2048-token spans so the split's 512-row minimum is reachable. | off |
| `TURBOSPARK_ANE_PREFILL_STATS` | Runtime | Set `1` to print one counter line per split micro-batch to stderr, including the host handoff (wake and resume) that `tslab kernel --bench ane_handoff` times. | unset |
| `TURBOSPARK_ANE_PREFILL_FAULT` | Runtime | Diagnostic. Fails the prediction at split point `P` of the first micro-batch that reaches it, once per process, to exercise the Steel redo path. | unset |

**Baked INT4 programs.** `crates/coreml/tools/bake_ane_prefill_programs.py`
writes programs with the install's INT4 weights baked in (uint4 shift-scale, or
`--lut`), one per class, layer and width, as a format 2 manifest. Point
`TURBOSPARK_ANE_PREFILL_DIR` at that directory to use them. A layer with
neither a baked nor an F16 program runs on Steel. Measured result: the baked
programs match the F16 split's speed and hold 160 to 300 MiB wired each, so
the F16 programs stay the default (`docs/SPEEDUP_PLAYBOOK.md`).

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
| `TURBOSPARK_ENGAGEMENT` | Runtime, gpu | Set `1` to print `[engaged] NAME` to stderr the first time each opt-in lane runs (MMA8 tile, Steel prefill, rows attention, decode attention kernels, fused FFN, lazy GDN tape, the ANE split, routed read-ahead, probes and stand-ins). Prints only, changes nothing. `tslab` sets it on every run and refuses an arm whose log contradicts its switches. | unset |
| `TURBOSPARK_LOGIT_DUMP_IDS` | `crates/bench` `logit_dump` test | Path to a `{"prompt_ids": [...], "ids": [...]}` file. Replaces the quality corpus with that explicit trajectory (usually another engine's greedy path), so agreement is read along the path that engine took. | unset |

---

## TurboSpark Lab (tslab)

| Variable | Affected Tools | Description | Default |
| --- | --- | --- | --- |
| `TURBOSPARK_LAB_DIR` | `tslab`, skill scripts | Results root for every ledger. Checked before `TURBOSPARK_HOME`. | `~/.turbospark/lab` |
| `CARGO_TARGET_DIR` | `tslab run-experiment`, `doctor.sh` | Where the freshly built `tslab-worker` is found (`<target>/release/tslab-worker`). | `target` |
| `TSLAB_ANE_MODEL` | `turbospark-lab` ignored test | Dense Qwen install that the real-program `ane_handoff` test runs on. Needs `--features ane-prefill`. | unset |

tslab clears the environment for every worker it starts. A worker sees only
`PATH`, `HOME`, `TMPDIR`, `USER`, `LANG`, `LC_ALL`, `TURBOSPARK_HOME`,
`TURBOSPARK_LAB_DIR`, `TURBOSPARK_ENGAGEMENT=1`, and the variables the arm
sets. To A/B any `TURBOSPARK_*` switch, put it on the arm
(`--candidate on:TURBOSPARK_X=1`) or on every arm (`--env KEY=V`), not in your
shell. `tslab knobs` lists every switch with its values and whether it is
bit-exact. See [TurboSpark Lab (tslab)](tslab.md).
