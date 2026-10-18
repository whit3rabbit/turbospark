# MoE prior art: jitllm, phhusson's NPU forks, and llama.cpp #30340

Reviewed 2026-10-14 for anything the routed-expert streamer
(`crates/streaming`) or the MoE layer paths could borrow. The `main-npu`
part was re-read on 2026-10-17, after the fork's author pointed to it in the
review of PR #151: the first pass took one number from that branch and
missed its quantized-weight path. Every source was read. Only `main-npu`, a
Core ML probe of this engine's prefill matmuls, and a prototype split inside
this engine's prefill were run, on one M4 Max (see "Measured on an M4 Max",
"Phase 0 in this engine", and "Phase 1 in this engine"). The engine
measurements items 1 to 3 cite live in `docs/EXPERT_ROUTING.md`. Any other
row is a claim from the source until the experiment it names has run on a
Mac.

Sources and the revisions read:

- [jitllm/jitllm](https://github.com/jitllm/jitllm) at `5f6a5cd`
  (2026-10-14), Go, Apache-2.0. Its own measurements live in
  `docs/engineering-history/placement.md` (cited below as PL) and
  `docs/design/block-paging.md`.
- [phhusson/npu_experimentations](https://github.com/phhusson/npu_experimentations)
  at `e5794cf`, a README with no license, and the forks it links:
  `phhusson/llama.cpp` branches `apple/bonsai-pq20` (tip `c6af028`) and
  `main-npu` (tip `d0e54af`), MIT; `phhusson/h3.c-ane` branch
  `dev/phh/mixed` (tip `fe44f7a`), MIT plus BSD-3-Clause notices.
  `main-npu` has since been rebased onto newer upstream (tip `689fd77`).
  Its `ggml/src/ggml-ane/ggml-ane.m`, `docs/backend/ANE.md`,
  `ggml/include/ggml-ane.h` and `scripts/ane-sweep.py` are byte-identical
  at both tips.
- [ggml-org/llama.cpp#30340](https://github.com/ggml-org/llama.cpp/pull/30340),
  dma-buf buffer sharing between ROCm and Vulkan, closed unmerged on
  2026-10-12 after a maintainer veto.

Read this page before proposing an expert-cache policy, a cross-layer slot
budget, a cold-read tuning pass, or Neural Engine offload.

## What it comes to

| Idea | Source | State here | Next step |
|---|---|---|---|
| Aging the cache's use counts | jitllm ROADMAP 12 (from ds4) | **measured, default unchanged**: the `TURBOSPARK_EXPERT_CACHE_DECAY` seam (2026-10-13) won at 16 slots and cost 17-50% of decode at the `auto` 96 on two families | a physical-bytes-aware arm, see `docs/EXPERT_ROUTING.md` |
| One slot budget shared by all layers | jitllm `format/jlm/read.go` `evictOne` | **closed** on the qwen4exp replay: `pool lru` buys no hit rate over per-layer LRU | only if a pooled decayed policy replays ahead first |
| An offline-optimal bound before policy work | jitllm `engine/model/route_test.go` `replayBelady` | **built here**: `scripts/expert_policy_replay.py` | run it on real captures |
| Queue depth for cold reads | jitllm `format/jlm/read.go`, PL 15r | **measured, constants unchanged**: on the disk-bound qwen4exp, 16-32 threads and whole-expert reads moved expert io 5% and decode within noise | none; depth is bounded by misses per layer |
| Per-expert contiguous pages, pread over cold mmap, concurrent misses, shared expert overlap | jitllm | already here | none |
| Cross-layer router prefetch | jitllm `Config.StreamPrefetch` | closed here (warm); jitllm's disk-bound result lowers the prior on the owed cold sweep | none new |
| Moving blocks between devices, routed experts on the CPU | jitllm `moveblocks.go`, `streamBank.runHost` | does not apply to unified memory | none |
| Neural Engine prefill matmuls, Metal converting weights and activations | phhusson `main-npu` | **Phase 0 measured** (2026-10-18): the fork loses on an M4 Max, but the same matmul through public Core ML, split with Steel, ran 1.10x to 1.77x faster per matmul (1.24x or more in 8 of 10 cases) | an opt-in Phase 1 split prototype with real-model gates, a separate decision from MoE work |
| Neural Engine row-split prefill on repacked 2-bit weights | phhusson `apple/bonsai-pq20`, `h3.c-ane` | not started; one encoding, private API | only if the `main-npu` route is taken and Bonsai needs more |
| Cross-device zero-copy buffers | llama.cpp #30340 | unified memory already has it | only a fence lesson, for ANE work |

## What could transfer, and how to price it first

All three rows below are POLICY questions, and every policy question on this
engine has been answered from a trace before it was answered in code
(`docs/EXPERT_ROUTING.md`). `scripts/expert_policy_replay.py` is that
instrument for these three. It reads the same
`TURBOSPARK_ROUTER_HIST` + `TURBOSPARK_ROUTER_TRACE=1` captures as
`scripts/pilot_ceiling.py` and reuses that script's validated port of the
engine's LFU, so the `lfu (engine)` row is the number `pilot_ceiling.py`
already reconciled against the engine's own counters.

```sh
python3 scripts/expert_policy_replay.py coding.json general.json \
  --slots 16,32 --skip 21
```

**Read the `oracle` row first.** It is Belady's rule per layer, checked
against exhaustive search in `scripts/test_expert_policy_replay.py`, and no
online policy at that slot count can miss less. If it sits within a point or
two of `lfu (engine)`, policy work is closed at that slot count and slots
remain the lever (`docs/DECODE_BUDGET.md`).

### 1. The LFU counts never age by default, and a server carries them across requests

**Measured since.** This section was written on a branch that did not yet
have the decay seam (`8f1ef1ab`, 2026-10-13). Replays on Gemma 4 and qwen4exp
then favored decay at every slot count. The real multi-request A/B on
2026-10-17 agreed at 16 slots (+3.0%) and cost 17-50% of decode at the
`auto` 96 slots on Ornith 35B and qwen4exp, because decay's misses were cold
device reads that no replay counts
(`docs/EXPERT_ROUTING.md`, last section). The default stays cumulative LFU.
The reasoning below is kept as the question that measurement answered.

`ExpertCache` (`crates/streaming/src/expert_cache.rs`) increments
`expert_use_count` on every request and, by default, never decays it.
`PreadExpertStreamer::open` builds one cache per layer per runner open, and
nothing resets it. So a long-lived runner, the server's normal shape, ranks
experts by their count over its whole lifetime. That behaves like a static
pin of the historically hottest set, and this repo has already measured that
a static pin loses to temporal locality (`docs/EXPERT_ROUTING.md`, "A pinned
coding set loses to the cache").

Every published hit rate is one short process. The 84.6% at 32 slots is a
220-pass capture, where the counts are small and the last-use tie-break
still steers eviction. The long-lived case has never been measured.

The failure is concrete. `StaleCountsAfterADomainChange` in the test file
builds a workload switch on the engine's own policy port. Phase A routes
`{0,1}` and `{2,3}`. Phase B routes `{4,5}` and `{6,7}`, which four slots
hold entirely. The engine's LFU keeps the two stale phase-A experts and
evicts phase B's own experts on every pass until B's counts outvote A's,
which takes about 50 passes. LRU and a decayed LFU both settle within a few
passes. That is a synthetic case built to show the mechanism, not a
measurement of how often real routing triggers it. The measured domain tilt
(Jaccard 0.455 between coding and general hot sets, same page) says real
routing changes enough for the question to be worth asking.

jitllm reached the same place from the other side. Its host pager evicts LRU
("routing is not a cyclic scan", `read.go` `evictOne`), and its ROADMAP
item 12 lists ds4's decayed-frequency eviction as unbuilt, to be priced "by
replaying real routing traces at identical byte budgets, including prompt,
decode and tool-result transitions". ds4 counts misses as well as hits, and
the engine already does.

The experiment: capture two or more requests of different kinds on one
install. Replay them back to back with `--half-life` swept, and read column
`#2` onwards. Change the policy only if a decayed row beats `lfu (engine)`
by more than the interleaved-pair noise, and recovers most of the gap to
`oracle`. Any change ships behind a seam with greedy and sampled byte
identity, because a new policy changes slot contents and not outputs. Slot
order stays router ranking order regardless.

### 2. One slot budget across all layers

jitllm keeps every expert page of every layer in one byte budget with one
LRU. This engine gives each layer `--expert-cache-slots` slots and no more.
The two only behave the same if layers are equally hard to cache, and they
are not. `docs/EXPERT_ROUTING.md` measures the 95%-mass set at 48 to 96
experts per layer on the real Gemma 4 install, so a fixed per-layer count
over-provisions concentrated layers and starves spread ones.

`ThePoolLendsSlotsAcrossLayers` shows the mechanism. A layer that only ever
routes one expert lends its idle slots to a layer cycling more experts than
its own share, and the pooled cache misses only first touches.

The replay prints `pool lru` and `pool oracle` at the same total slot
count. Read them against the per-layer rows with two caveats:

- **Same bytes only at a uniform stride.** A mixed sub-4-bit install
  (ROADMAP Phase S) has per-layer strides, and a pool there would need byte
  accounting rather than slot counting.
- **It is a larger change than a policy.** Today each layer's streamer owns
  its `AlignedSlot`s, and slot index is the binding the routed argument
  buffer uses. A pool means slots owned above the streamers and a plan that
  can take a victim from another layer's set. The batched prefill's
  `union(M) <= slot_count` sub-batching would then bind to a per-layer share
  that moves. Price it with the replay before costing it in code.

Mapped residency is unaffected: it has no slots.

**Replayed 2026-10-17, and closed.** The replay ran four qwen4exp captures
(512 experts, top-10, uniform stride) as one session. `pool lru` came in
below per-layer decay at 16, 32 and 64 slots (44.0, 60.2 and 74.9% against
50.4, 62.5 and 75.1%). Against per-layer LRU it trails by 4.4 points at 16
slots and is within 1.8 points from 32 up.

`pool oracle` sits 1.2 to 6.6 points above the per-layer oracle, so pooling
does raise the ceiling. No online pooled policy priced here gets near it,
and the change is larger than a policy. These are slot hit rates, which the
same day's decay A/B showed can mislead. The table is in
`docs/EXPERT_ROUTING.md`, "The oracle and pool rows for qwen4exp". One
family only, and a pooled decayed LFU was not replayed.

### 3. Cold reads want queue depth, and the tuning here is warm

jitllm's measurement, on Linux NVMe with a cold cache (PL 15r): one
`ReadAt` at a time reaches 1.53 GiB/s, 4 concurrent 2.49, 8 concurrent
3.90, 16 concurrent 4.70. An mmap fault reaches 1.28 to 2.22, and
`MADV_WILLNEED` recovered nothing at 1.31. Their conclusion is that a cold
read's cost is per REQUEST, not per byte.

This engine's `POOL_THREADS = 8` and `MISS_READ_CHUNK_BYTES = 840 KiB`
(ROADMAP dead end 5, `.claude/docs/modules/streaming.md` Gotchas 4 and 5)
were tuned where the read is a page-cache memcpy, which is a bandwidth
problem. Cold, the same read is a queue-depth problem, and those optima need
not carry over. Apple SSDs are not Linux NVMe, so jitllm's ladder is a
reason to measure, not a number to expect.

The seams for the condition already exist:
`TURBOSPARK_EXPERT_DISK_IO=1 TURBOSPARK_EXPERT_NOCACHE=1` after
`sudo purge`, read through the `expert bytes` phase row. Both constants are
`const`s, so the sweep is one build per value (8, 12, 16 threads crossed
with two chunk sizes). Run every bypassed arm consecutively after one purge,
per Gotcha 8's rule that the pair cannot be interleaved. A win here would
matter to `auto` mapped residency's cold first prefill and to
memory-tight machines. It would not touch warm decode.

**Measured 2026-10-17, page cache on.** On the disk-bound qwen4exp from
the T7, 16 threads changed nothing, and whole-expert reads at 16-32 threads
cut expert io 5% with decode inside run-to-run noise. Per-layer queue depth
is set by misses per layer (about 2.8 at the `auto` 96 slots), not by the
pool, so jitllm's ladder cannot be climbed one layer at a time
(`docs/EXPERT_ROUTING.md`, "Read-pool depth"). The purged `F_NOCACHE` arms
were not run.

The `MADV_WILLNEED` row is a Linux measurement. It is still evidence about
`docs/EXPERT_RESIDENCY.md`'s unbuilt `madvise(MADV_WILLNEED)` item and the
`TURBOSPARK_MAPPED_DEMAND_PREP=advice` experiment: expect little from advice
alone, and measure before building more of it.

## Already here

These jitllm mechanisms are the same decisions this engine made earlier,
usually with its own measurement. They are listed so they are not re-proposed
as imports:

- **One contiguous gate+up+down region per expert, page aligned**
  (`format/jlm/write.go` `layoutOf`). Here: `packed_experts/layer_NN.bin`
  with per-expert offsets (`stream_layout.rs`).
- **pread over mmap for cold reads, and parallel reads per layer**
  (`read.go`, `ReadThreads = 16`). Here: ROADMAP dead end 2, and Gotchas 4
  and 5's per-chunk pool. jitllm never maps container weights at all. This
  engine also has the warm mapped arm (`docs/EXPERT_RESIDENCY.md`), which
  jitllm did not measure.
- **The shared expert computed while the routed read runs** (`moe.go`).
  Here: `TURBOSPARK_SHARED_CB` (Gemma 4) and
  `TURBOSPARK_QWEN_SHARED_READ_OVERLAP`.
- **Pin before read, invalidate on refill** (`read.go` `Lease`/`Hold`, PL
  15v's stale-frame bug). Here: Gotcha 6's invalidate-before-read.
- **Prefill reads each expert once per chunk.** Here the batched routed
  prefill reads a sub-batch's union in one pooled round
  (`moe_prefill_pipeline.rs`). jitllm's expert-major prefill pins and reads
  one expert at a time (`moe.go` `moeBatchFrom`), which is the weaker of the
  two.
- **Reduction order.** jitllm's host sums experts in ascending id, so its
  expert-major prefill and its decode agree bit for bit, while its device
  sums in selection order. The two paths are gated by an error band, not
  identity. This engine's one rule (slot order is router ranking order,
  AGENTS.md) covers decode and the batched prefill pair alike, which is
  stricter. Keep it.

## Independent confirmations of closed decisions

- **Cross-layer router prefetch.** jitllm's probe names 60.5% of the next
  block's experts (chance 1.8%), close to this engine's 70.6% PILOT recall.
  The prefetch built on it made decode 2.2x slower (3.87 to 8.35 s/token),
  because speculative reads queued ahead of demand reads and the wrong 40%
  spent disk bandwidth. Once reads went parallel it was neutral, and it is
  off by default (PL 16c). That was a disk-bound CPU host, the regime where
  `docs/EXPERT_ROUTING.md` says this engine's warm-only verdict might
  reverse, and where its PILOT sweep is still owed. One outside result is
  not that sweep, but it is a reason to expect the cold answer to be no as
  well.
- **Cross-token union prefetch.** A union over the last W tokens reaches
  98.5% hits at W=16 for 5.02x the bytes, and "loses worse the better it
  predicts" (PL 15t). This is the same bandwidth-versus-coverage trade as
  the `PILOT_K` table.
- **Releasing expert pages after use.** On a cold Qwen3-30B-A3B, jitllm went
  from 26.0 to 13.9 s and from 19.4 to 2.1 GB RSS. That is the memory axis
  this engine's mapped arm already reports, measured the other way round.

## Not transferable

- **Moving layers between devices mid-conversation, and routed experts on
  the CPU while attention stays on the GPU** (`moveblocks.go`,
  `streamBank.runHost`). Both exist to avoid a PCIe crossing. On unified
  memory there is no link to avoid, and CPU and GPU draw on the same DRAM
  bandwidth, which is what decode is bound by. jitllm's own Metal rows for
  hybrid experts are "untested".
- **`O_DIRECT`.** jitllm uses it on Linux only, and on Darwin deliberately
  reads buffered (`direct_other.go`). The Linux seam here
  (`crates/streaming/src/linux_uring.rs`) is the analogue, and its first
  Linux run is recorded in that file's header.
- **`kern.memorystatus_level` x `hw.memsize` x 0.8 as the budget**
  (`engine/sched/budget_darwin.go`). That is a number that moves from one
  second to the next. This engine budgets from physical memory minus a fixed
  reserve on purpose, so that two opens are comparable, and reads the kernel's
  pressure verdict only to watch, never to size (`docs/LOAD_GUARD.md`). One
  jitllm finding does carry over as a caution: a budget one frame short of
  holding the working set is a cliff, not a slope (olmoe, 47 to 12.58
  tok/s when KV took one frame, PL 15t).

## The Neural Engine work (phhusson)

This is the only part of the three sources with Apple-specific compute.
None of it is MoE. There are two routes. `main-npu` is the general one and
takes ordinary quantized checkpoints, while `apple/bonsai-pq20` serves one
repacked 2-bit encoding.

### `main-npu`: whole matmuls on the ANE, conversions on Metal

**What it does.** A ggml backend (`ggml/src/ggml-ane/ggml-ane.m`) takes
contiguous two-dimensional `GGML_OP_MUL_MAT` and nothing else. K and M must
be multiples of 32 from 32 to 16384. N, the token count, must be a multiple
of 32 from 32 to 4096, so single-token decode never offloads. Each shape
gets one MIL program, compiled on first use through the private
`_ANEInMemoryModel` and cached for the backend's lifetime. Weights and
activations are IOSurface request inputs, so one program serves every layer
of that shape.

The program cuts K into 2048-element slices and adds the partial products in
F16. The ANE runs each offloaded matmul whole, and Metal does three
conversions around it:

- **Weights.** F16 weights go to the ANE as stored. A quantized weight is
  dequantized into one F16 scratch buffer just before the ANE reads it. Any
  type Metal can copy to F16 qualifies, and the branch added the Q4_K and
  Q6_K copies for this (`GGML_ANE_MTLQUANTS` narrows the set).
- **Activations.** Each token is scaled by a power of two chosen from its
  largest magnitude, then cast to F16. The ANE flushes FP16 subnormals, and
  without the scale the mean KL against Metal was about 0.406.
- **Output.** The ANE's F16 `W * A^T` is transposed, unscaled, and widened to
  ggml's F32 layout.

Ordering uses `MTLSharedEvent`s bridged to `_ANESharedWaitEvent` and
`_ANESharedSignalEvent` through the private
`-[MTLSharedEvent IOSurfaceSharedEvent]`, so no handoff waits on the CPU.
There is one weight scratch, so the next matmul's dequantization waits until
the ANE has finished reading the previous one.

**Which matmuls go.** A per-matmul gate decides. A matmul offloads only at
`GGML_ANE_MIN_MACS` or more (default 6e9) and `GGML_ANE_MIN_INTENSITY` or
more (default 45 MACs per estimated byte moved, scratch traffic included).
The defaults are calibrated on an M4, and the README's advice for another
machine is to rerun `scripts/ane-sweep.py` and reset them. The sweep times
each shape on the ANE and on Metal alone, and reads modeled energy from
IOReport. At the defaults a 4096 x 4096 projection needs a chunk of at least
384 tokens.

**What it reports** (not reproduced here):

- README, M4 Mac mini 16 GB, a Qwen 3.5 9B distill at Q4_K_M. Adding the
  ANE takes pp4096 from 202 to 258 tok/s (+28%). Modeled energy goes from
  2612 to 1043 J. The README says to compare those joules only with each
  other, because they include warmup and repeat loops. The fork's meter,
  `joules.py`, reads IOReport's Energy Model across all processes rather
  than powermetrics, so do not set them beside `docs/POWER_BASELINE.md`.
- `docs/backend/ANE.md`, M4, LFM2.5-2.6B F16, pp4096 at microbatch 2048:
  788.52 +/- 64.11 tok/s against 653.84 +/- 0.39 for Metal alone (+21% on
  the mean). The default microbatch 512 showed no gain. One 4096-token
  passage gave mean KL 0.000053 and 100% top-token agreement against Metal,
  which that doc calls a limited check.
- The README calls the datapath bf16. The MIL is `fp16` and the device
  describes itself as "Apple Neural Engine (F16)", so read it as F16.

### `apple/bonsai-pq20`: a row split for one encoding

**What it does.** The branch runs prefill `GGML_OP_MUL_MAT` through the same
private framework. Its MIL comes from coremltools at build time, with the
same per-shape programs and IOSurface weight inputs. The ANE takes the
leading 65% of a matmul's output rows and Metal the rest, concurrently.
Weights are rewritten once at load into a planar layout both devices read,
because the ANE has no bit operations and decodes 2-bit codes with
arithmetic. Decode stays on Metal.

**What it reports** (committed result JSON, M4 16 GB, macOS 26.6.2, not
reproduced here): Ternary-Bonsai-2-27B PQ2_0 pp4096 goes from 58.73 to
115.68 tok/s at the 0.65 split. The ANE alone reaches 85.41, and tg32 drops
3.6%. Over 736 checked matmuls the KL is at most 1.01e-6.

### Both routes

**Why it is not MoE work.** No branch routes `MUL_MAT_ID` to the ANE, and
`main-npu`'s backend refuses anything but a 2-D `MUL_MAT`. The README calls
expert layers "(much) harder". The one real attempt, on AMD
XDNA (`amd-npu-gguf`), offloaded padded 512-row expert jobs and measured
no gain (289.02 to 290.26 tok/s), while slowing the GPU's own matmuls 1.35
to 1.50 ms per layer. Fixed-shape programs and variable per-expert token
counts do not fit. On an MoE install here, the realistic ANE targets would
be the dense parts of prefill: attention projections, GDN in-proj, and the
shared expert.

**Where it would touch this engine.** `main-npu` widens this. Metal
dequantizes to F16 before the ANE sees a weight, so the ANE program does not
depend on the weight encoding. The route could therefore serve every
family's dense prefill here, not only Bonsai.

This engine has no such conversion. Its GEMV and GEMM kernels, Steel
included (`crates/gpu/src/dequant_int4_steel.rs`), dequantize inside the
matmul and never write an F16 weight matrix. A port needs a materializing
kernel per resident format, an F16 scratch, the scale and unscale-transpose
kernels, and the private-API program. The scratch adds about 4 bytes of
traffic per weight (2 written, 2 read) that Steel's fused read does not pay.
That is why the gate asks for high intensity.

Precision changes too. Steel already takes F16 activations and accumulates
in FP32. The ANE path sums K slices in F16 and flushes the subnormals the
per-token scale does not lift. So it needs the real-model quality gates any
new prefill path here needs before it takes a family's default.

`apple/bonsai-pq20` is narrower. This engine already runs the
Ternary-Bonsai-2-27B line (`docs/BONSAI2.md`). The branch's numbers are for
its PQ2_0 GGUF, so they say nothing about this engine's encoding of the
same weights. Prefill is compute-bound, so the ANE adds compute rather than
competing for bandwidth, which is why the row split beats either device
alone.

**The hardware was the open question.** Every Apple number above is from a
plain M4, whose GPU has at most 10 cores. The ANE is the same 16-core unit
on M4 Pro and M4 Max, while the GPU grows to as many as 40 cores. So a plain
M4 is the case most favorable to the ANE. On one M4 Max, the fork's mixed
arm loses to Metal alone at every gate and microbatch tried.

**Measured on an M4 Max** (2026-10-17). These are measurements of the
llama.cpp fork (`main-npu` at `689fd77`, built with `-DGGML_ANE=ON`), not of
this engine. The host is an Apple M4 Max (14-core CPU, 32-core GPU,
16-core ANE, 36 GB) on macOS 26.6.2 (25G83). It ran on AC power with no other
GPU or build job running. The model is bartowski's
`MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf`, the file behind the README's +28%
row.

Each invocation of
`llama-bench -dev MTL0,ANE/MTL0 -sm layer -ts 1/0 -ngl 99 -p 4096 -n 32 -r 5`
came after a discarded warmup invocation, and the device order flipped every
round. A cell is the pp4096 mean and sample standard deviation in tok/s.
Each covers 20 timed runs (4 rounds of 5), or 10 for the fitted gate.

| Gate (`MIN_MACS`, `MIN_INTENSITY`) | Batch | Metal only | ANE + Metal | Change |
|---|---|---|---|---|
| Default (6e9, 45) | b 2048, ub 512 | 677.5 +/- 5.1 | 301.5 +/- 5.8 | -55% |
| Default (6e9, 45) | b 4096, ub 2048 | 672.4 +/- 5.6 | 461.3 +/- 6.6 | -31% |
| Fitted (0, 206.95) | b 2048, ub 512 | 681.8 +/- 16.7 | 149.8 +/- 0.4 | -78% |
| Fitted (0, 206.95) | b 4096, ub 2048 | 687.9 +/- 5.6 | 153.9 +/- 1.2 | -78% |
| Default, refused weights on Metal | b 2048, ub 512 | 686.9 +/- 16.5 | 375.0 +/- 6.2 | -45% |
| Default, refused weights on Metal | b 4096, ub 2048 | 692.3 +/- 3.1 | 512.7 +/- 18.9 | -26% |

The sweep agrees: `scripts/ane-sweep.py` timed 105 cases and found none
where the ANE beat Metal. They span Q4_K, Q6_K, and Q8_0, 32 to 2048
tokens, and seven shapes, among them this model's 1024 x 4096 K and V
projections. Each case ran twice. The best was 0.79x (Q6_K 4096 x 12288 at 2048
tokens). At 512 tokens, shapes of 4096 x 4096 and up ran at 0.42x to 0.53x.

The fit returned `GGML_ANE_MIN_MACS=0` and `GGML_ANE_MIN_INTENSITY=206.95`,
a bound just above the highest measured intensity, so it routes nothing.
The ANE did use 3.3x to 3.8x less modeled energy per matmul at 2048 tokens.
That is the fork's IOReport meter, whole machine, and comparable only within
the sweep.

Part of the loss is the fork's loader, not the ANE. With the ANE as the
layer device, every weight the ANE refuses at load goes to the CPU instead
of Metal. That is 242 tensors: norms, conv1d weights, SSM parameters, the
full-attention K and V projections, and the output head.
`docs/backend/ANE.md` describes a fallback to peer devices, but `src/` at
`689fd77` has none. The fork's author confirmed on PR #151 that the CPU
fallback is unintended, likely a rebase regression, so the `-ot` rows in the
table above are closer to the intended fork. The load probe is a 512-token
matmul, so placement follows the gate at 512 tokens whatever the run's
microbatch.

The graph then has 4815 splits against Metal's 34, and tg32 halves (56.0 to
27.0) though no decode matmul offloads. The fitted gate refuses every
matmul, which puts nearly all layer weights on the CPU. That is the 150
tok/s row, and it says nothing about the ANE.

The last two rows are a diagnostic, not the fork as shipped. An `-ot`
override puts the refused tensors on Metal. Only the token embedding stays
on the CPU, and tg32 comes back to 55 to 56 tok/s. Prefill still loses 26%
to 45%. On this machine an offloaded matmul runs slower on the ANE than on
Metal, and the graph waits for it, so every offload costs time.

That closes the speed case for this M4 Max. The 40-core part was not
measured. It pairs the same ANE with a larger GPU, so it should do no
better, but that is an inference. The result fits the core-count reading of
the fork's +28% on a 10-core M4. M4 Pro, other models and F16 checkpoints
were not measured here. Energy is below the sweep table.


**Per-matmul energy, from the same sweep.** The meter is the fork's IOReport
model of the whole machine, so these ratios compare only within the sweep.
Each row is the median over the 21 cases at that token count:

| Tokens per matmul | ANE speed vs Metal | ANE energy vs Metal |
|---:|---:|---:|
| 32 | 0.09x | 3.90x |
| 128 | 0.25x | 1.34x |
| 512 | 0.42x | 0.56x |
| 1024 | 0.54x | 0.36x |
| 2048 | 0.64x | 0.30x |

Routing every case where the ANE is cheaper (50 of 105) would take 1.58x
Metal's matmul time for 0.36x its modeled energy. Below about 512 tokens
the ANE costs energy as well as time.

**Whole-prompt energy, measured.** Same build, model and machine (2026-10-17).
Metal alone against ANE + Metal with the `-ot` placement, which is the
diagnostic arm and not the fork as shipped. Default gate, `-p 4096 -n 0`.
The fork's `joules.py` was read at `llama-bench`'s per-rep progress markers,
so load, warmup and teardown are excluded. Each arm ran 24 reps per setting
in AMMA then MAAM order, all clean, with three 60 s idle windows per setting
(about 2.5 W).

| Microbatch | Arm | s per prompt | J per prompt, idle-subtracted | Average W |
|---|---|---:|---:|---:|
| 512 | Metal | 6.00 | 335 | 58 |
| 512 | ANE + Metal | 10.67 | 160 | 17.5 |
| 2048 | Metal | 5.99 | 339 | 59 |
| 2048 | ANE + Metal | 8.25 | 125 | 17.7 |

On the mixed path a 4096-token prompt takes 1.38x to 1.78x as long. It
uses 0.37x to 0.48x the modeled energy. This is modeled energy, not wall
power, so compare the arms only with each other.

Metal's energy is almost all GPU (about 305 of 350 J). On the mixed path the
GPU draws 35 to 57 J and the ANE 38 to 40 J. Raw logs:
`~/models/ane-probe/results/energy/` on the test machine.

**Why a concurrent split might still win here.** phhusson's design note
([llama.cpp discussion #29861](https://github.com/ggml-org/llama.cpp/discussions/29861))
names three ways to share work. Layer split gives the NPU whole ops. Tensor
split gives the NPU and the GPU parts of one matmul's rows at once. Mode
split uses the NPU for prefill and the GPU for decode. His large wins came
from tensor plus mode split. Prefill went from 60 to 115 tok/s on the
base-M4 Bonsai branch, and from 150 to 330 on a Ryzen NPU.

`main-npu` does layer and mode split only, so an offloaded matmul wins only
when the ANE alone is faster. A row split adds the ANE's rate to the GPU's
instead. On this M4 Max it could beat Metal alone, although the ANE is the
slower device. For a quantized model the F16 weight conversion also runs on
the GPU and would eat into that. Phase 0 below measured it against Steel per
matmul, conversion included, at 1.10x to 1.77x faster.

**When each path is the better choice:**

| Situation | Better | Basis |
|---|---|---|
| Pro or Max GPU, speed | Metal alone | Measured here: 1.35x to 2.25x faster |
| Base chip (8 to 10 GPU cores), long prompt, microbatch 2048 | ANE + Metal | The fork's figures, not reproduced: +28% (9B Q4_K_M), +21% (2.6B F16), no gain at microbatch 512 |
| Battery or background prefill, 512+ tokens per batch | ANE + Metal | Measured here: 0.37x to 0.48x modeled energy per prompt, at 1.38x to 1.78x the time |
| Decode, or short prompts | Metal alone | No matmul under 32 tokens offloads, and small batches cost more energy on the ANE |
| Max GPU, concurrent row split, 512+ tokens per matmul | ANE + Steel split | Phase 0, per matmul and not end to end: 1.10x to 1.77x faster (1.24x or more in 8 of 10 cases) at 0.65x to 0.74x the modeled energy |
| Max GPU, dense qwen38 prefill at a 2048-row micro-batch | ANE + Steel split of gate and up (an uncommitted prototype) | Phase 1, end to end on one 2,940-token prompt: 1.142x faster at 0.896x the modeled energy, greedy output unchanged |

**A parked option, reopened by Phase 0.** Porting `main-npu` as written is
still not worth it. It needs private API and does not win on speed here.
Phase 0 below answered both for one route. Public Core ML runs the same
matmul on the ANE, and a concurrent split with Steel beats Steel per matmul.
That route still needs a per-format F16 conversion kernel and new precision
gates.

The fork's energy result made a low-power prefill mode the strongest case. It
used about half the energy for a prompt 1.4x to 1.8x slower. It still owes wall
power (`scripts/power.sh`) and a product call on that trade. This engine
already uses the ANE the supported way, through Core ML, for small
fixed-shape models where it wins (`docs/COREML.md`). Reopen the LLM prefill
case on any of these:

1. A base-chip target where `ane-sweep.py` shows the ANE ahead of Metal.
2. A low-power prefill mode whose whole-prompt energy beats Metal by enough
   to pay for the slower prompt.
3. A concurrent row-split prototype that beats Metal alone on a Max. Phase 0
   met this per matmul, and Phase 1 end to end for gate and up alone. What
   is left is the default call, and Phase 1 lists what it would owe.

**What it costs.**

- Private classes and selectors, validated on one machine and one OS build
  ("any macOS update can break it"). `main-npu` was tested on one M4 under
  macOS 26.6.2.
- Per-shape compile caches outside the app's control. aned's grew to 58 GB
  in a day in `h3.c-ane`, and `main-npu` writes its compiled models under the
  system temporary directory. A shape's first use includes its compile.
- In `main-npu`, a compiler failure fails the graph rather than falling back
  to Metal.
- Serialized ANE calls, and an input-binding order that misbinds past 8
  inputs.
- An FP16-only datapath that needs scaling guards. `main-npu` also sums its
  K slices in F16, which can round or overflow where an F32 path would not.
- On the ANE, child IOSurface offsets above 4 GiB wrap modulo 4 GiB and an
  exactly 4 GiB parent is rejected. `main-npu` therefore wraps each tensor in
  its own page-bounded surface over the parent's memory.

The LICENSE here reads "Clean-room implementation", so the forks' code
should be re-derived from the documented mechanism rather than copied, even
though MIT would allow copying. Treat the work as a separate go/no-go
decision with a dense-prefill scope, which no longer has to include private
API. If it is taken, pair `main-npu`'s Metal conversion with Phase 0's row
split. Then the ANE never sees the weight encoding, which
`apple/bonsai-pq20` ties to one. It is not a streaming change.

### Phase 0 in this engine: Core ML matmuls against Steel

Measured 2026-10-18 on the M4 Max above. This is a standalone probe, not a
runtime change. It asks two questions at the qwen38 (Qwen3.8-27B)
projection shapes. Can the public Core ML API run `main-npu`'s matmul on
the ANE? And does that matmul, alone or split with Steel, beat Steel?

The probe sources, raw logs, and summary are in `~/models/ane-phase0/` on
the test machine. None of it is in the repository.

**The program.** One Core ML ML program per shape and token count computes
`W * A^T`. Both operands are F16 inputs, so one compiled program serves
every layer of that shape. K is sliced at 2048 and the partial products are
summed in F16. That is `main-npu`'s program shape, re-derived from its
description.

It was built with coremltools 9.0 (the pin in
`crates/coreml/tools/gen_coreml_fixtures.py`) for macOS 15 at F16 compute
precision. The inputs and the output (through
`MLPredictionOptions.outputBackings`) are IOSurface-backed `MLMultiArray`s.
Metal reads and writes them in place, so nothing is copied. No private class
or selector is used.

**Placement.** The `coreml_plan` example (`crates/coreml`) with
`--units ane` placed every op on the ANE for all five shapes at 512 and 2048
tokens. That includes gate and up (M = 17408) and down (K = 17408), above
the 16384 maximum `main-npu` accepts. The IOReport ANE channel drew energy
during every prediction, so the ANE ran them.

**The arms.** The weights are synthetic INT4 in this engine's layout (group
64, BF16 scale and bias), made with the recipe in
`crates/gpu/tests/qwen38_int4_prefill_gemm_probe.rs`.

- **Steel.** The engine's own `dequant_int4_steel.metal`, one dispatch over
  all the tokens.
- **Whole ANE.** Metal dequantizes the weights to an F16 scratch and scales
  each token by a power of two. The ANE runs the program, and Metal
  transposes and unscales the F16 output. All three steps are in the time,
  and every handoff waits on the CPU (no shared events).
- **Split.** The ANE runs the first 47% of the rows (rounded to 64) through
  the same steps. Steel computes the rest at the same time. The conversion
  must run first and alone: committed beside Steel's buffer, it delayed the
  ANE, and the split ran slower than Steel.
- **Core ML only.** The prediction alone on static inputs, a diagnostic.

**Correctness.** Against Steel, the ANE's worst normalized row error was
0.00065 to 0.00148, inside the probe's 0.005 bound. Against an FP64 CPU
reference on sampled rows, the ANE was 0.0006 to 0.0012 off and Steel about
0.0004. Synthetic weights have no real activation outliers, so this is not
a quality gate.

**Method.** Energy is the sum of the 12 top-level IOReport Energy Model
channels, read at a handshake, so model load, compile, and warmup are
excluded. Each shape and token count got a 20 s idle window. Then each arm
ran for about 3 s, in the order Steel, whole ANE, split, Core ML only, then
reversed. The two rounds' times agree within 2%.

Each arm waited for about 10 s with no build or test process, GPU
utilization under 35%, and AC power. A watcher flagged any build that
started during a window. The three configurations with a flagged reading
were rerun clean.

Time is the ANE arm's ms over Steel's, and speed is Steel's ms over the
split's. Energy is idle-subtracted modeled energy per matmul, over Steel's.

| Shape (M x K) | Tokens | Steel ms | Whole ANE time | Whole ANE energy | Split speed | Split energy |
|---|---:|---:|---:|---:|---:|---:|
| gate, up (17408 x 5120) | 512 | 8.34 | 1.94x | 0.40x | 1.31x | 0.74x |
| gate, up (17408 x 5120) | 2048 | 32.78 | 1.30x | 0.26x | 1.77x | 0.67x |
| down (5120 x 17408) | 512 | 8.42 | 1.87x | 0.45x | 1.10x | 0.72x |
| down (5120 x 17408) | 2048 | 32.45 | 1.35x | 0.23x | 1.52x | 0.65x |
| attention q and gate (12288 x 5120) | 512 | 5.88 | 1.91x | 0.38x | 1.26x | 0.73x |
| attention q and gate (12288 x 5120) | 2048 | 22.95 | 1.33x | 0.25x | 1.73x | 0.66x |
| GDN qkv (10240 x 5120) | 512 | 4.95 | 1.94x | 0.35x | 1.24x | 0.73x |
| GDN qkv (10240 x 5120) | 2048 | 19.18 | 1.35x | 0.25x | 1.71x | 0.66x |
| o_proj, out_proj (5120 x 6144) | 512 | 3.06 | 2.12x | 0.42x | 1.10x | 0.73x |
| o_proj, out_proj (5120 x 6144) | 2048 | 11.52 | 1.44x | 0.25x | 1.53x | 0.65x |

Attention K and V (1024 x 5120) were not run.

**Go/no-go.** The bars were set before the pass.

- **Split, at least 1.1x Steel's speed: go.** It passes 9 of 10. Outside
  down and o_proj at 512 tokens it runs 1.24x to 1.77x. Down at 512 misses at
  1.099x, and o_proj at 512 passes at 1.103x. It costs 0.65x to 0.74x
  Steel's energy everywhere.
- **Energy mode, the whole matmul at no more than 0.6x Steel's energy within
  1.4x its time: go at 2048 tokens only.** Four of five shapes pass, at
  0.23x to 0.26x the energy and 1.30x to 1.35x the time. o_proj misses at
  1.44x. At 512 tokens every shape fails on time (1.87x to 2.12x), at 0.35x
  to 0.45x the energy.

At 512 tokens the split's ANE half finishes last: 5.6 ms against Steel's
4.3 ms at gate and up. A smaller ANE share should help there. At 2048 the
halves finish within 3% of each other for gate and up, q and gate, and GDN
qkv. Down and o_proj leave the ANE about 14% behind.

Whole-matmul offload behaves like the fork's sweep: slower than the GPU, and
cheaper in energy from 512 tokens up. The split is what changes the answer.
It adds the ANE's rate to the GPU's instead of replacing it.

**What it does not show.**

- It is per matmul, not end to end. The dense qwen prefill feeds Steel 64
  tokens at a time (`crates/runtime/src/families/qwen/prefill.rs`), so the
  runtime never issues a matmul this wide. In a quick ungated, unmetered
  pass, the same tokens in 64-token dispatches took 1.05x to 1.38x one-shot
  Steel's time. Part of any end-to-end gain would come from wider micro-batches
  alone. The split's 1.24x to 1.77x is against one-shot Steel at the same
  width. A metered follow-up did not reproduce that penalty (1.03x to 1.11x)
  and found wider micro-batches worth at most 3.7% end to end. See the next
  section.
- The split's Steel rows land in a separate buffer. Writing the ANE rows into
  the shared output at their row offset is not measured.
- Shapes are static, so each shape and token count is its own compiled
  program. Compile and load time was excluded and not measured.
- Synthetic weights, modeled energy (not wall power), 3 s arms, one machine.
  A steady 17% to 26% GPU load from outside the probe, likely the window
  server, is why the gate allowed 35%. It faded when the display slept
  partway through. Idle power fell with it, from about 3 W to as low as
  0.5 W, so compare raw energies only within a configuration.

**Phase 1, if taken.** An opt-in split prototype on the public Core ML API,
off by default. Steel stays the default. It needs:

- ANE micro-batches of 512 or 2048 tokens, with Steel taking the tail.
- The ANE rows written straight into the shared output.
- Per-shape program builds at first use, with their load cost measured.
- The real-model quality, memory, and paired prefill speed and energy gates
  any new prefill path here needs (`docs/BATCHED_PREFILL.md`), plus wall
  power.

Energy mode would be a second opt-in, for long prompts on battery. Phase 1
below took the split for gate and up.

### Micro-batch width and the prefill split, end to end

Wider micro-batches run correctly at every width from 128 to 2048 rows with
no new kernel, but they barely speed prefill up. The best width, 256, is
3.7% faster than the default 64. Steel is already compute-bound at 64 rows.
INT4 projections hold 71% to 73% of prefill GPU time, which bounds what any
ANE split can buy.

Measured 2026-10-18 on the same M4 Max (32-core GPU) with the real
`~/models/qwen38-27b.gturbo`. These are readings, not frozen rows in
`docs/BENCHMARKS.md`. Nothing below changed a default, and the measurement
seams were never committed.

**Method.** The prompt is the frozen `long-synthesis` text as one user
message through the chat template, 2,940 tokens, greedy. Each run is its own
process. A discarded warmup round comes first, then interleaved rounds that
alternate direction. Before every run the driver required AC power, no build,
test, or GPU process from another session, and GPU utilization under 10% for
5 s. A watcher flagged any overlap, and no timed run had one.

GPU time is the sum of each micro-batch command buffer's GPU interval. Width
came from a measurement-only env var read where the scratch is allocated. The
op-class split came from a second env var that dropped one class from the
Steel prefill arm. A class's cost is the drop in GPU time. Scripts, raw logs,
and the patch are in `~/models/qwen38-prefill-width/` on the test machine.

**Width.** Prefill tok/s is prompt tokens over prefill wall seconds, mean and
sample standard deviation. The first five rows are one session of 3 rounds,
the last two another session of 2 rounds.

| Width (outer chunk) | Prefill tok/s | GPU s |
|---|---:|---:|
| 64 (128), the default | 139.03 +/- 0.15 | 20.632 |
| 64 (512), control | 138.77 +/- 1.03 | 20.666 |
| 128 (128) | 142.49 +/- 0.64 | 20.141 |
| 256 (256) | 144.13 +/- 0.98 | 19.913 |
| 512 (512) | 141.70 +/- 1.31 | 20.249 |
| 1024 (1024) | 139.73 +/- 4.13 | 20.555 |
| 2048 (2048) | 139.00 +/- 3.02 | 20.604 |

From 512 up the gain fades and the spread grows. Across sessions, 512's GPU
total ran 20.04 to 20.59 s, against 20.60 to 20.65 s at 64. CPU encoding is
0.33 to 0.38 s of a 21 s prefill, so prefill is GPU-bound at about 7.0 ms of
GPU per token.

**Where prefill GPU time goes.** Three rounds per arm, each spread under
0.05 s.

| Class | Width 64 | Width 512 |
|---|---:|---:|
| INT4 projections (MLP gate/up/down, GDN qkv/z/a/b/out, attention q+gate/k/v/o) | 15.09 s, 73.2% | 14.27 s, 71.2% |
| GDN recurrence (conv prefill, conv tail, delta rule) | 1.42 s, 6.9% | 1.37 s, 6.8% |
| Attention (GQA rows partial) | 0.46 s, 2.2% | 0.45 s, 2.3% |
| Everything else (norms, RoPE, gates, residuals, embedding, head) | 3.63 s, 17.6% | 3.94 s, 19.7% |
| Total | 20.60 s | 20.04 s |

A second method agrees. A scratch test timed every projection that one
2,940-token prompt issues, at the real shapes and layer counts. It applied the
runtime's rule for Steel or the scalar kernel and rotated two weight copies so
nothing stayed cache-resident. It gives 15.25 s at width 64 and 14.40 s at
512, within 1% of the knockout.

Per-micro-batch GPU time against position has a slope of about 0.1 ms per
token per 1,000 positions, which matches attention near 2%.

**Why width barely helps.** Steel's time per matrix is close to linear in
rows. Gate or up (17408 x 5120) takes 1.09 ms at 64 rows, 8.46 ms at 512, and
33.8 ms at 2048. That is 10.4 TFLOPS at 64, rising to 10.8 by 512 and flat to
2048.

Down reaches 9.8 TFLOPS at 64 and 10.7 at 512. Projection time falls from
15.25 s to 14.40 s by width 512 and stays there.

The quick pass's 1.05x to 1.38x penalty did not reproduce. Eight 64-row
matrices against one 512-row matrix cost 1.03x (gate, up) to 1.11x (o_proj),
and down cost 1.09x. Phase 0's one-shot Steel times match this run's within
1% to 6% at both token counts, so the split ratios above rest on a sound
baseline. Only the quick pass's 64-row arm read high, and why was not
isolated.

**A cost the split exposed.** GDN `in_proj_a` and `in_proj_b` are 48 rows
each, not a whole 32-row Steel tile, so they run the scalar kernel. At that
size it is latency-bound: 0.24 ms per matrix per 64-row micro-batch, for
0.24 MB of weights.

Over 96 matrices per micro-batch that is 1.07 s per prompt, 5.2% of prefill
GPU time, at every width. Moving them onto Steel (for example, alongside
`in_proj_z`, 6,240 rows together) is unmeasured.

**Quality.** The greedy md5 of 96 generated tokens matched across all widths
from 64 to 2048, over 35 runs in two sessions. `qwen38_prefill_arm_probe`
was not run, so this is weaker evidence than that probe's KL bound.

**What blocks wider widths.** Nothing functional. Steel takes any row count
as one dispatch, and so do the GQA rows partial and the GDN conv and delta
kernels. The rows partial bounds each threadgroup's staged range by its own
last row, so masked work stays small. A default change would still owe:

- Scratch, about 337 KiB per row (computed, not measured). That is 21 MiB at
  64, 168 MiB at 512, and 674 MiB at 2048.
- The `qwen38_prefill_arm_probe` KL bound, measured here only at 64. Phase 1
  below measured it at 2048 too: 1.547e-5 nats mean, inside the bound.
- The coupling to `MAX_PREFILL_ROWS`, which also sizes `scratch.x` and the
  MTP state and bounds the MTP prime-row assert.
- The noise from 512 up, and a 0.25 to 0.31 s larger non-projection remainder
  at 512 whose cause was not isolated.

**What it means for Phase 1 (modeled, not measured).** Everything except the
projections takes 5.5 to 5.8 s. Even free projections would cap prefill at
about 3.5x. Wide micro-batches cost the GPU nothing, so the ANE can get the
512- or 2048-row matmuls it needs.

At 512 rows, Phase 0's split ratios applied to this run's Steel times give
about 1.20x on projections and 1.13x on prefill. At 2048 rows they give about
1.58x and 1.36x.

That model assumes z splits like GDN qkv and that K, V, a, and b do not split.
It also assumes the split runs inside the same serial command buffer at no
extra cost. Energy was not measured here.

### Phase 1 in this engine: the gate/up split, end to end

An opt-in prototype splits one projection class, the MLP gate and up
projections, between the ANE and Steel inside the real dense qwen38 chunked
prefill. On a 2,940-token prompt at a 2048-row micro-batch, it is 1.142x
faster than Steel alone at the same width. Modeled energy falls to 0.896x.
Breaking the micro-batch's command buffer at every split costs nothing
measurable on the GPU. Greedy output is unchanged, and the KL probe stays
inside its bound.

Measured 2026-10-18 on the same M4 Max with `~/models/qwen38-27b.gturbo` and
the width sweep's prompt. These are readings, not frozen rows in
`docs/BENCHMARKS.md`. Steel stays the default, and the prototype was never
committed. Its patch (on `cd34e2dd`, width seams included), programs,
scripts, and raw logs are in `~/models/ane-split/` on the test machine.

**The prototype.** It compiles only with the `ane-split-probe` cargo
feature, so the default build and the app link do not change. It runs only
when `TURBOSPARK_PROBE_ANE_SPLIT=ane` is set, at the width the sweep's seam
selects.

- **Program.** One Core ML program per token count (512, 1024, 2048) computes
  the ANE rows of gate and of up in one prediction, `wg * a^T` and
  `wu * a^T`. All three operands are F16 inputs, and K is sliced at 2048 as
  in Phase 0. `coreml_plan --units ane` places every operation on the ANE.
  The packages carry no weights and are 12 KiB each.
- **Rows.** The ANE takes the first 8,192 rows of gate and of up, Phase 0's
  47%. Steel computes the other 9,216 rows of each straight into the shared
  output. The vendored kernel is instantiated `aligned_N`, so its `N`
  argument is only the output row stride. Passing the full row count there
  writes the slice at its column offset with no shader change.
- **Conversions on Metal.** INT4 rows to an F16 IOSurface plane, a per-token
  power-of-two activation scale, and an exact transpose and unscale into the
  output. They cost 1.43 ms of GPU per layer at 2048 rows, 92 ms per
  micro-batch. Inputs and outputs are IOSurface-backed `MLMultiArray`s bound
  through `outputBackings`, so Core ML copies nothing.
- **The break.** One command buffer per micro-batch becomes 129. Each layer's
  segment waits on an `MTLSharedEvent`, converts the previous layer's ANE
  rows, finishes that layer's FFN, and runs the current layer up to its
  gate/up input and conversions. Steel's rows follow in a buffer of their
  own. The host waits for the segment, runs the prediction, and signals the
  event.
- **Control.** `TURBOSPARK_PROBE_ANE_SPLIT=break` keeps the same buffers,
  event waits, and host loop. Steel takes every row and nothing runs on the
  ANE.

Every buffer is encoded before any is committed, the default arm's
encode-then-run shape, so overlapping encode with GPU work does not flatter
the split. A default command queue holds 64 buffers, and creating the 65th
blocks until one completes. The first version deadlocked on that, and the
feature build raises the capacity to 512.

**Correctness before the real model.** A synthetic test at the real gate/up
shape compares the split against one-shot Steel. Steel's rows are
bit-identical. The ANE rows' worst per-token normalized error is 0.00094,
against 0.00065 to 0.00148 in Phase 0. Without the per-token scale it is
0.97, so the ANE does flush the small rows.

**Method.** One process per run, a discarded warmup round, then 6
interleaved rounds that alternate direction. Before every run the driver
required:

- AC power.
- No build, test, or GPU process from another session.
- GPU utilization under 10% for 5 s.
- The CPUs at least 85% idle, and no other process above 50% CPU.

A run was dirty if any `top` sample during it showed another process above
50% CPU or all others above 150%. That rule was set before the campaign.
Dirty runs were redone in place, 5 of 29 timed runs, for Spotlight's `mds`
and for `mediaanalysisd`. One of them took 13% more GPU time than the clean
mean, which is why the rule exists.

Prefill time runs from the first micro-batch to the last, on runtime markers.
It leaves out the split's one-time program load: 0.19 to 0.20 s per process
once compiled and cached, 2.4 to 3.1 s cold. The CLI's prefill figure
includes it. Energy is IOReport's whole-machine Energy Model, read at the
same markers and idle-subtracted with the 20 s windows around each round
(0.63 to 0.96 W). It is modeled energy, not wall power.

**Results.** Width 2048, 2,940 tokens, six clean runs per arm.

| Arm | Prefill s | Speed | GPU s | Energy J, idle-subtracted |
|---|---:|---:|---:|---:|
| Steel only | 21.028 +/- 0.225 | 1.000x | 20.415 | 1034.9 |
| Break control | 20.818 +/- 0.020 | 1.010x | 20.243 | 1049.3 (1.014x) |
| Split, 8,192 ANE rows | 18.420 +/- 0.050 | 1.142x | 17.515 | 927.4 (0.896x) |
| Split, 7,168 rows for the tail program | 18.238 +/- 0.026 | 1.153x | 17.614 | 922.7 (0.892x) |

The prompt runs as micro-batches of 2048 and 892 rows. The 892-row one pads
to the 1024-token program, where 8,192 ANE rows leave the GPU waiting 229 ms
per micro-batch. A 7,168-row share for that program removes the wait. That
share was picked from earlier runs of this same prompt. So the last row is
tuned on its own evaluation prompt, and the third row is not.

The 2048-row micro-batch alone goes from 14.34 s to 12.32 to 12.34 s in the
two split arms, 1.162x to 1.164x. Per layer the prediction takes 34.3 ms and
Steel's rows take 36.1 ms, so the ANE finishes first. Average power is 50.0
to 51.4 W in every arm, so the energy saving is the time saving. GPU energy falls from 857 J to 719 to 724 J, and
the ANE adds 21 to 22 J and DRAM about 5 J.

**What the break costs.**

- **On the GPU, nothing measurable.** The break control runs within 1% of
  Steel alone, and its logits are bit-identical.
- **On the host, 32 to 44 ms per 2048-row micro-batch**, 0.5 to 0.7 ms per
  layer. In the tuned arm, 8.6 ms of that is the host noticing that segments
  finished. The GPU spends 0.5 ms waiting on late predictions and 22.7 ms
  resuming after the signals.
- **Under load it grows.** On a busier host the same counters reached 0.9 s
  per micro-batch. The prediction slowed to about 42 ms per layer, and the
  GPU waited on it. Each resume after a late signal took about 5 ms. A split
  whose ANE half finishes first does not pay that resume, so the balance
  matters more than the break.

The model above assumed the split runs inside the serial command buffer at
no extra cost, and for gate and up that holds. Steel spends 4.32 s of the
2048-row micro-batch on gate and up. Phase 0's 1.77x ratio applied to that
predicts 1.155x, and the micro-batch measured 1.162x to 1.164x.

**Quality.**

- Greedy md5 of 96 generated tokens: `86e4b4b0` in all 58 runs across both
  campaigns and every arm, the default width 64 included. The width sweep
  recorded the same md5.
- `qwen38_prefill_arm_probe`: final-position KL against the per-token
  reference over six prompts of 64 to 3000 tokens, bound 1e-3. Two
  measurement-only additions to the test set the outer chunk and add the
  break and split arms.

| Arm | Mean KL, nats | Top-1 |
|---|---:|---:|
| Default (Steel, width 64, chunk 128) | 1.893e-5 | 6/6 |
| Default, per-row attention | 2.031e-5 | 6/6 |
| Steel, width 2048, chunk 2048 | 1.547e-5 | 6/6 |
| Break, width 2048 | 1.547e-5 (bit-identical to Steel) | 6/6 |
| Split, width 2048 | 1.624e-5 | 6/6 |

The split ran on the 1000, 2000, and 3000-token prompts. The three shorter
ones fall under its 512-row floor and ran Steel. Its worst logit difference
against Steel at the same width is 0.0195.

**What it does not show.**

- Only gate and up are split. The 1.36x modeled above at 2048 rows needs
  every projection class split, and this run checks the model for one.
- One prompt and one machine. Other chips, installs, and families are
  unmeasured. A base-chip GPU is smaller against the same ANE, so its best
  share and its gain probably differ.
- Memory was computed, not measured. The planes take 600 to 630 MiB across
  the three programs, each with its own weight planes. That is on top of the
  674 MiB width-2048 scratch.
- ANE contention. Another process using the ANE at the same time was not
  tested. The busy-host campaign is the only hint, and it cut the gain.
- Failure handling. A failed prediction drains the GPU and returns an error,
  with no fallback to Steel. A non-finite ANE output is not detected.
- Wall power, sampled quality, and the memory oracle were not run.

**What a default would owe.** The split pays off on prompts long enough to
fill 512-row micro-batches. Below that it does nothing, and decode never uses
it. Before it takes any family's default on macOS it needs:

- The other projection classes split, or a measured reason to stop at gate
  and up.
- The real-model quality, memory, and paired speed and energy gates in
  `docs/BATCHED_PREFILL.md`, the memory oracle at width 2048 with the planes
  resident, and wall power.
- At least one more chip, a base M4 first, since the best share is per
  machine.
- A fallback to Steel when Core ML fails or the ANE runs slow, and a
  non-finite check on the ANE rows.
- Programs built or shipped per shape, and one weight plane shared across
  them.
- The width change's own debts, listed in the width section above.
- The app link, since IOSurface and CoreVideo join it: `make swift-app`,
  `scripts/verify-app-bundle.sh`, and the oldest-Xcode CI build.

## llama.cpp #30340

Linux dma-buf sharing so ROCm can prefill and Vulkan can decode from one
copy of the weights. Its own numbers did not beat the better single backend
on either axis (Qwen 27B IQ3_S: split pp512 107.70 against ROCm's 109.38;
tg 6.10 against Vulkan's 6.11). It has no MoE content. Apple unified memory
already gives Metal and the CPU one copy, so nothing here needs it.

One lesson transfers to any future ANE work. Reading another backend's
buffer zero-copy without a fence produced perplexity above 7,000 until the
scheduler synchronized shared producers (41.36 after). A shared buffer needs
a fence per handoff.

## Found while reviewing

The Linux `io_uring` + `O_DIRECT` read source jitllm's design invites
comparison with had never executed. Its first run, on this review's Linux
host, could not complete a read: four flag constants were off by bit
position and the CQ cursor was unmasked. It is fixed and has a parity test.
See `crates/streaming/src/linux_uring.rs`.
