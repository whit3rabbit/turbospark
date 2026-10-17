# MoE prior art: jitllm, phhusson's NPU forks, and llama.cpp #30340

Reviewed 2026-10-14 for anything the routed-expert streamer
(`crates/streaming`) or the MoE layer paths could borrow. The `main-npu`
part was re-read on 2026-10-17, after the fork's author pointed to it in the
review of PR #151: the first pass took one number from that branch and
missed its quantized-weight path. Every source was read, none was built or
run, and nothing here is a measurement on this engine. A row below is a
claim from the source until the experiment it names has run on a Mac.

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
| Queue depth for cold reads | jitllm `format/jlm/read.go`, PL 15r | **open**: `POOL_THREADS` and the chunk size were swept warm | re-sweep under the disk-bound seams |
| Per-expert contiguous pages, pread over cold mmap, concurrent misses, shared expert overlap | jitllm | already here | none |
| Cross-layer router prefetch | jitllm `Config.StreamPrefetch` | closed here (warm); jitllm's disk-bound result lowers the prior on the owed cold sweep | none new |
| Moving blocks between devices, routed experts on the CPU | jitllm `moveblocks.go`, `streamBank.runHost` | does not apply to unified memory | none |
| Neural Engine prefill matmuls, Metal converting weights and activations | phhusson `main-npu` | not started; dense matmuls only, any quant with an F16 conversion, private API | a separate decision, not MoE work; benchmark the fork on M4 Max first |
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

**The hardware is the open question.** Every Apple number above is from a
plain M4, whose GPU has at most 10 cores. The ANE is the same 16-core unit
on M4 Pro and M4 Max, while the GPU grows to as many as 40 cores. So a plain
M4 is the case most favorable to the ANE. On an M4 Max, expect the gate to offload
fewer shapes and each to gain less. That is an inference from core counts,
not a measurement.

The cheapest test needs no code here. Build the fork, run its
`llama-bench -dev MTL0,ANE/MTL0` comparison on a Q4_K_M checkpoint, and run
`scripts/ane-sweep.py`, both on an M4 Max on AC power. A Metal-only arm that
matches or beats the mixed arm there closes the question for that class of
machine.

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
though MIT would allow copying. Treat it as a separate go/no-go decision
about private API, with a dense-prefill scope. If it is taken, start from
`main-npu`'s route rather than the row split, because it does not tie the
ANE to one encoding. It is not a streaming change.

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
