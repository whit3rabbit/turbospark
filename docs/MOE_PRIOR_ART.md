# MoE prior art: jitllm, phhusson's NPU forks, and llama.cpp #30340

Reviewed 2026-10-14 for anything the routed-expert streamer
(`crates/streaming`) or the MoE layer paths could borrow. Every source was
read, none was built or run, and nothing here is a measurement on this
engine. A row below is a claim from the source until the experiment it names
has run on a Mac.

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
- [ggml-org/llama.cpp#30340](https://github.com/ggml-org/llama.cpp/pull/30340),
  dma-buf buffer sharing between ROCm and Vulkan, closed unmerged on
  2026-10-12 after a maintainer veto.

Read this page before proposing an expert-cache policy, a cross-layer slot
budget, a cold-read tuning pass, or Neural Engine offload.

## What it comes to

| Idea | Source | State here | Next step |
|---|---|---|---|
| Aging the cache's use counts | jitllm ROADMAP 12 (from ds4) | **open**: `ExpertCache` counts never decay | replay two captures back to back |
| One slot budget shared by all layers | jitllm `format/jlm/read.go` `evictOne` | **open**: every layer gets the same `--expert-cache-slots` | the replay's `pool` rows |
| An offline-optimal bound before policy work | jitllm `engine/model/route_test.go` `replayBelady` | **built here**: `scripts/expert_policy_replay.py` | run it on real captures |
| Queue depth for cold reads | jitllm `format/jlm/read.go`, PL 15r | **open**: `POOL_THREADS` and the chunk size were swept warm | re-sweep under the disk-bound seams |
| Per-expert contiguous pages, pread over cold mmap, concurrent misses, shared expert overlap | jitllm | already here | none |
| Cross-layer router prefetch | jitllm `Config.StreamPrefetch` | closed here (warm); jitllm's disk-bound result lowers the prior on the owed cold sweep | none new |
| Moving blocks between devices, routed experts on the CPU | jitllm `moveblocks.go`, `streamBank.runHost` | does not apply to unified memory | none |
| Neural Engine row-split prefill | phhusson `apple/bonsai-pq20`, `h3.c-ane` | not started; dense matmuls only, private API | a separate decision, not MoE work |
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

### 1. The LFU counts never age, and a server carries them across requests

`ExpertCache` (`crates/streaming/src/expert_cache.rs`) increments
`expert_use_count` on every request and never decays it.
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
None of it is MoE.

**What it does.** `apple/bonsai-pq20` runs `GGML_OP_MUL_MAT` for prefill on
the ANE through the private `AppleNeuralEngine.framework`. It uses
`_ANEInMemoryModel` with MIL text generated at build time by coremltools,
one program per fixed shape, and weights passed as IOSurface request inputs
so one program serves every layer of that shape. The ANE takes the leading
65% of a matmul's output rows and Metal the rest, concurrently.
`MTLSharedEvent` is bridged to `_ANESharedWaitEvent` and
`_ANESharedSignalEvent` through the private
`-[MTLSharedEvent IOSurfaceSharedEvent]`. Weights are rewritten once at
load into a planar layout both devices read, because the ANE has no bit
operations and decodes 2-bit codes with arithmetic. Decode stays on Metal.

**What it reports** (committed result JSON, M4 16 GB, macOS 26.6.2, not
reproduced here): Ternary-Bonsai-2-27B PQ2_0 pp4096 goes from 58.73 to
115.68 tok/s at the 0.65 split. The ANE alone reaches 85.41, and tg32 drops
3.6%. Over 736 checked matmuls the KL is at most 1.01e-6. `main-npu`
reports +21% prefill on an F16 2.6B, and notes that per-token power-of-two
activation scaling was required (KL 0.000053 with it, about 0.406 without)
because the ANE flushes FP16 subnormals.

**Why it is not MoE work.** No fork routes `MUL_MAT_ID` to the ANE. The
README calls expert layers "(much) harder". The one real attempt, on AMD
XDNA (`amd-npu-gguf`), offloaded padded 512-row expert jobs and measured
no gain (289.02 to 290.26 tok/s), while slowing the GPU's own matmuls 1.35
to 1.50 ms per layer. Fixed-shape programs and variable per-expert token
counts do not fit. On an MoE install here, the realistic ANE targets would
be the dense parts of prefill: attention projections, GDN in-proj, and the
shared expert.

**Where it would touch this engine.** This engine already runs the
Ternary-Bonsai-2-27B line (`docs/BONSAI2.md`). The fork's numbers are for
its PQ2_0 GGUF, so they say nothing about this engine's encoding of the same
weights. Prefill is compute-bound, so the ANE adds compute rather than
competing for bandwidth, which is why the row split beats either device
alone.

**What it costs.** Private classes and selectors validated on one machine
and one OS build ("any macOS update can break it"); per-shape compile caches
outside the app's control (aned's grew to 58 GB in a day in `h3.c-ane`);
serialized ANE calls; an input-binding order that misbinds past 8 inputs;
and an FP16-only datapath that needs scaling guards. The LICENSE here reads
"Clean-room implementation", so the forks' code should be re-derived from
the documented mechanism rather than copied, even though MIT would allow
copying. Treat it as a separate go/no-go decision about private API, with a
dense-prefill scope. It is not a streaming change.

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
