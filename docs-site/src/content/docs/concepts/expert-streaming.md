---
title: Expert Streaming and the Memory Model
description: Why MoE routed experts are never fully resident, where the ~2 GiB footprint figure comes from (and where it does not), and how the slot cache, the mapped-residency alternative, and the load guard decide what a session may commit.
diataxisType: explanation
---

A mixture-of-experts (MoE) checkpoint spends most of its bytes on routed
experts. On the real Gemma 4 26B-A4B install, 12.3 GB of the roughly 13 GB
install is the expert table (`crates/streaming/src/mapped_experts.rs`,
module header). This engine targets machines with 16 to 36 GB of unified
memory, so the design question is not "how do we load the model" but "how
do we run a model whose weights do not fit". The answer is that routed
experts are never made fully resident: a bounded slot cache holds a working
set, a `pread` streamer fills it on demand, and everything else in this
page is the arithmetic and the policy around that one idea.

That is also where the frequently quoted "~2 GiB" footprint figure comes
from. It is the size of a 16-slot expert cache on the fine-grained MoE
installs this engine runs, and it is emphatically not the size of the
model, not the KV cache, and not a number that exists at all on a dense
model. The sections below unpack each of those claims.

## Why the slot cache exists

The decode loop needs, for every token and every MoE layer, only the
experts the router selected for that token: 8 of 128 on Gemma 4, 2 of 8 on
Mixtral. Holding the whole table would pin 12.3 GB on Gemma and be
impossible on anything larger, while holding nothing would make every
expert weight a disk read on the critical path. The slot cache is the
middle term: a fixed number of pinned buffers per layer, filled with
whichever experts are hot, evicted when they are not.

The virtue being bought is a guarantee, not a speed. A pinned slot bounds
the working set exactly: the cache costs `slots x bytes-per-slot`, forever,
regardless of how the router behaves. The alternative discussed later
(mapped residency) is cheaper but gives that guarantee away to the
operating system.

The cost is measured rather than assumed. On the real Gemma 4 install, an
8/16/32-slot sweep (`docs/DECODE_BUDGET.md`, cited from
`crates/model-io/src/expert_cache_policy.rs`) shows GPU busy time flat at
10.26 / 10.24 / 9.99 ms per token while wall clock moves 25.07 to 19.54:
the whole variable is the exposed expert `pread` (7.62 to 3.13 ms) and the
GPU idling 52% to 40% of the token waiting on it. Buying cache is the only
lever on that axis. The alternatives are both measured dead ends.
Router-lookahead expert prefetch hit 7% cross-layer predictor accuracy in
the Swift benchmarks. Shrinking the experts so a miss costs less decoded
20%-smaller experts slower, at 2.0 to 2.4x the energy.

## How the slot cache works

One `PreadExpertStreamer` is opened per layer file
(`crates/streaming/src/pread_streamer.rs`). At open it:

- allocates every slot up front: one `posix_memalign` (2 MiB alignment)
  page-rounded allocation per slot, reused forever, so the decode hot path
  never allocates;
- constructs the `ExpertCache` policy object for that layer;
- checks the file is at least as long as the stream window, because a
  short file is the case where a bad layout is most likely.

Slot memory is deliberately host-side and page-aligned so a GPU backend can
wrap each slot zero-copy with `newBufferWithBytesNoCopy`
(`PreadExpertStreamer::slot_allocation`). The slot's bytes change as
experts stream through; the pointer does not.

### The plan cycle

`ExpertCache` (`crates/streaming/src/expert_cache.rs`) is pure policy: it
knows which expert each slot holds (`slot_expert`), a per-slot last-use
clock, per-expert hit counts, and which slots have speculative reads in
flight. It performs no I/O, which is what lets eviction behavior be tested
against scripted access traces with no model on disk.

A decode step drives it in three phases:

1. **Plan.** `plan_experts_cached(experts, avoiding_slots)` places every
   requested expert: already-resident experts keep their slots (hits),
   misses are assigned evictable slots chosen in eviction order. Slots
   named in `avoiding_slots` are reserved and never evicted, as are slots
   whose speculative reads are still in flight. If misses outnumber
   evictable slots, `plan_if_possible` returns `None` and `plan` panics
   with `expert cache cannot place requested misses`.
2. **Execute.** `execute_expert_cache_plan` reads every miss into its slot
   in parallel (next section), then `commit_plan` marks the miss slots as
   holding their experts.
3. **Advise.** Optionally, `advise_expert_cache_plan_misses` issues macOS
   `F_RDADVISE` kernel hints for the missed byte ranges, coalesced into
   minimal non-overlapping ranges by `coalesced_adjacent_advice_ranges`.

### Eviction, protection, and direct loads

The default policy is LFU (`ExpertCachePolicy::DEFAULT` is `Lfu`); LRU is
the alternative. LFU orders victims by per-expert hit count ascending,
tie-broken by last use, and always prefers an empty slot over an occupied
one. LRU orders by the use clock alone.

Two invariants protect correctness rather than performance:

- **Protected slots.** The caller passes `avoiding_slots` so a command
  buffer still reading a slot's bytes cannot have that slot reassigned
  mid-flight. In-flight speculative reservations serve the same role from
  inside the cache: a slot being filled from another task is reserved
  against the real plan.
- **Residency must match bytes.** `load_expert_into_slot` writes slot
  bytes outside any plan, so it drops that slot's residency record first
  (`ExpertCache::invalidate_slot`), before the read, not after. Otherwise the cache can
  score a hit on bytes belonging to a different expert: no error, no
  crash, fluent output computed from the wrong weights.
  Speculative reservations follow the same discipline on the publish side
  (`publish_speculative_reservation` fills residency only for reads that
  succeeded).

## The read path: chunks on a parked pool

The unit of parallelism is a CHUNK of one expert's blob, not a whole miss.
`MISS_READ_CHUNK_BYTES` is 840 KiB, a 4-way split of Gemma 4's real
3,358,720-byte stride. The Swift original ran one task per miss. That
silently degrades to a single-threaded copy on the common warm layer that
misses exactly once (1.3 misses per layer at 32 slots on the real 26B
install). Splitting decouples thread count from miss count: a lone miss
reads as wide as a busy layer.

The chunks run on `read_pool`
(`crates/streaming/src/read_pool.rs`), a process-wide pool of 8 parked
worker threads. The count was swept, not guessed: at 4/8/16 threads,
expert I/O read 3.97-4.03 / 3.65-3.74 / 3.70-3.76 ms per token. 8 beat 4
every round and tied 16, so the smaller pool wins on parked-thread cost.
One pool serves every layer because layers execute one at a time. `run_batch` blocks until every chunk completes, which is
also what makes its raw destination pointers and borrowed file descriptor
sound. A single-chunk batch runs inline: handing one ~800 KiB copy to
another thread and waiting is strictly slower.

One fact shapes every tuning decision here: with the install's expert
files in page cache, this `pread` is a memcpy, not disk I/O. Measured at
125 MiB per token in 5.26 ms on the real 26B (and 23.8 GiB/s on the
chunked path), far past any SSD. On a cold or memory-tight machine the
same path is genuinely disk-bound and the chunking buys much less.

Which regime a run is in is observable. `TURBOSPARK_EXPERT_DISK_IO=1`
samples physical disk bytes around each batch via
`PreadExpertStreamer::io_stats`, where zero samples means unmeasured,
never "no disk reads". `TURBOSPARK_EXPERT_NOCACHE=1` sets `F_NOCACHE` on
the blob descriptor to reproduce the disk-bound condition experimentally.
Under it, readahead advice is skipped and reported as skipped: readahead
and cache-bypass are contradictory instructions about one descriptor.

## The arithmetic that decides whether a model streams

Whether a checkpoint can use this engine at all is decided by the
granularity of its experts, not by the size of the model. The slot cache
costs:

```
slot cache bytes = slots x (sum over layers of that layer's expert stride)
                 = slots x bytes_per_slot
```

`bytes_per_slot` is the whole-model cost of ONE additional slot
(`ExpertCacheSlots::resolve` in
`crates/model-io/src/expert_cache_policy.rs`). The per-layer stride is
per-layer on purpose: a mixed sub-4-bit install is not uniform across
layers, and padding every layer to the model-wide maximum both over-writes
and over-reads.

Worked examples, at the 16 slots the benchmark protocol pins:

| Install | Experts/layer | One expert | Layers | One slot | 16 slots |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B | 128 (top-8) | ~3.2 MiB | 30 | ~96 MiB | ~1.5 GiB |
| Qwen3-30B-A3B | 128 (top-8) | ~2.5 MiB | 48 | ~120 MiB | ~1.9 GiB |
| qwen4_exp | fine-grained | 2.76 MB | 48 | 126.6 MiB | ~2.0 GiB |
| Mixtral 8x7B | 8 (top-2) | ~108.9 MiB | 32 | ~3.4 GiB | ~54.5 GiB |

That table is the whole story of the "~2 GiB figure":

- **Where it comes from.** Fine-grained mixtures with small experts and
  modest layer counts land their 16-slot cache near 1.5 to 2.0 GiB. Add
  the pinned resident core and KV and the measured peak footprints of
  these installs land near 2.1 to 2.9 GiB (the memory-oracle ceilings in
  `docs/BENCHMARKS.md`). The dominant term is the slot cache.
- **Where it does not come from.** Not from the weights: the 12.3 GB
  expert table streams from disk and is never resident. Not from KV: KV is
  a pure function of the context window and sits on top. And not on a
  dense model: a dense install has no routed experts, so
  `bytes_per_slot` is zero, every slot count describes the same empty
  cache, and there is no slot term at all.

The Mixtral row is the counterexample that proves granularity is the axis.
Mixtral is the SMALLER model by parameter count compared to Gemma 4's
class, and it cannot stream usefully at any setting: one slot costs
~3.4 GiB, 16 slots would want ~54.5 GiB of pinned memory, and raising
slots to the expert count (8) pins the entire 27.2 GiB table, which is not
streaming. `open_expert_streamers` caps the slot count at the expert
count and reports the working set when the streamer cannot get its memory,
so the failure names the arithmetic instead of reading like a leak
(AGENTS.md Gotcha 36).

### The allowlist caps residency independently of free RAM

`ALLOWED_CACHE_SLOTS` (`crates/core/src/runtime_config.rs`) is
`[8, 16, 24, 32, 48, 64, 96, 128]`. The largest cache any install can ask
for is `128 x bytes_per_slot`, however much memory the machine has. On a
fine-grained model that is a small fraction of the expert table: qwen4_exp
at 32 slots spends 3.95 GiB and holds roughly 11% of a 288-expert table.
A model can pass the fit test above and still be residency-starved by the
allowlist; that is a policy constant, not a property of the checkpoint.

:::note
Runtime configuration setters panic outside their allowed sets rather
than clamping, so a slot resolver may only ever return a value from this
list (`every_resolved_value_is_in_the_allowed_set` in
`expert_cache_policy.rs`).
:::

## How Auto resolves, and why it only climbs

`ExpertCacheSlots` has two arms: `Fixed(n)` (what every measuring harness
passes, pinning the variable) and `Auto` (size against the machine at
open, the default). Auto resolves as:

1. floor at `DEFAULT_CACHE_SLOTS` (16);
2. `free = physical - resident - HEADROOM_RESERVE_BYTES` (4 GiB held back
   for the KV cache, command-buffer churn, sampler scratch, and whatever
   else the user is running; a fixed figure because those terms do not
   scale with installed memory);
3. `budget = free x HEADROOM_FRACTION` (a quarter, so three other
   engine-sized things can run beside this one without swapping);
4. the largest allowed slot count at or above the floor whose
   `n x bytes_per_slot` fits the budget, else the floor.

**Auto may only ever climb.** The floor exists so no machine can be made
slower by the feature being on: a 13 GB install on a 16 GB machine has no
headroom by this arithmetic and gets exactly the 16 slots it always got.
An earlier draft floored at the bottom of the allowlist and would have
resolved that machine DOWN to 8, a regression for exactly the users least
able to absorb one. Concretely (from the policy's own tests): a 36 GB
machine with the 13 GB Gemma install resolves to 48 slots (4.5 GiB), a 27
GB machine lands between rungs at 24, machines at 18 GB and below stay at
16, and a coarse-grained Mixtral stays at the floor on any machine tested
up to 128 GB.

Nothing that measures ever passes `Auto`. The benchmark protocol pins its
slot count and the memory oracles and quality gates go through it, for the
same reason the power harness does not inherit an environment-sensing
power profile: when a knob has an environment-sensing default, the harness
measuring the knob is exactly the caller that must not sense. A frozen
footprint row taken at whatever the machine felt like that morning is not
a row.

## The mapped alternative: residency without a copy

The copy the streamer exists to perform is not always necessary. The MoE
decode kernels never required a slot: they read expert weights in place
from "streamer slots or any other page of memory" through the routed-blob
argument buffer, so a routed blob pointer can be
`mapped_layer_buffer + expert_offset` and the kernels do not change.

`MappedExpertLayer` (`crates/streaming/src/mapped_experts.rs`) is that
arm: one instance per layer, mapping the layer file's window read-only
and handing out page-aligned bytes plus per-expert offsets. It reuses the
same `StreamLayout` as the streamer, including the explicit per-expert
offset table (the writer need not emit dense `expert * stride` offsets,
and an assumed-uniform read of a non-uniform layout silently returns a
neighbour's blob), and it carries `ResidentBuffer`'s page-alignment shift
on every offset (dropping it reads the file's header instead of expert 0).

The obvious objection, that wrapping a multi-gigabyte mapping would pin
it, is measured false on the real install's 12.3 GB table: the `mmap`
itself charges 0.0 MiB of `phys_footprint`, wrapping all 30 layer files
charges 2.9 MiB, and reading one expert on each layer through the GPU,
about 96 MiB of fresh file-backed pages, charges a further 0.1 MiB. Clean
file-backed pages are excluded from the counted footprint whoever reads
them. End to end on the real Gemma 4 install
(`docs/EXPERT_RESIDENCY.md`): peak footprint 3,652 MiB streamed against
559 MiB mapped, decode 53.8 against 68.7 tok/s, output byte-identical on
greedy and sampled smokes. (Read that 1.28x as a ceiling: the capture
shared the machine, and contention costs the `pread` arm more than the
mapped one.)

The trade is why the streamer stays. Pages nobody is charged for are pages
the OS may evict. A pinned slot GUARANTEES a bounded working set; a
mapping hands residency to the kernel, which is the right answer on a
machine that can hold the table and the wrong one on a machine that
cannot, exactly the cold-or-memory-tight case where the streamer's read
path is disk-bound anyway. So mapped residency is a MODE
(`TURBOSPARK_EXPERT_RESIDENCY=mapped`, off by default), and it resolves
DOWN to the streamer rather than up: on a machine that cannot hold the
table, the mode must fall back, not fail.

## The dense counterpoint

Everything above is MoE-specific. On a dense install nothing streams:
there is no packed-experts region, no streamer, and no slot cache, and
`Auto` resolves to the default without even dividing (a zero
`bytes_per_slot` would otherwise divide by nothing).

The counterintuitive part is the footprint. A dense install's resident
weight mapping is largely invisible to the counted footprint: measured on
the real Mistral 7B Q4_K_M, 4.07 GiB of resident weights produced a peak
`phys_footprint` of 683.9 MiB, of which KV at 4,096 context (537 MiB) is
most of what IS counted (AGENTS.md Gotcha 40). So the counted footprint is
not the RAM requirement in either direction: a dense 7B runs in well under
a gigabyte of counted footprint while needing its weights mapped, and a
streamed MoE's counted footprint is dominated by a slot cache that is a
policy choice, not a property of the checkpoint.

## The load guard: what a session may commit

Slot-cache sizing and context sizing both budget from installed memory,
and `LoadGuard` (`crates/model-io/src/load_guard.rs`) decides how
conservative that budget is, in ordered tiers:

| Tier | Reserve | Fraction of the rest | Tight at | Refuses |
|---|---|---|---|---|
| `Off` | 0 | 1.0 | 0.9 | no |
| `Relaxed` (default) | 4 GiB | 0.25 | 0.9 | yes |
| `Balanced` | 8 GiB | 0.167 | 0.8 | yes |
| `Strict` | 12 GiB | 0.10 | 0.7 | yes |
| `Custom` | 4 GiB | 0.25 | 0.9 + hard cap | yes |

Parsed as `off` / `relaxed` / `balanced` / `strict` (`LoadGuard::parse`);
`Custom` carries a byte count so it is parsed by number, not by word.

Three properties are load-bearing:

- **`Relaxed` IS the pre-guard arithmetic.** It resolves to exactly the
  reserve and fraction every frozen peak in `docs/BENCHMARKS.md` and every
  `measured` block in the catalog was measured under. Moving the default
  would not fail anything; it would leave every published row quietly
  describing a configuration the engine no longer opens with. A test pins
  the constants rather than trusting the paragraph.
- **`Custom` caps allocated, not installed, bytes.** The ceiling is on
  counted bytes (slot cache plus KV), because exceeding memory with the
  MAPPED install is the streaming this engine is built around and costs
  throughput rather than correctness. A cap read against the install size
  would refuse a 13 GB model on a 16 GB machine, which runs.
- **The AutoFit floor is scoped to Auto.** `LoadPolicy` pairs a guard with
  `min_auto_context`, the fewest tokens an automatic resolution may land
  on. An explicit `--max-context 2048` is the user deciding how to spend
  their own machine and is stopped only when the number cannot work at
  all. The setting is named for AutoFit because it constrains the fit, not
  the user.

The guard itself is deliberately pure and portable: no GPU dependency, no
OS probe, every input a parameter. Watching actual memory pressure is a
runtime concern, not a guard concern: the decode loop samples memory
pressure during a turn and reports the worst level observed on the
completion result (`peak_memory_pressure`, with the `MemoryPressure` type
in `crates/runtime/src/power.rs`). The guard decides what may be committed
up front; the watcher reports what happened anyway.

## Limits and open items

- **The allowlist is the residency ceiling**, at 128 slots, independent of
  free RAM. Raising it is a constants change plus revalidation, not new
  machinery, but every frozen measurement taken at a given slot count
  describes that count.
- **`slots == top_k` fails in chunked prefill.** A top-8 model at exactly
  8 slots panics on the first multi-token prompt with `expert cache cannot
  place requested misses`: the prefill path's stale protected-slot
  reservation can leave zero room for the next token's misses, and the
  pipelined branch only engages at `slots >= 2 * top_k`
  (AGENTS.md Gotcha 64). Oracles run at 16 slots and cannot see it.
- **Mapped residency can be disk-bound.** On a cold or memory-tight
  machine the mapped path is genuinely slower, which is why the mode
  resolves down to the streamer rather than up.
- **`io_stats` physical bytes are sampled, not always on.** Zero samples
  means unmeasured, never "no disk reads"; read the sample count, not just
  the byte count.
- **`TURBOSPARK_READ_QOS=utility`** runs the read-pool workers on
  efficiency cores (off by default; the prior is that it loses, since the
  pool sits on the decode critical path doing a page-cache memcpy).

## Where the code lives

| File | Role |
|---|---|
| `crates/streaming/src/pread_streamer.rs` | `PreadExpertStreamer`: per-layer file, pre-allocated slots, plan/execute/advise cycle, chunked miss reads |
| `crates/streaming/src/expert_cache.rs` | `ExpertCache`: pure LFU/LRU placement and eviction, speculative reservations, advice-range coalescing |
| `crates/streaming/src/read_pool.rs` | Process-wide pool of 8 parked reader threads; `run_batch` chunk execution |
| `crates/streaming/src/mapped_experts.rs` | `MappedExpertLayer`: in-place mapped residency, the streamer's sibling |
| `crates/streaming/src/stream_layout.rs` | Expert blob offset arithmetic, including the per-expert offset table |
| `crates/streaming/src/rdadvice.rs`, `disk_io.rs` | `F_RDADVISE` hints, physical-disk-read probe, `F_NOCACHE` seam |
| `crates/model-io/src/expert_cache_policy.rs` | `ExpertCacheSlots` and the `Auto` resolution: floor, reserve, fraction, allowlist |
| `crates/model-io/src/load_guard.rs` | `LoadGuard` tiers, `GuardBudget`, `LoadPolicy` and the AutoFit floor |
| `crates/core/src/runtime_config.rs` | `ALLOWED_CACHE_SLOTS` and `DEFAULT_CACHE_SLOTS` |

Deeper measurement records live in `docs/EXPERT_RESIDENCY.md` (the mapped
versus streamed trade), `docs/EXPERT_ROUTING.md` (prefetch, a measured
negative), `docs/DECODE_BUDGET.md` (the slot sweep), and
`docs/LOAD_GUARD.md` (the guard's user-facing page).
