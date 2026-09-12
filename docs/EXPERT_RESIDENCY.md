# Expert residency: the slot cache is a copy nothing asked for

The routed experts of an MoE install can be read IN PLACE out of an `mmap`
instead of being `pread`-copied into a pinned slot. On the real Gemma 4 install
that is **84% less counted memory and 35% more decode throughput, with
byte-identical output**.

This page records what was measured, what it costs, and why the `pread`
streamer stays.

## The two paths

```text
STREAMED (default)
  packed_experts/layer_NN.bin --pread--> AlignedSlot --wrap--> MTLBuffer --> kernel
     (page cache)              memcpy      (pinned)    no copy    reads in place

MAPPED (TURBOSPARK_EXPERT_RESIDENCY=mapped)
  packed_experts/layer_NN.bin --mmap--> MTLBuffer --> kernel
     (page cache)              no copy    no copy     reads in place
```

Nothing downstream of the streamer ever wanted the copy. `crates/gpu`'s
`moe_decode.rs` header has always said the decode kernels read expert weights
in place from "streamer slots **or any other page of memory**" through the
`RoutedBlobs` argument buffer, and `RoutedBlobsBuffer::bind` has always taken
`(buffer, offset)` pairs. The slot cache exists because the streamer copies.

So the change is one expression at the dispatch site: a blob reference becomes
`(mapped_layer_buffer, expert_offset)` instead of `(slot_buffer, 0)`. No kernel
changed, no argument buffer changed, no shader changed.

## Measured

Apple M4 Max, 36 GB, macOS 26.5.2. Real `~/models/gemma4.gturbo` (Gemma 4
26B-A4B, 128 experts of 3.2 MiB over 30 layers, 12.3 GB expert table).

**RE-MEASURED 2026-08-29** on AC. The original capture (2026-08-23) predates
`--prefill-chunk` becoming the default on 2026-08-26, so it described a
configuration nobody runs; the superseded pair was 3,721 -> 606 MiB and
51.9 -> 69.8 tok/s. Both halves reproduced within drift, and the conclusion is
unchanged.

### Memory

`/usr/bin/time -l`'s `peak memory footprint` line, which IS `phys_footprint`.
Slot count `auto`, resolving to 32 on this machine -- recorded because a frozen
peak is a peak at ONE context window and ONE slot count (AGENTS.md Gotcha 58),
and the protocol's own runs pin 16 rather than 32.

| | streamed | mapped |
|---|---:|---:|
| peak phys_footprint | 3,652 MiB | **559 MiB** |
| maximum resident set size | 3,448 MiB | 432 MiB |

Two runs of each arm, and this is the reproducible half: the streamed peak read
3,652.5 and 3,652.0 MiB (0.01% apart) and the mapped 560.9 and 557.5 (0.6%).

The 3,093 MiB difference is the slot cache, and the arithmetic says so rather
than the label: `32 slots x 30 layers x 3.2 MiB` is 3,072 MiB.

### Throughput

Interleaved pairs after a discarded warmup, greedy, 400 new tokens.
Interleaved rather than batched because run-to-run spread here is wider than
many single changes (CLAUDE.local.md's standing rule) -- and on this capture
that rule is what made the number readable at all.

| pair | streamed tok/s | mapped tok/s | ratio |
|---|---:|---:|---:|
| 1 | *35.668* | 69.660 | *1.95* |
| 2 | 53.240 | 69.678 | 1.31 |
| 3 | 53.725 | 68.740 | 1.28 |
| 4 | 53.804 | 67.778 | 1.26 |
| 5 | *47.752* | 68.456 | *1.43* |
| 6 | 54.335 | 70.003 | 1.29 |

**1.28x decode**, from the four pairs in roman type: 1.31 / 1.28 / 1.26 / 1.29,
spread 3.8%. Median 53.8 tok/s streamed against 68.7 mapped.

**THE TWO ITALICISED PAIRS ARE EXCLUDED AND THE REASON IS IN THE STREAMED
COLUMN, NOT THE RATIO.** Pair 1's streamed arm reads 35.668 against a 53.2-54.3
cluster and pair 5's reads 47.752; the mapped column is flat across all six
(67.8-70.0, 3.2%). So both excluded ratios are a depressed DENOMINATOR rather
than a better numerator, which is what a ratio column alone cannot show and a
mean over all six would have hidden -- it would read 1.42x, higher than any
clean pair.

**Pair 1 is the trap this page already recorded, reproduced.** The first
measurement of this feature read 1.99x and was wrong, because its streamed arm
paid a cold GPU and a cold page cache (AGENTS.md Gotcha 20); pair 1 here reads
1.95x for the same reason, one warmup being enough for the GPU and not for the
page cache after the mapped footprint runs. The 2x figure should not be quoted.

**THE MACHINE WAS NOT QUIET AND THAT BIASES THIS UPWARD, not down.** Another
session was compiling throughout (two `rustc` at ~33%, load average ~4.5), and
the reproducibility gate this capture opened with -- three identical streamed
arms, `crates/bench` Gotcha 23 -- FAILED it at 52.2 / 46.8 / 43.8, a 19% spread.
Interleaving is what rescued it, and it licenses the RATIO rather than the
absolute rows. The direction of the residual bias is knowable: CPU contention
depresses throughput, the streamed arm does `pread` (CPU work) where the mapped
arm does none, so contention costs the streamed arm more and 1.28x is a ceiling
on the honest figure rather than a floor. Re-measure on a genuinely idle machine
before tightening it.

The mapped arm's own stability across all six pairs is a result in itself: no
`pread` means no cache-warming variance, which is the same property the phase
counters show directly below.

### Byte-identity

**A THREE-WAY comparison per sampling mode, not on-vs-off** (2026-08-29). An
on-vs-off comparison inside ONE binary cannot say the default path did not
move: both of its arms carry whatever the change did. The third arm is a
binary built from the commit before this landed, which is the only one that
can (`crates/runtime` Gotcha 22 set the precedent for the MXFP4 pair).

| arm | greedy | sampled |
|---|---|---|
| PRE-CHANGE binary | `cf23477ee3eb8fa753322c93544ee517` | `3b02cc85dccc90f8e985eb7ed525d168` |
| post-change, seam off | `cf23477ee3eb8fa753322c93544ee517` | `3b02cc85dccc90f8e985eb7ed525d168` |
| post-change, mapped | `cf23477ee3eb8fa753322c93544ee517` | `3b02cc85dccc90f8e985eb7ed525d168` |

Both modes are run, because greedy is `argmax` and `argmax` is invariant under
every monotone transform of the distribution -- it stays byte-identical to
correct through bugs that destroy sampling entirely (AGENTS.md Gotcha 16).

That identity is expected STRUCTURALLY rather than hoped for: the same bytes
reach the same kernels in the same order. The routed slots are still dispatched
in the router's own ranking in both modes, which is what keeps output stable
across residency modes for exactly the reason it is stable across slot counts
(AGENTS.md Gotcha 27).

### The mapped arm is proven to have ENGAGED

Byte-identity is necessary and NOT sufficient, and on this feature the
insufficiency is the whole hazard: a seam that silently did nothing would also
be byte-identical, and would then be measured as the streamed engine and
reported under the mapped label. That is not hypothetical --
`TURBOSPARK_ROUTED_BATCH` shipped with exactly that failure on the MoE `llama`
family (`crates/runtime` Gotcha 22).

`TURBOSPARK_PHASES=1` on the same install and prompt, 60 new tokens, settles it:

| | streamed | mapped |
|---|---:|---:|
| expert io (`pread`) | 605.6 ms (32.3% of the run) | **0.0 ms (0.0%)** |
| expert cache | 19,200 requests, 15,078 hits (78.5%), 4,122 misses | 19,200 requests, 19,200 hits (100%), 0 misses |

Zero `pread` time and a 100% hit rate are what "every expert is already
addressable" looks like from the counters, and neither is reachable by a mode
that quietly fell through to the streamer.

## The footprint result, stage by stage

`crates/bench/tests/mapped_expert_probe.rs` maps the whole 12.3 GB table as 30
per-layer files and samples `phys_footprint` at four points:

| stage | phys_footprint | delta |
|---|---:|---:|
| baseline | 31.9 MiB | |
| after `mmap` of 12.3 GB | 31.9 MiB | +0.0 |
| after 30 `newBufferWithBytesNoCopy` wraps | 34.8 MiB | +2.9 |
| after ONE GPU expert read | 174.1 MiB | +139.2 |
| after one expert on each of 30 layers | 174.2 MiB | +0.1 |

**The last row is the one that settles it.** That sweep faulted in ~96 MiB of
fresh file-backed pages across 30 separate mappings, read by the GPU, and moved
the counter by 0.1 MiB. The +139.2 above it is one-time MSL pipeline
compilation, not pages -- the sweep reuses the same pipeline and pays nothing.

This refutes AGENTS.md Gotcha 19 as it stood ("`newBufferWithBytesNoCopy` makes
Metal pin the range"). That claim was never measured, and it already
contradicted Gotcha 40 and the 2026-08-18 recommendation-engine work; two
entries disagreed with it and it survived anyway.

## What it costs, and why the streamer stays

**Pages nobody is charged for are pages the OS may evict.** The slot cache's
virtue is that it PINS a bounded working set. A mapping hands residency to the
kernel, which is right on a machine that can hold the table and wrong on one
that cannot. `crates/streaming` Gotcha 3 is explicit that on a cold or
memory-tight machine this path is genuinely disk-bound.

**A cold mapping pays its faults up front.** The very first mapped run on this
install read `prefill=21tok/74.77s` against the streamed arm's 2.53s, because
it faulted 12.3 GB in from disk. Warm, the same prefill reads 1.48-1.60s
against 0.97-1.06s -- still ~50% slower, because a page fault costs more than a
memcpy from an already-warm page cache. That per-page cost is paid ONCE; decode
over 400 tokens is where the 1.35x comes from. So the trade is a one-time
fault cost for a permanently faster steady state, and it is a bad trade for a
process that opens a model, emits ten tokens and exits.

Both of those are why this is a MODE and not a replacement, and why it must
resolve DOWN to the streamer rather than up.

## This does not contradict upstream, it extends it

`DEVIATIONS.md` records a Swift finding that bounded `pread` BEATS `mmap` for
cold experts, 2.79 against 9.88 ms, and that is why this port took the pread
path. That result reproduces here: the first mapped run on a cold page cache
prefills in 74.8s against 2.5s.

What upstream measured was COLD LATENCY. What is new is the WARM steady state
(1.35x decode, because a fault is paid once and a memcpy is paid every token)
and the MEMORY axis, which that experiment did not look at at all. Both
findings are correct and they are about different operating points, which is
why both paths ship rather than one replacing the other.

## Status

- **Implemented for FOUR of the five MoE flows since 2026-08-30**: Gemma 4
  (the original), `qwen` (`QwenGdnMoe`), `llama` (both `Llama` and
  `Qwen3Moe`, which share one flow), and `gptoss`. Each family's
  per-token routed encoder(s) carry the same fork Gemma 4's does: skip the
  plan/`pread` and read straight out of the mapping when
  `self.mapped.buffers[layer].is_some()`, otherwise the unchanged streamed
  path. `mapped_residency_refusal` (`real_forward_init.rs`) is the single
  whitelist gate, still a pure function of the family, and now admits
  `Gemma4 | QwenGdnMoe | Llama | Qwen3Moe | GptOss`. Verified on real
  installs: `~/.turbospark/models/qwen3moe.gturbo` (the `llama` flow's
  `Qwen3Moe` half, 128 experts/48 layers) and
  `~/.turbospark/models/gptoss-20b.gturbo` both reproduce byte-identical
  greedy output between the streamed and mapped arms, and
  `TURBOSPARK_PHASES=1` shows the same fingerprint Gemma 4's own capture
  does: `expert io (pread): 0.0 ms (0.0%)` and a 100% cache hit rate.
  `qwen`'s own family (`QwenGdnMoe`, e.g. Ornith 35B) has no real install on
  this machine at the time of writing (see `CLAUDE.local.md`'s artifact
  drift note), so it is verified on the synthetic fixture only
  (`crates/runtime/tests/mapped_expert_residency_qwen.rs`).
  **Only `DeepseekV4Flash` remains refused**, and it stays refused for an
  unrelated reason: compressed attention is not wired at all yet
  (`crates/runtime/src/real_forward_init.rs::validate_arch_config`), so no
  install of that family can open regardless of residency mode.
- **`TURBOSPARK_ROUTED_BATCH=1` and this mode cannot be combined, on every
  family that has a batched routed driver.** Gemma 4's own conflict lives at
  `families/gemma4/moe_batch.rs`; the same guard was added to
  `families/gptoss/moe_batch.rs` (its MXFP4 batched pair) and
  `families/qwen/moe_batch.rs` (its batched VERIFY pass, reachable only
  through a synthetic MoE+MTP fixture since no published MoE conversion of
  that architecture carries an ingestible speculative head). `llama` needs no
  such guard: it has no batched-routed-prefill driver at all, so there is no
  second seam to reconcile. Every guard binds one buffer per CACHE SLOT
  (`MoePrefillRoute::slot` indexes that array) and this mode has no slot
  cache: it has one buffer per layer plus a per-expert offset, and the
  experts to bind are the SUB-BATCH's union, which changes inside the layer
  loop. Serving both means re-binding per sub-batch rather than per layer,
  which is a change to that driver rather than a branch in it. Deleting the
  refusal was measured rather than reasoned about on Gemma 4: the run trips
  `assert!(!blobs.is_empty() ...)` in `gpu::moe_prefill_batch`'s argument
  encoder, a plain `assert!` that aborts in release, naming neither seam.
  The refusal is at the DRIVER rather than at `open` because
  `set_routed_batch_prefill` can flip that seam after open; confirmed firing
  by name on the real gptoss install (both seams named in the error).
- The DEFAULT chunked-prefill path is unaffected and needs no second arm:
  with `TURBOSPARK_ROUTED_BATCH` unset, Gemma 4's chunked driver runs its routed
  half through `encode_gemma4_layer_routed_moe` -- the same function the
  sequential decode path uses, and the one carrying the mapped branch.
- Behind `TURBOSPARK_EXPERT_RESIDENCY=mapped`, off by default, an A/B seam in
  the shape `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` already have:
  both arms must produce identical tokens, so it is a seam first and a feature
  second.
- No CLI flag yet, and deliberately: `turbospark-bench`, the memory oracles
  and the quality gates must not sense a knob that moves prefill and footprint
  by this much (AGENTS.md Gotcha 35). A flag needs the harnesses pinned first.
- Every frozen row in `crates/bench` is a STREAMED row and stands unchanged. A
  mapped row is a new row, not a re-freeze.
- **The VISION TOWER has its own mapped arm since 2026-08-30, behind its OWN
  seam, `TURBOSPARK_VISION_RESIDENCY=mapped`, never `TURBOSPARK_EXPERT_RESIDENCY`.**
  Reusing the routed variable would move the tower silently for anyone A/Bing
  routed residency, which is the exact silent-ignore failure
  `mapped_residency_refusal`'s own doc comment exists to prevent, one seam
  over. There is no per-family refusal function for it either: the tower is
  family-agnostic (any install with `arch.vision.is_active()` runs the same
  code), so the only gate is whether the tower opens at all, and
  `MappedExpertLayer::open`'s `layout.num_layers > 0` precondition is
  trivially satisfied by the tower's own `packed_vision/` layout, which is
  always exactly one "layer" of `depth` blocks.

  The shape differs from the routed case in one way that matters: the
  routed arm maps ONE `MetalBuffer` PER LAYER (`MappedResidency::buffers`
  is a `Vec`), while the tower has only one pseudo-layer, so
  `VisionTower` maps a SINGLE buffer over the WHOLE tower and
  `block::encode_block` gained a `base: u64` parameter
  (`mapped_layer.expert_offset(n)`) added to every `roles.at(role)` call, since
  a block's twelve named sub-tensor roles are offsets relative to the START of
  that one block's own blob -- true for the pread arm because
  `slot_buffers[slot]` wraps exactly one pread'd block starting at 0, and
  false for the mapped arm's buffer, which wraps every block concatenated. The
  pread call site passes `base = 0`, a no-op addition, which is what makes the
  change verifiable as byte-identical rather than merely argued.

  Landed in two steps, per this page's own rule for isolating causes: first
  the mapping alone with the per-block `commit_and_wait` unchanged for both
  arms, then a second step dropping that wait for the MAPPED arm only (the
  pread arm keeps it; it is what makes the synchronous `pread` into the next
  slot safe with no fence). The mapped arm has no pread step and no
  per-block-overwritten slot, so the wait there was pure CPU-side
  serialization; correctness rests on the same commit-order guarantee this
  page's own routed rows already rely on. The wait is NOT dropped when a
  per-block host readback is requested (`run_with_stages`'s cross-engine
  capture, or `TURBOSPARK_VISION_OVERFLOW`), since those need the GPU to have
  actually finished before reading `s.x` from the host -- both still
  `commit_and_wait` per block, unchanged.

  Verified on the synthetic fixture
  (`crates/runtime/tests/mapped_vision_residency.rs`): both arms produce
  byte-identical `VisionEmbedding` rows on the same image, AND
  `RealForwardRunner::vision_residency_is_mapped()` (`None` before the first
  image, `Some(bool)` after) proves the mapped arm actually engaged rather
  than silently falling through to pread -- mutation-checked, and the
  engagement mutation reddens ONLY that assertion while leaving the
  byte-identity one green, which is exactly the silent-fallback failure the
  accessor exists to catch. No real-install run yet; see Not done.

## Landed since: the flag, the mode-first budget, and the eviction probe
(2026-09-11, ROADMAP P1 item 3)

- **`--expert-residency auto|streamed|mapped`** exists on `turbospark-check`
  and `turbospark-server` (`model_io::ExpertResidency`, resolved by the ONE
  resolver `runtime::resolve_expert_residency`, which the open, both front
  ends' `committed_breakdown_with_residency` sizing and the startup lines all
  share so the budget arithmetic and the allocation cannot disagree about
  which mode was chosen). `Auto` defers to the `TURBOSPARK_EXPERT_RESIDENCY`
  seam when set (every mapped test and probe predates the flag and drives
  it) and otherwise resolves DOWN to streamed; under mapped residency the
  committed breakdown budgets ZERO slot bytes, which is the whole
  3-GiB-per-open difference between the two arms.
- **The eviction probe ran** (`crates/bench/tests/
  mapped_residency_eviction.rs`, real gemma4, Apple M4 Max, battery): a
  DIRTY allocation of the full 30.9 GB budget (physical minus resident
  minus a 6 GB reserve) never pushed the kernel past Nominal -- macOS
  absorbed it in compressed memory -- and the decode DURING that pressure
  read 66.2 tok/s against the 62.9 warm window (105%, within jitter), with
  +169 faults against the warm window's +0. After release, 62.6 tok/s
  (99.5% of warm). The honest reading is NARROW: on this machine, the
  strongest pressure this bounded probe can induce does not evict a HOT
  expert mapping (a decode touching all 30 layers' experts every token
  keeps them hot, and eviction targets idle clean pages), so no
  mid-decode cliff was observable. It is NOT a statement about exhaustion
  (the kernel never reached Warn) or about a COLD mapping after idle --
  the cold-prefill numbers above already cover that half. `auto` therefore
  still resolves down to streamed; flipping it up would want a probe that
  reaches real pressure (a bigger machine's workload or a lower reserve)
  and a doc update saying so.

## Not done

- `auto` resolving UP against machine memory. The eviction probe above is
  the measurement this was waiting on, and its answer on this machine is
  "no observable eviction at inducible pressure" -- which removes the
  known-cliff objection without establishing a WIN, so the conservative
  default stands until someone measures the other direction.
- `madvise(MADV_WILLNEED)` on the routed offsets, the mmap analogue of the
  `F_RDADVISE` hinting the pread path uses.
- Frozen mapped-arm memory-oracle rows for any family (every frozen row in
  `crates/bench` is still a STREAMED row, per the Status section above).
- Composing with the BATCHED routed pair (`TURBOSPARK_ROUTED_BATCH=1`), which is
  refused rather than served on every family that has one today. It needs the
  wide argument buffer bound per SUB-BATCH instead of per layer, since the
  union of experts to bind changes inside the layer loop where the slot array
  does not. Worth costing against what it buys: the two seams are the repo's
  two prefill levers and nobody has measured them together, so the win is
  unknown rather than known-small.
- A measurement of what happens when the OS evicts a mapped expert mid-decode,
  which is the failure mode this design accepts and the streamer does not have.
- A real `QwenGdnMoe` install on this machine to verify against (Ornith 35B
  was on disk when this feature was scoped; it no longer is -- see
  `CLAUDE.local.md`'s artifact inventory, which needs a re-check against
  what is actually present before the next session trusts it).
