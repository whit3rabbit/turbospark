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

MAPPED (MFERENCE_EXPERT_RESIDENCY=mapped)
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

### Memory

`/usr/bin/time -l`'s `peak memory footprint` line, which IS `phys_footprint`.
Slot count `auto`, resolving to 32 on this machine.

| | streamed | mapped |
|---|---:|---:|
| peak phys_footprint | 3,721 MiB | **606 MiB** |
| maximum resident set size | 3,529 MiB | 419 MiB |

The 3,115 MiB difference is the slot cache: `32 slots x 30 layers x 3.2 MiB`.

### Throughput

Three interleaved pairs after a discarded warmup, greedy, 400 new tokens.
Interleaved rather than batched because run-to-run spread here is wider than
many single changes (CLAUDE.local.md's standing rule).

| pair | streamed tok/s | mapped tok/s |
|---|---:|---:|
| 1 | 51.693 | 69.900 |
| 2 | 51.725 | 69.646 |
| 3 | 52.168 | 69.947 |
| spread | 0.9% | 0.4% |

**1.35x decode.** The sampled arm at the CLI defaults reads 48.839 against
68.684, i.e. 1.41x.

**The first measurement of this read 1.99x and was wrong**, because the
streamed arm was the first run after a build and paid both a cold GPU and a
cold page cache (AGENTS.md Gotcha 20). Discarding a warmup and interleaving
took it to 1.35x. The 2x figure should not be quoted.

### Byte-identity

Every arm above reproduces the standing frozen digests exactly:

| arm | md5 of generated text |
|---|---|
| greedy, streamed | `b2f166110a3c402dbc509f73b6c51c4a` |
| greedy, mapped | `b2f166110a3c402dbc509f73b6c51c4a` |
| sampled, streamed | `0c383ac0bd490b0e6adff0b3d55cf98d` |
| sampled, mapped | `0c383ac0bd490b0e6adff0b3d55cf98d` |

That is expected STRUCTURALLY rather than hoped for: the same bytes reach the
same kernels in the same order. The routed slots are still dispatched in the
router's own ranking in both modes, which is what keeps output stable across
residency modes for exactly the reason it is stable across slot counts
(AGENTS.md Gotcha 27).

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

- Implemented for the **Gemma 4** flow only. The other three MoE flows
  (`llama`, `gptoss`, `qwen`) **REFUSE the mode by name at `open`**, they do not
  take the streamed path silently. That matters because
  `open_expert_streamers` NULLS every `pread` streamer when this mode engages,
  so an unwired family would otherwise reach its own `.ok_or_else` and report
  "layer N has no packed-expert streamer" -- blaming the INSTALL for a mode the
  caller chose. `mapped_residency_refusal` is a pure function of the family so
  the rule is testable with no env var, no GPU and no install, and widening it
  is one edit there plus that family's dispatch arm.
- **`MFERENCE_ROUTED_BATCH=1` and this mode cannot be combined**, and the
  combination is refused by name at `families/gemma4/moe_batch.rs`. The batched
  routed pair binds one buffer per CACHE SLOT (`MoePrefillRoute::slot` indexes
  that array) and this mode has no slot cache: it has one buffer per layer plus
  a per-expert offset, and the experts to bind are the SUB-BATCH's union, which
  changes inside the layer loop. Serving both means re-binding per sub-batch
  rather than per layer, which is a change to that driver rather than a branch
  in it. Deleting the refusal was measured rather than reasoned about: the run
  trips `assert!(!blobs.is_empty() ...)` in `gpu::moe_prefill_batch`'s argument
  encoder, a plain `assert!` that aborts in release, naming neither seam.
  The refusal is at the DRIVER rather than at `open` because
  `set_routed_batch_prefill` can flip that seam after open.
- The DEFAULT chunked-prefill path is unaffected and needs no second arm:
  with `MFERENCE_ROUTED_BATCH` unset, Gemma 4's chunked driver runs its routed
  half through `encode_gemma4_layer_routed_moe` -- the same function the
  sequential decode path uses, and the one carrying the mapped branch.
- Behind `MFERENCE_EXPERT_RESIDENCY=mapped`, off by default, an A/B seam in
  the shape `MFERENCE_ROUTED_BATCH` and `MFERENCE_BATCHED_GEMV` already have:
  both arms must produce identical tokens, so it is a seam first and a feature
  second.
- No CLI flag yet, and deliberately: `turbospark-bench`, the memory oracles
  and the quality gates must not sense a knob that moves prefill and footprint
  by this much (AGENTS.md Gotcha 35). A flag needs the harnesses pinned first.
- Every frozen row in `crates/bench` is a STREAMED row and stands unchanged. A
  mapped row is a new row, not a re-freeze.

## Not done

- `auto` resolution against machine memory, which is what would let this be on
  by default. It needs the eviction behaviour measured under real memory
  pressure, which nothing here has done.
- `madvise(MADV_WILLNEED)` on the routed offsets, the mmap analogue of the
  `F_RDADVISE` hinting the pread path uses.
- The other three MoE families.
- Composing with the BATCHED routed pair (`MFERENCE_ROUTED_BATCH=1`), which is
  refused rather than served today. It needs the wide argument buffer bound per
  SUB-BATCH instead of per layer, since the union of experts to bind changes
  inside the layer loop where the slot array does not. Worth costing against
  what it buys: the two seams are the repo's two prefill levers and nobody has
  measured them together, so the win is unknown rather than known-small.
- A measurement of what happens when the OS evicts a mapped expert mid-decode,
  which is the failure mode this design accepts and the streamer does not have.
