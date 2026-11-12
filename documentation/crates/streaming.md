---
description: "Routed-expert streaming: pread-based slot cache, memory-mapped layer residency, LFU/LRU eviction policy, and OS read-ahead advice for expert weights."
icon: wave-square
---

# streaming

`crates/streaming` (package `turbospark-streaming`) moves routed-expert
weights from disk into buffers a forward pass can consume. A MoE install's
routed expert table is far larger than memory, so the crate provides two
residency strategies over the same on-disk layout: a `pread`-based streamer
that copies one expert into a page-aligned slot under an LFU/LRU cache
policy, and a read-only memory map that hands out expert bytes zero-copy. It
also issues kernel read-ahead advice for the experts a plan is about to
miss. This is one of the few crates where the workspace deliberately allows
a narrow amount of `unsafe` and platform-specific code, and it is unix-only
(`std::os::unix` throughout).

## What it does

| Responsibility | Where |
| --- | --- |
| Byte layout of one packed-expert layer file | `src/stream_layout.rs` (`StreamLayout`) |
| `pread` into page-aligned slots with a slot cache | `src/pread_streamer.rs` (`PreadExpertStreamer`) |
| Pure LFU/LRU/decayed-LFU slot policy | `src/expert_cache.rs` (`ExpertCache`, `ExpertCachePolicy`, `ExpertCachePlan`) |
| Read-only mapped residency and page preparation | `src/mapped_experts.rs` (`MappedExpertLayer`) |
| macOS `F_RDADVISE` read-ahead advisory | `src/rdadvice.rs` |
| Linux io_uring path | `src/linux_uring.rs` |
| Page-aligned slot allocation | `src/aligned_slot.rs` (`AlignedSlot`) |
| Disk I/O accounting | `src/disk_io.rs` (`ExpertIoStats`) |

## Where it sits

`turbospark-streaming` depends on `turbospark-core` (aliased `foundation`)
and `turbospark-model-io`: `StreamLayout::from_packed_experts_layer` is built
from model-io's decoded `packed_experts/layout.json` (`LayerLayout`). The
only consumer is `runtime`, which opens one streamer or mapping per streamed
routed-expert layer at model load. `libc` is a dependency on every unix
target because `pread_streamer` reaches for `posix_memalign` and `sysconf`
on all of them, not just on macOS.

## Public API tour

Root re-exports from `crates/streaming/src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `StreamLayout` | `src/stream_layout.rs` | Path, window, experts per layer, stride, and optional explicit per-expert offsets; `expert_offset(layer, expert)` |
| `PreadExpertStreamer::open(layout, slot_count, policy)` | `src/pread_streamer.rs` | Pre-allocates slots and opens the layer file with a size check |
| `PreadExpertStreamer::load_expert(layer, expert)` | `src/pread_streamer.rs` | Reads one expert into a slot, returning the slot index |
| `PreadExpertStreamer::load_experts_cached(&[usize])` | `src/pread_streamer.rs` | Plans, loads misses, and returns one slot per requested expert |
| `PreadExpertStreamer::slot_data(slot)` | `src/pread_streamer.rs` | The bytes a slot currently holds |
| `PreadExpertStreamer::advise_expert_cache_plan_misses(&plan)` | `src/pread_streamer.rs` | Read-ahead advice over a plan's misses |
| `ExpertCache::new(slot_count, policy, experts_per_layer)` | `src/expert_cache.rs` | The pure policy object; `plan`, `plan_if_possible`, `commit_plan` |
| `ExpertCachePolicy` | `src/expert_cache.rs` | `Lru`, `Lfu` (the default), or `LfuDecay { interval }`; also `from_env` for the decay A/B seam |
| `MappedExpertLayer::open(layout)` | `src/mapped_experts.rs` | Maps the layer window read-only; `expert_bytes(expert)`, `prefetch_expert(expert)`, `prepare_selected_experts(...)` |
| `MappedPagePreparationMode`, `MappedPagePreparationStats` | `src/mapped_experts.rs` | `Advice` or `Touch` preparation of demand-selected experts |
| `StreamerError` | `src/error.rs` | `OpenFailed`, `SizeMismatch`, `OffsetOutOfRange`, `PreadFailed`, `SlotOutOfRange`, `AllocationFailed` |
| `rdadvice_call`, `clipped_byte_count`, `RdAdviceCallResult` | `src/rdadvice.rs` | The `F_RDADVISE` call and its result |
| `AlignedSlot`, `ExpertIoStats`, `dio_aligned`, `dio_round_up`, `LinuxIoMode`, `DIO_ALIGNMENT` | `src/aligned_slot.rs`, `src/disk_io.rs`, `src/linux_uring.rs` | Allocation, accounting, and the Linux direct-I/O helpers |

## Examples

### Plan and commit against the slot cache

The cache policy is pure, so it is tested against scripted access traces
with no file behind it, as in `crates/streaming/tests/expert_cache.rs`. The
`avoiding_slots` set marks slots the caller cannot evict on this plan.

```rust
use std::collections::HashSet;
use turbospark_streaming::{ExpertCache, ExpertCachePolicy};

let mut cache = ExpertCache::new(4, ExpertCachePolicy::Lfu, 16);

// A cold cache: every request is a miss with an assigned slot.
let plan = cache.plan(&[1, 2, 3], &HashSet::new());
assert_eq!(plan.hits, 0);
assert_eq!(plan.misses, vec![0, 1, 2]);

// The loader reads the misses, then the plan is committed, which makes
// a repeated request a hit on the same slot.
cache.commit_plan(&plan);
let plan2 = cache.plan(&[1], &HashSet::new());
assert_eq!(plan2.hits, 1);
assert!(plan2.misses.is_empty());
assert_eq!(plan2.assigned_slots[0], plan.assigned_slots[0]);

// `plan_if_possible` returns None when a miss has no evictable slot:
// here a 2-slot cache is fully committed and both slots are avoided.
let mut small = ExpertCache::new(2, ExpertCachePolicy::Lfu, 16);
let p = small.plan(&[8, 9], &HashSet::new());
small.commit_plan(&p);
let both: HashSet<usize> = [0, 1].into_iter().collect();
assert!(small.plan_if_possible(&[7], &both).is_none());
```

### Stream experts through a pread streamer

Adapted from `crates/streaming/tests/pread_streamer.rs`, which writes a
synthetic layer file where expert `e`'s blob is every byte set to `e`, so a
wrong-expert read is visible by content.

```rust
use turbospark_streaming::{ExpertCachePolicy, PreadExpertStreamer, StreamLayout};

// From model-io's decoded packed-experts layout in real installs:
// let layout = StreamLayout::from_packed_experts_layer(&layer, &layout_dir);
let expert_stride: u64 = 64;
let experts_per_layer: usize = 4;
let layout = StreamLayout {
    path: layer_path.display().to_string(),
    stream_offset: 0,
    stream_size: expert_stride * experts_per_layer as u64,
    experts_per_layer: experts_per_layer,
    expert_stride: expert_stride,
    expert_offsets: None,
};

let mut streamer =
    PreadExpertStreamer::open(layout, 2, ExpertCachePolicy::Lfu).unwrap();

// One expert per slot; a repeat request reuses the resident slot.
let slots = streamer.load_experts_cached(&[0, 1]).unwrap();
assert!(streamer.slot_data(slots[0]).iter().all(|&b| b == 0));

// Physical I/O accounting for the reads that were issued.
let stats = streamer.io_stats();
assert!(stats.batches >= 1);
```

`open` fails with `StreamerError::SizeMismatch` when the file is shorter
than the layout window, the case where a stale layout is most likely.

### Map a layer and read expert bytes zero-copy

From `crates/streaming/tests/mapped_experts.rs`, whose load-bearing case is
that the mapped bytes equal the bytes the streamer would have copied.

```rust
use turbospark_streaming::MappedExpertLayer;

let mapped = MappedExpertLayer::open(layout).unwrap();
assert_eq!(mapped.experts_per_layer(), experts_per_layer);
assert_eq!(mapped.expert_stride(), expert_stride);

// Host-side read of one expert; the GPU path addresses the mapping by
// offset instead.
let bytes = mapped.expert_bytes(2).unwrap();
assert!(bytes.iter().all(|&b| b == 2));

// Warm one expert's pages without changing its bytes.
let touched = mapped.prefetch_expert(3).unwrap();
assert!(touched >= 1);
```

Every offset `MappedExpertLayer` hands out is relative to
`page_aligned_bytes()`, so a non-zero `stream_offset` cannot silently read
the wrong expert.

## Gotchas

{% hint style="warning" %}
`linux_uring.rs` is gated only by its Linux tests, not by a cross-target
check: its io_uring flags are hand-copied UAPI, and a wrong bit is just
another valid flag. Run the crate's tests on Linux after touching it, per
the crate's `AGENTS.md`.
{% endhint %}

{% hint style="warning" %}
`PreadExpertStreamer::nocache_active()` reports whether the descriptor is
actually bypassing the unified buffer cache; the `F_NOCACHE` request can be
refused. A disk-bound measurement that reads only the environment variable
risks publishing a warm-cache row as a disk-bound one. Relatedly, price any
`ExpertCache` policy change with `scripts/expert_policy_replay.py` before
building it.
{% endhint %}

Two lifetime rules from the source: an `AlignedSlot`'s pointer is valid only
for the streamer that owns it, and `MappedExpertLayer::page_aligned_bytes()`
is aliased zero-copy by GPU wrappers, so a wrapper must not outlive the
layer value it was made from.

## Related pages

- [core](core.md)
- [The .gturbo Format](../models/gturbo-format.md)
- [Memory and Capacity](../models/memory-and-capacity.md)
- [Environment Variables](../advanced/environment-variables.md)
