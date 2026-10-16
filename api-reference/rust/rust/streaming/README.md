---
description: "Load MoE expert weights on demand with a pread slot cache or a read-only mmap, plus platform readahead and I/O counters."
icon: box
---

# turbospark-streaming

`turbospark-streaming` loads Mixture-of-Experts expert weights from a layer file on demand. It either copies each requested expert into a fixed set of page-aligned slots through a per-layer cache (`PreadExpertStreamer`), or it maps the file and returns offsets into the mapping (`MappedExpertLayer`). Reach for it when the full expert table does not fit in memory and each decode step needs only a small, bounded set of experts.

This is an advanced, low-level crate. It hands out raw slot pointers, byte offsets, and platform-specific readahead calls, and it leaves the GPU wrapping to you. Read it only if you are writing your own MoE forward pass, your own expert store, or tooling that measures expert I/O. Most users get this behavior through the runtime crate.

## What it provides

- `PreadExpertStreamer` opens one layer file, allocates its slot memory once, and loads experts with `pread`. Cache misses are split into chunks and read in parallel on a process-wide pool of parked worker threads.
- `ExpertCache` is the LFU or LRU slot policy with no file I/O. The streamer owns one internally. You can also use it alone to test a policy against an access trace.
- `MappedExpertLayer` maps one layer file read-only. Its experts are read in place, so no slot copy is made. See [Mapped experts](mapped-experts.md).
- `StreamLayout` describes the byte window of one layer file. It holds the file path, the window offset and size, the expert count, the stride, and an optional table of per-expert offsets.
- `ExpertIoStats` and `ExpertIoAdviceResult` report requested bytes, sampled physical bytes, and readahead call results.

The crate does not depend on Metal or the GPU crate. It returns bytes and pointers, and your code decides how to wrap them.

## Platform support

The crate uses `std::os::unix` throughout. It builds on unix targets and fails on Windows. The table shows which parts behave differently by platform.

| Piece | macOS | Linux | Other unix |
| --- | --- | --- | --- |
| `PreadExpertStreamer`, `ExpertCache`, read pool | Yes. Pool threads run at utility QoS only when `TURBOSPARK_READ_QOS=utility` is set. | Yes. No QoS change. | Not checked here. |
| Readahead advice (`advise_experts` and friends) | `F_RDADVISE` through `fcntl` | `posix_fadvise(WILLNEED)` | No-op. Each call reports `succeeded: false`, so it counts as a failed call. |
| Cache bypass (`TURBOSPARK_EXPERT_NOCACHE=1`) | Sets `F_NOCACHE` on the blob descriptor | Never applied. `nocache_active()` returns `false`. | Never applied |
| Physical read counter (`TURBOSPARK_EXPERT_DISK_IO=1`) | Reads `proc_pid_rusage` | Not available. `io_stats().samples` stays 0. | Not available |
| `MappedExpertLayer` | Yes | Yes, through `memmap2` | Not checked here |
| `LinuxIoMode` (the `Mode` enum from `linux_uring`) and `dio_*` helpers | Exported | Exported. The io_uring path is scaffolding. | Exported |

Linux has two caveats. The repository gate for Linux is `cargo check --target x86_64-unknown-linux-gnu -p turbospark-streaming` (from `crates/streaming/AGENTS.md`). This audit did not re-run that check, so whether it passes is unverified here. The io_uring code has never run on a Linux machine, according to its own module header. `Mode::from_env()` defaults to `Mode::Pread`, and no streamer reads a `Mode::Uring` value, so selecting `uring` changes nothing in this crate.

## Add the crate

The workspace uses the package name `turbospark-streaming`. Add the package from crates.io once the version you need is published, or use a path or git dependency from a checkout.

```toml
[dependencies]
turbospark-streaming = { path = "../turbospark/crates/streaming" }
```

The crate also depends on `turbospark-core` and `turbospark-model-io`. Both are workspace path dependencies, so a checkout-based dependency needs them reachable by path as well. Inside the workspace, this crate is imported as `streaming`, and its dependencies appear as `foundation` and `model_io`. Those aliases are workspace-internal. Your code imports `turbospark_streaming`.

## Stream one layer

This example writes a small layer file where expert `e` is 64 bytes of value `e`. It opens a streamer with two slots, loads two experts, and checks the bytes in each slot.

```rust
use std::fs::File;
use std::io::Write;

use turbospark_streaming::{ExpertCachePolicy, PreadExpertStreamer, StreamLayout};

const STRIDE: u64 = 64;
const EXPERTS: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join("turbospark-streaming-example-layer.bin");
    let mut file = File::create(&path)?;
    for expert in 0..EXPERTS {
        file.write_all(&vec![expert as u8; STRIDE as usize])?;
    }
    drop(file);

    let layout = StreamLayout {
        path: path.display().to_string(),
        stream_offset: 0,
        stream_size: STRIDE * EXPERTS as u64,
        experts_per_layer: EXPERTS,
        expert_stride: STRIDE,
        expert_offsets: None,
    };

    // Two slots with LFU eviction. Slot memory is allocated once, here.
    let mut streamer = PreadExpertStreamer::open(layout, 2, ExpertCachePolicy::Lfu)?;

    let slots = streamer.load_experts_cached(&[1, 3])?;
    for (expert, slot) in [1usize, 3].iter().zip(&slots) {
        assert!(streamer.slot_data(*slot).iter().all(|&b| b == *expert as u8));
    }
    println!("requested bytes: {}", streamer.io_stats().bytes_requested);

    // Expert 1 is resident, so this call reads nothing.
    let again = streamer.load_experts_cached(&[1])?;
    assert_eq!(again[0], slots[0]);
    println!("requested bytes: {}", streamer.io_stats().bytes_requested);
    Ok(())
}
```

The first call reads two blobs and reports 128 requested bytes. The second call hits the cache and leaves the counter at 128. Requested bytes count misses only.

`load_experts_cached` plans and executes in one step. The slot indices come back in request order. Slot contents change when a later call evicts that slot, so copy out any bytes you need to keep, or wrap them before the next load.

## Two-step plans

Use the two-step form when some slots must not be overwritten during this load. `plan_experts_cached` takes a `HashSet<usize>` of slots to avoid. `execute_expert_cache_plan` performs the reads and commits the residency.

```rust
use std::collections::HashSet;

let plan = streamer.plan_experts_cached(&[2, 3], &HashSet::new());
let slots = streamer.execute_expert_cache_plan(&plan)?;
```

`plan_experts_cached_if_possible` returns `Option<ExpertCachePlan>` and gives `None` when the misses cannot fit. It still panics if you pass more experts than there are slots, because that assertion runs first.

## Key types and methods

Read the [docs.rs pages](https://docs.rs/turbospark-streaming/latest/turbospark_streaming/) for full signatures. The docs.rs pages need a published version. From a checkout, `cargo doc -p turbospark-streaming --open` builds the same pages locally.

| Item | What you use it for |
| --- | --- |
| `StreamLayout` | Describes the file window. All fields are public, and `from_packed_experts_layer` builds one from a `model_io::LayerLayout` and the layout directory. |
| `PreadExpertStreamer::open(layout, slot_count, policy)` | Opens the file, checks its size, and allocates `slot_count` slots. |
| `load_experts_cached(&[usize])` | Plans and reads the misses. Returns the slot for each request. |
| `plan_experts_cached` and `execute_expert_cache_plan` | The two-step form. |
| `load_expert(layer, expert)` and `load_expert_into_slot(layer, expert, slot)` | Read one expert outside the plan. The target slot's residency is cleared first, so a later cached plan does not treat stale bytes as a hit. |
| `slot_data(slot)` | Borrows a slot's bytes. The slice is valid until the next load that touches the slot. |
| `slot_allocation(slot)` | Returns the raw base pointer and length of a slot's allocation. The pointer stays valid for the streamer's lifetime. Dereferencing it is your `unsafe` responsibility. |
| `non_resident_experts(&[usize])` and `resident_experts_snapshot()` | Read residency without changing it. |
| `prefetch_expert(expert, scratch)` | Reads one blob through your scratch buffer to warm the page cache. It does not fill a slot. It refuses to run when `nocache_active()` is true. |
| `advise_experts`, `advise_expert_misses`, `advise_expert_cache_plan_misses` | Issue readahead for the given experts and return an `ExpertIoAdviceResult`. |
| `reserve_speculative_slots` and `execute_speculative_reservation` | Start reads for experts you expect to need soon. Only use these with a clear plan for the result. |
| `io_stats()` and `nocache_active()` | Read the counters and the cache-bypass state. |
| `ExpertCache::new(slot_count, policy, experts_per_layer)` | The policy alone. Useful for trace tests. |

## Errors

Most fallible calls return `Result<_, StreamerError>`. The enum is `Clone`, `PartialEq`, and `Eq`, and it implements `std::error::Error`.

| Variant | When it happens |
| --- | --- |
| `OpenFailed { path, detail }` | The layer file cannot be opened or its metadata cannot be read. |
| `SizeMismatch { expected, actual }` | The file is shorter than `stream_offset + stream_size` at open, or a read hits end of file. |
| `OffsetOutOfRange { offset }` | An expert's blob falls outside the window. `prefetch_expert` also rejects an index at or above `experts_per_layer`; the load paths check only the window. |
| `PreadFailed { detail }` | A read syscall failed, or `prefetch_expert` was called with an empty scratch buffer or with cache bypass active. |
| `SlotOutOfRange { slot }` | A slot index is at or above `slot_count`. |
| `AllocationFailed { detail }` | Allocating a slot with `posix_memalign` failed at open. |

Some misuse panics instead of returning an error. Check these in your own code:

- `PreadExpertStreamer::open` and `ExpertCache::new` panic when `slot_count` is zero.
- Planning panics when a request has more experts than `slot_count`. The message is `expert cache needs at least N slots`.
- `plan_experts_cached` panics when the misses cannot be placed, because every evictable slot is avoided or still being filled by a speculative read.

{% hint style="warning" %}
Pass distinct expert indices. Duplicates are not merged, so a repeated miss takes its own slot and reads the same blob again. The `requested` byte count also includes each duplicate. Deduplicate the list before you call the streamer.
{% endhint %}

## Measure before you report a number

`io_stats()` returns `ExpertIoStats`. `bytes_requested` is always counted. `bytes_physical` and `samples` are filled only when the physical-read probe is on, which means `TURBOSPARK_EXPERT_DISK_IO=1` on macOS. The probe costs a syscall pair per read batch, so it is off by default.

`amplification()` returns `None` when `samples` or `bytes_requested` is zero. Treat `None` as unmeasured, not as "no disk reads". Read `nocache_active()` before you label a run as cache-bypassed, because the environment variable alone does not prove the `fcntl` took effect.

`TURBOSPARK_EXPERT_NOCACHE=1` stops this process from adding pages to the buffer cache. It does not evict pages that a previous run already loaded. A disk-bound arm needs `sudo purge` before the run, and the result counts only when `bytes_physical` is non-zero. The protocol is in [BENCHMARKING.md](https://github.com/whit3rabbit/turbospark/blob/main/docs/BENCHMARKING.md).

## Where to go next

- [Mapped experts](mapped-experts.md) covers `MappedExpertLayer`, the in-place alternative, and when to choose it.
- [Streaming pipeline](https://github.com/whit3rabbit/turbospark/blob/main/docs/STREAMING.md) describes how the streamer fits into decode.
- [Expert routing](https://github.com/whit3rabbit/turbospark/blob/main/docs/EXPERT_ROUTING.md) and [expert residency](https://github.com/whit3rabbit/turbospark/blob/main/docs/EXPERT_RESIDENCY.md) hold the measurements and the design decisions behind the cache and the mapping mode.
- [turbospark-model-io](https://docs.rs/turbospark-model-io/latest/turbospark_model_io/) defines `LayerLayout` and `PackedExpertsLayout`, which `StreamLayout::from_packed_experts_layer` reads when you use a packed-experts install.
