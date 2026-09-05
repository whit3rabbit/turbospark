---
uuid: "64a37a85-8a76-4666-a754-d0a70b1ce915"
title: "turbospark-streaming"
summary: "Routed-expert pread streamer and slot cache. On a warm install the pread is a page-cache memcpy, not disk I/O, measured at 125 MiB/token in 5.26ms"
tags: ["crate", "streaming"]
source: "crates/streaming/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-streaming do?

Streams MoE routed-expert weights on demand via `pread`
(`PreadExpertStreamer`) into a per-layer slot cache (`ExpertCache`, pure
LFU/LRU eviction), with a persistent parked-thread read pool
(`read_pool.rs`) for parallel reads and a macOS `F_RDADVISE` kernel hint
(`rdadvice.rs`). `MappedExpertLayer` is the sibling path: routed experts
read IN PLACE from an `mmap`, no slot copy at all
(`TURBOSPARK_EXPERT_RESIDENCY=mapped`).

Contains `unsafe` in `rdadvice.rs` (macOS `F_RDADVISE` `fcntl`),
`disk_io.rs` (`proc_pid_rusage`, `F_NOCACHE`), and `read_pool.rs` (raw
destination pointers and a borrowed `RawFd` across parked worker threads,
sound only because `run_batch` blocks until every claim is dropped).

## Don't

- Don't optimize this crate's hot path as a disk-I/O problem by default.
  With the install's expert files in page cache, `pread` is a memcpy
  (measured: 274.9 MiB/token requested, 0.0 physical, 0.00x amplification
  when warm). It only becomes genuinely disk-bound on a cold or memory-
  tight machine. `TURBOSPARK_EXPERT_DISK_IO=1` makes the regime observable.
- Don't "simplify" `execute_expert_cache_plan` back to one read per miss.
  Chunking (`MISS_READ_CHUNK_BYTES`) is what lets a lone cache miss (the
  common warm-cache case, ~1.3 misses/layer at 32 slots) still read at full
  width, and removing it silently reverts to a single-threaded copy.
- Don't call `load_expert` / `load_expert_into_slot` without invalidating
  that slot's cached residency first. A slot's bytes and the cache's belief
  about what's in it are two separate facts, and only the cached
  plan-and-execute path keeps them together automatically. Get this wrong
  and a later lookup scores a HIT and hands out the wrong expert's weights,
  with no error or crash.
- Don't interleave `TURBOSPARK_EXPERT_NOCACHE=1` runs with normal (caching)
  runs when measuring the disk-bound condition. This is a real exception to
  the standard "alternate variants pair by pair" A/B rule: a caching run
  repopulates the page cache, `F_NOCACHE` doesn't evict what's already
  resident, and the very next bypassed run silently stops being disk-bound
  (measured going from 1.00x amplification to 0.00x in one sitting). Run
  every bypassed arm consecutively right after `sudo purge`.
- Don't read `ExpertIoStats::amplification()` returning `None` as "no disk
  reads." It means UNMEASURED (`samples == 0`), which looks identical to a
  fully warm run unless you check for it explicitly.
- Don't reach for an `io_uring`-style submission ring here. It was priced
  and declined: batched submission is worth at most ~0.65% of a token, and
  registered buffers and async completion are already had (one
  `posix_memalign`'d `AlignedSlot` per slot, a swept thread-count pool).
  The measured read is a ~23.8 GiB/s page-cache memcpy, not latency-bound
  I/O, so a ring optimizes a term that's already near zero.
