# turbospark-streaming

Routed-expert streamer (`PreadExpertStreamer`) for loading MoE expert weights on demand via `pread`, coupled with a per-layer slot cache (`ExpertCache`).

## Safety

- Contains `unsafe` code in `rdadvice.rs` for invoking macOS `F_RDADVISE` via `fcntl`, in `disk_io.rs` for `proc_pid_rusage` and the `F_NOCACHE` `fcntl`, and in `read_pool.rs` for shared memory/pointer management across parked reader threads.
- `disk_io.rs`'s probe passes a WHOLE `libc::rusage_info_v2`, never a prefix. `proc_pid_rusage` takes no count parameter (unlike the `task_info` call `crates/bench/src/memory.rs` truncates on purpose), so the kernel writes the full flavor unconditionally and a shortened struct is a stack overwrite rather than a smaller answer.

## Directory & File Structure

```
crates/streaming/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library root
|   +-- pread_streamer.rs   # PreadExpertStreamer for on-demand expert weight reading
|   +-- aligned_slot.rs     # AlignedSlot page-aligned buffer for slot streaming
|   +-- expert_cache.rs     # ExpertCache implementing pure LFU/LRU eviction policy
|   +-- read_pool.rs        # Process-wide pool of parked reader threads for parallel pread
|   +-- rdadvice.rs         # macOS F_RDADVISE kernel hint wrapper (unsafe)
|   +-- disk_io.rs          # physical-disk-read probe + F_NOCACHE seam, ExpertIoStats (unsafe)
|   +-- stream_layout.rs    # Expert blob offset and byte layout calculation helpers
|   \-- error.rs            # StreamingError enum definition
\-- tests/
    +-- expert_cache.rs     # Pure LFU/LRU eviction policy unit tests
    \-- pread_streamer.rs   # PreadExpertStreamer file reading & caching integration tests
```

## Key Modules

- `pread_streamer.rs`: `PreadExpertStreamer` for streaming expert weight blobs from `packed_experts/blobs.bin`.
- `expert_cache.rs`: `ExpertCache` implementing pure LFU/LRU eviction policy.
- `read_pool.rs`: Process-wide pool of parked reader threads (`run_batch`) for parallel pread of expert weight chunks.
- `rdadvice.rs`: Platform-gated `F_RDADVISE` kernel hint wrapper (macOS `fcntl`; documented no-op on other OSes).
- `stream_layout.rs`: Expert blob offset and byte layout helpers.

## Development & Test Commands

```sh
# Run tests for turbospark-streaming
cargo test -p turbospark-streaming
```

## Crate Gotchas

1. **Decoupled Cache Logic**: The `ExpertCache` eviction policy is deliberately pure logic with no direct file I/O calls. This design allows testing eviction behavior and trace hit rates in unit tests without requiring multi-gigabyte model installs.
2. **Platform Specifics**: `rdadvice.rs` uses macOS-specific `F_RDADVISE` hints to optimize kernel page cache behavior for expert streams. On non-macOS systems, this call compiles to a safe no-op.
3. **The `pread` is a memcpy, not disk I/O -- MEASURED at last, not inferred.** With the install's expert files in page cache, this path moved 125 MiB per token in 5.26 ms on the real 26B (far past any SSD). Optimize it as a bandwidth problem. On a cold or memory-tight machine it becomes genuinely disk-bound and the tuning below buys much less.

   That claim was an INFERENCE from the transfer rate for the whole life of this crate, and it is now an observation. Real Gemma 4 install, 16 slots, 21-token prompt, `--max-new 120`, 64.2% cache hit rate, three cache states measured 2026-08-29:

   | state | requested | physical | amplification |
   |---|---:|---:|---:|
   | warm | 274.9 MiB/token | **0.0** | 0.00x |
   | populating (first run after `purge`) | 274.9 | 54.3 | 0.20x |
   | `F_NOCACHE` on a purged machine | 274.9 | **274.9** | **1.00x** |

   So the warm claim holds exactly, and `F_RDADVISE` is NOT over-reading -- the disk-bound arm reads 1.00x rather than more, which is the same statement from the other side. **The tok/s column is deliberately absent**: every throughput reading taken that day shared the machine with four concurrent `cargo test --workspace` runs from other sessions, which is Gotcha 43's contamination and makes them unpublishable. The BYTES are unaffected by CPU contention, which is why this table is bytes only. Which regime a run was in is an OBSERVABLE now (`MFERENCE_EXPERT_DISK_IO=1`, Gotcha 8); `docs/BENCHMARKS.md` records a 2.6x cold-versus-warm throughput error found by accident, which is the cost of not having had the instrument.
4. **Parallelism is per CHUNK, not per miss.** Swift ran one task per miss (`DispatchQueue.concurrentPerform`), which silently degrades to a single-threaded copy on the common warm-cache layer that misses exactly once (1.3 misses/layer at 32 slots on the real 26B). `execute_expert_cache_plan` splits each miss into `MISS_READ_CHUNK_BYTES` pieces so a lone miss still reads wide. Do not "simplify" this back to one unit of work per miss.
5. **Chunking makes a persistent pool mandatory, not optional.** Splitting multiplies the threads a layer wants, and there are 30 layers per token on the decode critical path. `read_pool` parks `POOL_THREADS` workers once; going back to `std::thread::scope` would pay thread creation hundreds of times per token. Measured: chunking alone 5.26 -> 4.16 ms/token, plus the pool 4.16 -> 3.86.
6. **A slot's BYTES and the cache's belief about them are two facts, and only the cached path keeps them together.** `load_expert` / `load_expert_into_slot` write a slot outside any plan, so they must invalidate that slot's residency or a later `plan_experts_cached` scores a HIT and hands out bytes belonging to a different expert -- no error, no crash, just the wrong weights, which is Gotcha 27's failure mode reached through the API rather than through dispatch order. They do invalidate now, BEFORE the read (a failed read leaves the slot half-written, which is equally not the expert the cache thinks it is). Production never mixes the two paths -- the four family `moe.rs` files use `plan_experts_cached` + `execute_expert_cache_plan` exclusively -- so nothing but `a_direct_load_leaves_no_stale_residency_for_the_cache_to_hit_on` enforces it. Any new method that writes slot memory owes the same call.
7. **`read_pool` safety rests on `run_batch` blocking.** Workers get raw destination pointers and a borrowed `RawFd`, both valid only because the submitter blocks until every claim is dropped. A `Claim`'s `Drop` (not the worker's happy path) signals completion, so a panicking read cannot hang the submitter. Any change that lets `run_batch` return early breaks all of it.
8. **Two measurement seams, both OFF by default, and neither is an optimization.** `MFERENCE_EXPERT_DISK_IO=1` samples `proc_pid_rusage`'s `ri_diskio_bytesread` around each read batch, so `MFERENCE_PHASES=1`'s new `expert bytes` row reports requested MiB/token against PHYSICAL MiB/token and their ratio. `MFERENCE_EXPERT_NOCACHE=1` sets `F_NOCACHE` on the expert blob, which is an EXPERIMENTAL CONDITION in the shape of `scripts/power.sh COOLING=max` (AGENTS.md Gotcha 28): it makes the disk-bound arm that `docs/EXPERT_ROUTING.md` names as the prefetch-reversal condition reproducible, and it is expected to be SLOWER. Under it, `F_RDADVISE` is skipped and reported as `skipped` -- readahead and cache-bypass are contradictory instructions about one descriptor, and a run issuing both measures neither.

   **`F_NOCACHE` DOES NOT EVICT PAGES THAT ARE ALREADY RESIDENT, so the flag alone does not guarantee a disk-bound run.** Measured while writing `the_disk_read_counter_moves_on_a_bypassed_read`: an 8 MiB file written, `fsync`ed, closed and reopened with `F_NOCACHE` read back with a disk delta of **0**, because the write had left every page in the buffer cache; setting `F_NOCACHE` on the WRITE descriptor too took the same read to exactly 8,388,608, stable across three runs. An expert blob a previous run already faulted in stays resident, so pair the seam with `sudo purge` on a warm machine and confirm from a non-zero `bytes_physical` rather than from the flag.

   **This is not theoretical: the first real-install run of the seam FAILED to establish the condition, and the instrument is what said so.** On the warm Gemma 4 install the bypassed arm read **0.5 MiB/token physical against 274.9 requested (0.00x)**, i.e. no more device I/O than the warm arm, because prior runs had left the blobs resident. The throughput gap to the warm arm was ~3%, which anyone reading the FLAG instead of the BYTES would have published as a disk-bound row. After `sudo purge` the same command read **274.9 of 274.9, 1.00x**. That contrast is the whole argument for landing the two seams together.

   **THE PAIR CANNOT BE INTERLEAVED, which is a real exception to the standing A/B rule** (AGENTS.md Gotcha 20 and `CLAUDE.local.md` both say to alternate variants pair by pair). A CACHING run repopulates the buffer cache and `F_NOCACHE` does not evict, so the next bypassed run is served from memory and silently stops being disk-bound. Measured in one sitting: two bypassed runs after a purge read 1.00x, then two caching runs, then two more bypassed runs read **0.00x** -- the condition was destroyed by the arm it was being compared against. Run every bypassed arm consecutively after a purge, take the warm arms afterwards, and accept that this comparison is cross-capture with Gotcha 22's caveat rather than interleaved.

   **`samples == 0` means UNMEASURED, never "no disk reads".** `ExpertIoStats::amplification` returns `None` there and the CLI prints `n/a`, because a warm run and an uninstrumented one are identical in `bytes_physical` alone -- Gotcha 59's shape, where a degenerate input scores an instrument's best result.
9. **A cross-platform `io_uring`-style submission/completion layer is DECLINED, on arithmetic.** `io_uring` is Linux-only (macOS has no equivalent: POSIX AIO caps at `AIO_LISTIO_MAX` 16 with `kern.aioprocmax` 16 and `kern.aiothreads` 4, `kqueue` cannot drive regular-file reads at all, and `dispatch_io` is a GCD-scheduled thread pool underneath). Priced against its three wins here: **batched submission** is worth at most 0.65% (1.3 misses/layer at 32 slots, split 4 ways, 30 layers is ~156 preads/token, or 0.156 ms of a ~24 ms token at 1 us each, and a ring does not remove them anyway); **registered buffers** are already had, since `AlignedSlot` is one `posix_memalign` allocation per slot made at open and reused forever; **async completion** is what this pool already provides, at a thread count that was swept (Gotcha 5) rather than assumed. The deeper reason is that a submission ring hides I/O LATENCY and the measured read is a ~23.8 GiB/s page-cache memcpy, so the term it optimizes is near zero -- the same argument that closed prefetch in `docs/EXPERT_ROUTING.md`. Reopens only if Gotcha 8's instrument shows reads are genuinely latency-bound.
10. **Page-level dedup does not apply to this access shape.** `garnermccloud/sglang-ssd-stream`, which Gotchas 8 and 9 borrow from, sorts rows by filesystem page and reads each page once because its unit is 16 rows of 160 bytes scattered across a 47.68 GiB table. This crate's unit is 8 experts of ~3.2 MiB per layer per token, contiguous, page-aligned, and already deduplicated at the EXPERT level by `ExpertCache`, so two distinct experts share no page and the scan would find nothing. Their `POSIX_FADV_RANDOM` and this crate's `F_RDADVISE` are correct in OPPOSITE directions for the same reason, which is AGENTS.md Gotcha 36's granularity axis. Do not read their ~64 MiB working-set figure as a target for `--expert-cache-slots` either; the demand rates are three to four orders of magnitude apart.
