# turbospark-streaming

Routed-expert streamer (`PreadExpertStreamer`) for loading MoE expert weights on demand via `pread`, coupled with a per-layer slot cache (`ExpertCache`).

## Safety

- Contains `unsafe` code in `rdadvice.rs` for invoking macOS `F_RDADVISE` via `fcntl` and in `read_pool.rs` for shared memory/pointer management across parked reader threads.

## Directory & File Structure

```
crates/streaming/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library root
|   +-- pread_streamer.rs   # PreadExpertStreamer for on-demand expert weight reading
|   +-- expert_cache.rs     # ExpertCache implementing pure LFU/LRU eviction policy
|   +-- read_pool.rs        # Process-wide pool of parked reader threads for parallel pread
|   +-- rdadvice.rs         # macOS F_RDADVISE kernel hint wrapper (unsafe)
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
3. **The `pread` is a memcpy, not disk I/O.** With the install's expert files in page cache, this path moved 125 MiB per token in 5.26 ms on the real 26B (far past any SSD). Optimize it as a bandwidth problem. On a cold or memory-tight machine it becomes genuinely disk-bound and the tuning below buys much less.
4. **Parallelism is per CHUNK, not per miss.** Swift ran one task per miss (`DispatchQueue.concurrentPerform`), which silently degrades to a single-threaded copy on the common warm-cache layer that misses exactly once (1.3 misses/layer at 32 slots on the real 26B). `execute_expert_cache_plan` splits each miss into `MISS_READ_CHUNK_BYTES` pieces so a lone miss still reads wide. Do not "simplify" this back to one unit of work per miss.
5. **Chunking makes a persistent pool mandatory, not optional.** Splitting multiplies the threads a layer wants, and there are 30 layers per token on the decode critical path. `read_pool` parks `POOL_THREADS` workers once; going back to `std::thread::scope` would pay thread creation hundreds of times per token. Measured: chunking alone 5.26 -> 4.16 ms/token, plus the pool 4.16 -> 3.86.
6. **`read_pool` safety rests on `run_batch` blocking.** Workers get raw destination pointers and a borrowed `RawFd`, both valid only because the submitter blocks until every claim is dropped. A `Claim`'s `Drop` (not the worker's happy path) signals completion, so a panicking read cannot hang the submitter. Any change that lets `run_batch` return early breaks all of it.
