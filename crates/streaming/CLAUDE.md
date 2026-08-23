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
|   +-- aligned_slot.rs     # AlignedSlot page-aligned buffer for slot streaming
|   +-- expert_cache.rs     # ExpertCache implementing pure LFU/LRU eviction policy
|   +-- mapped_experts.rs   # MappedExpertLayer: the experts read in place from an mmap
|   +-- read_pool.rs        # Process-wide pool of parked reader threads for parallel pread
|   +-- rdadvice.rs         # macOS F_RDADVISE kernel hint wrapper (unsafe)
|   +-- stream_layout.rs    # Expert blob offset and byte layout calculation helpers
|   \-- error.rs            # StreamingError enum definition
\-- tests/
    +-- expert_cache.rs     # Pure LFU/LRU eviction policy unit tests
    +-- mapped_experts.rs   # MappedExpertLayer vs the streamer's copy, byte for byte
    \-- pread_streamer.rs   # PreadExpertStreamer file reading & caching integration tests
```

## Key Modules

- `pread_streamer.rs`: `PreadExpertStreamer` for streaming expert weight blobs from `packed_experts/blobs.bin`.
- `expert_cache.rs`: `ExpertCache` implementing pure LFU/LRU eviction policy.
- `read_pool.rs`: Process-wide pool of parked reader threads (`run_batch`) for parallel pread of expert weight chunks.
- `rdadvice.rs`: Platform-gated `F_RDADVISE` kernel hint wrapper (macOS `fcntl`; documented no-op on other OSes).
- `mapped_experts.rs`: `MappedExpertLayer`, the routed experts read IN PLACE out of one `mmap` per layer instead of `pread`-copied into a pinned slot. The streamer's sibling, not its replacement (Gotcha 8).
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
6. **A slot's BYTES and the cache's belief about them are two facts, and only the cached path keeps them together.** `load_expert` / `load_expert_into_slot` write a slot outside any plan, so they must invalidate that slot's residency or a later `plan_experts_cached` scores a HIT and hands out bytes belonging to a different expert -- no error, no crash, just the wrong weights, which is Gotcha 27's failure mode reached through the API rather than through dispatch order. They do invalidate now, BEFORE the read (a failed read leaves the slot half-written, which is equally not the expert the cache thinks it is). Production never mixes the two paths -- the four family `moe.rs` files use `plan_experts_cached` + `execute_expert_cache_plan` exclusively -- so nothing but `a_direct_load_leaves_no_stale_residency_for_the_cache_to_hit_on` enforces it. Any new method that writes slot memory owes the same call.
7. **`read_pool` safety rests on `run_batch` blocking.** Workers get raw destination pointers and a borrowed `RawFd`, both valid only because the submitter blocks until every claim is dropped. A `Claim`'s `Drop` (not the worker's happy path) signals completion, so a panicking read cannot hang the submitter. Any change that lets `run_batch` return early breaks all of it.

8. **THE COPY THIS CRATE EXISTS TO OPTIMIZE IS NOT ALWAYS NECESSARY, and `MappedExpertLayer` is the arm that skips it.** Gotcha 3 says the `pread` is a page-cache memcpy rather than disk I/O and should be optimized as a bandwidth problem. The stronger reading is that a memcpy from the page cache into a pinned slot, so a Metal buffer can wrap the slot, is a copy nothing downstream ever asked for: `crates/gpu`'s `moe_decode.rs` reads expert weights in place from "streamer slots or any other page of memory", so the file's own mapping is an acceptable source and a routed blob pointer can be `mapped_layer_buffer + expert_offset`.

   Measured on the real Gemma 4 install (`docs/EXPERT_RESIDENCY.md`): peak `phys_footprint` 3,721 MiB streamed against **606 MiB** mapped, decode 51.9 against 69.8 tok/s, and output byte-identical on both the greedy and sampled smokes. The slot cache is 70-90% of the measured peak of every MoE install and it holds a copy of bytes the GPU can read where they already are.

   **BOTH TYPES STAY, AND THE TRADE IS THE REASON.** A pinned slot GUARANTEES a bounded working set; a mapping hands residency to the OS, so pages nobody is charged for are pages that can be evicted. Mapped residency is right on a machine that can hold the table and wrong on one that cannot, which is exactly the cold-or-memory-tight case Gotcha 3 already warns about. It is a MODE (`MFERENCE_EXPERT_RESIDENCY=mapped`, off by default) and must resolve DOWN to the streamer rather than up.

   Two things about the implementation. It reuses `StreamLayout` rather than restating the offset arithmetic, INCLUDING the per-expert offset table -- the writer need not emit dense `expert * stride` offsets, and a mapped read that assumed it would silently return a neighbour's blob. And every offset it hands out carries `ResidentBuffer`'s `slice_shift`, because that type rounds the file offset DOWN to a page boundary to make the base page-aligned (what `newBufferWithBytesNoCopy` requires) and reports the difference; dropping the shift reads the file's header instead of expert 0, which is a wrong read rather than an error. Both are mutation-checked in `tests/mapped_experts.rs`, and each mutation reddens only its own case.
