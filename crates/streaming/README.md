# turbospark-streaming

Routed-expert weight streaming engine (`PreadExpertStreamer`) for loading MoE expert weights on demand via `pread`, coupled with a per-layer slot cache (`ExpertCache`), a parked worker thread pool for parallel chunk reading (`read_pool.rs`), kernel readahead advice (`rdadvice.rs`), memory-mapped in-place expert reading (`mapped_experts.rs`), and Linux `io_uring` readahead scaffolding (`linux_uring.rs`).

Downstream workspace crates import this package via the `streaming` alias:

```toml
[dependencies]
streaming = { package = "turbospark-streaming", path = "../streaming" }
```

## Purpose & Role

In Mixture-of-Experts (MoE) models, total parameter counts frequently exceed available RAM. `turbospark-streaming` implements high-bandwidth on-demand streaming of expert weight blobs from NVMe disk into RAM or Metal buffers during token decode, decoupling model capacity from physical memory limits.

## Safety

- Contains `unsafe` code strictly bounded to:
  - `rdadvice.rs`: Calling macOS `fcntl` with `F_RDADVISE` for kernel readahead hints.
  - `disk_io.rs`: Calling `proc_pid_rusage` and setting `F_NOCACHE` via `fcntl` for hardware I/O profiling.
  - `read_pool.rs`: Sharing raw pointer destinations across parked worker threads during coordinated parallel chunk reads.
  - `linux_uring.rs`: Direct Linux `io_uring_setup` and `mmap` syscalls without libc.

## Key Modules

- `pread_streamer.rs`: `PreadExpertStreamer` for streaming expert weight blobs on demand from `packed_experts/blobs.bin`.
- `expert_cache.rs`: `ExpertCache` implementing deterministic LFU and LRU slot eviction algorithms.
- `read_pool.rs`: Global pool of parked reader threads (`run_batch`) providing wide parallel I/O for single and batch expert cache misses.
- `rdadvice.rs`: Platform-gated `F_RDADVISE` kernel readahead advice wrapper (macOS `fcntl`; safe no-op on other operating systems).
- `mapped_experts.rs`: `MappedExpertLayer`, enabling routed experts to be read directly in place from an `mmap` per layer instead of `pread`-copied into pinned slots.
- `aligned_slot.rs`: Page-aligned buffer allocations required for unbuffered direct disk I/O.
- `disk_io.rs`: Physical hardware read accounting (`ExpertIoStats`) and the `F_NOCACHE` profiling seam.
- `stream_layout.rs`: Expert blob offset, stride, and byte layout calculation helpers.
- `linux_uring.rs`: Linux `io_uring` submission/completion queue implementation and Linux cgroup memory probe scaffolding.

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-streaming
cargo test -p turbospark-streaming
```

## Tests

- `tests/expert_cache.rs`: Validates cache hit/miss accounting, LRU/LFU eviction order, and capacity limits without requiring large disk files.
- `tests/mapped_experts.rs`: Verifies memory-mapped in-place expert reading and slice safety.
- `tests/pread_streamer.rs`: Tests file-backed pread streaming, chunk parallel reads, and readahead advice triggers.

## Crate Gotchas

1. **Decoupled Cache Logic**: The `ExpertCache` eviction policy contains pure cache state logic without direct file I/O calls. This enables testing eviction behavior and trace hit rates in unit tests without requiring multi-gigabyte checkpoint files.
2. **Chunk Parallelism**: `read_pool` splits individual expert blob reads into chunk-sized pieces across parked worker threads so that even a single expert cache miss executes with wide I/O parallelism.
3. **Thread Pool Synchronization**: Raw pointer safety in `read_pool` relies on `run_batch` blocking execution until all workers complete or drop their claims before buffers are accessed by the GPU dispatcher.
4. **Platform-Specific I/O Hooks**: `F_RDADVISE` is macOS-only, while `linux_uring` compiles only on Linux targets; both compile to safe fallbacks or no-ops when compiling on non-matching operating systems.
