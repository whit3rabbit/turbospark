# turbospark-streaming

Routed-expert weight streaming engine (`PreadExpertStreamer`) for loading MoE expert weights on demand via `pread`, coupled with a per-layer slot cache (`ExpertCache`), a parked worker thread pool for parallel chunk reading (`read_pool.rs`), kernel readahead advice (`rdadvice.rs`), memory-mapped in-place expert reading (`mapped_experts.rs`), and a Linux `io_uring` + `O_DIRECT` read source (`linux_uring.rs`; the ring is proven by a Linux-only test, and nothing selects it yet).

Downstream workspace crates import this package via the `streaming` alias:

```toml
[dependencies]
streaming = { package = "turbospark-streaming", path = "../streaming" }
```

## Purpose & Role

In Mixture-of-Experts (MoE) models, total parameter counts frequently exceed available RAM. `turbospark-streaming` implements high-bandwidth on-demand streaming of expert weight blobs from NVMe disk into RAM or Metal buffers during token decode, decoupling model capacity from physical memory limits.

## Safety

- Contains `unsafe` code strictly bounded to:
  - `rdadvice.rs`: Calling macOS `fcntl` with `F_RDADVISE` for kernel readahead hints, and Linux `posix_fadvise(WILLNEED)` (the platform analogue); a documented no-op on other operating systems.
  - `disk_io.rs`: Calling `proc_pid_rusage` and setting `F_NOCACHE` via `fcntl` for hardware I/O profiling.
  - `read_pool.rs`: Sharing raw pointer destinations across parked worker threads during coordinated parallel chunk reads.
  - `aligned_slot.rs`: `posix_memalign`/`free` slot allocation and raw slice views over the allocation.
  - `pread_streamer.rs`: Chunk-destination pointer arithmetic feeding the read pool.
  - `mapped_experts.rs`: `sysconf` page-size queries, `madvise(WILLNEED)`, and volatile page touches on the read-only mapping.
  - `linux_uring.rs`: Raw `io_uring_setup`/`io_uring_enter` through `libc::syscall` (no liburing binding) and the ring `mmap`s, with the UAPI constants pinned by a Linux-only test.

## Key Modules

- `pread_streamer.rs`: `PreadExpertStreamer` for streaming expert weight blobs on demand from the layer's file under `packed_experts/` (per-layer names like `layer_00.bin`; the vision tower streams `packed_vision/blobs.bin` through the same layout).
- `expert_cache.rs`: `ExpertCache` implementing deterministic LFU and LRU slot eviction algorithms, plus an opt-in decaying LFU policy (`LfuDecay`, selected by `TURBOSPARK_EXPERT_CACHE_DECAY=<N>`) whose frequency counts halve every `N` plans so a stale hot set stops outranking the current one.
- `read_pool.rs`: Global pool of parked reader threads (`run_batch`) providing wide parallel I/O for single and batch expert cache misses.
- `rdadvice.rs`: Platform-gated kernel readahead advice wrapper (macOS `F_RDADVISE` `fcntl`, Linux `posix_fadvise(WILLNEED)`; documented no-op on other operating systems).
- `mapped_experts.rs`: `MappedExpertLayer`, enabling routed experts to be read directly in place from an `mmap` per layer instead of `pread`-copied into pinned slots.
- `aligned_slot.rs`: Page-aligned buffer allocations required for unbuffered direct disk I/O.
- `disk_io.rs`: Physical hardware read accounting (`ExpertIoStats`) and the `F_NOCACHE` profiling seam.
- `stream_layout.rs`: Expert blob offset, stride, and byte layout calculation helpers.
- `linux_uring.rs`: Linux `io_uring` submission/completion queue implementation (the cgroup memory probe lives in `crates/model-io/src/cgroup.rs`).

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-streaming
cargo test -p turbospark-streaming
```

## Tests

- `tests/expert_cache.rs`: Validates cache hit/miss accounting, LRU/LFU eviction order, decaying-versus-cumulative LFU behavior on a drifting hot set, and capacity limits without requiring large disk files.
- `tests/mapped_experts.rs`: Verifies memory-mapped in-place expert reading and slice safety.
- `tests/pread_streamer.rs`: Tests file-backed pread streaming, chunk parallel reads, and readahead advice triggers.
- `tests/expert_prefetch.rs`: Validates that streamed and mapped bounded prefetch reads touch real ranges without changing slot-cache ownership or residency.
- `tests/mapped_page_preparation.rs`: Verifies demand-selected page preparation (touch and advice modes) is bounded, validated before touching pages, and read-only.

## Crate Gotchas

1. **Decoupled Cache Logic**: The `ExpertCache` eviction policy contains pure cache state logic without direct file I/O calls. This enables testing eviction behavior and trace hit rates in unit tests without requiring multi-gigabyte checkpoint files.
2. **Chunk Parallelism**: `read_pool` splits individual expert blob reads into chunk-sized pieces across parked worker threads so that even a single expert cache miss executes with wide I/O parallelism.
3. **Thread Pool Synchronization**: Raw pointer safety in `read_pool` relies on `run_batch` blocking execution until all workers complete or drop their claims before buffers are accessed by the GPU dispatcher.
4. **Platform-Specific I/O Hooks**: `F_RDADVISE` is macOS-only (Linux takes the `posix_fadvise` analogue, other systems a documented no-op), and only `linux_uring`'s raw ring module is compiled on Linux targets; its selector and alignment helpers compile and are unit-tested on every host.
