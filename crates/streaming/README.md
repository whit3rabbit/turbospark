# turbospark-streaming

Routed-expert streamer (`PreadExpertStreamer`) for loading MoE expert weights on demand via `pread`, coupled with a per-layer slot cache (`ExpertCache`), a process-wide worker pool for parallel chunk reading (`read_pool.rs`), and kernel readahead advice (`rdadvice.rs`).

Downstream workspace crates import this package via the `streaming` alias:

```toml
[dependencies]
streaming = { package = "turbospark-streaming", path = "../streaming" }
```

## Safety

- Contains `unsafe` code in `rdadvice.rs` for calling macOS `F_RDADVISE` via `fcntl`, in `disk_io.rs` for `proc_pid_rusage` and the `F_NOCACHE` `fcntl`, and in `read_pool.rs` for managing raw pointers across parked worker threads.

## Key Modules

- `pread_streamer.rs`: `PreadExpertStreamer` for streaming expert weight blobs on demand from `packed_experts/blobs.bin`.
- `expert_cache.rs`: `ExpertCache` implementing pure LFU/LRU eviction policy.
- `read_pool.rs`: Process-wide pool of parked reader threads (`run_batch`) for parallel pread of expert weight chunks.
- `rdadvice.rs`: Platform-gated `F_RDADVISE` kernel hint wrapper (macOS `fcntl`; safe no-op on other operating systems).
- `mapped_experts.rs`: `MappedExpertLayer`, the routed experts read in place out of an `mmap` per layer instead of `pread`-copied into a pinned slot.
- `disk_io.rs`: Physical-disk-read accounting (`ExpertIoStats`) and the `F_NOCACHE` seam that forces a disk-bound read, both off by default.
- `stream_layout.rs`: Expert blob offset and byte layout calculation helpers.

## Development & Test Commands

```sh
# Run tests for turbospark-streaming
cargo test -p turbospark-streaming
```

## Crate Gotchas

1. **Decoupled Cache Logic**: The `ExpertCache` eviction policy contains pure cache state logic without direct file I/O calls. This enables testing eviction behavior and trace hit rates in unit tests without requiring full multi-gigabyte checkpoint files.
2. **Chunk Parallelism**: `read_pool` splits expert blob reads into chunk-sized pieces across parked worker threads so even a single cache miss executes with wide I/O parallelism.
3. **Thread Pool Safety**: `read_pool` raw pointer safety relies on `run_batch` blocking execution until all workers complete or drop their claims.
