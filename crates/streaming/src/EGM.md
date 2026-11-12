# turbospark-streaming (src)

## Overview
Expert-weight streaming for MoE models: layout of packed experts, mmap and pread backends, an expert slot cache (Lru, Lfu, LFU with decay; default `Lfu` via [expert_cache.rs:ExpertCachePolicy] `DEFAULT`), aligned wired slots, a read thread pool, an unwired io_uring seam for Linux (`Mode` parsing, `Sqe`, DIO alignment helpers; no streamer reads through it), and read-advice/IO accounting.

## Interface
- Layout: [stream_layout.rs:StreamLayout] (`from_packed_experts_layer`, `from_packed_layer_in`, `expert_offset`).
- Mapped backend: [mapped_experts.rs:MappedExpertLayer] (`open`, `expert_bytes`, `prefetch_expert`, `prepare_selected_experts`, `page_aligned_bytes` (the pointer a caller wraps), `expert_stride`, `experts_per_layer`, `layout`), `MappedPagePreparationMode`, `MappedPagePreparationStats`.
- Pread backend: [pread_streamer.rs:PreadExpertStreamer] (`open`, `load_expert`, `load_experts_cached`, `plan_experts_cached`, `execute_expert_cache_plan`, `plan_experts_cached_if_possible`, `wire_slots`, `reserve_speculative_slots`, `advise_experts`, `io_stats`, `prefetch_expert` (refused under `F_NOCACHE`), `nocache_active` (the truth to report), `resident_experts_snapshot`, `cache_policy`, `layout`, `non_resident_experts`, `slot_data`, `slot_allocation` (raw pointer and length seam), `load_expert_into_slot`, `read_experts_into_bank`, `execute_speculative_reservation`, `advise_expert_misses`, `advise_expert_cache_plan_misses`).
- Cache: [expert_cache.rs:ExpertCache] (`plan_if_possible` returns `None` when too few slots are evictable but still panics if more experts are requested than `slot_count`, `slot_count`, `non_resident_experts`, `plan` the panicking variant, `commit_plan` the commit path, `reserve_speculative_slots`, `publish_speculative_reservation`, plus `resident_experts_snapshot`, `cache_policy`), [expert_cache.rs:ExpertCachePolicy] (`from_env`, `lfu_decay_from_str`, `describe`), [expert_cache.rs:ExpertCachePlan], [expert_cache.rs:ExpertIoAdviceResult], [expert_cache.rs:coalesced_adjacent_advice_ranges].
- Memory: [aligned_slot.rs:AlignedSlot] (`as_ptr`, `len`), [aligned_slot.rs:wire_limit_bytes].
- IO stats and advice: [disk_io.rs:ExpertIoStats] (`accumulate`, `amplification`), [rdadvice.rs:call], [rdadvice.rs:clipped_byte_count], [rdadvice.rs:RdAdviceCallResult].
- Linux IO: [linux_uring.rs:Mode] (re-exported as `LinuxIoMode`; `parse`, `from_env_default`), [linux_uring.rs:dio_aligned], [linux_uring.rs:dio_round_up], [linux_uring.rs:DIO_ALIGNMENT].
- Errors: [error.rs:StreamerError]. Re-export of `foundation::TokenId` in [lib.rs].
- Internal: [read_pool.rs:run_batch] (crate-private pool of 8 threads).

## Invariants
- `slot_count` must be positive [expert_cache.rs:ExpertCache], [pread_streamer.rs:PreadExpertStreamer].
- A slot a speculative read is still filling must not be handed out [expert_cache.rs:ExpertCache].
- Expert prefetch conflicts with `F_NOCACHE` and is refused; `TURBOSPARK_EXPERT_NOCACHE=1` alone does not guarantee a disk-bound read [pread_streamer.rs:PreadExpertStreamer], [disk_io.rs].
- Every `ReadChunk` in a batch addresses a byte range no other chunk touches (basis of the `Send`/`Sync` claim) [read_pool.rs:ReadChunk].
- io_uring `Sqe` is exactly 64 bytes [linux_uring.rs:Sqe].
- The mapped buffer wrapper must not outlive its mapping [mapped_experts.rs:MappedExpertLayer].

## Side Effects
- Reads env vars: `TURBOSPARK_EXPERT_CACHE_DECAY` [expert_cache.rs:ExpertCachePolicy], `TURBOSPARK_LINUX_IO` (auto, pread, uring) [linux_uring.rs:Mode] (parse-only seam: `from_env` has no production caller and nothing selects `Mode::Uring` yet), `TURBOSPARK_EXPERT_DISK_IO`, `TURBOSPARK_EXPERT_NOCACHE` [disk_io.rs], `TURBOSPARK_READ_QOS` [read_pool.rs].
- Opens and preads model files, sets `F_NOCACHE` only when `TURBOSPARK_EXPERT_NOCACHE=1` and the `fcntl` succeeds [disk_io.rs:nocache_requested], uses `posix_memalign` allocations [aligned_slot.rs:AlignedSlot]. Slot pages are `mlock`ed only when the caller invokes `wire_slots`, never in `open` [pread_streamer.rs:PreadExpertStreamer::wire_slots].
- Lazily spawns a process-wide read thread pool on the first multi-chunk batch; single-chunk batches run inline on the caller [read_pool.rs:POOL_THREADS], [read_pool.rs:run_batch].

## Verification
- Run: `cargo test -p turbospark-streaming`
