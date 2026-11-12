# gpu context

## Overview
Metal device, command queue and pipeline-cache context, the pass encoder wrapper, standalone dispatch helpers, buffer I/O helpers, and GPU error types.

## Interface
- [device.rs:MetalContext]: `new`, `pipeline`, `queue`, `device`, `begin_pass`, `begin_pass_labeled`, `begin_pass_concurrent_labeled`, `new_buffer_with_data`, `new_output_buffer`, `wait_until_idle`, `argument_encoder`, `buffer_allocation_count`, `compilation_stats`, `pipeline_inventory`, `flush_pipeline_cache`.
- [pass.rs:PassEncoder]: `encode_threadgroups`, `encode_threadgroups_3d`, `encode_threads_3d`, `use_read_buffer`, `memory_barrier_with_buffers`, `relabel`, `commit`, `commit_and_wait`, `commit_and_wait_checked`, `commit_and_wait_with_gpu_time`; [pass.rs:CommittedPass] (`waiting_on_drop`, `wait`, `wait_ref`, `wait_with_gpu_time`, `wait_gpu_start_end`).
- [pipeline_cache.rs:MetalCompilationStats] and the optional device-built pipeline archive.
- [dispatch.rs:dispatch_one_threadgroup_per_row], [dispatch.rs:dispatch_one_threadgroup_per_row_offsets], [dispatch.rs:dispatch_threads_3d].
- [buffer_io.rs:write_buffer_bytes], [buffer_io.rs:read_buffer_bytes], [buffer_io.rs:read_buffer_f16], [buffer_io.rs:read_buffer_f16_into].
- [error.rs:GpuError] (NoDevice, LibraryCompile, FunctionNotFound, PipelineCreate, BufferCreate, ...), [error.rs:autorelease_pool].

## Invariants
- `PassEncoder` ends encoding in `Drop` when not already ended; without that Metal aborts the process on dealloc [pass.rs:PassEncoder].
- A `CommittedPass` waits for completion on drop when `wait_on_drop` is set and reports command-buffer errors [pass.rs:CommittedPass].
- In-memory pipelines are address keyed, while disk archives are content keyed and include the compiler and device boundary [pipeline_cache.rs] (module doc).
- Repeated encoding loops must use `autorelease_pool` [error.rs:autorelease_pool].

## Side Effects
- Reads env vars `TURBOSPARK_METAL_PRECISE_MATH` (presence-only: any value, including "0", enables it), `TURBOSPARK_METAL_PRECOMPILED` (enabled only by exactly "1") [device.rs:MetalContext], `TURBOSPARK_METAL_PIPELINE_CACHE`, `TURBOSPARK_METAL_CACHE_DIR` [pipeline_cache.rs], and `TURBOSPARK_DISPATCH_PROFILE` [dispatch_profile.rs:enabled] (read outside this module; [pass.rs:PassEncoder] only rebinds per dispatch when it is on).
- Compiles MSL libraries and may write/read pipeline archives on disk only when `TURBOSPARK_METAL_PIPELINE_CACHE=1` and macOS is 15 or later [pipeline_cache.rs:supports_specialized_archives]. Archives load without content verification, so `TURBOSPARK_METAL_CACHE_DIR` must not be writable by other users [pipeline_cache.rs:PipelineDiskCache].
- Creates GPU buffers and command buffers, commits work to the queue [device.rs:MetalContext], [pass.rs:PassEncoder].

## Verification
- Run: `cargo test -p turbospark-gpu`
