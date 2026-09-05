---
uuid: "0fe75c5c-0632-4a32-8ea7-a5f2ae23bec8"
title: "turbospark-gpu"
summary: "Metal device context, pipeline cache, per-kernel dispatches, and the KV cache. macOS only, gated per-module, reduces to an empty crate elsewhere"
tags: ["crate", "gpu"]
depends_on: ["09ef101c-e821-4ff5-95ff-3bb53b5e05a3"]
source: "crates/gpu/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-gpu do?

Owns the Metal device context (`MetalContext`), the pipeline cache, command
pass encoders (`PassEncoder`, `CommittedPass`), the KV cache manager
(`KvCacheManager`), zero-copy resident weight buffers, and every per-kernel
dispatch (attention, MoE routing, RMSNorm, RoPE, dequant GEMVs for every
quant width, GDN, vision tower). Every source file is gated
`#[cfg(target_os = "macos")]`, so the crate compiles to an empty module off
macOS. Building or testing it needs a real Metal-capable device and
Xcode's `metal` toolchain (`xcrun -sdk macosx metal`).
`cargo test -p turbospark-gpu` is the whole dev loop.

## Don't

- Don't pass a shader source string to `MetalContext::pipeline` that isn't
  the exact same `&'static str` (an `include_str!` constant) every time.
  The pipeline cache keys on string memory ADDRESS, not contents, so two
  calls with byte-identical but differently-sourced strings silently miss
  the cache rather than reuse it.
- Don't write a repeated Metal encode loop (a command buffer or compute
  encoder created per iteration) without wrapping it in
  `gpu::autorelease_pool`. `commandBuffer`/`computeCommandEncoder` return
  autoreleased objects, and without an inner pool every one created in the
  process's life stays alive until exit (measured at ~180 KiB per decoded
  token on real hardware).
- Don't add a specialization axis (a mode, a width, a capability flag) as a
  `MTLFunctionConstant` without putting its value into the pipeline cache's
  `constants_key`. A byte that misses the key means two different
  configurations silently share one compiled pipeline: whichever compiled
  first wins on every later dispatch. This has actually happened (a KV ring
  capacity, a norm convention, a batch shape) and reads as correct-looking
  wrong output, never an error.
- Don't let a `?` or an early return happen between `begin_pass` and
  `commit` on a `PassEncoder` without going through its own `Drop`.
  Dropping an encoder that never got `endEncoding` sent aborts the process
  from inside `-[_MTLCommandEncoder dealloc]`, burying the real error that
  was still propagating up the call stack.
- Don't change `gdn_qk_norm` or `gdn_gated_norm`'s threadgroup size away
  from exactly 128 threads. Both reduce four SIMD partials with a hardcoded
  loop bound: fewer threads leaves slots uninitialized, more silently drops
  work. `NORM_THREADS` is the pin.
- Don't add a fence for a GPU-only intermediate one command buffer writes
  and the next on the SAME queue reads. Same-queue buffers execute in
  commit order already. The rule inverts for anything the HOST writes or
  reads back (routing weights, an argument buffer): that traffic never
  enters the queue and CAN race a buffer still in flight.
