---
description: "Low-level Metal GPU crate: device context, pass encoders, kernel dispatch, KV cache, and resident weight buffers."
icon: box
---

# turbospark-gpu

{% hint style="warning" %}
`turbospark-gpu` runs only on macOS with a Metal-capable device. On other platforms the Metal items are absent, so the crate only re-exports `TokenId`. It is also a low-level layer. You manage Metal buffers, encode kernels by hand, and track model state yourself.
{% endhint %}

Most integrators should use [turbospark-runtime](../runtime/README.md) instead. The runtime owns prefill, decode, and the model families, and it calls this crate for you. Reach for `turbospark-gpu` when you need to encode your own Metal passes, call one TurboSpark kernel directly, or manage the KV cache and resident weight buffers yourself.

## Requirements

- macOS on Apple Silicon or another Mac with a Metal-capable GPU. `MetalContext::new` returns `GpuError::NoDevice` when no device is available.
- The Metal toolchain from Xcode. Kernels compile from MSL source at first use unless you opt into bundled libraries (see [Environment variables](#environment-variables)).
- Rust 1.82 or newer, which is the workspace `rust-version`.
- Depend on the package `turbospark-gpu` and import it as `turbospark_gpu`. Some crates in this workspace use short aliases such as `model_io` in their own `Cargo.toml` files. Those aliases are internal. Depend on `turbospark-model-io` and import it as `turbospark_model_io` when you need its types.

Add the package from crates.io once the version you need is published. Until then, use a path or git dependency from a checkout. The example below builds against a path dependency:

```toml
[dependencies]
turbospark-gpu = { path = "../turbospark/crates/gpu" }
```

## Entry points

| Type or function | Use it for |
| --- | --- |
| [`MetalContext`](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/struct.MetalContext.html) | Owns the Metal device, the command queue, and the pipeline cache. Create one and keep it for the life of the work. |
| [`PassEncoder`](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/struct.PassEncoder.html) | One command buffer and one serial compute encoder. Encode many dispatches, then commit. |
| [`CommittedPass`](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/struct.CommittedPass.html) | A committed command buffer you can wait on later. |
| `encode_*` functions | One kernel per call, such as [`encode_residual_add`](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/fn.encode_residual_add.html). Most take `&mut MetalContext`, a `&PassEncoder`, and `(buffer, byte offset)` pairs. |
| [`KvCacheManager`](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/struct.KvCacheManager.html) | Persistent per-layer K and V buffers for decode. |
| [`ResidentGpuWeights`](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/struct.ResidentGpuWeights.html) | One zero-copy `MTLBuffer` over a resident weights mapping. |
| [`GpuError`](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/enum.GpuError.html) | The error type for every fallible call in this crate. |
| `autorelease_pool` | Wraps every repeated encode loop. See [Passes and autorelease pools](encoding-and-autorelease.md). |

## Minimal example

This example adds two half-precision vectors on the GPU. It compiled and ran on an Apple M4 Max with a path dependency on this checkout. It prints `after one pass: [3.0, 3.0, 3.0, 3.0]`.

```rust
use turbospark_gpu::{encode_residual_add, read_buffer_f16, MetalContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = MetalContext::new()?;

    // f16 bit patterns: 1.0 is 0x3C00, 2.0 is 0x4000.
    let hidden = context.new_buffer_with_data(&[0x3C00u16; 4]);
    let delta = context.new_buffer_with_data(&[0x4000u16; 4]);

    let pass = context.begin_pass_labeled("residual");
    encode_residual_add(&mut context, &pass, (&hidden, 0), (&delta, 0), 4)?;
    pass.commit_and_wait_checked()?;

    println!("after one pass: {:?}", read_buffer_f16(&hidden, 0, 4));
    Ok(())
}
```

The buffers hold raw bytes, so `new_buffer_with_data` takes any slice. Pass `u16` bit patterns for `f16` data and read results back with `read_buffer_f16` or `read_buffer_bytes`. Both reads are valid only after the pass that wrote them has completed.

## Buffers and weights

`MetalContext::new_buffer_with_data` and `new_output_buffer` allocate shared-storage buffers. Shared storage lets the CPU write and read the same memory, so you do not copy between host and device. Do not allocate on the decode hot path. The context counts every allocation, and `buffer_allocation_count` returns that count for steady-state checks.

`ResidentGpuWeights` wraps a resident weights mapping in one `MTLBuffer` without copying. Build the mapping with `turbospark_model_io::ResidentBuffer::map(file_path, file_offset, resident_size)`. Then call `ResidentGpuWeights::wrap(context.device(), resident)`, which returns `Result<ResidentGpuWeights, GpuError>`. Address a tensor with `gpu_offset(logical_offset)`, which returns the offset into `buffer()`.

`KvCacheManager::new` allocates every layer's K and V buffers when you construct it. Its arguments are the device, a `turbospark_model_io::ArchConfig`, the maximum context length, a ring flag for sliding-window layers, the optional sliding window, the prefill chunk size, and an optional ring capacity override. Use `new_with_kv_quant` to enable TurboQuant.

Four pitfalls apply to the cache:

- KV cache sizing and ring capacity affect correctness, not only memory. Let the constructor size the ring unless you have a specific reason to override it.
- `advance` and `advance_by` panic when the position passes `max_context`.
- `k_slot` and `v_slot` panic on linear and compressed layers, which have no per-token rows.
- `write_k` and `write_v` panic on TurboQuant-quantized layers. Those layers write through `encode_kv_quantize_tq` instead.

For a complete `ArchConfig` literal, read the `toy_arch` helper in the [KV cache test](https://github.com/whit3rabbit/turbospark/blob/main/crates/gpu/tests/kv_cache.rs). The test builds a real `KvCacheManager` on a device.

## Errors

Every fallible function returns `Result<_, GpuError>`. The variants are:

- `NoDevice`: no Metal device on this host.
- `LibraryCompile(String)`: the MSL source did not compile.
- `FunctionNotFound(String)`: the kernel name is not in the library.
- `PipelineCreate(String)`: the pipeline state could not be built.
- `BufferCreate(String)`: Metal rejected a buffer, including a resident mapping that Metal would not wrap.
- `ArgumentBufferMismatch(String)`: an argument buffer does not match the reflected layout it was allocated for.
- `InvalidInput(String)`: bad dimensions, ranges, plans, or buffers. A command buffer that did not complete also reports here through `commit_and_wait_checked`.

`GpuError` implements `std::error::Error`, so `?` converts it into `Box<dyn Error>`.

## Environment variables

`MetalContext::new` reads these variables once:

- `TURBOSPARK_METAL_PRECOMPILED=1` loads bundled precompiled libraries instead of compiling MSL at first use. It has no effect when precise math is on.
- `TURBOSPARK_METAL_PRECISE_MATH` turns off Metal fast math. Use it to separate compiler arithmetic from layout bugs, not for production.
- `TURBOSPARK_METAL_PIPELINE_CACHE=1` enables the on-disk pipeline archive. `TURBOSPARK_METAL_CACHE_DIR` sets its directory.
- `TURBOSPARK_DISPATCH_PROFILE=1` records per-dispatch timings. Profiled runs serialize the GPU, so their timings are not throughput numbers.

## Platform limits

A clean compile on macOS does not show that a kernel is correct on a device. The crate validates its kernels against the CPU reference kernels in [turbospark-compute](https://docs.rs/turbospark-compute/latest/turbospark_compute/). If you add a dispatch, compare its output with a CPU reference on real hardware. The [kernel reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/KERNELS.md) lists which kernels have that coverage.

Metal can finish a command buffer with an error and still return normally from a wait. The crate prints a `[gpu] command buffer ... finished with status` line to stderr in that case. Check stderr when a run produces wrong output with no error, and prefer `commit_and_wait_checked` when you read the results.

## Next

- [Passes and autorelease pools](encoding-and-autorelease.md) covers the encode lifecycle, the commit and wait choices, and custom pipelines.
- [turbospark-runtime](../runtime/README.md) is the supported entry point for running models.
- The [docs.rs reference](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/) lists every public item.
