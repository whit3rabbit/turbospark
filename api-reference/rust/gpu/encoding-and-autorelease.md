---
description: "How to encode Metal dispatches with PassEncoder, choose a commit and wait form, and wrap repeated loops in autorelease_pool."
icon: box
---

# Passes and autorelease pools

{% hint style="warning" %}
This page covers the low-level encode path. It applies only on macOS with a Metal-capable device. If you are running a model, use [turbospark-runtime](../runtime/README.md) and skip this page.
{% endhint %}

Open one pass per group of dispatches, encode every kernel into it, then commit once. Wait only when the CPU needs the output. Wrap every repeated encode loop in `autorelease_pool`, because Metal's command buffers and encoders are autoreleased objects. Skipping the pool does not fail a test or a build, because nothing crashes at first. Instead, a long-running process grows until it exits, and the growth is easy to miss in a short run.

## The pass lifecycle

`MetalContext::begin_pass` and `begin_pass_labeled` open one command buffer and one serial compute encoder. The label names the buffer in a GPU capture. Each `encode_*` call appends one dispatch to that encoder. Nothing runs until you commit.

Dispatches run in the order you encode them, and the source documents serial order as the guarantee that each dispatch sees the previous one's writes. `memory_barrier_with_buffers` is available for the case where a later dispatch reads output from an earlier one in the same pass. Test that path on a device before you rely on either behavior.

When the work is done, pick one finish:

- `commit_and_wait_checked()` blocks, then returns `Err(GpuError::InvalidInput)` if the command buffer did not complete. Use this when you read the output.
- `commit_and_wait()` blocks and does not return an error. It prints to stderr if the buffer fails. Prefer the checked form.
- `commit()` returns a `CommittedPass` without waiting. Call `wait()` or `wait_ref()` on it before you read any shared-storage output. Buffers committed to one queue run in commit order, so you can queue more work and wait on an earlier handle.
- `commit_and_wait_with_gpu_time()` blocks and returns the GPU-side busy time in seconds. Use it for measurement, not correctness.

Shared-storage output is CPU-readable only after the pass that wrote it completes. Reading before that point is a race, and the bytes you see may be partial.

## Dropping a pass

A `PassEncoder` dropped without a commit ends its encoder. The work it encoded never reaches the GPU. Every `?` in an encode sequence takes this path. Without the drop handler, Metal aborts the process with "Command encoder released without endEncoding", and the real error never reaches a caller.

`CommittedPass` works differently. Dropping it does not wait, so the GPU may still be running when the handle goes away. If the resources a pass uses can be released after an error, call `waiting_on_drop()` on the handle.

Metal keeps its own references to buffers. It does not keep external bytes-no-copy memory alive. Wrapped memory, such as a `ResidentGpuWeights` mapping, must outlive every pass that reads it.

## Repeated loops and autorelease pools

Every repeated encode loop needs `autorelease_pool`. `MTLCommandQueue.commandBuffer` and `computeCommandEncoder` return autoreleased objects. A Rust binary has one pool around `main`, so without an inner pool, every command buffer stays alive until the process exits. The runtime wraps its own encode loops in the same way, so you only need the pool in loops you write yourself.

The pool takes a closure and returns its value, so `?` works inside it. This example runs one thousand encode passes. It compiled and ran on an Apple M4 Max with a path dependency on this checkout. The buffer starts at 1.0 and gains 2.0 per pass, so the final print shows `[2001.0, 2001.0, 2001.0, 2001.0]`.

```rust
use turbospark_gpu::{autorelease_pool, encode_residual_add, read_buffer_f16, GpuError, MetalContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = MetalContext::new()?;
    let hidden = context.new_buffer_with_data(&[0x3C00u16; 4]); // 1.0
    let delta = context.new_buffer_with_data(&[0x4000u16; 4]); // 2.0

    for _ in 0..1000 {
        autorelease_pool(|| -> Result<(), GpuError> {
            let pass = context.begin_pass_labeled("residual");
            encode_residual_add(&mut context, &pass, (&hidden, 0), (&delta, 0), 4)?;
            pass.commit_and_wait_checked()
        })?;
    }

    println!("{:?}", read_buffer_f16(&hidden, 0, 4));
    Ok(())
}
```

Place the pool around one iteration, not around the whole loop. Keep it small. A pool around the whole loop still holds every iteration's objects until the loop ends. In a decode loop, that usually means one pool per token.

Helpers such as `dispatch_one_threadgroup_per_row` and `dispatch_threads_3d` create their own command buffer and wait for it. They also wrap their work in a pool, so you can call them directly for one-off checks. Inside a loop, use `begin_pass` so several dispatches share one command buffer.

## Custom pipelines

`MetalContext::pipeline` compiles and caches a compute pipeline for one function in an MSL source string. Its signature is:

```rust
pub fn pipeline(
    &mut self,
    source: &'static str,
    function_name: &'static str,
    constants: &FunctionConstantValues,
    constants_key: &[u8],
) -> Result<ComputePipelineState, GpuError>
```

Three rules apply:

- `source` must be `'static`. Load it with `include_str!` so the same address is used for every call. The cache keys on the source address, not its text.
- `FunctionConstantValues` comes from the `metal` crate. Add `metal = "0.33"` to match this crate. Metal requires every function constant the shader declares to be set before it builds the pipeline, even constants the shader guards and never reads.
- `constants_key` must include every value that changes the specialization. Two calls with different constants and the same key return the same cached pipeline, with no error. A ring capacity or layer shape reused this way is a silent correctness bug.

Compilation happens on first use. The first call to a new kernel pays the compile cost. When the disk cache is enabled, `MetalContext` flushes new specializations to it on drop. `pipeline_inventory` lists the cached specializations if you need to confirm which ones a run created.

## Checks

- Measure memory across many passes. The repository's steady-state test in [memory_oracle.rs](https://github.com/whit3rabbit/turbospark/blob/main/crates/bench/tests/memory_oracle.rs) is the check that catches a missing pool. Run it after you add a loop.
- Compare kernel output with a CPU reference on real hardware before you trust a new dispatch.
- When a run produces wrong output with no error, look at stderr. A `[gpu] command buffer ... finished with status` line means the GPU aborted the work, and the outputs cannot be trusted.

## Related

- [turbospark-gpu overview](README.md) lists the entry points and the cache and weight types.
- The [docs.rs reference](https://docs.rs/turbospark-gpu/latest/turbospark_gpu/) lists every public function and type.
