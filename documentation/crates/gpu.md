---
description: "Metal GPU backend: pipeline cache, kernel dispatch, and per-family shaders for the macOS runtime, validated against the compute crate's CPU reference kernels."
icon: bolt
---

# gpu

`crates/gpu` (package `turbospark-gpu`) is the Metal boundary of the engine. It
owns the device and command-queue context, the MSL shader sources under
`src/shaders/`, and one dispatch module per kernel group, and every dispatch is
parity-tested against the CPU reference kernels in `crates/compute`. The
runtime drives it for every decoded token and the image crate uses it for the
vision path. It requires macOS and a Metal-capable device; on other platforms
the crate compiles to almost nothing so the workspace still builds on Linux CI.

## What it does

| Area | Modules | Contents |
| --- | --- | --- |
| Device context | `src/context/` | `MetalContext` (device, queue, in-memory pipeline cache), `PassEncoder` and `CommittedPass`, dispatch helpers, buffer read/write helpers, `GpuError`, `autorelease_pool` |
| Dequant GEMV | `dequant_int4_gemv`, `dequant_int8_gemv`, `dequant_int1_gemv`, `dequant_int2_gemv`, `dequant_iq_gemv`, `dequant_q2_k_gemv` ... `dequant_q8_0_gemv`, `gemv_bf16` | One module per weight format: affine INT4/INT8/INT1/INT2, the GGUF IQ and K-quant families, and BF16, each with a one-shot wrapper, an encoder, a resident-matrix view, and an embedding lookup |
| Batched GEMM | `dequant_int4_batch`, `dequant_int8_batch`, `dequant_int1_gemm_batch`, `dequant_int2_gemm_batch`, `dequant_int4_steel` | Prefill-shaped batched matrix products, MMA variants, and the Steel tile pipeline (`BatchedScratch::steel_gemm`) |
| Attention | `attention_decode`, `attention_indexed`, `attention_tq`, `mla` | Two-pass split-KV decode (`attention_decode_gqa_partial` + `attention_decode_combine`), indexed and TurboQuant KV variants, MLA absorbed decode |
| MoE | `moe_decode`, `moe_gguf/`, `moe_prefill_batch`, `moe_prefill_batch_gguf` | Two-phase routed experts reading expert blobs zero-copy through an argument buffer, per-format phase-1 kernels, the Gemma router GEMV, batched prefill |
| State and KV | `kv_cache`, `kv_cache_mem`, `kv_cache_rewind`, `kv_quantize`, `kv_quant_tables`, `gdn`, `gdn_state`, `dsv4_state`, `qsa_indexer` | `KvCacheManager` with ring addressing and optional TurboQuant quantization, GDN linear-attention state, DSV4 state manager |
| Elementwise and position | `rms_norm`, `rope`, `utility`, `logit_softmax`, `hadamard`, `hyper_connection`, `ple` | Normalization (plain, centered, grouped, per-head), RoPE variants, activations, residual adds, gating, softcap, FWHT |
| Media frontends | `vision`, `whisper_conv`, `whisper_encoder`, `moonshine_frontend`, `music3/`, `dflash_conv` | Kernels for the vision tower, Whisper, Moonshine, Music 3 packed linears, and the DFlash drafter |
| Host-side helpers | `resident_metal`, `device_memory`, `power_state`, `warmup`, `dispatch_profile`, `prefill_scratch` | Zero-copy resident weight wrap, working-set and thermal queries, opt-in kernel warmup, dispatch profiling, chunked-prefill scratch layouts |

The shaders live in `src/shaders/*.metal` and each dispatch module names its
source with `include_str!`. `build.rs` extracts those same compositions from
the source tree and compiles them at build time, so a shader that no longer
compiles fails the Rust build before anything reaches a device.

## Where it sits

Depends on `turbospark-core` (aliased `foundation`, for `TokenId`) and
`turbospark-compute` always; the macOS target adds `metal`, `half`,
`turbospark-model-io` (resident buffers and KV strides), and `libc`. The
compute dependency is mostly the parity contract: `turbospark-compute`'s
kernels are the CPU reference every Metal dispatch is tested against, and the
one real runtime use is `kv_quant_tables`, which builds the TurboQuant
codebook tables host-side at model open. `turbospark-repack` and
`turbospark-audio` are macOS dev-dependencies used by the parity fixtures.

Used by `runtime` (the decode and prefill path), `image` (the vision path),
and `bench` (measurement harnesses).

## Public API tour

### Device context: `MetalContext`

Defined in `src/context/device.rs`. `MetalContext::new()` picks the system
Metal device and creates the command queue; it fails with
`GpuError::NoDevice` when no device exists. The important methods:

| Method | Meaning |
| --- | --- |
| `pipeline(source, function_name, constants, constants_key)` | Compile once and return the `ComputePipelineState` for a function, cached per (source, function, constants fingerprint) |
| `begin_pass()` / `begin_pass_labeled(label)` | Open a `PassEncoder`: many kernels, one command buffer |
| `new_buffer_with_data(bytes)` / `new_output_buffer(len)` | Upload a buffer / allocate an output buffer |
| `buffer_allocation_count()` | Running count of buffer allocations; the decode hot path is required to allocate none |
| `compilation_stats()`, `pipeline_inventory()`, `flush_pipeline_cache()` | Diagnostics and disk-cache publishing |
| `device()`, `queue()`, `wait_until_idle()` | Escape hatches and sync points |

`MetalContext::pipeline` is where the specialization contract lives. Metal
function constants have no introspection, so the caller passes a byte
fingerprint (`constants_key`) of every value that went into the
`FunctionConstantValues`; two calls with different constants then cache as
distinct pipelines. The cache key for the source is the `&'static str`'s
address, so always pass the same `include_str!(...)` binding for a given
shader file.

### Passes, dispatch, and buffer IO

`PassEncoder` (`src/context/pass.rs`) batches kernel launches into one
command buffer and commits with `commit_and_wait()`, `commit_and_wait_checked()`
or `commit()` (returning a `CommittedPass` you can wait on later).
`src/context/dispatch.rs` provides `dispatch_one_threadgroup_per_row` and
`dispatch_threads_3d` grid helpers, and `src/context/buffer_io.rs` provides
`write_buffer_bytes`, `read_buffer_bytes`, and `read_buffer_f16` for host
readback. The crate re-exports `metal::Buffer` as `MetalBuffer` so downstream
crates can hold GPU buffers without depending on `metal` themselves.

### Errors and the autorelease pool

`GpuError` (`src/context/error.rs`) covers device, compile, pipeline, buffer,
argument-ABI, and input errors. `autorelease_pool(f)` runs a closure inside an
Objective-C autorelease pool; see the gotchas below for why it is not
optional.

### Resident weights and KV cache

`ResidentGpuWeights::wrap` (`src/resident_metal.rs`) turns the whole mmap'd
resident weight region into a single shared-storage `MTLBuffer` with
`newBufferWithBytesNoCopy`: nothing is copied, and tensors are addressed as
offsets via `gpu_offset`. `wrap_page_aligned_no_copy` is the per-slot half of
the same trick for streamed experts. `KvCacheManager` (`src/kv_cache.rs`)
allocates per-layer K/V buffers with ring addressing, optional sliding-window
capacity, and optional TurboQuant quantization (`new_with_kv_quant`, with
`LayerKind` selecting full vs sliding-window layers).

## Examples

All three examples need macOS and a Metal device; the parity tests they are
adapted from are gated with `#![cfg(target_os = "macos")]`.

### Validate a kernel against the CPU reference

Adapted from `crates/gpu/tests/rms_norm_parity.rs`. Every kernel module
follows this shape: run the `turbospark-compute` CPU reference on the same
input and score the readback with the tolerance helpers.

```rust
use half::f16;
use turbospark_gpu::{rms_norm_no_scale, MetalContext};

let mut context = MetalContext::new().expect("Metal device available on this machine");
let x_f32: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.03).collect();
let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

let cpu = turbospark_compute::rms_norm(&x_f32, &vec![1.0f32; 256], 1e-6);
let gpu = rms_norm_no_scale(&mut context, &x_f16, 1e-6)
    .expect("GPU dispatch succeeds");

let err = turbospark_compute::max_abs_diff(
    &gpu.iter().map(|v| v.to_f32()).collect::<Vec<f32>>(),
    &cpu,
);
assert!(err < turbospark_compute::Tolerance::FP16_REDUCTION, "err = {err}");
```

### Encode a kernel through a PassEncoder

Also from `crates/gpu/tests/rms_norm_parity.rs`: the `encode_*` half of each
module takes buffers and offsets instead of host slices, so several kernels
share one command buffer and one commit.

```rust
use half::f16;
use turbospark_gpu::{encode_rms_norm_no_scale, read_buffer_f16, MetalContext};

let mut context = MetalContext::new().expect("Metal device available on this machine");
let d = 64usize;
let x_f16: Vec<f16> = (0..d).map(|i| f16::from_f32((i as f32 - 32.0) * 0.07)).collect();
let x_buf = context.new_buffer_with_data(
    &x_f16.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<u8>>(),
);
let out_buf = context.new_output_buffer((d * 2) as u64);

let pass = context.begin_pass();
encode_rms_norm_no_scale(&mut context, &pass, (&x_buf, 0), (&out_buf, 0), d as u32, 1e-6)
    .expect("encode");
pass.commit_and_wait();

let out: Vec<f32> = read_buffer_f16(&out_buf, 0, d)
    .iter()
    .map(|v| v.to_f32())
    .collect();
```

### Guard a repeated encode loop with autorelease_pool

Adapted from the timed loops in `crates/gpu/tests/gemv_bandwidth_bench.rs`
(simplified). Any loop that encodes work repeatedly must run inside
`autorelease_pool`, and a steady loop should reuse its buffers.

```rust
use turbospark_gpu::{autorelease_pool, encode_rms_norm_no_scale, MetalContext};

let mut context = MetalContext::new().expect("Metal device available on this machine");
// x_buf and out_buf prepared as in the previous example.
let repeats: u32 = 1000;
autorelease_pool(|| {
    let pass = context.begin_pass();
    for _ in 0..repeats {
        encode_rms_norm_no_scale(&mut context, &pass, (&x_buf, 0), (&out_buf, 0), d as u32, 1e-6)
            .expect("encode");
    }
    pass.commit_and_wait();
});
```

## Gotchas

{% hint style="warning" %}
Every value baked into a shader as a function constant MUST be part of the
`constants_key` passed to `MetalContext::pipeline`. The split-KV GQA partial
kernel (`attention_decode_gqa_partial`) bakes group, simdgroups, head_dim, and
unroll on top of the shared (scale, ring capacity, chunk count, sinks) set,
and all of them are in its key: a pipeline reused across shapes would read the
wrong number of Q rows or lanes, and the failure is silent rather than an
error.
{% endhint %}

{% hint style="warning" %}
Gate every repeated Metal encode loop with `autorelease_pool`. Command buffers
and encoders are autoreleased Objective-C objects; a plain Rust binary has one
pool around `main`, so without an inner pool every command object the process
ever creates stays alive until exit and memory grows without bound on long
decode runs.
{% endhint %}

More rules from the crate docs: gated GDN normalization uses exactly 128
threads per threadgroup, a layout contract that must be preserved. Steel
(`dequant_int4_steel`) is prefill-only, is not bit-exact against the scalar
INT4 kernels, and its shader is generated from pinned MLX sources by
`scripts/vendor_mlx_steel_qmm.py`: change the script and regenerate, never the
shader. The multi-row GQA partial serves the chunked-prefill scratch only,
never decode or speculative verify. GDN snapshots are GPU blits that do not
wait, so no host write to live state may land before a later command buffer on
the same queue has completed. `TURBOSPARK_ATTN_DECODE_KERNEL=barrier|gqa`
overrides the per-family decode-attention choice process-wide, and the
optional on-disk pipeline archive is enabled with
`TURBOSPARK_METAL_PIPELINE_CACHE=1` (directory via `TURBOSPARK_METAL_CACHE_DIR`).

## Related pages

- [Crate overview](overview.md)
- [compute](compute.md)
- [runtime](runtime.md)
- [Hardware Acceleration](../models/hardware-acceleration.md)
- [Testing and Verification](../developer/testing-and-verification.md)
