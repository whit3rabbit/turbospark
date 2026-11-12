# image shaders

## Overview
Metal Shading Language source for the native image pipeline (Z-Image
transformer, text encoder helpers, scheduler step, and VAE decoder). It is
embedded into the Rust binary at build time and compiled at runtime by the
Metal backend in the parent directory.

## Interface
- [image.metal:image_round_bf16] BF16 rounding of f32 buffers.
- [image.metal:image_linear], [image.metal:image_linear_tiled], [image.metal:image_linear_simd] linear (matmul) variants.
- [image.metal:image_rms_norm], [image.metal:image_layer_norm], [image.metal:image_group_norm] normalization kernels.
- [image.metal:image_lookup] embedding row lookup.
- [image.metal:image_rope_orthogonal], [image.metal:image_rope_adjacent] rotary embedding variants.
- [image.metal:image_attention], [image.metal:image_vae_attention] attention kernels.
- [image.metal:image_add], [image.metal:image_silu_mul], [image.metal:image_scale_shift], [image.metal:image_gate_add], [image.metal:image_silu] elementwise ops.
- [image.metal:image_scheduler_step] flow-match Euler update on device.
- [image.metal:image_conv2d], [image.metal:image_upsample_nearest] VAE convolution and 2x upsample.

## Invariants
- Every kernel declares `max_total_threads_per_threadgroup`: 256, except `image_attention` at 512, so host dispatch must respect that cap [image.metal:image_round_bf16, image.metal:image_attention].
- Elementwise kernels take a `constant uint &` count and guard `index < count`; the rest take a params struct and guard on rows, dims, or totals [image.metal:image_round_bf16, image.metal:image_add].
- Cooperative kernels need an exact threadgroup size, not just a cap: `image_rms_norm`, `image_rope_adjacent`, `image_group_norm`, `image_vae_attention` need 256 threads; `image_attention` needs 4 * head_dim threads and supports head_dim <= 128; the host also requires head_dim % 32 == 0 and q_heads % kv_heads == 0 [image.metal:image_attention, metal_ops.rs:attention].
- `image_vae_attention` holds `numerator[16]` per lane across 32 lanes, so the kernel covers at most 512 channels and would drop more. The host wrapper refuses channels outside 1..=512, channels not a multiple of 8, and an area that is zero or not a multiple of 64, so production callers never reach the drop [image.metal:image_vae_attention, metal_ops.rs:vae_attention].
- The host includes this file with `include_str!`; a kernel rename requires a matching change in the Rust pipeline setup [../metal_ops.rs:SOURCE].

## Side Effects
- None (pure device code; buffers are written only through caller-bound `device` arguments).

## Verification
- Run: `cargo test -p turbospark-image`
