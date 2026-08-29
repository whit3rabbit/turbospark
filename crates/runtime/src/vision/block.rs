//! One `qwen3_5` vision block, encoded against a streamer slot (ROADMAP
//! M-V4).
//!
//! The reference, `Qwen3VLMoEVisionBlock.__call__`:
//!
//! ```text
//! h = h + attn(norm1(h))
//! h = h + mlp(norm2(h))
//! ```
//!
//! Post-norm residuals in both halves, no sandwich norms, no gate, no mask.
//! The attention is bidirectional over the WHOLE page: the reference passes
//! `cu_seqlens = [0, seq_len]`, one segment, so every patch attends to every
//! patch and there is nothing to mask.
//!
//! # No qkv split kernel
//!
//! The fused `attn.qkv.weight` is `[3 * hidden, hidden]`, row-major by
//! OUTPUT, so its first `hidden` rows ARE q's projection. Running it as three
//! matmuls at three byte offsets into the same weight writes three contiguous
//! `[seq, hidden]` buffers the rope and attention kernels read directly, with
//! no scatter kernel and identical arithmetic. The reference reshapes to
//! `[seq, 3, heads, head_dim]` and splits on the `3`, which is the same
//! partition. This shape is `crates/gpu/tests/vision_block_parity.rs`'s and
//! is ported rather than rederived.
//!
//! # Two things one mutation away from a fluent wrong model
//!
//! **Rope reaches q and k and NOT v.** A rotated v keeps every shape, stays
//! finite, and reads as a slightly different image.
//!
//! **The MLP's activation is GELU tanh and the merger's is the erf form.**
//! They agree to about 3e-4, which is two orders of magnitude under what a
//! whole-block parity bound can resolve (`crates/gpu` Gotcha 9), so no
//! composition test can see the choice and the call site is where it is
//! pinned.

use crate::real_forward_types::RealForwardError;
use crate::vision::scratch::VisionScratch;
use crate::vision::shape::{VisionShape, VISION_LAYER_NORM_EPS};
use crate::vision::weights::{BlockRoles, Role};

/// Encode one block's whole forward pass into `pass`, reading its twelve
/// weights out of `slot`.
///
/// `slot` is a streamer slot buffer wrapping the block blob's page-aligned
/// host allocation, so a role's blob-relative offset is its buffer offset.
pub(crate) fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    slot: &gpu::MetalBuffer,
    roles: &BlockRoles,
    s: &VisionScratch,
    shape: &VisionShape,
) -> Result<(), RealForwardError> {
    let seq = s.seq as u32;
    let hidden = shape.hidden as u32;
    let inter = shape.intermediate as u32;
    let wide = (s.seq * shape.hidden) as u32;
    let gpu_err = RealForwardError::Gpu;

    // --- attention half ---

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        (slot, roles.at(Role::Ln1W)),
        (slot, roles.at(Role::Ln1B)),
        (&s.normed, 0),
        seq,
        hidden,
        VISION_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;

    // q, k, v at rows 0, hidden, 2 * hidden of the fused weight, and the
    // matching thirds of the fused bias.
    let row_bytes = (shape.hidden * shape.hidden * 2) as u64;
    let bias_bytes = (shape.hidden * 2) as u64;
    for (i, out) in [&s.q, &s.k, &s.v].into_iter().enumerate() {
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, 0),
            (slot, roles.at(Role::QkvW) + i as u64 * row_bytes),
            Some((slot, roles.at(Role::QkvB) + i as u64 * bias_bytes)),
            (out, 0),
            seq,
            hidden,
            hidden,
        )
        .map_err(gpu_err)?;
    }

    // Q AND K ONLY. `v` carries no positional rotation in any attention this
    // port implements, and rotating it here would keep every shape.
    for target in [&s.q, &s.k] {
        gpu::encode_vision_rope_2d(
            context,
            pass,
            (target, 0),
            (&s.freqs, 0),
            seq,
            shape.heads as u32,
            shape.head_dim as u32,
        )
        .map_err(gpu_err)?;
    }

    gpu::encode_vision_attention(
        context,
        pass,
        (&s.q, 0),
        (&s.k, 0),
        (&s.v, 0),
        (&s.attn, 0),
        seq,
        shape.heads as u32,
        shape.head_dim as u32,
        shape.attention_scale(),
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.attn, 0),
        (slot, roles.at(Role::ProjW)),
        Some((slot, roles.at(Role::ProjB))),
        (&s.proj, 0),
        seq,
        hidden,
        hidden,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide)
        .map_err(gpu_err)?;

    // --- MLP half ---

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        (slot, roles.at(Role::Ln2W)),
        (slot, roles.at(Role::Ln2B)),
        (&s.normed, 0),
        seq,
        hidden,
        VISION_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.normed, 0),
        (slot, roles.at(Role::Fc1W)),
        Some((slot, roles.at(Role::Fc1B))),
        (&s.h1, 0),
        seq,
        hidden,
        inter,
    )
    .map_err(gpu_err)?;

    // TANH here, ERF in the merger. `nn.GELU(approx="tanh")` is what the
    // reference's `MLP` constructs; its `PatchMerger` constructs a bare
    // `nn.GELU()`.
    gpu::encode_vision_gelu(
        context,
        pass,
        (&s.h1, 0),
        (s.seq * shape.intermediate) as u32,
        gpu::GeluKind::Tanh,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.h1, 0),
        (slot, roles.at(Role::Fc2W)),
        Some((slot, roles.at(Role::Fc2B))),
        (&s.proj, 0),
        seq,
        inter,
        hidden,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide)
        .map_err(gpu_err)?;

    Ok(())
}
