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
/// `slot` is either a streamer slot buffer wrapping exactly this one block's
/// page-aligned host allocation (the pread arm, `base == 0`), or the mapped
/// residency arm's single buffer over the WHOLE tower's mapped region, in
/// which case `base` is this block's own offset within it
/// (`MappedExpertLayer::expert_offset`). Either way a role's buffer offset is
/// `base + roles.at(role)`: `roles.at()` is always relative to the start of
/// this one block's own blob, and `base` places that blob within whichever
/// buffer `slot` is.
///
/// `tile_rows` is the MLP's row tile (Part B1, `scratch::VISION_MLP_TILE_ROWS`
/// by default): `fc1 -> gelu -> fc2` is row-independent, so the MLP half runs
/// in `ceil(seq / tile_rows)` iterations over `s.h1`, which is sized for one
/// tile rather than the whole page. Every dispatch in every iteration still
/// goes through this same `pass` -- no new command buffer, no commit mid-loop
/// -- so tile `t+1`'s `fc1` write to `s.h1` is issued on the same encoder
/// strictly after tile `t`'s `fc2` read of it, which is what makes reusing
/// one small buffer across tiles safe (`crates/gpu/CLAUDE.md` Gotcha 8's
/// commit-order guarantee, applied one grain finer than the block loop
/// already relies on it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    slot: &gpu::MetalBuffer,
    base: u64,
    roles: &BlockRoles,
    s: &VisionScratch,
    shape: &VisionShape,
    tile_rows: usize,
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
        (slot, base + roles.at(Role::Ln1W)),
        (slot, base + roles.at(Role::Ln1B)),
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
            (slot, base + roles.at(Role::QkvW) + i as u64 * row_bytes),
            Some((slot, base + roles.at(Role::QkvB) + i as u64 * bias_bytes)),
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
        (slot, base + roles.at(Role::ProjW)),
        Some((slot, base + roles.at(Role::ProjB))),
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
        (slot, base + roles.at(Role::Ln2W)),
        (slot, base + roles.at(Role::Ln2B)),
        (&s.normed, 0),
        seq,
        hidden,
        VISION_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;

    // Row-tiled (Part B1): `fc1 -> gelu -> fc2` is row-independent, so
    // looping over fixed-size row tiles of `s.normed`/`s.proj` through the
    // ONE tile-sized `s.h1` buffer is identical arithmetic to running every
    // row through a page-sized `h1` at once. An ordinary page (5,120 patches
    // at the default 2,048-row tile) still takes 3 iterations here, which is
    // intentional -- see the module doc.
    let tile = tile_rows.max(1);
    let mut t0 = 0usize;
    while t0 < s.seq {
        let rows = tile.min(s.seq - t0);
        let row_off = (t0 * shape.hidden * 2) as u64;

        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, row_off),
            (slot, base + roles.at(Role::Fc1W)),
            Some((slot, base + roles.at(Role::Fc1B))),
            (&s.h1, 0),
            rows as u32,
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
            (rows * shape.intermediate) as u32,
            gpu::GeluKind::Tanh,
        )
        .map_err(gpu_err)?;

        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.h1, 0),
            (slot, base + roles.at(Role::Fc2W)),
            Some((slot, base + roles.at(Role::Fc2B))),
            (&s.proj, row_off),
            rows as u32,
            inter,
            hidden,
        )
        .map_err(gpu_err)?;

        t0 += rows;
    }

    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide)
        .map_err(gpu_err)?;

    Ok(())
}
