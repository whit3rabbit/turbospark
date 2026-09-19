//! The three stages either side of the block loop: the patch embedding, the
//! interpolated position rows, and the merger (ROADMAP M-V4).

use crate::real_forward_types::RealForwardError;
use crate::vision::scratch::VisionScratch;
use crate::vision::shape::{VisionShape, VISION_LAYER_NORM_EPS};
use crate::vision::weights::VisionResident;

/// `x = patch_rows @ patch_embed.proj.weight^T + bias`.
///
/// The checkpoint ships that weight as a rank-5 `Conv3d` kernel, `[out, T,
/// P_h, P_w, C]`, and the walk records it flattened to `(hidden, patch_dim)`
/// -- which is the shape it is USED at, the conv having kernel == stride. So
/// there is no convolution here and no permutation anywhere: M-V1 emits patch
/// rows with `(T, P_h, P_w, C)` inside each row precisely so both operands
/// read row-major (`crates/vision-io` Gotcha 1).
pub(crate) fn encode_patch_embed(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    resident: &VisionResident,
    s: &VisionScratch,
    shape: &VisionShape,
) -> Result<(), RealForwardError> {
    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.rows, 0),
        (weights.buffer(), weights.gpu_offset(resident.patch_w)),
        Some((weights.buffer(), weights.gpu_offset(resident.patch_b))),
        (&s.x, 0),
        s.seq as u32,
        shape.patch_dim as u32,
        shape.hidden as u32,
    )
    .map_err(RealForwardError::Gpu)?;

    gpu::encode_vision_residual_add(
        context,
        pass,
        (&s.x, 0),
        (&s.pos, 0),
        (s.seq * shape.hidden) as u32,
    )
    .map_err(RealForwardError::Gpu)
}

/// The interpolated position rows, `[seq, hidden]` FP16, computed on the
/// HOST.
///
/// # Why the host, and why this is not a missing kernel
///
/// `pos_embed_weights` returns four INDICES and four WEIGHTS per patch and
/// deliberately gathers nothing (`crates/vision-io` Gotcha 11), because the
/// values being gathered live in a buffer that crate cannot see. This port
/// has no gather kernel and M-V2 shipped none, so the blend happens here: the
/// `[2304, 1152]` table is read off the resident mmap once at open (5.3 MB),
/// and this runs once per IMAGE rather than once per token.
///
/// The four weights sum to one per patch, so the blend cannot amplify: the
/// result is inside the table's own range and cannot push the FP16 stream
/// anywhere the table does not already reach.
pub(crate) fn pos_embed_rows(
    table: &[f32],
    entries: &turbospark_vision_io::PosEmbedTable,
    shape: &VisionShape,
) -> Result<Vec<half::f16>, RealForwardError> {
    let hidden = shape.hidden;
    let patches = entries.len();
    let mut out = vec![half::f16::ZERO; patches * hidden];
    let mut acc = vec![0.0f32; hidden];
    for patch in 0..patches {
        acc.fill(0.0);
        let dst = &mut out[patch * hidden..(patch + 1) * hidden];
        for corner in 0..4 {
            let row = entries.indices[corner][patch];
            let w = entries.weights[corner][patch];
            let src = table.get(row * hidden..(row + 1) * hidden).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "position embedding row {row} is past the {}-row table",
                    shape.pos_rows
                ))
            })?;
            for (a, v) in acc.iter_mut().zip(src) {
                *a += w * v;
            }
        }
        for (d, a) in dst.iter_mut().zip(&acc) {
            *d = half::f16::from_f32(*a);
        }
    }
    Ok(out)
}

/// `out = fc2(gelu_erf(fc1(norm(x) viewed as merged rows)))`.
///
/// # The norm is over `hidden`, and the reshape is a VIEW
///
/// The reference builds its `PatchMerger` with `use_postshuffle_norm=False`,
/// so `self.norm` is sized `config.hidden_size` (1152) and runs PER PATCH ROW
/// before the reshape to `[-1, hidden * merge^2]`. Norming the 4608-wide row
/// instead is a different function that runs and produces a plausible
/// embedding.
///
/// The reshape itself costs nothing and moves nothing: M-V1 emits patch rows
/// in merge-window order, so the `merge^2` patches of one output token are
/// already contiguous and `[seq, hidden]` reinterpreted as `[merged, hidden *
/// merge^2]` is the same bytes. That ordering decision is what buys this.
///
/// # The erf GELU
///
/// `nn.GELU()` with no `approx`, against the blocks' `approx="tanh"`. The two
/// agree to ~3e-4 and no whole-tower parity bound can separate them, so the
/// choice lives at this call site and in the per-kernel cases.
pub(crate) fn encode_merger(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    resident: &VisionResident,
    s: &VisionScratch,
    shape: &VisionShape,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let merger_in = shape.merger_input();

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        (weights.buffer(), weights.gpu_offset(resident.merger_norm_w)),
        (weights.buffer(), weights.gpu_offset(resident.merger_norm_b)),
        (&s.normed, 0),
        s.seq as u32,
        shape.hidden as u32,
        VISION_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;

    // `normed` is `[seq, hidden]`; read here as `[merged, merger_in]`. Same
    // bytes, no copy.
    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.normed, 0),
        (weights.buffer(), weights.gpu_offset(resident.merger_fc1_w)),
        Some((weights.buffer(), weights.gpu_offset(resident.merger_fc1_b))),
        (&s.m1, 0),
        s.merged as u32,
        merger_in as u32,
        merger_in as u32,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_gelu(
        context,
        pass,
        (&s.m1, 0),
        (s.merged * merger_in) as u32,
        gpu::GeluKind::Erf,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.m1, 0),
        (weights.buffer(), weights.gpu_offset(resident.merger_fc2_w)),
        Some((weights.buffer(), weights.gpu_offset(resident.merger_fc2_b))),
        (&s.out, 0),
        s.merged as u32,
        merger_in as u32,
        shape.out_hidden as u32,
    )
    .map_err(gpu_err)
}

/// `out = fc2(gelu_erf(fc1(norm(x viewed as merged rows))))` for ONE
/// deepstack merger.
///
/// # The norm is the POST-SHUFFLE one, the opposite of [`encode_merger`]'s
///
/// The reference builds the deepstack mergers with `use_postshuffle_norm=True`
/// (`mlx_vlm/models/qwen3_vl/vision.py`): the reshape to
/// `[-1, hidden * merge^2]` happens FIRST and the LayerNorm runs on the
/// concatenated row, where the main merger norms per patch row and reshapes
/// after. Same bytes in, a DIFFERENT function out -- norming the wrong width
/// is fluent-wrong in exactly the way `docs/VISION.md`'s "Six things" section
/// catalogs. `x` here is `s.x` read as `[merged, merger_in]`, which costs
/// nothing: the patch rows are emitted in merge-window order, so the view is
/// the same bytes reinterpreted.
///
/// The erf GELU matches the main merger (`nn.GELU()`, no `approx`), and the
/// result lands in `s.out`, the main merger's own output buffer -- the
/// deepstack mergers run MID-LOOP, each is read back to the host before
/// `s.out` is next written, and the main merger's final output is the last
/// thing written, so the reuse never overlaps a live value.
pub(crate) fn encode_deepstack_merger(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    merger: &super::weights::DeepstackMergerWeights,
    s: &VisionScratch,
    shape: &VisionShape,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let merger_in = shape.merger_input();

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        (weights.buffer(), weights.gpu_offset(merger.norm_w)),
        (weights.buffer(), weights.gpu_offset(merger.norm_b)),
        (&s.normed, 0),
        s.merged as u32,
        merger_in as u32,
        VISION_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.normed, 0),
        (weights.buffer(), weights.gpu_offset(merger.fc1_w)),
        Some((weights.buffer(), weights.gpu_offset(merger.fc1_b))),
        (&s.m1, 0),
        s.merged as u32,
        merger_in as u32,
        merger_in as u32,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_gelu(
        context,
        pass,
        (&s.m1, 0),
        (s.merged * merger_in) as u32,
        gpu::GeluKind::Erf,
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.m1, 0),
        (weights.buffer(), weights.gpu_offset(merger.fc2_w)),
        Some((weights.buffer(), weights.gpu_offset(merger.fc2_b))),
        (&s.out, 0),
        s.merged as u32,
        merger_in as u32,
        shape.out_hidden as u32,
    )
    .map_err(gpu_err)
}
