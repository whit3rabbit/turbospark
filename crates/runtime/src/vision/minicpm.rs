//! The MiniCPM-V 4.7 vision tower on Metal (`docs/MINICPMV_VISION_PHASE0.md`).
//!
//! A SigLIP encoder (27 pre-LayerNorm blocks, tanh GELU, bidirectional
//! attention over the whole image, no rope) with two merges folded in:
//!
//! ```text
//! x = patch_embed(rows) + position_embedding[bucket ids]
//! blocks 0..=6
//! x = vit_merger(x)         # 2x2 windows: attn over 4 tokens + MLP + window mean
//! blocks 7..=26             # on a quarter of the tokens
//! x = post_layernorm(x)
//! out = merger(x)           # 2x2 groups of 1152 -> LN(4608) -> 4608 -> gelu -> 2048
//! ```
//!
//! # The token order is a HOST permutation
//!
//! Blocks 0 to 6 and 7 to 26 carry no position-dependent op (the position is
//! a per-token add in the embedding), so permuting tokens before the first
//! block is exact. The tokens are ordered by 4x4 block (raster over blocks),
//! then 2x2 window inside the block (raster), then raster inside the window,
//! so that:
//!
//! - every `vit_merger` window is FOUR CONSECUTIVE ROWS (the reference's
//!   `reshape(merged_h, 2, merged_w, 2, d).transpose(0, 2, 1, 3, 4)`), and
//! - after it, the merged tokens of one 4x4 block are four consecutive rows in
//!   `(a, b)` raster order, which is exactly the final merger's `2x2` group
//!   with its `(gh, gw, dim)` concatenation, and the output rows come out in
//!   the page's final raster order with no un-permutation.
//!
//! # GELU
//!
//! Every GELU here is the tanh form. MLX spells it `"precise"` in the mergers
//! and `"tanh"` in the encoder blocks and the 4.7 `vit_merger`; they name the
//! same approximation, which the exact-erf form differs from by about 3e-4,
//! below what any composition test here resolves.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

use super::weights::{fp16_view, VISION_PREFIX};
use super::{VisionEmbedding, VisionStages, VISION_SLOTS};
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// LayerNorm epsilon of every norm in the tower, the mergers included.
const EPS: f32 = 1e-6;
/// The block after which the window merger runs.
const INSERT_LAYER: usize = 6;
/// MLP row tile for the encoder blocks, the qwen tower's Part B1 shape.
pub(crate) const MINICPM_MLP_TILE_ROWS: usize = 1024;

/// The blob roles in byte order: the repack writer's `MINICPM_BLOCK_ROLES`
/// first column, restated because repack is a dev-dependency.
pub(crate) const MINICPM_BLOCK_ROLES: [&str; 16] = [
    "ln1_w", "ln1_b", "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "o_w", "o_b", "ln2_w", "ln2_b",
    "fc1_w", "fc1_b", "fc2_w", "fc2_b",
];

#[derive(Debug, Clone, Copy)]
enum R {
    Ln1W = 0,
    Ln1B = 1,
    QW = 2,
    QB = 3,
    KW = 4,
    KB = 5,
    VW = 6,
    VB = 7,
    OW = 8,
    OB = 9,
    Ln2W = 10,
    Ln2B = 11,
    Fc1W = 12,
    Fc1B = 13,
    Fc2W = 14,
    Fc2B = 15,
}

const ALL_ROLES: [R; 16] = [
    R::Ln1W,
    R::Ln1B,
    R::QW,
    R::QB,
    R::KW,
    R::KB,
    R::VW,
    R::VB,
    R::OW,
    R::OB,
    R::Ln2W,
    R::Ln2B,
    R::Fc1W,
    R::Fc1B,
    R::Fc2W,
    R::Fc2B,
];

/// The tower's geometry, resolved once from the install's [`VisionConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MiniCpmDims {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_size: usize,
    pub(crate) patch_dim: usize,
    pub(crate) pos_rows: usize,
    pub(crate) pos_side: usize,
    pub(crate) out_hidden: usize,
}

impl MiniCpmDims {
    pub(crate) fn resolve(v: &VisionConfig) -> Result<Self, RealForwardError> {
        if v.tower != VisionTowerKind::MiniCpmV || !v.is_active() {
            return Err(RealForwardError::Unsupported(format!(
                "not an active minicpm_v vision tower (kind {})",
                v.tower.as_str()
            )));
        }
        let us = |x: i64, what: &str| {
            usize::try_from(x)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| RealForwardError::Unsupported(format!("minicpm vision {what} {x}")))
        };
        let d = Self {
            depth: us(v.depth, "depth")?,
            hidden: us(v.hidden_size, "hidden_size")?,
            intermediate: us(v.intermediate_size, "intermediate_size")?,
            heads: us(v.num_heads, "num_heads")?,
            head_dim: us(v.head_dim(), "head_dim")?,
            patch_size: us(v.patch_size, "patch_size")?,
            patch_dim: us(
                v.temporal_patch_size * v.patch_size * v.patch_size * v.in_channels,
                "patch_dim",
            )?,
            pos_rows: us(v.num_position_embeddings, "position rows")?,
            pos_side: {
                let rows = us(v.num_position_embeddings, "position rows")?;
                let side = (rows as f64).sqrt().round() as usize;
                if side * side != rows {
                    return Err(RealForwardError::Unsupported(format!(
                        "minicpm vision position table of {rows} rows is not square"
                    )));
                }
                side
            },
            out_hidden: us(v.out_hidden_size, "out_hidden_size")?,
        };
        if d.depth <= INSERT_LAYER
            || d.heads * d.head_dim != d.hidden
            || d.head_dim % 4 != 0
            || v.spatial_merge_size != 4
            || v.temporal_patch_size != 1
        {
            return Err(RealForwardError::Unsupported(format!(
                "minicpm vision geometry depth {} hidden {} heads {} merge {} is not the \
                 27-block, 16x tower this port runs",
                d.depth, d.hidden, d.heads, v.spatial_merge_size
            )));
        }
        if d.head_dim as u32 > gpu::MAX_ATTENTION_HEAD_DIM {
            return Err(RealForwardError::Unsupported(format!(
                "minicpm vision head_dim {} is past the attention kernel's {}",
                d.head_dim,
                gpu::MAX_ATTENTION_HEAD_DIM
            )));
        }
        Ok(d)
    }

    /// The window merger's concatenated width: four tokens.
    fn window(&self) -> usize {
        4 * self.hidden
    }

    fn role_elems(&self, role: R) -> usize {
        let (h, i) = (self.hidden, self.intermediate);
        match role {
            R::Ln1W | R::Ln1B | R::Ln2W | R::Ln2B => h,
            R::QB | R::KB | R::VB | R::OB => h,
            R::QW | R::KW | R::VW | R::OW => h * h,
            R::Fc1W | R::Fc2W => h * i,
            R::Fc1B => i,
            R::Fc2B => h,
        }
    }
}

/// One block's sixteen byte offsets within its blob.
#[derive(Debug, Clone)]
pub(crate) struct MiniCpmBlockRoles {
    offsets: [u64; 16],
}

impl MiniCpmBlockRoles {
    fn at(&self, role: R) -> u64 {
        self.offsets[role as usize]
    }

    pub(crate) fn resolve(
        entry: &model_io::ExpertEntry,
        stride: u64,
        d: &MiniCpmDims,
    ) -> Result<Self, RealForwardError> {
        let block = entry.expert;
        let mut offsets = [0u64; 16];
        for role in ALL_ROLES {
            let name = MINICPM_BLOCK_ROLES[role as usize];
            let sub = entry.sub_tensors.get(name).ok_or_else(|| {
                RealForwardError::MissingTensor(format!("minicpm vision block {block} {name}"))
            })?;
            let want = (d.role_elems(role) * 2) as u64;
            if sub.dtype != "fp16" || sub.size != want {
                return Err(RealForwardError::Unsupported(format!(
                    "minicpm vision block {block} {name}: {} bytes of {}, expected {want} of fp16",
                    sub.size, sub.dtype
                )));
            }
            if sub.offset % 4 != 0 || sub.offset + sub.size > stride {
                return Err(RealForwardError::Unsupported(format!(
                    "minicpm vision block {block} {name} at {}..{} does not fit a 4-byte aligned \
                     slot of a {stride}-byte blob",
                    sub.offset,
                    sub.offset + sub.size
                )));
            }
            offsets[role as usize] = sub.offset;
        }
        Ok(Self { offsets })
    }
}

pub(crate) struct BlockBinding<'a> {
    pub(crate) buffer: &'a gpu::MetalBuffer,
    pub(crate) roles: &'a MiniCpmBlockRoles,
}

impl BlockBinding<'_> {
    fn w(&self, role: R) -> (&gpu::MetalBuffer, u64) {
        (self.buffer, self.roles.at(role))
    }
}

/// The resident tensors outside the blocks, as offsets into the weights buffer.
struct MiniCpmResident {
    patch_w: u64,
    patch_b: u64,
    pos_table: u64,
    post_ln_w: u64,
    post_ln_b: u64,
    vit: VitMergerWeights,
    merger: MergerWeights,
}

struct VitMergerWeights {
    ln1_w: u64,
    ln1_b: u64,
    q_w: u64,
    q_b: u64,
    k_w: u64,
    k_b: u64,
    v_w: u64,
    v_b: u64,
    o_w: u64,
    o_b: u64,
    pre_w: u64,
    pre_b: u64,
    lin1_w: u64,
    lin1_b: u64,
    lin2_w: u64,
    lin2_b: u64,
}

struct MergerWeights {
    pre_w: u64,
    pre_b: u64,
    lin1_w: u64,
    lin1_b: u64,
    lin2_w: u64,
    lin2_b: u64,
}

impl MiniCpmResident {
    fn resolve(index: &ResidentIndex, d: &MiniCpmDims) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let (h, w4) = (d.hidden, d.window());
        let w4i = 4 * d.intermediate;
        let v = |tail: &str, elems: usize| fp16_view(index, &name(tail), elems);
        Ok(Self {
            patch_w: v("embeddings.patch_embedding.weight", h * d.patch_dim)?,
            patch_b: v("embeddings.patch_embedding.bias", h)?,
            pos_table: v("embeddings.position_embedding.weight", d.pos_rows * h)?,
            post_ln_w: v("post_layernorm.weight", h)?,
            post_ln_b: v("post_layernorm.bias", h)?,
            vit: VitMergerWeights {
                ln1_w: v("vit_merger.layer_norm1.weight", h)?,
                ln1_b: v("vit_merger.layer_norm1.bias", h)?,
                q_w: v("vit_merger.self_attn.q_proj.weight", h * h)?,
                q_b: v("vit_merger.self_attn.q_proj.bias", h)?,
                k_w: v("vit_merger.self_attn.k_proj.weight", h * h)?,
                k_b: v("vit_merger.self_attn.k_proj.bias", h)?,
                v_w: v("vit_merger.self_attn.v_proj.weight", h * h)?,
                v_b: v("vit_merger.self_attn.v_proj.bias", h)?,
                o_w: v("vit_merger.self_attn.out_proj.weight", h * h)?,
                o_b: v("vit_merger.self_attn.out_proj.bias", h)?,
                pre_w: v("vit_merger.pre_norm.weight", w4)?,
                pre_b: v("vit_merger.pre_norm.bias", w4)?,
                lin1_w: v("vit_merger.linear_1.weight", w4 * w4i)?,
                lin1_b: v("vit_merger.linear_1.bias", w4i)?,
                lin2_w: v("vit_merger.linear_2.weight", h * w4i)?,
                lin2_b: v("vit_merger.linear_2.bias", h)?,
            },
            merger: MergerWeights {
                pre_w: v("merger.mlp.0.pre_norm.weight", w4)?,
                pre_b: v("merger.mlp.0.pre_norm.bias", w4)?,
                lin1_w: v("merger.mlp.0.linear_1.weight", w4 * w4)?,
                lin1_b: v("merger.mlp.0.linear_1.bias", w4)?,
                lin2_w: v("merger.mlp.0.linear_2.weight", d.out_hidden * w4)?,
                lin2_b: v("merger.mlp.0.linear_2.bias", d.out_hidden)?,
            },
        })
    }
}

/// Decode `elems` FP16 values at a resident offset to host f32.
fn fp16_host(
    weights: &gpu::ResidentGpuWeights,
    local: u64,
    elems: usize,
) -> Result<Vec<f32>, RealForwardError> {
    let at = local as usize;
    let bytes = weights.data().get(at..at + elems * 2).ok_or_else(|| {
        RealForwardError::Unsupported(format!(
            "minicpm vision tensor at resident offset {at} spans past the mapped region"
        ))
    })?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect())
}

/// For each token in tower order, its index in the page's raster order: 4x4
/// blocks in raster order, 2x2 windows in raster order inside a block, raster
/// inside a window. `rows` and `cols` must be multiples of 4.
pub(crate) fn tower_order(rows: usize, cols: usize) -> Result<Vec<usize>, RealForwardError> {
    if rows == 0 || cols == 0 || rows % 4 != 0 || cols % 4 != 0 {
        return Err(RealForwardError::Unsupported(format!(
            "minicpm vision grid {rows}x{cols} is not a whole number of 4x4 patch blocks"
        )));
    }
    let mut order = Vec::with_capacity(rows * cols);
    for br in 0..rows / 4 {
        for bc in 0..cols / 4 {
            for wa in 0..2 {
                for wb in 0..2 {
                    for gh in 0..2 {
                        for gw in 0..2 {
                            let r = br * 4 + wa * 2 + gh;
                            let c = bc * 4 + wb * 2 + gw;
                            order.push(r * cols + c);
                        }
                    }
                }
            }
        }
    }
    Ok(order)
}

/// The reference's `_build_position_buckets` for one axis of `n` patches
/// against a table of `side` per axis: `frac = clip(arange(n) / n, 0,
/// 1 - 1e-6)` in f32 and the bucket is how many boundaries `<= frac`. The
/// boundaries are `np.arange(1 / side, 1.0, 1 / side)` built in f64 (numpy's
/// fill is `first + i * ((first + step) - first)`) and cast to f32, as the
/// reference does, because an exact tie (`2 / 28` against `5 / 70`) lands in
/// the adjacent bucket if the boundary is built in f32.
pub(crate) fn position_buckets(n: usize, side: usize) -> Vec<usize> {
    let n = n.max(1);
    let step = 1.0f64 / side as f64;
    let first = step;
    let delta = (first + step) - first;
    let len = (((1.0f64 - first) / step).ceil().max(0.0)) as usize;
    let boundaries: Vec<f32> = (0..len)
        .map(|i| (first + i as f64 * delta) as f32)
        .collect();
    let max = (1.0f64 - 1e-6) as f32;
    (0..n)
        .map(|i| {
            let frac = ((i as f32) / (n as f32)).clamp(0.0, max);
            boundaries.iter().filter(|&&b| frac >= b).count()
        })
        .collect()
}

/// Per-image GPU scratch. The five `seq x hidden` buffers are reused for the
/// quarter-size token set after the window merger.
pub(crate) struct MiniCpmScratch {
    seq: usize,
    tile: usize,
    rows: gpu::MetalBuffer,
    pos: gpu::MetalBuffer,
    x: gpu::MetalBuffer,
    normed: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    k: gpu::MetalBuffer,
    v: gpu::MetalBuffer,
    attn: gpu::MetalBuffer,
    proj: gpu::MetalBuffer,
    h1: gpu::MetalBuffer,
    /// The window MLP's `[merged, 4 * 4304]` hidden layer.
    window_hidden: gpu::MetalBuffer,
    out: gpu::MetalBuffer,
}

impl MiniCpmScratch {
    fn allocate(
        context: &gpu::MetalContext,
        d: &MiniCpmDims,
        seq: usize,
        tile: usize,
        rows: &[half::f16],
        pos: &[half::f16],
    ) -> Self {
        let wide = (seq * d.hidden * 2) as u64;
        let tile = tile.min(seq).max(1);
        let windows = seq / 4;
        let tokens = seq / 16;
        Self {
            seq,
            tile,
            rows: context.new_buffer_with_data(rows),
            pos: context.new_buffer_with_data(pos),
            x: context.new_output_buffer(wide),
            normed: context.new_output_buffer(wide),
            q: context.new_output_buffer(wide),
            k: context.new_output_buffer(wide),
            v: context.new_output_buffer(wide),
            attn: context.new_output_buffer(wide),
            proj: context.new_output_buffer(wide),
            h1: context.new_output_buffer((tile * d.intermediate * 2) as u64),
            window_hidden: context.new_output_buffer((windows * 4 * d.intermediate * 2) as u64),
            out: context.new_output_buffer((tokens * d.out_hidden * 2) as u64),
        }
    }
}

/// Encode one encoder block over the first `rows` rows of `s.x`, in place.
fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    b: &BlockBinding<'_>,
    s: &MiniCpmScratch,
    d: &MiniCpmDims,
    rows: usize,
) -> Result<(), RealForwardError> {
    let e = RealForwardError::Gpu;
    let seq = rows as u32;
    let hidden = d.hidden as u32;
    let wide = (rows * d.hidden) as u32;
    let scale = (d.head_dim as f32).powf(-0.5);

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        b.w(R::Ln1W),
        b.w(R::Ln1B),
        (&s.normed, 0),
        seq,
        hidden,
        EPS,
    )
    .map_err(e)?;
    for (out, w, bias) in [
        (&s.q, R::QW, R::QB),
        (&s.k, R::KW, R::KB),
        (&s.v, R::VW, R::VB),
    ] {
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, 0),
            b.w(w),
            Some(b.w(bias)),
            (out, 0),
            seq,
            hidden,
            hidden,
        )
        .map_err(e)?;
    }
    gpu::encode_vision_attention(
        context,
        pass,
        (&s.q, 0),
        (&s.k, 0),
        (&s.v, 0),
        (&s.attn, 0),
        seq,
        d.heads as u32,
        d.head_dim as u32,
        scale,
    )
    .map_err(e)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.attn, 0),
        b.w(R::OW),
        Some(b.w(R::OB)),
        (&s.proj, 0),
        seq,
        hidden,
        hidden,
    )
    .map_err(e)?;
    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide).map_err(e)?;

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        b.w(R::Ln2W),
        b.w(R::Ln2B),
        (&s.normed, 0),
        seq,
        hidden,
        EPS,
    )
    .map_err(e)?;
    let inter = d.intermediate as u32;
    let mut t0 = 0usize;
    while t0 < rows {
        let n = s.tile.min(rows - t0);
        let row_off = (t0 * d.hidden * 2) as u64;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, row_off),
            b.w(R::Fc1W),
            Some(b.w(R::Fc1B)),
            (&s.h1, 0),
            n as u32,
            hidden,
            inter,
        )
        .map_err(e)?;
        gpu::encode_vision_gelu(
            context,
            pass,
            (&s.h1, 0),
            (n * d.intermediate) as u32,
            gpu::GeluKind::Tanh,
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.h1, 0),
            b.w(R::Fc2W),
            Some(b.w(R::Fc2B)),
            (&s.proj, row_off),
            n as u32,
            inter,
            hidden,
        )
        .map_err(e)?;
        t0 += n;
    }
    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide).map_err(e)
}

/// The 2x2 window attention-merger over the `rows` tokens of `s.x`, whose
/// consecutive groups of four are the windows. Leaves the `rows / 4` merged
/// tokens in `s.x` (swapped in from `s.attn`'s buffer, so no copy kernel).
#[allow(clippy::too_many_arguments)]
fn encode_vit_merger(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    w: &gpu::ResidentGpuWeights,
    r: &VitMergerWeights,
    s: &mut MiniCpmScratch,
    d: &MiniCpmDims,
    rows: usize,
) -> Result<(), RealForwardError> {
    let e = RealForwardError::Gpu;
    let hidden = d.hidden as u32;
    let wide = (rows * d.hidden) as u32;
    let windows = rows / 4;
    let at = |local: u64| (w.buffer(), w.gpu_offset(local));
    let scale = (d.head_dim as f32).powf(-0.5);

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        at(r.ln1_w),
        at(r.ln1_b),
        (&s.normed, 0),
        rows as u32,
        hidden,
        EPS,
    )
    .map_err(e)?;
    for (out, wt, bias) in [
        (&s.q, r.q_w, r.q_b),
        (&s.k, r.k_w, r.k_b),
        (&s.v, r.v_w, r.v_b),
    ] {
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, 0),
            at(wt),
            Some(at(bias)),
            (out, 0),
            rows as u32,
            hidden,
            hidden,
        )
        .map_err(e)?;
    }
    // Attention INSIDE each window of four tokens: the same kernel at each
    // window's row offset, bidirectional, no mask.
    for window in 0..windows {
        let off = (window * 4 * d.hidden * 2) as u64;
        gpu::encode_vision_attention(
            context,
            pass,
            (&s.q, off),
            (&s.k, off),
            (&s.v, off),
            (&s.attn, off),
            4,
            d.heads as u32,
            d.head_dim as u32,
            scale,
        )
        .map_err(e)?;
    }
    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.attn, 0),
        at(r.o_w),
        Some(at(r.o_b)),
        (&s.proj, 0),
        rows as u32,
        hidden,
        hidden,
    )
    .map_err(e)?;
    // windows = windows + attn(norm(windows))
    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide).map_err(e)?;

    // merged = linear_2(gelu(linear_1(pre_norm(windows.reshape(M, 4 * 1152)))))
    let w4 = d.window() as u32;
    let w4i = (4 * d.intermediate) as u32;
    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        at(r.pre_w),
        at(r.pre_b),
        (&s.normed, 0),
        windows as u32,
        w4,
        EPS,
    )
    .map_err(e)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.normed, 0),
        at(r.lin1_w),
        Some(at(r.lin1_b)),
        (&s.window_hidden, 0),
        windows as u32,
        w4,
        w4i,
    )
    .map_err(e)?;
    gpu::encode_vision_gelu(
        context,
        pass,
        (&s.window_hidden, 0),
        (windows * 4 * d.intermediate) as u32,
        gpu::GeluKind::Tanh,
    )
    .map_err(e)?;
    // The merged rows go to `s.attn` (free now), then the mean of the windows
    // `x` still holds is added on top.
    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.window_hidden, 0),
        at(r.lin2_w),
        Some(at(r.lin2_b)),
        (&s.attn, 0),
        windows as u32,
        w4i,
        hidden,
    )
    .map_err(e)?;
    gpu::encode_vision_window_mean4_add(
        context,
        pass,
        (&s.attn, 0),
        (&s.x, 0),
        windows as u32,
        hidden,
    )
    .map_err(e)?;
    std::mem::swap(&mut s.x, &mut s.attn);
    Ok(())
}

/// What a diagnostic run reads back, for the parity test.
#[derive(Debug, Default)]
pub(crate) struct MiniCpmStages {
    pub(crate) patch_embed: Vec<u16>,
    pub(crate) block_0: Vec<u16>,
    pub(crate) vit_merger: Vec<u16>,
    pub(crate) last_block: Vec<u16>,
}

fn read_bits(buffer: &gpu::MetalBuffer, count: usize) -> Vec<u16> {
    gpu::read_buffer_f16(buffer, 0, count)
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The MiniCPM-V tower opened against an install: encoder blocks streamed from
/// `packed_vision/` into [`VISION_SLOTS`] slots, residents resolved off the
/// install's index. Opened lazily on the first image. Combined installs only:
/// a sidecar cannot pair it safely (`catalog::stream::stream_vision_sidecar`).
pub(crate) struct MiniCpmVisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<MiniCpmBlockRoles>,
    dims: MiniCpmDims,
    resident: MiniCpmResident,
    /// The position table decoded to host f32 at open (22 MB).
    pos_table: Vec<f32>,
    pub(crate) last_scratch_bytes: u64,
}

impl MiniCpmVisionTower {
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let dims = MiniCpmDims::resolve(&arch.vision)?;
        let layout = model_io::load_packed_layout_from(
            dir,
            model_io::PACKED_VISION_DIR,
            PACKED_LAYOUT_MAX_BYTES,
        )
        .map_err(RealForwardError::Model)?;
        let layer = layout.layers.iter().find(|l| l.layer == 0).ok_or_else(|| {
            RealForwardError::Unsupported("packed_vision/layout.json has no layer 0".to_string())
        })?;
        if layer.experts.len() != dims.depth {
            return Err(RealForwardError::Unsupported(format!(
                "packed_vision carries {} blocks but the manifest declares {}",
                layer.experts.len(),
                dims.depth
            )));
        }
        let mut blocks: Vec<Option<MiniCpmBlockRoles>> = vec![None; dims.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, dims.depth
                ))
            })?;
            if slot
                .replace(MiniCpmBlockRoles::resolve(
                    entry,
                    layer.expert_stride,
                    &dims,
                )?)
                .is_some()
            {
                return Err(RealForwardError::Unsupported(format!(
                    "packed_vision names block {} twice",
                    entry.expert
                )));
            }
        }
        let blocks = blocks
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                b.ok_or_else(|| {
                    RealForwardError::MissingTensor(format!("packed_vision has no block {i}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let stream_layout =
            streaming::StreamLayout::from_packed_layer_in(layer, dir, model_io::PACKED_VISION_DIR);
        let streamer = streaming::PreadExpertStreamer::open(
            stream_layout,
            VISION_SLOTS,
            streaming::ExpertCachePolicy::DEFAULT,
        )
        .map_err(|e| RealForwardError::Unsupported(format!("minicpm vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }
        let resident = MiniCpmResident::resolve(index, &dims)?;
        let pos_table = fp16_host(weights, resident.pos_table, dims.pos_rows * dims.hidden)?;
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            dims,
            resident,
            pos_table,
            last_scratch_bytes: 0,
        })
    }

    /// Encode one image to the `[merged, out_hidden]` rows the trunk takes.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &PreprocessedImage,
        params: &PreprocessParams,
        mut capture: Option<&mut MiniCpmStages>,
    ) -> Result<VisionEmbedding, RealForwardError> {
        let d = self.dims;
        if params.patch_dim() != d.patch_dim || params.patch_size != d.patch_size {
            return Err(RealForwardError::Unsupported(format!(
                "preprocessing produced patch {} / {}-value rows but this tower takes patch {} / {}",
                params.patch_size,
                params.patch_dim(),
                d.patch_size,
                d.patch_dim
            )));
        }
        let GridThw { t, h: gh, w: gw } = image.grid;
        let seq = image.grid.patches();
        if t != 1 || seq == 0 || image.patch_rows.len() != seq * d.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "minicpm vision input: {} values for a {t}x{gh}x{gw} grid of {}-value patches",
                image.patch_rows.len(),
                d.patch_dim
            )));
        }
        let order = tower_order(gh, gw)?;

        // Host prep in TOWER order: patch rows to FP16, and each token's
        // position row gathered by its bucket ids (computed on the raster
        // grid, where the reference computes them).
        let mut rows16 = Vec::with_capacity(seq * d.patch_dim);
        for &raster in &order {
            rows16.extend(
                image.patch_rows[raster * d.patch_dim..(raster + 1) * d.patch_dim]
                    .iter()
                    .map(|&v| half::f16::from_f32(v)),
            );
        }
        let (bh, bw) = (
            position_buckets(gh, d.pos_side),
            position_buckets(gw, d.pos_side),
        );
        let mut pos16 = Vec::with_capacity(seq * d.hidden);
        for &raster in &order {
            let id = bh[raster / gw] * d.pos_side + bw[raster % gw];
            pos16.extend(
                self.pos_table[id * d.hidden..(id + 1) * d.hidden]
                    .iter()
                    .map(|&v| half::f16::from_f32(v)),
            );
        }

        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let resident = &self.resident;
        let last_scratch = &mut self.last_scratch_bytes;
        let merged_out = seq / 16;

        gpu::autorelease_pool(|| {
            let e = RealForwardError::Gpu;
            let mut s =
                MiniCpmScratch::allocate(context, &d, seq, MINICPM_MLP_TILE_ROWS, &rows16, &pos16);
            *last_scratch = (seq * d.hidden * 2 * 7
                + s.tile * d.intermediate * 2
                + (seq / 4) * 4 * d.intermediate * 2
                + merged_out * d.out_hidden * 2) as u64;
            let wide = seq * d.hidden;
            let at = |local: u64| (weights.buffer(), weights.gpu_offset(local));

            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.rows, 0),
                at(resident.patch_w),
                Some(at(resident.patch_b)),
                (&s.x, 0),
                seq as u32,
                d.patch_dim as u32,
                d.hidden as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_residual_add(context, &pass, (&s.x, 0), (&s.pos, 0), wide as u32)
                .map_err(e)?;
            pass.commit_and_wait();
            if let Some(c) = capture.as_deref_mut() {
                c.patch_embed = read_bits(&s.x, wide);
            }

            let mut rows = seq;
            for n in 0..d.depth {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|err| {
                    RealForwardError::Unsupported(format!("minicpm vision block {n} read: {err}"))
                })?;
                let binding = BlockBinding {
                    buffer: &slots[slot],
                    roles: &blocks[n],
                };
                let pass = context.begin_pass();
                encode_block(context, &pass, &binding, &s, &d, rows)?;
                if n == INSERT_LAYER {
                    encode_vit_merger(context, &pass, weights, &resident.vit, &mut s, &d, rows)?;
                    rows /= 4;
                }
                pass.commit_and_wait();
                if let Some(c) = capture.as_deref_mut() {
                    if n == 0 {
                        c.block_0 = read_bits(&s.x, rows * d.hidden);
                    }
                    if n == INSERT_LAYER {
                        c.vit_merger = read_bits(&s.x, rows * d.hidden);
                    }
                    if n + 1 == d.depth {
                        c.last_block = read_bits(&s.x, rows * d.hidden);
                    }
                }
            }

            // post_layernorm, then the final merger over groups of four rows.
            let groups = rows / 4;
            let w4 = d.window() as u32;
            let pass = context.begin_pass();
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&s.x, 0),
                at(resident.post_ln_w),
                at(resident.post_ln_b),
                (&s.normed, 0),
                rows as u32,
                d.hidden as u32,
                EPS,
            )
            .map_err(e)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&s.normed, 0),
                at(resident.merger.pre_w),
                at(resident.merger.pre_b),
                (&s.proj, 0),
                groups as u32,
                w4,
                EPS,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.proj, 0),
                at(resident.merger.lin1_w),
                Some(at(resident.merger.lin1_b)),
                (&s.q, 0),
                groups as u32,
                w4,
                w4,
            )
            .map_err(e)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&s.q, 0),
                groups as u32 * w4,
                gpu::GeluKind::Tanh,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.q, 0),
                at(resident.merger.lin2_w),
                Some(at(resident.merger.lin2_b)),
                (&s.out, 0),
                groups as u32,
                w4,
                d.out_hidden as u32,
            )
            .map_err(e)?;
            pass.commit_and_wait();

            Ok(VisionEmbedding {
                rows: read_bits(&s.out, groups * d.out_hidden),
                merged_tokens: groups,
                out_hidden: d.out_hidden,
                grid: image.grid,
                deepstack: Vec::new(),
            })
        })
    }
}

/// Folds a diagnostic run into the shared stage record.
pub(crate) fn into_vision_stages(stages: MiniCpmStages) -> VisionStages {
    VisionStages {
        patch_embed: stages.patch_embed,
        block_first: stages.block_0,
        block_last: stages.last_block,
        ..VisionStages::default()
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn tower_order_makes_every_window_four_consecutive_raster_cells() {
        let (rows, cols) = (8, 12);
        let order = tower_order(rows, cols).unwrap();
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..rows * cols).collect::<Vec<_>>());
        for window in order.chunks_exact(4) {
            let (r0, c0) = (window[0] / cols, window[0] % cols);
            assert_eq!(r0 % 2, 0);
            assert_eq!(c0 % 2, 0);
            assert_eq!(
                window,
                &[
                    r0 * cols + c0,
                    r0 * cols + c0 + 1,
                    (r0 + 1) * cols + c0,
                    (r0 + 1) * cols + c0 + 1
                ]
            );
        }
        // Sixteen consecutive tokens are one 4x4 block, and blocks come in
        // raster order, so the final merger's rows are in output order.
        for (b, block) in order.chunks_exact(16).enumerate() {
            let (br, bc) = (b / (cols / 4), b % (cols / 4));
            for &cell in block {
                assert_eq!((cell / cols) / 4, br);
                assert_eq!((cell % cols) / 4, bc);
            }
        }
        // Inside a block the four windows are in (a, b) raster order.
        let block = &order[..16];
        let window_origin = |k: usize| (block[4 * k] / cols, block[4 * k] % cols);
        assert_eq!(
            [0, 1, 2, 3].map(window_origin),
            [(0, 0), (0, 2), (2, 0), (2, 2)]
        );
    }

    #[test]
    fn a_misaligned_grid_is_refused() {
        assert!(tower_order(6, 8).is_err());
        assert!(tower_order(8, 0).is_err());
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            assert!(x.is_finite() && y.is_finite(), "non-finite value");
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x) * f64::from(*x);
            nb += f64::from(*y) * f64::from(*y);
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn env_path(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(
            std::env::var(name).unwrap_or_else(|_| panic!("set {name} (see this test's doc)")),
        )
    }

    /// FP16 bits to f32, rows regrouped by `raster[k]` = destination row of row
    /// `k` (the tower's order back to the reference's raster order).
    fn to_raster(bits: &[u16], width: usize, raster: &[usize]) -> Vec<f32> {
        let mut out = vec![0.0f32; bits.len()];
        for (k, &dst) in raster.iter().enumerate() {
            for c in 0..width {
                out[dst * width + c] = half::f16::from_bits(bits[k * width + c]).to_f32();
            }
        }
        out
    }

    /// The real tower on a combined install against
    /// `scripts/minicpm_v_tower_reference.py`.
    ///
    /// That script is a numpy transcription of mlx-vlm's tower written from
    /// its source (read, not executed) in the reference's own raster
    /// formulation, so it shares no code with this tower and does not share
    /// its host token permutation: the stages are mapped back to raster order
    /// here before they are compared. Patch rows come from
    /// `crates/vision-io`, so this holds the preprocessing and the tower
    /// together.
    ///
    /// ```sh
    /// TURBOSPARK_HOME=/Volumes/T7/tshome-minicpm turbospark-model pull --repo \
    ///   mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4@810419dd09762d63aad6da5b6a418c4af8946e60 \
    ///   --alias minicpm-v47-vision
    /// TURBOSPARK_MINICPM_INSTALL_DIR=/Volumes/T7/tshome-minicpm/models/text/minicpm-v47-vision.gturbo \
    /// TURBOSPARK_MINICPM_MODEL_DIR=/Volumes/T7/hf-models/MiniCPM-V-4.7-35B-A3B-mxfp4 \
    /// TURBOSPARK_MINICPM_IMAGE=/Volumes/T7/vision-pages/shapes.png \
    /// TURBOSPARK_MINICPM_SCRATCH=/Volumes/T7/vision-dumps/minicpm-shapes \
    ///   cargo test -p turbospark-runtime --release --lib \
    ///   the_real_tower_matches_the_numpy_reference -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a MiniCPM-V combined install, the HF shards 1 and 5, an image and python3 with numpy"]
    fn the_real_tower_matches_the_numpy_reference() {
        let install = env_path("TURBOSPARK_MINICPM_INSTALL_DIR");
        let model_dir = env_path("TURBOSPARK_MINICPM_MODEL_DIR");
        let image = env_path("TURBOSPARK_MINICPM_IMAGE");
        let scratch = env_path("TURBOSPARK_MINICPM_SCRATCH");
        std::fs::create_dir_all(&scratch).unwrap();

        // A combined install, or the tower-only fixture
        // `crates/repack/examples/minicpm_tower_local.rs` writes from local
        // shards (a test fixture only: the runtime refuses to attach it).
        let arch = if model_io::is_sidecar_dir(&install) {
            let (_record, vision) = model_io::load_vision_sidecar(&install).expect("fixture");
            model_io::sidecar_arch(
                model_io::ModelFamily::QwenGdnMoe,
                vision.out_hidden_size,
                &vision,
            )
        } else {
            turbospark_repack::peek_manifest_arch(&install).expect("install manifest")
        };
        assert_eq!(arch.vision.tower, VisionTowerKind::MiniCpmV);
        let params = turbospark_vision_io::MiniCpmParams::from_preprocessor_config_json(
            &std::fs::read_to_string(install.join("preprocessor_config.json")).unwrap(),
        )
        .expect("preprocessor_config.json");
        let rgb = turbospark_vision_io::decode_image_file(&image).expect("decode");
        let pre = turbospark_vision_io::minicpm_preprocess(&rgb, &params).expect("preprocess");
        let (gh, gw) = (pre.grid.h, pre.grid.w);
        println!(
            "grid {gh}x{gw} ({} patches, {} tokens), resized {:?}",
            pre.grid.patches(),
            pre.merged_tokens,
            pre.resized
        );

        let rows_path = scratch.join("rows.bin");
        std::fs::write(
            &rows_path,
            pre.patch_rows
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let reference = scratch.join("reference");
        let status = std::process::Command::new("python3")
            .arg("-I")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../scripts/minicpm_v_tower_reference.py"
            ))
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("--rows")
            .arg(&rows_path)
            .args(["--grid", &gh.to_string(), &gw.to_string()])
            .arg("--out-dir")
            .arg(&reference)
            .status()
            .expect("python3");
        assert!(status.success(), "the numpy reference failed");

        let mut context = gpu::MetalContext::new().expect("Metal device");
        let index = model_io::load_resident_index(&install.join("model_weights.bin")).unwrap();
        let buffer = model_io::ResidentBuffer::map(
            &install.join("model_weights.bin"),
            index.header.index_size,
            index.header.resident_size,
        )
        .unwrap();
        let weights = gpu::ResidentGpuWeights::wrap(context.device(), buffer).unwrap();
        let mut tower =
            MiniCpmVisionTower::open(&install, &context, &weights, &index, &arch).expect("open");
        let mut stages = MiniCpmStages::default();
        let embedding = tower
            .run(
                &mut context,
                &weights,
                &pre,
                &params.as_preprocess_params(),
                Some(&mut stages),
            )
            .expect("tower run");

        // Tower order back to the reference's raster order. The merged window
        // tokens are the windows in (block, a, b) order.
        let order = tower_order(gh, gw).unwrap();
        let h = 1152;
        let windows: Vec<usize> = (0..gh * gw / 4)
            .map(|k| {
                let (block, within) = (k / 4, k % 4);
                let (br, bc) = (block / (gw / 4), block % (gw / 4));
                (br * 2 + within / 2) * (gw / 2) + (bc * 2 + within % 2)
            })
            .collect();
        let got = [
            ("patch_embed", to_raster(&stages.patch_embed, h, &order)),
            ("block_0", to_raster(&stages.block_0, h, &order)),
            ("vit_merger", to_raster(&stages.vit_merger, h, &windows)),
            ("last_block", to_raster(&stages.last_block, h, &windows)),
            (
                "output",
                embedding
                    .rows
                    .iter()
                    .map(|&b| half::f16::from_bits(b).to_f32())
                    .collect(),
            ),
        ];
        let mut worst = 1.0f64;
        for (name, got) in &got {
            let want = read_f32(&reference.join(format!("{name}.bin")));
            let c = cosine(got, &want);
            println!("{name:12} cosine {c:.8}");
            worst = worst.min(c);
        }
        assert!(
            worst > 0.999,
            "the tower disagrees with the reference (worst stage cosine {worst})"
        );
    }

    #[test]
    fn position_buckets_cover_the_table_monotonically() {
        for n in [4usize, 28, 32, 36, 70, 140] {
            let b = position_buckets(n, 70);
            assert_eq!(b.len(), n);
            assert_eq!(b[0], 0);
            assert!(b.windows(2).all(|w| w[0] <= w[1]), "n={n}");
            assert!(*b.last().unwrap() < 70, "n={n}");
        }
        // Values numpy produces for the reference's own arithmetic (f32 frac,
        // f64-built boundaries cast to f32). n = 28 is a TIE case: 2 / 28 is
        // exactly 5 / 70, and the reference counts the tie (`>=`), so index 2
        // lands in bucket 5, not 4.
        let b28 = position_buckets(28, 70);
        assert_eq!(&b28[..12], &[0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27]);
        assert_eq!(b28.iter().sum::<usize>(), 938);
        assert_eq!(
            &position_buckets(32, 70)[..12],
            &[0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 24]
        );
        assert_eq!(position_buckets(32, 70).iter().sum::<usize>(), 1070);
        assert_eq!(
            &position_buckets(36, 70)[..12],
            &[0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        );
        assert_eq!(position_buckets(140, 70).iter().sum::<usize>(), 4830);
        // 70 patches against a 70-wide table is the identity.
        assert_eq!(position_buckets(70, 70), (0..70).collect::<Vec<_>>());
        // 32 patches pick roughly every 2.19th row: frac 31/32 -> 67.
        assert_eq!(position_buckets(32, 70)[31], 67);
    }
}
