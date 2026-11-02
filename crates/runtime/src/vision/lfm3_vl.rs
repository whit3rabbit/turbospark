//! The LFM3-VL vision tower on Metal (`docs/LFM3_VL_VISION_PHASE0.md`).
//!
//! A sibling of the qwen and Gemma 4 towers in this module. Shared with them:
//! the destination format (blocks streamed two slots at a time from
//! `packed_vision/`, residents at FP16 under `vision.`) and the FP16 kernels
//! (matmul, bidirectional attention, 2-D rope, norms, residual add, mul).
//! New: the conv block's gated depthwise 3x3 (`encode_vision_gated_dwconv3x3`)
//! and SiLU.
//!
//! ```text
//! h = embedding_norm(patchifier(rows))                      // RMSNorm, no bias
//! for block in 14:                                          // conv or attention
//!     x = operator_norm(h)
//!     h = h + (conv(x) | attn(x))
//!     h = h + w2(silu(w1(ffn_norm(h))) * w3(ffn_norm(h)))
//! out = linear_2(gelu(linear_1(layer_norm(h) viewed as [seq/4, 4096])))
//! ```
//!
//! Tokens are in merge-window order throughout, which is the order the
//! projector's `[seq / 4, 4 * hidden]` view needs (four consecutive rows are
//! one 2x2 window) and the order the conv kernel gathers neighbours in.
//!
//! The 2-D rope is `vision_rope_freq_rows` at theta 1e6 on the 64-wide head:
//! row angles then column angles, both copies of the half implied by the
//! kernel's half-split pairing, which is the reference's `[angles, angles]`
//! concatenation.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};

use super::weights::{fp16_view, VISION_PREFIX};
use super::{VisionEmbedding, VisionStages, VISION_SLOTS};
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// RMS epsilon of every block, q/k and embedding norm (`norm_eps`).
const NORM_EPS: f32 = 1e-5;
/// The projector's LayerNorm epsilon (`projector_layer_norm_eps`).
const PROJECTOR_LN_EPS: f32 = 1e-6;
/// The 2-D rope base (`rope_parameters.rope_theta`).
const ROPE_THETA: f32 = 1_000_000.0;
/// The projector's hidden layer width (`projector_hidden_size`).
pub(crate) const PROJECTOR_HIDDEN: usize = 4096;
/// MLP row tile, as the other towers' Part B1: the `[rows, intermediate]`
/// gate and up buffers are sized for a tile, not the page.
pub(crate) const LFM3_VL_MLP_TILE_ROWS: usize = 1024;
const FP16_MAX: f32 = 65_504.0;
/// The largest page, in patches: `16_777_216 / 16^2`, the checkpoint's own
/// `longest_edge` ceiling.
pub(crate) const MAX_PATCHES: usize = 65_536;

/// The blob roles in a block's order. A conv block carries the first five and
/// `in_proj`, `conv_w`, `out_proj`; an attention block the first five and
/// `q_w`..`k_norm`. Restated rather than imported for the other towers'
/// reason: repack is a dev-dependency of this crate.
pub(crate) const LFM3_VL_ROLES: [&str; 14] = [
    "op_norm", "ffn_norm", "w1", "w3", "w2", "in_proj", "conv_w", "out_proj", "q_w", "k_w", "v_w",
    "o_w", "q_norm", "k_norm",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum R {
    OpNorm = 0,
    FfnNorm = 1,
    W1 = 2,
    W3 = 3,
    W2 = 4,
    InProj = 5,
    ConvW = 6,
    OutProj = 7,
    Q = 8,
    K = 9,
    V = 10,
    O = 11,
    QNorm = 12,
    KNorm = 13,
}

const COMMON: [R; 5] = [R::OpNorm, R::FfnNorm, R::W1, R::W3, R::W2];
const CONV: [R; 3] = [R::InProj, R::ConvW, R::OutProj];
const ATTN: [R; 6] = [R::Q, R::K, R::V, R::O, R::QNorm, R::KNorm];

/// The tower's geometry, resolved once from the install's [`VisionConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lfm3VlDims {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_dim: usize,
    pub(crate) merge: usize,
    pub(crate) text_hidden: usize,
    /// The projector's hidden width. [`PROJECTOR_HIDDEN`] in production; a
    /// field so a test can run a tower small enough to read.
    pub(crate) proj_hidden: usize,
}

impl Lfm3VlDims {
    pub(crate) fn resolve(v: &VisionConfig) -> Result<Self, RealForwardError> {
        if v.tower != VisionTowerKind::Lfm3Vl || !v.is_active() {
            return Err(RealForwardError::Unsupported(format!(
                "not an active lfm3_vl vision tower (kind {})",
                v.tower.as_str()
            )));
        }
        let us = |x: i64, what: &str| {
            usize::try_from(x)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| RealForwardError::Unsupported(format!("lfm3_vl vision {what} {x}")))
        };
        let d = Self {
            depth: us(v.depth, "depth")?,
            hidden: us(v.hidden_size, "hidden_size")?,
            intermediate: us(v.intermediate_size, "intermediate_size")?,
            heads: us(v.num_heads, "num_heads")?,
            head_dim: us(v.head_dim(), "head_dim")?,
            patch_dim: us(
                v.temporal_patch_size * v.patch_size * v.patch_size * v.in_channels,
                "patch_dim",
            )?,
            merge: us(v.spatial_merge_size, "spatial_merge_size")?,
            text_hidden: us(v.out_hidden_size, "out_hidden_size")?,
            proj_hidden: PROJECTOR_HIDDEN,
        };
        if d.heads * d.head_dim != d.hidden || d.head_dim % 4 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "lfm3_vl vision hidden {} over {} heads does not give a head_dim divisible by 4",
                d.hidden, d.heads
            )));
        }
        if d.head_dim as u32 > gpu::MAX_ATTENTION_HEAD_DIM {
            return Err(RealForwardError::Unsupported(format!(
                "lfm3_vl vision head_dim {} is past the attention kernel's {}",
                d.head_dim,
                gpu::MAX_ATTENTION_HEAD_DIM
            )));
        }
        Ok(d)
    }

    fn role_elems(&self, role: R) -> usize {
        let (h, i) = (self.hidden, self.intermediate);
        match role {
            R::OpNorm | R::FfnNorm => h,
            R::QNorm | R::KNorm => self.head_dim,
            R::W1 | R::W3 | R::W2 => i * h,
            R::InProj => 3 * h * h,
            R::ConvW => h * 9,
            R::OutProj | R::Q | R::K | R::V | R::O => h * h,
        }
    }
}

/// One block's byte offsets within its blob, and which kind it is.
#[derive(Debug, Clone)]
pub(crate) struct Lfm3VlBlockRoles {
    offsets: [u64; 14],
    is_conv: bool,
}

impl Lfm3VlBlockRoles {
    fn at(&self, role: R) -> u64 {
        self.offsets[role as usize]
    }

    /// Resolve and size-check one block's layout entry. The kind is the one
    /// whose full role set the blob carries; a blob carrying neither, or
    /// both, is refused.
    pub(crate) fn resolve(
        entry: &model_io::ExpertEntry,
        stride: u64,
        d: &Lfm3VlDims,
    ) -> Result<Self, RealForwardError> {
        let block = entry.expert;
        let has = |role: R| entry.sub_tensors.contains_key(LFM3_VL_ROLES[role as usize]);
        let is_conv = CONV.iter().all(|&r| has(r));
        let is_attn = ATTN.iter().all(|&r| has(r));
        if is_conv == is_attn || CONV.iter().any(|&r| has(r)) && ATTN.iter().any(|&r| has(r)) {
            return Err(RealForwardError::Unsupported(format!(
                "lfm3_vl vision block {block} is not exactly one of a conv block and an \
                 attention block"
            )));
        }
        let mut offsets = [0u64; 14];
        let kind: &[R] = if is_conv { &CONV } else { &ATTN };
        for &role in COMMON.iter().chain(kind) {
            let name = LFM3_VL_ROLES[role as usize];
            let sub = entry.sub_tensors.get(name).ok_or_else(|| {
                RealForwardError::MissingTensor(format!("lfm3_vl vision block {block} {name}"))
            })?;
            let want = (d.role_elems(role) * 2) as u64;
            if sub.dtype != "fp16" || sub.size != want {
                return Err(RealForwardError::Unsupported(format!(
                    "lfm3_vl vision block {block} {name}: {} bytes of {}, expected {want} of fp16",
                    sub.size, sub.dtype
                )));
            }
            if sub.offset % 4 != 0 || sub.offset + sub.size > stride {
                return Err(RealForwardError::Unsupported(format!(
                    "lfm3_vl vision block {block} {name} at {}..{} does not fit a 4-byte aligned \
                     slot of a {stride}-byte blob",
                    sub.offset,
                    sub.offset + sub.size
                )));
            }
            offsets[role as usize] = sub.offset;
        }
        Ok(Self { offsets, is_conv })
    }
}

/// Where one block's weights are bound from: a streamer slot (base 0) or a
/// test buffer holding the blob at `base`.
pub(crate) struct BlockBinding<'a> {
    pub(crate) buffer: &'a gpu::MetalBuffer,
    pub(crate) base: u64,
    pub(crate) roles: &'a Lfm3VlBlockRoles,
}

impl BlockBinding<'_> {
    fn w(&self, role: R) -> (&gpu::MetalBuffer, u64) {
        (self.buffer, self.base + self.roles.at(role))
    }
}

/// Per-image GPU scratch.
///
/// `a` is the normed input and, for a conv block, the conv core's output; for
/// an attention block it doubles as the attention output (last read by the v
/// projection, which is encoded before attention writes it). `q` is also the
/// projector's `[seq / 4, 4096]` hidden buffer, which has the same byte size.
pub(crate) struct Lfm3VlScratch {
    seq: usize,
    tile: usize,
    h: gpu::MetalBuffer,
    a: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    k: gpu::MetalBuffer,
    v: gpu::MetalBuffer,
    o: gpu::MetalBuffer,
    /// The conv block's `in_proj` output, `[seq, 3 * hidden]`.
    p3: gpu::MetalBuffer,
    g: gpu::MetalBuffer,
    u: gpu::MetalBuffer,
    freqs: gpu::MetalBuffer,
}

impl Lfm3VlScratch {
    fn allocate(
        context: &gpu::MetalContext,
        d: &Lfm3VlDims,
        seq: usize,
        tile: usize,
        freqs: &[f32],
    ) -> Self {
        let wide = (seq * d.hidden * 2) as u64;
        let tile = tile.min(seq).max(1);
        let mlp = (tile * d.intermediate * 2) as u64;
        Self {
            seq,
            tile,
            h: context.new_output_buffer(wide),
            a: context.new_output_buffer(wide),
            q: context.new_output_buffer(wide),
            k: context.new_output_buffer(wide),
            v: context.new_output_buffer(wide),
            o: context.new_output_buffer(wide),
            p3: context.new_output_buffer(3 * wide),
            g: context.new_output_buffer(mlp),
            u: context.new_output_buffer(mlp),
            freqs: context.new_buffer_with_data(freqs),
        }
    }
}

/// Encode one block over `s.h` in place. `grid` is the patch grid `(height,
/// width)` the conv block convolves over.
pub(crate) fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    b: &BlockBinding<'_>,
    s: &Lfm3VlScratch,
    d: &Lfm3VlDims,
    grid: (usize, usize),
) -> Result<(), RealForwardError> {
    let e = RealForwardError::Gpu;
    let (seq, hid, nh, hd, inter) = (
        s.seq as u32,
        d.hidden as u32,
        d.heads as u32,
        d.head_dim as u32,
        d.intermediate as u32,
    );
    let wide = seq * hid;

    // Operator half: h += out_proj(...) | o_proj(attn(...)).
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.h, 0),
        Some(b.w(R::OpNorm)),
        (&s.a, 0),
        seq,
        hid,
        NORM_EPS,
    )
    .map_err(e)?;
    if b.roles.is_conv {
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.a, 0),
            b.w(R::InProj),
            None,
            (&s.p3, 0),
            seq,
            hid,
            3 * hid,
        )
        .map_err(e)?;
        gpu::encode_vision_gated_dwconv3x3(
            context,
            pass,
            (&s.a, 0),
            (&s.p3, 0),
            b.w(R::ConvW),
            grid.0 as u32,
            grid.1 as u32,
            d.merge as u32,
            hid,
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.a, 0),
            b.w(R::OutProj),
            None,
            (&s.o, 0),
            seq,
            hid,
            hid,
        )
        .map_err(e)?;
    } else {
        let head_rows = seq * nh;
        for (w, out) in [(R::Q, &s.q), (R::K, &s.k), (R::V, &s.v)] {
            gpu::encode_vision_matmul(
                context,
                pass,
                (&s.a, 0),
                b.w(w),
                None,
                (out, 0),
                seq,
                hid,
                hid,
            )
            .map_err(e)?;
        }
        // Per-head RMSNorm on q and k (v carries none). K was expanded to the
        // full head count at repack, so both are `seq * heads` rows.
        for (w, x) in [(R::QNorm, &s.q), (R::KNorm, &s.k)] {
            gpu::encode_vision_rms_norm(
                context,
                pass,
                (x, 0),
                Some(b.w(w)),
                (x, 0),
                head_rows,
                hd,
                NORM_EPS,
            )
            .map_err(e)?;
        }
        // Q AND K ONLY, as every attention this port implements.
        for qk in [&s.q, &s.k] {
            gpu::encode_vision_rope_2d(context, pass, (qk, 0), (&s.freqs, 0), seq, nh, hd)
                .map_err(e)?;
        }
        gpu::encode_vision_attention(
            context,
            pass,
            (&s.q, 0),
            (&s.k, 0),
            (&s.v, 0),
            (&s.a, 0),
            seq,
            nh,
            hd,
            (d.head_dim as f32).powf(-0.5),
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.a, 0),
            b.w(R::O),
            None,
            (&s.o, 0),
            seq,
            hid,
            hid,
        )
        .map_err(e)?;
    }
    gpu::encode_vision_residual_add(context, pass, (&s.h, 0), (&s.o, 0), wide).map_err(e)?;

    // MLP half, row-tiled: h += w2(silu(w1(m)) * w3(m)).
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.h, 0),
        Some(b.w(R::FfnNorm)),
        (&s.a, 0),
        seq,
        hid,
        NORM_EPS,
    )
    .map_err(e)?;
    let mut r0 = 0usize;
    while r0 < s.seq {
        let rows = (s.seq - r0).min(s.tile) as u32;
        let at = (r0 * d.hidden * 2) as u64;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.a, at),
            b.w(R::W1),
            None,
            (&s.g, 0),
            rows,
            hid,
            inter,
        )
        .map_err(e)?;
        gpu::encode_vision_silu(context, pass, (&s.g, 0), rows * inter).map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.a, at),
            b.w(R::W3),
            None,
            (&s.u, 0),
            rows,
            hid,
            inter,
        )
        .map_err(e)?;
        gpu::encode_vision_mul(context, pass, (&s.g, 0), (&s.u, 0), rows * inter).map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.g, 0),
            b.w(R::W2),
            None,
            (&s.o, at),
            rows,
            inter,
            hid,
        )
        .map_err(e)?;
        r0 += s.tile;
    }
    gpu::encode_vision_residual_add(context, pass, (&s.h, 0), (&s.o, 0), wide).map_err(e)
}

/// Resident weights as `(buffer, offset)` pairs, resolved once.
pub(crate) struct Lfm3VlResidents<'a> {
    pub(crate) patchifier: (&'a gpu::MetalBuffer, u64),
    pub(crate) embedding_norm: (&'a gpu::MetalBuffer, u64),
    pub(crate) ln_w: (&'a gpu::MetalBuffer, u64),
    pub(crate) ln_b: (&'a gpu::MetalBuffer, u64),
    pub(crate) lin1_w: (&'a gpu::MetalBuffer, u64),
    pub(crate) lin1_b: (&'a gpu::MetalBuffer, u64),
    pub(crate) lin2_w: (&'a gpu::MetalBuffer, u64),
    pub(crate) lin2_b: (&'a gpu::MetalBuffer, u64),
}

/// Each stage's rows (FP16 bits), for a parity test that must say where it
/// diverged.
#[derive(Debug, Default)]
pub(crate) struct Lfm3VlStages {
    pub(crate) patch_embed: Vec<u16>,
    pub(crate) blocks: Vec<Vec<u16>>,
}

fn read_bits(buffer: &gpu::MetalBuffer, count: usize) -> Vec<u16> {
    gpu::read_buffer_f16(buffer, 0, count)
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

fn to_f16(values: &[f32], what: &str) -> Result<Vec<half::f16>, RealForwardError> {
    values
        .iter()
        .map(|&v| {
            if v.is_finite() && v.abs() <= FP16_MAX {
                Ok(half::f16::from_f32(v))
            } else {
                Err(RealForwardError::Unsupported(format!(
                    "lfm3_vl vision {what} holds {v}, outside FP16; the tower would saturate"
                )))
            }
        })
        .collect()
}

/// The whole tower and projector for one image.
///
/// Generic over where blocks live, so a test can bind them from memory:
/// `encode_nth_block` encodes block `n` into the pass it is given (the
/// install's version loads the block into a streamer slot first). One command
/// buffer per stage, waited, which is what makes reusing a slot for block
/// `n + 2` safe. Returns `[seq / merge^2, text_hidden]` FP16 bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    context: &mut gpu::MetalContext,
    d: &Lfm3VlDims,
    pixels: &[f32],
    freqs: &[f32],
    grid: (usize, usize),
    res: &Lfm3VlResidents<'_>,
    tile_rows: usize,
    mut encode_nth_block: impl FnMut(
        &mut gpu::MetalContext,
        &gpu::PassEncoder,
        usize,
        &Lfm3VlScratch,
    ) -> Result<(), RealForwardError>,
    mut capture: Option<&mut Lfm3VlStages>,
) -> Result<Vec<u16>, RealForwardError> {
    let bad = |m: String| RealForwardError::Unsupported(m);
    let (gh, gw) = grid;
    let n = gh * gw;
    let per_token = d.merge * d.merge;
    // The checkpoint's declared pixel ceiling (16,777,216) is exactly this many
    // 16x16 patches. The tower's scratch is eight page-sized FP16 buffers, so
    // a larger page is refused by name here rather than failing in the
    // allocator; this tower is not covered by the Qwen pixel-budget resolver
    // (it has no position table to shape).
    if n > MAX_PATCHES {
        return Err(bad(format!(
            "lfm3_vl vision page of {n} patches is past the {MAX_PATCHES}-patch ceiling; lower \
             the image's max_pixels"
        )));
    }
    if n == 0 || gh % d.merge != 0 || gw % d.merge != 0 || pixels.len() != n * d.patch_dim {
        return Err(bad(format!(
            "lfm3_vl vision input: {gh}x{gw} patches and {} values; expected whole {}x{} merge \
             windows of {}-value patches",
            pixels.len(),
            d.merge,
            d.merge,
            d.patch_dim
        )));
    }
    if freqs.len() != n * d.head_dim / 2 {
        return Err(bad(format!(
            "lfm3_vl vision rope rows hold {} values, expected {}",
            freqs.len(),
            n * d.head_dim / 2
        )));
    }
    let soft = n / per_token;
    let hid = d.hidden;
    let pixels16 = to_f16(pixels, "pixel rows")?;

    gpu::autorelease_pool(|| {
        let e = RealForwardError::Gpu;
        let s = Lfm3VlScratch::allocate(context, d, n, tile_rows, freqs);
        let px = context.new_buffer_with_data(&pixels16);

        let pass = context.begin_pass();
        gpu::encode_vision_matmul(
            context,
            &pass,
            (&px, 0),
            res.patchifier,
            None,
            (&s.h, 0),
            n as u32,
            d.patch_dim as u32,
            hid as u32,
        )
        .map_err(e)?;
        gpu::encode_vision_rms_norm(
            context,
            &pass,
            (&s.h, 0),
            Some(res.embedding_norm),
            (&s.h, 0),
            n as u32,
            hid as u32,
            NORM_EPS,
        )
        .map_err(e)?;
        pass.commit_and_wait();
        if let Some(c) = capture.as_deref_mut() {
            c.patch_embed = read_bits(&s.h, n * hid);
        }

        for block in 0..d.depth {
            let pass = context.begin_pass();
            encode_nth_block(context, &pass, block, &s)?;
            pass.commit_and_wait();
            if let Some(c) = capture.as_deref_mut() {
                c.blocks.push(read_bits(&s.h, n * hid));
            }
        }

        // The residual stream leaves the tower here; refuse a saturated one
        // by name rather than project it.
        let last = read_bits(&s.h, n * hid);
        if let Some(i) = last
            .iter()
            .position(|&b| !half::f16::from_bits(b).to_f32().is_finite())
        {
            return Err(bad(format!(
                "lfm3_vl vision tower output is non-finite at {i}; the FP16 residual stream \
                 overflowed"
            )));
        }

        // Projector: LayerNorm per row, then the same bytes viewed as
        // `[soft, merge^2 * hidden]` rows (four consecutive merge-window
        // tokens are one 2x2 window).
        let out = context.new_output_buffer((soft * d.text_hidden * 2) as u64);
        let pass = context.begin_pass();
        gpu::encode_vision_layer_norm(
            context,
            &pass,
            (&s.h, 0),
            res.ln_w,
            res.ln_b,
            (&s.a, 0),
            n as u32,
            hid as u32,
            PROJECTOR_LN_EPS,
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            &pass,
            (&s.a, 0),
            res.lin1_w,
            Some(res.lin1_b),
            (&s.q, 0),
            soft as u32,
            (hid * per_token) as u32,
            d.proj_hidden as u32,
        )
        .map_err(e)?;
        gpu::encode_vision_gelu(
            context,
            &pass,
            (&s.q, 0),
            (soft * d.proj_hidden) as u32,
            gpu::GeluKind::Erf,
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            &pass,
            (&s.q, 0),
            res.lin2_w,
            Some(res.lin2_b),
            (&out, 0),
            soft as u32,
            d.proj_hidden as u32,
            d.text_hidden as u32,
        )
        .map_err(e)?;
        pass.commit_and_wait();
        Ok(read_bits(&out, soft * d.text_hidden))
    })
}

/// The LFM3-VL tower opened against an install: blocks streamed from
/// `packed_vision/` into [`VISION_SLOTS`] slots, residents resolved off the
/// trunk's own index. Opened lazily on the first image.
pub(crate) struct Lfm3VlVisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<Lfm3VlBlockRoles>,
    dims: Lfm3VlDims,
    /// Resident byte offsets: patchifier, embedding norm, then the
    /// projector's six.
    offsets: [u64; 8],
    pub(crate) slot_bytes: u64,
}

impl Lfm3VlVisionTower {
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let dims = Lfm3VlDims::resolve(&arch.vision)?;
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
        // By the entry's own index, never by position: the writer may pack
        // out of order.
        let mut blocks: Vec<Option<Lfm3VlBlockRoles>> = vec![None; dims.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, dims.depth
                ))
            })?;
            if slot
                .replace(Lfm3VlBlockRoles::resolve(
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
        .map_err(|e| RealForwardError::Unsupported(format!("lfm3_vl vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }

        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let (h, p, t) = (dims.hidden, dims.proj_hidden, dims.text_hidden);
        let offsets = [
            fp16_view(index, &name("patchifier.weight"), h * dims.patch_dim)?,
            fp16_view(index, &name("embedding_norm.weight"), h)?,
            fp16_view(index, &name("projector.layer_norm.weight"), h)?,
            fp16_view(index, &name("projector.layer_norm.bias"), h)?,
            fp16_view(
                index,
                &name("projector.linear_1.weight"),
                p * h * dims.merge * dims.merge,
            )?,
            fp16_view(index, &name("projector.linear_1.bias"), p)?,
            fp16_view(index, &name("projector.linear_2.weight"), t * p)?,
            fp16_view(index, &name("projector.linear_2.bias"), t)?,
        ];
        let slot_bytes = VISION_SLOTS as u64 * layer.expert_stride;
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            dims,
            offsets,
            slot_bytes,
        })
    }

    /// Encode one image to `[merged, text_hidden]` FP16 bits, with the
    /// per-stage rows when `stages` is given.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &turbospark_vision_io::PreprocessedImage,
        params: &turbospark_vision_io::PreprocessParams,
        stages: Option<&mut Lfm3VlStages>,
    ) -> Result<VisionEmbedding, RealForwardError> {
        let d = self.dims;
        if image.grid.t != 1 {
            return Err(RealForwardError::Unsupported(format!(
                "lfm3_vl vision takes still images (one temporal group), got {}",
                image.grid.t
            )));
        }
        let freqs =
            turbospark_vision_io::vision_rope_freq_rows(image.grid, d.head_dim, ROPE_THETA, params)
                .map_err(|e| RealForwardError::Unsupported(format!("lfm3_vl rope rows: {e}")))?;
        let at = |i: usize| (weights.buffer(), weights.gpu_offset(self.offsets[i]));
        let res = Lfm3VlResidents {
            patchifier: at(0),
            embedding_norm: at(1),
            ln_w: at(2),
            ln_b: at(3),
            lin1_w: at(4),
            lin1_b: at(5),
            lin2_w: at(6),
            lin2_b: at(7),
        };
        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let grid = (image.grid.h, image.grid.w);
        let rows = forward(
            context,
            &d,
            &image.patch_rows,
            &freqs,
            grid,
            &res,
            LFM3_VL_MLP_TILE_ROWS,
            |context, pass, n, s| {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|e| {
                    RealForwardError::Unsupported(format!("lfm3_vl vision block {n} read: {e}"))
                })?;
                let b = BlockBinding {
                    buffer: &slots[slot],
                    base: 0,
                    roles: &blocks[n],
                };
                encode_block(context, pass, &b, s, &d, grid)
            },
            stages,
        )?;
        Ok(VisionEmbedding {
            merged_tokens: rows.len() / d.text_hidden,
            out_hidden: d.text_hidden,
            grid: image.grid,
            rows,
            deepstack: Vec::new(),
        })
    }
}

/// The runner-facing stage rows: first and last block, and the patch embedding.
pub(crate) fn into_vision_stages(s: Lfm3VlStages) -> VisionStages {
    VisionStages {
        patch_embed: s.patch_embed,
        block_first: s.blocks.first().cloned().unwrap_or_default(),
        block_last: s.blocks.last().cloned().unwrap_or_default(),
        ..VisionStages::default()
    }
}
