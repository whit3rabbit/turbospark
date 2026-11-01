//! The Gemma 4 vision tower on Metal (`docs/GEMMA4_VISION_PHASE0.md`).
//!
//! A sibling of the qwen tower in this module, not a mode of it: the block,
//! the rope, the position table and the output stage all differ, and only
//! the destination format (blocks streamed two slots at a time from
//! `packed_vision/`, residents at FP16 under `vision.`) and four kernels
//! (matmul, bidirectional attention, tanh GELU, residual add) are shared.
//! `turbospark_compute::vision_gemma4` is the definition of what this
//! computes, and the unit tests below hold it to mlx-vlm's own fixture.
//!
//! # What runs where
//!
//! - HOST: `2 * (x - 0.5)` on the pixels, the two axial position rows per
//!   patch (a gather from a `[2, rows, hidden]` table, as the qwen tower's
//!   position blend is host-side), the rope angles, and the whole pooling
//!   stage. Pooling, the `sqrt(hidden)` scale and the standardize run in F32
//!   because the scaled rows leave FP16's range before the standardize brings
//!   them back (Phase 0 trap 5); the result is range-checked before it is
//!   narrowed.
//! - GPU: the patch projection, every block, the projector's scale-free
//!   pre-norm and the projection.
//!
//! # Attention scale 1.0, v normed without a weight
//!
//! The q/k norms carry the scale and `v` is RMS-normed per head with no
//! weight. Both are pinned by `compute`'s mutation-checked reference test and
//! repeated here only as call-site arguments.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};

use super::weights::{fp16_view, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

const EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 100.0;
/// MLP row tile, the qwen tower's Part B1 shape: the `[rows, intermediate]`
/// gate and up buffers are sized for a tile, not the page.
pub(crate) const GEMMA4_MLP_TILE_ROWS: usize = 1024;
const FP16_MAX: f32 = 65_504.0;

/// The blob roles in byte order, the repack writer's
/// `GEMMA4_BLOCK_ROLES` first column. Restated rather than imported for the
/// qwen table's reason (`weights::BLOCK_ROLES`): repack is a dev-dependency.
pub(crate) const GEMMA4_BLOCK_ROLES: [&str; 13] = [
    "ln_in",
    "q_w",
    "k_w",
    "v_w",
    "q_norm",
    "k_norm",
    "o_w",
    "ln_post_attn",
    "ln_pre_ffw",
    "gate_w",
    "up_w",
    "down_w",
    "ln_post_ffw",
];

#[derive(Debug, Clone, Copy)]
enum R {
    LnIn = 0,
    Q = 1,
    K = 2,
    V = 3,
    QNorm = 4,
    KNorm = 5,
    O = 6,
    LnPostAttn = 7,
    LnPreFfw = 8,
    Gate = 9,
    Up = 10,
    Down = 11,
    LnPostFfw = 12,
}

/// The tower's geometry, resolved once from the install's [`VisionConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gemma4Dims {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_dim: usize,
    pub(crate) pos_rows: usize,
    pub(crate) pool: usize,
    pub(crate) text_hidden: usize,
}

impl Gemma4Dims {
    pub(crate) fn resolve(v: &VisionConfig) -> Result<Self, RealForwardError> {
        if v.tower != VisionTowerKind::Gemma4 || !v.is_active() {
            return Err(RealForwardError::Unsupported(format!(
                "not an active gemma4 vision tower (kind {})",
                v.tower.as_str()
            )));
        }
        let us = |x: i64, what: &str| {
            usize::try_from(x)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| RealForwardError::Unsupported(format!("gemma4 vision {what} {x}")))
        };
        let d = Self {
            depth: us(v.depth, "depth")?,
            hidden: us(v.hidden_size, "hidden_size")?,
            intermediate: us(v.intermediate_size, "intermediate_size")?,
            heads: us(v.num_heads, "num_heads")?,
            head_dim: us(v.head_dim(), "head_dim")?,
            patch_dim: us(v.patch_size * v.patch_size * v.in_channels, "patch_dim")?,
            pos_rows: us(v.num_position_embeddings, "position rows")?,
            pool: us(v.spatial_merge_size, "pooling kernel")?,
            text_hidden: us(v.out_hidden_size, "out_hidden_size")?,
        };
        if d.heads * d.head_dim != d.hidden || d.head_dim % 4 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "gemma4 vision hidden {} over {} heads does not give a head_dim divisible by 4",
                d.hidden, d.heads
            )));
        }
        if d.head_dim as u32 > gpu::MAX_ATTENTION_HEAD_DIM {
            return Err(RealForwardError::Unsupported(format!(
                "gemma4 vision head_dim {} is past the attention kernel's {}",
                d.head_dim,
                gpu::MAX_ATTENTION_HEAD_DIM
            )));
        }
        Ok(d)
    }

    fn role_elems(&self, role: R) -> usize {
        let (h, i) = (self.hidden, self.intermediate);
        match role {
            R::LnIn | R::LnPostAttn | R::LnPreFfw | R::LnPostFfw => h,
            R::QNorm | R::KNorm => self.head_dim,
            R::Q | R::K | R::V | R::O => h * h,
            R::Gate | R::Up | R::Down => i * h,
        }
    }
}

/// One block's thirteen byte offsets within its blob.
#[derive(Debug, Clone)]
pub(crate) struct Gemma4BlockRoles {
    offsets: [u64; 13],
}

impl Gemma4BlockRoles {
    fn at(&self, role: R) -> u64 {
        self.offsets[role as usize]
    }

    /// Resolve and size-check one block's layout entry.
    pub(crate) fn resolve(
        entry: &model_io::ExpertEntry,
        stride: u64,
        d: &Gemma4Dims,
    ) -> Result<Self, RealForwardError> {
        const ALL: [R; 13] = [
            R::LnIn,
            R::Q,
            R::K,
            R::V,
            R::QNorm,
            R::KNorm,
            R::O,
            R::LnPostAttn,
            R::LnPreFfw,
            R::Gate,
            R::Up,
            R::Down,
            R::LnPostFfw,
        ];
        let block = entry.expert;
        let mut offsets = [0u64; 13];
        for role in ALL {
            let name = GEMMA4_BLOCK_ROLES[role as usize];
            let sub = entry.sub_tensors.get(name).ok_or_else(|| {
                RealForwardError::MissingTensor(format!("gemma4 vision block {block} {name}"))
            })?;
            let want = (d.role_elems(role) * 2) as u64;
            if sub.dtype != "fp16" || sub.size != want {
                return Err(RealForwardError::Unsupported(format!(
                    "gemma4 vision block {block} {name}: {} bytes of {}, expected {want} of fp16",
                    sub.size, sub.dtype
                )));
            }
            if sub.offset % 4 != 0 || sub.offset + sub.size > stride {
                return Err(RealForwardError::Unsupported(format!(
                    "gemma4 vision block {block} {name} at {}..{} does not fit a 4-byte aligned \
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

/// Where one block's weights are bound from: a streamer slot (base 0) or a
/// test buffer holding the blob at `base`.
pub(crate) struct BlockBinding<'a> {
    pub(crate) buffer: &'a gpu::MetalBuffer,
    pub(crate) base: u64,
    pub(crate) roles: &'a Gemma4BlockRoles,
}

impl BlockBinding<'_> {
    fn w(&self, role: R) -> (&gpu::MetalBuffer, u64) {
        (self.buffer, self.base + self.roles.at(role))
    }
}

/// Per-image GPU scratch. `a` doubles as the attention output: it is last
/// read by the v projection, which is encoded before attention writes it.
pub(crate) struct Gemma4Scratch {
    seq: usize,
    tile: usize,
    h: gpu::MetalBuffer,
    a: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    k: gpu::MetalBuffer,
    v: gpu::MetalBuffer,
    o: gpu::MetalBuffer,
    g: gpu::MetalBuffer,
    u: gpu::MetalBuffer,
    freqs: gpu::MetalBuffer,
}

impl Gemma4Scratch {
    fn allocate(
        context: &gpu::MetalContext,
        d: &Gemma4Dims,
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
            g: context.new_output_buffer(mlp),
            u: context.new_output_buffer(mlp),
            freqs: context.new_buffer_with_data(freqs),
        }
    }
}

/// Encode one block over `s.h` in place.
pub(crate) fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    b: &BlockBinding<'_>,
    s: &Gemma4Scratch,
    d: &Gemma4Dims,
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
    let head_rows = seq * nh;

    // Attention half: h += post_attn_norm(o(attn(rope(norm(q)), rope(norm(k)), norm(v)))).
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.h, 0),
        Some(b.w(R::LnIn)),
        (&s.a, 0),
        seq,
        hid,
        EPS,
    )
    .map_err(e)?;
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
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.q, 0),
        Some(b.w(R::QNorm)),
        (&s.q, 0),
        head_rows,
        hd,
        EPS,
    )
    .map_err(e)?;
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.k, 0),
        Some(b.w(R::KNorm)),
        (&s.k, 0),
        head_rows,
        hd,
        EPS,
    )
    .map_err(e)?;
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.v, 0),
        None,
        (&s.v, 0),
        head_rows,
        hd,
        EPS,
    )
    .map_err(e)?;
    for qk in [&s.q, &s.k] {
        gpu::encode_vision_rope_axial(context, pass, (qk, 0), (&s.freqs, 0), seq, nh, hd)
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
        1.0,
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
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.o, 0),
        Some(b.w(R::LnPostAttn)),
        (&s.o, 0),
        seq,
        hid,
        EPS,
    )
    .map_err(e)?;
    gpu::encode_vision_residual_add(context, pass, (&s.h, 0), (&s.o, 0), wide).map_err(e)?;

    // MLP half, row-tiled: h += post_ffw_norm(down(gelu_tanh(gate(m)) * up(m))).
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.h, 0),
        Some(b.w(R::LnPreFfw)),
        (&s.a, 0),
        seq,
        hid,
        EPS,
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
            b.w(R::Gate),
            None,
            (&s.g, 0),
            rows,
            hid,
            inter,
        )
        .map_err(e)?;
        gpu::encode_vision_gelu(context, pass, (&s.g, 0), rows * inter, gpu::GeluKind::Tanh)
            .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.a, at),
            b.w(R::Up),
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
            b.w(R::Down),
            None,
            (&s.o, at),
            rows,
            inter,
            hid,
        )
        .map_err(e)?;
        r0 += s.tile;
    }
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.o, 0),
        Some(b.w(R::LnPostFfw)),
        (&s.o, 0),
        seq,
        hid,
        EPS,
    )
    .map_err(e)?;
    gpu::encode_vision_residual_add(context, pass, (&s.h, 0), (&s.o, 0), wide).map_err(e)
}

/// Per patch, the x part's `head_dim / 4` angles then the y part's, F32.
pub(crate) fn freq_rows(positions: &[(usize, usize)], head_dim: usize) -> Vec<f32> {
    let part = head_dim / 2;
    let quarter = head_dim / 4;
    let mut out = Vec::with_capacity(positions.len() * part);
    for &(x, y) in positions {
        for p in [x, y] {
            for i in 0..quarter {
                out.push(p as f32 * ROPE_THETA.powf(-(2.0 * i as f32) / part as f32));
            }
        }
    }
    out
}

/// The host-resident pieces of the tower, decoded once at open.
pub(crate) struct Gemma4HostTables {
    /// `[2, pos_rows, hidden]`: the x table, then the y table.
    pub(crate) pos_table: Vec<f32>,
    pub(crate) std_bias: Vec<f32>,
    pub(crate) std_scale: Vec<f32>,
}

/// One image as the tower takes it.
pub(crate) struct Gemma4Input<'a> {
    /// `[patches, patch_dim]` in `[0, 1]`, row-major over the grid.
    pub(crate) pixels: &'a [f32],
    /// `(x, y)` per patch.
    pub(crate) positions: &'a [(usize, usize)],
    pub(crate) grid_w: usize,
}

/// Each stage's rows, for a parity test that must say where it diverged.
#[derive(Debug, Default)]
pub(crate) struct Gemma4Stages {
    pub(crate) patch_embed: Vec<f32>,
    pub(crate) blocks: Vec<Vec<f32>>,
    pub(crate) tower: Vec<f32>,
}

fn read_f32(buffer: &gpu::MetalBuffer, count: usize) -> Vec<f32> {
    gpu::read_buffer_f16(buffer, 0, count)
        .into_iter()
        .map(|v| v.to_f32())
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
                    "gemma4 vision {what} holds {v}, outside FP16; the tower would saturate"
                )))
            }
        })
        .collect()
}

/// The whole tower and projector for one image.
///
/// Generic over where weights live, so a test can bind them from memory:
/// `input_proj` and `projector` are resident `(buffer, offset)` pairs, and
/// `encode_nth_block` encodes block `n` into the pass it is given (the
/// install's version loads the block into a streamer slot first). One
/// command buffer per stage, waited, which is what makes reusing a slot for
/// block `n + 2` safe. Returns `[soft, text_hidden]` FP16 bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward(
    context: &mut gpu::MetalContext,
    d: &Gemma4Dims,
    input: &Gemma4Input<'_>,
    host: &Gemma4HostTables,
    input_proj: (&gpu::MetalBuffer, u64),
    projector: (&gpu::MetalBuffer, u64),
    tile_rows: usize,
    mut encode_nth_block: impl FnMut(
        &mut gpu::MetalContext,
        &gpu::PassEncoder,
        usize,
        &Gemma4Scratch,
    ) -> Result<(), RealForwardError>,
    mut capture: Option<&mut Gemma4Stages>,
) -> Result<Vec<u16>, RealForwardError> {
    let n = input.positions.len();
    let per_token = d.pool * d.pool;
    let bad = |m: String| RealForwardError::Unsupported(m);
    if n == 0
        || input.pixels.len() != n * d.patch_dim
        || input.grid_w == 0
        || input.grid_w % d.pool != 0
        || n % input.grid_w != 0
        || (n / input.grid_w) % d.pool != 0
    {
        return Err(bad(format!(
            "gemma4 vision input: {n} patches, {} values, grid width {}; expected whole {}x{} \
             pool blocks of {}-value patches",
            input.pixels.len(),
            input.grid_w,
            d.pool,
            d.pool,
            d.patch_dim
        )));
    }
    if let Some(&(x, y)) = input
        .positions
        .iter()
        .find(|&&(x, y)| x >= d.pos_rows || y >= d.pos_rows)
    {
        return Err(bad(format!(
            "gemma4 vision patch at ({x}, {y}) is past the {}-row position table",
            d.pos_rows
        )));
    }
    let soft = n / per_token;
    let hid = d.hidden;

    // Host prep.
    let scaled: Vec<f32> = input.pixels.iter().map(|p| 2.0 * (p - 0.5)).collect();
    let mut pos = vec![0.0f32; n * hid];
    let y_table = d.pos_rows * hid;
    for (t, &(x, y)) in input.positions.iter().enumerate() {
        let row = &mut pos[t * hid..(t + 1) * hid];
        let xr = &host.pos_table[x * hid..(x + 1) * hid];
        let yr = &host.pos_table[y_table + y * hid..y_table + (y + 1) * hid];
        for ((o, a), b) in row.iter_mut().zip(xr).zip(yr) {
            *o = a + b;
        }
    }
    let freqs = freq_rows(input.positions, d.head_dim);
    let pixels16 = to_f16(&scaled, "pixel rows")?;
    let pos16 = to_f16(&pos, "position rows")?;

    gpu::autorelease_pool(|| {
        let e = RealForwardError::Gpu;
        let s = Gemma4Scratch::allocate(context, d, n, tile_rows, &freqs);
        let px = context.new_buffer_with_data(&pixels16);
        let pb = context.new_buffer_with_data(&pos16);

        let pass = context.begin_pass();
        gpu::encode_vision_matmul(
            context,
            &pass,
            (&px, 0),
            input_proj,
            None,
            (&s.h, 0),
            n as u32,
            d.patch_dim as u32,
            hid as u32,
        )
        .map_err(e)?;
        gpu::encode_vision_residual_add(context, &pass, (&s.h, 0), (&pb, 0), (n * hid) as u32)
            .map_err(e)?;
        pass.commit_and_wait();
        if let Some(c) = capture.as_deref_mut() {
            c.patch_embed = read_f32(&s.h, n * hid);
        }

        for block in 0..d.depth {
            let pass = context.begin_pass();
            encode_nth_block(context, &pass, block, &s)?;
            pass.commit_and_wait();
            if let Some(c) = capture.as_deref_mut() {
                c.blocks.push(read_f32(&s.h, n * hid));
            }
        }

        // Pool, scale and standardize in F32 on the host.
        let h = read_f32(&s.h, n * hid);
        if let Some(i) = h.iter().position(|v| !v.is_finite()) {
            return Err(bad(format!(
                "gemma4 vision tower output is non-finite at {i}; the FP16 residual stream \
                 overflowed"
            )));
        }
        let root = (hid as f32).sqrt();
        let per_row = input.grid_w / d.pool;
        let mut pooled = vec![0.0f32; soft * hid];
        let inv = 1.0 / per_token as f32;
        for (t, &(x, y)) in input.positions.iter().enumerate() {
            let k = x / d.pool + per_row * (y / d.pool);
            for c in 0..hid {
                pooled[k * hid + c] += h[t * hid + c] * inv;
            }
        }
        for row in pooled.chunks_exact_mut(hid) {
            for (c, v) in row.iter_mut().enumerate() {
                *v = (*v * root - host.std_bias[c]) * host.std_scale[c];
            }
        }
        if let Some(c) = capture {
            c.tower = pooled.clone();
        }
        let tower16 = to_f16(&pooled, "standardized soft tokens")?;

        let tb = context.new_buffer_with_data(&tower16);
        let out = context.new_output_buffer((soft * d.text_hidden * 2) as u64);
        let pass = context.begin_pass();
        gpu::encode_vision_rms_norm(
            context,
            &pass,
            (&tb, 0),
            None,
            (&tb, 0),
            soft as u32,
            hid as u32,
            EPS,
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            &pass,
            (&tb, 0),
            projector,
            None,
            (&out, 0),
            soft as u32,
            hid as u32,
            d.text_hidden as u32,
        )
        .map_err(e)?;
        pass.commit_and_wait();
        Ok(gpu::read_buffer_f16(&out, 0, soft * d.text_hidden)
            .into_iter()
            .map(|v| v.to_bits())
            .collect())
    })
}

/// The Gemma 4 tower opened against an install: blocks streamed from
/// `packed_vision/` into [`VISION_SLOTS`] slots, residents resolved off the
/// trunk's own index. Opened lazily on the first image.
pub(crate) struct Gemma4VisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<Gemma4BlockRoles>,
    dims: Gemma4Dims,
    input_proj: u64,
    projector: u64,
    host: Gemma4HostTables,
}

fn fp16_host(
    weights: &gpu::ResidentGpuWeights,
    local: u64,
    elems: usize,
) -> Result<Vec<f32>, RealForwardError> {
    let at = local as usize;
    let bytes = weights.data().get(at..at + elems * 2).ok_or_else(|| {
        RealForwardError::Unsupported(format!(
            "gemma4 vision tensor at resident offset {at} spans past the mapped region"
        ))
    })?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect())
}

impl Gemma4VisionTower {
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let dims = Gemma4Dims::resolve(&arch.vision)?;
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
        let mut blocks: Vec<Option<Gemma4BlockRoles>> = vec![None; dims.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, dims.depth
                ))
            })?;
            if slot
                .replace(Gemma4BlockRoles::resolve(
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
        .map_err(|e| RealForwardError::Unsupported(format!("gemma4 vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }

        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let h = dims.hidden;
        let input_proj = fp16_view(
            index,
            &name("patch_embedder.input_proj.weight"),
            h * dims.patch_dim,
        )?;
        let projector = fp16_view(
            index,
            &name("embedding_projection.weight"),
            dims.text_hidden * h,
        )?;
        let pos = fp16_view(
            index,
            &name("patch_embedder.position_embedding_table"),
            2 * dims.pos_rows * h,
        )?;
        let bias = fp16_view(index, &name("std_bias"), h)?;
        let scale = fp16_view(index, &name("std_scale"), h)?;
        let host = Gemma4HostTables {
            pos_table: fp16_host(weights, pos, 2 * dims.pos_rows * h)?,
            std_bias: fp16_host(weights, bias, h)?,
            std_scale: fp16_host(weights, scale, h)?,
        };
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            dims,
            input_proj,
            projector,
            host,
        })
    }

    /// Encode one image to `[soft, text_hidden]` FP16 bits.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        patches: &turbospark_vision_io::Gemma4Patches,
    ) -> Result<Vec<u16>, RealForwardError> {
        if patches.pixels.len() != patches.num_patches() * self.dims.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "gemma4 patches are {}-value rows but this tower's patch embedding takes {}",
                patches.pixels.len() / patches.num_patches().max(1),
                self.dims.patch_dim
            )));
        }
        let input = Gemma4Input {
            pixels: &patches.pixels,
            positions: &patches.positions,
            grid_w: patches.grid_w,
        };
        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let dims = self.dims;
        forward(
            context,
            &dims,
            &input,
            &self.host,
            (weights.buffer(), weights.gpu_offset(self.input_proj)),
            (weights.buffer(), weights.gpu_offset(self.projector)),
            GEMMA4_MLP_TILE_ROWS,
            |context, pass, n, s| {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|e| {
                    RealForwardError::Unsupported(format!("gemma4 vision block {n} read: {e}"))
                })?;
                let b = BlockBinding {
                    buffer: &slots[slot],
                    base: 0,
                    roles: &blocks[n],
                };
                encode_block(context, pass, &b, s, &dims)
            },
            None,
        )
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    mod oracle {
        #![allow(dead_code, clippy::approx_constant)]
        include!("../../../compute/tests/generated/gemma4_vision_oracle.rs");
    }

    fn dims() -> Gemma4Dims {
        Gemma4Dims {
            depth: oracle::BLOCKS,
            hidden: oracle::HIDDEN,
            intermediate: oracle::INTERMEDIATE,
            heads: oracle::HEADS,
            head_dim: oracle::HEAD_DIM,
            patch_dim: oracle::PATCH_DIM,
            pos_rows: oracle::POS_TABLE_ROWS,
            pool: oracle::POOL,
            text_hidden: oracle::TEXT_HIDDEN,
        }
    }

    /// The fixture's blocks packed into one FP16 buffer the way the writer
    /// lays a blob out: roles in byte order, each 4-byte aligned.
    fn pack_blocks(d: &Gemma4Dims) -> (Vec<half::f16>, Vec<(u64, Gemma4BlockRoles)>) {
        let fixture_role = |r: usize| -> usize {
            // The fixture's role names, in the same forward order.
            const NAMES: [&str; 13] = [
                "ln_in",
                "q",
                "k",
                "v",
                "q_norm",
                "k_norm",
                "o",
                "ln_post_attn",
                "ln_pre_ffw",
                "gate",
                "up",
                "down",
                "ln_post_ffw",
            ];
            oracle::BLOCK_ROLES
                .iter()
                .position(|n| *n == NAMES[r])
                .expect("fixture role")
        };
        let mut data = Vec::new();
        let mut out = Vec::new();
        for block in oracle::BLOCK_WEIGHTS {
            let base = (data.len() * 2) as u64;
            let mut offsets = [0u64; 13];
            for (r, slot) in offsets.iter_mut().enumerate() {
                if data.len() % 2 != 0 {
                    data.push(half::f16::ZERO);
                }
                *slot = (data.len() * 2) as u64 - base;
                data.extend(
                    block[fixture_role(r)]
                        .iter()
                        .map(|&v| half::f16::from_f32(v)),
                );
            }
            let _ = d;
            out.push((base, Gemma4BlockRoles { offsets }));
        }
        (data, out)
    }

    fn compare(name: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{name} length");
        let (mut dot, mut ng, mut nw, mut worst) = (0f64, 0f64, 0f64, 0f32);
        let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        for (g, w) in got.iter().zip(want) {
            assert!(g.is_finite(), "{name}: non-finite");
            dot += f64::from(*g) * f64::from(*w);
            ng += f64::from(*g).powi(2);
            nw += f64::from(*w).powi(2);
            worst = worst.max((g - w).abs());
        }
        let cos = dot / (ng.sqrt() * nw.sqrt());
        assert!(cos > 0.9999, "{name}: cosine {cos}");
        assert!(
            worst <= tol * scale,
            "{name}: worst {worst} at scale {scale}"
        );
    }

    fn run(mutate: impl Fn(&mut Vec<half::f16>)) -> (Gemma4Stages, Vec<f32>) {
        let d = dims();
        let mut context = gpu::MetalContext::new().expect("Metal device");
        let (mut blob, roles) = pack_blocks(&d);
        mutate(&mut blob);
        let blob = context.new_buffer_with_data(&blob);
        let ip: Vec<half::f16> = oracle::W_INPUT_PROJ
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let pj: Vec<half::f16> = oracle::W_PROJ
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let ip = context.new_buffer_with_data(&ip);
        let pj = context.new_buffer_with_data(&pj);
        let positions: Vec<(usize, usize)> = oracle::POSITIONS
            .chunks_exact(2)
            .map(|p| (p[0], p[1]))
            .collect();
        let host = Gemma4HostTables {
            pos_table: oracle::W_POS_TABLE.to_vec(),
            std_bias: oracle::W_STD_BIAS.to_vec(),
            std_scale: oracle::W_STD_SCALE.to_vec(),
        };
        let input = Gemma4Input {
            pixels: oracle::PIXELS,
            positions: &positions,
            grid_w: oracle::GRID_WH.0,
        };
        let mut stages = Gemma4Stages::default();
        // Tile 4 rows so the 54-row page runs the MLP loop many times.
        let out = forward(
            &mut context,
            &d,
            &input,
            &host,
            (&ip, 0),
            (&pj, 0),
            4,
            |context, pass, n, s| {
                let (base, r) = &roles[n];
                encode_block(
                    context,
                    pass,
                    &BlockBinding {
                        buffer: &blob,
                        base: *base,
                        roles: r,
                    },
                    s,
                    &d,
                )
            },
            Some(&mut stages),
        )
        .expect("forward");
        let out = out
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        (stages, out)
    }

    /// Every stage against mlx-vlm's own run of the same tiny tower. FP16
    /// storage on the GPU against mlx's float32, so the bound is relative to
    /// each stage's magnitude.
    #[test]
    fn the_metal_tower_matches_mlx_vlm_stage_by_stage() {
        let (s, out) = run(|_| {});
        compare("patch_embed", &s.patch_embed, oracle::OUT_PATCH_EMBED, 2e-3);
        compare("block_0", &s.blocks[0], oracle::OUT_BLOCK_0, 5e-3);
        compare("block_1", &s.blocks[1], oracle::OUT_BLOCK_1, 5e-3);
        compare("tower", &s.tower, oracle::OUT_TOWER, 1e-2);
        compare("embedded", &out, oracle::OUT_EMBEDDED, 1e-2);
    }

    /// The blob binding is live: one flipped weight in block 1's v projection
    /// moves the output. A tower reading zeros or the wrong slot would not.
    #[test]
    fn a_block_weight_perturbation_moves_the_output() {
        let (_, base) = run(|_| {});
        let d = dims();
        let (_, roles) = pack_blocks(&d);
        let (b1, r1) = &roles[1];
        let at = ((b1 + r1.at(R::V)) / 2) as usize;
        let (_, moved) = run(|blob| {
            for v in &mut blob[at..at + 8] {
                *v = half::f16::from_f32(v.to_f32() + 1.0);
            }
        });
        let diff = base
            .iter()
            .zip(&moved)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff > 1e-3, "perturbing v moved the output by only {diff}");
    }

    /// The real tower's widths (hidden 1152, 16 heads of 72, intermediate
    /// 4304, a 10240-row table, projector to 2816) on one block and a 6 x 3
    /// grid, Metal against the CPU reference. The fixture above runs at
    /// head_dim 8; this is what exercises the rope's 18-pair quarters and
    /// the attention kernel's register tile at 72.
    #[test]
    fn the_real_widths_match_the_cpu_reference() {
        use compute::vision_gemma4 as cpu;
        let d = Gemma4Dims {
            depth: 1,
            hidden: 1152,
            intermediate: 4304,
            heads: 16,
            head_dim: 72,
            patch_dim: 768,
            pos_rows: 10240,
            pool: 3,
            text_hidden: 2816,
        };
        // Deterministic, small enough that FP16 holds the residual stream.
        let wave = |n: usize, seed: f32, amp: f32, center: f32| -> Vec<f32> {
            (0..n)
                .map(|i| center + amp * ((i as f32) * 0.618 + seed).sin())
                .collect()
        };
        // FP16-round every weight once, so both sides see the same values.
        let r = |v: Vec<f32>| -> Vec<f32> {
            v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect()
        };
        let (h, i) = (d.hidden, d.intermediate);
        let lin =
            |n: usize, k: usize, seed: f32| r(wave(n * k, seed, 1.0 / (k as f32).sqrt(), 0.0));
        let norm = |n: usize, seed: f32| r(wave(n, seed, 0.2, 1.0));
        let roles: Vec<Vec<f32>> = vec![
            norm(h, 1.0),
            lin(h, h, 2.0),
            lin(h, h, 3.0),
            lin(h, h, 4.0),
            // The real checkpoint's q/k norm weights are constant per block
            // at about 1.0 and 0.45 (range-read from blocks 0, 13 and 26),
            // which puts attention logits near +-32 at scale 1.0.
            r(vec![1.0; d.head_dim]),
            r(vec![0.45; d.head_dim]),
            lin(h, h, 7.0),
            norm(h, 8.0),
            norm(h, 9.0),
            lin(i, h, 10.0),
            lin(i, h, 11.0),
            lin(h, i, 12.0),
            norm(h, 13.0),
        ];
        let input_proj = lin(h, d.patch_dim, 14.0);
        let pos_table = r(wave(2 * d.pos_rows * h, 15.0, 0.3, 0.0));
        let std_bias = r(wave(h, 16.0, 0.5, 0.0));
        let std_scale = r(wave(h, 17.0, 0.02, 0.05));
        let proj = lin(d.text_hidden, h, 18.0);

        let (gw, gh) = (6usize, 3usize);
        let positions: Vec<(usize, usize)> = (0..gh)
            .flat_map(|y| (0..gw).map(move |x| (x * 40, y * 55)))
            .collect();
        // Positions spread across the table so far rows are read too; the
        // pool groups by INDEX order, so give it the grid the patches came
        // from rather than the table coordinates.
        let pool_positions: Vec<(usize, usize)> =
            (0..gh).flat_map(|y| (0..gw).map(move |x| (x, y))).collect();
        let pixels: Vec<f32> = wave(positions.len() * d.patch_dim, 19.0, 0.5, 0.5);

        // CPU: the reference takes one position list for both the table and
        // the pool, so run it on the grid coordinates and the GPU likewise.
        let cpu_dims = cpu::Gemma4VisionDims {
            hidden: h,
            intermediate: i,
            heads: d.heads,
            head_dim: d.head_dim,
            patch_dim: d.patch_dim,
            pos_table_rows: d.pos_rows,
            pool: d.pool,
            text_hidden: d.text_hidden,
            rope_theta: ROPE_THETA,
            eps: EPS,
        };
        let block = cpu::Gemma4VisionBlock {
            ln_in: &roles[0],
            q: &roles[1],
            k: &roles[2],
            v: &roles[3],
            q_norm: &roles[4],
            k_norm: &roles[5],
            o: &roles[6],
            ln_post_attn: &roles[7],
            ln_pre_ffw: &roles[8],
            gate: &roles[9],
            up: &roles[10],
            down: &roles[11],
            ln_post_ffw: &roles[12],
        };
        let weights = cpu::Gemma4VisionWeights {
            input_proj: &input_proj,
            pos_table: &pos_table,
            blocks: vec![block],
            std_bias: &std_bias,
            std_scale: &std_scale,
            proj: &proj,
        };
        let pixels16 = r(pixels.iter().map(|p| 2.0 * (p - 0.5)).collect());
        // Undo the scale the reference applies, on FP16-rounded values, so
        // both sides start from the same patch rows.
        let pixels_in: Vec<f32> = pixels16.iter().map(|v| v / 2.0 + 0.5).collect();
        let want = cpu::gemma4_vision_forward(&pixels_in, &pool_positions, gw, &weights, &cpu_dims);
        let _ = positions;

        let mut context = gpu::MetalContext::new().expect("Metal device");
        let mut blob: Vec<half::f16> = Vec::new();
        let mut offsets = [0u64; 13];
        for (slot, w) in offsets.iter_mut().zip(&roles) {
            *slot = (blob.len() * 2) as u64;
            blob.extend(w.iter().map(|&v| half::f16::from_f32(v)));
        }
        let roles_b = Gemma4BlockRoles { offsets };
        let to16 =
            |v: &[f32]| -> Vec<half::f16> { v.iter().map(|&x| half::f16::from_f32(x)).collect() };
        let blob = context.new_buffer_with_data(&blob);
        let ip = context.new_buffer_with_data(&to16(&input_proj));
        let pj = context.new_buffer_with_data(&to16(&proj));
        let host = Gemma4HostTables {
            pos_table: pos_table.clone(),
            std_bias: std_bias.clone(),
            std_scale: std_scale.clone(),
        };
        let input = Gemma4Input {
            pixels: &pixels_in,
            positions: &pool_positions,
            grid_w: gw,
        };
        let mut stages = Gemma4Stages::default();
        let out = forward(
            &mut context,
            &d,
            &input,
            &host,
            (&ip, 0),
            (&pj, 0),
            GEMMA4_MLP_TILE_ROWS,
            |context, pass, _, s| {
                encode_block(
                    context,
                    pass,
                    &BlockBinding {
                        buffer: &blob,
                        base: 0,
                        roles: &roles_b,
                    },
                    s,
                    &d,
                )
            },
            Some(&mut stages),
        )
        .expect("forward");
        let out: Vec<f32> = out
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();
        compare("patch_embed", &stages.patch_embed, &want.patch_embed, 2e-3);
        // Attention at scale 1.0 on RMS-normed q and k peaks near logit 32
        // with these weights, so one-ulp differences in q and k move the
        // softmax by a percent or two. No FP16 pipeline can bit-match
        // another here: a CPU block that rounds to FP16 wherever this one
        // stores (`cpu_block_fp16_storage`) sits as far from F32 as the GPU
        // does, and each kernel matches the CPU to about one ulp on identical
        // inputs. So the claim checked is that the GPU is no worse than an
        // FP16-storage pipeline, not that it bit-matches one.
        let matched = cpu_block_fp16_storage(&stages.patch_embed, &pool_positions, &roles, &d);
        let worst =
            |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let gpu_err = worst(&stages.blocks[0], &want.blocks[0]);
        let fp16_err = worst(&matched, &want.blocks[0]);
        assert!(
            gpu_err <= 1.5 * fp16_err,
            "the GPU block is {gpu_err} from F32 where an FP16-storage CPU block is {fp16_err}"
        );
        compare("block_0", &stages.blocks[0], &want.blocks[0], 1e-2);
        compare("tower", &stages.tower, &want.tower, 2e-2);
        compare("embedded", &out, &want.embedded, 2e-2);
    }

    /// One block on the CPU, rounding to FP16 wherever `encode_block`
    /// stores a buffer: `a`, the q/k/v projections, their norms, the rope,
    /// the attention output, `o`, the gate (before and after GELU), up, the
    /// product, down, its norm, and `h` after each residual add.
    fn cpu_block_fp16_storage(
        h0: &[f32],
        positions: &[(usize, usize)],
        w: &[Vec<f32>],
        d: &Gemma4Dims,
    ) -> Vec<f32> {
        use compute::rms_norm::rms_norm;
        use compute::vision::{bidirectional_attention, gelu_tanh_vision, matmul_bias};
        use compute::vision_gemma4::axial_rope_2d;
        let f = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter()
                .map(|x| half::f16::from_f32(x).to_f32())
                .collect()
        };
        let rows = |x: &[f32], wt: Option<&[f32]>, dim: usize| -> Vec<f32> {
            let ones = vec![1.0f32; dim];
            let wt = wt.unwrap_or(&ones);
            f(x.chunks_exact(dim)
                .flat_map(|r| rms_norm(r, wt, EPS))
                .collect())
        };
        let (n, hid, hd, nh, inter) = (
            positions.len(),
            d.hidden,
            d.head_dim,
            d.heads,
            d.intermediate,
        );
        let mm = |a: &[f32], b: &[f32], m: usize, k: usize, o: usize| {
            f(matmul_bias(a, b, None, m, k, o))
        };
        let h = h0.to_vec();
        let a = rows(&h, Some(&w[0]), hid);
        let mut q = rows(&mm(&a, &w[1], n, hid, hid), Some(&w[4]), hd);
        let mut k = rows(&mm(&a, &w[2], n, hid, hid), Some(&w[5]), hd);
        let v = rows(&mm(&a, &w[3], n, hid, hid), None, hd);
        for (t, &(x, y)) in positions.iter().enumerate() {
            for head in 0..nh {
                let at = (t * nh + head) * hd;
                axial_rope_2d(&mut q[at..at + hd], x, y, ROPE_THETA);
                axial_rope_2d(&mut k[at..at + hd], x, y, ROPE_THETA);
            }
        }
        let (q, k) = (f(q), f(k));
        let attn = f(bidirectional_attention(&q, &k, &v, n, nh, hd, 1.0));
        let o = rows(&mm(&attn, &w[6], n, hid, hid), Some(&w[7]), hid);
        let h: Vec<f32> = f(h.iter().zip(&o).map(|(a, b)| a + b).collect());
        let m = rows(&h, Some(&w[8]), hid);
        let g = f(gelu_tanh_vision(&mm(&m, &w[9], n, hid, inter)));
        let u = mm(&m, &w[10], n, hid, inter);
        let gu = f(g.iter().zip(&u).map(|(a, b)| a * b).collect());
        let dn = rows(&mm(&gu, &w[11], n, inter, hid), Some(&w[12]), hid);
        f(h.iter().zip(&dn).map(|(a, b)| a + b).collect())
    }

    /// The real checkpoint's tower against mlx-vlm, from the dump
    /// `scripts/gemma4_vision_real_dump.py` writes (about 2.2 GB, so not in
    /// the repo). Ignored: it needs `TURBOSPARK_GEMMA4_VISION_DUMP`.
    ///
    /// The bar is the reference's own precision: at every compared stage the
    /// FP16 Metal tower must be at least as close to mlx-vlm's FLOAT32 run as
    /// mlx-vlm's BF16 run is, which is how the reference normally executes.
    #[test]
    #[ignore = "needs TURBOSPARK_GEMMA4_VISION_DUMP from scripts/gemma4_vision_real_dump.py"]
    fn the_real_tower_matches_mlx_vlm() {
        use std::path::PathBuf;
        let dir = PathBuf::from(
            std::env::var("TURBOSPARK_GEMMA4_VISION_DUMP")
                .expect("set TURBOSPARK_GEMMA4_VISION_DUMP"),
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let f32s = |p: PathBuf| -> Vec<f32> {
            std::fs::read(&p)
                .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let f16s = |name: &str| -> Vec<half::f16> {
            std::fs::read(dir.join("weights").join(format!("{name}.f16")))
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        };
        let d = Gemma4Dims {
            depth: 27,
            hidden: 1152,
            intermediate: 4304,
            heads: 16,
            head_dim: 72,
            patch_dim: 768,
            pos_rows: 10240,
            pool: 3,
            text_hidden: manifest["text_hidden"].as_u64().unwrap() as usize,
        };
        // The checkpoint suffix of each blob role, in blob order (the repack
        // writer's `GEMMA4_BLOCK_ROLES`).
        const SUFFIXES: [&str; 13] = [
            "input_layernorm.weight",
            "self_attn.q_proj.linear.weight",
            "self_attn.k_proj.linear.weight",
            "self_attn.v_proj.linear.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.linear.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "mlp.gate_proj.linear.weight",
            "mlp.up_proj.linear.weight",
            "mlp.down_proj.linear.weight",
            "post_feedforward_layernorm.weight",
        ];
        let mut blob: Vec<half::f16> = Vec::new();
        let mut blocks = Vec::new();
        for b in 0..d.depth {
            let base = (blob.len() * 2) as u64;
            let mut offsets = [0u64; 13];
            for (slot, suffix) in offsets.iter_mut().zip(SUFFIXES) {
                *slot = (blob.len() * 2) as u64 - base;
                blob.extend(f16s(&format!("encoder.layers.{b}.{suffix}")));
            }
            blocks.push((base, Gemma4BlockRoles { offsets }));
        }
        let to32 = |v: Vec<half::f16>| -> Vec<f32> { v.into_iter().map(|x| x.to_f32()).collect() };
        let host = Gemma4HostTables {
            pos_table: to32(f16s("patch_embedder.position_embedding_table")),
            std_bias: to32(f16s("std_bias")),
            std_scale: to32(f16s("std_scale")),
        };
        let pixels = f32s(dir.join("pixels.f32"));
        let positions: Vec<(usize, usize)> = std::fs::read(dir.join("positions.u32"))
            .unwrap()
            .chunks_exact(8)
            .map(|b| {
                (
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize,
                    u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize,
                )
            })
            .collect();
        let grid_w = manifest["grid"][0].as_u64().unwrap() as usize;

        let mut context = gpu::MetalContext::new().expect("Metal device");
        let blob = context.new_buffer_with_data(&blob);
        let ip = context.new_buffer_with_data(&f16s("patch_embedder.input_proj.weight"));
        let pj = context.new_buffer_with_data(&f16s("embedding_projection.weight"));
        let input = Gemma4Input {
            pixels: &pixels,
            positions: &positions,
            grid_w,
        };
        let mut stages = Gemma4Stages::default();
        let started = std::time::Instant::now();
        let out = forward(
            &mut context,
            &d,
            &input,
            &host,
            (&ip, 0),
            (&pj, 0),
            GEMMA4_MLP_TILE_ROWS,
            |context, pass, n, s| {
                let (base, roles) = &blocks[n];
                encode_block(
                    context,
                    pass,
                    &BlockBinding {
                        buffer: &blob,
                        base: *base,
                        roles,
                    },
                    s,
                    &d,
                )
            },
            Some(&mut stages),
        )
        .expect("forward");
        eprintln!(
            "metal tower + projector: {:.0} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        let embedded: Vec<f32> = out
            .iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect();

        let cos = |a: &[f32], b: &[f32]| -> f64 {
            assert_eq!(a.len(), b.len());
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                assert!(x.is_finite(), "non-finite Metal value");
                dot += f64::from(*x) * f64::from(*y);
                na += f64::from(*x).powi(2);
                nb += f64::from(*y).powi(2);
            }
            dot / (na.sqrt() * nb.sqrt())
        };
        let mut ours: Vec<(String, Vec<f32>)> =
            vec![("patch_embed".into(), stages.patch_embed.clone())];
        for b in manifest["keep_blocks"].as_array().unwrap() {
            let b = b.as_u64().unwrap() as usize;
            ours.push((format!("block_{b}"), stages.blocks[b].clone()));
        }
        ours.push(("tower".into(), stages.tower.clone()));
        ours.push(("embedded".into(), embedded));
        let mut worse = Vec::new();
        for (name, got) in &ours {
            let truth = f32s(dir.join("f32").join(format!("{name}.f32")));
            let bf16 = f32s(dir.join("bf16").join(format!("{name}.f32")));
            let (metal, refbf) = (cos(got, &truth), cos(&bf16, &truth));
            eprintln!("{name:12} metal vs f32 {metal:.8}   mlx bf16 vs f32 {refbf:.8}   metal vs mlx bf16 {:.8}", cos(got, &bf16));
            if metal < refbf {
                worse.push(format!("{name}: {metal:.8} < {refbf:.8}"));
            }
        }
        assert!(
            worse.is_empty(),
            "the Metal tower is further from F32 than mlx BF16 at: {worse:?}"
        );
    }

    #[test]
    fn freq_rows_put_x_angles_first() {
        let rows = freq_rows(&[(3, 0)], 8);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0], 3.0);
        assert!(rows[1] > 0.0 && rows[2] == 0.0 && rows[3] == 0.0);
    }
}
