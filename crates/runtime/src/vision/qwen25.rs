//! The Qwen2.5-VL vision tower on Metal (`docs/QWEN25_VL_VISION_PHASE0.md`).
//!
//! A sibling of the qwen3 tower in this module rather than a mode of it: the
//! block norms are RMSNorm, the MLP is gated SiLU, there is no absolute
//! position table, and 28 of the 32 blocks attend inside windows of the page
//! instead of over all of it. What it shares is the destination format
//! (blocks streamed two slots at a time from `packed_vision/`, residents at
//! FP16 under `vision.`), the kernels (matmul, bidirectional attention, the 2-D
//! rope, GELU, residual add, RMSNorm, mul), and the result type: it returns the
//! same [`VisionEmbedding`] the qwen3 tower does, so the prompt builders and
//! `set_prompt_vision` need no second path.
//!
//! # Window order is a HOST permutation
//!
//! The reference reorders the page's merged tokens (each a 2x2 patch group)
//! into windows of 4x4 merged tokens, runs every block on that order, applies
//! the merger, and puts the merged rows back with `argsort(window_index)`. The
//! window blocks attend inside each window (a segment), the full blocks (7,
//! 15, 23, 31) over the whole page. Here the patch rows and the rope rows are
//! permuted before upload and the merged output rows are put back on readback,
//! so no kernel learns about windows: a window block is the bidirectional
//! attention kernel dispatched once per segment at that segment's row offset.
//!
//! # The mutation that makes a fluent wrong model
//!
//! Attention over the whole page on every block is the same tensor shapes and
//! a plausible image. [`Qwen25VisionTower::run`]'s window arm and the unit test
//! `window_layout_matches_the_reference_order` are what pin it.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

use super::weights::{fp16_view, VISION_PREFIX};
use super::{VisionEmbedding, VisionStages, VISION_SLOTS};
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// RMSNorm epsilon: both block norms and the merger's `ln_q` construct
/// `nn.RMSNorm(dim, eps=1e-6)`.
const EPS: f32 = 1e-6;

/// MLP row tile: the `[rows, intermediate]` gate and up buffers are sized for
/// a tile, not the page (the qwen3 tower's Part B1 shape).
pub(crate) const QWEN25_MLP_TILE_ROWS: usize = 1024;

/// The attention window in pixels. A property of the tower kind, checked
/// against the checkpoint at intake (`repack::qwen36_config`).
const WINDOW_PIXELS: usize = 112;

/// The blob roles in byte order, the repack writer's `QWEN25_BLOCK_ROLES`
/// first column. Restated rather than imported because repack is a
/// dev-dependency of this crate.
pub(crate) const QWEN25_BLOCK_ROLES: [&str; 12] = [
    "n1_w", "qkv_w", "qkv_b", "proj_w", "proj_b", "n2_w", "gate_w", "gate_b", "up_w", "up_b",
    "down_w", "down_b",
];

#[derive(Debug, Clone, Copy)]
enum R {
    N1W = 0,
    QkvW = 1,
    QkvB = 2,
    ProjW = 3,
    ProjB = 4,
    N2W = 5,
    GateW = 6,
    GateB = 7,
    UpW = 8,
    UpB = 9,
    DownW = 10,
    DownB = 11,
}

const ALL_ROLES: [R; 12] = [
    R::N1W,
    R::QkvW,
    R::QkvB,
    R::ProjW,
    R::ProjB,
    R::N2W,
    R::GateW,
    R::GateB,
    R::UpW,
    R::UpB,
    R::DownW,
    R::DownB,
];

/// The tower's geometry, resolved once from the install's [`VisionConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen25Dims {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_dim: usize,
    pub(crate) patch_size: usize,
    pub(crate) merge: usize,
    pub(crate) out_hidden: usize,
}

impl Qwen25Dims {
    pub(crate) fn resolve(v: &VisionConfig) -> Result<Self, RealForwardError> {
        if v.tower != VisionTowerKind::Qwen25Vl || !v.is_active() {
            return Err(RealForwardError::Unsupported(format!(
                "not an active qwen2.5-vl vision tower (kind {})",
                v.tower.as_str()
            )));
        }
        let us = |x: i64, what: &str| {
            usize::try_from(x)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| RealForwardError::Unsupported(format!("qwen25 vision {what} {x}")))
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
            patch_size: us(v.patch_size, "patch_size")?,
            merge: us(v.spatial_merge_size, "spatial_merge_size")?,
            out_hidden: us(v.out_hidden_size, "out_hidden_size")?,
        };
        if d.heads * d.head_dim != d.hidden || d.head_dim % 4 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "qwen25 vision hidden {} over {} heads does not give a head_dim divisible by 4",
                d.hidden, d.heads
            )));
        }
        if d.head_dim as u32 > gpu::MAX_ATTENTION_HEAD_DIM {
            return Err(RealForwardError::Unsupported(format!(
                "qwen25 vision head_dim {} is past the attention kernel's {}",
                d.head_dim,
                gpu::MAX_ATTENTION_HEAD_DIM
            )));
        }
        if WINDOW_PIXELS % (d.patch_size * d.merge) != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "qwen25 vision window of {WINDOW_PIXELS} px is not a whole number of {}x{} merged \
                 tokens",
                d.merge, d.merge
            )));
        }
        Ok(d)
    }

    /// Merged tokens per window edge: 112 / 2 / 14 = 4.
    fn window_units(&self) -> usize {
        WINDOW_PIXELS / self.merge / self.patch_size
    }

    fn merged_width(&self) -> usize {
        self.merge * self.merge * self.hidden
    }

    /// A block is full-attention on every eighth block counting from the
    /// eighth (7, 15, 23, 31), which the repack parser verified.
    fn is_full_attention(block: usize) -> bool {
        block % 8 == 7
    }

    fn role_elems(&self, role: R) -> usize {
        let (h, i) = (self.hidden, self.intermediate);
        match role {
            R::N1W | R::N2W | R::ProjB | R::DownB => h,
            R::QkvW => 3 * h * h,
            R::QkvB => 3 * h,
            R::ProjW => h * h,
            R::GateW | R::UpW => i * h,
            R::GateB | R::UpB => i,
            R::DownW => h * i,
        }
    }
}

/// One block's twelve byte offsets within its blob.
#[derive(Debug, Clone)]
pub(crate) struct Qwen25BlockRoles {
    offsets: [u64; 12],
}

impl Qwen25BlockRoles {
    fn at(&self, role: R) -> u64 {
        self.offsets[role as usize]
    }

    /// Resolve and size-check one block's layout entry.
    pub(crate) fn resolve(
        entry: &model_io::ExpertEntry,
        stride: u64,
        d: &Qwen25Dims,
    ) -> Result<Self, RealForwardError> {
        let block = entry.expert;
        let mut offsets = [0u64; 12];
        for role in ALL_ROLES {
            let name = QWEN25_BLOCK_ROLES[role as usize];
            let sub = entry.sub_tensors.get(name).ok_or_else(|| {
                RealForwardError::MissingTensor(format!("qwen25 vision block {block} {name}"))
            })?;
            let want = (d.role_elems(role) * 2) as u64;
            if sub.dtype != "fp16" || sub.size != want {
                return Err(RealForwardError::Unsupported(format!(
                    "qwen25 vision block {block} {name}: {} bytes of {}, expected {want} of fp16",
                    sub.size, sub.dtype
                )));
            }
            if sub.offset % 4 != 0 || sub.offset + sub.size > stride {
                return Err(RealForwardError::Unsupported(format!(
                    "qwen25 vision block {block} {name} at {}..{} does not fit a 4-byte aligned \
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
    pub(crate) roles: &'a Qwen25BlockRoles,
}

impl BlockBinding<'_> {
    fn w(&self, role: R) -> (&gpu::MetalBuffer, u64) {
        (self.buffer, self.base + self.roles.at(role))
    }
}

/// Per-image GPU scratch.
pub(crate) struct Qwen25Scratch {
    seq: usize,
    tile: usize,
    /// Patch rows (FP16), window order.
    rows: gpu::MetalBuffer,
    /// The residual stream.
    x: gpu::MetalBuffer,
    normed: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    k: gpu::MetalBuffer,
    v: gpu::MetalBuffer,
    attn: gpu::MetalBuffer,
    proj: gpu::MetalBuffer,
    gate: gpu::MetalBuffer,
    up: gpu::MetalBuffer,
    /// Merger hidden layer, `[merged, 4 * hidden]`.
    merger_hidden: gpu::MetalBuffer,
    out: gpu::MetalBuffer,
    freqs: gpu::MetalBuffer,
    /// Rows of each attention segment of a WINDOW block, in order.
    segments: Vec<usize>,
}

impl Qwen25Scratch {
    fn allocate(
        context: &gpu::MetalContext,
        d: &Qwen25Dims,
        seq: usize,
        tile: usize,
        rows: &[half::f16],
        freqs: &[f32],
        segments: Vec<usize>,
    ) -> Self {
        let merged = seq / (d.merge * d.merge);
        let wide = (seq * d.hidden * 2) as u64;
        let tile = tile.min(seq).max(1);
        let mlp = (tile * d.intermediate * 2) as u64;
        Self {
            seq,
            tile,
            rows: context.new_buffer_with_data(rows),
            x: context.new_output_buffer(wide),
            normed: context.new_output_buffer(wide),
            q: context.new_output_buffer(wide),
            k: context.new_output_buffer(wide),
            v: context.new_output_buffer(wide),
            attn: context.new_output_buffer(wide),
            proj: context.new_output_buffer(wide),
            gate: context.new_output_buffer(mlp),
            up: context.new_output_buffer(mlp),
            merger_hidden: context.new_output_buffer((merged * d.merged_width() * 2) as u64),
            out: context.new_output_buffer((merged * d.out_hidden * 2) as u64),
            freqs: context.new_buffer_with_data(freqs),
            segments,
        }
    }
}

/// The page's merged tokens in window order, and the patch-row length of each
/// non-empty attention window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowLayout {
    /// `unit_order[j]` is the original merged-token index that sits at window
    /// position `j`.
    pub(crate) unit_order: Vec<usize>,
    /// Patch rows per non-empty window, in window order.
    pub(crate) segments: Vec<usize>,
}

/// The reference's `get_window_index` for one image: windows of
/// `window_units x window_units` merged tokens, row-major inside a window,
/// row-major over windows, one frame at a time.
///
/// The reference pads each edge by `window - edge % window`, which adds a
/// whole EMPTY window row when the edge divides evenly; it then drops
/// zero-length segments, so the padding changes nothing and this does not
/// build it.
pub(crate) fn window_layout(
    grid: GridThw,
    merge: usize,
    window_units: usize,
) -> Result<WindowLayout, RealForwardError> {
    if merge == 0
        || window_units == 0
        || grid.h == 0
        || grid.w == 0
        || grid.h % merge != 0
        || grid.w % merge != 0
    {
        return Err(RealForwardError::Unsupported(format!(
            "qwen25 vision grid {}x{}x{} is not a whole number of {merge}x{merge} merge groups",
            grid.t, grid.h, grid.w
        )));
    }
    let (rows, cols) = (grid.h / merge, grid.w / merge);
    let per_patch = merge * merge;
    let mut unit_order = Vec::with_capacity(grid.t * rows * cols);
    let mut segments = Vec::new();
    for t in 0..grid.t {
        for wy in 0..rows.div_ceil(window_units) {
            for wx in 0..cols.div_ceil(window_units) {
                let before = unit_order.len();
                for y in wy * window_units..((wy + 1) * window_units).min(rows) {
                    for x in wx * window_units..((wx + 1) * window_units).min(cols) {
                        unit_order.push((t * rows + y) * cols + x);
                    }
                }
                let count = unit_order.len() - before;
                if count > 0 {
                    segments.push(count * per_patch);
                }
            }
        }
    }
    Ok(WindowLayout {
        unit_order,
        segments,
    })
}

/// Rows of `width` values regrouped by merge unit: output unit `j` is input
/// unit `order[j]`, each unit being `per_unit` consecutive rows.
pub(crate) fn permute_units<T: Copy>(
    rows: &[T],
    width: usize,
    per_unit: usize,
    order: &[usize],
) -> Vec<T> {
    let unit = width * per_unit;
    let mut out = Vec::with_capacity(rows.len());
    for &u in order {
        out.extend_from_slice(&rows[u * unit..(u + 1) * unit]);
    }
    out
}

/// Encode one block over `s.x` in place.
pub(crate) fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    b: &BlockBinding<'_>,
    s: &Qwen25Scratch,
    d: &Qwen25Dims,
    block: usize,
) -> Result<(), RealForwardError> {
    let e = RealForwardError::Gpu;
    let seq = s.seq as u32;
    let hidden = d.hidden as u32;
    let wide = (s.seq * d.hidden) as u32;
    let scale = (d.head_dim as f32).powf(-0.5);

    // --- attention half ---
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.x, 0),
        Some(b.w(R::N1W)),
        (&s.normed, 0),
        seq,
        hidden,
        EPS,
    )
    .map_err(e)?;

    // q, k, v are the fused weight's three row blocks and the fused bias's
    // three thirds, read at byte offsets rather than split by a kernel.
    let w_block = (d.hidden * d.hidden * 2) as u64;
    let b_block = (d.hidden * 2) as u64;
    let (qkv_w, qkv_w_off) = b.w(R::QkvW);
    let (qkv_b, qkv_b_off) = b.w(R::QkvB);
    for (i, out) in [&s.q, &s.k, &s.v].into_iter().enumerate() {
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, 0),
            (qkv_w, qkv_w_off + i as u64 * w_block),
            Some((qkv_b, qkv_b_off + i as u64 * b_block)),
            (out, 0),
            seq,
            hidden,
            hidden,
        )
        .map_err(e)?;
    }

    // Q AND K ONLY: v carries no positional rotation.
    for target in [&s.q, &s.k] {
        gpu::encode_vision_rope_2d(
            context,
            pass,
            (target, 0),
            (&s.freqs, 0),
            seq,
            d.heads as u32,
            d.head_dim as u32,
        )
        .map_err(e)?;
    }

    // Full-attention blocks see the whole page; every other block attends
    // inside each window, which is the same kernel at each segment's offset.
    if Qwen25Dims::is_full_attention(block) {
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
    } else {
        let mut start = 0usize;
        for &len in &s.segments {
            let off = (start * d.hidden * 2) as u64;
            gpu::encode_vision_attention(
                context,
                pass,
                (&s.q, off),
                (&s.k, off),
                (&s.v, off),
                (&s.attn, off),
                len as u32,
                d.heads as u32,
                d.head_dim as u32,
                scale,
            )
            .map_err(e)?;
            start += len;
        }
        debug_assert_eq!(start, s.seq);
    }

    gpu::encode_vision_matmul(
        context,
        pass,
        (&s.attn, 0),
        b.w(R::ProjW),
        Some(b.w(R::ProjB)),
        (&s.proj, 0),
        seq,
        hidden,
        hidden,
    )
    .map_err(e)?;
    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide).map_err(e)?;

    // --- MLP half: down(silu(gate(x)) * up(x)), row-tiled ---
    gpu::encode_vision_rms_norm(
        context,
        pass,
        (&s.x, 0),
        Some(b.w(R::N2W)),
        (&s.normed, 0),
        seq,
        hidden,
        EPS,
    )
    .map_err(e)?;
    let inter = d.intermediate as u32;
    let mut t0 = 0usize;
    while t0 < s.seq {
        let rows = s.tile.min(s.seq - t0);
        let row_off = (t0 * d.hidden * 2) as u64;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, row_off),
            b.w(R::GateW),
            Some(b.w(R::GateB)),
            (&s.gate, 0),
            rows as u32,
            hidden,
            inter,
        )
        .map_err(e)?;
        gpu::encode_vision_silu(context, pass, (&s.gate, 0), (rows * d.intermediate) as u32)
            .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, row_off),
            b.w(R::UpW),
            Some(b.w(R::UpB)),
            (&s.up, 0),
            rows as u32,
            hidden,
            inter,
        )
        .map_err(e)?;
        gpu::encode_vision_mul(
            context,
            pass,
            (&s.gate, 0),
            (&s.up, 0),
            (rows * d.intermediate) as u32,
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.gate, 0),
            b.w(R::DownW),
            Some(b.w(R::DownB)),
            (&s.proj, row_off),
            rows as u32,
            inter,
            hidden,
        )
        .map_err(e)?;
        t0 += rows;
    }
    gpu::encode_vision_residual_add(context, pass, (&s.x, 0), (&s.proj, 0), wide).map_err(e)
}

/// The resident tensors a tower reads outside its blocks.
struct Qwen25Resident {
    patch_embed: u64,
    ln_q: u64,
    fc1_w: u64,
    fc1_b: u64,
    fc2_w: u64,
    fc2_b: u64,
}

impl Qwen25Resident {
    fn resolve(index: &ResidentIndex, d: &Qwen25Dims) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let mw = d.merged_width();
        Ok(Self {
            patch_embed: fp16_view(
                index,
                &name("patch_embed.proj.weight"),
                d.hidden * d.patch_dim,
            )?,
            ln_q: fp16_view(index, &name("merger.ln_q.weight"), d.hidden)?,
            fc1_w: fp16_view(index, &name("merger.mlp.0.weight"), mw * mw)?,
            fc1_b: fp16_view(index, &name("merger.mlp.0.bias"), mw)?,
            fc2_w: fp16_view(index, &name("merger.mlp.2.weight"), d.out_hidden * mw)?,
            fc2_b: fp16_view(index, &name("merger.mlp.2.bias"), d.out_hidden)?,
        })
    }
}

/// The Qwen2.5-VL tower opened against an install: blocks streamed from
/// `packed_vision/` into [`VISION_SLOTS`] slots, residents resolved off the
/// trunk's (or an attached sidecar's) index. Opened lazily on the first image.
pub(crate) struct Qwen25VisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<Qwen25BlockRoles>,
    dims: Qwen25Dims,
    resident: Qwen25Resident,
    /// A sidecar's own weights, `Some` only when opened against one.
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
    pub(crate) last_scratch_bytes: u64,
}

impl Qwen25VisionTower {
    /// Open against an install that carries the tower itself.
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let dims = Qwen25Dims::resolve(&arch.vision)?;
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
        let mut blocks: Vec<Option<Qwen25BlockRoles>> = vec![None; dims.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, dims.depth
                ))
            })?;
            if slot
                .replace(Qwen25BlockRoles::resolve(
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
        .map_err(|e| RealForwardError::Unsupported(format!("qwen25 vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }
        let resident = Qwen25Resident::resolve(index, &dims)?;
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            dims,
            resident,
            sidecar_weights: None,
            last_scratch_bytes: 0,
        })
    }

    /// Open against a standalone sidecar directory: its own resident index
    /// and its own mapping, kept alive for the tower's life.
    pub(crate) fn open_with_sidecar(
        sidecar_dir: &Path,
        context: &gpu::MetalContext,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let index = model_io::load_resident_index(&sidecar_dir.join("model_weights.bin"))
            .map_err(RealForwardError::Model)?;
        if let Some(entry) = index
            .entries
            .values()
            .find(|e| !crate::real_forward_layout::readable_resident_dtype(&e.name, e.dtype))
        {
            return Err(RealForwardError::Unsupported(format!(
                "vision sidecar tensor {} carries resident dtype {}, which no reader in this \
                 crate honours",
                entry.name, entry.dtype
            )));
        }
        let buffer = model_io::ResidentBuffer::map(
            &sidecar_dir.join("model_weights.bin"),
            index.header.index_size,
            index.header.resident_size,
        )
        .map_err(RealForwardError::Model)?;
        let weights = gpu::ResidentGpuWeights::wrap(context.device(), buffer)
            .map_err(RealForwardError::Gpu)?;
        let mut tower = Self::open(sidecar_dir, context, &index, arch)?;
        tower.sidecar_weights = Some(weights);
        Ok(tower)
    }

    pub(crate) fn is_sidecar(&self) -> bool {
        self.sidecar_weights.is_some()
    }

    /// Encode one image to the `[merged, out_hidden]` rows the trunk takes, in
    /// the page's original merge order.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &PreprocessedImage,
        params: &PreprocessParams,
        mut capture: Option<VisionStages>,
    ) -> Result<(VisionEmbedding, Option<VisionStages>), RealForwardError> {
        let d = self.dims;
        if params.patch_dim() != d.patch_dim
            || params.merge_size != d.merge
            || params.patch_size != d.patch_size
        {
            return Err(RealForwardError::Unsupported(format!(
                "preprocessing produced patch {} / merge {} / {}-value rows but this tower takes \
                 patch {} / merge {} / {}",
                params.patch_size,
                params.merge_size,
                params.patch_dim(),
                d.patch_size,
                d.merge,
                d.patch_dim
            )));
        }
        let seq = image.grid.patches();
        if seq == 0 || image.patch_rows.len() != seq * d.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "qwen25 vision input: {} values for {seq} patches of {}",
                image.patch_rows.len(),
                d.patch_dim
            )));
        }
        let layout = window_layout(image.grid, d.merge, d.window_units())?;
        let per_unit = d.merge * d.merge;

        // Host prep, all in window order: the patch rows, and the rope rows
        // (kept F32 to the GPU, the angle at pair 0 is the raw coordinate).
        let rows16: Vec<half::f16> =
            permute_units(&image.patch_rows, d.patch_dim, per_unit, &layout.unit_order)
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
        let freqs_orig = turbospark_vision_io::rope::vision_rope_freq_rows_default(
            image.grid, d.head_dim, params,
        )
        .map_err(|e| RealForwardError::Unsupported(format!("vision rope rows: {e}")))?;
        let freqs = permute_units(&freqs_orig, d.head_dim / 2, per_unit, &layout.unit_order);

        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let merged = seq / per_unit;
        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let resident = &self.resident;
        let last_scratch = &mut self.last_scratch_bytes;
        let order = &layout.unit_order;

        gpu::autorelease_pool(|| {
            let e = RealForwardError::Gpu;
            let s = Qwen25Scratch::allocate(
                context,
                &d,
                seq,
                QWEN25_MLP_TILE_ROWS,
                &rows16,
                &freqs,
                layout.segments.clone(),
            );
            *last_scratch = (seq * d.hidden * 2 * 7
                + 2 * s.tile * d.intermediate * 2
                + merged * d.merged_width() * 2
                + merged * d.out_hidden * 2) as u64;
            let wide = seq * d.hidden;

            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.rows, 0),
                (w.buffer(), w.gpu_offset(resident.patch_embed)),
                None,
                (&s.x, 0),
                seq as u32,
                d.patch_dim as u32,
                d.hidden as u32,
            )
            .map_err(e)?;
            pass.commit_and_wait();
            if let Some(c) = capture.as_mut() {
                c.patch_embed = read_bits(&s.x, wide);
            }

            for (n, block_roles) in blocks.iter().enumerate() {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|err| {
                    RealForwardError::Unsupported(format!("qwen25 vision block {n} read: {err}"))
                })?;
                let binding = BlockBinding {
                    buffer: &slots[slot],
                    base: 0,
                    roles: block_roles,
                };
                let pass = context.begin_pass();
                encode_block(context, &pass, &binding, &s, &d, n)?;
                pass.commit_and_wait();
                if let Some(c) = capture.as_mut() {
                    if n == 0 {
                        c.block_first = read_bits(&s.x, wide);
                    }
                    if n + 1 == d.depth {
                        c.block_last = read_bits(&s.x, wide);
                    }
                }
            }

            // The merger: RMSNorm over each patch row, then the four patches
            // of a merge group read as one 4 * hidden row (a view, the rows
            // are already grouped), fc1, erf GELU, fc2.
            let mw = d.merged_width() as u32;
            let pass = context.begin_pass();
            gpu::encode_vision_rms_norm(
                context,
                &pass,
                (&s.x, 0),
                Some((w.buffer(), w.gpu_offset(resident.ln_q))),
                (&s.normed, 0),
                seq as u32,
                d.hidden as u32,
                EPS,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.normed, 0),
                (w.buffer(), w.gpu_offset(resident.fc1_w)),
                Some((w.buffer(), w.gpu_offset(resident.fc1_b))),
                (&s.merger_hidden, 0),
                merged as u32,
                mw,
                mw,
            )
            .map_err(e)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&s.merger_hidden, 0),
                (merged * d.merged_width()) as u32,
                gpu::GeluKind::Erf,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.merger_hidden, 0),
                (w.buffer(), w.gpu_offset(resident.fc2_w)),
                Some((w.buffer(), w.gpu_offset(resident.fc2_b))),
                (&s.out, 0),
                merged as u32,
                mw,
                d.out_hidden as u32,
            )
            .map_err(e)?;
            pass.commit_and_wait();

            // Put the merged rows back on the page's own order.
            let in_window_order = read_bits(&s.out, merged * d.out_hidden);
            let mut rows = vec![0u16; in_window_order.len()];
            for (j, &u) in order.iter().enumerate() {
                rows[u * d.out_hidden..(u + 1) * d.out_hidden]
                    .copy_from_slice(&in_window_order[j * d.out_hidden..(j + 1) * d.out_hidden]);
            }
            Ok((
                VisionEmbedding {
                    rows,
                    merged_tokens: merged,
                    out_hidden: d.out_hidden,
                    grid: image.grid,
                    deepstack: Vec::new(),
                },
                capture,
            ))
        })
    }
}

fn read_bits(buffer: &gpu::MetalBuffer, count: usize) -> Vec<u16> {
    gpu::read_buffer_f16(buffer, 0, count)
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// The reference's `get_window_index` on a grid whose edges neither fill
    /// nor align with the window: a 10x14 merged grid (patch grid 20x28) has
    /// windows of 4x4 with ragged right and bottom edges.
    #[test]
    fn window_layout_matches_the_reference_order() {
        let layout = window_layout(GridThw::new(1, 20, 28), 2, 4).unwrap();
        // Every merged token exactly once.
        let mut seen = layout.unit_order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..10 * 14).collect::<Vec<_>>());
        // Windows: 3 rows x 4 columns, all non-empty. Edge windows are 2 or 4
        // wide and 2 or 4 tall.
        assert_eq!(layout.segments.len(), 12);
        assert_eq!(layout.segments.iter().sum::<usize>(), 20 * 28);
        assert_eq!(layout.segments[0], 4 * 4 * 4);
        assert_eq!(layout.segments[3], 4 * 2 * 4);
        assert_eq!(layout.segments[8], 2 * 4 * 4);
        assert_eq!(layout.segments[11], 2 * 2 * 4);
        // The first window is the top-left 4x4 of merged tokens, row-major.
        assert_eq!(
            &layout.unit_order[..8],
            &[0, 1, 2, 3, 14, 15, 16, 17],
            "the first window reads its first two rows"
        );
        // The second window starts at merged column 4.
        assert_eq!(layout.unit_order[16], 4);
    }

    #[test]
    fn a_grid_that_fills_the_window_adds_no_empty_segment() {
        // 8x8 merged tokens: exactly 2x2 windows of 4x4, no ragged edge.
        let layout = window_layout(GridThw::new(1, 16, 16), 2, 4).unwrap();
        assert_eq!(layout.segments, vec![64, 64, 64, 64]);
    }

    #[test]
    fn a_grid_smaller_than_one_window_is_one_segment_in_page_order() {
        let layout = window_layout(GridThw::new(1, 6, 8), 2, 4).unwrap();
        assert_eq!(layout.segments, vec![6 * 8]);
        assert_eq!(layout.unit_order, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn permute_units_moves_whole_merge_groups() {
        // 3 units of 2 rows of width 2.
        let rows: Vec<u16> = (0..12).collect();
        assert_eq!(
            permute_units(&rows, 2, 2, &[2, 0, 1]),
            vec![8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn a_misaligned_grid_is_refused() {
        assert!(window_layout(GridThw::new(1, 5, 8), 2, 4).is_err());
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

    /// The tower on the REAL checkpoint against `scripts/qwen25_vl_tower_reference.py`.
    ///
    /// That script is a numpy transcription of mlx-vlm's tower written from
    /// its source, not mlx-vlm executed, so it can share a misreading of the
    /// reference with this code; what it does not share is any code. Patch
    /// rows come from `crates/vision-io`, so this holds the TOWER and the
    /// preprocessing together, and the dumped stage names localize a gap the
    /// way `vision_tower_parity.rs` does: patch embed, the first block, the
    /// last block (all in window order), then the merger in page order.
    ///
    /// ```sh
    /// turbospark-model pull-vision --repo mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
    ///   --alias qwen25vl-3b-vision --out /Volumes/T7/turbospark/vision/qwen25vl-3b.gturbo-vision
    /// TURBOSPARK_QWEN25VL_SIDECAR_DIR=/Volumes/T7/turbospark/vision/qwen25vl-3b.gturbo-vision \
    /// TURBOSPARK_QWEN25VL_MODEL_DIR=/Volumes/T7/hf-models/Qwen2.5-VL-3B-Instruct-4bit \
    /// TURBOSPARK_QWEN25VL_IMAGE=/Volumes/T7/vision-pages/page-small.png \
    /// TURBOSPARK_QWEN25VL_SCRATCH=/Volumes/T7/vision-dumps/qwen25vl-3b-small \
    ///   cargo test -p turbospark-runtime --release --lib \
    ///   the_real_tower_matches_the_numpy_reference -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Qwen2.5-VL tower sidecar, the HF checkpoint dir, an image and python3 with numpy"]
    fn the_real_tower_matches_the_numpy_reference() {
        let sidecar = env_path("TURBOSPARK_QWEN25VL_SIDECAR_DIR");
        let model_dir = env_path("TURBOSPARK_QWEN25VL_MODEL_DIR");
        let image = env_path("TURBOSPARK_QWEN25VL_IMAGE");
        let scratch = env_path("TURBOSPARK_QWEN25VL_SCRATCH");
        std::fs::create_dir_all(&scratch).unwrap();

        let (_record, vision) = model_io::load_vision_sidecar(&sidecar).expect("sidecar");
        let arch = model_io::sidecar_arch(
            model_io::ModelFamily::Qwen2Dense,
            vision.out_hidden_size,
            &vision,
        );
        let params = PreprocessParams::from_preprocessor_config_json(
            &std::fs::read_to_string(sidecar.join("preprocessor_config.json")).unwrap(),
        )
        .expect("preprocessor_config.json");
        let rgb = turbospark_vision_io::decode::decode_image_file(&image).expect("decode");
        let pre = turbospark_vision_io::preprocess(&rgb, &params).expect("preprocess");
        let g = pre.grid;
        println!(
            "grid {}x{}x{} ({} patches, {} merged), resized {:?}",
            g.t,
            g.h,
            g.w,
            g.patches(),
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
                "/../../scripts/qwen25_vl_tower_reference.py"
            ))
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("--rows")
            .arg(&rows_path)
            .args([
                "--grid",
                &g.t.to_string(),
                &g.h.to_string(),
                &g.w.to_string(),
            ])
            .arg("--out-dir")
            .arg(&reference)
            .status()
            .expect("python3");
        assert!(status.success(), "the numpy reference failed");

        let mut context = gpu::MetalContext::new().expect("Metal device");
        let mut tower =
            Qwen25VisionTower::open_with_sidecar(&sidecar, &context, &arch).expect("open tower");
        let dummy = {
            // The sidecar tower binds its own residents; the trunk-weights
            // argument is only the fallback when no sidecar is attached.
            let index = model_io::load_resident_index(&sidecar.join("model_weights.bin")).unwrap();
            let buffer = model_io::ResidentBuffer::map(
                &sidecar.join("model_weights.bin"),
                index.header.index_size,
                index.header.resident_size,
            )
            .unwrap();
            gpu::ResidentGpuWeights::wrap(context.device(), buffer).unwrap()
        };
        let (embedding, stages) = tower
            .run(
                &mut context,
                &dummy,
                &pre,
                &params,
                Some(VisionStages::default()),
            )
            .expect("tower run");
        let stages = stages.expect("captured");
        let to_f32 = |bits: &[u16]| -> Vec<f32> {
            bits.iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect()
        };
        let last = vision.depth - 1;
        let pairs = [
            ("patch_embed", to_f32(&stages.patch_embed)),
            ("block_0", to_f32(&stages.block_first)),
            (&format!("block_{last}") as &str, to_f32(&stages.block_last)),
            ("merger", to_f32(&embedding.rows)),
        ];
        let mut worst = 1.0f64;
        for (name, got) in &pairs {
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
}
