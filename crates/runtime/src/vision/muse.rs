//! The Muse-Glimmer vision tower on Metal (`docs/MUSEGLIMMER_VISION_PHASE0.md`).
//!
//! A sibling of the Qwen2.5-VL tower (`qwen25.rs`) in this module and built the
//! same way: blocks streamed two slots at a time from `packed_vision/`, FP16
//! residents under `vision.`, the shared kernels, and the same
//! [`VisionEmbedding`] result, so `set_prompt_vision` and every front end are
//! unchanged. What differs from every other tower here:
//!
//! - **Tokens are in RASTER order**, not merge-window order, and a row is
//!   `(T, C, P, P)`. Windows are 32 PATCHES square (`pos_emb_height`), not 4
//!   merged tokens, and 38 of 50 blocks attend inside them.
//! - **The position table is added before everything else**, interpolated with
//!   HALF-PIXEL centres `(i + 0.5) * (32 / n) - 0.5` and per-corner validity
//!   masks, not the align-corners rule of the Qwen3 tower. It is a host step.
//! - **`ln_pre` follows the position add and `ln_post` follows the last
//!   block.** LayerNorm with bias, eps 1e-5.
//! - **The merge is a DIM-MAJOR pixel shuffle**: a merged row is
//!   `[d0 of its 4 patches, d1 of its 4 patches, ...]` where Qwen's mergers
//!   concatenate whole patch rows. Done on the host after `ln_post`.
//! - **The tail** is `gelu(fc2(gelu(fc1(x))))`, a projection into the trunk's
//!   width, and a NO-SCALE RMSNorm, all bias-free.
//!
//! Window order is a host permutation exactly as in `qwen25.rs`; this module
//! calls the same [`window_layout`] with a merge of 1 (units are patches).

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

use super::qwen25::{permute_units, window_layout};
use super::shape::VisionShape;
use super::weights::{fp16_view, resolve_block_roles, BlockRoles, Role, VISION_PREFIX};
use super::{VisionEmbedding, VisionStages, VISION_SLOTS};
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// LayerNorm epsilon of the blocks and of `ln_pre`/`ln_post`
/// (`layer_norm_eps`, checked at intake).
const LN_EPS: f32 = 1e-5;
/// The final no-scale RMSNorm's epsilon: the TRUNK's `rms_norm_eps`
/// (`RMSNormNoScale(config.text_config.rms_norm_eps)`), 1e-5.
const OUT_RMS_EPS: f32 = 1e-5;
/// Rope theta (checked at intake).
const ROPE_THETA: f32 = 10_000.0;
/// MLP row tile, as the other towers.
pub(crate) const MUSE_MLP_TILE_ROWS: usize = 1024;
/// Attention window edge in patches (checked at intake).
const WINDOW_PATCHES: usize = 32;

/// Full attention on every fourth block and the last, else window.
fn is_full_attention(block: usize, depth: usize) -> bool {
    (block + 1) % 4 == 0 || block + 1 == depth
}

/// The half-pixel bilinear position rows for a `t x h x w` patch grid, raster
/// order, `[t * h * w, hidden]` F32. The reference's `_position_embeddings`.
///
/// `table` is `[side * side, hidden]`. Per axis the sample centre is
/// `(i + 0.5) * (side / n) - 0.5`; its floor and floor + 1 are the two taps; a
/// tap outside `[0, side)` has weight zero (and is clamped for the gather).
pub(crate) fn position_rows(table: &[f32], side: usize, hidden: usize, grid: GridThw) -> Vec<f32> {
    struct Axis {
        lo: usize,
        hi: usize,
        w_lo: f32,
        w_hi: f32,
    }
    let axis = |n: usize| -> Vec<Axis> {
        let scale = (side as f64 / n as f64) as f32;
        (0..n)
            .map(|i| {
                let g = (i as f32 + 0.5) * scale - 0.5;
                let lo = g.floor();
                let frac = g - lo;
                let (lo_i, hi_i) = (lo as i64, lo as i64 + 1);
                let valid = |v: i64| v >= 0 && v < side as i64;
                let clamp = |v: i64| v.clamp(0, side as i64 - 1) as usize;
                Axis {
                    lo: clamp(lo_i),
                    hi: clamp(hi_i),
                    w_lo: if valid(lo_i) { 1.0 - frac } else { 0.0 },
                    w_hi: if valid(hi_i) { frac } else { 0.0 },
                }
            })
            .collect()
    };
    let (rows_ax, cols_ax) = (axis(grid.h), axis(grid.w));
    let mut out = vec![0.0f32; grid.patches() * hidden];
    for t in 0..grid.t {
        for (y, ry) in rows_ax.iter().enumerate() {
            for (x, cx) in cols_ax.iter().enumerate() {
                let dst = &mut out[((t * grid.h + y) * grid.w + x) * hidden..][..hidden];
                // The reference sums four taps in the order
                // (lo, lo), (lo, hi), (hi, lo), (hi, hi) over (row, col).
                for (r, wr) in [(ry.lo, ry.w_lo), (ry.hi, ry.w_hi)] {
                    for (c, wc) in [(cx.lo, cx.w_lo), (cx.hi, cx.w_hi)] {
                        let w = wr * wc;
                        let src = &table[(r * side + c) * hidden..][..hidden];
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += s * w;
                        }
                    }
                }
            }
        }
    }
    out
}

/// The rope angle rows, `[patches, head_dim / 2]` F32, raster order: per patch
/// the width angles then the height angles, each `head_dim / 4` long, at the
/// 1-INDEXED positions `(col + 1, row + 1)`. The reference builds
/// `cat(freq_w, freq_h, freq_w, freq_h)`; the half-split rope kernel reads the
/// first half of that, which is this row.
pub(crate) fn rope_rows(grid: GridThw, head_dim: usize) -> Vec<f32> {
    let quarter = head_dim / 4;
    let spatial = head_dim / 2;
    let inv: Vec<f32> = (0..quarter)
        .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / spatial as f32))
        .collect();
    let mut out = Vec::with_capacity(grid.patches() * spatial);
    for _t in 0..grid.t {
        for y in 0..grid.h {
            for x in 0..grid.w {
                let (w, h) = ((x + 1) as f32, (y + 1) as f32);
                out.extend(inv.iter().map(|f| w * f));
                out.extend(inv.iter().map(|f| h * f));
            }
        }
    }
    out
}

/// The dim-major pixel shuffle: `src` is `[t * h * w, dim]` in raster patch
/// order; the result is `[t * (h/m) * (w/m), dim * m * m]` in merged raster
/// order with element `d * m * m + mh * m + mw` taken from dim `d` of patch
/// `(hb * m + mh, wb * m + mw)`. The reference's `reshape(t, h/m, m, w/m, m, d)`
/// transposed `(0, 1, 3, 5, 2, 4)`.
pub(crate) fn pixel_shuffle<T: Copy + Default>(
    src: &[T],
    dim: usize,
    grid: GridThw,
    m: usize,
) -> Vec<T> {
    let (hb_n, wb_n) = (grid.h / m, grid.w / m);
    let width = dim * m * m;
    let mut out = vec![T::default(); grid.t * hb_n * wb_n * width];
    for t in 0..grid.t {
        for hb in 0..hb_n {
            for wb in 0..wb_n {
                let dst = &mut out[((t * hb_n + hb) * wb_n + wb) * width..][..width];
                for mh in 0..m {
                    for mw in 0..m {
                        let patch = (t * grid.h + hb * m + mh) * grid.w + wb * m + mw;
                        let row = &src[patch * dim..][..dim];
                        for (d, v) in row.iter().enumerate() {
                            dst[d * m * m + mh * m + mw] = *v;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Per-image GPU scratch.
struct MuseScratch {
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
    mlp: gpu::MetalBuffer,
    freqs: gpu::MetalBuffer,
    segments: Vec<usize>,
}

/// Encode one block over `s.x` in place.
fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    slot: &gpu::MetalBuffer,
    roles: &BlockRoles,
    s: &MuseScratch,
    shape: &VisionShape,
    block: usize,
) -> Result<(), RealForwardError> {
    let e = RealForwardError::Gpu;
    let seq = s.seq as u32;
    let hidden = shape.hidden as u32;
    let inter = shape.intermediate as u32;
    let wide = (s.seq * shape.hidden) as u32;
    let at = |r: Role| (slot, roles.at(r));
    let scale = shape.attention_scale();

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        at(Role::Ln1W),
        at(Role::Ln1B),
        (&s.normed, 0),
        seq,
        hidden,
        LN_EPS,
    )
    .map_err(e)?;

    // q, k, v: the fused weight's three row blocks and the fused bias's
    // three thirds (repack concatenated them in that order).
    let w_block = (shape.hidden * shape.hidden * 2) as u64;
    let b_block = (shape.hidden * 2) as u64;
    for (i, out) in [&s.q, &s.k, &s.v].into_iter().enumerate() {
        let (w, w_off) = at(Role::QkvW);
        let (b, b_off) = at(Role::QkvB);
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, 0),
            (w, w_off + i as u64 * w_block),
            Some((b, b_off + i as u64 * b_block)),
            (out, 0),
            seq,
            hidden,
            hidden,
        )
        .map_err(e)?;
    }
    // Q AND K ONLY.
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
        .map_err(e)?;
    }

    if is_full_attention(block, shape.depth) {
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
            scale,
        )
        .map_err(e)?;
    } else {
        let mut start = 0usize;
        for &len in &s.segments {
            let off = (start * shape.hidden * 2) as u64;
            gpu::encode_vision_attention(
                context,
                pass,
                (&s.q, off),
                (&s.k, off),
                (&s.v, off),
                (&s.attn, off),
                len as u32,
                shape.heads as u32,
                shape.head_dim as u32,
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
        at(Role::ProjW),
        Some(at(Role::ProjB)),
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
        at(Role::Ln2W),
        at(Role::Ln2B),
        (&s.normed, 0),
        seq,
        hidden,
        LN_EPS,
    )
    .map_err(e)?;
    let mut t0 = 0usize;
    while t0 < s.seq {
        let rows = s.tile.min(s.seq - t0);
        let row_off = (t0 * shape.hidden * 2) as u64;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, row_off),
            at(Role::Fc1W),
            Some(at(Role::Fc1B)),
            (&s.mlp, 0),
            rows as u32,
            hidden,
            inter,
        )
        .map_err(e)?;
        // The exact erf form: the reference's MLP is `nn.gelu`, not tanh.
        gpu::encode_vision_gelu(
            context,
            pass,
            (&s.mlp, 0),
            (rows * shape.intermediate) as u32,
            gpu::GeluKind::Erf,
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.mlp, 0),
            at(Role::Fc2W),
            Some(at(Role::Fc2B)),
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

/// Resident tensors outside the blocks, as local byte offsets.
struct MuseResident {
    patch_embed: u64,
    pos_table: u64,
    ln_pre_w: u64,
    ln_pre_b: u64,
    ln_post_w: u64,
    ln_post_b: u64,
    fc1: u64,
    fc2: u64,
    projection: u64,
    /// The adapter's inner width (4096), recovered from `fc2`'s size.
    projector_hidden: usize,
}

impl MuseResident {
    fn resolve(index: &ResidentIndex, shape: &VisionShape) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let h = shape.hidden;
        let fc2_name = name("adapter.fc2.weight");
        let fc2_bytes = index
            .entries
            .get(&fc2_name)
            .ok_or_else(|| RealForwardError::MissingTensor(fc2_name.clone()))?
            .size_bytes as usize;
        let ph = ((fc2_bytes / 2) as f64).sqrt().round() as usize;
        if ph * ph * 2 != fc2_bytes {
            return Err(RealForwardError::Unsupported(format!(
                "{fc2_name}: {fc2_bytes} bytes is not a square FP16 matrix"
            )));
        }
        Ok(Self {
            patch_embed: fp16_view(index, &name("patch_embed.proj.weight"), h * shape.patch_dim)?,
            pos_table: fp16_view(index, &name("pos_embed.weight"), shape.pos_rows * h)?,
            ln_pre_w: fp16_view(index, &name("ln_pre.weight"), h)?,
            ln_pre_b: fp16_view(index, &name("ln_pre.bias"), h)?,
            ln_post_w: fp16_view(index, &name("ln_post.weight"), h)?,
            ln_post_b: fp16_view(index, &name("ln_post.bias"), h)?,
            fc1: fp16_view(
                index,
                &name("adapter.fc1.weight"),
                ph * shape.merger_input(),
            )?,
            fc2: fp16_view(index, &fc2_name, ph * ph)?,
            projection: fp16_view(index, &name("projection.weight"), shape.out_hidden * ph)?,
            projector_hidden: ph,
        })
    }
}

/// The Muse-Glimmer tower opened against an install (or a sidecar).
pub(crate) struct MuseVisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<BlockRoles>,
    shape: VisionShape,
    resident: MuseResident,
    /// `pos_embed.weight` decoded once, `[side * side, hidden]` F32.
    pos_table: Vec<f32>,
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
    pub(crate) last_scratch_bytes: u64,
}

fn read_fp16_host(
    weights: &gpu::ResidentGpuWeights,
    local: u64,
    elems: usize,
) -> Result<Vec<f32>, RealForwardError> {
    let at = local as usize;
    let bytes = weights.data().get(at..at + elems * 2).ok_or_else(|| {
        RealForwardError::Unsupported(format!(
            "muse vision tensor at resident offset {at} spans past the mapped region"
        ))
    })?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect())
}

fn require_muse(v: &VisionConfig) -> Result<(), RealForwardError> {
    if v.tower != VisionTowerKind::MuseGlimmer || !v.is_active() {
        return Err(RealForwardError::Unsupported(format!(
            "not an active muse_glimmer vision tower (kind {})",
            v.tower.as_str()
        )));
    }
    Ok(())
}

impl MuseVisionTower {
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        require_muse(&arch.vision)?;
        let shape = VisionShape::resolve(&arch.vision)?;
        let layout = model_io::load_packed_layout_from(
            dir,
            model_io::PACKED_VISION_DIR,
            PACKED_LAYOUT_MAX_BYTES,
        )
        .map_err(RealForwardError::Model)?;
        let layer = layout.layers.iter().find(|l| l.layer == 0).ok_or_else(|| {
            RealForwardError::Unsupported("packed_vision/layout.json has no layer 0".to_string())
        })?;
        if layer.experts.len() != shape.depth {
            return Err(RealForwardError::Unsupported(format!(
                "packed_vision carries {} blocks but the manifest declares {}",
                layer.experts.len(),
                shape.depth
            )));
        }
        let mut blocks: Vec<Option<BlockRoles>> = vec![None; shape.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, shape.depth
                ))
            })?;
            if slot
                .replace(resolve_block_roles(
                    entry,
                    entry.expert,
                    layer.expert_stride,
                    &shape,
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
        .map_err(|e| RealForwardError::Unsupported(format!("muse vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }
        let resident = MuseResident::resolve(index, &shape)?;
        let pos_table = read_fp16_host(weights, resident.pos_table, shape.pos_rows * shape.hidden)?;
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            shape,
            resident,
            pos_table,
            sidecar_weights: None,
            last_scratch_bytes: 0,
        })
    }

    /// Open against a standalone sidecar: its own resident index and mapping,
    /// kept alive for the tower's life.
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
        let mut tower = Self::open(sidecar_dir, context, &weights, &index, arch)?;
        tower.sidecar_weights = Some(weights);
        Ok(tower)
    }

    pub(crate) fn is_sidecar(&self) -> bool {
        self.sidecar_weights.is_some()
    }

    /// Encode one image to the `[merged, out_hidden]` rows the trunk takes, in
    /// merged raster order.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &PreprocessedImage,
        params: &PreprocessParams,
        mut capture: Option<VisionStages>,
    ) -> Result<(VisionEmbedding, Option<VisionStages>), RealForwardError> {
        let shape = self.shape.clone();
        if params.patch_dim() != shape.patch_dim
            || params.merge_size != shape.merge
            || params.patch_size != shape.patch_size
        {
            return Err(RealForwardError::Unsupported(format!(
                "preprocessing produced patch {} / merge {} / {}-value rows but this tower takes \
                 patch {} / merge {} / {}",
                params.patch_size,
                params.merge_size,
                params.patch_dim(),
                shape.patch_size,
                shape.merge,
                shape.patch_dim
            )));
        }
        let grid = image.grid;
        let seq = grid.patches();
        if seq == 0 || image.patch_rows.len() != seq * shape.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "muse vision input: {} values for {seq} patches of {}",
                image.patch_rows.len(),
                shape.patch_dim
            )));
        }
        if grid.h % shape.merge != 0 || grid.w % shape.merge != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "muse vision grid {}x{} is not a whole number of {m}x{m} merge groups",
                grid.h,
                grid.w,
                m = shape.merge
            )));
        }
        // Windows are in PATCHES here: a merge of 1 makes every unit a patch.
        let layout = window_layout(grid, 1, WINDOW_PATCHES)?;
        let merged = seq / (shape.merge * shape.merge);

        // Host prep, all in window order. The position rows are added BEFORE
        // the permutation in the reference; both operands are permuted by the
        // same order and the patch embedding and the add are row-wise, so the
        // result is the same.
        let rows16: Vec<half::f16> =
            permute_units(&image.patch_rows, shape.patch_dim, 1, &layout.unit_order)
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
        let pos = position_rows(&self.pos_table, shape.pos_side, shape.hidden, grid);
        let pos16: Vec<half::f16> = permute_units(&pos, shape.hidden, 1, &layout.unit_order)
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let freqs = permute_units(
            &rope_rows(grid, shape.head_dim),
            shape.head_dim / 2,
            1,
            &layout.unit_order,
        );

        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let resident = &self.resident;
        let order = &layout.unit_order;
        let last_scratch = &mut self.last_scratch_bytes;
        let ph = resident.projector_hidden;
        let mw = shape.merger_input();

        gpu::autorelease_pool(|| {
            let e = RealForwardError::Gpu;
            let wide = (seq * shape.hidden * 2) as u64;
            let tile = MUSE_MLP_TILE_ROWS.min(seq).max(1);
            let s = MuseScratch {
                seq,
                tile,
                rows: context.new_buffer_with_data(&rows16),
                pos: context.new_buffer_with_data(&pos16),
                x: context.new_output_buffer(wide),
                normed: context.new_output_buffer(wide),
                q: context.new_output_buffer(wide),
                k: context.new_output_buffer(wide),
                v: context.new_output_buffer(wide),
                attn: context.new_output_buffer(wide),
                proj: context.new_output_buffer(wide),
                mlp: context.new_output_buffer((tile * shape.intermediate * 2) as u64),
                freqs: context.new_buffer_with_data(&freqs),
                segments: layout.segments.clone(),
            };
            *last_scratch = wide * 8 + (tile * shape.intermediate * 2) as u64;
            let n_wide = seq * shape.hidden;

            // Patch embedding (no bias), the position add, then ln_pre.
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.rows, 0),
                (w.buffer(), w.gpu_offset(resident.patch_embed)),
                None,
                (&s.proj, 0),
                seq as u32,
                shape.patch_dim as u32,
                shape.hidden as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&s.proj, 0),
                (&s.pos, 0),
                n_wide as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&s.proj, 0),
                (w.buffer(), w.gpu_offset(resident.ln_pre_w)),
                (w.buffer(), w.gpu_offset(resident.ln_pre_b)),
                (&s.x, 0),
                seq as u32,
                shape.hidden as u32,
                LN_EPS,
            )
            .map_err(e)?;
            pass.commit_and_wait();
            if let Some(c) = capture.as_mut() {
                c.patch_embed = read_bits(&s.x, n_wide);
            }

            for n in 0..shape.depth {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|err| {
                    RealForwardError::Unsupported(format!("muse vision block {n} read: {err}"))
                })?;
                let pass = context.begin_pass();
                encode_block(context, &pass, &slots[slot], &blocks[n], &s, &shape, n)?;
                pass.commit_and_wait();
                if let Some(c) = capture.as_mut() {
                    if n == 0 {
                        c.block_first = read_bits(&s.x, n_wide);
                    }
                    if n + 1 == shape.depth {
                        c.block_last = read_bits(&s.x, n_wide);
                    }
                }
            }

            // ln_post, then back to the host for the un-permute and the
            // dim-major shuffle (a gather no kernel here does).
            let pass = context.begin_pass();
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&s.x, 0),
                (w.buffer(), w.gpu_offset(resident.ln_post_w)),
                (w.buffer(), w.gpu_offset(resident.ln_post_b)),
                (&s.normed, 0),
                seq as u32,
                shape.hidden as u32,
                LN_EPS,
            )
            .map_err(e)?;
            pass.commit_and_wait();
            let in_window_order = read_bits(&s.normed, n_wide);
            let mut raster = vec![0u16; in_window_order.len()];
            for (j, &patch) in order.iter().enumerate() {
                raster[patch * shape.hidden..(patch + 1) * shape.hidden]
                    .copy_from_slice(&in_window_order[j * shape.hidden..(j + 1) * shape.hidden]);
            }
            let shuffled = pixel_shuffle(&raster, shape.hidden, grid, shape.merge);
            let adapter_in = context.new_buffer_with_data(&shuffled);

            // Adapter: gelu(fc2(gelu(fc1(x)))), projection, no-scale RMSNorm.
            let h1 = context.new_output_buffer((merged * ph * 2) as u64);
            let h2 = context.new_output_buffer((merged * ph * 2) as u64);
            let out = context.new_output_buffer((merged * shape.out_hidden * 2) as u64);
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&adapter_in, 0),
                (w.buffer(), w.gpu_offset(resident.fc1)),
                None,
                (&h1, 0),
                merged as u32,
                mw as u32,
                ph as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&h1, 0),
                (merged * ph) as u32,
                gpu::GeluKind::Erf,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&h1, 0),
                (w.buffer(), w.gpu_offset(resident.fc2)),
                None,
                (&h2, 0),
                merged as u32,
                ph as u32,
                ph as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&h2, 0),
                (merged * ph) as u32,
                gpu::GeluKind::Erf,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&h2, 0),
                (w.buffer(), w.gpu_offset(resident.projection)),
                None,
                (&out, 0),
                merged as u32,
                ph as u32,
                shape.out_hidden as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_rms_norm(
                context,
                &pass,
                (&out, 0),
                None,
                (&out, 0),
                merged as u32,
                shape.out_hidden as u32,
                OUT_RMS_EPS,
            )
            .map_err(e)?;
            pass.commit_and_wait();

            let rows = read_bits(&out, merged * shape.out_hidden);
            if let Some(i) = rows
                .iter()
                .position(|&b| !half::f16::from_bits(b).is_finite())
            {
                return Err(RealForwardError::Unsupported(format!(
                    "muse vision output is non-finite at element {i}; the FP16 tower overflowed"
                )));
            }
            Ok((
                VisionEmbedding {
                    rows,
                    merged_tokens: merged,
                    out_hidden: shape.out_hidden,
                    grid,
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

    #[test]
    fn full_attention_is_every_fourth_block_and_the_last() {
        let full: Vec<usize> = (0..50).filter(|&b| is_full_attention(b, 50)).collect();
        assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 49]);
    }

    #[test]
    fn windows_are_32_patches_with_ragged_edges_in_raster_order() {
        // 40 x 70 patches: windows 32x32 with a 8-row bottom band and a
        // 6-column right band. 2 x 3 windows, all non-empty.
        let layout = window_layout(GridThw::new(1, 40, 70), 1, WINDOW_PATCHES).unwrap();
        assert_eq!(
            layout.segments,
            vec![32 * 32, 32 * 32, 32 * 6, 8 * 32, 8 * 32, 8 * 6]
        );
        assert_eq!(layout.segments.iter().sum::<usize>(), 40 * 70);
        // The first window reads its first row then its second row.
        assert_eq!(&layout.unit_order[..3], &[0, 1, 2]);
        assert_eq!(layout.unit_order[32], 70);
        // The second window starts at column 32 of row 0.
        assert_eq!(layout.unit_order[1024], 32);
    }

    #[test]
    fn a_grid_inside_one_window_is_one_segment_in_page_order() {
        let layout = window_layout(GridThw::new(1, 30, 32), 1, WINDOW_PATCHES).unwrap();
        assert_eq!(layout.segments, vec![30 * 32]);
        assert_eq!(layout.unit_order, (0..30 * 32).collect::<Vec<_>>());
    }

    /// A 2 x 2 table whose rows are 0, 1, 2, 3 in every channel makes the
    /// interpolated value readable as a number.
    #[test]
    fn position_rows_use_half_pixel_centres_and_zero_the_off_table_tap() {
        let table: Vec<f32> = (0..4).flat_map(|r| [r as f32, r as f32]).collect();
        // Grid 4x4 over a 2x2 table: centres (i + 0.5) * 0.5 - 0.5 =
        // -0.25, 0.25, 0.75, 1.25.
        let rows = position_rows(&table, 2, 2, GridThw::new(1, 4, 4));
        let at = |y: usize, x: usize| rows[(y * 4 + x) * 2];
        // Interior (0.25, 0.25): taps rows {0,1} x cols {0,1} at fractions
        // .25/.25 -> 0*.5625 + 1*.1875 + 2*.1875 + 3*.0625 = 0.75.
        assert!((at(1, 1) - 0.75).abs() < 1e-6, "{}", at(1, 1));
        // (-0.25, -0.25): lo = -1 is off the table (weight 0), hi = 0 has
        // weight frac = 0.75 per axis: 0 * 0.5625 = 0. The align-corners rule
        // would clamp the sample to the edge instead and read the same 0, so
        // this corner alone cannot tell them apart; the next one can.
        assert!(at(0, 0).abs() < 1e-6);
        // (1.25, 1.25): hi = 2 is off the table, lo = 1 has weight 1 - 0.25 =
        // 0.75 per axis, so the value is 3 * 0.5625 = 1.6875, NOT the clamped
        // 3.0 an edge-replicating interpolation would give.
        assert!((at(3, 3) - 1.6875).abs() < 1e-5, "{}", at(3, 3));
    }

    #[test]
    fn rope_rows_are_one_indexed_width_first() {
        let rows = rope_rows(GridThw::new(1, 2, 3), 96);
        let width = 48;
        // Patch (row 0, col 0): position (w, h) = (1, 1); pair 0's inverse
        // frequency is 1, so both its width and height angles are 1.
        assert!((rows[0] - 1.0).abs() < 1e-6);
        assert!((rows[24] - 1.0).abs() < 1e-6);
        // Patch (row 1, col 2): w = 3, h = 2, width angles first.
        let p = (1 * 3 + 2) * width;
        assert!((rows[p] - 3.0).abs() < 1e-6);
        assert!((rows[p + 24] - 2.0).abs() < 1e-6);
        assert_eq!(rows.len(), 6 * width);
    }

    #[test]
    fn the_pixel_shuffle_is_dim_major() {
        // A 2 x 2 patch grid of 3-dim rows, patch p dim d = 10 p + d.
        let src: Vec<u16> = (0..4u16)
            .flat_map(|p| (0..3u16).map(move |d| 10 * p + d))
            .collect();
        let out = pixel_shuffle(&src, 3, GridThw::new(1, 2, 2), 2);
        // One merged row of 12: dim 0 of patches 0..3, then dim 1, then dim 2.
        assert_eq!(out, vec![0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32]);
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

    /// The tower on the REAL checkpoint against
    /// `scripts/muse_glimmer_tower_reference.py`, a numpy transcription of the
    /// reference (not the reference executed). Patch rows come from
    /// `crates/vision-io`, so this holds the preprocessing and the tower
    /// together. Stage names localize a gap: patch embed (after `ln_pre`),
    /// the first and last blocks (window order), then the final rows.
    ///
    /// ```sh
    /// turbospark-model pull-vision --repo mlx-community/Muse-Glimmer-30B-4bit \
    ///   --alias muse-glimmer-vision --out /Volumes/T7/turbospark/vision/muse-glimmer.gturbo-vision
    /// TURBOSPARK_MUSE_SIDECAR_DIR=/Volumes/T7/turbospark/vision/muse-glimmer.gturbo-vision \
    /// TURBOSPARK_MUSE_MODEL_DIR=/Volumes/T7/hf-models/Muse-Glimmer-30B-4bit \
    /// TURBOSPARK_MUSE_IMAGE=/Volumes/T7/vision-pages/page-560x700.png \
    /// TURBOSPARK_MUSE_SCRATCH=/Volumes/T7/vision-dumps/muse-560x700 \
    ///   cargo test -p turbospark-runtime --release --lib \
    ///   the_real_tower_matches_the_numpy_reference -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a Muse tower sidecar, the HF checkpoint dir (shards 3 and 4), an image and python3 with numpy"]
    fn the_real_tower_matches_the_numpy_reference() {
        let sidecar = env_path("TURBOSPARK_MUSE_SIDECAR_DIR");
        let model_dir = env_path("TURBOSPARK_MUSE_MODEL_DIR");
        let image = env_path("TURBOSPARK_MUSE_IMAGE");
        let scratch = env_path("TURBOSPARK_MUSE_SCRATCH");
        std::fs::create_dir_all(&scratch).unwrap();

        let (_record, vision) = model_io::load_vision_sidecar(&sidecar).expect("sidecar");
        let arch = model_io::sidecar_arch(
            model_io::ModelFamily::MuseGlimmer,
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
                "/../../scripts/muse_glimmer_tower_reference.py"
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
        let index = model_io::load_resident_index(&sidecar.join("model_weights.bin")).unwrap();
        let buffer = model_io::ResidentBuffer::map(
            &sidecar.join("model_weights.bin"),
            index.header.index_size,
            index.header.resident_size,
        )
        .unwrap();
        let weights = gpu::ResidentGpuWeights::wrap(context.device(), buffer).unwrap();
        let mut tower =
            MuseVisionTower::open_with_sidecar(&sidecar, &context, &arch).expect("open tower");
        let (embedding, stages) = tower
            .run(
                &mut context,
                &weights,
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
            worst > 0.9999,
            "the tower disagrees with the reference (worst stage cosine {worst})"
        );
    }

    #[test]
    fn the_pixel_shuffle_walks_merged_blocks_in_raster_order() {
        // 2 x 4 patches: two merged blocks side by side, 1-dim rows.
        let src: Vec<u16> = (0..8).collect();
        let out = pixel_shuffle(&src, 1, GridThw::new(1, 2, 4), 2);
        // Block 0 = patches (0,0) (0,1) (1,0) (1,1) = 0 1 4 5.
        assert_eq!(out, vec![0, 1, 4, 5, 2, 3, 6, 7]);
    }
}
