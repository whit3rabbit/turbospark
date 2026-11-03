//! The SigLIP-so400m tower DeepSeek-VL2 carries
//! (`docs/DEEPSEEK_VL2_VISION_PHASE0.md`).
//!
//! A sibling of the CLIP tower ([`super::clip`]) and shaped like it: blocks
//! stream from `packed_vision/` in the qwen tower's twelve-role blob, the
//! residents are `vision.`-prefixed FP16, the sidecar opens through the same
//! two constructors. What differs is the forward pass and its batching.
//!
//! # The forward pass
//!
//! ```text
//! x   = patch_embed(rows) + pos                      # per tile [729, 1152]
//! for blk in 0..27:                                  # no pre-norm, no class row
//!     x += attn(ln1(x))                              # eps 1e-6, no rope, no mask
//!     x += fc2(gelu_tanh(fc1(ln2(x))))               # eps 1e-6
//! y   = ln(x)                                        # eps 1e-5, NOT 1e-6
//! u   = unfold_2x2(zero_pad_28(y))                   # [196, 4608], channel-major
//! out = linear_2(gelu_erf(linear_1(u)))              # [196, 1280]
//! ```
//!
//! # Every tile of an image runs in one pass
//!
//! DeepSeek-VL2 cuts an image into a global view plus up to nine local
//! tiles, each its own 729-row sequence. Attention never crosses a tile, but
//! everything else is row-wise, so the matmuls, norms and activations run over
//! all `tiles * 729` rows at once and only attention dispatches per tile, at
//! a byte offset. That reads each block from disk once per IMAGE instead of
//! once per tile (the blocks are 30 MiB each, so the difference is 800 MiB of
//! pread per extra tile).
//!
//! The tower's last stage (the unfold) is a pure data rearrangement, done on
//! the host between two GPU passes. The final output is the whole prompt run:
//! the projector's rows interleaved with the learned `image_newline` and
//! `view_separator` embeddings in the reference's layout
//! ([`turbospark_vision_io::deepseek_vl2_row_layout`]).

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{deepseek_vl2_row_layout, DeepseekVl2Row, DeepseekVl2Tiles};

use super::shape::VisionShape;
use super::weights::{fp16_view, resolve_block_roles, BlockRoles, Role, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// The blocks' LayerNorm epsilon (`VisionConfig.layer_norm_eps` in the
/// reference). A kind CONSTANT, as the other towers' are (AGENTS.md Gotcha 24).
const BLOCK_LAYER_NORM_EPS: f32 = 1e-6;

/// The final norm's epsilon: `nn.LayerNorm(config.width)` with no `eps`, i.e.
/// MLX's default 1e-5, NOT the blocks' 1e-6. A one-line trap, pinned by name.
const FINAL_LAYER_NORM_EPS: f32 = 1e-5;

/// The residual stream at the points the cross-engine gate compares. Each is
/// `[tiles * 729, hidden]` FP16 bits except `projector`, which is
/// `[tiles * 196, out_hidden]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SiglipStages {
    /// After the patch embedding and the position add.
    pub embed: Vec<u16>,
    /// After block 0.
    pub block_first: Vec<u16>,
    /// After the last block.
    pub block_last: Vec<u16>,
    /// After the final LayerNorm: the tower's output.
    pub normed: Vec<u16>,
    /// The projector's rows, before the newline and separator interleave.
    pub projector: Vec<u16>,
}

/// The eleven resident tensors' offsets into the resident region.
#[derive(Debug, Clone)]
struct SiglipResident {
    patch_w: u64,
    patch_b: u64,
    pos_embed: u64,
    norm_w: u64,
    norm_b: u64,
    proj1_w: u64,
    proj1_b: u64,
    proj2_w: u64,
    proj2_b: u64,
    image_newline: u64,
    view_separator: u64,
}

impl SiglipResident {
    fn resolve(index: &ResidentIndex, shape: &VisionShape) -> Result<Self, RealForwardError> {
        let h = shape.hidden;
        let o = shape.out_hidden;
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        Ok(Self {
            patch_w: fp16_view(index, &name("patch_embed.proj.weight"), h * shape.patch_dim)?,
            patch_b: fp16_view(index, &name("patch_embed.proj.bias"), h)?,
            pos_embed: fp16_view(index, &name("pos_embed.weight"), shape.pos_rows * h)?,
            norm_w: fp16_view(index, &name("norm.weight"), h)?,
            norm_b: fp16_view(index, &name("norm.bias"), h)?,
            proj1_w: fp16_view(
                index,
                &name("projector.layers.0.weight"),
                o * shape.merger_input(),
            )?,
            proj1_b: fp16_view(index, &name("projector.layers.0.bias"), o)?,
            proj2_w: fp16_view(index, &name("projector.layers.2.weight"), o * o)?,
            proj2_b: fp16_view(index, &name("projector.layers.2.bias"), o)?,
            image_newline: fp16_view(index, &name("image_newline"), o)?,
            view_separator: fp16_view(index, &name("view_separator"), o)?,
        })
    }
}

/// The tower's dimensions, in the qwen tower's [`VisionShape`] so the block
/// role resolver is shared. `merge` is the projector's unfold factor and
/// `pos_side` the patch grid's side.
fn siglip_shape(vision: &VisionConfig) -> Result<VisionShape, RealForwardError> {
    let bad = |detail: String| RealForwardError::Unsupported(detail);
    if vision.tower != VisionTowerKind::SigLip || !vision.is_active() {
        return Err(bad(
            "this install declares no SigLIP vision tower".to_string()
        ));
    }
    let u = |v: i64| usize::try_from(v).unwrap_or(0);
    let (depth, hidden, inter, heads) = (
        u(vision.depth),
        u(vision.hidden_size),
        u(vision.intermediate_size),
        u(vision.num_heads),
    );
    let (patch, out_hidden, pos_rows, merge) = (
        u(vision.patch_size),
        u(vision.out_hidden_size),
        u(vision.num_position_embeddings),
        u(vision.spatial_merge_size),
    );
    for (label, value) in [
        ("depth", depth),
        ("hidden_size", hidden),
        ("intermediate_size", inter),
        ("num_heads", heads),
        ("patch_size", patch),
        ("out_hidden_size", out_hidden),
        ("num_position_embeddings", pos_rows),
        ("spatial_merge_size", merge),
    ] {
        if value == 0 {
            return Err(bad(format!("SigLIP config field {label} is zero")));
        }
    }
    if (vision.temporal_patch_size, vision.in_channels) != (1, 3) {
        return Err(bad(format!(
            "SigLIP tower with temporal {}, channels {}; need 1, 3",
            vision.temporal_patch_size, vision.in_channels
        )));
    }
    if merge != 2 {
        return Err(bad(format!(
            "SigLIP projector unfold {merge}; only the 2x2 unfold is implemented"
        )));
    }
    if hidden % heads != 0 {
        return Err(bad(format!(
            "SigLIP hidden_size {hidden} is not divisible by num_heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    if head_dim > gpu::MAX_ATTENTION_HEAD_DIM as usize {
        return Err(bad(format!(
            "SigLIP head_dim {head_dim} exceeds the attention kernel's tile ({})",
            gpu::MAX_ATTENTION_HEAD_DIM
        )));
    }
    let side = (pos_rows as f64).sqrt().round() as usize;
    if side * side != pos_rows {
        return Err(bad(format!(
            "SigLIP position table of {pos_rows} rows is not a square grid"
        )));
    }
    Ok(VisionShape {
        depth,
        hidden,
        intermediate: inter,
        heads,
        head_dim,
        merge,
        out_hidden,
        patch_dim: 3 * patch * patch,
        pos_rows,
        pos_side: side,
        patch_size: patch,
        deepstack: Vec::new(),
    })
}

/// The SigLIP tower: its streamer, its blocks' roles and its resident offsets.
pub(crate) struct SiglipVisionTower {
    /// Declared BEFORE `streamer` so the buffers drop before the allocations
    /// they alias.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<BlockRoles>,
    resident: SiglipResident,
    shape: VisionShape,
    /// `image_newline` and `view_separator`, FP16 bytes, copied out at open:
    /// the interleave is a host assembly.
    newline_row: Vec<u8>,
    separator_row: Vec<u8>,
    /// The sidecar's own resident weights, `Some` when the tower was opened
    /// against a standalone sidecar rather than the trunk's install.
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
}

impl SiglipVisionTower {
    /// Open against a combined install's own resident weights.
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        Self::build(dir, context, weights, index, arch)
    }

    /// Open against a standalone sidecar directory, with its own resident
    /// mapping, exactly as [`super::clip::ClipVisionTower::open_with_sidecar`]
    /// does.
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
        let mut tower = Self::build(sidecar_dir, context, &weights, &index, arch)?;
        tower.sidecar_weights = Some(weights);
        Ok(tower)
    }

    fn build(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let shape = siglip_shape(&arch.vision)?;
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
        // By the entry's own index, never by position: the writer may pack out
        // of order.
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
        .map_err(|e| RealForwardError::Unsupported(format!("SigLIP vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }

        let resident = SiglipResident::resolve(index, &shape)?;
        let copy_row = |offset: u64, what: &str| -> Result<Vec<u8>, RealForwardError> {
            let at = offset as usize;
            Ok(weights
                .data()
                .get(at..at + shape.out_hidden * 2)
                .ok_or_else(|| {
                    RealForwardError::Unsupported(format!(
                        "SigLIP {what} spans past the mapped region"
                    ))
                })?
                .to_vec())
        };
        let newline_row = copy_row(resident.image_newline, "image_newline")?;
        let separator_row = copy_row(resident.view_separator, "view_separator")?;
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            resident,
            shape,
            newline_row,
            separator_row,
            sidecar_weights: None,
        })
    }

    /// Whether this tower is sidecar-backed (test-only engagement proof).
    #[allow(dead_code)]
    pub(crate) fn is_sidecar(&self) -> bool {
        self.sidecar_weights.is_some()
    }

    /// Encode one image to `[image_tokens, out_hidden]` FP16 bits: the whole
    /// prompt run, newline and separator rows included.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        tiles: &DeepseekVl2Tiles,
        want_stages: bool,
    ) -> Result<(Vec<u16>, Option<SiglipStages>), RealForwardError> {
        let s = &self.shape;
        let per = tiles.patches_per_side * tiles.patches_per_side;
        if per != s.pos_rows {
            return Err(RealForwardError::Unsupported(format!(
                "the image's tiles have {per} patches but this tower's position table covers {}; \
                 SigLIP does not interpolate, and the processor always tiles at one size",
                s.pos_rows
            )));
        }
        let n_tiles = tiles.tiles();
        let seq = n_tiles * per;
        if tiles.pixels.len() != seq * s.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "the image's patch rows are {}-wide but this tower's patch embedding takes {}",
                tiles.pixels.len() / seq.max(1),
                s.patch_dim
            )));
        }
        let merged = tiles.merged_per_side;
        let grid = tiles.patches_per_side;
        if merged != grid.div_ceil(s.merge) {
            return Err(RealForwardError::Unsupported(format!(
                "tiles carry a {merged}-wide merged grid but a {grid}-wide patch grid unfolds to {}",
                grid.div_ceil(s.merge)
            )));
        }
        let hidden = s.hidden;
        let o = s.out_hidden;
        let wide = seq * hidden;
        let merged_rows = n_tiles * merged * merged;
        let unfold_width = s.merger_input();
        let rows: Vec<half::f16> = tiles
            .pixels
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let gpu_err = RealForwardError::Gpu;
        let bytes = |elems: usize| (elems * 2) as u64;

        let result = gpu::autorelease_pool(|| {
            let rows_buf = context.new_buffer_with_data(&rows);
            let x = context.new_output_buffer(bytes(wide));
            let normed = context.new_output_buffer(bytes(wide));
            let q = context.new_output_buffer(bytes(wide));
            let k = context.new_output_buffer(bytes(wide));
            let v = context.new_output_buffer(bytes(wide));
            let attn = context.new_output_buffer(bytes(wide));
            let proj = context.new_output_buffer(bytes(wide));
            let h1 = context.new_output_buffer(bytes(seq * s.intermediate));
            let read = |b: &gpu::MetalBuffer, count: usize| -> Vec<u16> {
                gpu::read_buffer_f16(b, 0, count)
                    .into_iter()
                    .map(|v| v.to_bits())
                    .collect()
            };
            let mut stages = want_stages.then(SiglipStages::default);

            // Embedding: every tile's patch rows through the conv-as-GEMM,
            // then each tile's own copy of the position table.
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&rows_buf, 0),
                (w.buffer(), w.gpu_offset(self.resident.patch_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.patch_b))),
                (&x, 0),
                seq as u32,
                s.patch_dim as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;
            for tile in 0..n_tiles {
                gpu::encode_vision_residual_add(
                    context,
                    &pass,
                    (&x, bytes(tile * per * hidden)),
                    (w.buffer(), w.gpu_offset(self.resident.pos_embed)),
                    (per * hidden) as u32,
                )
                .map_err(gpu_err)?;
            }
            pass.commit_and_wait();
            if let Some(st) = stages.as_mut() {
                st.embed = read(&x, wide);
            }

            let scratch = BlockScratch {
                x: &x,
                normed: &normed,
                q: &q,
                k: &k,
                v: &v,
                attn: &attn,
                proj: &proj,
                h1: &h1,
                rows: seq,
                tiles: n_tiles,
                per_tile: per,
            };
            for blk in 0..s.depth {
                let slot = blk % VISION_SLOTS;
                self.streamer
                    .load_expert_into_slot(0, blk, slot)
                    .map_err(|e| {
                        RealForwardError::Unsupported(format!(
                            "SigLIP vision block {blk} read: {e}"
                        ))
                    })?;
                let pass = context.begin_pass();
                encode_siglip_block(
                    context,
                    &pass,
                    &self.slot_buffers[slot],
                    &self.blocks[blk],
                    &scratch,
                    s,
                )?;
                pass.commit_and_wait();
                if let Some(st) = stages.as_mut() {
                    if blk == 0 {
                        st.block_first = read(&x, wide);
                    }
                    if blk + 1 == s.depth {
                        st.block_last = read(&x, wide);
                    }
                }
            }

            // The final norm, in place; its output is the tower's.
            let pass = context.begin_pass();
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&x, 0),
                (w.buffer(), w.gpu_offset(self.resident.norm_w)),
                (w.buffer(), w.gpu_offset(self.resident.norm_b)),
                (&x, 0),
                seq as u32,
                hidden as u32,
                FINAL_LAYER_NORM_EPS,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            let tower_out = read(&x, wide);

            // The unfold is data movement only: pad each tile's grid to an even
            // side with zero rows and gather 2x2 neighbourhoods channel-major.
            let unfolded = unfold_2x2(&tower_out, n_tiles, grid, merged, hidden);
            let unfolded_f16: Vec<half::f16> =
                unfolded.iter().map(|&b| half::f16::from_bits(b)).collect();
            let unfolded_buf = context.new_buffer_with_data(&unfolded_f16);
            let mid = context.new_output_buffer(bytes(merged_rows * o));
            let out = context.new_output_buffer(bytes(merged_rows * o));

            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&unfolded_buf, 0),
                (w.buffer(), w.gpu_offset(self.resident.proj1_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.proj1_b))),
                (&mid, 0),
                merged_rows as u32,
                unfold_width as u32,
                o as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&mid, 0),
                (merged_rows * o) as u32,
                gpu::GeluKind::Erf,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&mid, 0),
                (w.buffer(), w.gpu_offset(self.resident.proj2_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.proj2_b))),
                (&out, 0),
                merged_rows as u32,
                o as u32,
                o as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();

            let projected = read(&out, merged_rows * o);
            if let Some(st) = stages.as_mut() {
                st.normed = tower_out;
                st.projector = projected.clone();
            }
            Ok::<_, RealForwardError>((projected, stages))
        })?;
        let (projected, stages) = result;

        // Interleave the learned rows, in the reference's layout.
        let le = |row: &[u8]| -> Vec<u16> {
            row.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        let newline = le(&self.newline_row);
        let separator = le(&self.separator_row);
        let layout = deepseek_vl2_row_layout(tiles.tiles_w, tiles.tiles_h, merged);
        let mut out = Vec::with_capacity(layout.len() * o);
        let per_merged = merged * merged;
        for item in layout {
            match item {
                DeepseekVl2Row::Feature { tile, row } => {
                    let at = (tile * per_merged + row) * o;
                    out.extend_from_slice(&projected[at..at + o]);
                }
                DeepseekVl2Row::Newline => out.extend_from_slice(&newline),
                DeepseekVl2Row::Separator => out.extend_from_slice(&separator),
            }
        }
        Ok((out, stages))
    }
}

/// The projector's `unfold`: per tile, the `grid x grid` tower rows are padded
/// at the bottom and right to `2 * merged` with zero rows, then each 2x2
/// neighbourhood becomes one row of `hidden * 4` values laid out
/// channel-major: element `c * 4 + dy * 2 + dx`. That is
/// `x.transpose(B, C, H, W)[:, :, i:i+2, j:j+2].reshape(B, -1)` in the
/// reference, and the order the first projector layer's weight was trained on.
pub(crate) fn unfold_2x2(
    tower_rows: &[u16],
    tiles: usize,
    grid: usize,
    merged: usize,
    hidden: usize,
) -> Vec<u16> {
    let side = 2 * merged;
    let mut out = vec![0u16; tiles * merged * merged * hidden * 4];
    for t in 0..tiles {
        for i in 0..merged {
            for j in 0..merged {
                let dst_row = (t * merged * merged + i * merged + j) * hidden * 4;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (y, x) = (2 * i + dy, 2 * j + dx);
                        if y >= grid || x >= grid || y >= side || x >= side {
                            continue; // the zero pad
                        }
                        let src = ((t * grid + y) * grid + x) * hidden;
                        for c in 0..hidden {
                            out[dst_row + c * 4 + dy * 2 + dx] = tower_rows[src + c];
                        }
                    }
                }
            }
        }
    }
    out
}

/// The per-run buffers one block reads and writes.
pub(super) struct BlockScratch<'a> {
    pub(super) x: &'a gpu::MetalBuffer,
    pub(super) normed: &'a gpu::MetalBuffer,
    pub(super) q: &'a gpu::MetalBuffer,
    pub(super) k: &'a gpu::MetalBuffer,
    pub(super) v: &'a gpu::MetalBuffer,
    pub(super) attn: &'a gpu::MetalBuffer,
    pub(super) proj: &'a gpu::MetalBuffer,
    pub(super) h1: &'a gpu::MetalBuffer,
    /// Rows in the batch: `tiles * per_tile`.
    pub(super) rows: usize,
    pub(super) tiles: usize,
    pub(super) per_tile: usize,
}

/// One SigLIP block: `x += attn(ln1(x)); x += fc2(gelu_tanh(fc1(ln2(x))))`.
///
/// The CLIP block with this kind's epsilon and activation, batched over tiles:
/// the row-wise stages see all rows, attention is dispatched once per tile.
pub(super) fn encode_siglip_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    slot: &gpu::MetalBuffer,
    roles: &BlockRoles,
    b: &BlockScratch<'_>,
    shape: &VisionShape,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let rows = b.rows as u32;
    let hidden = shape.hidden as u32;
    let inter = shape.intermediate as u32;
    let wide = (b.rows * shape.hidden) as u32;
    let at = |role: Role| roles.at(role);

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (b.x, 0),
        (slot, at(Role::Ln1W)),
        (slot, at(Role::Ln1B)),
        (b.normed, 0),
        rows,
        hidden,
        BLOCK_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;

    // q, k, v at rows 0, hidden, 2 * hidden of the fused weight.
    let row_bytes = (shape.hidden * shape.hidden * 2) as u64;
    let bias_bytes = (shape.hidden * 2) as u64;
    for (i, out) in [b.q, b.k, b.v].into_iter().enumerate() {
        gpu::encode_vision_matmul(
            context,
            pass,
            (b.normed, 0),
            (slot, at(Role::QkvW) + i as u64 * row_bytes),
            Some((slot, at(Role::QkvB) + i as u64 * bias_bytes)),
            (out, 0),
            rows,
            hidden,
            hidden,
        )
        .map_err(gpu_err)?;
    }

    // Attention never crosses a tile: one dispatch per tile at its byte offset.
    let tile_bytes = (b.per_tile * shape.hidden * 2) as u64;
    for tile in 0..b.tiles {
        let off = tile as u64 * tile_bytes;
        gpu::encode_vision_attention(
            context,
            pass,
            (b.q, off),
            (b.k, off),
            (b.v, off),
            (b.attn, off),
            b.per_tile as u32,
            shape.heads as u32,
            shape.head_dim as u32,
            shape.attention_scale(),
        )
        .map_err(gpu_err)?;
    }

    gpu::encode_vision_matmul(
        context,
        pass,
        (b.attn, 0),
        (slot, at(Role::ProjW)),
        Some((slot, at(Role::ProjB))),
        (b.proj, 0),
        rows,
        hidden,
        hidden,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_residual_add(context, pass, (b.x, 0), (b.proj, 0), wide).map_err(gpu_err)?;

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (b.x, 0),
        (slot, at(Role::Ln2W)),
        (slot, at(Role::Ln2B)),
        (b.normed, 0),
        rows,
        hidden,
        BLOCK_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (b.normed, 0),
        (slot, at(Role::Fc1W)),
        Some((slot, at(Role::Fc1B))),
        (b.h1, 0),
        rows,
        hidden,
        inter,
    )
    .map_err(gpu_err)?;
    // `nn.GELU(approx="precise")` in MLX is the tanh form.
    gpu::encode_vision_gelu(context, pass, (b.h1, 0), rows * inter, gpu::GeluKind::Tanh)
        .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (b.h1, 0),
        (slot, at(Role::Fc2W)),
        Some((slot, at(Role::Fc2B))),
        (b.proj, 0),
        rows,
        inter,
        hidden,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_residual_add(context, pass, (b.x, 0), (b.proj, 0), wide).map_err(gpu_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VisionConfig {
        VisionConfig {
            depth: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 1,
            in_channels: 3,
            spatial_merge_size: 2,
            num_position_embeddings: 729,
            out_hidden_size: 1280,
            mrope_section: [0, 0, 0],
            deepstack_visual_indexes: Vec::new(),
            vision_start_token_id: 0,
            vision_end_token_id: 0,
            image_token_id: 128_815,
            video_token_id: 0,
            tower: VisionTowerKind::SigLip,
        }
    }

    #[test]
    fn the_deepseek_vl2_shape_resolves_with_a_27_wide_grid() {
        let s = siglip_shape(&config()).unwrap();
        assert_eq!((s.hidden, s.heads, s.head_dim), (1152, 16, 72));
        assert_eq!((s.patch_dim, s.pos_side, s.merger_input()), (588, 27, 4608));
    }

    #[test]
    fn a_shape_the_tower_cannot_run_is_refused() {
        let mut c = config();
        c.tower = VisionTowerKind::Clip;
        assert!(siglip_shape(&c).is_err());
        let mut c = config();
        c.num_position_embeddings = 730;
        assert!(siglip_shape(&c).is_err());
        let mut c = config();
        c.spatial_merge_size = 3;
        assert!(siglip_shape(&c).is_err());
        let mut c = config();
        c.hidden_size = 1000; // not divisible by 16
        assert!(siglip_shape(&c).is_err());
    }

    #[test]
    fn unfold_is_channel_major_and_zero_pads_the_odd_edge() {
        // A 3x3 grid (odd, like 27) of 2-channel rows whose value encodes its
        // own position: row r, channel c -> 10 * r + c + 1.
        let (grid, merged, hidden) = (3usize, 2usize, 2usize);
        let rows: Vec<u16> = (0..grid * grid)
            .flat_map(|r| (0..hidden).map(move |c| (10 * r + c + 1) as u16))
            .collect();
        let out = unfold_2x2(&rows, 1, grid, merged, hidden);
        assert_eq!(out.len(), merged * merged * hidden * 4);
        // Patch (0, 0) gathers rows (0,0) (0,1) (1,0) (1,1) = 0, 1, 3, 4.
        // Channel 0 first: element c * 4 + dy * 2 + dx.
        assert_eq!(&out[0..4], &[1, 11, 31, 41]);
        assert_eq!(&out[4..8], &[2, 12, 32, 42]);
        // Patch (0, 1): dx = 1 falls on column 3, past the grid, so zero.
        let p01 = hidden * 4;
        assert_eq!(&out[p01..p01 + 4], &[21, 0, 51, 0]);
        // Patch (1, 1) is mostly pad: only row (2, 2) = 8 is real.
        let p11 = 3 * hidden * 4;
        assert_eq!(&out[p11..p11 + 4], &[81, 0, 0, 0]);
    }
}
