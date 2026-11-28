//! The variable-size SigLIP tower the Idefics3 and SmolVLM wrappers carry
//! (`docs/IDEFICS3_VISION_PHASE0.md`).
//!
//! The block is [`super::siglip`]'s exactly (LayerNorm 1e-6, biased
//! attention, tanh GELU, no class row, no rope, no mask), so its encoder and
//! the twelve-role blob resolver are shared. What differs:
//!
//! ```text
//! x   = patch_embed(rows) + pos          # per frame [side^2, hidden]
//! for blk: SigLIP block
//! y   = ln(x)                            # eps 1e-6 (HF), not SigLIP's 1e-5
//! u   = pixel_shuffle(y, scale)          # [(side/scale)^2, hidden*scale^2]
//! out = u @ proj                         # no bias, no activation
//! ```
//!
//! The reference rescales position ids by each frame's valid patch count.
//! The processor only emits full square frames, where that rule is the
//! identity, so the table is added row for row and a frame of any other size
//! is refused. HF builds the post-norm with the config's 1e-6; mlx-vlm leaves
//! it at MLX's 1e-5. This follows HF, the training definition.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{idefics3_pixel_shuffle, Idefics3Tiles};

use super::shape::VisionShape;
use super::siglip::{encode_siglip_block, BlockScratch};
use super::weights::{fp16_view, resolve_block_roles, BlockRoles, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

const POST_LAYER_NORM_EPS: f32 = 1e-6;

/// The tower's dimensions. `merge` is the pixel shuffle's factor.
fn idefics3_shape(vision: &VisionConfig) -> Result<VisionShape, RealForwardError> {
    let bad = |detail: String| RealForwardError::Unsupported(detail);
    if vision.tower != VisionTowerKind::SigLipNaViT || !vision.is_active() {
        return Err(bad(
            "this install declares no siglip_navit vision tower".to_string()
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
    if [
        depth, hidden, inter, heads, patch, out_hidden, pos_rows, merge,
    ]
    .contains(&0)
    {
        return Err(bad("siglip_navit config has a zero field".to_string()));
    }
    if (vision.temporal_patch_size, vision.in_channels) != (1, 3) {
        return Err(bad(format!(
            "siglip_navit tower with temporal {}, channels {}; need 1, 3",
            vision.temporal_patch_size, vision.in_channels
        )));
    }
    if hidden % heads != 0 {
        return Err(bad(format!(
            "siglip_navit hidden_size {hidden} is not divisible by num_heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    if head_dim > gpu::MAX_ATTENTION_HEAD_DIM as usize {
        return Err(bad(format!(
            "siglip_navit head_dim {head_dim} exceeds the attention kernel's tile ({})",
            gpu::MAX_ATTENTION_HEAD_DIM
        )));
    }
    let side = (pos_rows as f64).sqrt().round() as usize;
    if side * side != pos_rows || side % merge != 0 {
        return Err(bad(format!(
            "siglip_navit position table of {pos_rows} rows is not a square grid that \
             shuffles by {merge}"
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

/// The six resident tensors' offsets into the resident region.
#[derive(Debug, Clone)]
struct Resident {
    patch_w: u64,
    patch_b: u64,
    pos_embed: u64,
    norm_w: u64,
    norm_b: u64,
    proj_w: u64,
}

impl Resident {
    fn resolve(index: &ResidentIndex, shape: &VisionShape) -> Result<Self, RealForwardError> {
        let h = shape.hidden;
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        Ok(Self {
            patch_w: fp16_view(index, &name("patch_embed.weight"), h * shape.patch_dim)?,
            patch_b: fp16_view(index, &name("patch_embed.bias"), h)?,
            pos_embed: fp16_view(index, &name("pos_embed.weight"), shape.pos_rows * h)?,
            norm_w: fp16_view(index, &name("post_norm.weight"), h)?,
            norm_b: fp16_view(index, &name("post_norm.bias"), h)?,
            proj_w: fp16_view(
                index,
                &name("projector.proj.weight"),
                shape.out_hidden * shape.merger_input(),
            )?,
        })
    }
}

/// The tower: its streamer, its blocks' roles and its resident offsets.
pub(crate) struct Idefics3VisionTower {
    /// Declared BEFORE `streamer` so the buffers drop before the allocations
    /// they alias.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<BlockRoles>,
    resident: Resident,
    shape: VisionShape,
}

impl Idefics3VisionTower {
    /// Open against a combined install's own resident weights.
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let shape = idefics3_shape(&arch.vision)?;
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
        // By the entry's own index, never by position.
        let mut blocks: Vec<Option<BlockRoles>> = vec![None; shape.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, shape.depth
                ))
            })?;
            let roles = resolve_block_roles(entry, entry.expert, layer.expert_stride, &shape)?;
            if slot.replace(roles).is_some() {
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
        .map_err(|e| RealForwardError::Unsupported(format!("siglip_navit streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }
        let resident = Resident::resolve(index, &shape)?;
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            resident,
            shape,
        })
    }

    /// Encode one image: `[frames * (side/scale)^2, out_hidden]` FP16 bits,
    /// frame by frame in the processor's order.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        tiles: &Idefics3Tiles,
    ) -> Result<Vec<u16>, RealForwardError> {
        let s = &self.shape;
        let per = tiles.patches_per_side * tiles.patches_per_side;
        let n_tiles = tiles.frames;
        if tiles.patches_per_side != s.pos_side || n_tiles == 0 {
            return Err(RealForwardError::Unsupported(format!(
                "the image has {n_tiles} frame(s) of {per} patches but this tower's position \
                 table covers {}",
                s.pos_rows
            )));
        }
        let seq = n_tiles * per;
        if tiles.pixels.len() != seq * s.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "{} pixel values for {seq} patch rows of {}",
                tiles.pixels.len(),
                s.patch_dim
            )));
        }
        let hidden = s.hidden;
        let o = s.out_hidden;
        let wide = seq * hidden;
        let merged = s.pos_side / s.merge;
        let merged_rows = n_tiles * merged * merged;
        let unfolded_width = s.merger_input();
        let rows: Vec<half::f16> = tiles
            .pixels
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let gpu_err = RealForwardError::Gpu;
        let bytes = |elems: usize| (elems * 2) as u64;

        gpu::autorelease_pool(|| {
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

            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&rows_buf, 0),
                (weights.buffer(), weights.gpu_offset(self.resident.patch_w)),
                Some((weights.buffer(), weights.gpu_offset(self.resident.patch_b))),
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
                    (
                        weights.buffer(),
                        weights.gpu_offset(self.resident.pos_embed),
                    ),
                    (per * hidden) as u32,
                )
                .map_err(gpu_err)?;
            }
            pass.commit_and_wait();

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
                        RealForwardError::Unsupported(format!("siglip_navit block {blk} read: {e}"))
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
            }

            let pass = context.begin_pass();
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&x, 0),
                (weights.buffer(), weights.gpu_offset(self.resident.norm_w)),
                (weights.buffer(), weights.gpu_offset(self.resident.norm_b)),
                (&x, 0),
                seq as u32,
                hidden as u32,
                POST_LAYER_NORM_EPS,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            let tower_out = read(&x, wide);

            // Data movement only, done on the host between two passes. The
            // shuffled rows hold the same `wide` values, `hidden * merge^2`
            // to a row.
            let mut shuffled = Vec::with_capacity(wide);
            for frame in tower_out.chunks_exact(per * hidden) {
                let rows = idefics3_pixel_shuffle(frame, s.pos_side, hidden, s.merge)
                    .map_err(|e| RealForwardError::Unsupported(format!("pixel shuffle: {e}")))?;
                shuffled.extend(rows.into_iter().map(half::f16::from_bits));
            }
            let shuffled_buf = context.new_buffer_with_data(&shuffled);
            let out = context.new_output_buffer(bytes(merged_rows * o));
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&shuffled_buf, 0),
                (weights.buffer(), weights.gpu_offset(self.resident.proj_w)),
                None,
                (&out, 0),
                merged_rows as u32,
                unfolded_width as u32,
                o as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            Ok(read(&out, merged_rows * o))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mlx-community/SmolVLM-Instruct-4bit`'s tower.
    fn config() -> VisionConfig {
        VisionConfig {
            depth: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 1,
            in_channels: 3,
            spatial_merge_size: 3,
            num_position_embeddings: 729,
            out_hidden_size: 2048,
            mrope_section: [0, 0, 0],
            deepstack_visual_indexes: Vec::new(),
            vision_start_token_id: 0,
            vision_end_token_id: 0,
            image_token_id: 49153,
            video_token_id: 0,
            tower: VisionTowerKind::SigLipNaViT,
        }
    }

    #[test]
    fn the_published_shapes_resolve_and_the_wrong_ones_refuse() {
        let s = idefics3_shape(&config()).unwrap();
        assert_eq!((s.head_dim, s.pos_side, s.merger_input()), (72, 27, 10368));
        let mut c = config();
        c.tower = VisionTowerKind::SigLip;
        assert!(idefics3_shape(&c).is_err());
        let mut c = config();
        c.spatial_merge_size = 2; // 27 % 2
        assert!(idefics3_shape(&c).is_err());
        let mut c = config();
        c.num_position_embeddings = 730;
        assert!(idefics3_shape(&c).is_err());
    }
}
