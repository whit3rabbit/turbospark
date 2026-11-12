//! The SigLIP tower LLaVA-OneVision carries
//! (`docs/LLAVA_ONEVISION_VISION_PHASE0.md`).
//!
//! The DeepSeek-VL2 tower's blocks ([`super::siglip`]) with a different head:
//!
//! ```text
//! x   = patch_embed(rows) + pos                      # per tile [729, 1152]
//! for blk in 0..26:                                  # no pre-norm, no class row
//!     x += attn(ln1(x))                              # eps 1e-6, no rope, no mask
//!     x += fc2(gelu_tanh(fc1(ln2(x))))               # eps 1e-6
//! out = linear_2(gelu_erf(linear_1(x)))              # every row; NO final norm,
//!                                                    # NO unfold: [tiles * 729, 3584]
//! ```
//!
//! Every tile of an image (the base tile and the anyres grid) runs in one
//! pass, attention dispatched per tile at a byte offset, so each 30 MiB block
//! is read from disk once per IMAGE. The projected rows are then packed on the
//! host ([`turbospark_vision_io::ov_pack_features`]): the base tile's 729
//! rows, then the grid's rows unpadded, pooled when over budget, with the
//! learned `image_newline` after every row.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{ov_pack_features, ov_plan, ClipPatches, OV_TILE_PATCHES};

use super::clip::ClipStages;
use super::shape::VisionShape;
use super::siglip::{encode_siglip_block, BlockScratch};
use super::weights::{fp16_view, resolve_block_roles, BlockRoles, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// The resident tensors' offsets into the resident region.
#[derive(Debug, Clone)]
struct OneVisionResident {
    patch_w: u64,
    patch_b: u64,
    pos_embed: u64,
    proj1_w: u64,
    proj1_b: u64,
    proj2_w: u64,
    proj2_b: u64,
    image_newline: u64,
}

impl OneVisionResident {
    fn resolve(index: &ResidentIndex, shape: &VisionShape) -> Result<Self, RealForwardError> {
        let h = shape.hidden;
        let o = shape.out_hidden;
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        Ok(Self {
            patch_w: fp16_view(index, &name("patch_embed.proj.weight"), h * shape.patch_dim)?,
            patch_b: fp16_view(index, &name("patch_embed.proj.bias"), h)?,
            pos_embed: fp16_view(index, &name("pos_embed.weight"), shape.pos_rows * h)?,
            proj1_w: fp16_view(index, &name("projector.layers.0.weight"), o * h)?,
            proj1_b: fp16_view(index, &name("projector.layers.0.bias"), o)?,
            proj2_w: fp16_view(index, &name("projector.layers.2.weight"), o * o)?,
            proj2_b: fp16_view(index, &name("projector.layers.2.bias"), o)?,
            image_newline: fp16_view(index, &name("image_newline"), o)?,
        })
    }
}

/// The tower's dimensions in the qwen tower's [`VisionShape`], so the block
/// role resolver is shared. `merge` is 1: nothing is folded.
fn onevision_shape(vision: &VisionConfig) -> Result<VisionShape, RealForwardError> {
    let bad = |detail: String| RealForwardError::Unsupported(detail);
    if vision.tower != VisionTowerKind::SigLipOv || !vision.is_active() {
        return Err(bad(
            "this install declares no OneVision SigLIP tower".to_string()
        ));
    }
    let u = |v: i64| usize::try_from(v).unwrap_or(0);
    let (depth, hidden, inter, heads) = (
        u(vision.depth),
        u(vision.hidden_size),
        u(vision.intermediate_size),
        u(vision.num_heads),
    );
    let (patch, out_hidden, pos_rows) = (
        u(vision.patch_size),
        u(vision.out_hidden_size),
        u(vision.num_position_embeddings),
    );
    for (label, value) in [
        ("depth", depth),
        ("hidden_size", hidden),
        ("intermediate_size", inter),
        ("num_heads", heads),
        ("patch_size", patch),
        ("out_hidden_size", out_hidden),
        ("num_position_embeddings", pos_rows),
    ] {
        if value == 0 {
            return Err(bad(format!("OneVision config field {label} is zero")));
        }
    }
    if (
        vision.temporal_patch_size,
        vision.in_channels,
        vision.spatial_merge_size,
    ) != (1, 3, 1)
    {
        return Err(bad(format!(
            "OneVision tower with temporal {}, channels {}, merge {}; need 1, 3, 1",
            vision.temporal_patch_size, vision.in_channels, vision.spatial_merge_size
        )));
    }
    if pos_rows != OV_TILE_PATCHES || patch != 14 {
        return Err(bad(format!(
            "OneVision tower with patch {patch} and {pos_rows} positions; the tiling is 14 px \
             patches, {OV_TILE_PATCHES} per tile"
        )));
    }
    if hidden % heads != 0 {
        return Err(bad(format!(
            "OneVision hidden_size {hidden} is not divisible by num_heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    if head_dim > gpu::MAX_ATTENTION_HEAD_DIM as usize {
        return Err(bad(format!(
            "OneVision head_dim {head_dim} exceeds the attention kernel's tile ({})",
            gpu::MAX_ATTENTION_HEAD_DIM
        )));
    }
    Ok(VisionShape {
        depth,
        hidden,
        intermediate: inter,
        heads,
        head_dim,
        merge: 1,
        out_hidden,
        patch_dim: 3 * patch * patch,
        pos_rows,
        pos_side: 27,
        patch_size: patch,
        deepstack: Vec::new(),
    })
}

/// The OneVision tower: its streamer, its blocks' roles and its resident
/// offsets.
pub(crate) struct OneVisionTower {
    /// Declared BEFORE `streamer` so the buffers drop before the allocations
    /// they alias.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<BlockRoles>,
    resident: OneVisionResident,
    shape: VisionShape,
    /// `image_newline`, FP16 bits, copied out at open: the packing is a host
    /// assembly.
    newline: Vec<u16>,
    /// The sidecar's own resident weights, `Some` when opened against a
    /// standalone sidecar rather than the trunk's install.
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
}

impl OneVisionTower {
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
    /// mapping, exactly as [`super::siglip::SiglipVisionTower::open_with_sidecar`]
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
        let shape = onevision_shape(&arch.vision)?;
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
        .map_err(|e| RealForwardError::Unsupported(format!("OneVision vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }

        let resident = OneVisionResident::resolve(index, &shape)?;
        let at = resident.image_newline as usize;
        let newline = weights
            .data()
            .get(at..at + shape.out_hidden * 2)
            .ok_or_else(|| {
                RealForwardError::Unsupported(
                    "OneVision image_newline spans past the mapped region".to_string(),
                )
            })?
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            resident,
            shape,
            newline,
            sidecar_weights: None,
        })
    }

    /// Whether this tower is sidecar-backed (test-only engagement proof).
    #[allow(dead_code)]
    pub(crate) fn is_sidecar(&self) -> bool {
        self.sidecar_weights.is_some()
    }

    /// Encode one image to `[image_tokens, out_hidden]` FP16 bits: the whole
    /// prompt run, newline rows included. The stages, when asked for, are
    /// `[tiles * 729, hidden]` after the embedding, block 0 and the last block.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        patches: &ClipPatches,
        want_stages: bool,
    ) -> Result<(Vec<u16>, Option<ClipStages>), RealForwardError> {
        let s = &self.shape;
        let anyres = patches.anyres.ok_or_else(|| {
            RealForwardError::Unsupported(
                "the OneVision tower needs anyres patches (base tile plus grid)".to_string(),
            )
        })?;
        let n_tiles = 1 + anyres.tiles_h * anyres.tiles_w;
        let per = OV_TILE_PATCHES;
        let seq = n_tiles * per;
        if patches.pixels.len() != seq * s.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "{} patch values for {n_tiles} tiles of {per} rows of {}",
                patches.pixels.len(),
                s.patch_dim
            )));
        }
        let plan = ov_plan(anyres.orig_h, anyres.orig_w, anyres.tiles_h, anyres.tiles_w);
        if plan.total_rows() != patches.grid * patches.grid_w {
            return Err(RealForwardError::Unsupported(format!(
                "the packing plan has {} rows but the prompt reserved {}",
                plan.total_rows(),
                patches.grid * patches.grid_w
            )));
        }
        let hidden = s.hidden;
        let o = s.out_hidden;
        let wide = seq * hidden;
        let rows: Vec<half::f16> = patches
            .pixels
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let gpu_err = RealForwardError::Gpu;
        let bytes = |elems: usize| (elems * 2) as u64;

        let (projected, stages) = gpu::autorelease_pool(|| {
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
            let mut stages = want_stages.then(ClipStages::default);

            // Embedding: every tile's patch rows through the conv-as-GEMM
            // (with its bias), then each tile's own copy of the position table.
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
                            "OneVision vision block {blk} read: {e}"
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

            // The projector on every row; there is no final norm to run first.
            let mid = context.new_output_buffer(bytes(seq * o));
            let out = context.new_output_buffer(bytes(seq * o));
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&x, 0),
                (w.buffer(), w.gpu_offset(self.resident.proj1_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.proj1_b))),
                (&mid, 0),
                seq as u32,
                hidden as u32,
                o as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&mid, 0),
                (seq * o) as u32,
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
                seq as u32,
                o as u32,
                o as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            Ok::<_, RealForwardError>((read(&out, seq * o), stages))
        })?;

        // Host assembly in f32: the base tile, then the unpadded, pooled grid
        // with the learned newline after every row.
        let f32s = |bits: &[u16]| -> Vec<f32> {
            bits.iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect()
        };
        let packed = ov_pack_features(&f32s(&projected), o, &f32s(&self.newline), &plan)
            .map_err(|e| RealForwardError::Unsupported(format!("OneVision packing: {e}")))?;
        let packed = packed
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        Ok((packed, stages))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VisionConfig {
        VisionConfig {
            depth: 26,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 1,
            in_channels: 3,
            spatial_merge_size: 1,
            num_position_embeddings: 729,
            out_hidden_size: 3584,
            mrope_section: [0, 0, 0],
            deepstack_visual_indexes: Vec::new(),
            vision_start_token_id: 0,
            vision_end_token_id: 0,
            image_token_id: 151646,
            video_token_id: 151647,
            tower: VisionTowerKind::SigLipOv,
        }
    }

    #[test]
    fn the_onevision_shape_resolves_with_no_unfold() {
        let s = onevision_shape(&config()).unwrap();
        assert_eq!((s.depth, s.hidden, s.head_dim, s.merge), (26, 1152, 72, 1));
        // The projector's first layer reads one hidden row, not four.
        assert_eq!(s.merger_input(), 1152);
        assert_eq!((s.pos_rows, s.pos_side, s.patch_dim), (729, 27, 588));
    }

    #[test]
    fn a_shape_the_tower_cannot_run_is_refused() {
        let mut c = config();
        c.spatial_merge_size = 2;
        assert!(onevision_shape(&c).is_err());
        let mut c = config();
        c.num_position_embeddings = 576;
        assert!(onevision_shape(&c).is_err());
        let mut c = config();
        c.tower = VisionTowerKind::SigLip;
        assert!(onevision_shape(&c).is_err());
        let mut c = config();
        c.hidden_size = 1150;
        assert!(onevision_shape(&c).is_err());
    }
}
