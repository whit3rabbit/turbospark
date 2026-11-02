//! The CLIP vision tower LLaVA-1.5 carries (`docs/LLAVA_VISION_PHASE0.md`).
//!
//! A sibling of the qwen tower ([`super::VisionTower`]) and of Gemma 4's
//! ([`super::gemma4`]), not a variant of either: the qwen tower's host prep is
//! built around rope rows, an interpolated position table and a merger, and
//! none of those exist here. What it does share is the weight plumbing. The
//! blocks stream from `packed_vision/` through the same pread streamer, in the
//! qwen tower's twelve-role blob (the repack fuses q, k and v into one `qkv`,
//! so `weights::resolve_block_roles` and its role table apply unchanged), and
//! the residents are `vision.`-prefixed FP16 behind `weights::fp16_view`.
//!
//! # The forward pass
//!
//! ```text
//! x      = [cls ; patch_embed(rows)] + pos           # [1 + 576, 1024]
//! x      = ln_pre(x)                                 # eps 1e-5
//! for blk in 0..23:                                  # NOT 24, see below
//!     x += attn(ln1(x))                              # bidirectional, no rope
//!     x += fc2(quick_gelu(fc1(ln2(x))))
//! out    = linear_2(gelu_erf(linear_1(x[1..])))      # class row dropped
//! ```
//!
//! **Twenty-three blocks.** LLaVA-1.5 reads `hidden_states[-2]`, the output of
//! the second-to-last block, so the checkpoint's 24th block and its
//! `post_layernorm` are never run. The repack does not pack them.
//!
//! **No rope and no mask.** CLIP attends over all 577 rows, class row
//! included, with the plain `head_dim ** -0.5` scale. The position signal is
//! the learned absolute table added once at the start.
//!
//! **The class row is dropped AFTER the last block and BEFORE the
//! projector**, which is `vision_feature_select_strategy = "default"`. The
//! 576 rows the projector sees are the patches in raster order, which is the
//! order the trunk's placeholder run consumes them in.
//!
//! The grid is fixed: the position table has 577 rows, and the processor
//! always crops to 336x336, so there is nothing to interpolate and an input
//! with a different patch count is refused rather than resized.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};

use super::shape::VisionShape;
use super::weights::{fp16_view, resolve_block_roles, BlockRoles, Role, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// CLIP's LayerNorm epsilon (`CLIPVisionConfig.layer_norm_eps`). A kind
/// CONSTANT beside the qwen tower's 1e-6, for the reason that one is: a float
/// in the manifest would not round-trip (AGENTS.md Gotcha 24).
const CLIP_LAYER_NORM_EPS: f32 = 1e-5;

/// The residual stream at three points, for the cross-engine parity gate.
/// Each is `[1 + patches, hidden]` FP16 bits (the class row first).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClipStages {
    /// After the patch embedding, the class row, the position add and
    /// `ln_pre`: `hidden_states[0]` in the reference.
    pub embed: Vec<u16>,
    /// After block 0: `hidden_states[1]`.
    pub block_first: Vec<u16>,
    /// After the last block that runs: `hidden_states[feature_layer]`.
    pub block_last: Vec<u16>,
}

/// The nine resident tensors' offsets into the resident region.
#[derive(Debug, Clone)]
struct ClipResident {
    patch_w: u64,
    class_embedding: u64,
    pos_embed: u64,
    ln_pre_w: u64,
    ln_pre_b: u64,
    proj1_w: u64,
    proj1_b: u64,
    proj2_w: u64,
    proj2_b: u64,
}

impl ClipResident {
    fn resolve(index: &ResidentIndex, shape: &VisionShape) -> Result<Self, RealForwardError> {
        let h = shape.hidden;
        let o = shape.out_hidden;
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        Ok(Self {
            patch_w: fp16_view(index, &name("patch_embed.proj.weight"), h * shape.patch_dim)?,
            class_embedding: fp16_view(index, &name("class_embedding"), h)?,
            pos_embed: fp16_view(index, &name("pos_embed.weight"), shape.pos_rows * h)?,
            ln_pre_w: fp16_view(index, &name("ln_pre.weight"), h)?,
            ln_pre_b: fp16_view(index, &name("ln_pre.bias"), h)?,
            proj1_w: fp16_view(index, &name("projector.linear_1.weight"), o * h)?,
            proj1_b: fp16_view(index, &name("projector.linear_1.bias"), o)?,
            proj2_w: fp16_view(index, &name("projector.linear_2.weight"), o * o)?,
            proj2_b: fp16_view(index, &name("projector.linear_2.bias"), o)?,
        })
    }
}

/// The tower's dimensions, in the qwen tower's [`VisionShape`] so the block
/// role resolver is shared. `merge` is 1, there is no deepstack, and
/// `pos_side` is the PATCH grid's side (the class row is `pos_rows - 1`).
fn clip_shape(vision: &VisionConfig) -> Result<VisionShape, RealForwardError> {
    let bad = |detail: String| RealForwardError::Unsupported(detail);
    if vision.tower != VisionTowerKind::Clip || !vision.is_active() {
        return Err(bad("this install declares no CLIP vision tower".to_string()));
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
            return Err(bad(format!("CLIP config field {label} is zero")));
        }
    }
    if (vision.temporal_patch_size, vision.spatial_merge_size, vision.in_channels) != (1, 1, 3) {
        return Err(bad(format!(
            "CLIP tower with temporal {}, merge {}, channels {}; need 1, 1, 3",
            vision.temporal_patch_size, vision.spatial_merge_size, vision.in_channels
        )));
    }
    if hidden % heads != 0 {
        return Err(bad(format!(
            "CLIP hidden_size {hidden} is not divisible by num_heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    if head_dim > gpu::MAX_ATTENTION_HEAD_DIM as usize {
        return Err(bad(format!(
            "CLIP head_dim {head_dim} exceeds the attention kernel's tile ({})",
            gpu::MAX_ATTENTION_HEAD_DIM
        )));
    }
    let side = ((pos_rows - 1) as f64).sqrt().round() as usize;
    if side * side != pos_rows - 1 {
        return Err(bad(format!(
            "CLIP position table of {pos_rows} rows is not a class row plus a square grid"
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
        pos_side: side,
        patch_size: patch,
        deepstack: Vec::new(),
    })
}

/// The CLIP tower: its streamer, its blocks' roles and its resident offsets.
pub(crate) struct ClipVisionTower {
    /// Declared BEFORE `streamer` so the buffers drop before the allocations
    /// they alias.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<BlockRoles>,
    resident: ClipResident,
    shape: VisionShape,
    /// The class embedding, FP16 bytes, copied out at open: it is written
    /// into row 0 of the residual stream from the host, since no kernel
    /// scatters one row.
    class_row: Vec<u8>,
    /// The sidecar's own resident weights, `Some` when the tower was opened
    /// against a standalone sidecar rather than the trunk's install.
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
}

impl ClipVisionTower {
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
    /// mapping, exactly as [`super::VisionTower::open_with_sidecar`] does.
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
        let shape = clip_shape(&arch.vision)?;
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
        .map_err(|e| RealForwardError::Unsupported(format!("CLIP vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }

        let resident = ClipResident::resolve(index, &shape)?;
        let at = resident.class_embedding as usize;
        let class_row = weights
            .data()
            .get(at..at + shape.hidden * 2)
            .ok_or_else(|| {
                RealForwardError::Unsupported(
                    "CLIP class embedding spans past the mapped region".to_string(),
                )
            })?
            .to_vec();
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            resident,
            shape,
            class_row,
            sidecar_weights: None,
        })
    }

    /// Whether this tower is sidecar-backed (test-only engagement proof).
    #[allow(dead_code)]
    pub(crate) fn is_sidecar(&self) -> bool {
        self.sidecar_weights.is_some()
    }

    /// Encode one image to `[patches, out_hidden]` FP16 bits.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        patches: &turbospark_vision_io::ClipPatches,
        want_stages: bool,
    ) -> Result<(Vec<u16>, Option<ClipStages>), RealForwardError> {
        let s = &self.shape;
        let n = patches.num_patches();
        if n != s.pos_rows - 1 {
            return Err(RealForwardError::Unsupported(format!(
                "the image has {n} patches but this tower's position table covers {}; CLIP \
                 does not interpolate, and the processor always crops to one size",
                s.pos_rows - 1
            )));
        }
        if patches.pixels.len() != n * s.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "the image's patch rows are {}-wide but this tower's patch embedding takes {}",
                patches.pixels.len() / n.max(1),
                s.patch_dim
            )));
        }
        let tokens = n + 1;
        let hidden = s.hidden;
        let wide = tokens * hidden;
        let rows: Vec<half::f16> = patches
            .pixels
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let gpu_err = RealForwardError::Gpu;
        let bytes = |elems: usize| (elems * 2) as u64;

        let stages_out = gpu::autorelease_pool(|| {
            let rows_buf = context.new_buffer_with_data(&rows);
            let x = context.new_output_buffer(bytes(wide));
            let normed = context.new_output_buffer(bytes(wide));
            let q = context.new_output_buffer(bytes(wide));
            let k = context.new_output_buffer(bytes(wide));
            let v = context.new_output_buffer(bytes(wide));
            let attn = context.new_output_buffer(bytes(wide));
            let proj = context.new_output_buffer(bytes(wide));
            let h1 = context.new_output_buffer(bytes(tokens * s.intermediate));
            let mid = context.new_output_buffer(bytes(n * s.out_hidden));
            let out = context.new_output_buffer(bytes(n * s.out_hidden));
            let read = |b: &gpu::MetalBuffer| -> Vec<u16> {
                gpu::read_buffer_f16(b, 0, wide)
                    .into_iter()
                    .map(|v| v.to_bits())
                    .collect()
            };
            let mut stages = want_stages.then(ClipStages::default);

            // Embedding. The patch rows land at row 1 onward; row 0 is the
            // class embedding, written from the host before the pass is
            // committed (the same moment-of-write argument the trunk's embed
            // blit relies on). The position table then adds over all 577
            // rows, and `ln_pre` runs in place (each thread reads and writes
            // only its own elements after the row reductions finish).
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&rows_buf, 0),
                (w.buffer(), w.gpu_offset(self.resident.patch_w)),
                None,
                (&x, bytes(hidden)),
                n as u32,
                s.patch_dim as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;
            gpu::write_buffer_bytes(&x, 0, &self.class_row);
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&x, 0),
                (w.buffer(), w.gpu_offset(self.resident.pos_embed)),
                wide as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&x, 0),
                (w.buffer(), w.gpu_offset(self.resident.ln_pre_w)),
                (w.buffer(), w.gpu_offset(self.resident.ln_pre_b)),
                (&x, 0),
                tokens as u32,
                hidden as u32,
                CLIP_LAYER_NORM_EPS,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            if let Some(st) = stages.as_mut() {
                st.embed = read(&x);
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
                tokens,
            };
            for blk in 0..s.depth {
                let slot = blk % VISION_SLOTS;
                self.streamer
                    .load_expert_into_slot(0, blk, slot)
                    .map_err(|e| {
                        RealForwardError::Unsupported(format!("CLIP vision block {blk} read: {e}"))
                    })?;
                let pass = context.begin_pass();
                encode_clip_block(
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
                        st.block_first = read(&x);
                    }
                    if blk + 1 == s.depth {
                        st.block_last = read(&x);
                    }
                }
            }

            // The projector over the 576 patch rows: the class row is the
            // first row of `x`, so skipping it is a byte offset.
            let pass = context.begin_pass();
            let o = s.out_hidden;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&x, bytes(hidden)),
                (w.buffer(), w.gpu_offset(self.resident.proj1_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.proj1_b))),
                (&mid, 0),
                n as u32,
                hidden as u32,
                o as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_gelu(context, &pass, (&mid, 0), (n * o) as u32, gpu::GeluKind::Erf)
                .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&mid, 0),
                (w.buffer(), w.gpu_offset(self.resident.proj2_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.proj2_b))),
                (&out, 0),
                n as u32,
                o as u32,
                o as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();

            let rows: Vec<u16> = gpu::read_buffer_f16(&out, 0, n * o)
                .into_iter()
                .map(|v| v.to_bits())
                .collect();
            Ok::<_, RealForwardError>((rows, stages))
        })?;
        Ok(stages_out)
    }
}

/// The per-run buffers one block reads and writes.
struct BlockScratch<'a> {
    x: &'a gpu::MetalBuffer,
    normed: &'a gpu::MetalBuffer,
    q: &'a gpu::MetalBuffer,
    k: &'a gpu::MetalBuffer,
    v: &'a gpu::MetalBuffer,
    attn: &'a gpu::MetalBuffer,
    proj: &'a gpu::MetalBuffer,
    h1: &'a gpu::MetalBuffer,
    tokens: usize,
}

/// One CLIP block: `x += attn(ln1(x)); x += fc2(quick_gelu(fc1(ln2(x))))`.
///
/// The qwen block (`block.rs`) minus its rope pair, with this kind's epsilon
/// and activation. **Rope is not skipped by a flag: it is not here**, because
/// a rope on a CLIP tower would keep every shape and rotate the position
/// information the learned table already carries.
fn encode_clip_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    slot: &gpu::MetalBuffer,
    roles: &BlockRoles,
    b: &BlockScratch<'_>,
    shape: &VisionShape,
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let seq = b.tokens as u32;
    let hidden = shape.hidden as u32;
    let inter = shape.intermediate as u32;
    let wide = (b.tokens * shape.hidden) as u32;
    let at = |role: Role| roles.at(role);

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (b.x, 0),
        (slot, at(Role::Ln1W)),
        (slot, at(Role::Ln1B)),
        (b.normed, 0),
        seq,
        hidden,
        CLIP_LAYER_NORM_EPS,
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
            seq,
            hidden,
            hidden,
        )
        .map_err(gpu_err)?;
    }

    gpu::encode_vision_attention(
        context,
        pass,
        (b.q, 0),
        (b.k, 0),
        (b.v, 0),
        (b.attn, 0),
        seq,
        shape.heads as u32,
        shape.head_dim as u32,
        shape.attention_scale(),
    )
    .map_err(gpu_err)?;

    gpu::encode_vision_matmul(
        context,
        pass,
        (b.attn, 0),
        (slot, at(Role::ProjW)),
        Some((slot, at(Role::ProjB))),
        (b.proj, 0),
        seq,
        hidden,
        hidden,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_residual_add(context, pass, (b.x, 0), (b.proj, 0), wide)
        .map_err(gpu_err)?;

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (b.x, 0),
        (slot, at(Role::Ln2W)),
        (slot, at(Role::Ln2B)),
        (b.normed, 0),
        seq,
        hidden,
        CLIP_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (b.normed, 0),
        (slot, at(Role::Fc1W)),
        Some((slot, at(Role::Fc1B))),
        (b.h1, 0),
        seq,
        hidden,
        inter,
    )
    .map_err(gpu_err)?;
    // QUICK here: `x * sigmoid(1.702 x)`, CLIP's `hidden_act`. Neither the
    // qwen tower's tanh nor the merger's erf, and unlike that pair it is far
    // enough from both for a parity bound to see.
    gpu::encode_vision_gelu(
        context,
        pass,
        (b.h1, 0),
        seq * inter,
        gpu::GeluKind::Quick,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (b.h1, 0),
        (slot, at(Role::Fc2W)),
        Some((slot, at(Role::Fc2B))),
        (b.proj, 0),
        seq,
        inter,
        hidden,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_residual_add(context, pass, (b.x, 0), (b.proj, 0), wide)
        .map_err(gpu_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VisionConfig {
        VisionConfig {
            depth: 23,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 1,
            in_channels: 3,
            spatial_merge_size: 1,
            num_position_embeddings: 577,
            out_hidden_size: 4096,
            mrope_section: [0, 0, 0],
            deepstack_visual_indexes: Vec::new(),
            vision_start_token_id: 0,
            vision_end_token_id: 0,
            image_token_id: 32000,
            video_token_id: 0,
            tower: VisionTowerKind::Clip,
        }
    }

    #[test]
    fn the_llava_shape_resolves_with_a_24_wide_grid() {
        let s = clip_shape(&config()).unwrap();
        assert_eq!((s.hidden, s.heads, s.head_dim), (1024, 16, 64));
        assert_eq!(s.patch_dim, 588);
        assert_eq!(s.pos_side, 24);
        assert_eq!(s.pos_rows - 1, 576);
    }

    #[test]
    fn a_shape_the_tower_cannot_run_is_refused() {
        let mut c = config();
        c.tower = VisionTowerKind::QwenSiglip;
        assert!(clip_shape(&c).is_err());
        let mut c = config();
        c.num_position_embeddings = 580; // not class + square
        assert!(clip_shape(&c).is_err());
        let mut c = config();
        c.spatial_merge_size = 2;
        assert!(clip_shape(&c).is_err());
        let mut c = config();
        c.hidden_size = 1000; // not divisible by 16
        assert!(clip_shape(&c).is_err());
    }
}
