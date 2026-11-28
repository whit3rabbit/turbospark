//! The InternViT vision tower the InternVL wrappers carry
//! (`docs/INTERNVL_VISION_PHASE0.md`).
//!
//! A sibling of the CLIP tower ([`super::clip`]) it structurally most
//! resembles: a class row, learned absolute positions, bidirectional
//! attention with biases, LayerNorm blocks, no rope, no mask. It differs in
//! four ways, each its own kernel or refusal:
//!
//! - **Layer scale.** Every block scales its attention and MLP outputs by a
//!   per-block learned `[hidden]` vector (`ls1`/`ls2` on the
//!   `internvl_chat` flavor, `lambda_1`/`lambda_2` on the `internvl` one)
//!   before adding it to the residual. The repack packs them as the
//!   fourteen-role blob (the qwen tower's twelve plus two), and the residual
//!   add is `vision_residual_scaled_add_fp16`.
//! - **Activations.** The block MLP is `nn.GELU(approx="precise")` in
//!   mlx-vlm, which is the TANH form (mlx spells the exact erf one
//!   `approx="none"`), for BOTH wrappers -- the `internvl` wrapper imports
//!   the same `MLP` class. The projector's activation is per flavor:
//!   `nn.GELU()` (exact erf) in `internvl_chat`'s `mlp1`, tanh in
//!   `internvl`'s `MultiModalProjector`. The published wrappers pair
//!   wrapper-flavor to trunk one to one (`internvl_chat` over `qwen2`,
//!   `internvl` over `qwen3`), so the projector arm keys on the trunk
//!   FAMILY as the flavor proxy, and a future wrapper that breaks that
//!   pairing takes the tanh arm plus a deviation note rather than a
//!   silent wrong activation on a gated path.
//! - **No pre-embedding norm and no final norm.** The checkpoint carries
//!   neither (verified against both published indexes); the position add
//!   feeds block 0 directly and the last block feeds the projector.
//! - **Pixel shuffle before the projector.** The class row is dropped, each
//!   tile's patch rows are shuffled into `merge x merge`-smaller rows of
//!   `merge^2 * hidden` channels (a pure permutation, done on the host over
//!   FP16 bits), and the projector's LayerNorm runs on the wide row.
//!
//! Tiles are separate runs: one image's dynamic tiles never attend across
//! each other; the tower runs per tile and the rows concatenate in tile
//! order, which is the order the single `<img> ... </img>` span consumes.
//!
//! The grid is fixed: the position table has `(image_size / patch_size)^2 + 1`
//! rows, and the dynamic tiling always feeds tiles of exactly the tower's own
//! `image_size` edge, so an input with a different patch count is refused
//! rather than interpolated (the CLIP rule; the reference interpolates for
//! fine-tuned resolutions this port has not met in the wild).

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};

use super::weights::{fp16_view, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;
use turbospark_vision_io::InternVlTiles;

/// The InternViT block's fourteen roles, in forward-pass order, as the
/// repack's `INTERNVL_BLOCK_ROLES` spells them. Restated here rather than
/// imported: `crates/repack` is a dev-dependency of this crate (the qwen
/// role table's reason), and the runtime never reads a repack constant.
const BLOCK_ROLES: [&str; 14] = [
    "ln1_w", "ln1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "ln2_w", "ln2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b", "ls1", "ls2",
];

/// The install names of the tower's residents, as the repack writes them.
const CLS_TOKEN: &str = "cls_token";
const PATCH_W: &str = "patch_embed.weight";
const PATCH_B: &str = "patch_embed.bias";
const POS_EMBED: &str = "pos_embed.weight";
/// The projector's install-name level, the LFM3 projector's.
const PROJECTOR_LEVEL: &str = "projector.";

/// InternViT's LayerNorm epsilon (`vision_config.layer_norm_eps`, 1e-6 on
/// every published config; a kind constant for the manifest's float-free
/// rule, AGENTS.md Gotcha 24).
const INTERNVL_LAYER_NORM_EPS: f32 = 1e-6;

/// One block's fourteen sub-tensor offsets, relative to the blob's start.
#[derive(Debug, Clone)]
struct InternVlBlockRoles {
    offsets: [u64; 14],
}

impl InternVlBlockRoles {
    fn at(&self, role: usize) -> u64 {
        self.offsets[role]
    }
}

/// Expected byte size of a block role given the tower's shape.
fn role_size(role: &str, hidden: usize, intermediate: usize) -> Option<u64> {
    let h = hidden as u64;
    let inter = intermediate as u64;
    let elems = match role {
        "ln1_w" | "ln1_b" | "ln2_w" | "ln2_b" | "proj_b" | "fc2_b" | "ls1" | "ls2" => h,
        "qkv_w" => 3 * h * h,
        "qkv_b" => 3 * h,
        "proj_w" => h * h,
        "fc1_w" => inter * h,
        "fc1_b" => inter,
        "fc2_w" => h * inter,
        _ => return None,
    };
    Some(elems * 2)
}

fn resolve_block_roles(
    entry: &model_io::ExpertEntry,
    block: usize,
    stride: u64,
    hidden: usize,
    intermediate: usize,
) -> Result<InternVlBlockRoles, RealForwardError> {
    let mut offsets = [0u64; 14];
    for (i, role) in BLOCK_ROLES.iter().enumerate() {
        let sub = entry.sub_tensors.get(*role).ok_or_else(|| {
            RealForwardError::MissingTensor(format!(
                "vision block {block} is missing the sub-tensor {role}"
            ))
        })?;
        if sub.dtype != "fp16" {
            return Err(RealForwardError::Unsupported(format!(
                "vision block {block} role {role} carries dtype {:?}, but the tower's kernels \
                 bind FP16; a same-width type would be read as a different number",
                sub.dtype
            )));
        }
        if let Some(expected) = role_size(role, hidden, intermediate) {
            if sub.size != expected {
                return Err(RealForwardError::Unsupported(format!(
                    "vision block {block} role {role} size {} does not match expected size {expected}",
                    sub.size
                )));
            }
        }
        if sub.offset % 4 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "vision block {block} role {role} sits at blob offset {}, which is not 4-byte \
                 aligned; Metal cannot bind a buffer there",
                sub.offset
            )));
        }
        if sub.offset + sub.size > stride {
            return Err(RealForwardError::Unsupported(format!(
                "vision block {block} role {role} spans {}..{} of a {stride}-byte blob",
                sub.offset,
                sub.offset + sub.size
            )));
        }
        offsets[i] = sub.offset;
    }
    Ok(InternVlBlockRoles { offsets })
}

/// The tower's dimensions, refused rather than derived where a wrong value
/// would silently change the graph. Returns
/// `(depth, hidden, intermediate, heads, head_dim, out_hidden)`.
fn internvl_shape(
    vision: &VisionConfig,
) -> Result<(usize, usize, usize, usize, usize, usize), RealForwardError> {
    let bad = |detail: String| RealForwardError::Unsupported(detail);
    if vision.tower != VisionTowerKind::InternViT || !vision.is_active() {
        return Err(bad(
            "this install declares no InternViT vision tower".to_string()
        ));
    }
    let u = |v: i64| usize::try_from(v).unwrap_or(0);
    let (depth, hidden, inter, heads) = (
        u(vision.depth),
        u(vision.hidden_size),
        u(vision.intermediate_size),
        u(vision.num_heads),
    );
    let (out_hidden, pos_rows, merge) = (
        u(vision.out_hidden_size),
        u(vision.num_position_embeddings),
        u(vision.spatial_merge_size),
    );
    for (label, value) in [
        ("depth", depth),
        ("hidden_size", hidden),
        ("intermediate_size", inter),
        ("num_heads", heads),
        ("out_hidden_size", out_hidden),
        ("num_position_embeddings", pos_rows),
        ("spatial_merge_size", merge),
    ] {
        if value == 0 {
            return Err(bad(format!("InternViT config field {label} is zero")));
        }
    }
    if (
        vision.temporal_patch_size,
        vision.in_channels,
        vision.patch_size,
    ) != (1, 3, 14)
    {
        return Err(bad(format!(
            "InternViT tower with temporal {}, channels {}, patch {}; need 1, 3, 14",
            vision.temporal_patch_size, vision.in_channels, vision.patch_size
        )));
    }
    if hidden % heads != 0 {
        return Err(bad(format!(
            "InternViT hidden_size {hidden} is not divisible by num_heads {heads}"
        )));
    }
    let head_dim = hidden / heads;
    if head_dim > gpu::MAX_ATTENTION_HEAD_DIM as usize {
        return Err(bad(format!(
            "InternViT head_dim {head_dim} exceeds the attention kernel's tile ({})",
            gpu::MAX_ATTENTION_HEAD_DIM
        )));
    }
    // Class row plus a square patch grid that shuffles whole merge blocks.
    if pos_rows < 2 || merge == 0 || merge > 4 {
        return Err(bad(format!(
            "InternViT position table of {pos_rows} rows with shuffle factor {merge} is not a \
             shape this tower runs"
        )));
    }
    let side = ((pos_rows - 1) as f64).sqrt().round() as usize;
    if side * side != pos_rows - 1 || side % merge != 0 {
        return Err(bad(format!(
            "InternViT position table of {pos_rows} rows is not a class row plus a square grid \
             that shuffles whole {merge}-wide blocks"
        )));
    }
    Ok((depth, hidden, inter, heads, head_dim, out_hidden))
}

/// The tower's resident offsets into the resident region, projector included.
#[derive(Debug, Clone)]
struct InternVlResident {
    cls_token: u64,
    patch_w: u64,
    patch_b: u64,
    /// HOST offset (into the mmap): the position table and class row are
    /// copied to host memory at open and a run composes the embedding's
    /// first row and the table add from them.
    pos_embed_host: usize,
    proj_norm_w: u64,
    proj_norm_b: u64,
    proj1_w: u64,
    proj1_b: u64,
    proj2_w: u64,
    proj2_b: u64,
}

impl InternVlResident {
    fn resolve(
        index: &ResidentIndex,
        hidden: usize,
        out_hidden: usize,
        pos_rows: usize,
        unfolded: usize,
    ) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let pos_entry = index
            .entries
            .get(&name(POS_EMBED))
            .ok_or_else(|| RealForwardError::MissingTensor(name(POS_EMBED)))?;
        let _ = fp16_view(index, &name(POS_EMBED), pos_rows * hidden)?;
        Ok(Self {
            cls_token: fp16_view(index, &name(CLS_TOKEN), hidden)?,
            patch_w: fp16_view(index, &name(PATCH_W), hidden * 588)?,
            patch_b: fp16_view(index, &name(PATCH_B), hidden)?,
            pos_embed_host: (pos_entry.file_offset - index.header.index_size) as usize,
            proj_norm_w: fp16_view(
                index,
                &name(&format!("{PROJECTOR_LEVEL}layer_norm.weight")),
                unfolded,
            )?,
            proj_norm_b: fp16_view(
                index,
                &name(&format!("{PROJECTOR_LEVEL}layer_norm.bias")),
                unfolded,
            )?,
            proj1_w: fp16_view(
                index,
                &name(&format!("{PROJECTOR_LEVEL}linear_1.weight")),
                out_hidden * unfolded,
            )?,
            proj1_b: fp16_view(
                index,
                &name(&format!("{PROJECTOR_LEVEL}linear_1.bias")),
                out_hidden,
            )?,
            proj2_w: fp16_view(
                index,
                &name(&format!("{PROJECTOR_LEVEL}linear_2.weight")),
                out_hidden * out_hidden,
            )?,
            proj2_b: fp16_view(
                index,
                &name(&format!("{PROJECTOR_LEVEL}linear_2.bias")),
                out_hidden,
            )?,
        })
    }
}

/// The InternViT tower: its streamer, its blocks' roles, its residents and
/// the position table's host copy.
pub(crate) struct InternVlVisionTower {
    /// Declared BEFORE `streamer` so the buffers drop before the allocations
    /// they alias.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<InternVlBlockRoles>,
    resident: InternVlResident,
    /// The learned position table, FP16 bits, copied out at open: a run
    /// uploads it beside the patch rows and adds it per element, and the
    /// class row plus the table's first row compose the embedding's row 0.
    pos_table: Vec<u8>,
    /// The class row, FP16 bits, written into row 0 from the host (the CLIP
    /// tower's argument: no kernel scatters one row).
    cls_row: Vec<u8>,
    depth: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    head_dim: usize,
    out_hidden: usize,
    pos_rows: usize,
    merge: usize,
    patch_dim: usize,
    /// The projector's activation: erf on the `internvl_chat` flavor (the
    /// gated one), tanh on the `internvl` one; the trunk family is the
    /// flavor proxy (see the module header).
    projector_tanh: bool,
}

impl InternVlVisionTower {
    /// Open against a combined install's own resident weights. Sidecars are
    /// refused by the caller: the sidecar pairing record names no InternViT
    /// kind yet.
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let (depth, hidden, intermediate, heads, head_dim, out_hidden) =
            internvl_shape(&arch.vision)?;
        let pos_rows = usize::try_from(arch.vision.num_position_embeddings).unwrap_or(0);
        let merge = usize::try_from(arch.vision.spatial_merge_size).unwrap_or(2);
        let unfolded = hidden * merge * merge;
        let layout = model_io::load_packed_layout_from(
            dir,
            model_io::PACKED_VISION_DIR,
            PACKED_LAYOUT_MAX_BYTES,
        )
        .map_err(RealForwardError::Model)?;
        let layer = layout.layers.iter().find(|l| l.layer == 0).ok_or_else(|| {
            RealForwardError::Unsupported("packed_vision/layout.json has no layer 0".to_string())
        })?;
        if layer.experts.len() != depth {
            return Err(RealForwardError::Unsupported(format!(
                "packed_vision carries {} blocks but the manifest declares {depth}",
                layer.experts.len()
            )));
        }
        let mut blocks: Vec<Option<InternVlBlockRoles>> = vec![None; depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {depth}",
                    entry.expert
                ))
            })?;
            if slot
                .replace(resolve_block_roles(
                    entry,
                    entry.expert,
                    layer.expert_stride,
                    hidden,
                    intermediate,
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
        .map_err(|e| RealForwardError::Unsupported(format!("InternViT vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }

        let resident = InternVlResident::resolve(index, hidden, out_hidden, pos_rows, unfolded)?;
        let pos_len = pos_rows * hidden * 2;
        let mut pos_table = vec![0u8; pos_len];
        pos_table.copy_from_slice(
            &weights.data()[resident.pos_embed_host..resident.pos_embed_host + pos_len],
        );
        let at = resident.cls_token as usize;
        let cls_row = weights
            .data()
            .get(at..at + hidden * 2)
            .ok_or_else(|| {
                RealForwardError::Unsupported(
                    "InternViT class token spans past the mapped region".to_string(),
                )
            })?
            .to_vec();
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            resident,
            pos_table,
            cls_row,
            depth,
            hidden,
            intermediate,
            heads,
            head_dim,
            out_hidden,
            pos_rows,
            merge,
            patch_dim: 588,
            // The flavor proxy: `internvl_chat` wraps a qwen2 trunk and its
            // projector is the exact erf GELU; the `internvl` wrapper's is
            // tanh. A future wrapper that breaks the flavor-to-trunk pairing
            // takes the tanh arm plus a deviation note (the module header).
            projector_tanh: arch.family == model_io::ModelFamily::Qwen3Dense,
        })
    }

    /// Encode one image (all of its tiles) to `[total_tokens, out_hidden]`
    /// FP16 bits, in tile order. One `<img> ... </img>` span consumes the
    /// whole return value.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        tiles: &InternVlTiles,
    ) -> Result<Vec<u16>, RealForwardError> {
        let bad = |detail: String| RealForwardError::Unsupported(detail);
        let patches = self.pos_rows - 1;
        if tiles.side_patches * tiles.side_patches != patches {
            return Err(bad(format!(
                "a tile has {} patches but this tower's position table covers {patches}; the \
                 dynamic tiling feeds tiles of the tower's own image size",
                tiles.side_patches * tiles.side_patches
            )));
        }
        if tiles.patch_dim != self.patch_dim {
            return Err(bad(format!(
                "a tile's patch rows are {}-wide but this tower's patch embedding takes {}",
                tiles.patch_dim, self.patch_dim
            )));
        }
        if tiles.rows.len() != tiles.tiles * patches * self.patch_dim {
            return Err(bad(format!(
                "{} tile row values for {} run(s) of {patches} patches x {}",
                tiles.rows.len(),
                tiles.tiles,
                self.patch_dim
            )));
        }
        let mut merged = Vec::new();
        for tile in 0..tiles.tiles {
            let start = tile * patches * self.patch_dim;
            let end = start + patches * self.patch_dim;
            let tile_rows: Vec<half::f16> = tiles.rows[start..end]
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
            let rows = self.run_one_tile(context, weights, &tile_rows)?;
            merged.extend_from_slice(&rows);
        }
        Ok(merged)
    }

    /// One tower run: patch embed, class row, position add, `depth` blocks,
    /// drop the class row, pixel shuffle, projector. Returns
    /// `[(pos_rows - 1) / merge^2, out_hidden]` FP16 bits.
    #[allow(clippy::too_many_lines)]
    fn run_one_tile(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        tile_rows: &[half::f16],
    ) -> Result<Vec<u16>, RealForwardError> {
        let hidden = self.hidden;
        let inter = self.intermediate;
        let out_hidden = self.out_hidden;
        let heads = self.heads as u32;
        let head_dim = self.head_dim as u32;
        let patches = self.pos_rows - 1;
        let tokens = patches + 1;
        let wide = tokens * hidden;
        let gpu_err = RealForwardError::Gpu;
        let bytes = |elems: usize| (elems * 2) as u64;
        let side = (patches as f64).sqrt().round() as usize;
        debug_assert_eq!(side * side, patches);
        let merged_rows = (side / self.merge) * (side / self.merge);

        gpu::autorelease_pool(|| {
            let rows_buf = context.new_buffer_with_data(tile_rows);
            let x = context.new_output_buffer(bytes(wide));
            let normed = context.new_output_buffer(bytes(wide));
            let q = context.new_output_buffer(bytes(wide));
            let k = context.new_output_buffer(bytes(wide));
            let v = context.new_output_buffer(bytes(wide));
            let attn = context.new_output_buffer(bytes(wide));
            let proj = context.new_output_buffer(bytes(wide));
            let h1 = context.new_output_buffer(bytes(tokens * inter));
            let pos_buf = context.new_buffer_with_data(&self.pos_table);

            // Embedding. The patch GEMM writes rows 1.. (row 0 is scratch);
            // the class row + the table's row 0 compose row 0 on the HOST
            // (one fp32 add per element, one fp16 rounding -- the same
            // rounding the GPU add applies), and the table's rows 1.. are
            // ADDED by the residual kernel over the patch rows.
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&rows_buf, 0),
                (weights.buffer(), weights.gpu_offset(self.resident.patch_w)),
                Some((weights.buffer(), weights.gpu_offset(self.resident.patch_b))),
                (&x, bytes(hidden)),
                patches as u32,
                self.patch_dim as u32,
                hidden as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();

            let cls: Vec<u16> = f16_bits(&self.cls_row);
            let pos: Vec<u16> = f16_bits(&self.pos_table);
            let mut row0 = vec![0u16; hidden];
            for (i, slot) in row0.iter_mut().enumerate() {
                let sum =
                    half::f16::from_bits(cls[i]).to_f32() + half::f16::from_bits(pos[i]).to_f32();
                *slot = half::f16::from_f32(sum).to_bits();
            }
            gpu::write_buffer_bytes(&x, 0, &le_bytes(&row0));
            let pass = context.begin_pass();
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&x, bytes(hidden)),
                (&pos_buf, bytes(hidden)),
                (patches * hidden) as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();

            // The blocks. q, k, v read the fused weight's rows at 0, hidden,
            // 2*hidden (the chat flavor's `attn.qkv` layout; the plain
            // flavor's separate projections are concatenated into it at
            // repack).
            for blk in 0..self.depth {
                let slot = blk % VISION_SLOTS;
                self.streamer
                    .load_expert_into_slot(0, blk, slot)
                    .map_err(|e| {
                        RealForwardError::Unsupported(format!(
                            "InternViT vision block {blk} read: {e}"
                        ))
                    })?;
                let roles = &self.blocks[blk];
                let slot_buf = &self.slot_buffers[slot];
                let at = |role: usize| roles.at(role);
                let pass = context.begin_pass();
                gpu::encode_vision_layer_norm(
                    context,
                    &pass,
                    (&x, 0),
                    (slot_buf, at(0)),
                    (slot_buf, at(1)),
                    (&normed, 0),
                    tokens as u32,
                    hidden as u32,
                    INTERNVL_LAYER_NORM_EPS,
                )
                .map_err(gpu_err)?;
                // q, k, v at rows 0, hidden, 2*hidden of the fused weight.
                let row_bytes = (hidden * hidden * 2) as u64;
                let bias_bytes = (hidden * 2) as u64;
                for (i, out) in [&q, &k, &v].into_iter().enumerate() {
                    gpu::encode_vision_matmul(
                        context,
                        &pass,
                        (&normed, 0),
                        (slot_buf, at(2) + i as u64 * row_bytes),
                        Some((slot_buf, at(3) + i as u64 * bias_bytes)),
                        (out, 0),
                        tokens as u32,
                        hidden as u32,
                        hidden as u32,
                    )
                    .map_err(gpu_err)?;
                }
                gpu::encode_vision_attention(
                    context,
                    &pass,
                    (&q, 0),
                    (&k, 0),
                    (&v, 0),
                    (&attn, 0),
                    tokens as u32,
                    heads,
                    head_dim,
                    (head_dim as f32).powf(-0.5),
                )
                .map_err(gpu_err)?;
                gpu::encode_vision_matmul(
                    context,
                    &pass,
                    (&attn, 0),
                    (slot_buf, at(4)),
                    Some((slot_buf, at(5))),
                    (&proj, 0),
                    tokens as u32,
                    hidden as u32,
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                // x += proj * ls1, fused (the layer-scale kernel).
                gpu::encode_vision_residual_scaled_add(
                    context,
                    &pass,
                    (&x, 0),
                    (&proj, 0),
                    (slot_buf, at(12)),
                    hidden as u32,
                    wide as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_vision_layer_norm(
                    context,
                    &pass,
                    (&x, 0),
                    (slot_buf, at(6)),
                    (slot_buf, at(7)),
                    (&normed, 0),
                    tokens as u32,
                    hidden as u32,
                    INTERNVL_LAYER_NORM_EPS,
                )
                .map_err(gpu_err)?;
                gpu::encode_vision_matmul(
                    context,
                    &pass,
                    (&normed, 0),
                    (slot_buf, at(8)),
                    Some((slot_buf, at(9))),
                    (&h1, 0),
                    tokens as u32,
                    hidden as u32,
                    inter as u32,
                )
                .map_err(gpu_err)?;
                // The block MLP's activation is the TANH form for both
                // wrappers (`nn.GELU(approx="precise")`; see the header).
                gpu::encode_vision_gelu(
                    context,
                    &pass,
                    (&h1, 0),
                    (tokens * inter) as u32,
                    gpu::GeluKind::Tanh,
                )
                .map_err(gpu_err)?;
                gpu::encode_vision_matmul(
                    context,
                    &pass,
                    (&h1, 0),
                    (slot_buf, at(10)),
                    Some((slot_buf, at(11))),
                    (&proj, 0),
                    tokens as u32,
                    inter as u32,
                    hidden as u32,
                )
                .map_err(gpu_err)?;
                gpu::encode_vision_residual_scaled_add(
                    context,
                    &pass,
                    (&x, 0),
                    (&proj, 0),
                    (slot_buf, at(13)),
                    hidden as u32,
                    wide as u32,
                )
                .map_err(gpu_err)?;
                pass.commit_and_wait();
            }

            // Drop the class row and pixel shuffle on the host: the tower's
            // output rows [1..] are [patches, hidden] FP16, and the shuffle
            // is a pure permutation of them into
            // [merged_rows, hidden * merge^2].
            let tower_rows: Vec<u16> = gpu::read_buffer_f16(&x, hidden * 2, patches * hidden)
                .into_iter()
                .map(|v| v.to_bits())
                .collect();
            let shuffled =
                turbospark_vision_io::internvl_pixel_shuffle(&tower_rows, side, hidden, self.merge)
                    .map_err(|e| {
                        RealForwardError::Unsupported(format!("InternViT pixel shuffle: {e}"))
                    })?;
            run_projector(
                context,
                &ProjectorWeights {
                    buffer: weights.buffer(),
                    norm_w: weights.gpu_offset(self.resident.proj_norm_w),
                    norm_b: weights.gpu_offset(self.resident.proj_norm_b),
                    linear_1_w: weights.gpu_offset(self.resident.proj1_w),
                    linear_1_b: weights.gpu_offset(self.resident.proj1_b),
                    linear_2_w: weights.gpu_offset(self.resident.proj2_w),
                    linear_2_b: weights.gpu_offset(self.resident.proj2_b),
                },
                &shuffled,
                merged_rows,
                out_hidden,
                if self.projector_tanh {
                    gpu::GeluKind::Tanh
                } else {
                    gpu::GeluKind::Erf
                },
            )
        })
    }
}

/// Where the projector's six tensors sit in one weight buffer.
struct ProjectorWeights<'a> {
    buffer: &'a gpu::MetalBuffer,
    norm_w: u64,
    norm_b: u64,
    linear_1_w: u64,
    linear_1_b: u64,
    linear_2_w: u64,
    linear_2_b: u64,
}

/// The projector over the pixel-shuffled rows: LayerNorm on the unfolded row,
/// Linear, GELU, Linear. Returns `[merged_rows, out_hidden]` FP16 bits.
fn run_projector(
    context: &mut gpu::MetalContext,
    w: &ProjectorWeights<'_>,
    shuffled: &[u16],
    merged_rows: usize,
    out_hidden: usize,
    gelu: gpu::GeluKind,
) -> Result<Vec<u16>, RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let unfolded = shuffled.len() / merged_rows;
    let shuffled_buf = context.new_buffer_with_data(shuffled);
    // The norm writes the UNFOLDED width (`merge^2 * hidden`), several times
    // `out_hidden`; Metal does not bounds-check a kernel's writes, so each
    // buffer is sized from the width its own writer emits.
    let normed = context.new_output_buffer((merged_rows * unfolded * 2) as u64);
    let mid = context.new_output_buffer((merged_rows * out_hidden * 2) as u64);
    let out = context.new_output_buffer((merged_rows * out_hidden * 2) as u64);
    let pass = context.begin_pass();
    gpu::encode_vision_layer_norm(
        context,
        &pass,
        (&shuffled_buf, 0),
        (w.buffer, w.norm_w),
        (w.buffer, w.norm_b),
        (&normed, 0),
        merged_rows as u32,
        unfolded as u32,
        INTERNVL_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        &pass,
        (&normed, 0),
        (w.buffer, w.linear_1_w),
        Some((w.buffer, w.linear_1_b)),
        (&mid, 0),
        merged_rows as u32,
        unfolded as u32,
        out_hidden as u32,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_gelu(
        context,
        &pass,
        (&mid, 0),
        (merged_rows * out_hidden) as u32,
        gelu,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        &pass,
        (&mid, 0),
        (w.buffer, w.linear_2_w),
        Some((w.buffer, w.linear_2_b)),
        (&out, 0),
        merged_rows as u32,
        out_hidden as u32,
        out_hidden as u32,
    )
    .map_err(gpu_err)?;
    pass.commit_and_wait();
    Ok(gpu::read_buffer_f16(&out, 0, merged_rows * out_hidden)
        .into_iter()
        .map(|v| v.to_bits())
        .collect())
}

/// FP16 little-endian bytes as u16 bits.
fn f16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// u16 bits as little-endian bytes.
fn le_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VisionConfig {
        VisionConfig {
            depth: 24,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 1,
            in_channels: 3,
            spatial_merge_size: 2,
            num_position_embeddings: 1025,
            out_hidden_size: 896,
            mrope_section: [0, 0, 0],
            deepstack_visual_indexes: Vec::new(),
            vision_start_token_id: 0,
            vision_end_token_id: 0,
            image_token_id: 151667,
            video_token_id: 0,
            tower: VisionTowerKind::InternViT,
        }
    }

    #[test]
    fn the_published_shape_resolves_with_a_32_grid() {
        let (depth, hidden, inter, heads, head_dim, out_hidden) =
            internvl_shape(&config()).unwrap();
        assert_eq!((depth, hidden, inter), (24, 1024, 4096));
        assert_eq!((heads, head_dim), (16, 64));
        assert_eq!(out_hidden, 896);
    }

    #[test]
    fn a_shape_the_tower_cannot_run_is_refused() {
        let mut c = config();
        c.tower = VisionTowerKind::QwenSiglip;
        assert!(internvl_shape(&c).is_err());
        let mut c = config();
        c.num_position_embeddings = 1026; // not class + square
        assert!(internvl_shape(&c).is_err());
        let mut c = config();
        c.spatial_merge_size = 3; // 32 does not shuffle whole 3-wide blocks
        assert!(internvl_shape(&c).is_err());
        let mut c = config();
        c.hidden_size = 1000; // not divisible by 16
        assert!(internvl_shape(&c).is_err());
        let mut c = config();
        c.patch_size = 16; // the tower's patch conv and preprocessing are 14
        assert!(internvl_shape(&c).is_err());
    }

    /// MUTATION GUARD: the role table maps blob sub-tensors to weights by
    /// position. Swapping the two layer-scale rows (ls1 onto ls2's slot)
    /// must be caught by the size check -- both are `[hidden]`, so the size
    /// does NOT catch it, and the guard is the exact table content itself.
    /// Pin it by name so any edit is a decision, and pin the role sizes the
    /// runtime validates against.
    #[test]
    fn the_role_table_and_sizes_are_pinned() {
        assert_eq!(BLOCK_ROLES[12], "ls1");
        assert_eq!(BLOCK_ROLES[13], "ls2");
        let h = 1024usize;
        let inter = 4096usize;
        assert_eq!(role_size("qkv_w", h, inter), Some((3 * h * h * 2) as u64));
        assert_eq!(role_size("qkv_b", h, inter), Some((3 * h * 2) as u64));
        assert_eq!(role_size("ls1", h, inter), Some((h * 2) as u64));
        assert_eq!(role_size("ls2", h, inter), Some((h * 2) as u64));
        assert_eq!(role_size("fc1_w", h, inter), Some((inter * h * 2) as u64));
        assert_eq!(role_size("unknown", h, inter), None);
    }

    /// The projector at the published row count and unfolded width against a
    /// CPU reference. `unfolded` is far wider than `out_hidden`, so a norm
    /// output sized by the latter writes past its buffer and the rows differ.
    #[test]
    fn the_projector_matches_a_cpu_reference_when_the_norm_is_the_widest_row() {
        const ROWS: usize = 256;
        const UNFOLDED: usize = 4096;
        const OUT: usize = 64;
        let fp16 = |i: usize, scale: f32| half::f16::from_f32((i as f32 * 0.7131).sin() * scale);
        let lens = [UNFOLDED, UNFOLDED, OUT * UNFOLDED, OUT, OUT * OUT, OUT];
        let scales = [1.0, 0.1, 0.03, 0.1, 0.2, 0.1];
        let mut bytes: Vec<half::f16> = Vec::new();
        let mut offsets = [0u64; 6];
        for (slot, (&len, &scale)) in lens.iter().zip(&scales).enumerate() {
            offsets[slot] = (bytes.len() * 2) as u64;
            let at = bytes.len();
            bytes.extend((0..len).map(|i| fp16(at + i, scale)));
        }
        let part = |slot: usize| -> Vec<f32> {
            let at = offsets[slot] as usize / 2;
            bytes[at..at + lens[slot]]
                .iter()
                .map(|v| v.to_f32())
                .collect()
        };
        let rows: Vec<half::f16> = (0..ROWS * UNFOLDED).map(|i| fp16(i * 3 + 1, 0.8)).collect();
        let shuffled: Vec<u16> = rows.iter().map(|v| v.to_bits()).collect();

        let mut context = gpu::MetalContext::new().expect("a Metal device");
        let buffer = context.new_buffer_with_data(&bytes);
        let got = run_projector(
            &mut context,
            &ProjectorWeights {
                buffer: &buffer,
                norm_w: offsets[0],
                norm_b: offsets[1],
                linear_1_w: offsets[2],
                linear_1_b: offsets[3],
                linear_2_w: offsets[4],
                linear_2_b: offsets[5],
            },
            &shuffled,
            ROWS,
            OUT,
            gpu::GeluKind::Erf,
        )
        .expect("the projector runs");

        let (nw, nb, w1, b1, w2, b2) = (part(0), part(1), part(2), part(3), part(4), part(5));
        let linear = |x: &[f32], w: &[f32], b: &[f32]| -> Vec<f32> {
            b.iter()
                .zip(w.chunks_exact(x.len()))
                .map(|(bias, row)| bias + row.iter().zip(x).map(|(a, v)| a * v).sum::<f32>())
                .collect()
        };
        for r in 0..ROWS {
            let x: Vec<f32> = rows[r * UNFOLDED..(r + 1) * UNFOLDED]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let normed = compute::vision::layer_norm(&x, &nw, &nb, INTERNVL_LAYER_NORM_EPS);
            let mid = compute::vision::gelu_erf(&linear(&normed, &w1, &b1));
            let want = linear(&mid, &w2, &b2);
            for (c, want) in want.iter().enumerate() {
                let got = half::f16::from_bits(got[r * OUT + c]).to_f32();
                assert!(
                    (got - want).abs() <= 0.02 + 0.01 * want.abs(),
                    "row {r} col {c}: {got} against {want}"
                );
            }
        }
    }

    #[test]
    fn f16_bits_round_trip() {
        let bytes = 1.5f32;
        let v = half::f16::from_f32(bytes);
        assert_eq!(f16_bits(&v.to_le_bytes())[0], v.to_bits());
        assert_eq!(le_bytes(&[v.to_bits()]), v.to_le_bytes().to_vec());
    }
}
