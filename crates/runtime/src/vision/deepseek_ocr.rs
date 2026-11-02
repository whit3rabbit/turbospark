//! The DeepSeek-OCR encoder: a SAM-ViT-B feeding a CLIP-L
//! (`docs/DEEPSEEK_OCR_VISION_PHASE0.md`).
//!
//! # The forward pass (BASE mode, one 1024 x 1024 view)
//!
//! ```text
//! x   = im2col16(img) @ patch_w + patch_b + pos            # [4096, 768]
//! for blk in 0..12:                                        # SAM, resident
//!     x += attn(ln1(x))      # eps 1e-6; windows of 14 (zero padded 64 -> 70)
//!                            # except blocks 2, 5, 8, 11 (all 4096 tokens);
//!                            # relative-position bias, padded keys unmasked
//!     x += lin2(gelu_erf(lin1(ln2(x))))                    # eps 1e-6
//! s   = net_3(net_2(ln(conv3x3(ln(conv1x1(x))))))         # [256, 1024]
//! c   = [cls; s] + clip_pos, then ln_pre (eps 1e-5)        # [257, 1024]
//! for blk in 0..24:                                        # CLIP-L, streamed
//!     c += attn(ln1(c)); c += fc2(quick_gelu(fc1(ln2(c))))  # eps 1e-5
//! rows = [c[1:], s] @ proj_w + proj_b                      # [256, 1280]
//! ```
//!
//! then the 16 rows of 16 features each closed by `image_newline`, and one
//! `view_separator`: 273 positions ([`turbospark_vision_io::deepseek_ocr_row_layout`]).
//!
//! # Where the weights live, and why the two towers differ
//!
//! The CLIP-L's blocks are the CLIP and SigLIP towers' twelve-role blob and
//! stream from `packed_vision/` through the same streamer. The SAM's twelve
//! blocks are RESIDENT FP16: 14 tensors each with two relative-position
//! tables the blob has no role for, 768 wide where the streamed blocks are
//! 1024, and 190 MiB in all, mapped lazily from the sidecar file. Both halves
//! run one block per GPU pass with a wait between, as the other towers do.
//!
//! The convolutions are im2col followed by the shared GEMM against the weight
//! as the checkpoint ships it (`[O, kh, kw, I]`), so no convolution kernel
//! exists. The SAM window partition (zero padded) and its inverse are one
//! gather kernel with two index lists built once at open.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{
    deepseek_ocr_row_layout, DeepseekOcrRow, DeepseekOcrView, OCR_BASE_SIZE, OCR_GRID, OCR_PATCH,
};

use super::shape::VisionShape;
use super::weights::{fp16_view, resolve_block_roles, BlockRoles, Role, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// SAM geometry: properties of the kind (`vision_deepseek_ocr.rs` in repack
/// checks the checkpoint against the same numbers).
const SAM_DEPTH: usize = 12;
const SAM_WIDTH: usize = 768;
const SAM_HEADS: usize = 12;
const SAM_MLP: usize = 3072;
const SAM_WINDOW: usize = 14;
const SAM_GRID: usize = OCR_BASE_SIZE / OCR_PATCH;
const SAM_GLOBAL_BLOCKS: [usize; 4] = [2, 5, 8, 11];
const SAM_NECK: usize = 256;
const SAM_NET2: usize = 512;
const SAM_NET3: usize = 1024;

/// SAM block and neck LayerNorm epsilon (`partial(nn.LayerNorm, eps=1e-6)`,
/// `LayerNorm2d(eps=1e-6)`).
const SAM_LAYER_NORM_EPS: f32 = 1e-6;
/// CLIP-L layer and pre-LayerNorm epsilon (`layernorm_epsilon` and
/// `pre_layernorm_epsilon` in the official `deepencoder.py`, 1e-5). mlx-vlm's
/// `VisionConfig.layer_norm_eps` defaults to 1e-6 for the layers; the
/// checkpoint's config sets neither, and the official code is followed.
const CLIP_LAYER_NORM_EPS: f32 = 1e-5;

const CLIP_WIDTH: usize = 1024;
const CLIP_DEPTH: usize = 24;
const CLIP_ROWS: usize = OCR_GRID * OCR_GRID + 1;
/// The SAM output grid feeding the CLIP-L: 16 x 16 tokens.
const SAM_OUT_TOKENS: usize = OCR_GRID * OCR_GRID;

/// The residual stream at the points the real-tower parity test compares,
/// as FP16 bits. Row-major, the shapes in the names' comments.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeepseekOcrStages {
    /// `(name, FP16 bits)` in execution order.
    pub named: Vec<(&'static str, Vec<u16>)>,
}

impl DeepseekOcrStages {
    pub fn get(&self, name: &str) -> Option<&[u16]> {
        self.named
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.as_slice())
    }
}

/// One SAM block's resident tensors (offsets into the resident region).
#[derive(Debug, Clone)]
struct SamBlock {
    norm1_w: u64,
    norm1_b: u64,
    qkv_w: u64,
    qkv_b: u64,
    proj_w: u64,
    proj_b: u64,
    rel_h: u64,
    rel_w: u64,
    norm2_w: u64,
    norm2_b: u64,
    lin1_w: u64,
    lin1_b: u64,
    lin2_w: u64,
    lin2_b: u64,
}

/// Everything resident the tower reads.
#[derive(Debug, Clone)]
struct OcrResident {
    patch_w: u64,
    patch_b: u64,
    pos_embed: u64,
    blocks: Vec<SamBlock>,
    neck0_w: u64,
    neck1_w: u64,
    neck1_b: u64,
    neck2_w: u64,
    neck3_w: u64,
    neck3_b: u64,
    net2_w: u64,
    net3_w: u64,
    class_embed: u64,
    clip_pos: u64,
    pre_w: u64,
    pre_b: u64,
    proj_w: u64,
    proj_b: u64,
    image_newline: u64,
    view_separator: u64,
}

impl OcrResident {
    fn resolve(index: &ResidentIndex, out_hidden: usize) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let sam = |tail: &str| name(&format!("sam.{tail}"));
        let view = |n: String, elems: usize| fp16_view(index, &n, elems);
        let d = SAM_WIDTH / SAM_HEADS;
        let mut blocks = Vec::with_capacity(SAM_DEPTH);
        for i in 0..SAM_DEPTH {
            let global = SAM_GLOBAL_BLOCKS.contains(&i);
            let rel = if global {
                2 * SAM_GRID - 1
            } else {
                2 * SAM_WINDOW - 1
            };
            let b = |tail: &str, elems: usize| view(sam(&format!("blocks.{i}.{tail}")), elems);
            blocks.push(SamBlock {
                norm1_w: b("norm1.weight", SAM_WIDTH)?,
                norm1_b: b("norm1.bias", SAM_WIDTH)?,
                qkv_w: b("attn.qkv.weight", 3 * SAM_WIDTH * SAM_WIDTH)?,
                qkv_b: b("attn.qkv.bias", 3 * SAM_WIDTH)?,
                proj_w: b("attn.proj.weight", SAM_WIDTH * SAM_WIDTH)?,
                proj_b: b("attn.proj.bias", SAM_WIDTH)?,
                rel_h: b("attn.rel_pos_h", rel * d)?,
                rel_w: b("attn.rel_pos_w", rel * d)?,
                norm2_w: b("norm2.weight", SAM_WIDTH)?,
                norm2_b: b("norm2.bias", SAM_WIDTH)?,
                lin1_w: b("mlp.lin1.weight", SAM_MLP * SAM_WIDTH)?,
                lin1_b: b("mlp.lin1.bias", SAM_MLP)?,
                lin2_w: b("mlp.lin2.weight", SAM_WIDTH * SAM_MLP)?,
                lin2_b: b("mlp.lin2.bias", SAM_WIDTH)?,
            });
        }
        Ok(Self {
            patch_w: view(
                sam("patch_embed.proj.weight"),
                SAM_WIDTH * OCR_PATCH * OCR_PATCH * 3,
            )?,
            patch_b: view(sam("patch_embed.proj.bias"), SAM_WIDTH)?,
            pos_embed: view(sam("pos_embed"), SAM_GRID * SAM_GRID * SAM_WIDTH)?,
            blocks,
            neck0_w: view(sam("neck.0.weight"), SAM_NECK * SAM_WIDTH)?,
            neck1_w: view(sam("neck.1.weight"), SAM_NECK)?,
            neck1_b: view(sam("neck.1.bias"), SAM_NECK)?,
            neck2_w: view(sam("neck.2.weight"), SAM_NECK * 9 * SAM_NECK)?,
            neck3_w: view(sam("neck.3.weight"), SAM_NECK)?,
            neck3_b: view(sam("neck.3.bias"), SAM_NECK)?,
            net2_w: view(sam("net_2.weight"), SAM_NET2 * 9 * SAM_NECK)?,
            net3_w: view(sam("net_3.weight"), SAM_NET3 * 9 * SAM_NET2)?,
            class_embed: view(name("clip.class_embedding"), CLIP_WIDTH)?,
            clip_pos: view(name("clip.pos_embed"), CLIP_ROWS * CLIP_WIDTH)?,
            pre_w: view(name("clip.pre_layrnorm.weight"), CLIP_WIDTH)?,
            pre_b: view(name("clip.pre_layrnorm.bias"), CLIP_WIDTH)?,
            proj_w: view(
                name("projector.weight"),
                out_hidden * (CLIP_WIDTH + SAM_NET3),
            )?,
            proj_b: view(name("projector.bias"), out_hidden)?,
            image_newline: view(name("image_newline"), out_hidden)?,
            view_separator: view(name("view_separator"), out_hidden)?,
        })
    }
}

/// The streamed CLIP-L's dimensions, in the qwen tower's [`VisionShape`] so
/// the block role resolver is shared.
fn ocr_shape(vision: &VisionConfig) -> Result<VisionShape, RealForwardError> {
    let bad = |detail: String| RealForwardError::Unsupported(detail);
    if vision.tower != VisionTowerKind::DeepseekOcr || !vision.is_active() {
        return Err(bad(
            "this install declares no DeepSeek-OCR vision tower".to_string()
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
    if (depth, hidden, inter, heads) != (CLIP_DEPTH, CLIP_WIDTH, 4096, 16)
        || patch != OCR_PATCH
        || pos_rows != CLIP_ROWS
        || out_hidden == 0
    {
        return Err(bad(format!(
            "DeepSeek-OCR tower of depth {depth}, width {hidden}, mlp {inter}, heads {heads}, \
             patch {patch}, {pos_rows} positions, out {out_hidden}; only the 24-layer CLIP-L over \
             a 16 px SAM with {CLIP_ROWS} positions is implemented"
        )));
    }
    let head_dim = hidden / heads;
    if head_dim > gpu::MAX_ATTENTION_HEAD_DIM as usize {
        return Err(bad(format!(
            "CLIP head_dim {head_dim} exceeds the attention kernel's tile ({})",
            gpu::MAX_ATTENTION_HEAD_DIM
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
        pos_side: OCR_GRID,
        patch_size: patch,
        deepstack: Vec::new(),
    })
}

/// Index list that gathers the zero-padded window partition: output row
/// `(window, i, j)` reads token `(wy * window + i, wx * window + j)` of the
/// `grid x grid` image, or the zero sentinel where that is past the edge.
/// Windows run row-major over the padded grid.
fn window_gather_index(grid: usize, window: usize) -> Vec<u32> {
    let per_side = grid.div_ceil(window);
    let mut index = Vec::with_capacity(per_side * per_side * window * window);
    for wy in 0..per_side {
        for wx in 0..per_side {
            for i in 0..window {
                for j in 0..window {
                    let (y, x) = (wy * window + i, wx * window + j);
                    index.push(if y < grid && x < grid {
                        (y * grid + x) as u32
                    } else {
                        gpu::VISION_GATHER_ZERO
                    });
                }
            }
        }
    }
    index
}

/// The inverse: token `(y, x)` reads window row `(y / w, x / w, y % w, x % w)`;
/// the padding rows are never read, which is the crop.
fn window_scatter_index(grid: usize, window: usize) -> Vec<u32> {
    let per_side = grid.div_ceil(window);
    let mut index = Vec::with_capacity(grid * grid);
    for y in 0..grid {
        for x in 0..grid {
            let win = (y / window) * per_side + x / window;
            index.push((((win * window) + y % window) * window + x % window) as u32);
        }
    }
    index
}

/// The tower: its streamer, its CLIP blocks' roles and its resident offsets.
pub(crate) struct DeepseekOcrVisionTower {
    /// Declared BEFORE `streamer` so the buffers drop before the allocations
    /// they alias.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<BlockRoles>,
    resident: OcrResident,
    shape: VisionShape,
    /// `class_embedding`, `image_newline` and `view_separator`, FP16 bytes,
    /// copied out at open: the class row and the layout are host assembly.
    class_row: Vec<u8>,
    newline_row: Vec<u8>,
    separator_row: Vec<u8>,
    /// The window partition's and its inverse's row indexes (`u32`).
    window_gather: gpu::MetalBuffer,
    window_scatter: gpu::MetalBuffer,
    /// The sidecar's own resident weights, `Some` when the tower was opened
    /// against a standalone sidecar rather than the trunk's install.
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
}

impl DeepseekOcrVisionTower {
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
    /// mapping, exactly as the other towers do.
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
        let shape = ocr_shape(&arch.vision)?;
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
        .map_err(|e| RealForwardError::Unsupported(format!("DeepSeek-OCR vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }

        let resident = OcrResident::resolve(index, shape.out_hidden)?;
        let copy_row =
            |offset: u64, width: usize, what: &str| -> Result<Vec<u8>, RealForwardError> {
                let at = offset as usize;
                Ok(weights
                    .data()
                    .get(at..at + width * 2)
                    .ok_or_else(|| {
                        RealForwardError::Unsupported(format!(
                            "DeepSeek-OCR {what} spans past the mapped region"
                        ))
                    })?
                    .to_vec())
            };
        let class_row = copy_row(resident.class_embed, CLIP_WIDTH, "class_embedding")?;
        let newline_row = copy_row(resident.image_newline, shape.out_hidden, "image_newline")?;
        let separator_row = copy_row(resident.view_separator, shape.out_hidden, "view_separator")?;
        let window_gather =
            context.new_buffer_with_data(&window_gather_index(SAM_GRID, SAM_WINDOW));
        let window_scatter =
            context.new_buffer_with_data(&window_scatter_index(SAM_GRID, SAM_WINDOW));
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            resident,
            shape,
            class_row,
            newline_row,
            separator_row,
            window_gather,
            window_scatter,
            sidecar_weights: None,
        })
    }

    /// Whether this tower is sidecar-backed (test-only engagement proof).
    #[allow(dead_code)]
    pub(crate) fn is_sidecar(&self) -> bool {
        self.sidecar_weights.is_some()
    }

    /// Encode one image to `[273, out_hidden]` FP16 bits: the whole prompt run,
    /// newline and separator rows included.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        view: &DeepseekOcrView,
        want_stages: bool,
    ) -> Result<(Vec<u16>, Option<DeepseekOcrStages>), RealForwardError> {
        if view.pixels.len() != OCR_BASE_SIZE * OCR_BASE_SIZE * 3 {
            return Err(RealForwardError::Unsupported(format!(
                "the image view has {} values; this tower reads a {OCR_BASE_SIZE} x {OCR_BASE_SIZE} \
                 RGB view",
                view.pixels.len()
            )));
        }
        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let gpu_err = RealForwardError::Gpu;
        let bytes = |elems: usize| (elems * 2) as u64;
        let tokens = SAM_GRID * SAM_GRID;
        let window_rows = SAM_GRID.div_ceil(SAM_WINDOW).pow(2) * SAM_WINDOW * SAM_WINDOW;
        let pixels: Vec<half::f16> = view
            .pixels
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let read = |b: &gpu::MetalBuffer, count: usize| -> Vec<u16> {
            gpu::read_buffer_f16(b, 0, count)
                .into_iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let mut stages = want_stages.then(DeepseekOcrStages::default);
        let mut stage = |name: &'static str, data: Vec<u16>| {
            if let Some(st) = stages.as_mut() {
                st.named.push((name, data));
            }
        };

        let result = gpu::autorelease_pool(|| {
            // ---- SAM: patch embedding and position add --------------------
            let img = context.new_buffer_with_data(&pixels);
            let cols = context.new_output_buffer(bytes(tokens * OCR_PATCH * OCR_PATCH * 3));
            let x = context.new_output_buffer(bytes(tokens * SAM_WIDTH));
            let sam = SamScratch {
                x: &x,
                normed: &context.new_output_buffer(bytes(tokens * SAM_WIDTH)),
                win: &context.new_output_buffer(bytes(window_rows * SAM_WIDTH)),
                q: &context.new_output_buffer(bytes(window_rows * SAM_WIDTH)),
                k: &context.new_output_buffer(bytes(window_rows * SAM_WIDTH)),
                v: &context.new_output_buffer(bytes(window_rows * SAM_WIDTH)),
                attn: &context.new_output_buffer(bytes(window_rows * SAM_WIDTH)),
                proj: &context.new_output_buffer(bytes(window_rows * SAM_WIDTH)),
                y: &context.new_output_buffer(bytes(tokens * SAM_WIDTH)),
                h1: &context.new_output_buffer(bytes(tokens * SAM_MLP)),
            };
            let pass = context.begin_pass();
            gpu::encode_vision_im2col(
                context,
                &pass,
                (&img, 0),
                (&cols, 0),
                OCR_BASE_SIZE as u32,
                OCR_BASE_SIZE as u32,
                3,
                OCR_PATCH as u32,
                OCR_PATCH as u32,
                0,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&cols, 0),
                (w.buffer(), w.gpu_offset(self.resident.patch_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.patch_b))),
                (&x, 0),
                tokens as u32,
                (OCR_PATCH * OCR_PATCH * 3) as u32,
                SAM_WIDTH as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&x, 0),
                (w.buffer(), w.gpu_offset(self.resident.pos_embed)),
                (tokens * SAM_WIDTH) as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            stage("sam_patch", read(&x, tokens * SAM_WIDTH));

            // ---- SAM blocks ------------------------------------------------
            for (i, blk) in self.resident.blocks.iter().enumerate() {
                let pass = context.begin_pass();
                encode_sam_block(
                    context,
                    &pass,
                    w,
                    blk,
                    SAM_GLOBAL_BLOCKS.contains(&i),
                    &sam,
                    (&self.window_gather, &self.window_scatter),
                )?;
                pass.commit_and_wait();
                match i {
                    0 => stage("sam_b0", read(&x, tokens * SAM_WIDTH)),
                    2 => stage("sam_b2", read(&x, tokens * SAM_WIDTH)),
                    5 => stage("sam_b5", read(&x, tokens * SAM_WIDTH)),
                    11 => stage("sam_b11", read(&x, tokens * SAM_WIDTH)),
                    _ => {}
                }
            }

            // ---- SAM neck and the two stride-2 convs -----------------------
            let t0 = context.new_output_buffer(bytes(tokens * SAM_NECK));
            let t1 = context.new_output_buffer(bytes(tokens * SAM_NECK));
            let cols3 = context.new_output_buffer(bytes(tokens * 9 * SAM_NECK));
            let half_grid = SAM_GRID / 2;
            let quarter_grid = SAM_GRID / 4;
            let cols_s2 = context.new_output_buffer(bytes(half_grid * half_grid * 9 * SAM_NECK));
            let n2 = context.new_output_buffer(bytes(half_grid * half_grid * SAM_NET2));
            let cols_s3 =
                context.new_output_buffer(bytes(quarter_grid * quarter_grid * 9 * SAM_NET2));
            let sam_out = context.new_output_buffer(bytes(SAM_OUT_TOKENS * SAM_NET3));
            let grid = SAM_GRID as u32;
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&x, 0),
                (w.buffer(), w.gpu_offset(self.resident.neck0_w)),
                None,
                (&t0, 0),
                tokens as u32,
                SAM_WIDTH as u32,
                SAM_NECK as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&t0, 0),
                (w.buffer(), w.gpu_offset(self.resident.neck1_w)),
                (w.buffer(), w.gpu_offset(self.resident.neck1_b)),
                (&t0, 0),
                tokens as u32,
                SAM_NECK as u32,
                SAM_LAYER_NORM_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_im2col(
                context,
                &pass,
                (&t0, 0),
                (&cols3, 0),
                grid,
                grid,
                SAM_NECK as u32,
                3,
                1,
                1,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&cols3, 0),
                (w.buffer(), w.gpu_offset(self.resident.neck2_w)),
                None,
                (&t1, 0),
                tokens as u32,
                (9 * SAM_NECK) as u32,
                SAM_NECK as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&t1, 0),
                (w.buffer(), w.gpu_offset(self.resident.neck3_w)),
                (w.buffer(), w.gpu_offset(self.resident.neck3_b)),
                (&t1, 0),
                tokens as u32,
                SAM_NECK as u32,
                SAM_LAYER_NORM_EPS,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            stage("sam_neck", read(&t1, tokens * SAM_NECK));

            let pass = context.begin_pass();
            gpu::encode_vision_im2col(
                context,
                &pass,
                (&t1, 0),
                (&cols_s2, 0),
                grid,
                grid,
                SAM_NECK as u32,
                3,
                2,
                1,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&cols_s2, 0),
                (w.buffer(), w.gpu_offset(self.resident.net2_w)),
                None,
                (&n2, 0),
                (half_grid * half_grid) as u32,
                (9 * SAM_NECK) as u32,
                SAM_NET2 as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_im2col(
                context,
                &pass,
                (&n2, 0),
                (&cols_s3, 0),
                half_grid as u32,
                half_grid as u32,
                SAM_NET2 as u32,
                3,
                2,
                1,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&cols_s3, 0),
                (w.buffer(), w.gpu_offset(self.resident.net3_w)),
                None,
                (&sam_out, 0),
                SAM_OUT_TOKENS as u32,
                (9 * SAM_NET2) as u32,
                SAM_NET3 as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            stage("sam_n2", read(&n2, half_grid * half_grid * SAM_NET2));
            let sam_bits = read(&sam_out, SAM_OUT_TOKENS * SAM_NET3);
            stage("sam_out", sam_bits.clone());

            // ---- CLIP-L: the SAM grid is its patch embedding ---------------
            let le = |row: &[u8]| -> Vec<u16> {
                row.chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect()
            };
            let mut clip_in: Vec<u16> = Vec::with_capacity(CLIP_ROWS * CLIP_WIDTH);
            clip_in.extend(le(&self.class_row));
            clip_in.extend_from_slice(&sam_bits);
            let clip_in: Vec<half::f16> = clip_in.into_iter().map(half::f16::from_bits).collect();
            let hidden = self.shape.hidden;
            let rows = CLIP_ROWS;
            let c = context.new_buffer_with_data(&clip_in);
            let clip = ClipScratch {
                x: &c,
                normed: &context.new_output_buffer(bytes(rows * hidden)),
                q: &context.new_output_buffer(bytes(rows * hidden)),
                k: &context.new_output_buffer(bytes(rows * hidden)),
                v: &context.new_output_buffer(bytes(rows * hidden)),
                attn: &context.new_output_buffer(bytes(rows * hidden)),
                proj: &context.new_output_buffer(bytes(rows * hidden)),
                h1: &context.new_output_buffer(bytes(rows * self.shape.intermediate)),
                rows,
            };
            let pass = context.begin_pass();
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&c, 0),
                (w.buffer(), w.gpu_offset(self.resident.clip_pos)),
                (rows * hidden) as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&c, 0),
                (w.buffer(), w.gpu_offset(self.resident.pre_w)),
                (w.buffer(), w.gpu_offset(self.resident.pre_b)),
                (&c, 0),
                rows as u32,
                hidden as u32,
                CLIP_LAYER_NORM_EPS,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            stage("clip_pre", read(&c, rows * hidden));

            for blk in 0..self.shape.depth {
                let slot = blk % VISION_SLOTS;
                self.streamer
                    .load_expert_into_slot(0, blk, slot)
                    .map_err(|e| {
                        RealForwardError::Unsupported(format!(
                            "DeepSeek-OCR CLIP block {blk} read: {e}"
                        ))
                    })?;
                let pass = context.begin_pass();
                encode_clip_block(
                    context,
                    &pass,
                    &self.slot_buffers[slot],
                    &self.blocks[blk],
                    &clip,
                    &self.shape,
                )?;
                pass.commit_and_wait();
                match blk {
                    0 => stage("clip_l0", read(&c, rows * hidden)),
                    11 => stage("clip_l11", read(&c, rows * hidden)),
                    _ => {}
                }
            }
            let clip_bits = read(&c, rows * hidden);
            stage("clip_out", clip_bits.clone());

            // ---- Concatenate [CLIP[1:], SAM] and project -------------------
            let feat_width = CLIP_WIDTH + SAM_NET3;
            let mut feat: Vec<half::f16> = Vec::with_capacity(SAM_OUT_TOKENS * feat_width);
            for r in 0..SAM_OUT_TOKENS {
                let clip_row = &clip_bits[(r + 1) * CLIP_WIDTH..(r + 2) * CLIP_WIDTH];
                let sam_row = &sam_bits[r * SAM_NET3..(r + 1) * SAM_NET3];
                feat.extend(clip_row.iter().map(|&b| half::f16::from_bits(b)));
                feat.extend(sam_row.iter().map(|&b| half::f16::from_bits(b)));
            }
            let feat_buf = context.new_buffer_with_data(&feat);
            let out_hidden = self.shape.out_hidden;
            let proj = context.new_output_buffer(bytes(SAM_OUT_TOKENS * out_hidden));
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&feat_buf, 0),
                (w.buffer(), w.gpu_offset(self.resident.proj_w)),
                Some((w.buffer(), w.gpu_offset(self.resident.proj_b))),
                (&proj, 0),
                SAM_OUT_TOKENS as u32,
                feat_width as u32,
                out_hidden as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
            let projected = read(&proj, SAM_OUT_TOKENS * out_hidden);
            stage("proj", projected.clone());
            Ok::<_, RealForwardError>(projected)
        });
        let projected = result?;

        // Interleave the learned rows, in the reference's base-mode layout.
        let o = self.shape.out_hidden;
        let le = |row: &[u8]| -> Vec<u16> {
            row.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        let newline = le(&self.newline_row);
        let separator = le(&self.separator_row);
        let layout = deepseek_ocr_row_layout();
        let mut out = Vec::with_capacity(layout.len() * o);
        for item in layout {
            match item {
                DeepseekOcrRow::Feature(row) => {
                    out.extend_from_slice(&projected[row * o..(row + 1) * o])
                }
                DeepseekOcrRow::Newline => out.extend_from_slice(&newline),
                DeepseekOcrRow::Separator => out.extend_from_slice(&separator),
            }
        }
        if let Some(st) = stages.as_mut() {
            st.named.push(("rows", out.clone()));
        }
        Ok((out, stages))
    }
}

/// The SAM block's per-run buffers. `win`, `q`, `k`, `v`, `attn` and `proj`
/// hold the partitioned windows (4900 rows) in a window block and the plain
/// 4096 rows in a global one.
struct SamScratch<'a> {
    x: &'a gpu::MetalBuffer,
    normed: &'a gpu::MetalBuffer,
    win: &'a gpu::MetalBuffer,
    q: &'a gpu::MetalBuffer,
    k: &'a gpu::MetalBuffer,
    v: &'a gpu::MetalBuffer,
    attn: &'a gpu::MetalBuffer,
    proj: &'a gpu::MetalBuffer,
    y: &'a gpu::MetalBuffer,
    h1: &'a gpu::MetalBuffer,
}

/// One SAM block: `x += attn(ln1(x)); x += lin2(gelu(lin1(ln2(x))))`, the
/// attention over zero-padded windows or the whole grid.
fn encode_sam_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    w: &gpu::ResidentGpuWeights,
    blk: &SamBlock,
    global: bool,
    s: &SamScratch<'_>,
    (gather, scatter): (&gpu::MetalBuffer, &gpu::MetalBuffer),
) -> Result<(), RealForwardError> {
    let gpu_err = RealForwardError::Gpu;
    let tokens = (SAM_GRID * SAM_GRID) as u32;
    let width = SAM_WIDTH as u32;
    let off = |o: u64| w.gpu_offset(o);
    let wb = w.buffer();

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (s.x, 0),
        (wb, off(blk.norm1_w)),
        (wb, off(blk.norm1_b)),
        (s.normed, 0),
        tokens,
        width,
        SAM_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;

    // Window blocks attend inside zero-padded 14 x 14 windows of the NORMED
    // grid (the pad is zeros AFTER norm1, so padded tokens carry the qkv bias
    // as keys); global blocks attend over all tokens.
    let (src, rows, gh, windows) = if global {
        (s.normed, tokens, SAM_GRID as u32, 1u32)
    } else {
        let window_rows = (SAM_GRID.div_ceil(SAM_WINDOW).pow(2) * SAM_WINDOW * SAM_WINDOW) as u32;
        gpu::encode_vision_gather_rows(
            context,
            pass,
            (s.normed, 0),
            (gather, 0),
            (s.win, 0),
            window_rows,
            width,
        )
        .map_err(gpu_err)?;
        (
            s.win,
            window_rows,
            SAM_WINDOW as u32,
            window_rows / (SAM_WINDOW * SAM_WINDOW) as u32,
        )
    };

    // q, k, v at rows 0, width, 2 * width of the fused weight.
    let row_bytes = (SAM_WIDTH * SAM_WIDTH * 2) as u64;
    let bias_bytes = (SAM_WIDTH * 2) as u64;
    for (i, out) in [s.q, s.k, s.v].into_iter().enumerate() {
        gpu::encode_vision_matmul(
            context,
            pass,
            (src, 0),
            (wb, off(blk.qkv_w) + i as u64 * row_bytes),
            Some((wb, off(blk.qkv_b) + i as u64 * bias_bytes)),
            (out, 0),
            rows,
            width,
            width,
        )
        .map_err(gpu_err)?;
    }
    gpu::encode_vision_sam_attention(
        context,
        pass,
        (s.q, 0),
        (s.k, 0),
        (s.v, 0),
        (wb, off(blk.rel_h)),
        (wb, off(blk.rel_w)),
        (s.attn, 0),
        gh,
        gh,
        windows,
        SAM_HEADS as u32,
        1.0 / ((SAM_WIDTH / SAM_HEADS) as f32).sqrt(),
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (s.attn, 0),
        (wb, off(blk.proj_w)),
        Some((wb, off(blk.proj_b))),
        (s.proj, 0),
        rows,
        width,
        width,
    )
    .map_err(gpu_err)?;
    if global {
        gpu::encode_vision_residual_add(context, pass, (s.x, 0), (s.proj, 0), tokens * width)
            .map_err(gpu_err)?;
    } else {
        // Crop the padding off: token (y, x) reads its window row back.
        gpu::encode_vision_gather_rows(
            context,
            pass,
            (s.proj, 0),
            (scatter, 0),
            (s.y, 0),
            tokens,
            width,
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_residual_add(context, pass, (s.x, 0), (s.y, 0), tokens * width)
            .map_err(gpu_err)?;
    }

    gpu::encode_vision_layer_norm(
        context,
        pass,
        (s.x, 0),
        (wb, off(blk.norm2_w)),
        (wb, off(blk.norm2_b)),
        (s.normed, 0),
        tokens,
        width,
        SAM_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (s.normed, 0),
        (wb, off(blk.lin1_w)),
        Some((wb, off(blk.lin1_b))),
        (s.h1, 0),
        tokens,
        width,
        SAM_MLP as u32,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_gelu(
        context,
        pass,
        (s.h1, 0),
        tokens * SAM_MLP as u32,
        gpu::GeluKind::Erf,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_matmul(
        context,
        pass,
        (s.h1, 0),
        (wb, off(blk.lin2_w)),
        Some((wb, off(blk.lin2_b))),
        (s.y, 0),
        tokens,
        SAM_MLP as u32,
        width,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_residual_add(context, pass, (s.x, 0), (s.y, 0), tokens * width)
        .map_err(gpu_err)
}

/// The CLIP block's per-run buffers (one sequence of 257 rows).
struct ClipScratch<'a> {
    x: &'a gpu::MetalBuffer,
    normed: &'a gpu::MetalBuffer,
    q: &'a gpu::MetalBuffer,
    k: &'a gpu::MetalBuffer,
    v: &'a gpu::MetalBuffer,
    attn: &'a gpu::MetalBuffer,
    proj: &'a gpu::MetalBuffer,
    h1: &'a gpu::MetalBuffer,
    rows: usize,
}

/// One CLIP-L block: `x += attn(ln1(x)); x += fc2(quick_gelu(fc1(ln2(x))))`,
/// the SigLIP block's structure with this kind's epsilon and activation
/// (`vision::siglip::encode_siglip_block`, which it mirrors).
fn encode_clip_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    slot: &gpu::MetalBuffer,
    roles: &BlockRoles,
    b: &ClipScratch<'_>,
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
        CLIP_LAYER_NORM_EPS,
    )
    .map_err(gpu_err)?;
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
    gpu::encode_vision_attention(
        context,
        pass,
        (b.q, 0),
        (b.k, 0),
        (b.v, 0),
        (b.attn, 0),
        rows,
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
        rows,
        hidden,
        inter,
    )
    .map_err(gpu_err)?;
    gpu::encode_vision_gelu(context, pass, (b.h1, 0), rows * inter, gpu::GeluKind::Quick)
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
            depth: 24,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_heads: 16,
            patch_size: 16,
            temporal_patch_size: 1,
            in_channels: 3,
            spatial_merge_size: 4,
            num_position_embeddings: 257,
            out_hidden_size: 1280,
            mrope_section: [0, 0, 0],
            deepstack_visual_indexes: Vec::new(),
            vision_start_token_id: 0,
            vision_end_token_id: 0,
            image_token_id: 128_815,
            video_token_id: 0,
            tower: VisionTowerKind::DeepseekOcr,
        }
    }

    #[test]
    fn the_deepseek_ocr_shape_resolves() {
        let s = ocr_shape(&config()).unwrap();
        assert_eq!((s.hidden, s.heads, s.head_dim, s.depth), (1024, 16, 64, 24));
        assert_eq!((s.out_hidden, s.pos_rows), (1280, 257));
    }

    #[test]
    fn a_shape_the_tower_cannot_run_is_refused() {
        let mut c = config();
        c.tower = VisionTowerKind::SigLip;
        assert!(ocr_shape(&c).is_err());
        let mut c = config();
        c.depth = 12;
        assert!(ocr_shape(&c).is_err());
        let mut c = config();
        c.num_position_embeddings = 577;
        assert!(ocr_shape(&c).is_err());
        let mut c = config();
        c.patch_size = 14;
        assert!(ocr_shape(&c).is_err());
    }

    #[test]
    fn the_window_partition_pads_to_70_and_its_inverse_undoes_it() {
        let (grid, window) = (64usize, 14usize);
        let gather = window_gather_index(grid, window);
        let scatter = window_scatter_index(grid, window);
        assert_eq!(gather.len(), 5 * 5 * 14 * 14);
        assert_eq!(scatter.len(), 64 * 64);
        let pad = gather
            .iter()
            .filter(|&&i| i == gpu::VISION_GATHER_ZERO)
            .count();
        assert_eq!(pad, 70 * 70 - 64 * 64);
        // Every real token appears exactly once in the partition ...
        let mut seen = vec![0u8; grid * grid];
        for &i in &gather {
            if i != gpu::VISION_GATHER_ZERO {
                seen[i as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
        // ... and the crop reads each token's own window row back.
        for t in 0..grid * grid {
            assert_eq!(gather[scatter[t] as usize] as usize, t, "token {t}");
        }
        // Window (0, 0) is the top-left 14 x 14 block in raster order; window
        // (0, 4) starts at column 56, so its columns 0..=7 are real (56..=63)
        // and the last 6 are padding.
        assert_eq!(&gather[0..3], &[0, 1, 2]);
        assert_eq!(gather[14], 64);
        let w04 = 4 * 14 * 14;
        assert_eq!(gather[w04], 56);
        assert_eq!(gather[w04 + 7], 63);
        assert_eq!(gather[w04 + 8], gpu::VISION_GATHER_ZERO);
        // Window (4, 0) is the bottom strip: its rows 8.. are padding.
        let w40 = 4 * 5 * 14 * 14;
        assert_eq!(gather[w40], (56 * 64) as u32);
        assert_eq!(gather[w40 + 8 * 14], gpu::VISION_GATHER_ZERO);
    }
}
