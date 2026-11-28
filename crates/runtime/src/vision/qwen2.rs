//! The Qwen2-VL vision tower on Metal (`docs/QWEN2VL_VISION_PHASE0.md`).
//!
//! A sibling of the Qwen2.5-VL tower in [`super::qwen25`], sharing the
//! destination format (blocks streamed two slots at a time from
//! `packed_vision/`, residents at FP16 under `vision.`), the kernels, the
//! patchify/2-D-rope machinery, and the result type: it returns the same
//! [`VisionEmbedding`], so the prompt builders and `set_prompt_vision` need
//! no second path. It differs from the 2.5 tower in exactly the three places
//! the reference (`mlx_vlm/models/qwen2_vl/vision.py`) differs from its
//! sibling, and nowhere else:
//!
//! - **LayerNorm blocks.** `norm1`/`norm2` and the merger's `ln_q` are
//!   `nn.LayerNorm(dim, eps=1e-6)` -- weight AND bias -- where the 2.5 tower
//!   has weight-only RMSNorm. This is the `encode_vision_layer_norm` kernel,
//!   not `encode_vision_rms_norm`.
//! - **A plain two-layer GELU MLP.** `mlp.fc1` / `mlp.fc2` with a bias each,
//!   no gate/up split. The activation is the reference's own
//!   `nn.GELU(approx="fast")`, which mlx spells as the fast SIGMOID
//!   approximation (`x * sigmoid(1.702 x)`; `GeluKind::Quick` here), the
//!   same function as transformers' `quick_gelu`. The merger's own GELU is a bare
//!   `nn.GELU()`, the exact erf form, same as 2.5.
//! - **Full attention on every block.** The reference has no window
//!   machinery at all (no `get_window_index`, no `fullatt_block_indexes`);
//!   attention is bidirectional over each image's whole patch sequence.
//!   Preprocessing feeds one image per call, so that is one
//!   `encode_vision_attention` over the page -- no permutation, no segments,
//!   and the page's own order from patch rows to returned merger rows.
//!
//! The block MLP of the 2.5 tower is gated SiLU and the merger's `ln_q` an
//! RMSNorm; a blob layout read with the wrong role table mixes gate rows
//! into an fc1 slot, which is exactly the shape-correct, fluent-wrong
//! failure [`Qwen2Dims::resolve`] and the runtime block resolver's size
//! checks exist to refuse.
//!
//! **Mage-ViT** (`VisionTowerKind::MageVit`, `docs/MAGE_VL_VISION_PHASE0.md`)
//! runs here too: the same twelve roles, full attention and merger, with a
//! LayerNorm between the patch embedding and block 0, the exact erf GELU in
//! the block MLP, and its own three-axis interleaved rope.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{PreprocessParams, PreprocessedImage};

use super::weights::{fp16_view, VISION_PREFIX};
use super::{VisionEmbedding, VisionStages, VISION_SLOTS};
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// LayerNorm epsilon: the block norms and the merger's `ln_q` are
/// `nn.LayerNorm(dim, eps=1e-6)` in the reference (the config omits the key,
/// so this is the reference's own default; intake checks it when a
/// checkpoint declares it). A KIND CONSTANT, never a manifest float.
const EPS: f32 = 1e-6;

/// MLP row tile: the `[rows, intermediate]` hidden buffer is sized for a
/// tile, not the page (the 2.5 tower's shape).
pub(crate) const QWEN2_MLP_TILE_ROWS: usize = 1024;

/// The blob roles in byte order, the repack writer's `QWEN2_VL_BLOCK_ROLES`
/// first column. Restated rather than imported because repack is a
/// dev-dependency of this crate.
pub(crate) const QWEN2_BLOCK_ROLES: [&str; 12] = [
    "n1_w", "n1_b", "qkv_w", "qkv_b", "proj_w", "proj_b", "n2_w", "n2_b", "fc1_w", "fc1_b",
    "fc2_w", "fc2_b",
];

#[derive(Debug, Clone, Copy)]
enum R {
    N1W = 0,
    N1B = 1,
    QkvW = 2,
    QkvB = 3,
    ProjW = 4,
    ProjB = 5,
    N2W = 6,
    N2B = 7,
    Fc1W = 8,
    Fc1B = 9,
    Fc2W = 10,
    Fc2B = 11,
}

const ALL_ROLES: [R; 12] = [
    R::N1W,
    R::N1B,
    R::QkvW,
    R::QkvB,
    R::ProjW,
    R::ProjB,
    R::N2W,
    R::N2B,
    R::Fc1W,
    R::Fc1B,
    R::Fc2W,
    R::Fc2B,
];

/// The tower's geometry, resolved once from the install's [`VisionConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen2Dims {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_dim: usize,
    pub(crate) patch_size: usize,
    pub(crate) merge: usize,
    pub(crate) out_hidden: usize,
    /// The tower is Mage-ViT rather than Qwen2-VL.
    pub(crate) mage: bool,
}

impl Qwen2Dims {
    pub(crate) fn resolve(v: &VisionConfig) -> Result<Self, RealForwardError> {
        let mage = v.tower == VisionTowerKind::MageVit;
        if !(mage || v.tower == VisionTowerKind::Qwen2Vl) || !v.is_active() {
            return Err(RealForwardError::Unsupported(format!(
                "not an active qwen2-vl vision tower (kind {})",
                v.tower.as_str()
            )));
        }
        let us = |x: i64, what: &str| {
            usize::try_from(x)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| RealForwardError::Unsupported(format!("qwen2 vision {what} {x}")))
        };
        let d = Self {
            depth: us(v.depth, "depth")?,
            hidden: us(v.hidden_size, "embed_dim")?,
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
            mage,
        };
        if mage && d.head_dim % 32 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "mage_vit head_dim {} is not a multiple of 32, which its 4:6:6 rope needs",
                d.head_dim
            )));
        }
        if d.heads * d.head_dim != d.hidden || d.head_dim % 4 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "qwen2 vision embed_dim {} over {} heads does not give a head_dim divisible \
                 by 4",
                d.hidden, d.heads
            )));
        }
        if d.head_dim as u32 > gpu::MAX_ATTENTION_HEAD_DIM {
            return Err(RealForwardError::Unsupported(format!(
                "qwen2 vision head_dim {} is past the attention kernel's {}",
                d.head_dim,
                gpu::MAX_ATTENTION_HEAD_DIM
            )));
        }
        // Every block attends over the whole page and the merger reads merge
        // groups, so a page whose patch count is not a whole number of merge
        // groups has nothing to reshape into.
        if d.merge * d.merge == 0 {
            return Err(RealForwardError::Unsupported(
                "qwen2 vision spatial_merge_size is zero".to_string(),
            ));
        }
        Ok(d)
    }

    fn merged_width(&self) -> usize {
        self.merge * self.merge * self.hidden
    }

    fn role_elems(&self, role: R) -> usize {
        let (h, i) = (self.hidden, self.intermediate);
        match role {
            R::N1W | R::N1B | R::N2W | R::N2B | R::ProjB | R::Fc2B => h,
            R::QkvW => 3 * h * h,
            R::QkvB => 3 * h,
            R::ProjW => h * h,
            R::Fc1W => i * h,
            R::Fc1B => i,
            R::Fc2W => h * i,
        }
    }
}

/// One block's twelve byte offsets within its blob.
#[derive(Debug, Clone)]
pub(crate) struct Qwen2BlockRoles {
    offsets: [u64; 12],
}

impl Qwen2BlockRoles {
    fn at(&self, role: R) -> u64 {
        self.offsets[role as usize]
    }

    /// Resolve and size-check one block's layout entry.
    pub(crate) fn resolve(
        entry: &model_io::ExpertEntry,
        stride: u64,
        d: &Qwen2Dims,
    ) -> Result<Self, RealForwardError> {
        let block = entry.expert;
        let mut offsets = [0u64; 12];
        for role in ALL_ROLES {
            let name = QWEN2_BLOCK_ROLES[role as usize];
            let sub = entry.sub_tensors.get(name).ok_or_else(|| {
                RealForwardError::MissingTensor(format!("qwen2 vision block {block} {name}"))
            })?;
            let want = (d.role_elems(role) * 2) as u64;
            if sub.dtype != "fp16" || sub.size != want {
                return Err(RealForwardError::Unsupported(format!(
                    "qwen2 vision block {block} {name}: {} bytes of {}, expected {want} of \
                     fp16",
                    sub.size, sub.dtype
                )));
            }
            if sub.offset % 4 != 0 || sub.offset + sub.size > stride {
                return Err(RealForwardError::Unsupported(format!(
                    "qwen2 vision block {block} {name} at {}..{} does not fit a 4-byte \
                     aligned slot of a {stride}-byte blob",
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
    pub(crate) roles: &'a Qwen2BlockRoles,
}

impl BlockBinding<'_> {
    fn w(&self, role: R) -> (&gpu::MetalBuffer, u64) {
        (self.buffer, self.base + self.roles.at(role))
    }
}

/// Device bytes `Qwen2Scratch::allocate` asks for on a page of `seq` patches.
/// `vision::budget` solves the pixel ceiling against this, so it lists every
/// buffer that struct holds; a unit test sums the real lengths against it.
/// Full attention adds no `seq x seq` buffer: the kernel streams the keys
/// through threadgroup memory, so scratch is linear in the page.
pub(crate) fn scratch_bytes(d: &Qwen2Dims, seq: usize, tile: usize) -> u64 {
    let merged = seq / (d.merge * d.merge);
    let tile = tile.min(seq).max(1);
    let fp16 = seq * d.patch_dim // rows
        + seq * d.hidden * 7 // x, normed, q, k, v, attn, proj
        + tile * d.intermediate // h1
        + merged * d.merged_width() // merger_hidden
        + merged * d.out_hidden; // out
    let f32s = seq * (d.head_dim / 2); // freqs
    (fp16 * 2 + f32s * 4) as u64
}

/// Per-image GPU scratch.
pub(crate) struct Qwen2Scratch {
    seq: usize,
    tile: usize,
    /// Patch rows (FP16), page order (there is no window order on this kind).
    rows: gpu::MetalBuffer,
    /// The residual stream.
    x: gpu::MetalBuffer,
    normed: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    k: gpu::MetalBuffer,
    v: gpu::MetalBuffer,
    attn: gpu::MetalBuffer,
    proj: gpu::MetalBuffer,
    /// The block MLP's hidden layer, `[tile, intermediate]`.
    h1: gpu::MetalBuffer,
    /// Merger hidden layer, `[merged, merge^2 * hidden]`.
    merger_hidden: gpu::MetalBuffer,
    out: gpu::MetalBuffer,
    freqs: gpu::MetalBuffer,
}

impl Qwen2Scratch {
    fn allocate(
        context: &gpu::MetalContext,
        d: &Qwen2Dims,
        seq: usize,
        tile: usize,
        rows: &[half::f16],
        freqs: &[f32],
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
            h1: context.new_output_buffer(mlp),
            merger_hidden: context.new_output_buffer((merged * d.merged_width() * 2) as u64),
            out: context.new_output_buffer((merged * d.out_hidden * 2) as u64),
            freqs: context.new_buffer_with_data(freqs),
        }
    }
}

/// Encode one block over `s.x` in place. Every block runs the same full
/// attention; there is no window index to consult on this kind.
pub(crate) fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    b: &BlockBinding<'_>,
    s: &Qwen2Scratch,
    d: &Qwen2Dims,
) -> Result<(), RealForwardError> {
    let e = RealForwardError::Gpu;
    let seq = s.seq as u32;
    let hidden = d.hidden as u32;
    let wide = (s.seq * d.hidden) as u32;
    let scale = (d.head_dim as f32).powf(-0.5);

    // --- attention half ---
    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        b.w(R::N1W),
        b.w(R::N1B),
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
        let rope = if d.mage {
            gpu::encode_vision_rope_interleaved
        } else {
            gpu::encode_vision_rope_2d
        };
        rope(
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

    // FULL attention on every block: the reference has no window machinery,
    // so the whole page is one bidirectional segment.
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

    // --- MLP half: fc2(gelu(fc1(x))), row-tiled ---
    gpu::encode_vision_layer_norm(
        context,
        pass,
        (&s.x, 0),
        b.w(R::N2W),
        b.w(R::N2B),
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
            b.w(R::Fc1W),
            Some(b.w(R::Fc1B)),
            (&s.h1, 0),
            rows as u32,
            hidden,
            inter,
        )
        .map_err(e)?;
        // The reference's `nn.GELU(approx="fast")`: the fast SIGMOID
        // approximation (see the module doc), NOT the erf form and NOT tanh.
        gpu::encode_vision_gelu(
            context,
            pass,
            (&s.h1, 0),
            (rows * d.intermediate) as u32,
            if d.mage {
                gpu::GeluKind::Erf
            } else {
                gpu::GeluKind::Quick
            },
        )
        .map_err(e)?;
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.h1, 0),
            b.w(R::Fc2W),
            Some(b.w(R::Fc2B)),
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
struct Qwen2Resident {
    patch_embed: u64,
    /// Mage-ViT's `layernorm_pre` weight and bias.
    pre_norm: Option<(u64, u64)>,
    ln_q_w: u64,
    ln_q_b: u64,
    fc1_w: u64,
    fc1_b: u64,
    fc2_w: u64,
    fc2_b: u64,
}

impl Qwen2Resident {
    fn resolve(index: &ResidentIndex, d: &Qwen2Dims) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let mw = d.merged_width();
        Ok(Self {
            patch_embed: fp16_view(
                index,
                &name("patch_embed.proj.weight"),
                d.hidden * d.patch_dim,
            )?,
            pre_norm: if d.mage {
                Some((
                    fp16_view(index, &name("layernorm_pre.weight"), d.hidden)?,
                    fp16_view(index, &name("layernorm_pre.bias"), d.hidden)?,
                ))
            } else {
                None
            },
            ln_q_w: fp16_view(index, &name("merger.ln_q.weight"), d.hidden)?,
            ln_q_b: fp16_view(index, &name("merger.ln_q.bias"), d.hidden)?,
            fc1_w: fp16_view(index, &name("merger.mlp.0.weight"), mw * mw)?,
            fc1_b: fp16_view(index, &name("merger.mlp.0.bias"), mw)?,
            fc2_w: fp16_view(index, &name("merger.mlp.2.weight"), d.out_hidden * mw)?,
            fc2_b: fp16_view(index, &name("merger.mlp.2.bias"), d.out_hidden)?,
        })
    }
}

/// The Qwen2-VL tower opened against an install: blocks streamed from
/// `packed_vision/` into [`VISION_SLOTS`] slots, residents resolved off the
/// trunk's (or an attached sidecar's) index. Opened lazily on the first
/// image.
pub(crate) struct Qwen2VisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<Qwen2BlockRoles>,
    dims: Qwen2Dims,
    resident: Qwen2Resident,
    /// A sidecar's own weights, `Some` only when opened against one.
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
    pub(crate) last_scratch_bytes: u64,
}

impl Qwen2VisionTower {
    /// Open against an install that carries the tower itself.
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let dims = Qwen2Dims::resolve(&arch.vision)?;
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
        let mut blocks: Vec<Option<Qwen2BlockRoles>> = vec![None; dims.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, dims.depth
                ))
            })?;
            if slot
                .replace(Qwen2BlockRoles::resolve(entry, layer.expert_stride, &dims)?)
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
        .map_err(|e| RealForwardError::Unsupported(format!("qwen2 vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }
        let resident = Qwen2Resident::resolve(index, &dims)?;
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

    /// Encode one image to the `[merged, out_hidden]` rows the trunk takes,
    /// in the page's own order (there is no window permutation to undo).
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
                "preprocessing produced patch {} / merge {} / {}-value rows but this tower \
                 takes patch {} / merge {} / {}",
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
                "qwen2 vision input: {} values for {seq} patches of {}",
                image.patch_rows.len(),
                d.patch_dim
            )));
        }
        if seq % (d.merge * d.merge) != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "qwen2 vision grid {}x{}x{} gives {seq} patches, which is not a whole number \
                 of {}x{} merge groups",
                image.grid.t, image.grid.h, image.grid.w, d.merge, d.merge
            )));
        }

        // Host prep, all in page order: the patch rows, and the rope rows
        // (kept F32 to the GPU, the angle at pair 0 is the raw coordinate).
        let rows16: Vec<half::f16> = image
            .patch_rows
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        let rope_rows = if d.mage {
            turbospark_vision_io::rope::mage_vit_rope_rows
        } else {
            turbospark_vision_io::rope::vision_rope_freq_rows_default
        };
        let freqs = rope_rows(image.grid, d.head_dim, params)
            .map_err(|e| RealForwardError::Unsupported(format!("vision rope rows: {e}")))?;

        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let per_unit = d.merge * d.merge;
        let merged = seq / per_unit;
        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let resident = &self.resident;
        let last_scratch = &mut self.last_scratch_bytes;

        gpu::autorelease_pool(|| {
            let e = RealForwardError::Gpu;
            let s = Qwen2Scratch::allocate(context, &d, seq, QWEN2_MLP_TILE_ROWS, &rows16, &freqs);
            *last_scratch = scratch_bytes(&d, seq, QWEN2_MLP_TILE_ROWS);
            let wide = seq * d.hidden;

            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.rows, 0),
                (w.buffer(), w.gpu_offset(resident.patch_embed)),
                None,
                (
                    if resident.pre_norm.is_some() {
                        &s.normed
                    } else {
                        &s.x
                    },
                    0,
                ),
                seq as u32,
                d.patch_dim as u32,
                d.hidden as u32,
            )
            .map_err(e)?;
            if let Some((weight, bias)) = resident.pre_norm {
                gpu::encode_vision_layer_norm(
                    context,
                    &pass,
                    (&s.normed, 0),
                    (w.buffer(), w.gpu_offset(weight)),
                    (w.buffer(), w.gpu_offset(bias)),
                    (&s.x, 0),
                    seq as u32,
                    d.hidden as u32,
                    EPS,
                )
                .map_err(e)?;
            }
            pass.commit_and_wait();
            if let Some(c) = capture.as_mut() {
                c.patch_embed = read_bits(&s.x, wide);
            }

            for (n, block_roles) in blocks.iter().enumerate() {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|err| {
                    RealForwardError::Unsupported(format!("qwen2 vision block {n} read: {err}"))
                })?;
                let binding = BlockBinding {
                    buffer: &slots[slot],
                    base: 0,
                    roles: block_roles,
                };
                let pass = context.begin_pass();
                encode_block(context, &pass, &binding, &s, &d)?;
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

            // The merger: LayerNorm (weight AND bias) over each patch row,
            // then the four patches of a merge group read as one
            // merge^2 * hidden row (a view, the rows are already grouped),
            // fc1, exact-erf GELU, fc2.
            let mw = d.merged_width() as u32;
            let pass = context.begin_pass();
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&s.x, 0),
                (w.buffer(), w.gpu_offset(resident.ln_q_w)),
                (w.buffer(), w.gpu_offset(resident.ln_q_b)),
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
            // The merger's activation is a bare `nn.GELU()`, the exact erf
            // form -- same as the 2.5 tower's merger, unlike this kind's own
            // block MLP above.
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

            // The rows are already in the page's own order.
            let rows = read_bits(&s.out, merged * d.out_hidden);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `mlx-community/Qwen2-VL-2B-Instruct-4bit` tower:
    /// embed_dim 1280, 16 heads, mlp_ratio 4 (5120), depth 32, patch 14,
    /// temporal 2, merge 2, trunk width 1536.
    fn config() -> VisionConfig {
        VisionConfig {
            depth: 32,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            spatial_merge_size: 2,
            num_position_embeddings: 0,
            out_hidden_size: 1536,
            mrope_section: [16, 24, 24],
            deepstack_visual_indexes: Vec::new(),
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            image_token_id: 151655,
            video_token_id: 151656,
            tower: VisionTowerKind::Qwen2Vl,
        }
    }

    #[test]
    fn the_published_shape_resolves() {
        let d = Qwen2Dims::resolve(&config()).unwrap();
        assert_eq!(
            (
                d.depth,
                d.hidden,
                d.intermediate,
                d.heads,
                d.head_dim,
                d.patch_dim,
                d.out_hidden
            ),
            (32, 1280, 5120, 16, 80, 2 * 14 * 14 * 3, 1536)
        );
    }

    #[test]
    fn a_shape_the_tower_cannot_run_is_refused() {
        let mut c = config();
        c.tower = VisionTowerKind::Qwen25Vl;
        assert!(Qwen2Dims::resolve(&c).is_err(), "foreign kind refused");
        let mut c = config();
        c.hidden_size = 1000; // not divisible by 16 heads
        assert!(Qwen2Dims::resolve(&c).is_err());
        let mut c = config();
        c.num_heads = 15; // 1280 / 15 is not whole
        assert!(Qwen2Dims::resolve(&c).is_err());
        let mut c = config();
        c.depth = 0; // the inactive sentinel
        assert!(Qwen2Dims::resolve(&c).is_err());
        let mut c = config();
        c.intermediate_size = -4;
        assert!(Qwen2Dims::resolve(&c).is_err());
    }

    /// MUTATION GUARD: the role table maps blob sub-tensors to weights by
    /// NAME and size. The two LayerNorm weights and the two fc biases are
    /// the same width, so a swap between same-shaped slots would pass every
    /// size check; the guard is the exact table content itself, pinned so
    /// any edit here is a decision. It also pins the three roles the real
    /// gate's tower run leans on: qkv (fused 3h), fc1 (i x h) and fc2
    /// (h x i).
    #[test]
    fn the_role_table_and_sizes_are_pinned() {
        assert_eq!(QWEN2_BLOCK_ROLES.len(), 12);
        assert_eq!(QWEN2_BLOCK_ROLES[0], "n1_w");
        assert_eq!(QWEN2_BLOCK_ROLES[1], "n1_b");
        assert_eq!(QWEN2_BLOCK_ROLES[8], "fc1_w");
        assert_eq!(QWEN2_BLOCK_ROLES[10], "fc2_w");
        assert_eq!(QWEN2_BLOCK_ROLES[11], "fc2_b");
        let (h, i) = (1280usize, 5120usize);
        let elems = |name: &str| {
            let role = QWEN2_BLOCK_ROLES
                .iter()
                .position(|r| *r == name)
                .map(|p| ALL_ROLES[p])
                .unwrap();
            Qwen2Dims {
                depth: 32,
                hidden: h,
                intermediate: i,
                heads: 16,
                head_dim: 80,
                patch_dim: 1176,
                patch_size: 14,
                merge: 2,
                out_hidden: 1536,
                mage: false,
            }
            .role_elems(role)
        };
        assert_eq!(elems("n1_b"), h);
        assert_eq!(elems("qkv_w"), 3 * h * h);
        assert_eq!(elems("qkv_b"), 3 * h);
        assert_eq!(elems("fc1_w"), i * h);
        assert_eq!(elems("fc1_b"), i);
        assert_eq!(elems("fc2_w"), h * i);
        assert_eq!(elems("fc2_b"), h);
    }

    /// The merger's input width on the published tower: the shuffle makes
    /// fc1 SQUARE at merge^2 * embed_dim (4 x 1280 = 5120), and fc2 maps
    /// that into the 1536-wide trunk. Pinning the arithmetic keeps a config
    /// change away from a silent stride change.
    #[test]
    fn the_merger_width_is_the_shuffle_then_the_trunk() {
        let d = Qwen2Dims::resolve(&config()).unwrap();
        assert_eq!(d.merged_width(), 5120);
        assert_eq!(d.out_hidden, 1536);
    }

    #[test]
    fn the_roles_match_the_repack_writer_by_name_and_suffix() {
        let written = turbospark_repack::VISION_QWEN2_VL_BLOCK_ROLES;
        assert_eq!(written.map(|(role, _)| role), QWEN2_BLOCK_ROLES);
        assert_eq!(
            written.map(|(_, suffix)| suffix),
            [
                "norm1.weight",
                "norm1.bias",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "norm2.weight",
                "norm2.bias",
                "mlp.fc1.weight",
                "mlp.fc1.bias",
                "mlp.fc2.weight",
                "mlp.fc2.bias",
            ]
        );
    }

    /// `x @ w^T + b`, `w` row-major by output, as `nn.Linear` ships it.
    fn linear(x: &[f32], w: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
        x.chunks(k)
            .flat_map(|row| {
                (0..n).map(move |o| {
                    b[o] + row.iter().zip(&w[o * k..]).map(|(a, w)| a * w).sum::<f32>()
                })
            })
            .collect()
    }

    fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        x.chunks(w.len())
            .flat_map(|row| {
                let n = row.len() as f32;
                let mean = row.iter().sum::<f32>() / n;
                let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
                let inv = 1.0 / (var + EPS).sqrt();
                row.iter()
                    .enumerate()
                    .map(move |(i, v)| (v - mean) * inv * w[i] + b[i])
            })
            .collect()
    }

    /// The budget formula against the buffers really allocated, with a tile
    /// smaller than the page so the MLP cap is exercised.
    #[cfg(target_os = "macos")]
    #[test]
    fn scratch_bytes_is_the_sum_of_the_allocated_buffers() {
        let d = Qwen2Dims {
            depth: 1,
            hidden: 16,
            intermediate: 40,
            heads: 2,
            head_dim: 8,
            patch_dim: 12,
            patch_size: 14,
            merge: 2,
            out_hidden: 24,
            mage: false,
        };
        let (seq, tile) = (32usize, 5usize);
        let context = gpu::MetalContext::new().expect("a Metal device");
        let rows = vec![half::f16::ZERO; seq * d.patch_dim];
        let freqs = vec![0f32; seq * d.head_dim / 2];
        let s = Qwen2Scratch::allocate(&context, &d, seq, tile, &rows, &freqs);
        // Exhaustive on purpose: a new buffer stops this compiling until it is
        // counted here and in `scratch_bytes`.
        let Qwen2Scratch {
            seq: _,
            tile: _,
            rows,
            x,
            normed,
            q,
            k,
            v,
            attn,
            proj,
            h1,
            merger_hidden,
            out,
            freqs,
        } = &s;
        let total: u64 = [
            rows,
            x,
            normed,
            q,
            k,
            v,
            attn,
            proj,
            h1,
            merger_hidden,
            out,
            freqs,
        ]
        .iter()
        .map(|b| b.length())
        .sum();
        assert_eq!(scratch_bytes(&d, seq, tile), total);
    }

    /// One block on Metal against a plain CPU transcription of the
    /// reference's `Qwen2VLVisionBlock`, at tiny widths with an MLP tile
    /// smaller than the page.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_block_matches_a_cpu_transcription_of_the_reference() {
        let d = Qwen2Dims {
            depth: 1,
            hidden: 16,
            intermediate: 32,
            heads: 2,
            head_dim: 8,
            patch_dim: 4,
            patch_size: 14,
            merge: 2,
            out_hidden: 8,
            mage: false,
        };
        let (h, seq, half) = (d.hidden, 8usize, d.head_dim / 2);
        let mut seed = 0u32;
        let mut noise = |len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    seed += 1;
                    // Rounded through FP16 so both sides read the same value.
                    half::f16::from_f32((seed as f32 * 12.9898).sin() * scale).to_f32()
                })
                .collect()
        };
        let weights: Vec<Vec<f32>> = ALL_ROLES
            .iter()
            .map(|&r| noise(d.role_elems(r), 0.6))
            .collect();
        let x = noise(seq * h, 1.5);
        let freqs = noise(seq * half, 3.0);
        let w = |r: R| &weights[r as usize][..];

        // CPU reference, rounded to FP16 wherever the GPU stores a buffer.
        let stored = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter()
                .map(|v| half::f16::from_f32(v).to_f32())
                .collect()
        };
        let normed = stored(layer_norm(&x, w(R::N1W), w(R::N1B)));
        let qkv: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                stored(linear(
                    &normed,
                    &w(R::QkvW)[i * h * h..][..h * h],
                    &w(R::QkvB)[i * h..][..h],
                    h,
                    h,
                ))
            })
            .collect();
        let rotate = |t: &[f32]| -> Vec<f32> {
            let mut out = t.to_vec();
            for s in 0..seq {
                for head in 0..d.heads {
                    let base = s * h + head * d.head_dim;
                    for i in 0..half {
                        let (sin, cos) = freqs[s * half + i].sin_cos();
                        let (a, b) = (t[base + i], t[base + i + half]);
                        out[base + i] = a * cos - b * sin;
                        out[base + i + half] = b * cos + a * sin;
                    }
                }
            }
            out
        };
        let (q, k, v) = (stored(rotate(&qkv[0])), stored(rotate(&qkv[1])), &qkv[2]);
        let mut attn = vec![0f32; seq * h];
        for head in 0..d.heads {
            let at = |s: usize| s * h + head * d.head_dim;
            for i in 0..seq {
                let scores: Vec<f32> = (0..seq)
                    .map(|j| {
                        (0..d.head_dim)
                            .map(|e| q[at(i) + e] * k[at(j) + e])
                            .sum::<f32>()
                            / (d.head_dim as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::MIN, f32::max);
                let total: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                for (j, score) in scores.iter().enumerate() {
                    let p = (score - max).exp() / total;
                    for e in 0..d.head_dim {
                        attn[at(i) + e] += p * v[at(j) + e];
                    }
                }
            }
        }
        let proj = stored(linear(&stored(attn), w(R::ProjW), w(R::ProjB), h, h));
        let mid = stored(x.iter().zip(&proj).map(|(a, b)| a + b).collect());
        let normed = stored(layer_norm(&mid, w(R::N2W), w(R::N2B)));
        let mut up = stored(linear(&normed, w(R::Fc1W), w(R::Fc1B), h, d.intermediate));
        for v in &mut up {
            *v /= 1.0 + (-1.702 * *v).exp();
        }
        let down = stored(linear(
            &stored(up),
            w(R::Fc2W),
            w(R::Fc2B),
            d.intermediate,
            h,
        ));
        let want = stored(mid.iter().zip(&down).map(|(a, b)| a + b).collect());

        // Metal, from one blob holding the twelve roles back to back.
        let mut context = gpu::MetalContext::new().expect("a Metal device");
        let to16 =
            |v: &[f32]| -> Vec<half::f16> { v.iter().map(|&v| half::f16::from_f32(v)).collect() };
        let mut blob = Vec::new();
        let mut offsets = [0u64; 12];
        for (offset, values) in offsets.iter_mut().zip(&weights) {
            *offset = (blob.len() * 2) as u64;
            blob.extend(to16(values));
        }
        let blob = context.new_buffer_with_data(&blob);
        let roles = Qwen2BlockRoles { offsets };
        let rows = vec![half::f16::ZERO; seq * d.patch_dim];
        let mut s = Qwen2Scratch::allocate(&context, &d, seq, 3, &rows, &freqs);
        s.x = context.new_buffer_with_data(&to16(&x));
        let binding = BlockBinding {
            buffer: &blob,
            base: 0,
            roles: &roles,
        };
        let pass = context.begin_pass();
        encode_block(&mut context, &pass, &binding, &s, &d).expect("encodes");
        pass.commit_and_wait();
        let got = gpu::read_buffer_f16(&s.x, 0, seq * h);

        let worst = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g.to_f32()).abs() / w.abs().max(1.0))
            .fold(0f32, f32::max);
        eprintln!("worst relative error {worst}");
        assert!(worst < 6e-3, "block output differs by {worst}");
    }
}
