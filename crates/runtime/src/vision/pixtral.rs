//! The Pixtral vision tower on Metal (`docs/PIXTRAL_VISION_PHASE0.md`).
//!
//! A sibling of the Qwen2.5 tower in this module and of the CLIP one: blocks
//! stream two slots at a time from `packed_vision/`, residents are FP16 under
//! `vision.`, and the kernels are the shared ones (matmul, bidirectional
//! attention, the 2-D rope, SiLU, mul, RMSNorm, GELU, residual add). What is
//! Pixtral's own is the block (RMSNorm, no biases anywhere, a gated SiLU MLP),
//! the rope row layout (`vision_io::pixtral_rope_freq_rows`), and the whole
//! image attending to itself with no windows, no class row and no position
//! table.
//!
//! The output is `[patches, out_hidden]` FP16 rows, row-major over the image's
//! grid. The prompt puts `[IMG_BREAK]` between patch rows, so the caller
//! injects them one grid row at a time (`set_prompt_vision_flat`).

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::ClipPatches;

use super::clip::ClipStages;
use super::weights::{fp16_view, VISION_PREFIX};
use super::VISION_SLOTS;
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// RMSNorm epsilon: `ln_pre` constructs `nn.RMSNorm(hidden)` with MLX's
/// default and every block norm states `rms_norm_eps` 1e-5.
const EPS: f32 = 1e-5;

/// The tower's rope theta (`vision_config.rope_theta`), checked at intake.
const ROPE_THETA: f32 = 10_000.0;

/// MLP row tile: the `[rows, intermediate]` gate and up buffers are sized for
/// a tile, not the image.
pub(crate) const PIXTRAL_MLP_TILE_ROWS: usize = 1024;

/// An image past this many patches is refused rather than allocated: the
/// preprocessor caps a side at `longest_edge / patch` = 64, so 4096 is the
/// largest image the shipped processor emits.
const MAX_PATCHES: usize = 4096;

/// The blob roles in byte order, the repack writer's `PIXTRAL_BLOCK_ROLES`.
/// Restated rather than imported because repack is a dev-dependency.
pub(crate) const PIXTRAL_BLOCK_ROLES: [&str; 7] = [
    "n1_w", "qkv_w", "proj_w", "n2_w", "gate_w", "up_w", "down_w",
];

#[derive(Debug, Clone, Copy)]
enum R {
    N1W = 0,
    QkvW = 1,
    ProjW = 2,
    N2W = 3,
    GateW = 4,
    UpW = 5,
    DownW = 6,
}

const ALL_ROLES: [R; 7] = [
    R::N1W,
    R::QkvW,
    R::ProjW,
    R::N2W,
    R::GateW,
    R::UpW,
    R::DownW,
];

/// The tower's geometry, resolved once from the install's [`VisionConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PixtralDims {
    pub(crate) depth: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) patch_dim: usize,
    pub(crate) out_hidden: usize,
}

impl PixtralDims {
    pub(crate) fn resolve(v: &VisionConfig) -> Result<Self, RealForwardError> {
        if v.tower != VisionTowerKind::Pixtral || !v.is_active() {
            return Err(RealForwardError::Unsupported(format!(
                "not an active pixtral vision tower (kind {})",
                v.tower.as_str()
            )));
        }
        let us = |x: i64, what: &str| {
            usize::try_from(x)
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| RealForwardError::Unsupported(format!("pixtral vision {what} {x}")))
        };
        let d = Self {
            depth: us(v.depth, "depth")?,
            hidden: us(v.hidden_size, "hidden_size")?,
            intermediate: us(v.intermediate_size, "intermediate_size")?,
            heads: us(v.num_heads, "num_heads")?,
            head_dim: us(v.head_dim(), "head_dim")?,
            patch_dim: us(3 * v.patch_size * v.patch_size, "patch_dim")?,
            out_hidden: us(v.out_hidden_size, "out_hidden_size")?,
        };
        if d.heads * d.head_dim != d.hidden || d.head_dim % 4 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "pixtral vision hidden {} over {} heads does not give a head_dim divisible by 4",
                d.hidden, d.heads
            )));
        }
        if d.head_dim as u32 > gpu::MAX_ATTENTION_HEAD_DIM {
            return Err(RealForwardError::Unsupported(format!(
                "pixtral vision head_dim {} is past the attention kernel's {}",
                d.head_dim,
                gpu::MAX_ATTENTION_HEAD_DIM
            )));
        }
        Ok(d)
    }

    fn role_elems(&self, role: R) -> usize {
        let (h, i) = (self.hidden, self.intermediate);
        match role {
            R::N1W | R::N2W => h,
            R::QkvW => 3 * h * h,
            R::ProjW => h * h,
            R::GateW | R::UpW | R::DownW => i * h,
        }
    }
}

/// One block's seven byte offsets within its blob.
#[derive(Debug, Clone)]
pub(crate) struct PixtralBlockRoles {
    offsets: [u64; 7],
}

impl PixtralBlockRoles {
    fn at(&self, role: R) -> u64 {
        self.offsets[role as usize]
    }

    /// Resolve and size-check one block's layout entry.
    pub(crate) fn resolve(
        entry: &model_io::ExpertEntry,
        stride: u64,
        d: &PixtralDims,
    ) -> Result<Self, RealForwardError> {
        let block = entry.expert;
        let mut offsets = [0u64; 7];
        for role in ALL_ROLES {
            let name = PIXTRAL_BLOCK_ROLES[role as usize];
            let sub = entry.sub_tensors.get(name).ok_or_else(|| {
                RealForwardError::MissingTensor(format!("pixtral vision block {block} {name}"))
            })?;
            let want = (d.role_elems(role) * 2) as u64;
            if sub.dtype != "fp16" || sub.size != want {
                return Err(RealForwardError::Unsupported(format!(
                    "pixtral vision block {block} {name}: {} bytes of {}, expected {want} of fp16",
                    sub.size, sub.dtype
                )));
            }
            if sub.offset % 4 != 0 || sub.offset + sub.size > stride {
                return Err(RealForwardError::Unsupported(format!(
                    "pixtral vision block {block} {name} at {}..{} does not fit a 4-byte \
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

/// Where one block's weights are bound from: a streamer slot (base 0).
pub(crate) struct BlockBinding<'a> {
    pub(crate) buffer: &'a gpu::MetalBuffer,
    pub(crate) base: u64,
    pub(crate) roles: &'a PixtralBlockRoles,
}

impl BlockBinding<'_> {
    fn w(&self, role: R) -> (&gpu::MetalBuffer, u64) {
        (self.buffer, self.base + self.roles.at(role))
    }
}

/// Per-image GPU scratch.
pub(crate) struct PixtralScratch {
    seq: usize,
    tile: usize,
    rows: gpu::MetalBuffer,
    x: gpu::MetalBuffer,
    normed: gpu::MetalBuffer,
    q: gpu::MetalBuffer,
    k: gpu::MetalBuffer,
    v: gpu::MetalBuffer,
    attn: gpu::MetalBuffer,
    proj: gpu::MetalBuffer,
    gate: gpu::MetalBuffer,
    up: gpu::MetalBuffer,
    /// The projector's hidden layer and output, `[seq, out_hidden]` each.
    proj_hidden: gpu::MetalBuffer,
    out: gpu::MetalBuffer,
    freqs: gpu::MetalBuffer,
}

impl PixtralScratch {
    fn allocate(
        context: &gpu::MetalContext,
        d: &PixtralDims,
        seq: usize,
        tile: usize,
        rows: &[half::f16],
        freqs: &[f32],
    ) -> Self {
        let wide = (seq * d.hidden * 2) as u64;
        let tile = tile.min(seq).max(1);
        let mlp = (tile * d.intermediate * 2) as u64;
        let proj = (seq * d.out_hidden * 2) as u64;
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
            proj_hidden: context.new_output_buffer(proj),
            out: context.new_output_buffer(proj),
            freqs: context.new_buffer_with_data(freqs),
        }
    }

    fn bytes(&self, d: &PixtralDims) -> u64 {
        (self.seq * d.hidden * 2 * 7
            + 2 * self.tile * d.intermediate * 2
            + 2 * self.seq * d.out_hidden * 2) as u64
    }
}

/// Encode one block over `s.x` in place.
pub(crate) fn encode_block(
    context: &mut gpu::MetalContext,
    pass: &gpu::PassEncoder,
    b: &BlockBinding<'_>,
    s: &PixtralScratch,
    d: &PixtralDims,
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

    // q, k, v are the fused weight's three row blocks, read at byte offsets.
    // No biases: `Attention` constructs every Linear with `bias=False`.
    let w_block = (d.hidden * d.hidden * 2) as u64;
    let (qkv_w, qkv_off) = b.w(R::QkvW);
    for (i, out) in [&s.q, &s.k, &s.v].into_iter().enumerate() {
        gpu::encode_vision_matmul(
            context,
            pass,
            (&s.normed, 0),
            (qkv_w, qkv_off + i as u64 * w_block),
            None,
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

    // The whole image attends to itself: one image, so the reference's
    // block-diagonal mask is a single all-zero block.
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
        None,
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
            None,
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
            None,
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
            None,
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
struct PixtralResident {
    patch_embed: u64,
    ln_pre: u64,
    linear_1_w: u64,
    linear_1_b: u64,
    linear_2_w: u64,
    linear_2_b: u64,
}

impl PixtralResident {
    fn resolve(index: &ResidentIndex, d: &PixtralDims) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        Ok(Self {
            patch_embed: fp16_view(
                index,
                &name("patch_embed.proj.weight"),
                d.hidden * d.patch_dim,
            )?,
            ln_pre: fp16_view(index, &name("ln_pre.weight"), d.hidden)?,
            linear_1_w: fp16_view(
                index,
                &name("projector.linear_1.weight"),
                d.out_hidden * d.hidden,
            )?,
            linear_1_b: fp16_view(index, &name("projector.linear_1.bias"), d.out_hidden)?,
            linear_2_w: fp16_view(
                index,
                &name("projector.linear_2.weight"),
                d.out_hidden * d.out_hidden,
            )?,
            linear_2_b: fp16_view(index, &name("projector.linear_2.bias"), d.out_hidden)?,
        })
    }
}

/// The Pixtral tower opened against an install: blocks streamed from
/// `packed_vision/` into [`VISION_SLOTS`] slots, residents resolved off the
/// trunk's (or an attached sidecar's) index. Opened lazily on the first image.
pub(crate) struct PixtralVisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<PixtralBlockRoles>,
    dims: PixtralDims,
    resident: PixtralResident,
    /// A sidecar's own weights, `Some` only when opened against one.
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
    pub(crate) last_scratch_bytes: u64,
}

impl PixtralVisionTower {
    /// Open against an install that carries the tower itself.
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let dims = PixtralDims::resolve(&arch.vision)?;
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
        let mut blocks: Vec<Option<PixtralBlockRoles>> = vec![None; dims.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names block {} past depth {}",
                    entry.expert, dims.depth
                ))
            })?;
            if slot
                .replace(PixtralBlockRoles::resolve(
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
        .map_err(|e| RealForwardError::Unsupported(format!("pixtral vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }
        let resident = PixtralResident::resolve(index, &dims)?;
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

    /// Encode one image to `[patches, out_hidden]` FP16 bits, row-major over
    /// the image's grid, plus the residual stream at three points when
    /// `capture` is set.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        patches: &ClipPatches,
        capture: bool,
    ) -> Result<(Vec<u16>, Option<ClipStages>), RealForwardError> {
        let d = self.dims;
        let seq = patches.num_patches();
        if seq == 0 || seq > MAX_PATCHES || patches.pixels.len() != seq * d.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "pixtral vision input: {} values for {seq} patches of {} (at most {MAX_PATCHES})",
                patches.pixels.len(),
                d.patch_dim
            )));
        }
        let rows16: Vec<half::f16> = patches
            .pixels
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        // F32 to the GPU: the angle at the first pair is the raw patch
        // coordinate and reaches the tens on a wide grid.
        let freqs = turbospark_vision_io::pixtral_rope_freq_rows(
            patches.grid,
            patches.grid_w,
            d.head_dim,
            ROPE_THETA,
        )
        .map_err(|e| RealForwardError::Unsupported(format!("pixtral rope rows: {e}")))?;

        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let resident = &self.resident;
        let last_scratch = &mut self.last_scratch_bytes;
        let mut stages = capture.then(ClipStages::default);

        gpu::autorelease_pool(|| {
            let e = RealForwardError::Gpu;
            let s =
                PixtralScratch::allocate(context, &d, seq, PIXTRAL_MLP_TILE_ROWS, &rows16, &freqs);
            *last_scratch = s.bytes(&d);
            let wide = seq * d.hidden;

            // Patch conv as a GEMM over the channel-major rows, then ln_pre.
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
            gpu::encode_vision_rms_norm(
                context,
                &pass,
                (&s.x, 0),
                Some((w.buffer(), w.gpu_offset(resident.ln_pre))),
                (&s.x, 0),
                seq as u32,
                d.hidden as u32,
                EPS,
            )
            .map_err(e)?;
            pass.commit_and_wait();
            if let Some(c) = stages.as_mut() {
                c.embed = read_bits(&s.x, wide);
            }

            for (n, block_roles) in blocks.iter().enumerate() {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|err| {
                    RealForwardError::Unsupported(format!("pixtral vision block {n} read: {err}"))
                })?;
                let binding = BlockBinding {
                    buffer: &slots[slot],
                    base: 0,
                    roles: block_roles,
                };
                let pass = context.begin_pass();
                encode_block(context, &pass, &binding, &s, &d)?;
                pass.commit_and_wait();
                if let Some(c) = stages.as_mut() {
                    if n == 0 {
                        c.block_first = read_bits(&s.x, wide);
                    }
                    if n + 1 == d.depth {
                        c.block_last = read_bits(&s.x, wide);
                    }
                }
            }

            // The projector: Linear (bias), exact-erf GELU, Linear (bias).
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.x, 0),
                (w.buffer(), w.gpu_offset(resident.linear_1_w)),
                Some((w.buffer(), w.gpu_offset(resident.linear_1_b))),
                (&s.proj_hidden, 0),
                seq as u32,
                d.hidden as u32,
                d.out_hidden as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&s.proj_hidden, 0),
                (seq * d.out_hidden) as u32,
                gpu::GeluKind::Erf,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.proj_hidden, 0),
                (w.buffer(), w.gpu_offset(resident.linear_2_w)),
                Some((w.buffer(), w.gpu_offset(resident.linear_2_b))),
                (&s.out, 0),
                seq as u32,
                d.out_hidden as u32,
                d.out_hidden as u32,
            )
            .map_err(e)?;
            pass.commit_and_wait();

            Ok((read_bits(&s.out, seq * d.out_hidden), stages))
        })
    }
}

fn read_bits(buffer: &gpu::MetalBuffer, count: usize) -> Vec<u16> {
    gpu::read_buffer_f16(buffer, 0, count)
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}
