//! The `gemma4_unified` encoder-free image embedder
//! (`docs/GEMMA4_UNIFIED_VISION_PHASE0.md`), after mlx-vlm
//! `gemma4_unified.VisionEmbedder` and `gemma4.MultimodalEmbedder`:
//!
//! ```text
//! h = LayerNorm(dense(LayerNorm(patch)))      # one 48 px patch per soft token
//! h = LayerNorm(h + pos[x][0] + pos[y][1])
//! t = RMSNorm_noscale(h) @ projection^T
//! ```
//!
//! The pixels stay in `[0, 1]`: the Gemma 4 tower's `2 * (x - 0.5)` has no
//! counterpart here. The reference also pads each image to 280 rows and drops
//! the padding afterwards; nothing mixes rows, so only real rows run.

use std::path::Path;

use model_io::{ArchConfig, VisionConfig, VisionTowerKind};

use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// The three LayerNorms: `nn.LayerNorm`'s default, which the config does not
/// override. NOT the config's `rms_norm_eps`.
const LAYER_NORM_EPS: f32 = 1e-5;
/// The projector's no-scale RMSNorm (`vision_config.rms_norm_eps`).
const RMS_EPS: f32 = 1e-6;

/// The blob roles, the repack writer's `GEMMA4_UNIFIED_ROLES` first column.
/// Restated because repack is a dev-dependency of this crate.
const ROLES: [&str; 10] = [
    "ln1_w", "ln1_b", "dense_w", "dense_b", "ln2_w", "ln2_b", "pos", "posln_w", "posln_b", "proj_w",
];

#[derive(Debug, Clone, Copy)]
struct Dims {
    patch: usize,
    hidden: usize,
    pos_rows: usize,
    text_hidden: usize,
}

impl Dims {
    fn resolve(v: &VisionConfig) -> Result<Self, RealForwardError> {
        let bad = || {
            RealForwardError::Unsupported(format!(
                "not a usable gemma4_unified vision config (kind {}, hidden {}, positions {})",
                v.tower.as_str(),
                v.hidden_size,
                v.num_position_embeddings
            ))
        };
        // The same bound intake applies: these multiply into buffer sizes.
        let dim = |x: i64| {
            usize::try_from(x)
                .ok()
                .filter(|n| (1..=1 << 16).contains(n))
                .ok_or_else(bad)
        };
        if v.tower != VisionTowerKind::Gemma4Unified
            || !v.is_active()
            || (v.patch_size, v.spatial_merge_size, v.in_channels) != (16, 3, 3)
        {
            return Err(bad());
        }
        Ok(Self {
            patch: 48 * 48 * 3,
            hidden: dim(v.hidden_size)?,
            pos_rows: dim(v.num_position_embeddings)?,
            text_hidden: dim(v.out_hidden_size)?,
        })
    }

    fn role_elems(&self, role: &str) -> usize {
        match role {
            "ln1_w" | "ln1_b" => self.patch,
            "dense_w" => self.hidden * self.patch,
            "pos" => self.pos_rows * 2 * self.hidden,
            "proj_w" => self.text_hidden * self.hidden,
            _ => self.hidden,
        }
    }
}

/// The embedder opened against an install. Its one blob is read from
/// `packed_vision/` on the first image.
pub(crate) struct Gemma4UnifiedEmbedder {
    /// Declared before `streamer`: the buffer aliases its allocation.
    blob: gpu::MetalBuffer,
    streamer: streaming::PreadExpertStreamer,
    offsets: [u64; 10],
    dims: Dims,
}

impl Gemma4UnifiedEmbedder {
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let dims = Dims::resolve(&arch.vision)?;
        let layout = model_io::load_packed_layout_from(
            dir,
            model_io::PACKED_VISION_DIR,
            PACKED_LAYOUT_MAX_BYTES,
        )
        .map_err(RealForwardError::Model)?;
        let bad = |m: String| RealForwardError::Unsupported(format!("gemma4_unified vision: {m}"));
        let layer = layout
            .layers
            .iter()
            .find(|l| l.layer == 0)
            .ok_or_else(|| bad("packed_vision/layout.json has no layer 0".into()))?;
        let entry = match layer.experts.as_slice() {
            [entry] if entry.expert == 0 => entry,
            other => return Err(bad(format!("expected one blob, found {}", other.len()))),
        };
        let mut offsets = [0u64; 10];
        for (slot, role) in ROLES.iter().enumerate() {
            let sub = entry.sub_tensors.get(*role).ok_or_else(|| {
                RealForwardError::MissingTensor(format!("gemma4_unified vision {role}"))
            })?;
            // Every dispatch below reads exactly this many bytes of the role.
            let want = (dims.role_elems(role) * 2) as u64;
            let fits = sub
                .offset
                .checked_add(sub.size)
                .is_some_and(|end| end <= layer.expert_stride);
            if sub.dtype != "fp16" || sub.size != want || sub.offset % 4 != 0 || !fits {
                return Err(bad(format!(
                    "{role}: {} bytes of {} at {}, expected {want} of fp16",
                    sub.size, sub.dtype, sub.offset
                )));
            }
            offsets[slot] = sub.offset;
        }
        let stream_layout =
            streaming::StreamLayout::from_packed_layer_in(layer, dir, model_io::PACKED_VISION_DIR);
        let mut streamer = streaming::PreadExpertStreamer::open(
            stream_layout,
            1,
            streaming::ExpertCachePolicy::DEFAULT,
        )
        .map_err(|e| bad(format!("streamer: {e}")))?;
        streamer
            .load_expert_into_slot(0, 0, 0)
            .map_err(|e| bad(format!("blob read: {e}")))?;
        let (ptr, len) = streamer.slot_allocation(0);
        let blob = gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
            .map_err(RealForwardError::Gpu)?;
        Ok(Self {
            blob,
            streamer,
            offsets,
            dims,
        })
    }

    fn at(&self, role: &str) -> (&gpu::MetalBuffer, u64) {
        let slot = ROLES.iter().position(|r| *r == role).expect("a known role");
        (&self.blob, self.offsets[slot])
    }

    /// Encode one image: `[soft_tokens, text_hidden]` FP16 bits.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        patches: &turbospark_vision_io::Gemma4Patches,
    ) -> Result<Vec<u16>, RealForwardError> {
        // Keeps the streamer's allocation, which `blob` aliases, in use.
        self.streamer
            .load_expert_into_slot(0, 0, 0)
            .map_err(|e| RealForwardError::Unsupported(format!("gemma4_unified vision: {e}")))?;
        let d = self.dims;
        let bad = |m: String| RealForwardError::Unsupported(format!("gemma4_unified vision: {m}"));
        let (gw, gh) = (patches.grid_w, patches.grid_h);
        if gw == 0
            || gh == 0
            || gw % 3 != 0
            || gh % 3 != 0
            || gw.checked_mul(gh) != Some(patches.num_patches())
            || patches.pixels.len() != patches.num_patches() * 16 * 16 * 3
        {
            return Err(bad(format!(
                "a {gw}x{gh} grid with {} patches is not whole 48 px blocks",
                patches.num_patches()
            )));
        }
        let (pixels, positions) = patches.model_patches();
        let n = positions.len();
        if let Some(&(x, y)) = positions
            .iter()
            .find(|&&(x, y)| x >= d.pos_rows || y >= d.pos_rows)
        {
            return Err(bad(format!(
                "patch at ({x}, {y}) is past the {}-row position table",
                d.pos_rows
            )));
        }

        // The table is `[positions, 2, hidden]`: x reads column 0, y column 1.
        let (table, table_at) = self.at("pos");
        let row = |p: usize, axis: usize| {
            gpu::read_buffer_f16(
                table,
                table_at as usize + (p * 2 + axis) * d.hidden * 2,
                d.hidden,
            )
        };
        let mut pos = Vec::with_capacity(n * d.hidden);
        for &(x, y) in &positions {
            pos.extend(
                row(x, 0)
                    .into_iter()
                    .zip(row(y, 1))
                    .map(|(a, b)| half::f16::from_f32(a.to_f32() + b.to_f32())),
            );
        }
        let pixels: Vec<half::f16> = pixels.into_iter().map(half::f16::from_f32).collect();
        let (rows, patch, hidden, out) = (
            n as u32,
            d.patch as u32,
            d.hidden as u32,
            d.text_hidden as u32,
        );

        let rows_out = gpu::autorelease_pool(|| {
            let e = RealForwardError::Gpu;
            let patch_bytes = (n * d.patch * 2) as u64;
            let hidden_bytes = (n * d.hidden * 2) as u64;
            let px = context.new_buffer_with_data(&pixels);
            let pb = context.new_buffer_with_data(&pos);
            let normed_px = context.new_output_buffer(patch_bytes);
            let h = context.new_output_buffer(hidden_bytes);
            let h2 = context.new_output_buffer(hidden_bytes);
            let normed = context.new_output_buffer(hidden_bytes);
            let t = context.new_output_buffer((n * d.text_hidden * 2) as u64);

            let pass = context.begin_pass();
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&px, 0),
                self.at("ln1_w"),
                self.at("ln1_b"),
                (&normed_px, 0),
                rows,
                patch,
                LAYER_NORM_EPS,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&normed_px, 0),
                self.at("dense_w"),
                Some(self.at("dense_b")),
                (&h, 0),
                rows,
                patch,
                hidden,
            )
            .map_err(e)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&h, 0),
                self.at("ln2_w"),
                self.at("ln2_b"),
                (&h2, 0),
                rows,
                hidden,
                LAYER_NORM_EPS,
            )
            .map_err(e)?;
            gpu::encode_vision_residual_add(context, &pass, (&h2, 0), (&pb, 0), rows * hidden)
                .map_err(e)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&h2, 0),
                self.at("posln_w"),
                self.at("posln_b"),
                (&normed, 0),
                rows,
                hidden,
                LAYER_NORM_EPS,
            )
            .map_err(e)?;
            gpu::encode_vision_rms_norm(
                context,
                &pass,
                (&normed, 0),
                None,
                (&normed, 0),
                rows,
                hidden,
                RMS_EPS,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&normed, 0),
                self.at("proj_w"),
                None,
                (&t, 0),
                rows,
                hidden,
                out,
            )
            .map_err(e)?;
            pass.commit_and_wait();
            Ok::<_, RealForwardError>(gpu::read_buffer_f16(&t, 0, n * d.text_hidden))
        })?;
        if let Some(i) = rows_out.iter().position(|v| !v.is_finite()) {
            return Err(bad(format!("output is non-finite at {i}")));
        }
        Ok(rows_out.into_iter().map(half::f16::to_bits).collect())
    }
}
