//! The Inkling 2 Mini vision tower on Metal (`docs/INKLING2_VISION_PHASE0.md`).
//!
//! Built from the Muse-Glimmer tower's parts (`muse.rs`): the same streamed
//! blocks, the same block encoder and the same kernels. The repack made the
//! weights fit them: q/k/v fused, q and k permuted from interleaved to
//! half-split rope pairs, and the layer scale folded into `out_proj` and
//! `c_proj` (`crates/repack/src/gemma4_checkpoint/vision_inkling2.rs`). What
//! differs at run time:
//!
//! - **A class row leads each image's sequence.** Its patch row is zero (so
//!   the bias-free patch GEMM leaves it zero) and its position row is
//!   `class_embedding + cls_position_embedding`, so the one position add
//!   places it. Its rope angles are zero (position `(0, 0)`).
//! - **Everything is in merge-block order** (tile row, tile col, row in tile,
//!   col in tile), the order `crates/vision-io`'s `inkling2_preprocess` emits.
//!   The merger is then a plain reshape of rows `1..` into `[tiles, 4096]`,
//!   read straight out of the residual buffer.
//! - **Attention is the whole image** on every block: one segment.
//! - **The position table is resampled with border clamping**
//!   (`F.interpolate(bilinear, align_corners=False)`), NOT Muse's
//!   zero-weight off-table taps.
//! - **No `ln_post`, no output norm**: `proj_in`, erf GELU, `proj_out`.

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex, VisionConfig, VisionTowerKind};
use turbospark_vision_io::{GridThw, Inkling2Patches, INKLING2_MERGE};

use super::muse::{encode_block, read_bits, read_fp16_host, MuseScratch};
use super::shape::VisionShape;
use super::weights::{fp16_view, resolve_block_roles, BlockRoles, VISION_PREFIX};
use super::{VisionEmbedding, VisionStages, VISION_SLOTS};
use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;

/// Rope theta (checked at intake).
const ROPE_THETA: f32 = 10_000.0;
/// LayerNorm epsilon of `ln_pre` (the blocks' is Muse's, the same 1e-5).
const LN_EPS: f32 = 1e-5;
/// MLP row tile, as the other towers.
pub(crate) const INKLING2_MLP_TILE_ROWS: usize = 1024;

/// `(row, col)` of every patch of an `R x C` tile image, merge-block order.
pub(crate) fn merge_block_coords(tile_rows: usize, tile_cols: usize) -> Vec<(usize, usize)> {
    let m = INKLING2_MERGE;
    let mut out = Vec::with_capacity(tile_rows * tile_cols * m * m);
    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            for pr in 0..m {
                for pc in 0..m {
                    out.push((tr * m + pr, tc * m + pc));
                }
            }
        }
    }
    out
}

/// One axis's two taps and weights for `n` target positions over a `side`
/// table: `src = (i + 0.5) * side / n - 0.5`, taps `floor(src)` and
/// `floor(src) + 1` clamped to the table, weights `1 - |src - raw tap|`
/// (transformers `_interpolation_axis_taps_weights`, border padding). A tap
/// clamped onto the edge keeps its weight, which is `F.interpolate`'s edge
/// replication.
fn axis_taps(n: usize, side: usize) -> Vec<[(usize, f32); 2]> {
    (0..n)
        .map(|i| {
            let src = (i as f32 + 0.5) * side as f32 / n as f32 - 0.5;
            let floor = src.floor();
            let tap = |off: f32| {
                let raw = floor + off;
                let w = (1.0 - (src - raw).abs()).max(0.0);
                let idx = raw.clamp(0.0, (side - 1) as f32) as usize;
                (idx, w)
            };
            [tap(0.0), tap(1.0)]
        })
        .collect()
}

/// The interpolated position rows for every patch, merge-block order,
/// `[4 * R * C, hidden]` F32. `table` is `[side * side, hidden]`.
pub(crate) fn position_rows(
    table: &[f32],
    side: usize,
    hidden: usize,
    tile_rows: usize,
    tile_cols: usize,
) -> Vec<f32> {
    let (gh, gw) = (tile_rows * INKLING2_MERGE, tile_cols * INKLING2_MERGE);
    let (ax_h, ax_w) = (axis_taps(gh, side), axis_taps(gw, side));
    let coords = merge_block_coords(tile_rows, tile_cols);
    let mut out = vec![0.0f32; coords.len() * hidden];
    for (p, &(y, x)) in coords.iter().enumerate() {
        let dst = &mut out[p * hidden..][..hidden];
        // The reference sums the four taps (h0, w0), (h0, w1), (h1, w0),
        // (h1, w1).
        for &(hy, wy) in &ax_h[y] {
            for &(wx, ww) in &ax_w[x] {
                let w = wy * ww;
                let src = &table[(hy * side + wx) * hidden..][..hidden];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += s * w;
                }
            }
        }
    }
    out
}

/// The rope angle rows, `[1 + 4 * R * C, head_dim / 2]` F32: row 0 (the
/// class row) all zero, then per patch in merge-block order the width angles
/// then the height angles, each `head_dim / 4` long, at the 1-INDEXED
/// position `(col + 1, row + 1)`. With q and k permuted at repack, the
/// half-split kernel reading this row computes the reference's interleaved
/// rotation `cat(w * inv, h * inv).repeat_interleave(2)`.
pub(crate) fn rope_rows(tile_rows: usize, tile_cols: usize, head_dim: usize) -> Vec<f32> {
    let quarter = head_dim / 4;
    let spatial = head_dim / 2;
    let inv: Vec<f32> = (0..quarter)
        .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / spatial as f32))
        .collect();
    let coords = merge_block_coords(tile_rows, tile_cols);
    let mut out = vec![0.0f32; spatial];
    out.reserve(coords.len() * spatial);
    for (y, x) in coords {
        let (w, h) = ((x + 1) as f32, (y + 1) as f32);
        out.extend(inv.iter().map(|f| w * f));
        out.extend(inv.iter().map(|f| h * f));
    }
    out
}

/// Resident tensors outside the blocks, as local byte offsets.
struct Inkling2Resident {
    patch_embed: u64,
    pos_table: u64,
    cls: u64,
    ln_pre_w: u64,
    ln_pre_b: u64,
    fc1: u64,
    fc2: u64,
    projector_hidden: usize,
}

impl Inkling2Resident {
    fn resolve(index: &ResidentIndex, shape: &VisionShape) -> Result<Self, RealForwardError> {
        let name = |tail: &str| format!("{VISION_PREFIX}{tail}");
        let h = shape.hidden;
        // proj_in is square: [4096, 4 * 1024].
        let ph = shape.merger_input();
        Ok(Self {
            patch_embed: fp16_view(index, &name("patch_embed.proj.weight"), h * shape.patch_dim)?,
            pos_table: fp16_view(index, &name("pos_embed.weight"), shape.pos_rows * h)?,
            cls: fp16_view(index, &name("cls_embed.weight"), h)?,
            ln_pre_w: fp16_view(index, &name("ln_pre.weight"), h)?,
            ln_pre_b: fp16_view(index, &name("ln_pre.bias"), h)?,
            fc1: fp16_view(index, &name("merger.fc1.weight"), ph * shape.merger_input())?,
            fc2: fp16_view(index, &name("merger.fc2.weight"), shape.out_hidden * ph)?,
            projector_hidden: ph,
        })
    }
}

fn require_inkling2(v: &VisionConfig) -> Result<(), RealForwardError> {
    if v.tower != VisionTowerKind::Inkling2 || !v.is_active() {
        return Err(RealForwardError::Unsupported(format!(
            "not an active inkling2 vision tower (kind {})",
            v.tower.as_str()
        )));
    }
    Ok(())
}

/// The Inkling 2 tower opened against a sidecar (or an install carrying it).
pub(crate) struct Inkling2VisionTower {
    /// Declared before `streamer`: the buffers alias its allocations.
    slot_buffers: Vec<gpu::MetalBuffer>,
    streamer: streaming::PreadExpertStreamer,
    blocks: Vec<BlockRoles>,
    shape: VisionShape,
    resident: Inkling2Resident,
    /// `pos_embed.weight` decoded once, `[side * side, hidden]` F32.
    pos_table: Vec<f32>,
    /// The class row, `[hidden]` F32.
    cls: Vec<f32>,
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
    pub(crate) last_scratch_bytes: u64,
}

impl Inkling2VisionTower {
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        require_inkling2(&arch.vision)?;
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
        .map_err(|e| RealForwardError::Unsupported(format!("inkling2 vision streamer: {e}")))?;
        let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
        for slot in 0..VISION_SLOTS {
            let (ptr, len) = streamer.slot_allocation(slot);
            slot_buffers.push(
                gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                    .map_err(RealForwardError::Gpu)?,
            );
        }
        let resident = Inkling2Resident::resolve(index, &shape)?;
        let pos_table = read_fp16_host(weights, resident.pos_table, shape.pos_rows * shape.hidden)?;
        let cls = read_fp16_host(weights, resident.cls, shape.hidden)?;
        Ok(Self {
            slot_buffers,
            streamer,
            blocks,
            shape,
            resident,
            pos_table,
            cls,
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

    /// Encode one image to `[tiles, out_hidden]` FP16 rows, tiles in raster
    /// order: the rows that replace the image's placeholder run.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &Inkling2Patches,
        mut capture: Option<VisionStages>,
    ) -> Result<(VisionEmbedding, Option<VisionStages>), RealForwardError> {
        let shape = self.shape.clone();
        let tiles = image.tokens();
        let patches = image.patches();
        if tiles == 0 || image.pixels.len() != patches * shape.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "inkling2 vision input: {} values for {patches} patches of {}",
                image.pixels.len(),
                shape.patch_dim
            )));
        }
        if shape.merge != INKLING2_MERGE {
            return Err(RealForwardError::Unsupported(format!(
                "inkling2 vision merge {} is not {INKLING2_MERGE}",
                shape.merge
            )));
        }
        let seq = 1 + patches;
        let h = shape.hidden;

        // Host prep. Row 0 is the class row: a zero patch row and the class
        // embedding as its "position", so the patch GEMM and the one add
        // place every row.
        let mut rows16 = vec![half::f16::ZERO; shape.patch_dim];
        rows16.extend(image.pixels.iter().map(|&v| half::f16::from_f32(v)));
        let mut pos = self.cls.clone();
        pos.extend(position_rows(
            &self.pos_table,
            shape.pos_side,
            h,
            image.rows,
            image.cols,
        ));
        let pos16: Vec<half::f16> = pos.iter().map(|&v| half::f16::from_f32(v)).collect();
        let freqs = rope_rows(image.rows, image.cols, shape.head_dim);

        let w = self.sidecar_weights.as_ref().unwrap_or(weights);
        let (streamer, slots, blocks) = (&mut self.streamer, &self.slot_buffers, &self.blocks);
        let resident = &self.resident;
        let last_scratch = &mut self.last_scratch_bytes;
        let ph = resident.projector_hidden;
        let mw = shape.merger_input();
        let grid = GridThw::new(1, image.rows * INKLING2_MERGE, image.cols * INKLING2_MERGE);

        gpu::autorelease_pool(|| {
            let e = RealForwardError::Gpu;
            let wide = (seq * h * 2) as u64;
            let tile = INKLING2_MLP_TILE_ROWS.min(seq).max(1);
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
                // One segment: every block attends over the whole image.
                segments: vec![seq],
            };
            *last_scratch = wide * 8 + (tile * shape.intermediate * 2) as u64;
            let n_wide = seq * h;

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
                h as u32,
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
                h as u32,
                LN_EPS,
            )
            .map_err(e)?;
            pass.commit_and_wait();
            if let Some(c) = capture.as_mut() {
                c.patch_embed = read_bits(&s.x, n_wide);
            }

            for (n, block_roles) in blocks.iter().enumerate() {
                let slot = n % VISION_SLOTS;
                streamer.load_expert_into_slot(0, n, slot).map_err(|err| {
                    RealForwardError::Unsupported(format!("inkling2 vision block {n} read: {err}"))
                })?;
                let pass = context.begin_pass();
                encode_block(context, &pass, &slots[slot], block_roles, &s, &shape, n)?;
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

            // The merger reads rows 1.. of the residual stream as
            // [tiles, 4 * hidden]: merge-block order makes the 2x2 concat a
            // plain reshape, and skipping row 0 drops the class row.
            let h1 = context.new_output_buffer((tiles * ph * 2) as u64);
            let out = context.new_output_buffer((tiles * shape.out_hidden * 2) as u64);
            let pass = context.begin_pass();
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&s.x, (h * 2) as u64),
                (w.buffer(), w.gpu_offset(resident.fc1)),
                None,
                (&h1, 0),
                tiles as u32,
                mw as u32,
                ph as u32,
            )
            .map_err(e)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&h1, 0),
                (tiles * ph) as u32,
                gpu::GeluKind::Erf,
            )
            .map_err(e)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&h1, 0),
                (w.buffer(), w.gpu_offset(resident.fc2)),
                None,
                (&out, 0),
                tiles as u32,
                ph as u32,
                shape.out_hidden as u32,
            )
            .map_err(e)?;
            pass.commit_and_wait();

            let rows = read_bits(&out, tiles * shape.out_hidden);
            if let Some(i) = rows
                .iter()
                .position(|&b| !half::f16::from_bits(b).is_finite())
            {
                return Err(RealForwardError::Unsupported(format!(
                    "inkling2 vision output is non-finite at element {i}; the FP16 tower \
                     overflowed"
                )));
            }
            Ok((
                VisionEmbedding {
                    rows,
                    merged_tokens: tiles,
                    out_hidden: shape.out_hidden,
                    grid,
                    deepstack: Vec::new(),
                },
                capture,
            ))
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn coordinates_walk_tiles_then_the_2x2_inside() {
        let c = merge_block_coords(1, 2);
        assert_eq!(
            c,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3)
            ]
        );
    }

    #[test]
    fn position_rows_clamp_to_the_border_not_zero() {
        // A 2 x 2 table whose rows are 0..3 in every channel; a 4 x 4 patch
        // grid (one 2x2 tile grid) samples at -0.25, 0.25, 0.75, 1.25.
        let table: Vec<f32> = (0..4).flat_map(|r| [r as f32, r as f32]).collect();
        let rows = position_rows(&table, 2, 2, 2, 2);
        let coords = merge_block_coords(2, 2);
        let at = |y: usize, x: usize| {
            let p = coords.iter().position(|&c| c == (y, x)).unwrap();
            rows[p * 2]
        };
        // (1.25, 1.25): taps 1 and 2 -> clamped to 1, weights 0.75 and 0.25
        // both land on table row 3: the edge value 3, where Muse's
        // zero-weight rule gives 1.6875.
        assert!((at(3, 3) - 3.0).abs() < 1e-6, "{}", at(3, 3));
        // (-0.25, -0.25): both taps clamp to 0.
        assert!(at(0, 0).abs() < 1e-6);
        // Interior (0.25, 0.25): 0*.5625 + 1*.1875 + 2*.1875 + 3*.0625.
        assert!((at(1, 1) - 0.75).abs() < 1e-6, "{}", at(1, 1));
    }

    #[test]
    fn rope_rows_lead_with_a_zero_class_row_then_width_first() {
        let rows = rope_rows(1, 1, 64);
        assert_eq!(rows.len(), 5 * 32);
        assert!(rows[..32].iter().all(|&v| v == 0.0));
        // Patch 1 of tile (0, 0) is (row 0, col 1): w = 2, h = 1; pair 0's
        // inverse frequency is 1.
        let p = 2 * 32;
        assert!((rows[p] - 2.0).abs() < 1e-6);
        assert!((rows[p + 16] - 1.0).abs() < 1e-6);
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

    /// The tower on the REAL sidecar against
    /// `scripts/inkling2_vision_reference.py`, a numpy transcription of the
    /// transformers reference (not the reference executed). The image is raw
    /// RGB8 bytes, preprocessed by BOTH sides independently, so this holds
    /// `crates/vision-io`, the repack transforms and the tower together.
    ///
    /// ```sh
    /// TURBOSPARK_INKLING2_SIDECAR_DIR=~/models/inkling2-vision/inkling2-mini.gturbo-vision \
    /// TURBOSPARK_INKLING2_VISION_SAFETENSORS=~/models/inkling2-vision/vision.safetensors \
    /// TURBOSPARK_INKLING2_IMAGE=~/models/inkling2-vision/ref/page.rgb \
    /// TURBOSPARK_INKLING2_IMAGE_HW="83 121" \
    /// TURBOSPARK_INKLING2_SCRATCH=~/models/inkling2-vision/ref/run \
    ///   cargo test -p turbospark-runtime --release --lib \
    ///   inkling2_real_tower_matches_the_numpy_reference -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs the Inkling 2 vision sidecar, the fetched BF16 tower, an RGB8 image and python3 with numpy"]
    fn inkling2_real_tower_matches_the_numpy_reference() {
        let sidecar = env_path("TURBOSPARK_INKLING2_SIDECAR_DIR");
        let weights_file = env_path("TURBOSPARK_INKLING2_VISION_SAFETENSORS");
        let image = env_path("TURBOSPARK_INKLING2_IMAGE");
        let scratch = env_path("TURBOSPARK_INKLING2_SCRATCH");
        let hw = std::env::var("TURBOSPARK_INKLING2_IMAGE_HW").expect("H W");
        let mut it = hw.split_whitespace().map(|v| v.parse::<usize>().unwrap());
        let (ih, iw) = (it.next().unwrap(), it.next().unwrap());
        std::fs::create_dir_all(&scratch).unwrap();

        let (_record, vision) = model_io::load_vision_sidecar(&sidecar).expect("sidecar");
        let arch = model_io::sidecar_arch(
            model_io::ModelFamily::Inkling2Loop,
            vision.out_hidden_size,
            &vision,
        );
        let raw = std::fs::read(&image).unwrap();
        let rgb = turbospark_vision_io::Rgb8Image::new(iw, ih, raw).expect("rgb8");
        let pre = turbospark_vision_io::inkling2_preprocess(
            &rgb,
            turbospark_vision_io::INKLING2_MAX_TOKENS,
        )
        .expect("preprocess");
        println!("tiles {}x{} ({} tokens)", pre.rows, pre.cols, pre.tokens());

        let reference = scratch.join("reference");
        let status = std::process::Command::new("python3")
            .arg("-I")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../scripts/inkling2_vision_reference.py"
            ))
            .arg("--weights")
            .arg(&weights_file)
            .arg("--image")
            .arg(&image)
            .args(["--size", &ih.to_string(), &iw.to_string()])
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
            Inkling2VisionTower::open_with_sidecar(&sidecar, &context, &arch).expect("open");
        let (embedding, stages) = tower
            .run(&mut context, &weights, &pre, Some(VisionStages::default()))
            .expect("tower run");
        let stages = stages.expect("captured");
        let to_f32 = |bits: &[u16]| -> Vec<f32> {
            bits.iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect()
        };
        let pairs = [
            ("embed", to_f32(&stages.patch_embed)),
            ("block_0", to_f32(&stages.block_first)),
            ("block_22", to_f32(&stages.block_last)),
            ("merger", to_f32(&embedding.rows)),
        ];
        let mut failures = Vec::new();
        for (name, got) in &pairs {
            let want = read_f32(&reference.join(format!("{name}.bin")));
            let c = cosine(got, &want);
            let max_abs = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("{name:10} cosine {c:.8}  max|diff| {max_abs:.4e}");
            let bar = if *name == "merger" { 0.9999 } else { 0.999 };
            if c <= bar {
                failures.push(format!("{name} {c:.6} (bar {bar})"));
            }
        }
        // Dump the rows for the llama.cpp cross-check.
        std::fs::write(
            scratch.join("ours_merger.bin"),
            to_f32(&embedding.rows)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        assert!(
            failures.is_empty(),
            "the tower disagrees with the reference: {failures:?}"
        );
    }
}
