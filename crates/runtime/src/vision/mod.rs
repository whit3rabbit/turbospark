//! The `qwen3_5` vision tower's streamed forward pass (ROADMAP M-V4).
//!
//! Takes a preprocessed image and returns the `[merged_tokens,
//! out_hidden_size]` FP16 rows the trunk's residual stream wants, which is
//! what M-V5 injects at the image-pad positions. Nothing consumes them yet.
//!
//! # The streaming contract is M-V3's, reused rather than rebuilt
//!
//! `packed_vision/` reuses the `PackedExpertsLayout` schema VERBATIM: the
//! tower is one "layer" of `depth` "experts", and `StreamLayout` interprets
//! neither word -- an expert is a fixed-stride blob in a file and a layer is
//! which file. So `PreadExpertStreamer`, `ExpertCache` and the read pool
//! serve a block loop with no new I/O code, which was the design bet M-V3
//! was written to pay off. `load_packed_layout_from` and
//! `StreamLayout::from_packed_layer_in` take the subdirectory as a parameter
//! for exactly this.
//!
//! # Two slots, and an honest note about what they buy TODAY
//!
//! The streamer opens at [`VISION_SLOTS`] and block `n` is read into slot
//! `n % 2`. With a `commit_and_wait` between blocks -- which is what v1 does
//! -- ONE slot would be correct: the host cannot overwrite a slot the GPU is
//! still reading, because the GPU has finished. The second slot is the shape
//! the later read-pool prefetch needs, and alternating now makes that a
//! one-line change rather than a restructure. It is not load-bearing for
//! correctness at this stage and this comment exists so nobody concludes it
//! is.
//!
//! # Everything here is FP16 and nothing else may read these tensors
//!
//! `readable_resident_dtype` honours dtype tag 2 under the `vision.` prefix
//! and refuses it everywhere else, because the text path's unquantized
//! readers are dtype-BLIND (AGENTS.md Gotcha 45). This module is the whole of
//! what makes that scoping true: it is the only reader of `vision.` tensors,
//! and `weights::fp16_view` checks the TAG as well as the width so it cannot
//! quietly become a general one.

mod block;
mod budget;
mod inject;
mod overflow;
mod scratch;
mod shape;
mod stages;
mod weights;

pub use budget::{PixelBudget, VisionBudgetTooSmall};
pub use inject::{PromptVision, RopePosition};

use std::path::Path;

use model_io::{ArchConfig, ResidentIndex};
use turbospark_vision_io::{PreprocessParams, PreprocessedImage};

use crate::real_forward::PACKED_LAYOUT_MAX_BYTES;
use crate::real_forward_types::RealForwardError;
use scratch::VisionScratch;
use shape::VisionShape;
use weights::{BlockRoles, VisionResident};

/// Slots the tower's streamer opens with.
///
/// TWO, for double buffering. Deliberately not `ALLOWED_CACHE_SLOTS`'
/// minimum: that set is the CLI flag's, for the routed-expert cache whose
/// working set is a routing decision. A tower reads its blocks in a fixed
/// order exactly once per image, so there is no cache to size -- the number
/// is a pipelining depth, and `PreadExpertStreamer::open` takes any positive
/// count.
pub const VISION_SLOTS: usize = 2;

/// Reads the tower's OWN residency seam, deliberately never
/// `TURBOSPARK_EXPERT_RESIDENCY` (the routed-expert one). Reusing that variable
/// would move the tower silently for anyone A/Bing routed residency, which is
/// exactly the silent-ignore failure `real_forward_init::mapped_residency_
/// refusal`'s own doc comment exists to prevent for the routed case.
///
/// Unlike the routed case, there is no per-family refusal function: the tower
/// is family-agnostic (any install with `arch.vision.is_active()` runs the
/// same code), so the only gate is whether the tower opens at all.
fn vision_mapped_residency_requested() -> bool {
    std::env::var("TURBOSPARK_VISION_RESIDENCY")
        .map(|v| v.eq_ignore_ascii_case("mapped"))
        .unwrap_or(false)
}

/// One image's contribution to the trunk's residual stream.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionEmbedding {
    /// `merged_tokens * out_hidden` FP16 values, as raw bits.
    ///
    /// Host rows rather than a retained GPU buffer, because the injection
    /// M-V5 performs writes per-position rows into `DecodeScratch::x`, which
    /// is shared storage the host writes; a GPU-side copy would need a kernel
    /// that does not exist. 13 MB on a 1024x1280 page.
    pub rows: Vec<u16>,
    pub merged_tokens: usize,
    pub out_hidden: usize,
    /// The patch grid the rows came from, carried so a caller can check a
    /// span length against it rather than recomputing the merge arithmetic.
    pub grid: turbospark_vision_io::GridThw,
    /// The deepstack mergers' outputs, one per `VisionShape::deepstack`
    /// entry, in that order: entry `k` is injected after TRUNK layer `k` at
    /// this image's positions. Each carries `merged_tokens * out_hidden`
    /// values -- the same merge geometry and output width as `rows` -- and
    /// EMPTY for a tower without deepstack (every `qwen3_5` tower).
    ///
    /// Host rows for the same reason `rows` is: the trunk-side add writes
    /// per-position rows from host-visible storage.
    pub deepstack: Vec<DeepstackRows>,
}

/// One deepstack merger's output rows, `merged_tokens * out_hidden` FP16
/// bits. A named struct rather than a bare `Vec<u16>` so a caller cannot
/// confuse an injection's rows with the main embedding's.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepstackRows {
    pub rows: Vec<u16>,
}

/// The residual stream at three points inside the tower, for the
/// cross-engine parity gate.
///
/// Each is `[patches, hidden]` FP16 bits. Not a debugging aid left on by
/// default and not an env var: it is reached through
/// [`VisionTower::run_with_stages`] alone, so the ordinary path cannot
/// acquire it.
///
/// **The naming is by POSITION, not by number.** `block_last` rather than
/// `block_26`, because the depth is the checkpoint's -- a comparison against
/// a reference that named a fixed index would silently compare the wrong
/// block on a tower of another depth.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisionStages {
    /// After the patch embedding AND the position add, which is where the
    /// reference's own trace takes it: `h = patch_embed(x); h = h + pos`.
    pub patch_embed: Vec<u16>,
    /// After block 0.
    pub block_first: Vec<u16>,
    /// After the last block, before the merger.
    pub block_last: Vec<u16>,
    /// Each deepstack merger's output, in the config's index order; EMPTY
    /// for a tower without deepstack. Same merge geometry as the merger
    /// stage's own rows, `[merged, out_hidden]` FP16 bits.
    pub deepstack: Vec<Vec<u16>>,
}

/// The tower: its streamer, its weights, and the position table.
pub struct VisionTower {
    /// One zero-copy `MTLBuffer` per slot, wrapped once over the slot's
    /// page-aligned allocation. **Declared BEFORE `streamer`** so the buffers
    /// drop before the allocations they alias -- the same ordering
    /// `RealForwardRunner` states for its routed slot buffers, and for the
    /// same reason.
    slot_buffers: Vec<gpu::MetalBuffer>,
    /// `None` under mapped residency: the mapped arm opens no streamer at
    /// all, and the block loop branches on `mapped_buffer.is_some()` rather
    /// than on a mode flag it could disagree with.
    streamer: Option<streaming::PreadExpertStreamer>,
    /// Zero-copy `MTLBuffer` over the WHOLE tower's mapped region (all
    /// `depth` blocks concatenated), `Some` only under
    /// `TURBOSPARK_VISION_RESIDENCY=mapped`. Declared BEFORE `mapped_layer` for
    /// the same reason `slot_buffers` precedes `streamer`: the buffer aliases
    /// the mapping with no deallocator, so the mapping must outlive it and
    /// Rust drops fields in declaration order.
    mapped_buffer: Option<gpu::MetalBuffer>,
    mapped_layer: Option<streaming::MappedExpertLayer>,
    /// Per-block sub-tensor offsets, indexed by block.
    blocks: Vec<BlockRoles>,
    resident: VisionResident,
    /// `pos_embed.weight` decoded to host f32 once at open (5.3 MB on the
    /// real tower). See `stages::pos_embed_rows` for why the blend is a host
    /// operation.
    pos_table: Vec<f32>,
    shape: VisionShape,
    /// Bytes the slot cache pins, `VISION_SLOTS * block_stride`. Reported so
    /// a memory assertion compares against what was allocated rather than
    /// against a recomputation of it.
    pub(crate) slot_bytes: u64,
    /// What the last page's scratch actually cost, recorded by [`Self::run`].
    ///
    /// Kept beside [`Self::scratch_bytes`], which is the same quantity
    /// PREDICTED from the page size. Having both is the point: a gate that
    /// compared the formula against itself would assert nothing, and the
    /// formula is what a caller sizing a page budget would use.
    pub(crate) last_scratch_bytes: u64,
    /// `Some` only when `TURBOSPARK_VISION_OVERFLOW` named an output path at
    /// open. `None` means every call site below dispatches no readback at
    /// all, which is what keeps the ordinary path byte-identical to the
    /// engine that shipped before this existed (`overflow.rs`'s module
    /// doc).
    overflow: Option<overflow::VisionOverflowCapture>,
    /// The sidecar's OWN resident weights (vision memory sidecar, Part A2),
    /// `Some` only when this tower was opened via [`Self::open_with_sidecar`]
    /// against a standalone sidecar directory rather than baked into the
    /// trunk's own install.
    ///
    /// Independent of `slot_buffers`/`streamer` above: `ResidentGpuWeights`
    /// owns its `ResidentBuffer` mapping internally (see `crates/gpu`'s
    /// `resident_metal.rs` module doc), so there is no separate mapping this
    /// field must be declared ahead of to keep alive -- it is a
    /// self-contained unit, kept here only because nothing else in a
    /// standalone-sidecar session holds a reference to it once `open_inner`
    /// returns (the trunk case borrows the RUNNER's `weights` field at every
    /// call site instead).
    sidecar_weights: Option<gpu::ResidentGpuWeights>,
    /// The MLP's row tile (Part B1), `scratch::VISION_MLP_TILE_ROWS` at open
    /// and overridable afterward by the `#[doc(hidden)]` test knob
    /// `RealForwardRunner::set_vision_mlp_tile_rows` -- a per-RUNNER-session
    /// value rather than a global constant read at the call site, so a test
    /// can force several loop iterations on a page far smaller than any real
    /// one would need tiling for.
    pub(crate) mlp_tile_rows: usize,
}

impl VisionTower {
    /// Open the tower against an install that declares one, using the
    /// TRUNK's own resident weights and index (a combined install).
    ///
    /// Called lazily, on the first image: `arch.vision` is read by nothing
    /// else in this crate, and opening at `RealForwardRunner::open` would
    /// charge every text-only session on a vision install ~58 MiB of pinned
    /// slots plus the layout parse for a component it never touches.
    pub(crate) fn open(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        Self::build(dir, context, weights, index, arch)
    }

    /// Open the tower against a standalone SIDECAR directory (vision memory
    /// sidecar, Part A2) rather than the trunk's own install.
    ///
    /// Opens the sidecar directory's OWN `model_weights.bin` -- its own
    /// resident index and its own zero-copy `ResidentGpuWeights` mapping,
    /// entirely separate from the trunk runner's `weights`/`index` fields --
    /// and keeps that mapping alive in [`Self::sidecar_weights`] for the
    /// tower's whole life, since nothing else holds a reference to it.
    /// `packed_vision/` and the nine resident tensors are then read out of
    /// `sidecar_dir` exactly as [`Self::open`] reads them out of a combined
    /// install's own directory -- same [`Self::build`] body, different
    /// directory and different weights/index pair.
    ///
    /// The dtype backstop (`readable_resident_dtype`'s NAME-SCOPED FP16
    /// exception, AGENTS.md Gotcha 24) is re-run here on the sidecar's own
    /// resident index for the same reason `RealForwardRunner::open_inner`
    /// runs it on a trunk's: every entry under a sidecar built by
    /// `crates/repack`'s `write_vision_sidecar` is `vision.`-prefixed by
    /// construction, so this should never fire on a well-formed sidecar, but
    /// a hand-edited or corrupted one is exactly the case a backstop exists
    /// for.
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
                 crate honours; a well-formed sidecar carries only `vision.`-prefixed FP16 \
                 tensors",
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

    /// The shared body of [`Self::open`] and [`Self::open_with_sidecar`]:
    /// everything that does not depend on WHERE the resident weights came
    /// from. `sidecar_weights` starts `None` here; `open_with_sidecar` fills
    /// it in after this returns, once its own local `weights` (borrowed for
    /// this call) is free to be moved into the built tower.
    fn build(
        dir: &Path,
        context: &gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
    ) -> Result<Self, RealForwardError> {
        let shape = VisionShape::resolve(&arch.vision)?;

        let layout = model_io::load_packed_layout_from(
            dir,
            model_io::PACKED_VISION_DIR,
            PACKED_LAYOUT_MAX_BYTES,
        )
        .map_err(RealForwardError::Model)?;
        let layer = layout.layers.iter().find(|l| l.layer == 0).ok_or_else(|| {
            RealForwardError::Unsupported(
                "packed_vision/layout.json declares no layer 0; the tower is written as one \
                 layer of `depth` blobs"
                    .to_string(),
            )
        })?;
        if layer.experts.len() != shape.depth {
            // Refused rather than clamped to whichever is smaller: too few
            // blobs runs a shallower tower and too many runs the declared one
            // while the install carries something else, and both produce a
            // plausible embedding.
            return Err(RealForwardError::Unsupported(format!(
                "packed_vision carries {} blocks but the manifest declares a depth of {}",
                layer.experts.len(),
                shape.depth
            )));
        }

        // Indexed by the entry's own `expert` field rather than by position,
        // because the writer is free to pack out of order and `StreamLayout`
        // carries explicit offsets for that reason. A positional read would
        // agree on every install written so far and silently permute the
        // blocks of one that did not.
        let mut blocks: Vec<Option<BlockRoles>> = vec![None; shape.depth];
        for entry in &layer.experts {
            let slot = blocks.get_mut(entry.expert).ok_or_else(|| {
                RealForwardError::Unsupported(format!(
                    "packed_vision names a block {} past the declared depth of {}",
                    entry.expert, shape.depth
                ))
            })?;
            if slot
                .replace(weights::resolve_block_roles(
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
        let blocks: Vec<BlockRoles> = blocks
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                b.ok_or_else(|| {
                    RealForwardError::MissingTensor(format!("packed_vision has no block {i}"))
                })
            })
            .collect::<Result<_, _>>()?;

        let stream_layout =
            streaming::StreamLayout::from_packed_layer_in(layer, dir, model_io::PACKED_VISION_DIR);

        let (streamer, slot_buffers, mapped_buffer, mapped_layer, slot_bytes) =
            if vision_mapped_residency_requested() {
                // No per-family refusal: the tower is family-agnostic, and
                // `layer.experts.len() == shape.depth` is already checked
                // above, so `stream_layout` always describes exactly one
                // "layer" of `depth` blocks whenever the tower opens at all.
                let mapped_layer =
                    streaming::MappedExpertLayer::open(stream_layout).map_err(|e| {
                        RealForwardError::Unsupported(format!("mapped vision residency: {e}"))
                    })?;
                let bytes = mapped_layer.page_aligned_bytes();
                let buffer =
                    gpu::wrap_page_aligned_no_copy(context.device(), bytes.as_ptr(), bytes.len())
                        .map_err(RealForwardError::Gpu)?;
                (None, Vec::new(), Some(buffer), Some(mapped_layer), 0u64)
            } else {
                let streamer = streaming::PreadExpertStreamer::open(
                    stream_layout,
                    VISION_SLOTS,
                    streaming::ExpertCachePolicy::DEFAULT,
                )
                .map_err(|e| {
                    RealForwardError::Unsupported(format!(
                        "vision block streamer: {e} ({VISION_SLOTS} slots x {:.1} MiB per block)",
                        layer.expert_stride as f64 / (1024.0 * 1024.0),
                    ))
                })?;

                let mut slot_buffers = Vec::with_capacity(VISION_SLOTS);
                for slot in 0..VISION_SLOTS {
                    let (ptr, len) = streamer.slot_allocation(slot);
                    slot_buffers.push(
                        gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                            .map_err(RealForwardError::Gpu)?,
                    );
                }
                (
                    Some(streamer),
                    slot_buffers,
                    None,
                    None,
                    VISION_SLOTS as u64 * layer.expert_stride,
                )
            };

        let resident = VisionResident::resolve(index, &shape)?;
        let pos_table = read_fp16_host(
            weights,
            resident.pos_embed_host,
            shape.pos_rows * shape.hidden,
        )?;

        Ok(Self {
            slot_buffers,
            streamer,
            mapped_buffer,
            mapped_layer,
            blocks,
            resident,
            pos_table,
            shape,
            slot_bytes,
            last_scratch_bytes: 0,
            overflow: overflow::VisionOverflowCapture::from_env(),
            sidecar_weights: None,
            mlp_tile_rows: scratch::VISION_MLP_TILE_ROWS,
        })
    }

    /// Whether this tower was opened against a standalone sidecar directory
    /// ([`Self::open_with_sidecar`]) rather than baked into the trunk's own
    /// install ([`Self::open`]).
    ///
    /// Test-only proof of engagement, the same shape as
    /// [`Self::is_mapped_residency`]'s own doc: a byte-identity check
    /// between a sidecar-backed run and a combined-install run cannot on its
    /// own distinguish "the sidecar path ran and produced the same output"
    /// from "the sidecar path silently fell through to the trunk's own
    /// (nonexistent, on a text-only trunk) tower" -- both would trivially
    /// agree in the second case if the trunk happened to have one too.
    pub(crate) fn is_sidecar(&self) -> bool {
        self.sidecar_weights.is_some()
    }

    /// Whether this tower opened under `TURBOSPARK_VISION_RESIDENCY=mapped`.
    ///
    /// Test-only proof of engagement: a byte-identity check between the two
    /// arms cannot on its own distinguish "the mapped arm ran and produced
    /// the same output" from "the mapped arm silently fell through to
    /// pread", since both would pass parity trivially in the second case.
    #[allow(dead_code)]
    pub(crate) fn is_mapped_residency(&self) -> bool {
        self.mapped_buffer.is_some()
    }

    /// Run one image through the whole tower.
    ///
    /// # Command buffers
    ///
    /// One `commit_and_wait` per stage: the patch embedding and position add,
    /// then one per block, then the merger. `depth + 2` buffers per image,
    /// and the wait between blocks is what makes a synchronous `pread` into
    /// the next slot safe with no fence.
    ///
    /// The whole call runs inside an autorelease pool. Command buffers and
    /// encoders are AUTORELEASED objects and a plain Rust binary has exactly
    /// one pool, around `main`, so without an inner one every buffer a
    /// bulk-OCR loop ever created stays alive to exit (AGENTS.md Gotcha 17).
    /// At 29 buffers per page that is the same failure the decode loop has.
    pub(crate) fn run(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &PreprocessedImage,
        params: &PreprocessParams,
    ) -> Result<VisionEmbedding, RealForwardError> {
        self.run_inner(context, weights, image, params, None)
            .map(|(e, _)| e)
    }

    /// [`Self::run`] plus the residual stream at three intermediate points.
    ///
    /// The cross-engine parity gate's entry point and nothing else's. A gap
    /// that shows as 0.99 at the merger localizes immediately when the patch
    /// embedding is exact and block 26 is not; merger-only cannot do that,
    /// and bisecting it by rebuilding the tower per stage would cost a run
    /// per stage.
    ///
    /// It costs three extra readbacks of `[seq, hidden]` and NOTHING on the
    /// ordinary path, which passes `None` -- there is no flag to leave on by
    /// accident. The reads need no extra synchronization: the driver already
    /// waits on every stage's command buffer.
    pub(crate) fn run_with_stages(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &PreprocessedImage,
        params: &PreprocessParams,
    ) -> Result<(VisionEmbedding, VisionStages), RealForwardError> {
        let (embedding, stages) = self.run_inner(
            context,
            weights,
            image,
            params,
            Some(VisionStages::default()),
        )?;
        Ok((embedding, stages.expect("requested just above")))
    }

    fn run_inner(
        &mut self,
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        image: &PreprocessedImage,
        params: &PreprocessParams,
        mut capture: Option<VisionStages>,
    ) -> Result<(VisionEmbedding, Option<VisionStages>), RealForwardError> {
        // `weights` (the parameter) is the RUNNER's own trunk buffer,
        // `&self.weights`, passed in unconditionally by every call site in
        // `real_forward_api.rs` whether or not a sidecar is attached.
        // `self.sidecar_weights.as_ref().unwrap_or(weights)` picks the right
        // one, inline, at each of the two places below that actually bind a
        // resident buffer (patch embed, merger) -- a DIRECT field
        // projection rather than a `&self`-taking helper method, which
        // matters here: a helper would borrow the whole `self` for as long
        // as its result is alive, conflicting with the block loop's
        // `&mut self.streamer`/`self.mapped_buffer` accesses in between,
        // where a direct `self.sidecar_weights` projection is a normal
        // disjoint field borrow the compiler can interleave with those.
        //
        // The caller preprocessed with SOME parameters and the tower was
        // built from the install's. If they disagree the patch rows are the
        // wrong width or the merge windows the wrong size, and both produce
        // a correctly-shaped GEMM over the wrong data.
        if params.patch_dim() != self.shape.patch_dim {
            return Err(RealForwardError::Unsupported(format!(
                "preprocessing produced {}-element patch rows but this tower's patch embedding \
                 takes {}",
                params.patch_dim(),
                self.shape.patch_dim
            )));
        }
        if params.merge_size != self.shape.merge {
            return Err(RealForwardError::Unsupported(format!(
                "preprocessing used a merge size of {} but this tower declares {}",
                params.merge_size, self.shape.merge
            )));
        }

        let seq = image.grid.patches();
        let rows: Vec<half::f16> = image
            .patch_rows
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();
        // The `_default` variant, so `VISION_ROPE_THETA` stays a property of
        // the crate that measured it against the reference rather than a
        // literal restated here. Kept in F32 all the way to the GPU buffer
        // (AGENTS.md/CLAUDE.md B7): the angle at pair 0 equals the raw patch
        // coordinate and reaches the tens on a wide grid, where narrowing to
        // FP16 here would round it by a real, avoidable amount before the
        // shader ever sees it.
        let freqs: Vec<f32> = turbospark_vision_io::rope::vision_rope_freq_rows_default(
            image.grid,
            self.shape.head_dim,
            params,
        )
        .map_err(|e| RealForwardError::Unsupported(format!("vision rope rows: {e}")))?;
        let table =
            turbospark_vision_io::pos_embed_weights(image.grid, self.shape.pos_side, params)
                .map_err(|e| {
                    RealForwardError::Unsupported(format!("vision position table: {e}"))
                })?;
        let pos = stages::pos_embed_rows(&self.pos_table, &table, &self.shape)?;

        gpu::autorelease_pool(|| {
            let mut deepstack_out: Vec<DeepstackRows> =
                Vec::with_capacity(self.shape.deepstack.len());
            let s = VisionScratch::allocate(
                context,
                &self.shape,
                seq,
                self.mlp_tile_rows,
                &rows,
                &freqs,
                &pos,
            )?;
            self.last_scratch_bytes = s.bytes;

            let pass = context.begin_pass();
            stages::encode_patch_embed(
                context,
                &pass,
                self.sidecar_weights.as_ref().unwrap_or(weights),
                &self.resident,
                &s,
                &self.shape,
            )?;
            pass.commit_and_wait();
            let wide = seq * self.shape.hidden;
            if let Some(c) = capture.as_mut() {
                c.patch_embed = read_stage(&s.x, wide);
            }
            if let Some(o) = self.overflow.as_mut() {
                o.note_image_start();
            }

            for n in 0..self.shape.depth {
                let pass = context.begin_pass();
                let mapped_active = self.mapped_buffer.is_some();
                if let Some(mapped_buffer) = self.mapped_buffer.as_ref() {
                    let mapping = self
                        .mapped_layer
                        .as_ref()
                        .expect("mapped_buffer implies mapped_layer");
                    let base = mapping.expert_offset(n).map_err(|e| {
                        RealForwardError::Unsupported(format!(
                            "vision block {n} mapped offset: {e}"
                        ))
                    })?;
                    block::encode_block(
                        context,
                        &pass,
                        mapped_buffer,
                        base,
                        &self.blocks[n],
                        &s,
                        &self.shape,
                        self.mlp_tile_rows,
                    )?;
                } else {
                    let slot = n % VISION_SLOTS;
                    // Layer 0 because the tower IS one layer; `n` is the blob
                    // index inside it.
                    self.streamer
                        .as_mut()
                        .expect("pread arm: streamer opened in VisionTower::open")
                        .load_expert_into_slot(0, n, slot)
                        .map_err(|e| {
                            RealForwardError::Unsupported(format!("vision block {n} read: {e}"))
                        })?;
                    block::encode_block(
                        context,
                        &pass,
                        &self.slot_buffers[slot],
                        0,
                        &self.blocks[n],
                        &s,
                        &self.shape,
                        self.mlp_tile_rows,
                    )?;
                }
                // The mapped arm has no pread step and no per-block
                // overwritten slot (the mapped buffer is read-only for the
                // whole run), so waiting per block is pure CPU-side
                // serialization with no data-hazard purpose: commands on one
                // Metal queue execute in commit order (crates/gpu/CLAUDE.md
                // Gotcha 8), so `s.x`'s per-block residual read/write stays
                // correct with no host wait between blocks. Skip the wait
                // there UNLESS a per-block host readback is requested
                // (`capture`/`overflow` below), which needs the GPU to have
                // actually finished before `read_stage`/`check_block` reads
                // `s.x` from the host. The pread arm always waits, unchanged.
                let needs_sync = capture.is_some() || self.overflow.is_some();
                if mapped_active && !needs_sync {
                    pass.commit();
                } else {
                    pass.commit_and_wait();
                }
                if let Some(c) = capture.as_mut() {
                    if n == 0 {
                        c.block_first = read_stage(&s.x, wide);
                    }
                    if n + 1 == self.shape.depth {
                        c.block_last = read_stage(&s.x, wide);
                    }
                }
                if let Some(o) = self.overflow.as_mut() {
                    o.check_block(n, &s.x, wide)?;
                }

                // A deepstack block's output feeds its own merger INSTEAD of
                // being carried anywhere: the block loop's residual in `s.x`
                // keeps running, the merger reads it, and the merger's
                // `[merged, out_hidden]` result is read back to the host,
                // where the trunk's prefill adds it after layer `k`. The
                // merger gets its own pass with its own wait, because the
                // host readback needs the GPU done even on the mapped arm
                // (which skips the per-block wait when nobody is reading).
                if let Some(k) = self.shape.deepstack.iter().position(|&i| i == n) {
                    let merger = self
                        .resident
                        .deepstack_mergers
                        .get(k)
                        .ok_or_else(|| {
                            RealForwardError::Unsupported(format!(
                                "deepstack block {n} is slot {k} of {} resolved mergers; the \
                                 install's resident index is missing the merger this config \
                                 declares",
                                self.resident.deepstack_mergers.len()
                            ))
                        })?;
                    let pass = context.begin_pass();
                    stages::encode_deepstack_merger(
                        context,
                        &pass,
                        self.sidecar_weights.as_ref().unwrap_or(weights),
                        merger,
                        &s,
                        &self.shape,
                    )?;
                    pass.commit_and_wait();
                    let count = s.merged * self.shape.out_hidden;
                    let ds_rows: Vec<u16> = gpu::read_buffer_f16(&s.out, 0, count)
                        .into_iter()
                        .map(|v| v.to_bits())
                        .collect();
                    if let Some(c) = capture.as_mut() {
                        c.deepstack.push(ds_rows.clone());
                    }
                    deepstack_out.push(DeepstackRows { rows: ds_rows });
                }
            }

            let pass = context.begin_pass();
            stages::encode_merger(
                context,
                &pass,
                self.sidecar_weights.as_ref().unwrap_or(weights),
                &self.resident,
                &s,
                &self.shape,
            )?;
            pass.commit_and_wait();

            let count = s.merged * self.shape.out_hidden;
            let rows: Vec<u16> = gpu::read_buffer_f16(&s.out, 0, count)
                .into_iter()
                .map(|v| v.to_bits())
                .collect();
            Ok((
                VisionEmbedding {
                    rows,
                    merged_tokens: s.merged,
                    out_hidden: self.shape.out_hidden,
                    grid: image.grid,
                    deepstack: deepstack_out,
                },
                capture,
            ))
        })
    }

    /// What the scratch for a page of `seq` patches costs, without allocating
    /// it.
    ///
    /// Exposed so the memory gate can assert the formula rather than
    /// restating it, which is the shape that rots. Calls
    /// [`scratch::scratch_bytes`] with THIS tower's own current
    /// [`Self::mlp_tile_rows`] (`scratch::VISION_MLP_TILE_ROWS` unless a test
    /// has overridden it) rather than restating the formula here a second
    /// time -- the exact shape [`VisionScratch::allocate`] itself now uses,
    /// so a caller sizing a page budget and the real allocator can never
    /// independently drift, whichever tile is in effect.
    pub(crate) fn scratch_bytes(&self, seq: usize) -> u64 {
        scratch::scratch_bytes(&self.shape, seq, self.mlp_tile_rows)
    }
}

/// Read `count` FP16 values out of a GPU buffer as raw bits.
fn read_stage(buffer: &gpu::MetalBuffer, count: usize) -> Vec<u16> {
    gpu::read_buffer_f16(buffer, 0, count)
        .into_iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Decode a resident FP16 tensor to host `f32`.
///
/// The sibling of `real_forward_utils::read_bf16_host`, and a separate
/// function rather than a dtype parameter on it: that one is the TEXT path's
/// reader and the whole basis of the name-scoped FP16 exception is that no
/// text-path helper reaches a `vision.` tensor.
fn read_fp16_host(
    weights: &gpu::ResidentGpuWeights,
    local_offset: usize,
    elems: usize,
) -> Result<Vec<f32>, RealForwardError> {
    let bytes = weights
        .data()
        .get(local_offset..local_offset + elems * 2)
        .ok_or_else(|| {
            RealForwardError::Unsupported(format!(
                "vision tensor at resident offset {local_offset} spans past the mapped region"
            ))
        })?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
