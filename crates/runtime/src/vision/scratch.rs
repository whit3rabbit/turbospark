//! Per-page GPU scratch for one tower run (ROADMAP M-V4).
//!
//! # Sized by the page, allocated per image, dropped after the readback
//!
//! Unlike `DecodeScratch`, which is allocated once at open because every
//! decode step is the same shape, a tower run's shape is the image's: a
//! 1024x1280 OCR page is 5,120 patches and the largest page the processor
//! accepts is 64,516. So this is built per `encode_image` and dropped with
//! the embedding, which is the bulk-OCR constant-memory story -- peak vision
//! residency is `2 x block_stride` of pinned slots plus one page's scratch,
//! and the scratch goes away between pages.
//!
//! At the real widths that was about 152 MB for a 5,120-patch page and about
//! 1.9 GB for the 64,516-patch extreme, before [`VISION_MLP_TILE_ROWS`]
//! (Part B1): `fc1 -> gelu -> fc2` is row-independent, so a fixed row tile
//! caps [`VisionScratch::h1`] (the 555 MB term at the extreme page) with
//! identical arithmetic -- the loop runs the same three dispatches once per
//! tile instead of once for the whole page. Attention cannot be tiled the
//! same way -- it is bidirectional over the whole page -- so that lever
//! bounds the MLP term alone and nothing else. The tiling is unconditional
//! rather than gated on page size: an ordinary 1024x1280 OCR page is 5,120
//! patches, already more than one [`VISION_MLP_TILE_ROWS`]-row tile, so the
//! byte saving and the loop both apply to it too, not only to the extreme
//! case that motivated the lever.
//!
//! # What is reused and what is not
//!
//! `normed` serves BOTH the block norms and the merger's norm, and that is
//! not a saving so much as the merger's own shape: its norm is over `hidden`
//! per patch row, and the reshape to `[merged, hidden * merge^2]` that
//! follows is a pure VIEW of the same bytes, because M-V1 already emits patch
//! rows in merge-window order. There is no permutation to do and therefore no
//! second buffer to do it into.
//!
//! `proj` serves both the attention projection and the MLP's `fc2`, which are
//! never live at once. `q`/`k`/`v` are separate because the fused `qkv`
//! weight is run as three matmuls at three row offsets (the shape
//! `crates/gpu/tests/vision_block_parity.rs` established) and each writes a
//! contiguous `[seq, hidden]` the rope and attention kernels can read
//! directly.
//!
//! # Buffer aliasing (Part B2)
//!
//! `attn`, `proj` and `m1` are ALIASES rather than separate allocations --
//! `metal::Buffer::clone()` retains the same underlying `MTLBuffer`, so
//! `s.attn`, `s.proj` and `s.m1` are each a second Rust handle onto bytes
//! `s.normed`, `s.q` and `s.k` already own. This is sound under the SAME
//! commit-order guarantee `crate::vision::mod`'s block loop already states
//! for the two streamer slots (`crates/gpu/CLAUDE.md` Gotcha 8: dispatches on
//! one command buffer execute in commit order), applied at a finer grain --
//! every read of an aliased buffer under its old name is dispatched, in this
//! pass, strictly before the next write under its new name:
//!
//! - **`attn` aliases `normed`.** `normed`'s three reads (the qkv matmuls)
//!   all happen before `attn` is first written (the attention kernel);
//!   `attn`'s one read (the output projection) happens before `normed` is
//!   next written (block N's `norm2`, or the merger's own norm after the
//!   last block).
//! - **`proj` aliases `q`.** `q`'s last read (the attention kernel) happens
//!   before `proj` is first written (the output projection); `proj` is then
//!   read (the first residual add) and overwritten (the MLP's `fc2`) and
//!   read again (the second residual add) entirely within the SAME block,
//!   before the next block's `q` write.
//! - **`m1` aliases `k`.** `k` is read by every block's rope-and-attention
//!   pair and never touched by the merger; `m1` is written and read ONLY by
//!   the merger, which runs once, after the last block. The two are also the
//!   SAME byte count for any valid page: `merged * merger_input ==
//!   (seq / merge^2) * (hidden * merge^2) == seq * hidden`, an algebraic
//!   identity rather than a coincidence of one shape, since `allocate`
//!   already refuses a `seq` that is not a whole number of merge windows.
//!
//! `v` is NOT aliased: it has no dead point before the block loop ends (every
//! block's attention kernel reads it), so there is no free buffer to fold it
//! into. Five physical wide (`seq * hidden`) allocations remain: `x`,
//! `normed`(+`attn`), `q`(+`proj`), `k`(+`m1`), `v` -- against seven before
//! this change, before counting `m1`'s own separate allocation as an eighth
//! it no longer needs.
//!
//! This rests on the SERIAL encoder: a future prefetch or overlap arm that
//! lets two blocks' dispatches interleave on different command buffers would
//! have to un-alias first, since the ordering argument above assumes one
//! command buffer's dispatches never race another's.

use crate::real_forward_types::RealForwardError;
use crate::vision::shape::VisionShape;

/// Every buffer one tower run needs, plus what they cost.
pub(crate) struct VisionScratch {
    /// Patch rows, `[seq, patch_dim]` FP16. The patch-embedding GEMM's `a`.
    pub(crate) rows: gpu::MetalBuffer,
    /// The residual stream, `[seq, hidden]`.
    pub(crate) x: gpu::MetalBuffer,
    /// Norm output, `[seq, hidden]`; also the merger's normed rows.
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) q: gpu::MetalBuffer,
    pub(crate) k: gpu::MetalBuffer,
    pub(crate) v: gpu::MetalBuffer,
    /// Attention output, `[seq, hidden]`. ALIASES `normed` (Part B2, see the
    /// module doc) -- not a separate allocation.
    pub(crate) attn: gpu::MetalBuffer,
    /// A sublayer's output before its residual add, `[seq, hidden]`. Serves
    /// the attention projection and the MLP's `fc2`. ALIASES `q` (Part B2)
    /// -- not a separate allocation.
    pub(crate) proj: gpu::MetalBuffer,
    /// The MLP's hidden activation, `[min(seq, tile_rows), intermediate]`
    /// (Part B1's row tiling; see [`VISION_MLP_TILE_ROWS`]) -- one tile's
    /// worth, reused across every tile of the page rather than sized for the
    /// whole thing.
    pub(crate) h1: gpu::MetalBuffer,
    /// Rope frequency rows, `[seq, head_dim / 2]` F32 (AGENTS.md/CLAUDE.md
    /// B7: the angle at pair 0 equals the raw patch coordinate and reaches
    /// the tens on a wide grid, where FP16's step is a real, avoidable
    /// error), one row per patch shared by every head.
    pub(crate) freqs: gpu::MetalBuffer,
    /// The interpolated position rows, `[seq, hidden]` FP16, added to the
    /// patch embedding. Uploaded rather than gathered: see
    /// `stages::pos_embed_rows`.
    pub(crate) pos: gpu::MetalBuffer,
    /// The merger's hidden activation, `[merged, merger_input]`. ALIASES `k`
    /// (Part B2) -- not a separate allocation; the two are the same byte
    /// count for any valid page (see the module doc).
    pub(crate) m1: gpu::MetalBuffer,
    /// The tower's output, `[merged, out_hidden]`.
    pub(crate) out: gpu::MetalBuffer,
    pub(crate) seq: usize,
    pub(crate) merged: usize,
    /// Total bytes allocated here. Reported rather than recomputed by a
    /// caller, so the residency assertion in the gate compares against what
    /// was actually asked for.
    pub(crate) bytes: u64,
}

/// FP16 bytes for `elems` elements.
fn fp16(elems: usize) -> u64 {
    (elems as u64) * 2
}

/// F32 bytes for `elems` elements. Only `freqs` is F32 (AGENTS.md/CLAUDE.md
/// B7); every other scratch buffer here is FP16.
fn f32_bytes(elems: usize) -> u64 {
    (elems as u64) * 4
}

/// Row tile for the MLP's `fc1 -> gelu -> fc2` (`crate::vision::block`'s
/// "--- MLP half ---" loop). `fc1 -> gelu -> fc2` is row-independent, so a
/// fixed tile caps [`VisionScratch::h1`] at
/// `min(seq, VISION_MLP_TILE_ROWS) * intermediate` elements regardless of
/// how many patches the page has, with identical arithmetic to running
/// every row at once.
///
/// Named exactly this because [`scratch_bytes`] and the tiling loop in
/// `block.rs` both reference it, and the two must never drift apart about
/// what "the tile" means.
pub(crate) const VISION_MLP_TILE_ROWS: usize = 2048;

/// Bytes one page of `seq` patches costs in scratch, at a given MLP row
/// tile.
///
/// The single formula both [`VisionScratch::allocate`] (which actually
/// allocates at this size) and `VisionTower::scratch_bytes` (which predicts
/// it for a caller sizing a page budget, and which passes its own current
/// [`VisionTower::mlp_tile_rows`] here) call, so the two can never
/// independently drift -- only whether either one is applied at all, which
/// is exactly the failure mode `VisionTower::scratch_bytes`'s own doc comment
/// exists to let a gate assert against.
///
/// [`VisionTower::mlp_tile_rows`]: super::VisionTower
pub(crate) fn scratch_bytes(shape: &VisionShape, seq: usize, tile: usize) -> u64 {
    let h = shape.hidden;
    let merged = seq / shape.patches_per_token();
    let h1_rows = tile.min(seq);
    fp16(seq * shape.patch_dim)
        + f32_bytes(seq * (shape.head_dim / 2))
        + fp16(seq * h)
        // x, normed(+attn), q(+proj), k(+m1), v -- five physical allocations
        // (Part B2's buffer aliasing; see the module doc). `m1`'s own term is
        // gone because it is `k`'s buffer under another name, never a
        // separate allocation, and the two are always the same byte count.
        + fp16(seq * h) * 5
        + fp16(h1_rows * shape.intermediate)
        + fp16(merged * shape.out_hidden)
}

impl VisionScratch {
    /// Allocate for a page of `seq` patches, with the MLP's hidden
    /// activation capped at `min(seq, tile_rows)` rows (Part B1's row
    /// tiling; see [`VISION_MLP_TILE_ROWS`]).
    ///
    /// `rows`, `freqs` and `pos` are uploaded WITH data because all three are
    /// host-computed per image; the rest are output buffers.
    pub(crate) fn allocate(
        context: &gpu::MetalContext,
        shape: &VisionShape,
        seq: usize,
        tile_rows: usize,
        rows: &[half::f16],
        freqs: &[f32],
        pos: &[half::f16],
    ) -> Result<Self, RealForwardError> {
        if seq == 0 {
            return Err(RealForwardError::Unsupported(
                "vision tower asked to run on a page of zero patches".to_string(),
            ));
        }
        let per_token = shape.patches_per_token();
        if seq % per_token != 0 {
            // Refused rather than truncated: the merger reshapes `[seq,
            // hidden]` into `[seq / merge^2, hidden * merge^2]`, and a
            // partial window would silently drop the tail patches into a row
            // that is short by exactly the amount nobody checks.
            return Err(RealForwardError::Unsupported(format!(
                "vision page has {seq} patches, which is not a whole number of {per_token}-patch \
                 merge windows"
            )));
        }
        let merged = seq / per_token;
        let h = shape.hidden;
        let m = shape.merger_input();

        for (label, have, want) in [
            ("patch rows", rows.len(), seq * shape.patch_dim),
            (
                "rope frequency rows",
                freqs.len(),
                seq * (shape.head_dim / 2),
            ),
            ("position rows", pos.len(), seq * h),
        ] {
            if have != want {
                return Err(RealForwardError::Unsupported(format!(
                    "vision {label}: {have} elements, expected {want}"
                )));
            }
        }

        let wide = fp16(seq * h);
        let h1_rows = tile_rows.min(seq);
        let bytes = scratch_bytes(shape, seq, tile_rows);

        // Part B2: `attn`, `proj` and `m1` are ALIASES (`Buffer::clone()`
        // retains the same underlying `MTLBuffer`), never separate
        // allocations -- see the module doc for why this is safe under the
        // serial encoder's commit order. `m1`'s alleged shape,
        // `[merged, merger_input]`, is `merged * m` elements, which the
        // module doc proves algebraically equal to `seq * h` for any valid
        // page, so aliasing `k`'s `wide`-sized buffer under that name loses
        // no capacity.
        debug_assert_eq!(
            merged * m,
            seq * h,
            "m1 and k must be the same element count for this alias to be sound"
        );
        let normed = context.new_output_buffer(wide);
        let q = context.new_output_buffer(wide);
        let k = context.new_output_buffer(wide);
        let attn = normed.clone();
        let proj = q.clone();
        let m1 = k.clone();

        Ok(Self {
            rows: context.new_buffer_with_data(rows),
            x: context.new_output_buffer(wide),
            normed,
            q,
            k,
            v: context.new_output_buffer(wide),
            attn,
            proj,
            h1: context.new_output_buffer(fp16(h1_rows * shape.intermediate)),
            freqs: context.new_buffer_with_data(freqs),
            pos: context.new_buffer_with_data(pos),
            m1,
            out: context.new_output_buffer(fp16(merged * shape.out_hidden)),
            seq,
            merged,
            bytes,
        })
    }
}
