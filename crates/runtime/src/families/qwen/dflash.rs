//! The DFlash2 block-diffusion drafter's state and context-KV half
//! (`docs/DFLASH2.md`). The draft forward itself is `dflash_draft/`.
//!
//! Two facts about this drafter organize everything below:
//!
//! 1. **Its KV cache holds TARGET-derived rows for committed positions.**
//!    Every round, the states the trunk's verify pass captured at layers
//!    `[5, 19, 33, 47, 61]` are fused by `fc`, normed by `hidden_norm`, and
//!    projected by the drafter's own `k_proj`/`v_proj` into the cache. The
//!    drafter never runs its layers over the context; a draft pass is
//!    always and only the `block + 1` query rows (bonus token + masks).
//! 2. **The capture buffer is the fc input, already laid out.** Five aux
//!    states per row, concatenated `[row][aux][hidden]`, is exactly the
//!    `[rows, 5 * hidden]` matrix `fc` multiplies, so nothing rearranges a
//!    captured row between the trunk pass and the context write.
//!
//! The cursor discipline, which is where a quiet bug would live:
//!
//! - `kv.position()` counts positions whose rows are FINAL (target-derived
//!   for committed tokens). Query rows written by a draft sit AT or BEYOND
//!   the cursor and are always rewritten before anything reads them again:
//!   the next round's context write covers the accepted prefix, and the
//!   next draft rewrites its own block.
//! - `capture_base`/`capture_rows` describe the capture buffer's contents:
//!   set by every trunk pass (`produce`: one row; `produce_batched`: the
//!   batch). A rollback's shortened re-verify overwrites the same leading
//!   rows with identical values, so capture contents never need unwinding.
//! - At `dflash_draft_block` entry the invariant is `kv.position() ==
//!   capture_base + accepted - 1`... in the loop's terms, the context
//!   write covers `[kv.position(), base)` from capture rows
//!   `[0, base - kv.position())`, and the invariant checked is simply
//!   `capture_base == kv.position()` once the accepted prefix has been
//!   committed. It fails loudly rather than drafting off stale rows.
//!
//! Like `MtpState`, this allocates NOTHING unless a drafter was asked for
//! at open, so a dflash-carrying install with drafting off is
//! byte-identical in output and footprint to a dflashless one.

use model_io::{ArchConfig, ResidentIndex};

use super::batched::BatchedScratch;
pub use super::dflash_state::{dflash_speculation_blocker, install_has_dflash};
pub(crate) use super::dflash_state::{
    DflashDraftPolicy, DflashShape, DFLASH_AUX_LAYERS, DFLASH_MASK_TOKEN, DFLASH_RESIDUAL_EPS,
    DFLASH_RESIDUAL_SCALE, DFLASH_RING_SLACK, DFLASH_SERVING_BLOCK, DFLASH_TOP_K, DFLASH_WINDOW,
};
use crate::real_forward::{RealForwardError, RealForwardRunner};

/// The drafter's per-open state.
pub(crate) struct DflashState {
    pub(crate) kv: gpu::KvCacheManager,
    /// `[MAX_BATCH_ROWS][aux_count][hidden]`, the fc input layout.
    pub(crate) capture: gpu::MetalBuffer,
    /// Position of capture row 0; set by every trunk pass.
    pub(crate) capture_base: usize,
    /// Valid leading rows in the capture buffer.
    pub(crate) capture_rows: usize,
    /// Proposals per round.
    pub(crate) block: usize,
    pub(crate) shape: DflashShape,
    /// The M-row scratch the trunk's verify pass runs on, owned here so a
    /// dflash-only install (no MTP head) can still verify batched.
    pub(crate) batched: BatchedScratch,
    // The draft pass's own buffers, sized MAX_BATCH_ROWS rows.
    pub(crate) x: gpu::MetalBuffer,
    pub(crate) normed: gpu::MetalBuffer,
    pub(crate) conv_p: gpu::MetalBuffer,
    pub(crate) conv_f: gpu::MetalBuffer,
    pub(crate) sub_out: gpu::MetalBuffer,
    pub(crate) coeffs: gpu::MetalBuffer,
    pub(crate) q: gpu::MetalBuffer,
    pub(crate) attn_out: gpu::MetalBuffer,
    pub(crate) attn: gpu::AttentionScratch,
    pub(crate) ffn_gate: gpu::MetalBuffer,
    pub(crate) ffn_up: gpu::MetalBuffer,
    pub(crate) ffn_act: gpu::MetalBuffer,
    /// fc / hidden_norm outputs for the context write, `[rows][hidden]`.
    pub(crate) ctx_combined: gpu::MetalBuffer,
    pub(crate) ctx_normed: gpu::MetalBuffer,
    /// The selector's `hidden_projection` output, `[rows][rank]`.
    pub(crate) hproj: gpu::MetalBuffer,
    /// The drafter head's logits, `[rows][vocab]` (the TRUNK's lm_head).
    pub(crate) logits: gpu::MetalBuffer,
}

impl DflashState {
    /// Returns `None` when no drafter was asked for; ERRORS when one was
    /// asked for and the install cannot serve it, exactly `MtpState`'s
    /// two-condition split.
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        index: &ResidentIndex,
        arch: &ArchConfig,
        max_context: usize,
        policy: DflashDraftPolicy,
        gdn_shape: gpu::GdnShape,
    ) -> Result<Option<Self>, RealForwardError> {
        let block = match policy {
            DflashDraftPolicy::Off => return Ok(None),
            DflashDraftPolicy::Auto => {
                if !install_has_dflash(index) {
                    return Ok(None);
                }
                DFLASH_SERVING_BLOCK
            }
            DflashDraftPolicy::Fixed(block) => {
                if !install_has_dflash(index) {
                    return Err(RealForwardError::Unsupported(format!(
                        "MFERENCE_DFLASH_DRAFT={block} asks for the DFlash2 drafter, but this \
                         install carries none (dflash.fc.weight is not in the resident index); \
                         stream it beside the trunk (docs/DFLASH2.md)"
                    )));
                }
                // Refused rather than promoted to 1: a caller that named a
                // block of zero named something no drafter can serve, and
                // silently drafting one token gives them a live drafter
                // whose every round the loop then refuses. `Off` is the
                // spelling for "no drafter".
                if block == 0 {
                    return Err(RealForwardError::Unsupported(
                        "the DFlash2 drafter was asked for at a block of 0; use Off (or \
                         MFERENCE_DFLASH_DRAFT=0) to turn drafting off"
                            .to_string(),
                    ));
                }
                block
            }
        };
        if block + 1 > gpu::MAX_BATCH_ROWS {
            return Err(RealForwardError::Unsupported(format!(
                "DFlash2 block {block} verifies {} rows, over the batched kernel's cap of {}",
                block + 1,
                gpu::MAX_BATCH_ROWS
            )));
        }

        let hidden = arch.hidden_size as usize;
        let shape = DflashShape::derive(index, hidden)?;
        // The install-side half of the aux contract: fc fuses as many
        // states as there are taps below.
        if shape.aux_count != DFLASH_AUX_LAYERS.len() {
            return Err(RealForwardError::Unsupported(format!(
                "dflash.fc.weight fuses {} aux states but this drafter taps {} trunk layers",
                shape.aux_count,
                DFLASH_AUX_LAYERS.len()
            )));
        }

        // The drafter's own KV: `layers` sliding layers at ITS kv width, in
        // rings of window + slack, built from a cloned arch whose kv fields
        // are the DRAFTER's and not the trunk's. Nothing else reads this
        // manager, so the clone's unrelated fields are inert.
        let mut kv_arch = arch.clone();
        kv_arch.num_layers = shape.layers as i64;
        kv_arch.full_attention_layer_mask = vec![0u8; shape.layers];
        kv_arch.num_kv_heads = shape.num_kv_heads as i64;
        kv_arch.num_full_kv_heads = shape.num_kv_heads as i64;
        kv_arch.head_dim = shape.head_dim as i64;
        kv_arch.full_head_dim = shape.head_dim as i64;
        let kv = gpu::KvCacheManager::new(
            context.device(),
            &kv_arch,
            max_context,
            true,
            Some(DFLASH_WINDOW),
            1,
            Some(DFLASH_WINDOW + DFLASH_RING_SLACK),
        )
        .map_err(RealForwardError::Gpu)?;
        let kv_stride = shape.num_kv_heads * shape.head_dim * 2;
        for layer in 0..shape.layers {
            if kv.stride(layer) != kv_stride {
                return Err(RealForwardError::Unsupported(format!(
                    "the drafter's KV stride {} is not the {kv_stride} bytes a batched \
                     projection writes per row",
                    kv.stride(layer)
                )));
            }
        }

        let rows = gpu::MAX_BATCH_ROWS as u64;
        let h = hidden as u64;
        let inter = shape.inter as u64;
        let vocab = arch.vocab_size as u64;
        let halfs = |n: u64| context.new_output_buffer(rows * n * 2);
        let state = Self {
            kv,
            capture: context.new_output_buffer(rows * shape.aux_count as u64 * h * 2),
            capture_base: 0,
            capture_rows: 0,
            block,
            attn: gpu::AttentionScratch::new(
                context,
                shape.num_heads as u32,
                shape.head_dim as u32,
            ),
            batched: BatchedScratch::new(context, arch, gdn_shape, block + 1),
            x: halfs(h),
            normed: halfs(h),
            conv_p: halfs(h),
            conv_f: halfs(h),
            sub_out: halfs(h),
            coeffs: context.new_output_buffer(rows * shape.conv_rows as u64 * 2),
            q: halfs(shape.num_heads as u64 * shape.head_dim as u64),
            attn_out: halfs(shape.num_heads as u64 * shape.head_dim as u64),
            ffn_gate: halfs(inter),
            ffn_up: halfs(inter),
            ffn_act: halfs(inter),
            ctx_combined: halfs(h),
            ctx_normed: halfs(h),
            hproj: context.new_output_buffer(rows * shape.rank as u64 * 2),
            logits: halfs(vocab),
            shape,
        };
        Ok(Some(state))
    }

    pub(crate) fn reset(&mut self) {
        self.kv.reset();
        self.capture_base = 0;
        self.capture_rows = 0;
    }

    /// Records what a trunk pass just captured. Called by the hooks in
    /// `produce_real_qwen` / `produce_batched` after their pass commits.
    pub(crate) fn note_capture(&mut self, base: usize, rows: usize) {
        self.capture_base = base;
        self.capture_rows = rows;
    }

    /// Which capture column trunk `layer` feeds, or `None` when the
    /// drafter does not tap it.
    pub(crate) fn aux_slot(&self, layer: usize) -> Option<usize> {
        DFLASH_AUX_LAYERS.iter().position(|&l| l == layer)
    }
}

impl RealForwardRunner {
    /// Why DFlash2 speculation cannot run on this runner, or `None` if it
    /// can; the CLI's hard-fail message for `--speculative-drafter dflash`.
    pub fn dflash_speculation_blocker(&self) -> Option<String> {
        dflash_speculation_blocker(&self.index, &self.arch, self.real_dflash.is_some())
    }
}
