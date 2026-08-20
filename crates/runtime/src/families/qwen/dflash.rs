//! The DFlash2 block-diffusion drafter's state and context-KV half
//! (`docs/DFLASH2.md`). The draft forward itself is `dflash_draft.rs`.
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
use crate::families::qwen::{prefixed_layer_tensor, TRUNK_PREFIX};
use crate::real_forward::{RealForwardError, RealForwardRunner};
use crate::real_forward_utils::entry;

/// The drafter's tensor-name prefix, matching the repack walk's.
pub(crate) const DFLASH_PREFIX: &str = "dflash";

/// The published drafter's TRAINED block: 8 proposals plus the bonus row,
/// and the widest this port will build. Not the serving default -- see
/// [`DFLASH_SERVING_BLOCK`].
pub const DFLASH_BLOCK: usize = 8;

/// What `auto` actually serves, and it is 2 rather than the trained 8.
///
/// **MEASURED THROUGH THE REAL GENERATION LOOP ON TWO WORKLOADS**, which is
/// the comparison that matters and the one a single-prompt probe cannot
/// make. `Qwen3.8-27B` + DFlash2, 200 greedy tokens, against a ~22.1 tok/s
/// non-speculative arm:
///
/// | prompt | block 2 | block 4 | block 8 |
/// | --- | ---: | ---: | ---: |
/// | code (acceptance 0.93-0.98) | 1.47x | - | 1.34x |
/// | prose (acceptance 0.66-0.83) | 0.97x | 0.73x | 0.48x |
///
/// Block 2 wins on BOTH and its downside is bounded; block 8 is better only
/// where acceptance is already near 1 and is catastrophic where it is not.
/// The mechanism is the rollback term `docs/MTP_SPECULATIVE.md` names: a
/// rejected batched round on this recurrent family restores a whole
/// gated-DeltaNet snapshot and replays the accepted prefix, and the odds of
/// paying it rose 9% -> 27% -> 98% across those blocks. So the trained block
/// is the wrong SERVING block, and the same 2 the MTP head already defaults
/// to (`DEFAULT_SPECULATION_BLOCK`) is the right one -- two drafters, two
/// architectures, one answer, which is what makes it a property of this
/// engine rather than of either drafter.
///
/// A caller who has measured their own workload names a block explicitly;
/// `auto` is for the caller who has not.
pub const DFLASH_SERVING_BLOCK: usize = 2;

// The serving block is what `Auto` BUILDS at and the trained block bounds
// every block a caller may name, so the first can never exceed the second.
// A `const` block rather than a test: this is a relationship between two
// literals in one file, and a divergence should fail the BUILD rather than
// wait for whichever target happens to assert it (the same reasoning the
// oracles' protocol-parameter drift guard uses).
const _: () = assert!(DFLASH_SERVING_BLOCK <= DFLASH_BLOCK && DFLASH_SERVING_BLOCK > 0);

/// The trunk layers whose OUTPUT residual streams the drafter conditions
/// on. A property of the training run, read off the checkpoint's
/// `target_layer_ids` and pinned here as constants; `fc`'s input width
/// (`5 * hidden`) is the install-side cross-check that the count agrees
/// with the bytes.
pub const DFLASH_AUX_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

/// What the draft pass DIVIDES its residual stream by, so that stream fits
/// FP16.
///
/// **MEASURED, not chosen for taste.** This drafter's residual peaks at
/// 113,920 over its five layers on the real install, against FP16's largest
/// finite value of 65,504 -- so an unscaled pass overflows to `inf` inside
/// layer 2 and the first RMS norm past it turns `inf/inf` into NaN. That is
/// how the whole vocab came back NaN and every proposal came back token 0.
/// The growth is the checkpoint's own: the embedding is +/-0.08, and a conv
/// whose coefficients run to ~10 multiplies sublayer outputs in the
/// thousands. Both references are BF16 throughout and never meet a ceiling.
///
/// **The scale cancels exactly**, which is what makes this a change of
/// storage and not of arithmetic. `x` is read by exactly two things: RMS
/// norms, and the residual add whose addend is scaled by the same factor at
/// the conv that produces it. A power of two shifts the exponent and leaves
/// the mantissa alone, so the stored values ARE the reference's, at another
/// exponent.
///
/// **The eps is what makes that true, and it is easy to miss.** RMS norm is
/// only scale-invariant where `eps` is negligible: dividing the input by `S`
/// turns `eps` into an effective `eps * S^2`, because
/// `(x/S) / sqrt(mean(x^2)/S^2 + eps)` is `x / sqrt(mean(x^2) + eps * S^2)`.
/// On the EMBEDDING row, whose mean square is ~4e-4, that is not a rounding
/// difference but a factor of ten. Every norm that reads `x` therefore takes
/// [`DFLASH_RESIDUAL_EPS`], and the identity is exact rather than
/// approximate.
///
/// 2^3 leaves 4.6x of headroom over the measured peak while keeping the
/// embedding's elements inside FP16's NORMAL range (a larger scale pushes
/// them subnormal and spends mantissa bits on the one row that starts the
/// pass). It is a margin rather than a proof, which is why `dflash_select`
/// REFUSES a non-finite row instead of proposing token 0 -- a checkpoint
/// that overflows anyway fails by name rather than by silence.
pub const DFLASH_RESIDUAL_SCALE: f32 = 8.0;

/// The RMS epsilon for the norms that read the SCALED residual stream, so
/// they compute what the reference's unscaled norm computes. See
/// [`DFLASH_RESIDUAL_SCALE`]'s third paragraph for the derivation.
pub const DFLASH_RESIDUAL_EPS: f32 =
    super::RMS_EPS / (DFLASH_RESIDUAL_SCALE * DFLASH_RESIDUAL_SCALE);

/// The drafter's sliding window (all five layers are sliding_attention).
pub const DFLASH_WINDOW: usize = 2048;

/// Ring slack over the window, the trunk runner's own 128.
pub const DFLASH_RING_SLACK: usize = 128;

/// The mask token the drafter's proposal rows are embedded as.
pub const DFLASH_MASK_TOKEN: i32 = 248_070;

/// The selector's candidate count per proposal step.
pub const DFLASH_TOP_K: usize = 16;

fn dflash_layer_tensor(layer: usize, suffix: &str) -> String {
    prefixed_layer_tensor(DFLASH_PREFIX, layer, suffix)
}

/// Whether this install carries a DFlash2 drafter, read off the resident
/// index the same way head presence is: no manifest field, so nothing can
/// disagree with the bytes.
pub fn install_has_dflash(index: &ResidentIndex) -> bool {
    index.entries.contains_key("dflash.fc.weight")
}

/// What a caller asked for, mirroring [`super::MtpDraftPolicy`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DflashDraftPolicy {
    /// Build a drafter iff the install carries one.
    Auto,
    /// Never build one. What every MEASURING caller passes.
    Off,
    /// Build one at this block, and ERROR if the install has no drafter.
    Fixed(usize),
}

impl DflashDraftPolicy {
    /// Reads `MFERENCE_DFLASH_DRAFT`, with unset and unparsable both
    /// meaning `Auto` for the same reason `MtpDraftPolicy` chose it: a typo
    /// should not silently disable a feature the install can serve.
    pub fn from_env() -> Self {
        match std::env::var("MFERENCE_DFLASH_DRAFT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            None => DflashDraftPolicy::Auto,
            Some(0) => DflashDraftPolicy::Off,
            Some(n) => DflashDraftPolicy::Fixed(n),
        }
    }
}

/// The drafter's shape fields, DERIVED from tensor shapes at build rather
/// than restated, so the synthetic fixture exercises the same code the
/// real install does and a checkpoint that moved a width cannot drift past
/// open.
pub(crate) struct DflashShape {
    pub(crate) layers: usize,
    pub(crate) hidden: usize,
    pub(crate) head_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) inter: usize,
    /// The conv's per-row projection width: `2 sides * 2 taps * groups`.
    pub(crate) conv_rows: usize,
    /// The selector's bilinear rank.
    pub(crate) rank: usize,
    /// How many aux states `fc` fuses (`fc.cols / hidden`).
    pub(crate) aux_count: usize,
}

impl DflashShape {
    fn derive(index: &ResidentIndex, hidden: usize) -> Result<Self, RealForwardError> {
        let shape_of = |name: &str| -> Result<(u32, u32), RealForwardError> {
            let e = entry(index, name)?;
            Ok((e.shape.0, e.shape.1))
        };
        let head_dim = {
            let e = entry(index, &dflash_layer_tensor(0, "self_attn.q_norm.weight"))?;
            (e.size_bytes / 2) as usize
        };
        let (q_rows, _) = shape_of(&dflash_layer_tensor(0, "self_attn.q_proj.weight"))?;
        let (k_rows, _) = shape_of(&dflash_layer_tensor(0, "self_attn.k_proj.weight"))?;
        let (gate_rows, _) = shape_of(&dflash_layer_tensor(0, "mlp.gate_proj.weight"))?;
        let (_, rank) = shape_of("dflash.candidate_selector.predecessor_codebook")?;
        let (_, fc_cols) = shape_of("dflash.fc.weight")?;
        let layers = (0..)
            .take_while(|&l| {
                index
                    .entries
                    .contains_key(&dflash_layer_tensor(l, "input_layernorm.weight"))
            })
            .count();
        if head_dim == 0 || q_rows as usize % head_dim != 0 || k_rows as usize % head_dim != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "the DFlash2 drafter's shapes do not divide: q {q_rows}, k {k_rows}, head_dim \
                 {head_dim}"
            )));
        }
        let aux_count = fc_cols as usize / hidden;
        if aux_count * hidden != fc_cols as usize || aux_count == 0 {
            return Err(RealForwardError::Unsupported(format!(
                "dflash.fc.weight is [{}, {fc_cols}], not a whole number of {hidden}-wide aux \
                 states",
                shape_of("dflash.fc.weight")?.0
            )));
        }
        // `conv_rows` is the one width this struct COMPUTES rather than
        // reads, because the conv kernel needs the tap count and the group
        // size as numbers and the checkpoint states neither. So it is
        // checked against the tensor it describes: a drafter published at
        // another `conv_kernel_size` or `conv_group_size` is refused here,
        // naming the shape that moved, rather than handing
        // `encode_gemm_any` a row count the weight does not have.
        let conv_rows = 2 * gpu::DFLASH_TAPS as usize * (hidden / gpu::DFLASH_GROUP_SIZE as usize);
        let (proj_rows, _) = shape_of(&dflash_layer_tensor(
            0,
            "attention_conv.kernel_projection.weight",
        ))?;
        if proj_rows as usize != conv_rows {
            return Err(RealForwardError::Unsupported(format!(
                "attention_conv.kernel_projection.weight has {proj_rows} rows against the \
                 {conv_rows} this port's conv dispatches (2 sides x {} taps x {hidden}/{} \
                 groups); this drafter's conv shape is not the published one",
                gpu::DFLASH_TAPS,
                gpu::DFLASH_GROUP_SIZE
            )));
        }
        Ok(Self {
            layers,
            hidden,
            head_dim,
            num_heads: q_rows as usize / head_dim,
            num_kv_heads: k_rows as usize / head_dim,
            inter: gate_rows as usize,
            conv_rows,
            rank: rank as usize,
            aux_count,
        })
    }
}

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

/// Why DFlash2 speculation cannot run on this runner, or `None` if it can.
/// The architectural checks are `MtpState::speculation_blocker`'s (the
/// verify pass is the same one); only the drafter-presence clause differs.
pub fn dflash_speculation_blocker(
    index: &ResidentIndex,
    arch: &ArchConfig,
    has_drafter: bool,
) -> Option<String> {
    if arch.num_experts != 0 {
        return Some(format!(
            "the batched verify is dense-only and this install routes to {} experts",
            arch.num_experts
        ));
    }
    let full = (0..arch.num_layers as usize).find(|&l| !arch.layer_is_linear(l))?;
    let probe = prefixed_layer_tensor(TRUNK_PREFIX, full, "self_attn.q_proj.weight");
    match index.entries.get(&probe) {
        None => Some(format!(
            "cannot tell whether the verify can run: {probe} is missing"
        )),
        Some(e) if e.dtype != 4 => Some(format!(
            "the batched verify is INT4-only and this install's {probe} is dtype {}",
            e.dtype
        )),
        Some(_) => {
            if !has_drafter {
                return Some(
                    "this install carries no DFlash2 drafter (dflash.fc.weight is not in the \
                     resident index)"
                        .to_string(),
                );
            }
            None
        }
    }
}

impl RealForwardRunner {
    /// Why DFlash2 speculation cannot run on this runner, or `None` if it
    /// can; the CLI's hard-fail message for `--speculative-drafter dflash`.
    pub fn dflash_speculation_blocker(&self) -> Option<String> {
        dflash_speculation_blocker(&self.index, &self.arch, self.real_dflash.is_some())
    }
}
