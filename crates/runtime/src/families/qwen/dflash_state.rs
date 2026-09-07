use model_io::{ArchConfig, ResidentIndex};

use crate::families::qwen::{prefixed_layer_tensor, MOE_SPECULATION_BLOCKER_MARKER, TRUNK_PREFIX};
use crate::real_forward::RealForwardError;
use crate::real_forward_utils::entry;

/// The drafter's tensor-name prefix, matching the repack walk's.
pub(crate) const DFLASH_PREFIX: &str = "dflash";

/// The published drafter's TRAINED block: 8 proposals plus the bonus row,
/// and the widest this port will build. Not the serving default -- see
/// [`DFLASH_SERVING_BLOCK`].
pub const DFLASH_BLOCK: usize = 8;

/// What `auto` actually serves, and it is 2 rather than the trained 8.
///
/// **MEASURED ON THREE WORKLOADS THROUGH `dflash2_accept_length_probe`**,
/// 600 greedy tokens each, `Qwen3.8-27B` + DFlash2. The accepted-per-round
/// and rollback columns are deterministic and reproduce to the last digit
/// across runs; the wall clock is not and is read for ORDERING only:
///
/// | workload | per-position acceptance | block 2 | 4 | 7 | 8 |
/// | --- | --- | ---: | ---: | ---: | ---: |
/// | prose | 0.53-0.81 | 1.23 | 1.60 | 1.85 | 1.84 |
/// | code | 0.84-0.94 | 1.74 | 2.98 | 4.37 | 4.56 |
/// | math | 0.86-0.98 | 1.89 | 3.49 | 5.59 | 5.67 |
///
/// (accepted per round; a bigger block always accepts more per round and
/// that is not the question.) The rollback RATE is:
///
/// | workload | block 2 | 4 | 7 | 8 |
/// | --- | ---: | ---: | ---: | ---: |
/// | prose | 52% | 84% | 94% | 96% |
/// | code | 17% | 38% | 59% | 67% |
/// | math | 8% | 20% | 37% | 45% |
///
/// **Block 2 is fastest on all three, and the ordering is monotone in the
/// block on all three.** The mechanism is the rollback term
/// `docs/MTP_SPECULATIVE.md` names: a rejected batched round on this
/// recurrent family restores a whole gated-DeltaNet snapshot and replays the
/// accepted prefix, and a block of B needs ALL B proposals to land, so even
/// per-position acceptance of 0.93 compounds to a ~45% rollback rate at 8.
/// So the trained block is the wrong SERVING block, and the same 2 the MTP
/// head already defaults to (`DEFAULT_SPECULATION_BLOCK`) is the right one --
/// two drafters, two architectures, one answer, which is what makes it a
/// property of this engine rather than of either drafter.
///
/// **A LARGE BLOCK IS NOT RESCUED BY HIGH ACCEPTANCE, which the two-workload
/// version of this table could still be read as implying.** `math` accepts
/// 0.86-0.98 per position -- higher than `code` -- and block 8 still loses to
/// block 2 there. Acceptance decides whether speculation pays AT ALL (prose
/// loses at every block but 2); the BLOCK is decided by compounding and
/// rollback cost, nearly independently of it.
///
/// A caller who has measured their own workload names a block explicitly;
/// `auto` is for the caller who has not.
pub const DFLASH_SERVING_BLOCK: usize = 2;

const _: () = assert!(DFLASH_SERVING_BLOCK <= DFLASH_BLOCK && DFLASH_SERVING_BLOCK > 0);

/// The trunk layers whose OUTPUT residual streams the drafter conditions
/// on. A property of the training run, read off the checkpoint's
/// `target_layer_ids` and pinned here as constants; `fc`'s input width
/// (`5 * hidden`) is the install-side cross-check that the count agrees
/// with the bytes.
pub const DFLASH_AUX_LAYERS: [usize; 5] = [5, 19, 33, 47, 61];

/// What the draft pass DIVIDES its residual stream by, so that stream fits
/// FP16.
pub const DFLASH_RESIDUAL_SCALE: f32 = 8.0;

/// The RMS epsilon for the norms that read the SCALED residual stream, so
/// they compute what the reference's unscaled norm computes.
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

pub(crate) fn dflash_layer_tensor(layer: usize, suffix: &str) -> String {
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
    /// Reads `TURBOSPARK_DFLASH_DRAFT` (or legacy `TURBOSPARK_DFLASH_DRAFT`),
    /// with unset and unparsable both meaning `Auto` for the same reason
    /// `MtpDraftPolicy` chose it: a typo should not silently disable a feature
    /// the install can serve.
    pub fn from_env() -> Self {
        match std::env::var("TURBOSPARK_DFLASH_DRAFT")
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
#[derive(Debug)]
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
    pub(crate) fn derive(
        index: &ResidentIndex,
        hidden: usize,
        vocab: usize,
    ) -> Result<Self, RealForwardError> {
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
        let (pred_rows, rank) = shape_of("dflash.candidate_selector.predecessor_codebook")?;
        let (succ_rows, succ_rank) = shape_of("dflash.candidate_selector.successor_codebook")?;
        if pred_rows as usize != vocab || succ_rows as usize != vocab || succ_rank != rank {
            return Err(RealForwardError::Unsupported(format!(
                "dflash candidate selector codebook shape mismatch: pred [{pred_rows}, {rank}], succ [{succ_rows}, {succ_rank}], expected [{vocab}, {rank}]"
            )));
        }
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

/// Why DFlash2 speculation cannot run on this runner, or `None` if it can.
///
/// **THE MoE ARM MIRRORS `speculation_blocker`'s AND IS A POLICY, NOT A
/// CAPABILITY.** The batched routed verify runs since ROADMAP Phase 3
/// (`moe_batch.rs`); what no MoE checkpoint of this architecture ships is a
/// drafter. The published DFlash2 checkpoint (`incoai/Qwen3.8-27B-DFlash2`,
/// `docs/DFLASH2.md`) targets the DENSE half, so an MoE install has nothing
/// to propose with however good the verify is. The INT4 arm below is the
/// other kind: that one really is a kernel this engine does not have.
pub fn dflash_speculation_blocker(
    index: &ResidentIndex,
    arch: &ArchConfig,
    has_drafter: bool,
) -> Option<String> {
    if arch.num_experts != 0 {
        return Some(format!(
            "{MOE_SPECULATION_BLOCKER_MARKER}: this install routes to {} experts, and no \
             published MoE checkpoint of this architecture ships a DFlash2 drafter (the \
             published one targets the dense half). The batched routed verify itself \
             runs, so this is a checkpoint gap and not a missing kernel",
            arch.num_experts
        ));
    }
    let full = (0..arch.num_layers as usize).find(|&l| !arch.layer_is_linear(l));
    let Some(full) = full else {
        return Some(
            "the batched verify needs a full-attention layer to probe and this \
             install declares none"
                .to_string(),
        );
    };
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
                // CARRIES THE POINTER `DflashState::build`'s open-time error
                // carries, because since the headless arm landed in
                // `draft_policies` this message is what a named block on a
                // DENSE drafter-less install actually sees. On a dense
                // install the advice is correct and actionable: the published
                // drafter targets this half of the architecture. The MoE arm
                // above is the one where it would be a wild goose chase, and
                // that arm returns first.
                return Some(
                    "this install carries no DFlash2 drafter (dflash.fc.weight is not in the \
                     resident index); stream it beside the trunk (docs/DFLASH2.md)"
                        .to_string(),
                );
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_install_with_no_full_attention_layer_is_refused_by_name() {
        let mut arch = turbospark_repack::tiny_qwen_gdn_dense_arch(256, 4);
        arch.full_attention_layer_mask = vec![2; 4];
        let index = ResidentIndex {
            header: model_io::ResidentIndexHeader {
                index_size: 0,
                resident_size: 0,
                entry_count: 0,
            },
            entries: std::collections::HashMap::new(),
        };
        let reason = dflash_speculation_blocker(&index, &arch, true)
            .expect("should be refused because install declares no full-attention layer");
        assert!(
            reason.contains("declares none"),
            "expected 'declares none' in refusal, got: {reason}"
        );
    }

    #[test]
    fn derive_refuses_mismatched_codebook_shape() {
        let mut entries = std::collections::HashMap::new();
        let make_entry = |shape: (u32, u32), size_bytes: u64| model_io::ResidentIndexEntry {
            name: String::new(),
            dtype: 0,
            file_offset: 0,
            size_bytes,
            shape: (shape.0, shape.1, 1, 1),
            scale_offset: 0,
            scale_size: 0,
            bias_offset: 0,
            bias_size: 0,
        };
        entries.insert(
            dflash_layer_tensor(0, "self_attn.q_norm.weight"),
            make_entry((64, 1), 128),
        );
        entries.insert(
            dflash_layer_tensor(0, "self_attn.q_proj.weight"),
            make_entry((64, 64), 8192),
        );
        entries.insert(
            dflash_layer_tensor(0, "self_attn.k_proj.weight"),
            make_entry((64, 64), 8192),
        );
        entries.insert(
            dflash_layer_tensor(0, "mlp.gate_proj.weight"),
            make_entry((128, 64), 16384),
        );
        entries.insert(
            "dflash.candidate_selector.predecessor_codebook".to_string(),
            make_entry((100, 32), 6400),
        );
        entries.insert(
            "dflash.candidate_selector.successor_codebook".to_string(),
            make_entry((200, 32), 12800),
        );
        let index = ResidentIndex {
            header: model_io::ResidentIndexHeader {
                index_size: 0,
                resident_size: 0,
                entry_count: entries.len() as u64,
            },
            entries,
        };
        let err = DflashShape::derive(&index, 64, 200)
            .expect_err("should refuse mismatched codebook shape");
        match err {
            RealForwardError::Unsupported(msg) => {
                assert!(
                    msg.contains("codebook shape mismatch"),
                    "unexpected msg: {msg}"
                );
            }
            other => panic!("expected Unsupported error, got {other:?}"),
        }
    }
}
