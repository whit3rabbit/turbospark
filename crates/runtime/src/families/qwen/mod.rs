//! The real-checkpoint Qwen decode flow for [`RealForwardRunner`], serving
//! BOTH the `qwen36` family and (ROADMAP's 1-bit entry) the dense `qwen3_5`
//! one, which differ in their FFN half and in nothing else.

mod attn;
mod batched;
mod batched_layers;
mod batched_scratch;
mod dense;
mod dflash;
mod dflash_draft;
mod dflash_state;
mod moe;
mod moe_batch;
mod mtp;
mod mtp_dump;
mod mtp_state;
mod produce;
mod state;

pub(crate) use attn::{encode_full_attention_block, QkNormConvention};
pub(crate) use batched_scratch::BatchedScratch;
pub(crate) use dflash::DflashState;
pub use dflash_state::{install_has_dflash, DflashDraftPolicy, DFLASH_BLOCK, DFLASH_SERVING_BLOCK};
pub(crate) use mtp_state::MtpState;
pub use mtp_state::{install_has_mtp_head, MtpDraftPolicy};
pub(crate) use state::RealQwenState;

/// Both drafters a speculative caller can ask for at open, as ONE argument
/// so the opener signatures do not grow a parameter per future drafter.
///
/// At most one is ever non-`Off` in practice -- a round drafts with one
/// model -- but nothing here enforces that, because the interesting
/// failure (both asked for) is better refused where the drafter is chosen,
/// with the caller's names in the message.
#[derive(Clone, Copy, Debug)]
pub struct DraftPolicies {
    pub mtp: MtpDraftPolicy,
    pub dflash: DflashDraftPolicy,
}

impl DraftPolicies {
    pub fn off() -> Self {
        Self {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Off,
        }
    }

    /// Both from the environment, for `open_with_slot_policy`.
    pub fn from_env() -> Self {
        Self {
            mtp: MtpDraftPolicy::from_env(),
            dflash: DflashDraftPolicy::from_env(),
        }
    }

    /// Only the MTP head asked for; what every existing MTP probe passes.
    pub fn mtp(policy: MtpDraftPolicy) -> Self {
        Self {
            mtp: policy,
            dflash: DflashDraftPolicy::Off,
        }
    }
}

pub(crate) const RMS_EPS: f32 = 1e-6;

use crate::real_forward::RealForwardRunner;

/// The trunk's tensor-name prefix.
pub(crate) const TRUNK_PREFIX: &str = "language_model.model";

/// The multi-token-prediction head's, for `prefixed_layer_tensor`.
///
/// **No trailing dot**, unlike `repack`'s `classify::MTP_PREFIX`. The two
/// answer different questions and are deliberately not shared: that one
/// MATCHES a name (`name.starts_with("mtp.")`) and this one BUILDS one
/// (`format!("{prefix}.layers.{layer}.{suffix}")`), so a single constant
/// would be wrong at one of the two sites.
pub(crate) const MTP_PREFIX: &str = "mtp";

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    prefixed_layer_tensor(TRUNK_PREFIX, layer, suffix)
}

/// [`layer_tensor`] under an explicit prefix.
///
/// The ONLY reason this exists is the multi-token-prediction head
/// (`docs/MTP_SPECULATIVE.md`), whose single block is shape-identical to a
/// trunk full-attention layer field for field and so can run the SAME
/// encoders under `mtp.layers.0.*` -- rather than a second copy of them,
/// which is what Gotcha 11 is about the cost of.
///
/// It is a STRING change and not a flow change: every caller passing
/// [`TRUNK_PREFIX`] resolves the byte-identical name it resolved before, and
/// `qwen38_quality_gate` was re-run to say so rather than to hope so.
pub(crate) fn prefixed_layer_tensor(prefix: &str, layer: usize, suffix: &str) -> String {
    format!("{prefix}.layers.{layer}.{suffix}")
}

impl RealForwardRunner {
    pub fn gdn_state_abs_max(&mut self, layer: usize) -> Option<f32> {
        let qwen = self.real_qwen.as_ref()?;
        if !qwen.gdn.is_linear(layer) {
            return None;
        }
        let buf = qwen.gdn.state_buffer(layer);
        let len = (buf.length() as usize) / 4;
        let contents = gpu::read_f32_buffer(buf, len);
        let mut max = 0.0f32;
        for &v in &contents {
            max = max.max(v.abs());
        }
        Some(max)
    }
}
