//! The humanizer rewrite over the C ABI.
//!
//! The model is a RAW COMPLETION rewriter (see `foundation::humanizer`), so
//! this path cannot go through [`super::generate`]: that renders messages
//! through the checkpoint's chat template, which its publisher says breaks
//! this model. It shares everything AFTER the prompt exists (`run_turn`) and
//! owns everything before it: the byte-exact wrapper, the forced sampling
//! recipe, the output budget, and the audit.
//!
//! **THE WRAPPER AND THE RECIPE LIVE HERE, NOT IN THE HOST.** A host that
//! assembled the prompt itself would carry a second copy of bytes whose
//! single-character drift moves the model off its training distribution
//! (`PROMPT_FINGERPRINT` exists to catch exactly that). The options below
//! therefore expose a budget and a seed and nothing else, and refuse unknown
//! keys by name: a host passing `temperature` is told it cannot, rather than
//! being silently ignored into the CLI-shaped defaults the publisher
//! documents as wrong for this model.

use foundation::humanizer::{
    audit_rewrite, build_humanize_prompt, default_max_new, HUMANIZER_CONTEXT,
    HUMANIZER_REPETITION_PENALTY, HUMANIZER_TEMPERATURE, HUMANIZER_TOP_K, HUMANIZER_TOP_P,
    REUSE_FLAG_THRESHOLD,
};
use serde::{Deserialize, Serialize};
use tokenizer::ReasoningEffort;

use super::{run_turn, ToolOffer};
use crate::session::Session;
use crate::wire::{GenerateOptions, GenerateResult};

/// Arguments to `ts_humanize`. `{}` and a null pointer both mean "the
/// defaults".
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct HumanizeOptions {
    /// The output budget. Absent takes the publisher's rule (2.5x the
    /// draft's tokens, floored at 256 and capped at 2048). Either way it is
    /// then clamped to what the model's 8,192 window leaves after the
    /// wrapped prompt.
    pub max_new_tokens: Option<u32>,
    /// Fixes the sampler's stream. The recipe samples at temperature 1.0, so
    /// without a seed two runs of one draft differ, which is the point of a
    /// "resample" button.
    pub seed: Option<u64>,
}

/// The audit, as numbers and a ready-to-show list. Mirrors
/// `foundation::humanizer::RewriteAudit` field for field, plus the
/// pre-rendered warning lines so a host does not re-derive the thresholds.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AuditWire {
    pub dropped_numbers: Vec<String>,
    pub invented_numbers: Vec<String>,
    pub language_drift: bool,
    pub reuse_ratio: f64,
    /// `reuseRatio` is above the level the model was trained against.
    pub reuse_flagged: bool,
    /// Empty when the rewrite is clean.
    pub warnings: Vec<String>,
}

/// `ts_humanize`'s `result_json`: the ordinary turn result (`content` is the
/// rewrite) with the audit beside it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanizeResult {
    #[serde(flatten)]
    pub generate: GenerateResult,
    pub audit: AuditWire,
}

/// Resolves the output budget: the caller's number or the publisher's rule,
/// clamped to the room the wrapped prompt leaves in the model's window.
///
/// A free function so the arithmetic is pinnable without a session.
pub(crate) fn output_budget(
    asked: Option<u32>,
    draft_tokens: usize,
    prompt_tokens: usize,
) -> Result<u32, String> {
    if prompt_tokens >= HUMANIZER_CONTEXT {
        return Err(format!(
            "the draft is too long: its wrapped prompt is {prompt_tokens} tokens and the \
             humanizer's window is {HUMANIZER_CONTEXT}, with nothing left for the rewrite"
        ));
    }
    let wanted = asked.unwrap_or_else(|| default_max_new(draft_tokens));
    Ok(wanted.min((HUMANIZER_CONTEXT - prompt_tokens) as u32))
}

/// Rewrites `draft`, calling `emit(kind, text, a, b)` per event exactly as
/// [`super::generate`] does.
pub(crate) fn humanize(
    session: &Session,
    draft: &str,
    options: &HumanizeOptions,
    emit: impl FnMut(i32, &str, u32, u32),
) -> Result<HumanizeResult, String> {
    if draft.trim().is_empty() {
        return Err("the draft is empty".to_string());
    }
    // BOS on, no template: this mode's contract (see the module doc).
    let prompt_ids = session
        .tokenizer
        .encode(&build_humanize_prompt(draft), true);
    // The draft's real token count, not its whitespace-word count: a Chinese
    // draft is one "word" to a whitespace split, which would floor its
    // budget at 256 tokens and truncate a long rewrite.
    let draft_tokens = session.tokenizer.encode(draft.trim(), false).len();
    let budget = output_budget(options.max_new_tokens, draft_tokens, prompt_ids.len())?;

    let shaping = selection::ShapingConfig::new(
        HUMANIZER_TEMPERATURE,
        HUMANIZER_TOP_K,
        Some(HUMANIZER_TOP_P),
        HUMANIZER_REPETITION_PENALTY,
        options.seed,
    )
    .map_err(|e| e.to_string())?;

    // EOS-ONLY STOP: the publisher is explicit that `###` must never be a
    // stop string on this model, so both stop sets stay empty.
    let turn_options = GenerateOptions {
        max_new_tokens: budget,
        stop: Vec::new(),
        stop_tokens: Vec::new(),
        ..GenerateOptions::default()
    };
    let generate = run_turn(
        session,
        &prompt_ids,
        &[],
        &ToolOffer::none(),
        ReasoningEffort::Off,
        shaping,
        &turn_options,
        emit,
    )?;

    let audit = audit_rewrite(draft, &generate.content);
    let audit = AuditWire {
        warnings: audit.warnings(),
        reuse_flagged: audit.reuse_ratio > REUSE_FLAG_THRESHOLD,
        dropped_numbers: audit.dropped_numbers,
        invented_numbers: audit.invented_numbers,
        language_drift: audit.language_drift,
        reuse_ratio: audit.reuse_ratio,
    };
    Ok(HumanizeResult { generate, audit })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_follows_the_publishers_rule_when_unasked() {
        // 100 draft tokens -> 250, inside the 256 floor? No: 250 < 256.
        assert_eq!(output_budget(None, 100, 300).unwrap(), 256);
        assert_eq!(output_budget(None, 400, 700).unwrap(), 1000);
        // The 2048 cap.
        assert_eq!(output_budget(None, 5000, 5100).unwrap(), 2048);
    }

    #[test]
    fn an_explicit_budget_wins_but_the_window_still_clamps_it() {
        assert_eq!(output_budget(Some(64), 400, 700).unwrap(), 64);
        // 8192 - 8000 = 192 left, so a request for 1000 is clamped.
        assert_eq!(output_budget(Some(1000), 100, 8000).unwrap(), 192);
    }

    #[test]
    fn a_prompt_that_fills_the_window_is_refused_by_name() {
        let err = output_budget(None, 8000, HUMANIZER_CONTEXT).unwrap_err();
        assert!(err.contains("too long"), "{err}");
        assert!(output_budget(None, 8000, HUMANIZER_CONTEXT - 1).is_ok());
    }

    #[test]
    fn unknown_option_keys_are_rejected_not_ignored() {
        // The trap this guards: a host sending `temperature` and silently
        // getting the recipe anyway, believing its value was honored.
        let err = serde_json::from_str::<HumanizeOptions>(r#"{"temperature":0.2}"#).unwrap_err();
        assert!(err.to_string().contains("temperature"), "{err}");
        let ok: HumanizeOptions = serde_json::from_str(r#"{"maxNewTokens":128,"seed":7}"#).unwrap();
        assert_eq!((ok.max_new_tokens, ok.seed), (Some(128), Some(7)));
    }
}
