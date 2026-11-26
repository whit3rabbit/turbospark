//! Quality and memory gates. The bounds live here, in code an experiment may
//! not edit (`integrity.rs` refuses a diff that touches this crate), and a
//! spec can only TIGHTEN them.
//!
//! Sources for the numbers:
//! - Dense mean KL 1e-3 nats: `qwen38_prefill_arm_probe` (Steel's rounding
//!   against the per-token reference, `docs/SPEEDUP_PLAYBOOK.md`).
//! - MoE sentinel 1e-2 nats with top-1 agreement on at least half the
//!   prompts: the qwen MoE batched arm's bound (`docs/BATCHED_PREFILL.md`),
//!   which is a wiring sentinel, not a claim that MoE meets the dense bar.
//! - Bit-exact arms must reproduce the reference's logits bits and tokens.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateClass {
    /// Output bits identical to the reference arm.
    BitExact,
    /// Dense family distribution bound.
    DenseKl,
    /// Routed-MoE sentinel bound (near-tie expert swaps are expected).
    MoeKl,
    /// Audio transcription: the transcript text must not change, while the
    /// logits under it may (a reduction-order change in an encoder kernel).
    /// Audio workloads only.
    TextExact,
    /// Speculative verify-width changes only (LAB-03): outputs may part, but
    /// no more often than the same arm parts from itself, and only where the
    /// baseline's choice was a near tie.
    TokenDivergence,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityBounds {
    pub mean_kl: f64,
    pub min_top1_fraction: f64,
}

impl GateClass {
    pub fn default_bounds(self) -> QualityBounds {
        match self {
            GateClass::BitExact | GateClass::TextExact | GateClass::TokenDivergence => {
                QualityBounds {
                    mean_kl: 0.0,
                    min_top1_fraction: 1.0,
                }
            }
            GateClass::DenseKl => QualityBounds {
                mean_kl: 1e-3,
                min_top1_fraction: 0.0,
            },
            GateClass::MoeKl => QualityBounds {
                mean_kl: 1e-2,
                min_top1_fraction: 0.5,
            },
        }
    }

    /// The class a KL-bounded change on an install is held to: the routed
    /// MoE sentinel when the install routes experts, the dense bound
    /// otherwise. Decided by the architecture, not the family name: dense
    /// Llama and Mixtral share a family, and only one of them routes.
    pub fn kl_class_for(routes_experts: bool) -> GateClass {
        if routes_experts {
            GateClass::MoeKl
        } else {
            GateClass::DenseKl
        }
    }

    /// Refuse a spec whose declared class is looser than its install's: the
    /// MoE sentinel on a dense install would grade a dense change at a
    /// bound ten times wider than the one it is owed. A stricter class (the
    /// dense bound on a MoE install, or bit-exact anywhere) is the author's
    /// call and passes.
    pub fn check_declared(self, routes_experts: bool) -> Result<(), String> {
        let owed = GateClass::kl_class_for(routes_experts);
        if self == GateClass::MoeKl && owed == GateClass::DenseKl {
            return Err(
                "the spec declares moe_kl, but this install routes no experts: \
                 declare dense_kl (or bit_exact)"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Combine the class default with a spec's request: the stricter of each.
/// A request that tries to loosen is refused rather than clamped, so a spec
/// that asked for a looser bound fails loudly instead of running under a
/// bound its author did not expect.
pub fn effective_bounds(
    class: GateClass,
    requested: Option<QualityBounds>,
) -> Result<QualityBounds, String> {
    let base = class.default_bounds();
    let Some(r) = requested else {
        return Ok(base);
    };
    if r.mean_kl > base.mean_kl || r.min_top1_fraction < base.min_top1_fraction {
        return Err(format!(
            "spec bounds {r:?} are looser than the {class:?} defaults {base:?}; a spec may only tighten"
        ));
    }
    Ok(r)
}

/// Per-prompt comparison of one arm's final-position logits against the
/// reference arm's.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptComparison {
    pub prompt_tokens: usize,
    pub kl: f64,
    pub max_abs_dlogit: f64,
    pub top1_agree: bool,
    pub top5_overlap: usize,
    pub logits_bits_equal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityResult {
    pub class: GateClass,
    pub bounds: QualityBounds,
    pub mean_kl: f64,
    pub top1_fraction: f64,
    pub all_bits_equal: bool,
    pub tokens_equal: bool,
    pub passed: bool,
    pub reasons: Vec<String>,
    pub prompts: Vec<PromptComparison>,
    /// The prefill logits at each chunk end (LAB-02). `passed` includes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positions: Option<DecodeResult>,
    /// The teacher-forced decode gate, when the workload decodes. `passed`
    /// above already includes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeResult>,
}

pub fn judge_quality(
    class: GateClass,
    bounds: QualityBounds,
    prompts: Vec<PromptComparison>,
    tokens_equal: bool,
) -> QualityResult {
    let n = prompts.len().max(1) as f64;
    let mean_kl = prompts.iter().map(|p| p.kl).sum::<f64>() / n;
    let top1_fraction = prompts.iter().filter(|p| p.top1_agree).count() as f64 / n;
    let all_bits_equal = prompts.iter().all(|p| p.logits_bits_equal);
    let mut reasons = Vec::new();
    if prompts.is_empty() {
        reasons.push("no prompts compared".to_string());
    }
    match class {
        GateClass::BitExact => {
            if !all_bits_equal {
                reasons.push("logits bits differ from the reference".into());
            }
            if !tokens_equal {
                reasons.push("generated tokens differ from the reference".into());
            }
        }
        GateClass::TextExact => {
            if !tokens_equal {
                reasons.push("the transcript differs from the reference".into());
            }
        }
        // Judged by `judge_divergence` over a probed speculative run; reaching
        // here means no such run happened.
        GateClass::TokenDivergence => {
            reasons.push("token_divergence needs speculative arms".into());
        }
        GateClass::DenseKl | GateClass::MoeKl => {
            // NaN fails closed: a broken arm must not pass by comparing false.
            if mean_kl.is_nan() || mean_kl > bounds.mean_kl {
                reasons.push(format!(
                    "mean KL {mean_kl:.3e} over the {:.1e} bound",
                    bounds.mean_kl
                ));
            }
            if top1_fraction < bounds.min_top1_fraction {
                reasons.push(format!(
                    "top-1 agreement {top1_fraction:.2} under {:.2}",
                    bounds.min_top1_fraction
                ));
            }
        }
    }
    QualityResult {
        class,
        bounds,
        mean_kl,
        top1_fraction,
        all_bits_equal,
        tokens_equal,
        passed: reasons.is_empty(),
        reasons,
        prompts,
        positions: None,
        decode: None,
    }
}

impl QualityResult {
    /// Fold the prefill-position half in; like the decode half it can only
    /// fail a result.
    pub fn with_positions(mut self, mut positions: DecodeResult) -> Self {
        for r in &mut positions.reasons {
            *r = r.replacen("decode:", "prefill positions:", 1);
        }
        self.passed &= positions.passed;
        self.reasons.extend(positions.reasons.iter().cloned());
        self.positions = Some(positions);
        self
    }

    /// Fold the decode gate in: it can only fail a result, never rescue one.
    pub fn with_decode(mut self, decode: DecodeResult) -> Self {
        self.passed &= decode.passed;
        self.reasons.extend(decode.reasons.iter().cloned());
        self.decode = Some(decode);
        self
    }
}

/// Greedy steps a forced decode records per prompt (LAB-01).
pub const DECODE_TRACE_TOKENS: usize = 256;

/// The teacher-forced decode half of a quality gate: both arms decode the
/// reference's token ids, judged position by position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeResult {
    pub passed: bool,
    pub reasons: Vec<String>,
    pub overall: crate::kl_stats::Summary,
    /// Per prompt, with its token count as the label.
    pub prompts: Vec<(usize, crate::kl_stats::Summary)>,
    /// Per text class (LAB-02), when the caller named them.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub classes: std::collections::BTreeMap<String, crate::kl_stats::Summary>,
}

/// `positions` per prompt as `(prompt_tokens, positions)`. The mean KL bound
/// is the class's own, applied over every decoded position; a class that
/// bounds top-1 applies it to the positions that did not flip. A flip where
/// the reference was sure fails any KL class: the mean cannot hide it.
pub fn judge_decode(
    class: GateClass,
    bounds: QualityBounds,
    prompts: &[(usize, Vec<crate::kl_stats::Position>)],
) -> DecodeResult {
    use crate::kl_stats::{summarize, LARGE_MARGIN};
    let all: Vec<_> = prompts
        .iter()
        .flat_map(|(_, p)| p.iter().copied())
        .collect();
    let overall = summarize(&all);
    let mut reasons = Vec::new();
    if all.is_empty() {
        reasons.push("decode: no positions compared".into());
    }
    match class {
        GateClass::BitExact => {
            if !overall.all_bits_equal {
                reasons.push("decode: logits bits differ from the reference".into());
            }
        }
        GateClass::TextExact | GateClass::TokenDivergence => {
            reasons.push(format!("decode: {class:?} has no decode-trace bound"));
        }
        GateClass::DenseKl | GateClass::MoeKl => {
            // NaN fails closed.
            if overall.mean_kl.is_nan() || overall.mean_kl > bounds.mean_kl {
                reasons.push(format!(
                    "decode: mean KL {:.3e} over the {:.1e} bound",
                    overall.mean_kl, bounds.mean_kl
                ));
            }
            if 1.0 - overall.flip_rate < bounds.min_top1_fraction {
                reasons.push(format!(
                    "decode: top-1 agreement {:.3} under {:.2}",
                    1.0 - overall.flip_rate,
                    bounds.min_top1_fraction
                ));
            }
            if overall.large_margin_flips > 0 {
                reasons.push(format!(
                    "decode: {} top-1 flips where the reference led by {LARGE_MARGIN} logits or more",
                    overall.large_margin_flips
                ));
            }
        }
    }
    DecodeResult {
        passed: reasons.is_empty(),
        reasons,
        overall,
        prompts: prompts.iter().map(|(n, p)| (*n, summarize(p))).collect(),
        classes: Default::default(),
    }
}

/// Hold each text class to the same bound on its own: a drift on one kind of
/// text must not be averaged away by the others. `labels` name `prompts` in
/// order. Like every other half, it can only fail a result.
pub fn judge_text_classes(
    mut result: DecodeResult,
    class: GateClass,
    bounds: QualityBounds,
    labels: &[&str],
    prompts: &[(usize, Vec<crate::kl_stats::Position>)],
) -> DecodeResult {
    type Prompt = (usize, Vec<crate::kl_stats::Position>);
    let mut by_class: std::collections::BTreeMap<&str, Vec<Prompt>> = Default::default();
    for (label, prompt) in labels.iter().zip(prompts) {
        by_class.entry(label).or_default().push(prompt.clone());
    }
    let several = by_class.len() > 1;
    for (label, members) in by_class {
        let judged = judge_decode(class, bounds, &members);
        // One class alone is the overall judgment again.
        if several {
            result.passed &= judged.passed;
            result
                .reasons
                .extend(judged.reasons.iter().map(|r| format!("{label} text: {r}")));
        }
        result.classes.insert(label.to_string(), judged.overall);
    }
    result
}

/// How far above the base install's KL a candidate may sit and still count
/// as no worse (LAB-22).
pub const QUANT_KL_EPS: f64 = 0.02;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantVerdict {
    pub passed: bool,
    pub reasons: Vec<String>,
}

/// Install against install on one reference: no more bytes than the base and
/// no worse a KL from the reference. A bit plan that loses at equal bytes is
/// declined; one that buys quality with bytes is a different question.
pub fn judge_quant(base_kl: f64, base_bytes: u64, cand_kl: f64, cand_bytes: u64) -> QuantVerdict {
    let mut reasons = Vec::new();
    if cand_bytes > base_bytes {
        reasons.push(format!(
            "{cand_bytes} bytes against the base's {base_bytes}"
        ));
    }
    // NaN on either side fails closed.
    let limit = base_kl * (1.0 + QUANT_KL_EPS);
    if cand_kl.is_nan() || limit.is_nan() || cand_kl > limit {
        reasons.push(format!("KL {cand_kl:.3e} over the base's {base_kl:.3e}"));
    }
    QuantVerdict {
        passed: reasons.is_empty(),
        reasons,
    }
}

/// One prompt of a divergence gate: where the two arms first parted, and how
/// many logits the baseline's choice led by there.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Divergence {
    pub first: Option<usize>,
    pub baseline_margin: Option<f64>,
}

/// `floor` is the share of prompts on which the SAME arm gave two outputs
/// (`tslab noise`); with no measurement it is zero, so any divergence fails.
/// A divergence with no recorded margin, or a margin that was not a near tie,
/// fails whatever the floor.
pub fn judge_divergence(prompts: &[Divergence], floor: f64) -> (bool, Vec<String>) {
    use crate::kl_stats::SMALL_MARGIN;
    let mut reasons = Vec::new();
    if prompts.is_empty() {
        reasons.push("no prompts compared".to_string());
    }
    let parted: Vec<&Divergence> = prompts.iter().filter(|p| p.first.is_some()).collect();
    let rate = parted.len() as f64 / prompts.len().max(1) as f64;
    if rate > floor {
        reasons.push(format!(
            "outputs part on {rate:.2} of prompts; the same arm parts from itself on {floor:.2}"
        ));
    }
    for p in parted {
        if p.baseline_margin
            .is_none_or(|m| m.is_nan() || m >= SMALL_MARGIN)
        {
            reasons.push(format!(
                "divergence at token {} was not a near tie (margin {:?})",
                p.first.unwrap_or(0),
                p.baseline_margin
            ));
        }
    }
    (reasons.is_empty(), reasons)
}

/// Peak footprint growth allowed for the candidate arm: the larger of
/// 64 MiB, 2% of the reference peak, and `noise` (twice the peak spread a
/// null A-against-A run measured on this host; identical arms were seen
/// 177 MiB apart on a short Qwen 3.6 prompt). A spec may only lower it.
pub fn memory_allowance(reference_peak: u64, requested: Option<u64>, noise: u64) -> u64 {
    let default = (64u64 << 20).max(reference_peak / 50).max(noise);
    requested.map_or(default, |r| r.min(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_exact_judges_the_transcript_not_the_bytes() {
        let cmp = |bytes: bool, text: bool| PromptComparison {
            prompt_tokens: 0,
            kl: 0.0,
            max_abs_dlogit: 0.0,
            top1_agree: text,
            top5_overlap: 0,
            logits_bits_equal: bytes,
        };
        let b = GateClass::TextExact.default_bounds();
        assert!(judge_quality(GateClass::TextExact, b, vec![cmp(false, true)], true).passed);
        assert!(!judge_quality(GateClass::TextExact, b, vec![cmp(true, false)], false).passed);
        assert!(!judge_quality(GateClass::BitExact, b, vec![cmp(false, true)], true).passed);
    }

    #[test]
    fn a_dense_install_refuses_the_moe_class_only() {
        assert!(GateClass::MoeKl.check_declared(false).is_err());
        assert!(GateClass::MoeKl.check_declared(true).is_ok());
        assert!(GateClass::DenseKl.check_declared(true).is_ok());
        assert!(GateClass::BitExact.check_declared(false).is_ok());
        assert_eq!(GateClass::kl_class_for(true), GateClass::MoeKl);
    }

    fn cmp(kl: f64, top1: bool, bits: bool) -> PromptComparison {
        PromptComparison {
            prompt_tokens: 64,
            kl,
            max_abs_dlogit: 0.0,
            top1_agree: top1,
            top5_overlap: 5,
            logits_bits_equal: bits,
        }
    }

    #[test]
    fn specs_may_tighten_but_not_loosen() {
        let tight = QualityBounds {
            mean_kl: 1e-4,
            min_top1_fraction: 0.0,
        };
        assert_eq!(
            effective_bounds(GateClass::DenseKl, Some(tight)).unwrap(),
            tight
        );
        let loose = QualityBounds {
            mean_kl: 1e-2,
            min_top1_fraction: 0.0,
        };
        assert!(effective_bounds(GateClass::DenseKl, Some(loose)).is_err());
        let fewer_top1 = QualityBounds {
            mean_kl: 1e-2,
            min_top1_fraction: 0.3,
        };
        assert!(effective_bounds(GateClass::MoeKl, Some(fewer_top1)).is_err());
    }

    #[test]
    fn kl_gate_enforces_both_terms() {
        let b = GateClass::MoeKl.default_bounds();
        assert!(judge_quality(GateClass::MoeKl, b, vec![cmp(5e-3, true, false); 4], false).passed);
        assert!(!judge_quality(GateClass::MoeKl, b, vec![cmp(2e-2, true, false); 4], false).passed);
        let mostly_wrong = vec![
            cmp(1e-3, false, false),
            cmp(1e-3, false, false),
            cmp(1e-3, true, false),
        ];
        assert!(!judge_quality(GateClass::MoeKl, b, mostly_wrong, false).passed);
    }

    #[test]
    fn bit_exact_needs_bits_and_tokens() {
        let b = GateClass::BitExact.default_bounds();
        assert!(judge_quality(GateClass::BitExact, b, vec![cmp(0.0, true, true)], true).passed);
        assert!(!judge_quality(GateClass::BitExact, b, vec![cmp(0.0, true, false)], true).passed);
        assert!(!judge_quality(GateClass::BitExact, b, vec![cmp(0.0, true, true)], false).passed);
    }

    #[test]
    fn the_decode_gate_catches_what_the_prefill_gate_cannot() {
        use crate::kl_stats::position;
        let same = position(&[3.0, 1.0, 0.0], &[3.0, 1.0, 0.0], 0).unwrap();
        // One sure flip among many identical positions: the mean KL stays
        // far under the bound, the flip must still fail.
        let mut tiny = vec![same; 20_000];
        tiny.push(position(&[3.0, 0.0, 0.0], &[0.0, 3.0, 0.0], 0).unwrap());
        let b = GateClass::DenseKl.default_bounds();
        let d = judge_decode(GateClass::DenseKl, b, &[(64, tiny)]);
        assert!(d.overall.mean_kl < b.mean_kl, "{}", d.overall.mean_kl);
        assert!(!d.passed);
        assert!(judge_decode(GateClass::DenseKl, b, &[(64, vec![same; 4])]).passed);
        assert!(!judge_decode(GateClass::DenseKl, b, &[]).passed);
        // A passing prefill gate does not rescue a failing decode.
        let prefill = judge_quality(GateClass::DenseKl, b, vec![cmp(0.0, true, true)], true);
        assert!(prefill.passed && !prefill.with_decode(d).passed);
    }

    #[test]
    fn decode_bit_exact_needs_bits_and_nan_fails_closed() {
        use crate::kl_stats::position;
        let b = GateClass::BitExact.default_bounds();
        let near = position(&[3.0, 1.0], &[3.0, 1.0000001], 0).unwrap();
        assert!(!judge_decode(GateClass::BitExact, b, &[(1, vec![near])]).passed);
        let broken = position(&[3.0, 1.0], &[f64::NAN, 1.0], 0).unwrap();
        let b = GateClass::MoeKl.default_bounds();
        assert!(!judge_decode(GateClass::MoeKl, b, &[(1, vec![broken])]).passed);
    }

    #[test]
    fn a_quant_candidate_must_not_buy_quality_with_bytes_or_lose_it() {
        assert!(judge_quant(0.10, 100, 0.10, 100).passed);
        assert!(judge_quant(0.10, 100, 0.101, 90).passed);
        assert!(!judge_quant(0.10, 100, 0.11, 90).passed);
        assert!(!judge_quant(0.10, 100, 0.05, 101).passed);
        assert!(!judge_quant(0.10, 100, f64::NAN, 90).passed);
        assert!(!judge_quant(f64::NAN, 100, 0.05, 90).passed);
    }

    #[test]
    fn a_drift_on_one_kind_of_text_is_not_averaged_away() {
        use crate::kl_stats::position;
        let same = position(&[3.0, 1.0, 0.0], &[3.0, 1.0, 0.0], 0).unwrap();
        let off = position(&[3.0, 1.0, 0.0], &[3.0, 1.2, 0.0], 0).unwrap();
        let b = GateClass::DenseKl.default_bounds();
        // Many clean document positions, a few drifted chat positions: the
        // pooled mean passes and the chat class alone does not.
        assert!(
            off.kl > b.mean_kl && off.kl * 20.0 / 20_020.0 < b.mean_kl,
            "{}",
            off.kl
        );
        let prompts = vec![(64, vec![same; 20_000]), (32, vec![off; 20])];
        let pooled = judge_decode(GateClass::DenseKl, b, &prompts);
        assert!(pooled.passed, "{:?}", pooled.reasons);
        let split = judge_text_classes(
            pooled.clone(),
            GateClass::DenseKl,
            b,
            &["documents", "chat"],
            &prompts,
        );
        assert!(!split.passed);
        assert!(
            split.reasons.iter().any(|r| r.starts_with("chat text:")),
            "{:?}",
            split.reasons
        );
        assert_eq!(split.classes.len(), 2);
        // One class: recorded, not judged twice.
        let one = judge_text_classes(pooled, GateClass::DenseKl, b, &["documents"; 2], &prompts);
        assert!(one.passed && one.reasons.is_empty() && one.classes.len() == 1);
    }

    #[test]
    fn a_divergence_passes_only_as_a_near_tie_within_the_floor() {
        let same = Divergence {
            first: None,
            baseline_margin: None,
        };
        let tie = Divergence {
            first: Some(3),
            baseline_margin: Some(0.1),
        };
        let sure = Divergence {
            first: Some(3),
            baseline_margin: Some(4.0),
        };
        let unknown = Divergence {
            first: Some(3),
            baseline_margin: None,
        };
        assert!(judge_divergence(&[same; 4], 0.0).0);
        // No measured floor: even a tie fails.
        assert!(!judge_divergence(&[same, tie], 0.0).0);
        assert!(judge_divergence(&[same, tie], 0.5).0);
        assert!(!judge_divergence(&[same, sure], 1.0).0);
        assert!(!judge_divergence(&[same, unknown], 1.0).0);
        assert!(!judge_divergence(&[], 1.0).0);
    }

    #[test]
    fn nan_kl_fails_closed() {
        let b = GateClass::DenseKl.default_bounds();
        assert!(
            !judge_quality(
                GateClass::DenseKl,
                b,
                vec![cmp(f64::NAN, true, false)],
                true
            )
            .passed
        );
    }

    #[test]
    fn memory_allowance_floor_and_tightening() {
        assert_eq!(memory_allowance(1 << 30, None, 0), 64 << 20);
        assert_eq!(memory_allowance(10 << 30, None, 0), (10u64 << 30) / 50);
        assert_eq!(memory_allowance(1 << 30, Some(1 << 20), 0), 1 << 20);
        assert_eq!(memory_allowance(1 << 30, Some(1 << 40), 0), 64 << 20);
        assert_eq!(memory_allowance(1 << 30, None, 400 << 20), 400 << 20);
    }
}
