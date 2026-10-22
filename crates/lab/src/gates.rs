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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityBounds {
    pub mean_kl: f64,
    pub min_top1_fraction: f64,
}

impl GateClass {
    pub fn default_bounds(self) -> QualityBounds {
        match self {
            GateClass::BitExact => QualityBounds {
                mean_kl: 0.0,
                min_top1_fraction: 1.0,
            },
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

    /// The class a family's KL-bounded changes are held to.
    pub fn kl_class_for_family(family: &str) -> GateClass {
        match family {
            "qwen36" | "gemma4" | "qwen3moe" | "gptOss" | "llama" | "deepseek2" | "minimax_m2"
            | "qwen4exp" | "flint" => GateClass::MoeKl,
            _ => GateClass::DenseKl,
        }
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
    }
}

/// Peak footprint growth allowed for the candidate arm: the larger of 64 MiB
/// and 2% of the reference peak, unless a spec asks for less.
pub fn memory_allowance(reference_peak: u64, requested: Option<u64>) -> u64 {
    let default = (64u64 << 20).max(reference_peak / 50);
    requested.map_or(default, |r| r.min(default))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(memory_allowance(1 << 30, None), 64 << 20);
        assert_eq!(memory_allowance(10 << 30, None), (10u64 << 30) / 50);
        assert_eq!(memory_allowance(1 << 30, Some(1 << 20)), 1 << 20);
        assert_eq!(memory_allowance(1 << 30, Some(1 << 40)), 64 << 20);
    }
}
