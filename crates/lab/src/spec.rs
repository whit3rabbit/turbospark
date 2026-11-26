//! The experiment spec: one JSON file per hypothesis.
//!
//! The autoresearch loop writes one of these per iteration and hands it to
//! `tslab run-experiment`. tslab freezes a copy (with its SHA-256) in the
//! experiment directory before measuring, so the verdict in the ledger names
//! the exact question it answered.

use serde::{Deserialize, Serialize};

use crate::gates::{effective_bounds, GateClass, QualityBounds};
use crate::paths::validate_name;
use crate::workload::{Arm, Metric, Workload};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentSpec {
    /// Experiment directory name under the lab dir.
    pub name: String,
    /// One or two sentences: what should get faster, and why.
    pub hypothesis: String,
    /// Install path or catalog alias.
    pub model: String,
    /// The switch under test. Required by `run-experiment`: a change that is
    /// not one env var away from the old path cannot be killed or A/B'd.
    #[serde(default)]
    pub knob_env: Option<String>,
    pub baseline: Arm,
    pub candidate: Arm,
    /// Differences between the arms beyond `knob_env`, each declared with a
    /// reason (LAB-07). `field` is an environment variable name, `open`,
    /// `speculation`, `contention`, `ballast_gib` or `cold`. An undeclared
    /// difference refuses the spec: a second flipped switch confounds the A/B.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub also_changes: Vec<AlsoChange>,
    pub workload: Workload,
    /// Held-out prompts for the single `tslab confirm` pass. Never used for
    /// the main verdict, so a change tuned to the main prompts is caught.
    #[serde(default)]
    pub confirm_workload: Option<Workload>,
    pub primary_metric: Metric,
    /// A second metric the candidate may not lose more than `max_regression`
    /// on (LAB-33), e.g. a decode win must not tax prefill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub secondary_guard: Option<SecondaryGuard>,
    /// Ask "not worse than this fraction" instead of only "better" (LAB-05),
    /// for default-on, cleanup and safety changes. A KEEP then needs the
    /// one-sided bound to clear the limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noninferiority_limit: Option<f64>,
    #[serde(default = "default_min_effect")]
    pub min_effect: f64,
    /// What switching the knob may do to output.
    pub exactness: GateClass,
    /// Stricter-than-default quality bounds, if any.
    #[serde(default)]
    pub quality_bounds: Option<QualityBounds>,
    /// Stricter-than-default peak growth allowance, bytes.
    #[serde(default)]
    pub memory_allowance_bytes: Option<u64>,
    /// `cargo test` argument strings that must pass before measuring, e.g.
    /// `-p turbospark-gpu --test my_kernel_parity`.
    #[serde(default)]
    pub required_tests: Vec<String>,
    #[serde(default = "default_budget")]
    pub time_budget_min: u32,
    #[serde(default = "default_min_pairs")]
    pub min_pairs: usize,
    #[serde(default = "default_max_pairs")]
    pub max_pairs: usize,
}

/// The widest regression a secondary guard may allow.
pub const MAX_GUARD_REGRESSION: f64 = 0.02;

/// The second metric of a spec (LAB-33).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecondaryGuard {
    pub metric: Metric,
    #[serde(default = "default_guard_regression")]
    pub max_regression: f64,
}

fn default_guard_regression() -> f64 {
    MAX_GUARD_REGRESSION
}

/// One declared extra difference between a spec's arms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlsoChange {
    pub field: String,
    pub reason: String,
}

/// Every way two arms differ: environment variable names, then the non-env
/// arm fields. `expect_engaged` is left out on purpose: it asserts that a
/// lane ran and changes nothing the worker does.
pub fn arm_differences(a: &Arm, b: &Arm) -> Vec<String> {
    let keys: std::collections::BTreeSet<&String> = a.env.keys().chain(b.env.keys()).collect();
    let mut out: Vec<String> = keys
        .into_iter()
        .filter(|k| a.env.get(*k) != b.env.get(*k))
        .cloned()
        .collect();
    for (name, differs) in [
        ("open", a.open != b.open),
        ("speculation", a.speculation != b.speculation),
        ("contention", a.contention != b.contention),
        ("ballast_gib", a.ballast_gib != b.ballast_gib),
        ("cold", a.cold != b.cold),
    ] {
        if differs {
            out.push(name.to_string());
        }
    }
    out
}

fn default_min_effect() -> f64 {
    crate::stats::MIN_EFFECT_FLOOR
}
fn default_budget() -> u32 {
    30
}
fn default_min_pairs() -> usize {
    crate::stats::MIN_PAIRS
}
fn default_max_pairs() -> usize {
    6
}

impl ExperimentSpec {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let spec: Self = serde_json::from_str(text).map_err(|e| format!("spec: {e}"))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Structural checks. `require_knob` is set by `run-experiment`.
    pub fn validate(&self) -> Result<(), String> {
        validate_name(&self.name)?;
        validate_name(&self.baseline.name)?;
        validate_name(&self.candidate.name)?;
        if self.baseline.name == self.candidate.name {
            return Err("baseline and candidate arms need different names".into());
        }
        if self.hypothesis.trim().is_empty() {
            return Err("hypothesis is empty".into());
        }
        for arm in [&self.baseline, &self.candidate] {
            for key in arm.env.keys() {
                if !key.starts_with("TURBOSPARK_") {
                    return Err(format!(
                        "arm {} sets {key}: arms may only set TURBOSPARK_* variables",
                        arm.name
                    ));
                }
            }
        }
        if let Some(knob) = &self.knob_env {
            if !knob.starts_with("TURBOSPARK_") {
                return Err(format!("knob_env {knob} must be a TURBOSPARK_* variable"));
            }
            if self.baseline.env.get(knob) == self.candidate.env.get(knob) {
                return Err(format!(
                    "baseline and candidate agree on {knob}; the arms must differ in the knob under test"
                ));
            }
            for field in arm_differences(&self.baseline, &self.candidate) {
                if field != *knob && !self.also_changes.iter().any(|c| c.field == field) {
                    return Err(format!(
                        "the arms also differ in {field}: change one thing at a time, or declare it in also_changes with a reason"
                    ));
                }
            }
        }
        for c in &self.also_changes {
            if c.reason.trim().is_empty() {
                return Err(format!("also_changes {}: the reason is empty", c.field));
            }
            if !arm_differences(&self.baseline, &self.candidate).contains(&c.field) {
                return Err(format!(
                    "also_changes {}: the arms do not differ there",
                    c.field
                ));
            }
        }
        if let Some(g) = &self.secondary_guard {
            if !(g.max_regression.is_finite()
                && (0.0..=MAX_GUARD_REGRESSION).contains(&g.max_regression))
            {
                return Err(format!(
                    "secondary_guard.max_regression must be within [0, {MAX_GUARD_REGRESSION}]"
                ));
            }
            if g.metric == self.primary_metric {
                return Err("secondary_guard.metric repeats primary_metric".into());
            }
        }
        if let Some(limit) = self.noninferiority_limit {
            if !(limit.is_finite()
                && (0.0..=crate::stats::MAX_NONINFERIORITY_LIMIT).contains(&limit))
            {
                return Err(format!(
                    "noninferiority_limit must be within [0, {}]",
                    crate::stats::MAX_NONINFERIORITY_LIMIT
                ));
            }
            if self.min_pairs < crate::stats::NONINFERIORITY_PAIRS {
                return Err(format!(
                    "noninferiority_limit needs a pre-declared min_pairs of at least {}",
                    crate::stats::NONINFERIORITY_PAIRS
                ));
            }
        }
        for w in std::iter::once(&self.workload).chain(self.confirm_workload.iter()) {
            if w.prompts.is_empty() {
                return Err("a workload has no prompts".into());
            }
        }
        if self.confirm_workload.as_ref() == Some(&self.workload) {
            return Err(
                "confirm_workload must differ from workload (it is the held-out set)".into(),
            );
        }
        if !(self.min_effect.is_finite() && self.min_effect >= 0.0) {
            return Err("min_effect must be a finite non-negative fraction".into());
        }
        if self.min_pairs < crate::stats::MIN_PAIRS {
            return Err(format!(
                "min_pairs {} is under the evaluator's {}",
                self.min_pairs,
                crate::stats::MIN_PAIRS
            ));
        }
        if self.max_pairs < self.min_pairs || self.max_pairs > 20 {
            return Err("max_pairs must be within [min_pairs, 20]".into());
        }
        if self.time_budget_min == 0 || self.time_budget_min > 240 {
            return Err("time_budget_min must be within [1, 240]".into());
        }
        if self.exactness != GateClass::BitExact {
            effective_bounds(self.exactness, self.quality_bounds)?;
        }
        if self.exactness == GateClass::TokenDivergence
            && self.baseline.effective_speculation(&self.workload)
                == self.candidate.effective_speculation(&self.workload)
        {
            return Err(
                "token_divergence is for a change of verify width: the arms' speculation must differ"
                    .into(),
            );
        }
        let audio = self.workload.audio.is_some();
        if self.exactness == GateClass::TextExact && !audio {
            return Err("text_exact gates an audio transcript; a text spec declares bit_exact, dense_kl or moe_kl".into());
        }
        // A waveform has no transcript to hold equal: a speak or music job
        // is judged on its bytes.
        if self.exactness == GateClass::TextExact
            && self
                .workload
                .audio
                .as_ref()
                .is_some_and(|j| j.task() != "transcribe")
        {
            return Err(
                "text_exact gates a transcript; a speak or music job declares bit_exact".into(),
            );
        }
        if audio
            && !matches!(
                self.primary_metric,
                Metric::AudioRunS | Metric::Rtf | Metric::PeakBytes
            )
        {
            return Err("an audio spec's primary_metric is audio_run_s, rtf or peak_bytes".into());
        }
        Ok(())
    }

    /// `run-experiment`'s extra demand: the knob under test is named.
    pub fn require_knob(&self) -> Result<&str, String> {
        self.knob_env.as_deref().ok_or_else(|| {
            "run-experiment needs knob_env: the new path must be one env var away".into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> serde_json::Value {
        serde_json::json!({
            "name": "fused-gate",
            "hypothesis": "Fusing the gate into the GEMV saves a dispatch per layer.",
            "model": "/m.gturbo",
            "knob_env": "TURBOSPARK_FOO",
            "baseline": {"name": "off"},
            "candidate": {"name": "on", "env": {"TURBOSPARK_FOO": "1"}},
            "workload": {"prompts": [{"protocol": "long-synthesis"}]},
            "confirm_workload": {"prompts": [{"protocol": "medium-review"}]},
            "primary_metric": "prefill_tok_s",
            "exactness": "dense_kl"
        })
    }

    #[test]
    fn a_good_spec_parses() {
        let spec = ExperimentSpec::from_json(&base().to_string()).unwrap();
        assert_eq!(spec.require_knob().unwrap(), "TURBOSPARK_FOO");
        assert_eq!(spec.max_pairs, 6);
    }

    #[test]
    fn a_second_difference_between_arms_must_be_declared() {
        let mut v = base();
        v["candidate"]["env"]["TURBOSPARK_BAR"] = serde_json::json!("1");
        let err = ExperimentSpec::from_json(&v.to_string()).unwrap_err();
        assert!(err.contains("TURBOSPARK_BAR"), "{err}");
        v["also_changes"] =
            serde_json::json!([{"field": "TURBOSPARK_BAR", "reason": "the lane needs it"}]);
        assert!(ExperimentSpec::from_json(&v.to_string()).is_ok());
        v["also_changes"][0]["reason"] = serde_json::json!(" ");
        assert!(ExperimentSpec::from_json(&v.to_string()).is_err());
        // Non-environment differences count too, and a declaration of a
        // difference that is not there is refused.
        let mut v = base();
        v["candidate"]["speculation"] = serde_json::json!({"block": 2});
        let err = ExperimentSpec::from_json(&v.to_string()).unwrap_err();
        assert!(err.contains("speculation"), "{err}");
        let mut v = base();
        v["also_changes"] = serde_json::json!([{"field": "open", "reason": "x"}]);
        assert!(ExperimentSpec::from_json(&v.to_string()).is_err());
        // A lane assertion is not a change in behavior.
        let mut v = base();
        v["candidate"]["expect_engaged"] = serde_json::json!(["foo_lane"]);
        assert!(ExperimentSpec::from_json(&v.to_string()).is_ok());
    }

    #[test]
    fn guards_and_noninferiority_limits_are_bounded() {
        let mut v = base();
        v["secondary_guard"] = serde_json::json!({"metric": "decode_tok_s"});
        assert!(ExperimentSpec::from_json(&v.to_string()).is_ok());
        v["secondary_guard"]["max_regression"] = serde_json::json!(0.1);
        assert!(ExperimentSpec::from_json(&v.to_string()).is_err());
        v["secondary_guard"] = serde_json::json!({"metric": "prefill_tok_s"});
        assert!(
            ExperimentSpec::from_json(&v.to_string()).is_err(),
            "repeats primary"
        );
        let mut v = base();
        v["noninferiority_limit"] = serde_json::json!(0.02);
        assert!(
            ExperimentSpec::from_json(&v.to_string()).is_err(),
            "3 pairs"
        );
        v["min_pairs"] = serde_json::json!(10);
        v["max_pairs"] = serde_json::json!(14);
        assert!(ExperimentSpec::from_json(&v.to_string()).is_ok());
        v["noninferiority_limit"] = serde_json::json!(0.2);
        assert!(ExperimentSpec::from_json(&v.to_string()).is_err());
    }

    #[test]
    fn refusals() {
        let cases: Vec<(&str, serde_json::Value)> = vec![
            ("/candidate/env/PATH", serde_json::json!("x")),
            ("/candidate/env/TURBOSPARK_FOO", serde_json::Value::Null),
            ("/min_pairs", serde_json::json!(2)),
            (
                "/quality_bounds",
                serde_json::json!({"mean_kl": 1.0, "min_top1_fraction": 0.0}),
            ),
            (
                "/confirm_workload",
                serde_json::json!({"prompts": [{"protocol": "long-synthesis"}]}),
            ),
            ("/unknown_field", serde_json::json!(1)),
        ];
        for (pointer, value) in cases {
            let mut v = base();
            let (parent, key) = pointer.rsplit_once('/').unwrap();
            let target = if parent.is_empty() {
                &mut v
            } else {
                v.pointer_mut(parent).unwrap()
            };
            if value.is_null() {
                target.as_object_mut().unwrap().remove(key);
            } else {
                target[key] = value;
            }
            assert!(
                ExperimentSpec::from_json(&v.to_string()).is_err(),
                "{pointer} was accepted"
            );
        }
    }
}
