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
    pub workload: Workload,
    /// Held-out prompts for the single `tslab confirm` pass. Never used for
    /// the main verdict, so a change tuned to the main prompts is caught.
    #[serde(default)]
    pub confirm_workload: Option<Workload>,
    pub primary_metric: Metric,
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
