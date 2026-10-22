//! What one worker run executes, and under which arm.
//!
//! These types cross the orchestrator/worker boundary as JSON, so they are
//! portable plain data. An arm is ENVIRONMENT plus open parameters only:
//! every runtime knob is read from the environment at open or through a
//! process-wide `OnceLock`, and a fresh process per run makes both kinds
//! apply the same way.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One prompt the workload prefills.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSource {
    /// A frozen protocol case id from `crates/bench/prompts/real-generation-v1`
    /// (`short-explanation`, `medium-review`, `long-synthesis`), chat-templated
    /// as the CLI templates it.
    Protocol(String),
    /// A `--messages-file`-shaped JSON conversation, chat-templated.
    MessagesFile(PathBuf),
    /// A raw text file tokenized without a template.
    TextFile(PathBuf),
    /// A slice of the distribution-probe corpus
    /// (`turbospark_bench::quality_corpus`) of this many tokens. The k-th
    /// corpus prompt in a workload takes the probes' k-th offset, so a
    /// workload of the probe lengths prefills the probes' exact token ids.
    Corpus(usize),
}

/// Expert cache slot policy for the open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Slots {
    #[default]
    Auto,
    Fixed(usize),
}

/// Expert residency for the open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    #[default]
    Auto,
    Streamed,
    Mapped,
}

/// Parameters that are part of the OPEN rather than the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OpenParams {
    #[serde(default)]
    pub slots: Slots,
    #[serde(default)]
    pub residency: Residency,
}

/// One arm: a name, the `TURBOSPARK_*` variables it sets, and its open
/// parameters. Variables not named here are cleared in the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Arm {
    pub name: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub open: OpenParams,
}

/// What a worker runs once the model is open.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    pub prompts: Vec<PromptSource>,
    /// Greedy decode steps after each prefill. A FIXED count that ignores
    /// end-of-turn, so both arms of an A/B do the same decode work even when
    /// a numerics change moves the stop point. 0 measures prefill only.
    #[serde(default)]
    pub decode_tokens: usize,
    /// Prefill chunk size, or `None` for the sequential per-token loop.
    #[serde(default = "default_chunk")]
    pub prefill_chunk: Option<usize>,
    /// KV window at open. Must cover the longest prompt plus `decode_tokens`.
    #[serde(default = "default_context")]
    pub max_context: u32,
}

fn default_chunk() -> Option<usize> {
    Some(128)
}

fn default_context() -> u32 {
    4096
}

/// Which part of a run a metric reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Prompt tokens per prefill wall second.
    PrefillTokS,
    /// Decode tokens per decode wall second.
    DecodeTokS,
    /// Prefill wall seconds.
    PrefillS,
    /// Peak `phys_footprint`, bytes.
    PeakBytes,
}

impl Metric {
    pub fn direction(self) -> crate::stats::Direction {
        use crate::stats::Direction::*;
        match self {
            Metric::PrefillTokS | Metric::DecodeTokS => HigherIsBetter,
            Metric::PrefillS | Metric::PeakBytes => LowerIsBetter,
        }
    }

    /// Read this metric from a worker record (`worker::RunRecord` as JSON).
    pub fn read(self, record: &serde_json::Value) -> Option<f64> {
        let totals = &record["totals"];
        match self {
            Metric::PrefillTokS => totals["prefill_tok_s"].as_f64(),
            Metric::DecodeTokS => totals["decode_tok_s"].as_f64(),
            Metric::PrefillS => totals["prefill_s"].as_f64(),
            Metric::PeakBytes => record["peak_footprint_bytes"].as_f64(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workload_round_trips_with_defaults() {
        let w: Workload = serde_json::from_str(
            r#"{"prompts":[{"protocol":"long-synthesis"},{"text_file":"/x.txt"}]}"#,
        )
        .unwrap();
        assert_eq!(w.prefill_chunk, Some(128));
        assert_eq!(w.max_context, 4096);
        assert_eq!(w.decode_tokens, 0);
        let back: Workload = serde_json::from_str(&serde_json::to_string(&w).unwrap()).unwrap();
        assert_eq!(back, w);
    }

    #[test]
    fn arm_open_params_parse() {
        let a: Arm = serde_json::from_str(
            r#"{"name":"s32","open":{"slots":{"fixed":32},"residency":"streamed"},"env":{"TURBOSPARK_X":"1"}}"#,
        )
        .unwrap();
        assert_eq!(a.open.slots, Slots::Fixed(32));
        assert_eq!(a.open.residency, Residency::Streamed);
        assert_eq!(a.env["TURBOSPARK_X"], "1");
    }
}
