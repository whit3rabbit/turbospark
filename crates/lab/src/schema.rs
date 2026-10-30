//! The output contract: one schema version for the ledger, the `--json`
//! envelope, kernel bench results and the exported series.
//!
//! Readers (the export, the HTML report, an agent, the claude.ai dashboard)
//! key on `schema` and `kind`. A ledger line without `schema` predates this
//! module and reads as version 0; `series.rs` adapts both. The JSON Schema
//! for all of it is `schema/tslab-1.schema.json`, printed by `tslab schema`.
//!
//! Changing a field's meaning is a new version, not an edit: dashboards and
//! old ledgers outlive any one build.

use serde::Serialize;
use serde_json::Value;

/// Stamped on every ledger line, envelope and export written by this build.
pub const SCHEMA: &str = "tslab/1";

/// The JSON Schema document, `tslab schema` prints it.
pub const SCHEMA_DOC: &str = include_str!("../schema/tslab-1.schema.json");

/// Every ledger `kind` this crate appends, with what its payload holds.
/// `tests/schema_contract.rs` holds the list to the source: an `append`
/// with a kind missing here, or from the schema document, fails there.
pub const KINDS: &[(&str, &str)] = &[
    (
        "run",
        "one worker process: arm, host samples, eligibility, the worker record",
    ),
    (
        "profile_summary",
        "per-arm medians over a profile's clean runs",
    ),
    (
        "null_calibration",
        "an arm against itself: the effect floor and peak noise",
    ),
    (
        "verdict",
        "a paired A/B: verdict report, memory check, pair run ids",
    ),
    (
        "sweep",
        "several candidates against one baseline, each a paired A/B: per-candidate verdicts and the best WIN",
    ),
    (
        "kernel_curve",
        "one kernel bench case: a BenchResult under `result`",
    ),
    (
        "hash_diff",
        "output-bit hashes of two kernel experiments compared",
    ),
    ("ane_plan", "Core ML compute-plan placement of one program"),
    (
        "expert_census",
        "one router capture run reduced: expert reuse within batches, union against slot counts, sub-batch replay",
    ),
    (
        "experiment_start",
        "run-experiment began: spec, base, integrity result",
    ),
    ("gate", "quality gate result for one spec"),
    ("experiment", "run-experiment's decision"),
    ("confirm", "confirm's decision on the held-out workload"),
    (
        "tree_replay",
        "tree against chain acceptance at equal verify rows",
    ),
    (
        "chain_capture",
        "per-position chained acceptance for depth replay",
    ),
    (
        "depth_replay",
        "controller settings priced over captured chains",
    ),
    (
        "hybrid_replay",
        "prompt lookup over the MTP chain against the head alone, per policy and case",
    ),
    (
        "round_cost",
        "measured speculative round ms per (install, width, KV bucket)",
    ),
    (
        "dispatch_probe",
        "chained-dispatch price fitted from probe arms",
    ),
    (
        "standin",
        "in-situ block cost from stand-in arms (unquotable)",
    ),
    (
        "projection",
        "a kernel mode priced from curves, counts and a measured base, before a model run",
    ),
    (
        "calibration",
        "a projection joined to the A/B verdict that followed: predicted, measured, miss",
    ),
    ("trace", "Metal System Trace tables (unquotable)"),
    ("workflow_step", "one step of a workflow run"),
    ("workflow_run", "a workflow run's summary"),
];

pub fn is_known_kind(kind: &str) -> bool {
    KINDS.iter().any(|(k, _)| *k == kind)
}

/// What a command hands back to `cli::main`: its exit code and the payload
/// the `--json` envelope carries.
#[derive(Debug, Clone)]
pub struct CmdOutput {
    pub code: i32,
    pub result: Value,
    pub warnings: Vec<String>,
}

impl CmdOutput {
    pub fn ok(result: Value) -> Self {
        Self {
            code: 0,
            result,
            warnings: Vec::new(),
        }
    }

    pub fn with_code(code: i32, result: Value) -> Self {
        Self {
            code,
            result,
            warnings: Vec::new(),
        }
    }

    pub fn warn(mut self, w: impl Into<String>) -> Self {
        self.warnings.push(w.into());
        self
    }
}

/// A ledger line this process appended, as the envelope lists it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RecordRef {
    pub id: String,
    pub kind: String,
    pub ledger: String,
}

/// The one object `--json` prints to stdout.
#[derive(Debug, Clone, Serialize)]
pub struct Envelope {
    pub schema: &'static str,
    pub command: String,
    pub ok: bool,
    pub exit_code: i32,
    /// run-experiment and confirm: keep, discard, inconclusive, ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    pub records: Vec<RecordRef>,
    pub result: Value,
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Envelope {
    pub fn new(
        command: &str,
        outcome: &Result<CmdOutput, String>,
        records: Vec<RecordRef>,
    ) -> Self {
        match outcome {
            Ok(out) => Self {
                schema: SCHEMA,
                command: command.to_string(),
                ok: true,
                exit_code: out.code,
                decision: out.result["decision"].as_str().map(str::to_string),
                records,
                result: out.result.clone(),
                warnings: out.warnings.clone(),
                error: None,
            },
            Err(e) => Self {
                schema: SCHEMA,
                command: command.to_string(),
                ok: false,
                exit_code: 1,
                decision: None,
                records,
                result: Value::Null,
                warnings: Vec::new(),
                error: Some(e.clone()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema_doc_parses_and_names_every_kind() {
        let doc: Value = serde_json::from_str(SCHEMA_DOC).expect("schema JSON");
        assert_eq!(doc["$id"].as_str(), Some(SCHEMA));
        let listed: Vec<&str> = doc["$defs"]["kind"]["enum"]
            .as_array()
            .expect("kind enum")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        for (k, _) in KINDS {
            assert!(listed.contains(k), "schema doc lacks kind {k}");
        }
        assert_eq!(listed.len(), KINDS.len(), "schema doc lists extra kinds");
    }

    #[test]
    fn envelope_carries_decision_and_error() {
        let ok = Ok(CmdOutput::with_code(3, json!({"decision": "discard"})));
        let e = Envelope::new("run-experiment", &ok, vec![]);
        assert_eq!(
            (e.ok, e.exit_code, e.decision.as_deref()),
            (true, 3, Some("discard"))
        );
        let err: Result<CmdOutput, String> = Err("boom".into());
        let e = Envelope::new("ab", &err, vec![]);
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["ok"], false);
        assert_eq!(v["exit_code"], 1);
        assert_eq!(v["error"], "boom");
        assert_eq!(v["schema"], SCHEMA);
    }
}
