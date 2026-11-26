//! `tslab report --md`: a text record of an experiment that carries its own
//! limits (LAB-32). Everything is read from ledger records; a missing field
//! prints `unrecorded` and is never inferred.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

const UNRECORDED: &str = "unrecorded";

fn text(v: &Value) -> String {
    match v {
        Value::Null => UNRECORDED.into(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn pct(v: &Value) -> String {
    v.as_f64()
        .map_or_else(|| UNRECORDED.into(), |x| format!("{:+.2}%", x * 100.0))
}

fn of_kind<'a>(records: &'a [Value], kind: &'a str) -> impl Iterator<Item = &'a Value> {
    records.iter().filter(move |r| r["kind"] == kind)
}

/// Why a reader must not take more from these records than they hold.
pub fn claims_not_made(records: &[Value]) -> Vec<String> {
    let mut out = Vec::new();
    let runs: Vec<&Value> = of_kind(records, "run").collect();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for r in runs.iter().filter(|r| r["eligible"] == false) {
        for reason in r["reasons"].as_array().into_iter().flatten() {
            *reasons.entry(text(reason)).or_default() += 1;
        }
    }
    let dirty = runs.iter().filter(|r| r["eligible"] == false).count();
    if dirty > 0 {
        out.push(format!(
            "{dirty} of {} runs were dirty and are in no verdict:",
            runs.len()
        ));
        out.extend(reasons.iter().map(|(r, n)| format!("  {n} x {r}")));
    }
    for (field, claim) in [
        (
            "ballast_gib",
            "Runs under memory ballast model a smaller Mac; they do not measure one.",
        ),
        (
            "controlled_contention",
            "Runs under deliberate CPU contention are not quotable.",
        ),
    ] {
        if runs.iter().any(|r| !r[field].is_null()) {
            out.push(claim.into());
        }
    }
    let verdicts: Vec<&Value> = of_kind(records, "verdict").collect();
    let gated = of_kind(records, "gate").next().is_some()
        || of_kind(records, "experiment").any(|r| !r["quality"].is_null())
        || verdicts.iter().any(|r| !r["quality"].is_null());
    if !gated {
        out.push("No quality gate is recorded: nothing here says the output is unchanged.".into());
    }
    if of_kind(records, "null_calibration").next().is_none() {
        out.push(
            "No null calibration in these records: the effect floor is the evaluator's default."
                .into(),
        );
    }
    if of_kind(records, "confirm").next().is_none() {
        out.push("No held-out confirm pass.".into());
    }
    let methods: BTreeSet<String> = verdicts
        .iter()
        .map(|r| text(&r["methodology"]["id"]))
        .collect();
    if methods.len() > 1 {
        out.push(format!(
            "Verdicts come from {} methodology ids and are NOT comparable with each other.",
            methods.len()
        ));
    }
    out.push("One machine and one chip; nothing here is a claim about another.".into());
    out
}

/// The whole record. `provenance` is a worker record's, when one was found.
pub fn render(name: &str, records: &[Value], provenance: &Value) -> String {
    let mut s = format!("# tslab experiment {name}\n\n## Environment\n\n");
    for key in [
        "chip",
        "macos",
        "macos_build",
        "toolchain_id",
        "git_rev",
        "git_dirty",
        "features",
        "turbospark_env",
    ] {
        s.push_str(&format!("- {key}: {}\n", text(&provenance[key])));
    }

    s.push_str("\n## Reproduce\n\n");
    match of_kind(records, "experiment_start").last() {
        Some(start) => s.push_str(&format!(
            "Spec sha256 {} against base {}. Write the spec below to spec.json and run \
             `tslab {} --spec spec.json`.\n\n```json\n{}\n```\n",
            text(&start["spec_sha256"]),
            text(&start["base"]),
            text(&start["purpose"]),
            serde_json::to_string_pretty(&start["spec"]).unwrap_or_default()
        )),
        None => s.push_str(&format!(
            "Command: {UNRECORDED} (no experiment_start record).\n"
        )),
    }

    s.push_str("\n## Verdicts\n\n");
    s.push_str(
        "| baseline | candidate | metric | verdict | pairs | median | 95% CI | null floor |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|\n");
    for v in of_kind(records, "verdict") {
        let r = &v["verdict"];
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} to {} | {} |\n",
            text(&v["baseline"]["name"]),
            text(&v["candidate"]["name"]),
            text(&v["metric"]),
            text(&r["verdict"]),
            text(&r["pairs"]),
            pct(&r["median"]),
            pct(&r["ci_low"]),
            pct(&r["ci_high"]),
            pct(&v["null_floor"]),
        ));
    }

    s.push_str("\n## Quality gates and decisions\n\n");
    for r in records {
        let kind = r["kind"].as_str().unwrap_or("");
        let q = &r["quality"];
        if matches!(kind, "gate" | "experiment") && !q.is_null() {
            s.push_str(&format!(
                "- {kind}: class {}, passed {}, mean KL {}\n",
                text(&q["class"]),
                text(&q["passed"]),
                text(&q["mean_kl"])
            ));
        }
        if matches!(kind, "experiment" | "confirm") {
            s.push_str(&format!(
                "- {kind} decision: {} ({})\n",
                text(&r["decision"]),
                text(&r["reason"])
            ));
        }
    }

    s.push_str("\n## Claims not made\n\n");
    for c in claims_not_made(records) {
        s.push_str(&format!("- {c}\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn limits_are_derived_from_the_records() {
        let records = vec![
            json!({"kind": "run", "eligible": false, "reasons": ["AC power was not observed"],
                   "ballast_gib": 8, "controlled_contention": null}),
            json!({"kind": "run", "eligible": true, "reasons": []}),
        ];
        let claims = claims_not_made(&records).join("\n");
        assert!(claims.contains("1 of 2 runs were dirty"), "{claims}");
        assert!(claims.contains("1 x AC power was not observed"));
        assert!(claims.contains("No quality gate"));
        assert!(claims.contains("ballast"));
        assert!(!claims.contains("contention"));
        let clean = vec![
            json!({"kind": "gate", "quality": {"passed": true}}),
            json!({"kind": "null_calibration"}),
            json!({"kind": "confirm", "decision": "keep"}),
            json!({"kind": "verdict", "methodology": {"id": "a"}}),
            json!({"kind": "verdict", "methodology": {"id": "b"}}),
        ];
        let claims = claims_not_made(&clean).join("\n");
        assert!(!claims.contains("No quality gate") && !claims.contains("No held-out"));
        assert!(claims.contains("2 methodology ids"), "{claims}");
    }

    #[test]
    fn missing_fields_read_as_unrecorded() {
        let out = render(
            "e",
            &[json!({"kind": "verdict", "verdict": {"verdict": "win"}})],
            &Value::Null,
        );
        assert!(out.contains("- chip: unrecorded"), "{out}");
        assert!(
            out.contains("| unrecorded | unrecorded | unrecorded | win |"),
            "{out}"
        );
        assert!(out.contains("Command: unrecorded"));
        assert!(out.is_ascii());
    }
}
