//! `tslab project` and `tslab calibrate` end to end on a synthetic lab
//! directory: ledger records in, a projection, a calibration, and the
//! bias table out. No GPU, no model.

use std::collections::BTreeMap;

use serde_json::{json, Value};
use turbospark_lab::args::Args;
use turbospark_lab::bench::{ArmCurve, Axis, BenchResult, Point};
use turbospark_lab::ledger::Ledger;
use turbospark_lab::projection::{calibrate_cmd, project_cmd};

fn args(v: &[&str]) -> Args {
    Args::new(v.iter().map(|s| s.to_string()).collect())
}

fn gdn_curve(ms: f64) -> BenchResult {
    let mut subject = BTreeMap::new();
    subject.insert("count".to_string(), json!(48));
    BenchResult {
        bench: "gdn_delta".into(),
        case: "gdn".into(),
        subject,
        x: Axis::new("m", "rows", "log2"),
        layout: "test".into(),
        arms: vec![ArmCurve {
            label: "prefill_lazy".into(),
            points: vec![Point::new(8.0, ms)],
            fit: None,
            unsupported: None,
        }],
        break_even: Vec::new(),
        notes: Vec::new(),
    }
}

fn put(lab: &std::path::Path, exp: &str, kind: &str, payload: Value) {
    Ledger::open(&lab.join(exp))
        .unwrap()
        .append(kind, payload)
        .unwrap();
}

#[test]
fn a_projection_is_calibrated_against_the_verdict_that_followed() {
    let lab = std::env::temp_dir().join(format!("tslab-projection-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&lab);
    // Single test per process: the lab dir is a process-wide variable.
    std::env::set_var("TURBOSPARK_LAB_DIR", &lab);

    // The same kernel timed by a runtime-Dk process and by a baked one.
    for (exp, ms) in [("runtime", 0.10), ("baked", 0.04)] {
        put(
            &lab,
            exp,
            "kernel_curve",
            json!({"bench": "gdn_delta", "case": "gdn", "eligible": true, "result": gdn_curve(ms)}),
        );
    }
    // 85 ms rounds at verify width 8, flag-less.
    for _ in 0..3 {
        put(
            &lab,
            "rounds",
            "round_cost",
            json!({"width": 8, "flags": "default", "eligible": true, "ms_per_round": 85.0,
                   "phases": {"verify": 66.8, "draft": 11.0}}),
        );
    }
    put(
        &lab,
        "rounds",
        "round_cost",
        json!({"width": 8, "flags": "default", "eligible": true, "ms_per_round": 900.0,
               "phases": {"verify": 800.0}}),
    );

    let out = project_cmd(args(&[
        "--knob",
        "gdn_dk128",
        "--default-from",
        "runtime",
        "--mode-from",
        "baked",
        "--x",
        "8",
        "--round-from",
        "rounds",
        "--width",
        "8",
        "--experiment",
        "proj",
    ]))
    .expect("project");
    let p = &out.result["projection"];
    // 48 layers * 0.06 ms saved = 2.88 ms of an 85 ms round. The base is a
    // median, so the 900 ms outlier does not move it (the 3x spike rule is
    // the same one `round_cost::by_depth` applies and is not what this checks).
    assert_eq!(p["base"]["ms"], 85.0, "{p}");
    assert!(
        (p["kernel_delta_ms"].as_f64().unwrap() + 2.88).abs() < 1e-9,
        "{p}"
    );
    let gain = p["improvement_throughput"].as_f64().unwrap();
    assert!((gain - (85.0 / (85.0 - 2.88) - 1.0)).abs() < 1e-9);
    assert_eq!(p["clears_floor"], true);
    assert!(
        Ledger::open(&lab.join("proj"))
            .unwrap()
            .of_kind("projection")
            .unwrap()
            .len()
            == 1
    );

    // A verdict that agrees (a run-experiment `confirm` shape), then one that
    // does not (a `tslab ab` verdict, a loss).
    put(
        &lab,
        "agrees",
        "confirm",
        json!({"decision": "keep", "verdict": {"median": 0.0407}}),
    );
    put(
        &lab,
        "disagrees",
        "verdict",
        json!({"metric": "decode_tok_s", "verdict": {"median": -0.05, "verdict": "loss"}}),
    );

    let ok = calibrate_cmd(args(&[
        "--projection",
        "proj",
        "--verdict",
        "agrees",
        "--experiment",
        "cal",
    ]))
    .expect("calibrate");
    assert_eq!(ok.result["calibration"]["miss"], false, "{}", ok.result);
    let bad = calibrate_cmd(args(&[
        "--projection",
        "proj",
        "--verdict",
        "disagrees",
        "--experiment",
        "cal",
    ]))
    .expect("calibrate");
    assert_eq!(bad.result["calibration"]["miss"], true);
    assert_eq!(bad.result["calibration"]["sign_flip"], true);

    // A typed-in number needs a source.
    assert!(calibrate_cmd(args(&["--projection", "proj", "--measured", "0.04"])).is_err());
    let doc = calibrate_cmd(args(&[
        "--projection",
        "proj",
        "--measured",
        "0.0407",
        "--source",
        "docs row",
        "--experiment",
        "cal",
    ]))
    .expect("doc-sourced");
    assert_eq!(doc.result["calibration"]["miss"], false);

    let all = calibrate_cmd(args(&["--all"])).expect("all");
    assert_eq!(all.result["calibrations"].as_array().unwrap().len(), 3);
    let bias = all.result["bias"].as_object().unwrap();
    assert_eq!(bias.len(), 1, "one bench and arm pair: {bias:?}");
    assert_eq!(bias.values().next().unwrap()["n"], 3);

    // A process-compared lane priced from one experiment would be one arm
    // against itself: refused with the way out.
    let same = project_cmd(args(&[
        "--knob",
        "gdn_dk128",
        "--curves",
        "runtime",
        "--x",
        "8",
        "--base-ms",
        "85",
    ]))
    .expect_err("same arm, one experiment");
    assert!(same.contains("--default-from"), "{same}");

    // A mode that is not a lane, and a base with two sources, are refused.
    assert!(project_cmd(args(&[
        "--knob",
        "batched_gemv_typo",
        "--curves",
        "runtime"
    ]))
    .is_err());
    assert!(project_cmd(args(&[
        "--knob",
        "gdn_dk128",
        "--default-from",
        "runtime",
        "--mode-from",
        "baked",
        "--x",
        "8",
        "--base-ms",
        "85",
        "--round-from",
        "rounds",
        "--width",
        "8",
    ]))
    .is_err());
    let _ = std::fs::remove_dir_all(&lab);
}
