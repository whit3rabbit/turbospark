//! The dashboard over a synthetic lab dir: the index counts what the ledgers
//! hold, every page renders self-contained, the compare view puts two
//! experiments on one chart, the family page links a ledger finding, and a
//! bad name is a 404 before any read.

use std::path::PathBuf;

use serde_json::json;
use turbospark_lab::dash::index::LabIndex;
use turbospark_lab::dash::pages::{self, Params};

fn lab_dir() -> PathBuf {
    // Tests run in parallel: the counter keeps each one in its own dir.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "tslab-dash-test-{}-{}-{}",
        std::process::id(),
        turbospark_lab::ledger::unix_ms(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    for (name, lines) in [("exp-a", ledger_a()), ("exp-b", ledger_b())] {
        let d = dir.join(name);
        std::fs::create_dir_all(d.join("runs")).unwrap();
        std::fs::write(d.join("ledger.jsonl"), lines.join("\n") + "\n").unwrap();
    }
    std::fs::write(
        dir.join("exp-a").join("notes.md"),
        "first line of notes\nmore\n",
    )
    .unwrap();
    // A directory without a ledger and one with a bad name are not experiments.
    std::fs::create_dir_all(dir.join("no-ledger")).unwrap();
    std::fs::create_dir_all(dir.join(".hidden")).unwrap();
    std::fs::write(dir.join(".hidden").join("ledger.jsonl"), "{}\n").unwrap();
    dir
}

fn kernel(id: &str) -> String {
    json!({"id": id, "kind": "kernel_curve", "ts_ms": 1793000000000u64, "schema": "tslab/1",
        "model": "/x/q.gturbo", "family": "qwen35",
        "result": {"bench": "attn_decode", "q_heads": 24, "kv_heads": 4, "head_dim": 256,
            "layers": 16, "gqa_supported": true,
            "points": [{"context": 512, "kernels": [{"kernel": "barrier", "ms": 1.0}, {"kernel": "gqa", "ms": 0.5}]},
                       {"context": 2048, "kernels": [{"kernel": "barrier", "ms": 3.0}, {"kernel": "gqa", "ms": 0.9}]}]}})
    .to_string()
}

fn ledger_a() -> Vec<String> {
    vec![
        json!({"id": "1793000000001-1-0", "kind": "run", "ts_ms": 1793000000001u64, "schema": "tslab/1",
            "arm": "plain", "round": 1, "warmup": false, "status": "ok", "eligible": false,
            "reasons": ["foreign CPU over 150% for 5s or more"], "family": "qwen35",
            "totals": {"prefill_tok_s": 150.0, "decode_tok_s": 22.0}, "peak_footprint_bytes": 700000000u64,
            "record": "/nonexistent/run.json", "run_id": "1793000000000-1-0"}).to_string(),
        json!({"id": "1793000000002-1-1", "kind": "run", "ts_ms": 1793000000002u64, "schema": "tslab/1",
            "arm": "mtp2", "round": 1, "warmup": false, "status": "ok", "eligible": true,
            "reasons": [], "family": "qwen35",
            "totals": {"prefill_tok_s": 140.0, "decode_tok_s": 34.0}, "peak_footprint_bytes": 900000000u64,
            "record": "/nonexistent/run2.json", "run_id": "1793000000000-1-1"}).to_string(),
        json!({"id": "1793000000003-1-2", "kind": "verdict", "ts_ms": 1793000000003u64, "schema": "tslab/1",
            "model": "/x/q.gturbo", "family": "qwen35", "metric": "decode_tok_s",
            "baseline": {"name": "plain"}, "candidate": {"name": "mtp2"},
            "verdict": {"verdict": "win", "median": 0.6, "ci_low": 0.5, "ci_high": 0.7, "pairs": 3,
                        "improvements": [0.55, 0.6, 0.65], "min_effect": 0.02},
            "null_floor": 0.01, "memory": {"passed": false}}).to_string(),
        json!({"id": "1793000000004-1-3", "kind": "sweep", "ts_ms": 1793000000004u64, "schema": "tslab/1",
            "sweep_id": "spec_block", "family": "qwen35", "model": "/x/q.gturbo",
            "baseline": {"name": "plain"}, "metric": "decode_tok_s", "floor": 0.02,
            "candidates": [{"arm": "mtp2", "verdict": {"verdict": "win", "median": 0.6, "ci_low": 0.5, "ci_high": 0.7, "pairs": 3}, "clean_pairs": 3, "dirty_pairs": 0},
                           {"arm": "mtp4", "verdict": {"verdict": "noise", "median": 0.05, "ci_low": -0.01, "ci_high": 0.1, "pairs": 3}, "clean_pairs": 3, "dirty_pairs": 1}],
            "best": {"arm": "mtp2", "median": 0.6}, "stopped": null}).to_string(),
        kernel("1793000000005-1-4"),
        "this line is damage".to_string(),
    ]
}

fn ledger_b() -> Vec<String> {
    vec![
        json!({"id": "1793000001000-2-0", "kind": "profile_summary", "ts_ms": 1793000001000u64, "schema": "tslab/1",
            "model": "/x/q.gturbo", "summary": {"auto": {"prefill_tok_s": 84.5, "decode_tok_s": 48.8, "peak_gib": 1.6, "clean_runs": 3}}}).to_string(),
        kernel("1793000001001-2-1"),
    ]
}

fn no_fetch(html: &str) {
    assert!(
        html.starts_with("<!doctype html>"),
        "{}",
        &html[..40.min(html.len())]
    );
    assert!(
        !html.contains("<script"),
        "a dashboard page carries no script"
    );
    assert!(
        !html.contains("http://") && !html.contains("https://"),
        "nothing fetched from anywhere"
    );
}

#[test]
fn index_counts_what_the_ledgers_hold() {
    let dir = lab_dir();
    let ix = LabIndex::build(&dir);
    assert_eq!(ix.experiments.len(), 2, "{:?}", ix.experiments.keys());
    let a = &ix.experiments["exp-a"].summary;
    assert_eq!(a.counts["run"], 2);
    assert_eq!(a.counts["verdict"], 1);
    assert_eq!(a.counts["sweep"], 1);
    assert_eq!((a.runs_clean, a.runs_dirty), (1, 1));
    assert_eq!(a.skipped_lines, 1, "the damaged line is counted, not fatal");
    assert!(a.has_notes);
    assert!(a.families.contains("qwen35"));
    assert!(a.models.contains("q.gturbo"));
    assert_eq!(ix.verdicts.len(), 1);
    assert_eq!(ix.sweeps.len(), 1);
    assert_eq!(ix.sweeps[0].best_arm.as_deref(), Some("mtp2"));
    assert_eq!(ix.profiles.len(), 1);
    assert_eq!(ix.profiles[0].decode_tok_s, Some(48.8));
    // Nothing changed: refresh is a no-op; an appended line is seen.
    assert!(!ix.clone().refresh());
    let mut ix2 = ix.clone();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(dir.join("exp-b").join("ledger.jsonl"))
        .unwrap();
    use std::io::Write as _;
    writeln!(f, "{}", kernel("1793000002000-2-2")).unwrap();
    assert!(ix2.refresh());
    assert_eq!(ix2.experiments["exp-b"].summary.counts["kernel_curve"], 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_page_renders_self_contained() {
    let dir = lab_dir();
    let ix = LabIndex::build(&dir);
    let home = pages::experiments_page(&ix, &Params::default());
    no_fetch(&home);
    assert!(home.contains("exp-a") && home.contains("exp-b"));
    assert!(home.contains("first line of notes"));
    assert!(home.contains("win"), "the best verdict shows on the table");
    let exp = pages::experiment_page(&ix, "exp-a").expect("exp-a exists");
    no_fetch(&exp);
    assert!(
        exp.contains("<svg"),
        "its kernel curve and sweep bars render"
    );
    assert!(exp.contains("foreign CPU"), "the dirty run's reason shows");
    assert!(pages::experiment_page(&ix, "missing").is_none());
    let v = pages::verdicts_page(&ix, &Params::default());
    no_fetch(&v);
    assert!(v.contains("plain vs mtp2"));
    let s = pages::sweeps_page(&ix, &Params::default());
    no_fetch(&s);
    assert!(s.contains("spec_block") && s.contains("mtp4"));
    let fam = pages::families_page(&ix);
    no_fetch(&fam);
    assert!(fam.contains("/family/qwen35"));
    let rec = pages::record_page(&ix, "exp-a", "1793000000003-1-2").expect("record exists");
    no_fetch(&rec);
    assert!(rec.contains("&quot;verdict&quot;"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compare_puts_two_experiments_on_one_chart() {
    let dir = lab_dir();
    let ix = LabIndex::build(&dir);
    let p = Params::from_query(
        &[("exp".to_string(), "exp-a,exp-b".to_string())]
            .into_iter()
            .collect(),
    );
    let html = pages::compare_page(&ix, &p);
    no_fetch(&html);
    // Both experiments' attn_decode curves share one card: one legend with
    // experiment-prefixed labels, not two cards.
    assert!(
        html.contains("exp-a: barrier") && html.contains("exp-b: barrier"),
        "labels carry the experiment"
    );
    // One kernel_ms card holds both experiments' curves, not one card each.
    assert_eq!(
        html.matches("<span class=\"sub\">kernel_ms</span>").count(),
        1,
        "both experiments' attn_decode curves share one card"
    );
    assert!(html.contains("Verdict board"));
    let empty = pages::compare_page(&ix, &Params::default());
    assert!(empty.contains("Pick experiments"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn family_page_links_a_ledger_finding_only_when_its_record_is_here() {
    let dir = lab_dir();
    let ix = LabIndex::build(&dir);
    let html = pages::family_page(&ix, "qwen35").expect("qwen35 profile");
    no_fetch(&html);
    assert!(html.contains("Recorded findings"));
    // The qwen35 profile's ledger findings point at experiments on the real
    // host, which this synthetic lab dir lacks.
    assert!(html.contains("not on this host"));
    assert!(
        html.contains("plain vs mtp2"),
        "the live verdict is listed under the family"
    );
    assert!(pages::family_page(&ix, "nope").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn router_serves_pages_and_json_and_refuses_bad_names() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt as _;
    use tower::ServiceExt as _;
    let dir = lab_dir();
    let app = turbospark_lab::dash::router(turbospark_lab::dash::App::new(&dir));
    let get = |uri: &str| Request::builder().uri(uri).body(Body::empty()).unwrap();
    let r = app.clone().oneshot(get("/")).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let r = app.clone().oneshot(get("/api/experiments")).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let body = r.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["schema"], "tslab/1");
    assert_eq!(v["experiments"].as_array().unwrap().len(), 2);
    let r = app
        .clone()
        .oneshot(get("/compare?exp=exp-a&exp=exp-b"))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let body = r.into_body().collect().await.unwrap().to_bytes();
    assert!(String::from_utf8_lossy(&body).contains("exp-b: barrier"));
    let r = app
        .clone()
        .oneshot(get("/experiment/..%2Fetc"))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    let r = app
        .clone()
        .oneshot(get("/record/exp-a/not-an-id"))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    let r = app
        .clone()
        .oneshot(get("/api/record/exp-a/1793000000003-1-2"))
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let r = app.oneshot(get("/nothing/here")).await.unwrap();
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    let _ = std::fs::remove_dir_all(&dir);
}
