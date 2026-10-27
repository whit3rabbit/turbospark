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
    // The one script is inline and loads nothing, and the one link is the
    // favicon as an inline data URI: a saved page still reads the same as a
    // live one, with or without either.
    assert!(
        !html.contains("<script src"),
        "a dashboard page loads no script from anywhere"
    );
    assert_eq!(
        html.matches("<link").count(),
        1,
        "the favicon is the only link on a dashboard page"
    );
    assert!(
        html.contains("<link rel=\"icon\" type=\"image/svg+xml\" href=\"data:image/svg+xml,"),
        "the one link is the inline favicon"
    );
    assert!(
        !html.contains("http://") && !html.contains("https://") && !html.contains("fetch("),
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
    assert!(home.contains("<title>TurboSpark Lab</title>"));
    assert!(
        home.contains("TurboSpark <span>Lab</span>"),
        "the top bar brands the site TurboSpark Lab"
    );
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

/// Appends one line to a fixture experiment's ledger.
fn append_line(dir: &std::path::Path, exp: &str, line: String) {
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(dir.join(exp).join("ledger.jsonl"))
        .unwrap();
    writeln!(f, "{line}").unwrap();
}

#[test]
fn a_poisoned_ledger_family_renders_escaped() {
    let dir = lab_dir();
    // The page shell legitimately holds one inline `<script>`, so the
    // payload is asserted by its distinctive prefix, not by bare tags.
    append_line(
        &dir,
        "exp-a",
        json!({"id": "1793000000006-1-5", "kind": "verdict", "ts_ms": 1793000000006u64,
            "schema": "tslab/1", "model": "/x/p.gturbo",
            "family": "fam<script>alert(1)</script>", "metric": "decode_tok_s",
            "baseline": {"name": "plain"}, "candidate": {"name": "mtp2"},
            "verdict": {"verdict": "win", "median": 0.6, "ci_low": 0.5, "ci_high": 0.7,
                        "pairs": 3, "improvements": [0.55, 0.6, 0.65], "min_effect": 0.02},
            "null_floor": 0.01, "memory": {"passed": true}})
        .to_string(),
    );
    let ix = LabIndex::build(&dir);
    let v = pages::verdicts_page(&ix, &Params::default());
    no_fetch(&v);
    assert!(
        v.contains("fam&lt;script&gt;"),
        "a ledger family is escaped before it reaches the page"
    );
    assert!(
        !v.contains("fam<script>"),
        "no markup crosses from a ledger value"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_run_id_cannot_walk_out_of_the_runs_dir() {
    let dir = lab_dir();
    // `lab/exp-a/runs/../../secret.stderr` resolves to the lab root, well
    // outside the runs dir the record page is supposed to read.
    std::fs::write(dir.join("secret.stderr"), "TSLAB-SECRET-98765\n").unwrap();
    append_line(
        &dir,
        "exp-a",
        json!({"id": "1793000000007-1-6", "kind": "run", "ts_ms": 1793000000007u64,
            "schema": "tslab/1", "arm": "plain", "round": 2, "warmup": false, "status": "ok",
            "eligible": true, "reasons": [], "family": "qwen35",
            "record": "/nonexistent/run7.json", "run_id": "../../secret"})
        .to_string(),
    );
    // A well-formed run id still finds its stderr tail.
    std::fs::write(
        dir.join("exp-a")
            .join("runs")
            .join("1793000000000-1-0.stderr"),
        "one line of stderr\n",
    )
    .unwrap();
    let ix = LabIndex::build(&dir);
    let bad = pages::record_page(&ix, "exp-a", "1793000000007-1-6").expect("record exists");
    no_fetch(&bad);
    assert!(
        !bad.contains("TSLAB-SECRET-98765"),
        "a crafted run_id must not read outside the runs dir"
    );
    let good = pages::record_page(&ix, "exp-a", "1793000000001-1-0").expect("record exists");
    assert!(
        good.contains("stderr (last"),
        "a well-formed run id still shows its stderr tail"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_record_pointer_is_read_only_inside_the_lab_dir() {
    let dir = lab_dir();
    // Both record files exist; only the one inside the lab dir may be read.
    let outside = dir
        .parent()
        .unwrap()
        .join(format!("tslab-dash-outside-{}.json", std::process::id()));
    std::fs::write(&outside, json!({"family": "outside-family"}).to_string()).unwrap();
    let inside = dir
        .join("exp-a")
        .join("runs")
        .join("1793000000008-1-7.json");
    std::fs::write(
        &inside,
        json!({"family": "inside-family", "provenance": {"chip": "M4",
               "git_rev": "0123456789abcdef"}})
        .to_string(),
    )
    .unwrap();
    for (id, path) in [
        ("1793000000009-1-8", outside.clone()),
        ("1793000000010-1-9", inside.clone()),
    ] {
        append_line(
            &dir,
            "exp-a",
            json!({"id": id, "kind": "run", "ts_ms": 1793000000011u64, "schema": "tslab/1",
                "arm": "plain", "round": 3, "warmup": false, "status": "ok", "eligible": true,
                "reasons": [], "family": "qwen35", "run_id": id,
                "record": path.display().to_string()})
            .to_string(),
        );
    }
    let ix = LabIndex::build(&dir);
    let s = &ix.experiments["exp-a"].summary;
    assert!(
        s.families.contains("inside-family") && s.chips.contains("M4"),
        "a record inside the lab dir is still read for provenance"
    );
    assert!(
        !s.families.contains("outside-family"),
        "a record pointer outside the lab dir is not read"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&outside);
}

#[tokio::test]
async fn the_loopback_router_refuses_a_foreign_host() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt as _;
    let dir = lab_dir();
    let app = turbospark_lab::dash::serve_router(turbospark_lab::dash::App::new(&dir), true);
    let get = |host: &str| {
        Request::builder()
            .uri("/")
            .header("host", host)
            .body(Body::empty())
            .unwrap()
    };
    // A rebound DNS name arrives with its own Host and is refused.
    let r = app.clone().oneshot(get("evil.example")).await.unwrap();
    assert_eq!(r.status(), StatusCode::FORBIDDEN);
    let r = app.clone().oneshot(get("evil.example:7878")).await.unwrap();
    assert_eq!(r.status(), StatusCode::FORBIDDEN);
    for host in [
        "127.0.0.1:7878",
        "localhost:7878",
        "[::1]:7878",
        "localhost",
    ] {
        let r = app.clone().oneshot(get(host)).await.unwrap();
        assert_eq!(r.status(), StatusCode::OK, "Host {host} is this machine");
        assert_eq!(
            r.headers()["x-content-type-options"],
            "nosniff",
            "every response is nosniff"
        );
        assert!(
            r.headers()["content-security-policy"]
                .to_str()
                .unwrap()
                .contains("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:"),
            "every response carries the CSP"
        );
    }
    // An off-loopback bind is the user's explicit choice and answers any Host.
    let lan = turbospark_lab::dash::serve_router(turbospark_lab::dash::App::new(&dir), false);
    let r = lan.oneshot(get("evil.example")).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    let _ = std::fs::remove_dir_all(&dir);
}
