//! The census arithmetic on a hand-built capture: every selection counted
//! once, inclusive thresholds, the union-fit table, and the routed
//! sub-batch replay against values worked out by hand. Offline, portable.

use serde_json::{json, Value};
use turbospark_lab::census::{parse_capture, routed_sub_batches, summarize};

/// One layer, 8 experts, top-2. A 4-token prefill row, a 1-token tail, a
/// decode row, and an incomplete row that must be dropped.
fn capture() -> Value {
    json!({
        "num_layers": 1, "num_experts": 8, "top_k": 2,
        "counts": [[4, 3, 3, 1, 1, 0, 0, 0]],
        "batches": {"version": 1, "truncated": false, "rows": [
            {"request": 0, "batch": 1, "phase": "prefill", "layer": 0, "token_start": 0,
             "token_count": 4, "trace_start": 0, "trace_end": 8,
             "routes": [[0, 1], [0, 2], [0, 3], [1, 2]], "completed": true,
             "cache_hits": 0, "cache_misses": 0, "bytes_requested": 0, "io_samples": 0},
            {"request": 0, "batch": 2, "phase": "prefill", "layer": 0, "token_start": 4,
             "token_count": 1, "trace_start": 8, "trace_end": 10,
             "routes": [[4, 1]], "completed": true,
             "cache_hits": 0, "cache_misses": 0, "bytes_requested": 0, "io_samples": 0},
            {"request": 0, "batch": 3, "phase": "decode", "layer": 0, "token_start": 5,
             "token_count": 1, "trace_start": 10, "trace_end": 12,
             "routes": [[0, 2]], "completed": true,
             "cache_hits": 0, "cache_misses": 0, "bytes_requested": 0, "io_samples": 0},
            {"request": 1, "batch": 1, "phase": "prefill", "layer": 0, "token_start": 0,
             "token_count": 2, "trace_start": 12, "trace_end": 14,
             "routes": [[5, 6]], "completed": false,
             "cache_hits": 0, "cache_misses": 0, "bytes_requested": 0, "io_samples": 0}
        ]}
    })
}

#[test]
fn parses_and_drops_incomplete_rows() {
    let c = parse_capture(&capture()).unwrap();
    assert_eq!(
        (c.num_experts, c.top_k, c.rows.len(), c.dropped),
        (8, 2, 3, 1)
    );
    let mut bad = capture();
    bad["batches"]["rows"][0]["routes"][0] = json!([0, 0]);
    assert!(
        parse_capture(&bad).is_err(),
        "a route must hold distinct experts"
    );
    let mut legacy = capture();
    legacy["batches"] = Value::Null;
    assert!(
        parse_capture(&legacy).is_err(),
        "a counts-only capture is refused"
    );
}

#[test]
fn reuse_is_inclusive_and_conserves_selections() {
    let s = summarize(&parse_capture(&capture()).unwrap());
    let p = &s["phases"]["prefill"];
    // Two complete prefill rows: 4 + 1 tokens, 10 selections.
    assert_eq!(p["rows"], 2);
    assert_eq!(p["selections"], 10);
    assert_eq!(
        p["largest_group"], 3,
        "expert 0 three times in the 4-token row"
    );
    // Thresholds are per row and inclusive. In the 4-token row the counts
    // are {0: 3, 1: 2, 2: 2, 3: 1}; the tail row has nothing above 1.
    let cov = |t: &str| p["reuse"][t]["selection_coverage"].as_f64().unwrap();
    assert!((cov("8") - 0.0).abs() < 1e-12);
    assert!((cov("16") - 0.0).abs() < 1e-12);
    // Decode: one row, one token, both experts used once.
    assert_eq!(s["phases"]["decode"]["selections"], 2);
    assert_eq!(s["phases"]["decode"]["largest_group"], 1);
    assert_eq!(s["complete_rows"], 3);
    assert_eq!(s["dropped_rows"], 1);
    assert_eq!(s["notes"].as_array().unwrap().len(), 0, "{}", s["notes"]);
}

#[test]
fn a_decode_only_capture_names_the_prefill_gap() {
    let mut v = capture();
    for row in v["batches"]["rows"].as_array_mut().unwrap() {
        row["phase"] = json!("decode");
    }
    let s = summarize(&parse_capture(&v).unwrap());
    assert!(s["phases"]["prefill"].is_null());
    let notes = s["notes"].as_array().unwrap();
    assert_eq!(notes.len(), 1);
    assert!(notes[0].as_str().unwrap().contains("no prefill rows"));
}

#[test]
fn threshold_boundary_is_visible_on_a_small_capture() {
    // Rebuild the 4-token row so the thresholds in the table bite: a row
    // with expert 0 used exactly 16 times among 16 tokens of top-2.
    let mut v = capture();
    let routes: Vec<Value> = (0..16).map(|i| json!([0, 1 + (i % 7)])).collect();
    v["batches"]["rows"][0]["token_count"] = json!(16);
    v["batches"]["rows"][0]["trace_end"] = json!(32);
    v["batches"]["rows"][0]["routes"] = Value::Array(routes);
    v["batches"]["rows"][1]["trace_start"] = json!(32);
    v["batches"]["rows"][1]["trace_end"] = json!(34);
    let s = summarize(&parse_capture(&v).unwrap());
    let p = &s["phases"]["prefill"];
    // 34 selections over both rows; expert 0 carries 16 of them in its row.
    assert_eq!(p["selections"], 34);
    let cov = |t: &str| p["reuse"][t]["selection_coverage"].as_f64().unwrap();
    assert!((cov("16") - 16.0 / 34.0).abs() < 1e-12, "{}", cov("16"));
    assert!(
        (cov("17") - 0.0).abs() < 1e-12,
        ">= 17 must not include a 16-use expert"
    );
    assert!((cov("8") - 16.0 / 34.0).abs() < 1e-12);
}

#[test]
fn union_fit_and_concentration() {
    let s = summarize(&parse_capture(&capture()).unwrap());
    let p = &s["phases"]["prefill"];
    // The full-width row is the 4-token one: union {0,1,2,3} = 4.
    assert_eq!(p["width"], 4);
    assert_eq!(p["full_rows"], 1);
    assert_eq!(p["union"]["median"], 4.0);
    assert_eq!(p["slots_fit"]["16"], 1.0);
    // Layer counts over both prefill rows: {0:3, 1:3, 2:2, 3:1, 4:1}; half
    // of 10 selections needs 2 experts, 80% needs 3, 95% needs 5.
    assert_eq!(p["concentration"]["50"]["experts_median_over_layers"], 2.0);
    assert_eq!(p["concentration"]["80"]["experts_median_over_layers"], 3.0);
    assert_eq!(p["concentration"]["95"]["experts_median_over_layers"], 5.0);
}

#[test]
fn routed_sub_batches_replay_the_greedy_shrink() {
    let routes: Vec<Vec<usize>> = vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![1, 2]];
    assert_eq!(routed_sub_batches(&routes, 2).unwrap(), vec![1, 1, 1, 1]);
    assert_eq!(routed_sub_batches(&routes, 3).unwrap(), vec![2, 1, 1]);
    assert_eq!(routed_sub_batches(&routes, 4).unwrap(), vec![4]);
    assert!(
        routed_sub_batches(&routes, 1).is_err(),
        "a token over the cap is refused"
    );
    let s = summarize(&parse_capture(&capture()).unwrap());
    let sb = &s["phases"]["prefill"]["sub_batches"];
    assert_eq!(sb["16"]["per_row_median"], 1.0);
    assert_eq!(sb["16"]["tokens_per_sub_batch_mean"], 4.0);
}
