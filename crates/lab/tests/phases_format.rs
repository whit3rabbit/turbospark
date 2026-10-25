#![cfg(target_os = "macos")]
//! The lab's bucket table must be the CLI's: same names, same order, same
//! "everything else" definition. A drift here means a tslab profile and a
//! `TURBOSPARK_PHASES=1` printout disagree about what a bucket is.

use runtime::PhaseCounters;
use turbospark_lab::phases::{delta, report, BUCKET_NAMES};

#[test]
fn bucket_labels_match_the_cli_printout() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cli/src/generate/format.rs"
    ))
    .unwrap();
    // The CLI's bucket rows are `("<label padded>", p.<field>)` tuples.
    let labels: Vec<String> = src
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            let rest = l.strip_prefix("(\"")?;
            let (label, tail) = rest.split_once('"')?;
            (tail.starts_with(", p.") || tail.starts_with(", other"))
                .then(|| label.trim().to_string())
        })
        .collect();
    assert_eq!(labels, BUCKET_NAMES.to_vec());
}

#[test]
fn buckets_sum_to_total_and_deltas_subtract() {
    let before = PhaseCounters {
        calls: 10,
        total_nanos: 1_000,
        gpu_wait_nanos: 100,
        ..Default::default()
    };
    let after = PhaseCounters {
        calls: 110,
        total_nanos: 101_000,
        gpu_wait_nanos: 30_100,
        final_wait_nanos: 5_000,
        router_nanos: 4_000,
        expert_io_nanos: 20_000,
        bind_nanos: 1_000,
        pipeline_wait_nanos: 10_000,
        expert_requests: 800,
        expert_hits: 600,
        ..Default::default()
    };
    let d = delta(&before, &after);
    assert_eq!(d.calls, 100);
    assert_eq!(d.total_nanos, 100_000);
    let r = report(&d);
    let sum: f64 = r.buckets.iter().map(|b| b.ms).sum();
    assert!((sum - r.total_ms).abs() < 1e-9, "{sum} vs {}", r.total_ms);
    let other = r.buckets.last().unwrap();
    assert_eq!(other.name, "encode + logit readbk");
    assert!((other.ms - 0.03).abs() < 1e-12, "{}", other.ms);
    assert_eq!(
        r.expert_bytes_physical, None,
        "unsampled physical must read None"
    );
}
