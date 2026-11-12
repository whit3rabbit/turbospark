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

mod moe_bound {
    use runtime::PhaseCounters;
    use turbospark_lab::phases::{moe_bound, report, MoeBoundClass};

    /// A 100-call stretch of 1 s: the named nanos set each share directly.
    fn stretch(routed_gpu: u64, expert_io: u64, router: u64) -> PhaseCounters {
        PhaseCounters {
            calls: 100,
            total_nanos: 1_000_000_000,
            expert_io_nanos: expert_io,
            router_nanos: router,
            routed_cb_gpu_nanos: routed_gpu,
            expert_requests: if expert_io > 0 || routed_gpu > 0 {
                800
            } else {
                0
            },
            ..Default::default()
        }
    }

    #[test]
    fn classes_follow_the_largest_term() {
        let io = moe_bound(&report(&stretch(100_000_000, 600_000_000, 100_000_000)));
        assert_eq!(io.class, MoeBoundClass::IoBound);
        assert!((io.io_share - 0.6).abs() < 1e-9, "{}", io.io_share);
        let loop_bound = moe_bound(&report(&stretch(100_000_000, 100_000_000, 200_000_000)));
        // "encode + logit readbk" is the unaccounted rest, 700 ms here (0.2 router + 0.7), and
        // it belongs to the host loop too.
        assert_eq!(loop_bound.class, MoeBoundClass::LoopBound);
        assert!(
            (loop_bound.loop_share - 0.9).abs() < 1e-9,
            "{}",
            loop_bound.loop_share
        );
        let compute = moe_bound(&report(&stretch(550_000_000, 150_000_000, 50_000_000)));
        assert_eq!(compute.class, MoeBoundClass::ComputeBound);
        assert!((compute.routed_gpu_share - 0.55).abs() < 1e-9);
        assert_eq!(
            moe_bound(&report(&stretch(0, 0, 0))).class,
            MoeBoundClass::Dense
        );
    }

    #[test]
    fn compute_bound_needs_the_gpu_half_not_only_quiet_io() {
        // Routed GPU at 30% with I/O at 10%: under the 50% rule this is not
        // compute bound. Swapping the two thresholds (0.2 GPU, 0.5 I/O) would
        // call it compute bound, which is the mutation this guards against.
        let b = moe_bound(&report(&stretch(300_000_000, 100_000_000, 50_000_000)));
        assert_ne!(b.class, MoeBoundClass::ComputeBound);
        assert_eq!(b.class, MoeBoundClass::LoopBound);
    }
}
