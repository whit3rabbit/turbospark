//! `run_sweep` end to end against a fake worker: a shell script the session
//! spawns in place of `tslab-worker`, steered by `TURBOSPARK_FAKE_*`
//! variables in each arm (the worker gets only the arm's environment). No
//! model, no GPU.
//!
//! The host monitor still judges every run, and a compile anywhere on the
//! machine dirties it, so verdict asserts accept INCONCLUSIVE when a pair was
//! not clean: a dirty pair must never produce a WIN or LOSS. Engagement,
//! refusal and pair order do not depend on the host and are asserted outright.
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use turbospark_lab::args::parse_arm;
use turbospark_lab::host::HostRules;
use turbospark_lab::orchestrate::{run_sweep, Session, SweepArm};
use turbospark_lab::stats::Verdict;
use turbospark_lab::workload::{Arm, Metric, OpenParams, PromptSource, Workload};

/// Writes the record the orchestrator reads, prints the engagement lines it
/// was told to, or refuses with the worker's refusal exit code.
const FAKE_WORKER: &str = r#"#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [ -n "$TURBOSPARK_FAKE_REFUSE" ]; then
  printf '{"status":"refused","refusal":"%s"}' "$TURBOSPARK_FAKE_REFUSE" > "$out"
  exit 3
fi
if [ -n "$TURBOSPARK_FAKE_ENGAGED" ]; then
  echo "$TURBOSPARK_FAKE_ENGAGED" | tr ',' '\n' | sed 's/^/[engaged] /' >&2
fi
tps="${TURBOSPARK_FAKE_TPS:-100}"
printf '{"status":"ok","family":"qwen35","totals":{"prefill_tok_s":%s,"decode_tok_s":%s},"prompts":[{"cache_class":"warm"}],"peak_footprint_bytes":1000000}' "$tps" "$tps" > "$out"
"#;

fn scratch() -> PathBuf {
    // Tests run in parallel: the counter keeps each one in its own dir.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "tslab-sweep-test-{}-{}-{}",
        std::process::id(),
        turbospark_lab::ledger::unix_ms(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn fake_worker(dir: &Path) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join("fake-worker.sh");
    std::fs::write(&p, FAKE_WORKER).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn session(dir: &Path, experiment: &str) -> Session {
    // Loose on everything a rule makes configurable: the asserts below hold
    // a dirty pair to INCONCLUSIVE instead. No quiet wait, so a busy host
    // does not stall the test for minutes.
    let rules = HostRules {
        max_total_cpu_percent: f64::MAX,
        max_single_cpu_percent: f64::MAX,
        sustained_s: f64::MAX,
        require_ac: false,
        quiet_wait_s: 0.0,
        min_free_gb: 0.0,
    };
    let mut s = Session::new(
        &dir.join("lab"),
        experiment,
        "/nonexistent/fake-model",
        rules,
        Duration::from_secs(30),
    )
    .unwrap()
    .with_worker(fake_worker(dir));
    s.verbose = false;
    s
}

fn arm(spec: &str) -> Arm {
    parse_arm(spec, OpenParams::default(), &BTreeMap::new()).unwrap()
}

fn workload() -> Workload {
    Workload {
        prompts: vec![PromptSource::Protocol("long-synthesis".into())],
        decode_tokens: 0,
        prefill_chunk: Some(128),
        max_context: 4096,
        speculation: None,
        audio: None,
    }
}

fn find<'a>(arms: &'a [SweepArm], name: &str) -> &'a SweepArm {
    arms.iter().find(|a| a.candidate.name == name).unwrap()
}

/// `want` when every pair was clean, INCONCLUSIVE otherwise.
fn assert_verdict(a: &SweepArm, want: Verdict) {
    let clean = a.verdict.pairs;
    if clean == a.pairs.len() {
        assert_eq!(
            a.verdict.verdict, want,
            "{}: {:?}",
            a.candidate.name, a.verdict
        );
    } else {
        assert_eq!(
            a.verdict.verdict,
            Verdict::Inconclusive,
            "{}: {clean} of {} pairs clean must not decide",
            a.candidate.name,
            a.pairs.len()
        );
    }
}

/// The run counter, the last field of a run id: spawn order within a process.
fn seq(run_id: &str) -> u64 {
    run_id.rsplit('-').next().unwrap().parse().unwrap()
}

#[test]
fn each_candidate_is_judged_against_the_baseline_in_alternating_order() {
    let dir = scratch();
    let s = session(&dir, "sweep-verdicts");
    let base = arm("base:TURBOSPARK_FAKE_TPS=100");
    let cands = [
        arm("fast:TURBOSPARK_FAKE_TPS=130"),
        arm("same:TURBOSPARK_FAKE_TPS=100"),
        arm("slow:TURBOSPARK_FAKE_TPS=70"),
    ];
    let run = run_sweep(
        &s,
        &base,
        &cands,
        &workload(),
        Metric::PrefillTokS,
        0.02,
        3,
        3,
        None,
    )
    .unwrap();
    assert!(run.stopped.is_none(), "{:?}", run.stopped);
    assert_eq!(run.arms.len(), 3);
    for a in &run.arms {
        assert!(a.refused.is_none());
        assert_eq!(a.pairs.len(), 3, "{}: max_pairs is 3", a.candidate.name);
        for (b, c) in &a.pairs {
            assert_eq!(b.arm, "base");
            assert_eq!(c.arm, a.candidate.name);
            assert_eq!(b.round, c.round);
            // Odd rounds run the baseline first, even rounds the candidate.
            let baseline_first = seq(&b.run_id) < seq(&c.run_id);
            assert_eq!(baseline_first, b.round % 2 == 1, "round {}", b.round);
        }
    }
    assert_verdict(find(&run.arms, "fast"), Verdict::Win);
    assert_verdict(find(&run.arms, "same"), Verdict::Noise);
    assert_verdict(find(&run.arms, "slow"), Verdict::Loss);
    // The warmup round is ledgered, never paired.
    let warm = s
        .ledger
        .read_all()
        .unwrap()
        .into_iter()
        .filter(|r| r["kind"] == "run" && r["warmup"] == true)
        .count();
    assert_eq!(warm, 4, "one discarded warmup per arm");
}

#[test]
fn an_ane_arm_that_never_engaged_is_dirty_not_measured() {
    let dir = scratch();
    let s = session(&dir, "sweep-engagement");
    let base = arm("steel:TURBOSPARK_FAKE_TPS=100");
    // Same speed-up on both: only the engagement line differs. The silent
    // one is what missing Core ML programs look like: the runtime warns
    // once and prefills on Steel under the ANE arm's label.
    let cands = [
        arm("silent:ane_prefill=split,TURBOSPARK_FAKE_TPS=135"),
        arm("split:ane_prefill=split,TURBOSPARK_FAKE_TPS=135,TURBOSPARK_FAKE_ENGAGED=ane_prefill"),
    ];
    let run = run_sweep(
        &s,
        &base,
        &cands,
        &workload(),
        Metric::PrefillTokS,
        0.02,
        3,
        3,
        None,
    )
    .unwrap();
    let silent = find(&run.arms, "silent");
    for (_, c) in &silent.pairs {
        assert!(!c.eligible);
        assert!(
            c.reasons
                .iter()
                .any(|r| r.starts_with("not engaged") && r.contains("ane_prefill")),
            "{:?}",
            c.reasons
        );
    }
    assert_eq!(silent.verdict.pairs, 0);
    assert_eq!(silent.verdict.verdict, Verdict::Inconclusive);
    let split = find(&run.arms, "split");
    for (_, c) in &split.pairs {
        assert!(
            !c.reasons.iter().any(|r| r.starts_with("not engaged")),
            "{:?}",
            c.reasons
        );
    }
    assert_verdict(split, Verdict::Win);
}

#[test]
fn a_refused_candidate_is_dropped_and_the_rest_still_run() {
    let dir = scratch();
    let s = session(&dir, "sweep-refusal");
    let base = arm("base:TURBOSPARK_FAKE_TPS=100");
    let cands = [
        arm("nope:TURBOSPARK_FAKE_REFUSE=not-wired"),
        arm("fast:TURBOSPARK_FAKE_TPS=130"),
    ];
    let run = run_sweep(
        &s,
        &base,
        &cands,
        &workload(),
        Metric::PrefillTokS,
        0.02,
        3,
        3,
        None,
    )
    .unwrap();
    let nope = find(&run.arms, "nope");
    assert_eq!(nope.refused.as_deref(), Some("not-wired"));
    assert!(nope.pairs.is_empty());
    assert_eq!(nope.verdict.verdict, Verdict::Inconclusive);
    assert_eq!(find(&run.arms, "fast").pairs.len(), 3);
}

#[test]
fn a_refused_baseline_stops_the_sweep() {
    let dir = scratch();
    let s = session(&dir, "sweep-refused-base");
    let base = arm("base:TURBOSPARK_FAKE_REFUSE=no-family");
    let cands = [arm("fast:TURBOSPARK_FAKE_TPS=130")];
    let err = run_sweep(
        &s,
        &base,
        &cands,
        &workload(),
        Metric::PrefillTokS,
        0.02,
        3,
        3,
        None,
    )
    .err()
    .unwrap();
    assert!(err.contains("baseline base refused: no-family"), "{err}");
}
