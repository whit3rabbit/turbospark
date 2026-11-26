//! Plumbing gate for the Inkling 2 LoopCD experiment
//! (`families/inkling2/loopcd.rs`), on a real install.
//!
//! ```text
//! INKLING2_INSTALL=~/.turbospark/models/text/inkling2-mini-mlx4.gturbo \
//!     cargo test --release -p turbospark-runtime \
//!     --test real_forward_inkling2_loopcd -- --ignored --nocapture
//! ```
//!
//! With the reference taken at the LAST core pass, every difference is zero,
//! so both variants must equal LoopCD off bit for bit. For `adaptive` that
//! also checks the second run of the final layers: its KV and conv slots must
//! be filled on the headless prefill tokens too, or its logits drift from the
//! main run's. With the default reference the output must change, so an arm
//! that never engaged cannot pass.

use std::path::{Path, PathBuf};

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const PROMPT: usize = 200;
const DECODE: usize = 12;

fn argmax(row: &[f16]) -> i32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .unwrap()
        .0 as i32
}

/// The last prompt row and `DECODE` rows after it, under the given settings.
/// `forced` replays another arm's tokens so every arm sees the same input.
fn rows(
    install: &Path,
    env: &[(&str, &str)],
    ids: &[i32],
    forced: Option<&[i32]>,
) -> (Vec<Vec<f16>>, Vec<i32>) {
    for name in ["", "_OMEGA", "_PASS"] {
        std::env::remove_var(format!("TURBOSPARK_INKLING2_LOOPCD{name}"));
    }
    for (k, v) in env {
        std::env::set_var(k, v);
    }
    let arch = model_io::inkling2_mini();
    let mut row = vec![f16::ZERO; arch.vocab_size as usize];
    let mut runner = RealForwardRunner::open_with_residency(
        install,
        arch,
        512,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies::off(),
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        turbospark_runtime::ExpertResidency::Resident,
    )
    .expect("open resident");
    runner.prefill_chunk(ids, 0, &mut row).expect("prefill");
    let mut out = vec![row.clone()];
    let mut fed = Vec::with_capacity(DECODE);
    let t = std::time::Instant::now();
    for i in 0..DECODE {
        let next = forced.map_or_else(|| argmax(&row), |f| f[i]);
        fed.push(next);
        runner
            .produce(next, ids.len() + i, &mut row)
            .expect("decode");
        out.push(row.clone());
    }
    eprintln!(
        "{env:?}: {:.1} decode tok/s (not a benchmark)",
        DECODE as f64 / t.elapsed().as_secs_f64()
    );
    (out, fed)
}

fn differing_rows(a: &[Vec<f16>], b: &[Vec<f16>]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.iter().zip(*y).any(|(p, q)| p.to_bits() != q.to_bits()))
        .count()
}

#[test]
#[ignore = "needs INKLING2_INSTALL (a real Inkling 2 install, resident experts)"]
fn loopcd_is_inert_at_the_last_pass_and_engages_at_the_first() {
    // LoopCD prefills one token at a time, so the off arm must too: the
    // multi-row prefill is not held bit-exact to it on every install. Set
    // before the first open (read once per process).
    std::env::set_var("TURBOSPARK_INKLING2_CHUNKED_PREFILL", "0");
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/INKLING2_PHASE0.md"),
    )
    .expect("prose");
    let mut ids = tok.encode(&text, false);
    assert!(ids.len() >= PROMPT);
    ids.truncate(PROMPT);

    let (off, fed) = rows(&install, &[], &ids, None);
    const MODE: &str = "TURBOSPARK_INKLING2_LOOPCD";
    const PASS: &str = "TURBOSPARK_INKLING2_LOOPCD_PASS";
    const OMEGA: &str = "TURBOSPARK_INKLING2_LOOPCD_OMEGA";
    for mode in ["hidden", "adaptive"] {
        for env in [[(MODE, mode), (PASS, "3")], [(MODE, mode), (OMEGA, "0")]] {
            let (got, _) = rows(&install, &env, &ids, Some(&fed));
            assert_eq!(differing_rows(&off, &got), 0, "{env:?} changed the output");
        }
        let (got, _) = rows(&install, &[(MODE, mode)], &ids, Some(&fed));
        let changed = differing_rows(&off, &got);
        let flips = off
            .iter()
            .zip(&got)
            .filter(|(a, b)| argmax(a) != argmax(b))
            .count();
        eprintln!(
            "{mode} default: {changed} of {} rows changed, {flips} top-1 flips",
            off.len()
        );
        assert_eq!(changed, off.len(), "{mode} did not engage");
        assert!(got.iter().flatten().all(|v| v.is_finite()));
    }
}
