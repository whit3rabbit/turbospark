#![cfg(target_os = "macos")]
//! The `TURBOSPARK_FLINT_CHUNKED_PREFILL=0` opt-out. The layer-batched driver
//! is the default (`crates/lab/families/flint.json`, finding
//! `flint.chunked-prefill-clean-2026-10-28`); this file pins the restored walk
//! to the same bar as its default-on twin
//! (`real_forward_flint_prefill.rs`): byte-identity with `produce`.
//!
//! A test binary is its own process, so the `OnceLock` behind the switch is
//! fresh here and the `=0` is safe to set before any runner opens.

use std::sync::Once;

use half::f16;
use turbospark_repack::build_synthetic_flint_install;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

static SWITCH: Once = Once::new();

fn switch_off() {
    SWITCH.call_once(|| std::env::set_var("TURBOSPARK_FLINT_CHUNKED_PREFILL", "0"));
}

const VOCAB: usize = 64;
const PROMPT_LEN: usize = 40;

fn prompt() -> Vec<i32> {
    (0..PROMPT_LEN)
        .map(|i| ((i * 7 + 3) % VOCAB) as i32)
        .collect()
}

fn open_runner(tag: &str, slots: usize) -> RealForwardRunner {
    switch_off();
    let dir = std::env::temp_dir().join(format!(
        "turbospark-flint-walk-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let arch = build_synthetic_flint_install(&dir, VOCAB as i64, "flint-toy").expect("install");
    RealForwardRunner::open_with_options(&dir, arch, 4096, slots).expect("a flint install opens")
}

fn bits(logits: &[f16]) -> Vec<u16> {
    logits.iter().map(|v| v.to_bits()).collect()
}

/// The reference: every prompt token through `produce_prefill` but the last,
/// which goes through `produce`, exactly as `run_raw_completion` does.
fn sequential_prefill(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<u16> {
    runner.reset();
    let mut logits = vec![f16::from_f32(0.0); VOCAB];
    let last = tokens.len() - 1;
    for (position, &token) in tokens.iter().enumerate() {
        if position == last {
            runner
                .produce(token, position, &mut logits)
                .expect("produce");
        } else {
            runner
                .produce_prefill(token, position, &mut logits)
                .expect("produce_prefill");
        }
    }
    bits(&logits)
}

/// The chunk entry point with the opt-out set must be the walk the old
/// default ran, at every chunk split (a whole-prompt chunk included: the
/// sequential path ignores the split).
#[test]
fn the_walk_opt_out_still_matches_sequential_produce() {
    let tokens = prompt();
    for slots in [4usize, 8] {
        let mut reference = open_runner(&format!("ref{slots}"), slots);
        let want = sequential_prefill(&mut reference, &tokens);

        for chunk in [PROMPT_LEN, 16, 13, 1] {
            let mut runner = open_runner(&format!("c{chunk}s{slots}"), slots);
            runner.reset();
            let mut logits = vec![f16::from_f32(0.0); VOCAB];
            let mut offset = 0;
            while offset < tokens.len() {
                let take = (tokens.len() - offset).min(chunk);
                runner
                    .prefill_chunk(&tokens[offset..offset + take], offset, &mut logits)
                    .expect("prefill_chunk");
                offset += take;
            }
            assert_eq!(
                bits(&logits),
                want,
                "chunk {chunk}, {slots} slots: the =0 opt-out no longer walks"
            );
        }
    }
}
