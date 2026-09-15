#![cfg(target_os = "macos")]

//! Can a SYNTHETIC dense `qwen3_5` install see the batched verify's
//! divergence from a sequential decode?
//!
//! On the real install the two part at the THIRD token and not before:
//! `produce_batched` and `produce` are bit-identical at spans 1 and 2 and
//! differ across ~88% of the vocabulary from span 3
//! (`crates/bench/tests/batched_forward_probe.rs`). Everything cheap has been
//! eliminated -- the batched INT4 GEMM is bit-exact against the GEMV on data
//! proven able to see a reassociation, the `gdn.metal` multi-row kernels are
//! bit-exact including the partial-conv-tail transient at exactly that row,
//! and `chunks_for` returns 1 until span 32 so the split-KV combine is not
//! running. What is left needs a per-layer bisect, and a bisect wants a
//! fixture that runs in seconds.
//!
//! **THIS FILE ASKS WHETHER THE FIXTURE CAN SEE IT AT ALL, WHICH IS THE
//! QUESTION TO SETTLE BEFORE BUILDING ANYTHING ON ONE.** A synthetic install
//! is untrained and small, and this repo has been caught twice by fixtures
//! that could not see the property under test -- the GEMM parity fixture sums
//! exactly in FP32 and reads 0 of 64 order-sensitive blocks, and the norm
//! fixture at weights near zero cannot tell `x * w` from `x * (1 + w)`
//! (AGENTS.md Gotchas 48, 50, 51). A green here would mean nothing on its
//! own; it is only informative beside the real install's red.
//!
//! It REPORTS rather than asserts, for that reason: neither outcome is a
//! defect in this fixture. If the step at span 3 reproduces, the bisect gets
//! a harness that costs seconds and whose layer count is a parameter. If it
//! does not, the fixture is blind and the bisect needs the 14 GB install.

use foundation::LogitValue;
use turbospark_repack::build_synthetic_qwen_gdn_dense_install_with_mtp;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
/// INT4. `encode_gemm_any` now also carries 1-bit (15) and 2-bit (16)
/// batched arms (ROADMAP P3.3), so 4 is a choice among supported widths
/// rather than the only one; the onset measurement this file exists for was
/// taken at this width and stays at it.
const BITS: u32 = 4;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-batched-onset-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open(dir: &std::path::Path) -> RealForwardRunner {
    let peeked = turbospark_repack::peek_manifest_arch(dir).expect("manifest peeks");
    RealForwardRunner::open_with_options_and_speculation(
        dir,
        peeked,
        4096,
        16,
        // The M-row scratch is allocated beside a drafter and `produce_batched`
        // refuses by name without it; this never drafts a token.
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(2)),
    )
    .expect("install opens")
}

#[test]
fn a_synthetic_dense_install_is_swept_for_the_span_three_onset() {
    let dir = temp_dir("onset");
    build_synthetic_qwen_gdn_dense_install_with_mtp(&dir, VOCAB, LAYERS, "tiny-qwen35-mtp", BITS)
        .expect("a dense qwen3_5 install WITH an MTP head builds");
    let mut runner = open(&dir);
    let vocab = VOCAB as usize;
    // Ordinary ids; what is compared is arithmetic, and a dense trunk routes
    // nothing that could make one token special.
    let tokens = [5i32, 9, 2, 11, 7, 3];

    println!("synthetic dense qwen3_5, {LAYERS} layers, vocab {vocab}, INT4:");
    let mut first_bad = None;
    for span in 1..=tokens.len() {
        let position = span - 1;

        // Both arms build the SAME history with `produce`, so only the row
        // under test differs -- the shape the real-install sweep uses.
        runner.reset();
        let mut sequential = vec![LogitValue::from_f32(0.0); vocab];
        for (i, &token) in tokens[..span].iter().enumerate() {
            runner
                .produce(token, i, &mut sequential)
                .expect("sequential produce");
        }

        runner.reset();
        let mut batched = vec![LogitValue::from_f32(0.0); vocab];
        for (i, &token) in tokens[..position].iter().enumerate() {
            runner.produce(token, i, &mut batched).expect("history");
        }
        runner
            .produce_batched(&tokens[position..span], position, &mut batched)
            .expect("batched at one row");

        // Finite BEFORE compared: NaN compares as stably as any other bit
        // pattern and would read as perfect agreement (Gotcha 59).
        assert!(
            sequential.iter().all(|v| v.to_f32().is_finite())
                && batched.iter().all(|v| v.to_f32().is_finite()),
            "span {span}: non-finite logits, so the comparison means nothing"
        );
        let differing = sequential
            .iter()
            .zip(batched.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = sequential
            .iter()
            .zip(batched.iter())
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0.0f32, f32::max);
        println!("  span {span}: {differing}/{vocab} logits differ, worst |delta| {worst:e}");
        if differing != 0 && first_bad.is_none() {
            first_bad = Some(span);
        }
    }

    match first_bad {
        Some(span) => println!(
            "\nREPRODUCES at span {span}. This fixture is a bisect harness: it costs \n\
             seconds and its layer count is a parameter, so the first layer at which \n\
             the two residual streams part can be found without the 14 GB install."
        ),
        None => println!(
            "\nDOES NOT REPRODUCE. Bit-identical at every span where the real install \n\
             differs on ~88% of the vocabulary from span 3, so this fixture is BLIND \n\
             to the property -- untrained weights at this scale do not put the two \n\
             paths anywhere near a rounding boundary. The bisect needs the real \n\
             install, and no assertion built on this fixture would have caught the \n\
             divergence (AGENTS.md Gotchas 48, 50, 51)."
        ),
    }

    let _ = std::fs::remove_dir_all(&dir);
}
