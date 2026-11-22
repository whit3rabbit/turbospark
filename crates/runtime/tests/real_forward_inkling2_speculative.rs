#![cfg(target_os = "macos")]
//! Greedy DSpark speculation on Inkling 2 against plain greedy decode, through
//! the shipped loops (`run_raw_completion` and
//! `run_raw_completion_speculative`), on a real install.
//!
//! ```text
//! INKLING2_INSTALL=~/models/inkling2-mini-gturbo \
//! TURBOSPARK_INKLING2_DSPARK=~/models/mlx/inkling2-mini-oq4e-mtp \
//! TURBOSPARK_SPEC_STATS=1 cargo test --release -p turbospark-runtime \
//!     --test real_forward_inkling2_speculative -- --ignored --nocapture
//! ```
//!
//! The committed token streams must be identical. A difference would have to
//! come from a verify row that differs from the sequential row, which the
//! verify gate (`real_forward_inkling2_verify.rs`) holds bit-exact, so this is
//! the end-to-end check of the loop wiring: drafter cursor, retaining
//! rollback, bonus token, and stop handling. Timings printed here come from
//! one process on whatever host ran it and are not protocol measurements.

use std::path::PathBuf;

use turbospark_runtime::{
    run_raw_completion, run_raw_completion_speculative, GenerationConfig, LogitProducer,
    RealForwardRunner,
};

fn prompts() -> Vec<(&'static str, String)> {
    let long = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/INKLING2_PHASE0.md"),
    )
    .expect("long prompt");
    // About 2,500 tokens of the doc, then a question about it.
    let cut: String = long.chars().take(9000).collect();
    vec![
        (
            "prose",
            "Write a short story, about 300 words, about a lighthouse keeper who finds a \
             mysterious letter washed up on the shore after a storm."
                .to_string(),
        ),
        (
            "code",
            "Write a Rust function that parses a CSV line into a Vec<String>, handling quoted \
             fields with embedded commas and escaped double quotes. Include unit tests."
                .to_string(),
        ),
        (
            "long",
            format!("{cut}\n\nSummarize the document above in five bullet points."),
        ),
    ]
}

#[test]
#[ignore = "needs INKLING2_INSTALL and TURBOSPARK_INKLING2_DSPARK"]
fn greedy_speculation_commits_the_plain_greedy_stream() {
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let max_ctx = 4096u32;
    let mut runner = RealForwardRunner::open_with_residency(
        &install,
        arch.clone(),
        max_ctx as usize,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies {
            mtp: turbospark_runtime::MtpDraftPolicy::Off,
            dflash: turbospark_runtime::DflashDraftPolicy::Fixed(7),
        },
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        turbospark_runtime::ExpertResidency::Resident,
    )
    .expect("open with the DSpark drafter");
    assert_eq!(runner.dflash_speculation_blocker(), None);
    let n_new = std::env::var("INKLING2_SPEC_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256u32);
    let config = GenerationConfig {
        shaping: selection::ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: n_new,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };
    assert!(
        config.shaping.is_deterministic(),
        "the default shaping is greedy"
    );
    let mut failures = Vec::new();
    for (kind, text) in prompts() {
        let ids = tok.encode_text_continuation(&text);
        runner.reset();
        let plain = run_raw_completion(&mut runner, &tok, &ids, &config, max_ctx, vocab, |_| {})
            .expect("plain");
        runner.reset();
        let spec = run_raw_completion_speculative(
            &mut runner,
            &tok,
            &ids,
            &config,
            max_ctx,
            vocab,
            7,
            |_| {},
        )
        .expect("speculative");
        let a = &plain.kv_backed_token_ids[ids.len()..];
        let b = &spec.kv_backed_token_ids[ids.len()..];
        let same = a.iter().zip(b).take_while(|(x, y)| x == y).count();
        eprintln!(
            "{kind}: prompt {} tokens; plain {} new at {:.1} tok/s, speculative {} new at {:.1} \
             tok/s; first {same} committed tokens identical (plain {} / spec {} kv rows)",
            ids.len(),
            plain.new_tokens,
            plain.new_tokens as f64 / plain.decode_seconds,
            spec.new_tokens,
            spec.new_tokens as f64 / spec.decode_seconds,
            a.len(),
            b.len()
        );
        if a != b || plain.new_tokens != spec.new_tokens {
            failures.push(format!("{kind}: streams diverge at generated token {same}"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
