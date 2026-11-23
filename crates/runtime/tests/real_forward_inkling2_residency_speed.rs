#![cfg(target_os = "macos")]
//! Not a gate: plain greedy decode speed of one Inkling 2 install under one
//! expert residency, to decide what `auto` should pick for an install format.
//! One residency per process (run it once per value, interleaved):
//!
//! ```text
//! INKLING2_INSTALL=<install> INKLING2_RESIDENCY=resident|streamed cargo test --release \
//!     -p turbospark-runtime --test real_forward_inkling2_residency_speed -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use turbospark_runtime::{run_raw_completion, GenerationConfig, LogitProducer, RealForwardRunner};

#[test]
#[ignore = "benchmark: needs INKLING2_INSTALL and INKLING2_RESIDENCY"]
fn plain_decode_speed_under_one_residency() {
    let install = PathBuf::from(std::env::var("INKLING2_INSTALL").expect("INKLING2_INSTALL"));
    let residency = match std::env::var("INKLING2_RESIDENCY").as_deref() {
        Ok("resident") => turbospark_runtime::ExpertResidency::Resident,
        Ok("streamed") => turbospark_runtime::ExpertResidency::Streamed,
        other => panic!("INKLING2_RESIDENCY must be resident or streamed, got {other:?}"),
    };
    let arch = model_io::inkling2_mini();
    let vocab = arch.vocab_size as usize;
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    let mut runner = RealForwardRunner::open_with_residency(
        &install,
        arch,
        4096,
        turbospark_runtime::ExpertCacheSlots::Auto,
        turbospark_runtime::DraftPolicies::off(),
        turbospark_runtime::SteeringPolicy::off(),
        1,
        turbospark_runtime::KvQuant::Off,
        residency,
    )
    .expect("open");
    let config = GenerationConfig {
        shaping: selection::ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: 128,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    };
    let ids = tok.encode_text_continuation(
        "Write a Rust function that parses a CSV line into a Vec<String>, handling quoted \
         fields with embedded commas and escaped double quotes. Include unit tests.",
    );
    // A warm run first: the first pass after open maps buffers and is slow.
    for pass in ["warm", "timed"] {
        runner.reset();
        let r = run_raw_completion(&mut runner, &tok, &ids, &config, 4096, vocab, |_| {})
            .expect("decode");
        eprintln!(
            "{pass}: {residency:?} {} new at {:.1} tok/s (prefill {:.2} s)",
            r.new_tokens,
            r.new_tokens as f64 / r.decode_seconds,
            r.prefill_seconds
        );
    }
}
