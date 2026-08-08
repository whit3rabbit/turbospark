#![cfg(target_os = "macos")]
//! THROWAWAY diagnostic probe for the Q8_0 GGUF non-determinism found in
//! ROADMAP Phase G Stage 2 item 8. Delete once the cause is fixed.
//!
//!   TURBOSPARK_PROBE_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
//!     cargo test -p turbospark-bench --test gguf_nondeterminism_probe --release -- --ignored --nocapture

use runtime::{run_raw_completion, GenerationConfig, RawDecodeProgress, RealForwardRunner};
use selection::ShapingConfig;
use tokenizer::{Message, MfTokenizer, Role};
use turbospark_bench::real_model::open_model_runner;

const RUNS: usize = 6;
const MAX_NEW: u32 = 48;
const MAX_CONTEXT: u32 = 4096;

fn one(runner: &mut RealForwardRunner, tokenizer: &MfTokenizer, prompt_ids: &[i32]) -> String {
    let config = GenerationConfig {
        shaping: ShapingConfig::new(0.0, 1, None, 1.0, None).expect("greedy"),
        max_new_tokens: MAX_NEW,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
    };
    let mut text = String::new();
    run_raw_completion(
        runner,
        tokenizer,
        prompt_ids,
        &config,
        MAX_CONTEXT,
        tokenizer.vocab_size,
        |event| match event {
            RawDecodeProgress::Token { delta, .. } => text.push_str(&delta),
            RawDecodeProgress::Tail(tail) => text.push_str(&tail),
            RawDecodeProgress::Prefill { .. } => {}
        },
    )
    .expect("generation runs");
    text
}

fn first_diff(a: &str, b: &str) -> Option<usize> {
    a.bytes().zip(b.bytes()).position(|(x, y)| x != y).or({
        if a.len() == b.len() {
            None
        } else {
            Some(a.len().min(b.len()))
        }
    })
}

#[test]
#[ignore = "diagnostic probe; needs a real install via TURBOSPARK_PROBE_INSTALL_DIR"]
fn greedy_repeats_identically() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_PROBE_INSTALL_DIR").expect("TURBOSPARK_PROBE_INSTALL_DIR"),
    );
    let slots: usize = std::env::var("TURBOSPARK_PROBE_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    eprintln!("probe: {} at {slots} slots", dir.display());

    let (mut runner, tokenizer) = open_model_runner(&dir, slots).expect("open");
    let messages = [Message::new(
        Role::User,
        "Explain how coastal wetlands reduce flood damage.",
    )];
    let rendered = tokenizer
        .apply_chat_template(&messages)
        .expect("chat template renders");
    let prompt_ids = tokenizer.encode(&rendered, false);

    let mut texts: Vec<String> = Vec::new();
    for i in 0..RUNS {
        let t = one(&mut runner, &tokenizer, &prompt_ids);
        let digest = model_io::hash_data(t.as_bytes());
        let against_first = match texts.first() {
            None => "  (first)".to_string(),
            Some(f) => match first_diff(f, &t) {
                None => "  same as run 0".to_string(),
                Some(at) => format!("  DIFFERS from run 0 at byte {at} of {}", f.len()),
            },
        };
        eprintln!(
            "probe: run {i} digest {} len {}{against_first}",
            &digest[..16],
            t.len()
        );
        texts.push(t);
    }

    let distinct: std::collections::BTreeSet<&String> = texts.iter().collect();
    eprintln!(
        "probe: {} distinct outputs across {RUNS} runs",
        distinct.len()
    );
    assert_eq!(distinct.len(), 1, "greedy generation is not deterministic");
}
