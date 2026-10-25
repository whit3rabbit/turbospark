#![cfg(target_os = "macos")]
//! Time to first token on a long multi-turn chat, for the GDN turn snapshot
//! (`TURBOSPARK_GDN_TURN_SNAPSHOT`, `crates/runtime/src/turn_snapshot.rs`).
//!
//! One process is one arm, because the switch is read once. A driver runs
//! the arms in alternating processes and compares them
//! (`docs/KAPITSA_KERNELS.md` lever 4 has the protocol and the numbers).
//!
//! The chat: turn 1 carries a long document (`TURBOSPARK_TTFT_DOC`, default
//! `docs/MTP.md`, cut to `TURBOSPARK_TTFT_DOC_TOKENS`, default 6000) and a
//! question, then four short follow-ups. Each reply goes back into the
//! history without its reasoning, the case the snapshot serves. Every turn
//! prints one `TTFT {json}` line: wall time from the generation call to the
//! first token's callback (so it includes the restore), the loop's own
//! prefill seconds, and the tokens reused.
//!
//! A warm pass of the turn-1 prompt with reuse off runs first and is not
//! reported, so Metal pipeline compiles and the first pass's low GPU clocks
//! land on neither arm.
//!
//! ```sh
//! TURBOSPARK_GDN_TURN_SNAPSHOT=1 \
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test turn_snapshot_ttft_probe --release -- --ignored --nocapture
//! ```

use std::time::Instant;

use runtime::{
    run_raw_completion_speculative, DraftPolicies, GenerationConfig, MtpDraftPolicy, RateControl,
    RawDecodeProgress, RealForwardRunner,
};
use selection::ShapingConfig;
use tokenizer::{Message, MfTokenizer, Role};

const MTP_DEPTH: usize = 3;
const BLOCK: usize = 2;
const SLOTS: usize = 16;
const GENERATE: u32 = 32;
const CONTEXT: usize = 16384;

const FOLLOW_UPS: [&str; 4] = [
    "Summarize the main idea in two sentences.",
    "Which measurement in it would you double-check first, and why?",
    "List the risks it names, briefly.",
    "What would you change about the design?",
];

fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        shaping: ShapingConfig::new(0.0, 0, None, 1.0, None).unwrap(),
        max_new_tokens: GENERATE,
        stop_strings: Vec::new(),
        extra_stop_tokens: Vec::new(),
        rate: RateControl::default(),
    }
}

fn visible_content(text: &str) -> String {
    text.rsplit_once("</think>")
        .map(|(_, after)| after.trim().to_string())
        .unwrap_or_default()
}

/// The document cut to at most `max_tokens` tokens, on a token boundary.
fn document(tokenizer: &MfTokenizer) -> String {
    let path = std::env::var("TURBOSPARK_TTFT_DOC")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/MTP.md").to_string());
    let max_tokens: usize = std::env::var("TURBOSPARK_TTFT_DOC_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6000);
    let text = std::fs::read_to_string(&path).expect("document reads");
    let ids = tokenizer.encode(&text, false);
    tokenizer.decode(&ids[..ids.len().min(max_tokens)], false)
}

struct Turn {
    tokens: Vec<i32>,
    text: String,
    ttft_s: f64,
    prefill_s: f64,
    reused: usize,
}

fn run_turn(runner: &mut RealForwardRunner, tokenizer: &MfTokenizer, prompt: &[i32]) -> Turn {
    let vocab = runner.vocab_size();
    let started = Instant::now();
    let mut first: Option<f64> = None;
    let mut events = Vec::new();
    let result = run_raw_completion_speculative(
        runner,
        tokenizer,
        prompt,
        &greedy_config(),
        CONTEXT as u32,
        vocab,
        BLOCK,
        |e| {
            if first.is_none() && matches!(e, RawDecodeProgress::Token { .. }) {
                first = Some(started.elapsed().as_secs_f64());
            }
            events.push(e);
        },
    )
    .expect("generation runs");
    let mut tokens = Vec::new();
    let mut text = String::new();
    for e in &events {
        match e {
            RawDecodeProgress::Token { id, delta, .. } => {
                tokens.push(*id);
                text.push_str(delta);
            }
            RawDecodeProgress::Tail(t) => text.push_str(t),
            _ => {}
        }
    }
    Turn {
        tokens,
        text,
        ttft_s: first.expect("at least one token"),
        prefill_s: result.prefill_seconds,
        reused: result.reused_prefix_tokens,
    }
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn ttft_over_a_long_chat() {
    let snapshot_on = std::env::var("TURBOSPARK_GDN_TURN_SNAPSHOT").as_deref() == Ok("1");
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let arch = repack::peek_manifest_arch(&dir).expect("manifest reads");
    let tokenizer = MfTokenizer::load_from_dir(&dir).expect("tokenizer loads");
    let mut runner = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        arch,
        CONTEXT,
        SLOTS,
        DraftPolicies::mtp(MtpDraftPolicy::Fixed(MTP_DEPTH)),
    )
    .expect("install opens");

    let doc = document(&tokenizer);
    let mut history = vec![Message::new(
        Role::User,
        format!("{doc}\n\nWhat problem does the document above solve?"),
    )];
    let render = |history: &[Message]| {
        tokenizer.encode(
            &tokenizer
                .apply_chat_template(history)
                .expect("chat template renders"),
            false,
        )
    };

    // Warm pass, unreported, reuse off.
    runner.set_prefix_reuse(false);
    let _ = run_turn(&mut runner, &tokenizer, &render(&history));

    runner.set_prefix_reuse(true);
    for turn in 0..=FOLLOW_UPS.len() {
        if turn > 0 {
            history.push(Message::new(Role::User, FOLLOW_UPS[turn - 1]));
        }
        let prompt = render(&history);
        let t = run_turn(&mut runner, &tokenizer, &prompt);
        println!(
            "TTFT {}",
            serde_json::json!({
                "arm": if snapshot_on { "on" } else { "off" },
                "turn": turn + 1,
                "prompt_tokens": prompt.len(),
                "reused": t.reused,
                "ttft_s": t.ttft_s,
                "prefill_s": t.prefill_s,
                "new_tokens": t.tokens.len(),
                "token_digest": format!("{:016x}", fnv(&t.tokens)),
            })
        );
        history.push(Message::new(Role::Assistant, visible_content(&t.text)));
    }
}

fn fnv(tokens: &[i32]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, t| {
        (h ^ (*t as u32 as u64)).wrapping_mul(0x0100_0000_01b3)
    })
}
