#![cfg(target_os = "macos")]
//! Does a multi-turn chat that continues from the GDN turn snapshot
//! (`TURBOSPARK_GDN_TURN_SNAPSHOT=1`, `crates/runtime/src/turn_snapshot.rs`)
//! produce the tokens a full prefill of the same prompt produces, on the
//! real install with its MTP drafter?
//!
//! The conversation is the case the snapshot exists for: every assistant
//! turn goes back into the history WITHOUT its reasoning, the way most
//! OpenAI-style clients send it, so the new prompt stops agreeing with what
//! was fed just before the previous prompt's end. Without the snapshot a
//! GDN model refuses that rewind and prefills the whole transcript.
//!
//! Two arms on one runner: the CONTINUING arm runs the turns with prefix
//! reuse on; the REFERENCE arm then reruns each of those prompts from zero
//! with reuse off, which is the path every turn takes today. Greedy tokens
//! must match turn for turn.
//!
//! With the switch on, turns after the first must also report reuse. With
//! it off they must report none: the refusal this replaces, and the A arm
//! for a memory comparison of the two processes.
//!
//! ```sh
//! TURBOSPARK_GDN_TURN_SNAPSHOT=1 \
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test turn_snapshot_gate --release -- --ignored --nocapture
//! ```

use runtime::{run_raw_completion_speculative, GenerationConfig, RateControl, RawDecodeProgress};
use selection::ShapingConfig;
use tokenizer::{Message, Role};
use turbospark_bench::real_model::open_model_runner_speculative;

const MTP_DEPTH: usize = 3;
const BLOCK: usize = 2;
const SLOTS: usize = 16;
const GENERATE: u32 = 96;
const CONTEXT: u32 = 4096;

const USER_TURNS: [&str; 3] = [
    "Write a Python function that removes duplicate records from a list of dicts, keyed \
     by their 'id' field. Keep it short.",
    "Change it so that when two records share an id, the one with the later 'updated' \
     timestamp wins.",
    "Now sort the result by 'updated', newest first, and add a docstring.",
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

fn tokens_of(events: &[RawDecodeProgress]) -> Vec<i32> {
    events
        .iter()
        .filter_map(|e| match e {
            RawDecodeProgress::Token { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

fn text_of(events: &[RawDecodeProgress]) -> String {
    events
        .iter()
        .map(|e| match e {
            RawDecodeProgress::Token { delta, .. } => delta.as_str(),
            RawDecodeProgress::Tail(t) => t.as_str(),
            _ => "",
        })
        .collect()
}

/// The reply as a client that drops reasoning sends it back: whatever
/// follows the last `</think>`, or nothing when the budget ran out inside
/// the reasoning.
fn visible_content(text: &str) -> String {
    text.rsplit_once("</think>")
        .map(|(_, after)| after.trim().to_string())
        .unwrap_or_default()
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn a_chat_continuing_from_the_turn_snapshot_matches_a_full_prefill() {
    let snapshot_on = std::env::var("TURBOSPARK_GDN_TURN_SNAPSHOT").as_deref() == Ok("1");
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(MTP_DEPTH)),
    )
    .expect("install opens");
    assert!(
        runner.mtp_draft_depth() > 0,
        "this install carries no MTP head; the gate needs the drafter case"
    );
    let vocab = runner.vocab_size();
    let config = greedy_config();

    // -- The continuing arm.
    runner.set_prefix_reuse(true);
    let mut history: Vec<Message> = Vec::new();
    let mut prompts: Vec<Vec<i32>> = Vec::new();
    let mut continuing: Vec<(Vec<i32>, usize, f64)> = Vec::new();
    for user in USER_TURNS {
        history.push(Message::new(Role::User, user));
        let rendered = tokenizer
            .apply_chat_template(&history)
            .expect("chat template renders");
        let prompt = tokenizer.encode(&rendered, false);
        let mut events = Vec::new();
        let result = run_raw_completion_speculative(
            &mut runner,
            &tokenizer,
            &prompt,
            &config,
            CONTEXT,
            vocab,
            BLOCK,
            |e| events.push(e),
        )
        .expect("continuing run");
        history.push(Message::new(
            Role::Assistant,
            visible_content(&text_of(&events)),
        ));
        continuing.push((
            tokens_of(&events),
            result.reused_prefix_tokens,
            result.prefill_seconds,
        ));
        prompts.push(prompt);
    }

    // -- The reference arm: every prompt from zero, today's path.
    runner.set_prefix_reuse(false);
    println!(
        "\nTURBOSPARK_GDN_TURN_SNAPSHOT={}, block {BLOCK}, {GENERATE} tokens a turn",
        u8::from(snapshot_on)
    );
    for (turn, prompt) in prompts.iter().enumerate() {
        let mut events = Vec::new();
        let result = run_raw_completion_speculative(
            &mut runner,
            &tokenizer,
            prompt,
            &config,
            CONTEXT,
            vocab,
            BLOCK,
            |e| events.push(e),
        )
        .expect("reference run");
        let (tokens, reused, prefill_s) = &continuing[turn];
        println!(
            "turn {}: prompt {} tokens; continuing reused {reused}, prefill {prefill_s:.3}s; \
             reference prefill {:.3}s",
            turn + 1,
            prompt.len(),
            result.prefill_seconds
        );
        assert_eq!(result.reused_prefix_tokens, 0, "the reference arm reused");
        assert_eq!(
            tokens,
            &tokens_of(&events),
            "turn {} continued from the snapshot produced different tokens",
            turn + 1
        );
        if turn == 0 {
            continue;
        }
        let previous = prompts[turn - 1].len();
        if snapshot_on {
            // The point sits a few tokens before the previous prompt's end
            // (`TURN_SNAPSHOT_MARGIN`), and the turns share everything
            // before it.
            assert!(
                *reused > 0 && *reused < previous && previous - *reused <= 16,
                "turn {} reused {reused} of a previous prompt of {previous}; \
                 the snapshot did not serve it",
                turn + 1
            );
        } else {
            assert_eq!(
                *reused,
                0,
                "turn {} reused without the snapshot; the GDN refusal it replaces did not fire",
                turn + 1
            );
        }
    }
    println!("identical tokens on every turn\n");
}
