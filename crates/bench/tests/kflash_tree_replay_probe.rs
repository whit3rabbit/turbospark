#![cfg(target_os = "macos")]
//! Would a KFlash-style tree verify beat the chain at the SAME row budget?
//! (`docs/KAPITSA_KERNELS.md`, KFlash section.)
//!
//! A host replay, not a tree verify: the trunk decodes greedily one token at
//! a time, and at every position the MTP head drafts a fresh depth-8 chain
//! whose top-4 per depth is kept. `turbospark_bench::tree_replay` then walks
//! the speculative loop for a chain and for a best-first tree over those
//! lattices. The count is exact (an accepted path is always the reference
//! prefix); the tree's own overhead is not modelled.
//!
//! The recorder and its bias note live in `tree_replay_record.rs`, shared
//! with `tslab tree-replay`, which runs the same check on any install.
//!
//! ```sh
//! TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo \
//!   cargo test -p turbospark-bench --test kflash_tree_replay_probe --release -- --ignored --nocapture
//! ```

use tokenizer::{Message, Role};
use turbospark_bench::protocol::PROTOCOL_CASES;
use turbospark_bench::real_model::open_model_runner_speculative;
use turbospark_bench::tree_replay::{replay, Shape};
use turbospark_bench::tree_replay_record::{plain, record_mtp};

/// Head capacity asked for at open; only `DEPTH` steps are drafted.
const MTP_DEPTH: usize = 16;
/// Draft depth per lattice: the largest block below.
const DEPTH: usize = 8;
/// Candidates kept per depth.
const TOP_K: usize = 4;
/// Proposals per round, i.e. verify rows minus the anchor.
const BLOCKS: [usize; 3] = [2, 4, 8];
const GENERATE: usize = 512;
const SLOTS: usize = 16;

/// Verify cost in decode-steps, the same composed column as
/// `mtp_accept_length_probe.rs` (`docs/MTP.md`). It predates the lazy tape
/// and was optimistic at blocks 4 and 8 against the live sweep.
fn break_even(block: usize) -> f64 {
    match block {
        2 => 1.71,
        4 => 2.86,
        _ => 4.53,
    }
}

fn render(tokenizer: &tokenizer::MfTokenizer, content: &str) -> Vec<i32> {
    let rendered = tokenizer
        .apply_chat_template(&[Message::new(Role::User, content)])
        .expect("chat template renders");
    tokenizer.encode(&rendered, false)
}

#[test]
#[ignore = "needs a real MTP install via TURBOSPARK_MTP_INSTALL_DIR"]
fn a_best_first_tree_against_the_chain_at_equal_rows() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("TURBOSPARK_MTP_INSTALL_DIR").expect("TURBOSPARK_MTP_INSTALL_DIR"),
    );
    let (mut runner, tokenizer) = open_model_runner_speculative(
        &dir,
        SLOTS,
        runtime::DraftPolicies::mtp(runtime::MtpDraftPolicy::Fixed(MTP_DEPTH)),
    )
    .expect("install opens");

    let mut cases: Vec<(&str, Vec<i32>)> = PROTOCOL_CASES[..2]
        .iter()
        .map(|c| (c.id, render(&tokenizer, c.content)))
        .collect();
    cases.push((
        "code",
        render(
            &tokenizer,
            "Write a Python function that merges two sorted lists, then explain \
             its time and space complexity in detail.",
        ),
    ));

    println!("\n{GENERATE} greedy tokens per case, depth {DEPTH}, top-{TOP_K} per depth\n");
    println!(
        "case                block  chain c/rd  tree c/rd  oracle c/rd  tree/chain  chain proj  tree proj"
    );
    for (id, prompt) in &cases {
        let reference = plain(&mut runner, prompt, GENERATE).expect("plain decode");
        let (stream, lattices) =
            record_mtp(&mut runner, prompt, GENERATE, DEPTH, TOP_K).expect("record");
        // Drafting must not perturb the trunk: otherwise the lattices were
        // drafted against some other stream.
        assert_eq!(
            reference, stream,
            "{id}: drafting changed the greedy stream"
        );

        // Where the reference token ranks at depth 0, over every position.
        let mut rank = [0usize; TOP_K + 1];
        for (i, l) in lattices.iter().enumerate() {
            let (Some(l), Some(&t)) = (l, stream.get(i + 1)) else {
                continue;
            };
            let r = l.depths[0].tokens.iter().position(|&c| c == t);
            rank[r.unwrap_or(TOP_K)] += 1;
        }

        for block in BLOCKS {
            let chain = replay(&lattices, &stream, block, Shape::Chain);
            let tree = replay(&lattices, &stream, block, Shape::BestFirst);
            let oracle = replay(&lattices, &stream, block, Shape::Oracle);
            let (c, t, o) = (
                chain.committed_per_round(),
                tree.committed_per_round(),
                oracle.committed_per_round(),
            );
            println!(
                "{id:<18}  {block:>5}  {c:>10.3}  {t:>9.3}  {o:>11.3}  {:>9.3}x  {:>9.2}x  {:>8.2}x",
                t / c,
                c / break_even(block),
                t / break_even(block),
            );
        }
        let total: usize = rank.iter().sum();
        let shares: Vec<String> = rank
            .iter()
            .map(|&n| format!("{:.3}", n as f64 / total.max(1) as f64))
            .collect();
        println!(
            "{id:<18}  depth-0 rank of the target (0..{TOP_K}, miss): {}\n",
            shares.join(" ")
        );
    }
    println!(
        "'c/rd' is committed tokens per verify round (accepted + bonus). 'proj' divides by\n\
         the composed break-even column, with NO tree overhead. Siblings are leaves (an MTP\n\
         head cannot draft under a token it was not fed), so a tree gains at most one token\n\
         per round over the chain.\n"
    );
}
