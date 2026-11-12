#![cfg(target_os = "macos")]
//! The DFlash2 tree verify (on by default, `TURBOSPARK_DFLASH_TREE=0` keeps
//! the chain) against the chain,
//! on the synthetic dflash install. Untrained weights, so nothing here says
//! a tree accepts more; what these pin is the MACHINERY's losslessness:
//!
//! 1. a tree shaped like a chain verifies, compacts and commits byte for
//!    byte like the chain round (the strongest form: same logits, same next
//!    step, same next draft);
//! 2. every slot of a branching tree carries the chain verify's logits over
//!    that slot's root path;
//! 3. a branching round whose accepted path is not `0..keep` leaves the
//!    trunk and the drafter exactly where sequential decode of that path
//!    leaves them, which is what the compaction is for.
//!
//! The switch is read once per process, so every test here clears it before
//! the first runner opens; the binary runs the default, which is trees.
//! `real_forward_qwen35_dflash_tree_off.rs` holds the opt-out.

use std::sync::atomic::{AtomicU64, Ordering};

use foundation::LogitValue;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install_with_dflash_wide_heads, peek_manifest_arch,
};
use turbospark_runtime::{
    DflashDraftPolicy, DraftPolicies, LogitProducer, MtpDraftPolicy, RealForwardRunner,
    SpeculativeProducer,
};

const VOCAB: i64 = 248_100;
const LAYERS: i64 = 64;
const BLOCK: usize = 7;
const PROMPT: [i32; 9] = [5, 7, 9, 11, 13, 15, 17, 19, 21];
/// The depth-7 tree of `tree_verify_cost_bench.rs`, as slot parents.
const BRANCHING: [i32; 8] = [-1, 0, 1, 2, 3, 0, 5, 2];
/// One token per slot: the anchor, then the nodes.
const FEED: [i32; 8] = [31, 33, 35, 37, 39, 41, 43, 45];
const NEXT: i32 = 47;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn install() -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("turbospark-dflash-tree-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    build_synthetic_qwen_gdn_dense_install_with_dflash_wide_heads(
        &dir,
        VOCAB,
        LAYERS,
        "dflash-tree",
        4,
    )
    .expect("the dense install with a DFlash2 drafter writes");
    dir
}

fn open(dir: &std::path::Path) -> RealForwardRunner {
    // SAFETY (edition 2021: not unsafe): every test clears the switch
    // before any runner reads it, so the binary tests the default.
    std::env::remove_var("TURBOSPARK_DFLASH_TREE");
    let arch = peek_manifest_arch(dir).expect("manifest peeks");
    RealForwardRunner::open_with_options_and_speculation(
        dir,
        arch,
        4096,
        16,
        DraftPolicies {
            mtp: MtpDraftPolicy::Off,
            dflash: DflashDraftPolicy::Fixed(BLOCK),
        },
    )
    .expect("the dflash install opens")
}

/// Whether this process's switches leave the tree verifiable: the lazy
/// GDN tape, the GQA-shared attention and the baked GDN key dim. Under any
/// other setting the runner must refuse trees, and the rest of this suite
/// has nothing to run.
fn tree_available() -> bool {
    let var = |k: &str| std::env::var(k).ok();
    var("TURBOSPARK_GDN_LAZY_TAPE").as_deref() != Some("0")
        && var("TURBOSPARK_GDN_DK128").as_deref() != Some("0")
        && var("TURBOSPARK_ATTN_DECODE_KERNEL").as_deref() != Some("barrier")
}

fn vocab() -> usize {
    VOCAB as usize
}

/// The prompt through the token walk, every position primed (an anchor
/// follows the last), so the drafter's cursor sits at the round's base.
fn prefill(runner: &mut RealForwardRunner) -> usize {
    let mut logits = vec![LogitValue::from_f32(0.0); vocab()];
    for (i, &t) in PROMPT.iter().enumerate() {
        runner.produce(t, i, &mut logits).expect("produce");
        runner.prime_drafter(0, i).expect("prime");
    }
    PROMPT.len()
}

fn bits(v: &[LogitValue]) -> Vec<u16> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// What the state after a round decides: the next draft's logits (every
/// query row) and the trunk's next step.
fn after_round(runner: &mut RealForwardRunner, at: usize) -> (Vec<u16>, Vec<u16>) {
    let mut proposals = Vec::new();
    runner
        .dflash_draft_block(NEXT, at, &mut proposals)
        .expect("the next round drafts");
    let mut row = vec![LogitValue::from_f32(0.0); vocab()];
    let mut draft = Vec::new();
    for r in 0..=BLOCK {
        runner.dflash_probe_logits(r, &mut row).expect("probe");
        draft.extend(bits(&row));
    }
    let mut logits = vec![LogitValue::from_f32(0.0); vocab()];
    runner.produce(NEXT, at, &mut logits).expect("next step");
    (draft, bits(&logits))
}

/// A tree round's tail as the loop runs it: compact, roll back to `keep`
/// rows, fold, rewind the drafter.
fn finish_round(
    runner: &mut RealForwardRunner,
    point: &turbospark_runtime::RollbackPoint,
    base: usize,
    path: &[u32],
    tree: bool,
) {
    let keep = path.len();
    if tree {
        runner.dflash_tree_compact(base, path).expect("compact");
    }
    if keep < FEED.len() {
        SpeculativeProducer::rollback_retaining(runner, point, keep).expect("retaining rollback");
    }
    SpeculativeProducer::commit_pending(runner, keep).expect("commit");
    SpeculativeProducer::rewind_drafter(runner, base + keep - 1).expect("rewind");
}

#[test]
fn the_runner_offers_tree_rounds_by_default() {
    let dir = install();
    let runner = open(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    let want = tree_available().then_some(BLOCK);
    assert_eq!(runner.dflash_tree_nodes(), want);
}

#[test]
fn a_chain_shaped_tree_round_is_the_chain_round() {
    if !tree_available() {
        return;
    }
    let dir = install();
    let mut chain = open(&dir);
    let mut tree = open(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    let base = prefill(&mut chain);
    prefill(&mut tree);
    let parents: Vec<i32> = (0..FEED.len() as i32).map(|s| s - 1).collect();
    let mut a = vec![LogitValue::from_f32(0.0); FEED.len() * vocab()];
    let mut b = a.clone();
    let pa = SpeculativeProducer::checkpoint(&mut chain);
    SpeculativeProducer::verify(&mut chain, &FEED, base, &mut a).expect("chain verify");
    let pb = SpeculativeProducer::checkpoint(&mut tree);
    tree.dflash_verify_tree(&FEED, base, &parents, &mut b)
        .expect("tree verify");
    assert!(bits(&a) == bits(&b), "the verify logits differ");
    // A partial round: three rows kept, so the retaining rollback runs.
    let keep = 3usize;
    let path: Vec<u32> = (0..keep as u32).collect();
    finish_round(&mut chain, &pa, base, &path, false);
    finish_round(&mut tree, &pb, base, &path, true);
    let want = after_round(&mut chain, base + keep);
    let got = after_round(&mut tree, base + keep);
    assert!(got.0 == want.0, "the next draft differs");
    assert!(got.1 == want.1, "the next trunk step differs");
}

fn root_path(slot: usize) -> Vec<usize> {
    let mut p = vec![slot];
    let mut at = BRANCHING[slot];
    while at >= 0 {
        p.push(at as usize);
        at = BRANCHING[at as usize];
    }
    p.reverse();
    p
}

#[test]
fn each_tree_slot_is_the_chain_verify_over_its_root_path() {
    if !tree_available() {
        return;
    }
    let dir = install();
    let mut tree = open(&dir);
    let base = prefill(&mut tree);
    let mut got = vec![LogitValue::from_f32(0.0); FEED.len() * vocab()];
    tree.dflash_verify_tree(&FEED, base, &BRANCHING, &mut got)
        .expect("tree verify");
    drop(tree);
    for slot in 0..FEED.len() {
        let path = root_path(slot);
        let feed: Vec<i32> = path.iter().map(|&s| FEED[s]).collect();
        let mut chain = open(&dir);
        prefill(&mut chain);
        let mut want = vec![LogitValue::from_f32(0.0); feed.len() * vocab()];
        SpeculativeProducer::verify(&mut chain, &feed, base, &mut want).expect("chain verify");
        let last = &want[(feed.len() - 1) * vocab()..];
        let row = &got[slot * vocab()..(slot + 1) * vocab()];
        assert!(bits(row) == bits(last), "slot {slot} on path {path:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The accepted path `[0, 5, 6]` is not `0..3`: without the compaction the
/// fold, the KV and the drafter's next context write would read slots 1
/// and 2, which belong to the other branch.
#[test]
fn a_branching_round_commits_what_sequential_decode_would() {
    if !tree_available() {
        return;
    }
    let dir = install();
    let mut tree = open(&dir);
    let mut seq = open(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    let base = prefill(&mut tree);
    prefill(&mut seq);
    let path = [0u32, 5, 6];

    let mut logits = vec![LogitValue::from_f32(0.0); FEED.len() * vocab()];
    let point = SpeculativeProducer::checkpoint(&mut tree);
    tree.dflash_verify_tree(&FEED, base, &BRANCHING, &mut logits)
        .expect("tree verify");
    finish_round(&mut tree, &point, base, &path, true);

    let mut step = vec![LogitValue::from_f32(0.0); vocab()];
    for (i, &slot) in path.iter().enumerate() {
        seq.produce(FEED[slot as usize], base + i, &mut step)
            .expect("sequential step");
        if i + 1 < path.len() {
            seq.prime_drafter(0, base + i).expect("prime");
        }
    }
    let want = after_round(&mut seq, base + path.len());
    let got = after_round(&mut tree, base + path.len());
    assert!(
        got.0 == want.0,
        "the next draft differs from sequential decode"
    );
    assert!(
        got.1 == want.1,
        "the next trunk step differs from sequential decode"
    );
}

fn argmax(row: &[LogitValue]) -> i32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, v) in row.iter().enumerate() {
        if v.to_f32() > best.1 {
            best = (i, v.to_f32());
        }
    }
    best.0 as i32
}

/// Teacher-forced tree rounds: each round drafts (for the drafter's context
/// write), then verifies a tree whose off-spine branch `0 -> 5 -> 6` carries
/// the sequential stream's next two tokens and whose spine is wrong, so
/// every round accepts a non-identity path, compacts it, and commits three
/// tokens. Untrained weights cannot make a drafter branch like this; the
/// planted tree does, and the committed stream must still be the sequential
/// one round after round.
#[test]
fn teacher_forced_branching_rounds_commit_the_sequential_stream() {
    if !tree_available() {
        return;
    }
    let dir = install();
    let mut tree = open(&dir);
    let mut seq = open(&dir);
    let _ = std::fs::remove_dir_all(&dir);

    // The sequential reference: the prompt, then greedy decode.
    let mut logits = vec![LogitValue::from_f32(0.0); vocab()];
    for (i, &t) in PROMPT.iter().enumerate() {
        seq.produce(t, i, &mut logits).expect("reference prompt");
    }
    let mut reference = vec![argmax(&logits)];
    while reference.len() < 16 {
        let at = PROMPT.len() + reference.len() - 1;
        seq.produce(*reference.last().unwrap(), at, &mut logits)
            .expect("reference step");
        reference.push(argmax(&logits));
    }

    // The tree walk, prefilled as the loop does (the last prompt row is left
    // for the first draft's context write).
    for (i, &t) in PROMPT.iter().enumerate() {
        tree.produce(t, i, &mut logits).expect("prompt");
        if i + 1 < PROMPT.len() {
            tree.prime_drafter(PROMPT[i + 1], i).expect("prime");
        }
    }
    let mut next = argmax(&logits);
    let mut committed: Vec<i32> = Vec::new();
    let mut batch = vec![LogitValue::from_f32(0.0); FEED.len() * vocab()];
    while committed.len() + 3 < reference.len() {
        committed.push(next);
        let k = committed.len() - 1;
        let base = PROMPT.len() + k;
        tree.dflash_draft_tree(next, base, BLOCK)
            .expect("the tree draft runs");
        let wrong = |t: i32| (t + 1) % VOCAB as i32;
        let mut feed = FEED;
        feed[0] = next;
        feed[1] = wrong(reference[k + 1]);
        feed[5] = reference[k + 1];
        feed[6] = reference[k + 2];
        let point = SpeculativeProducer::checkpoint(&mut tree);
        tree.dflash_verify_tree(&feed, base, &BRANCHING, &mut batch)
            .expect("tree verify");
        let row = |s: usize| argmax(&batch[s * vocab()..(s + 1) * vocab()]);
        // The walk: slot 0's pick must be slot 5's token, slot 5's slot 6's.
        assert_eq!(
            row(0),
            feed[5],
            "round at {base}: slot 0 picks the planted branch"
        );
        assert_eq!(row(5), feed[6], "round at {base}: slot 5 picks slot 6");
        let path = [0u32, 5, 6];
        finish_round(&mut tree, &point, base, &path, true);
        committed.extend_from_slice(&[feed[5], feed[6]]);
        next = row(6);
    }
    assert_eq!(committed, reference[..committed.len()]);
    assert_eq!(next, reference[committed.len()], "the last bonus");
}

/// The lazily scored tree is the builder over the research lattice (the
/// all-pairs scores `dflash_lattice` reads back from the same pass), and
/// that lattice's greedy path is the selector's chain: so the tree's
/// forced-greedy expansion is `dflash_select`'s proposals.
#[test]
fn the_tree_selector_scores_what_the_chain_selector_scores() {
    let dir = install();
    let mut a = open(&dir);
    let mut b = open(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    let base = prefill(&mut a);
    prefill(&mut b);
    let (tree, tokens) = a.dflash_draft_tree(NEXT, base, BLOCK).expect("tree draft");
    let lattice = a.dflash_lattice(NEXT).expect("lattice of the same pass");
    let k = lattice.width();
    let rebuilt = turbospark_runtime::dflash_best_first_tree(BLOCK, BLOCK, |d, p| match p {
        None => lattice.edges[d].clone(),
        Some(n) => lattice.edges[d][n.cand * k..(n.cand + 1) * k].to_vec(),
    });
    assert_eq!(tree, rebuilt);
    let lattice_tokens: Vec<i32> = tree
        .iter()
        .map(|n| lattice.candidates[n.depth][n.cand])
        .collect();
    assert_eq!(tokens, lattice_tokens);
    let mut proposals = Vec::new();
    b.dflash_draft_block(NEXT, base, &mut proposals)
        .expect("chain draft");
    let greedy: Vec<i32> = lattice
        .greedy_path()
        .iter()
        .enumerate()
        .map(|(d, &c)| lattice.candidates[d][c])
        .collect();
    assert_eq!(proposals, greedy);
}
