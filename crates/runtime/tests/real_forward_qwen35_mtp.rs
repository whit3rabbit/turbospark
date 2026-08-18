#![cfg(target_os = "macos")]
//! The multi-token-prediction head's draft step, against a synthetic install
//! (`docs/MTP_SPECULATIVE.md`, step 2).
//!
//! **These run in MILLISECONDS and needed no real install to write.**
//! `build_synthetic_qwen_gdn_dense_install_with_mtp` produces a
//! structurally-real head, so the whole draft path was exercised before the
//! 20-minute stream that gives it real weights finished -- which is
//! `crates/repack/CLAUDE.md` Gotcha 8's rule (build the fixture BEFORE the
//! download) applied to the runtime side.
//!
//! # What a synthetic install can and cannot say
//!
//! Its weights are deterministic and untrained, so drafted tokens are
//! meaningless by construction and nothing here may assert that a draft is
//! GOOD. Accept length is `crates/bench`'s question and needs the real
//! install. What these can say is that the step runs, that it reads the
//! HEAD's tensors rather than the trunk's, and that it has not changed --
//! and the last one needs a frozen digest rather than a perturbation.
//!
//! **AGENTS.md Gotcha 51 is why the digest is here from the start.** A file
//! of perturb-and-require-movement cases rebuilds its own baseline inside the
//! mutated binary, so a mutation that changes the MATH for every arm equally
//! leaves every case green: `real_forward_muse.rs` caught ONE of six that way
//! until a frozen digest took it to five. The digest is a CHANGE DETECTOR and
//! not a correctness claim -- untrained weights cannot say the arithmetic is
//! right, only that it is what it was -- so re-freezing it needs a stated
//! reason.
//!
//! # The env var, and why every test here sets the same depth
//!
//! `MFERENCE_MTP_DRAFT` is read once at `open`, and integration tests in one
//! file share a process and its environment. So every test here asks for the
//! same depth rather than toggling it, and the OFF path is covered by the
//! open-time refusal below rather than by unsetting the var mid-run.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_with_mtp,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const BITS: u32 = 1;
/// Every test in this binary asks for the same depth. See the header.
const DEPTH: &str = "2";
/// The change detector. Taken from a run of the test that asserts it, over a
/// deterministic fixture, and frozen. See that test's doc comment before
/// touching this.
/// Re-frozen 2026-08-18 a THIRD time, `7323c04a` -> `4406a9e2`, and this one
/// is the fix that made the head WORK: its norms are CENTERED (`x * (1 + w)`)
/// where this port was reading them plain. The published checkpoint stores
/// the offset-from-unity form for the head's five whole-vector norms while
/// the TRUNK's are plain, which is AGENTS.md Gotcha 50's "one model, two
/// conventions" on a second family. Measured effect on the real install:
/// the true next-next token went from median rank 248,308 of 248,320 to
/// median rank 0, and top-1 agreement from 0/24 to 23/24.
///
/// Re-frozen 2026-08-18 a SECOND time, `4a2e6af3` -> `7323c04a`, because the
/// head's hidden input changed: it is the trunk's POST-final-norm state now,
/// not its residual stream. That is what mlx-vlm's reference drafter is
/// handed (`Qwen3_5Model.__call__` returns `self.norm(h)` and the same value
/// feeds both `hidden_states[-1]` and `lm_head`), so this is a correctness
/// fix read off the reference rather than a tuning choice. NOTE it did NOT
/// fix the real head's anti-alignment -- see docs/MTP_SPECULATIVE.md step 3.
///
/// Re-frozen 2026-08-18 from `f9ead747`, and the reason is the head's KV.
/// `trunk_then_draft` now PRIMES every earlier position, so the draft attends
/// over rows the head actually wrote rather than over rows nobody did -- the
/// block's span is `[0, position]` and the head's cache used to start empty.
/// The step's arithmetic and dispatch order are untouched; what moved is its
/// input. Note every reachability case in this file stayed green across that
/// change, which is Gotcha 51's point restated: this constant was the only
/// thing that could see it.
const FROZEN_DRAFT_DIGEST: &str = "4406a9e2";

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-mtp-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Idempotent, and called by every test rather than once in a fixture: there
/// is no ordering guarantee between tests in a binary, so the var has to be
/// set before whichever one runs first reaches `open`.
fn ask_for_drafts() {
    std::env::set_var("MFERENCE_MTP_DRAFT", DEPTH);
}

fn build_with_head(dir: &std::path::Path) {
    build_synthetic_qwen_gdn_dense_install_with_mtp(dir, VOCAB, LAYERS, "mtp-toy", BITS)
        .expect("a dense qwen3_5 install WITH an MTP head builds");
}

fn open(dir: &std::path::Path) -> RealForwardRunner {
    let peeked = turbospark_repack::peek_manifest_arch(dir).expect("manifest peeks");
    RealForwardRunner::open(dir, peeked).expect("install opens")
}

/// Walks a few trunk tokens, PRIMING the head as it goes, then takes one
/// draft off the last one.
///
/// Two contracts, and the second was added after the first shipped. The ORDER
/// is one: `mtp_draft_step` reads the trunk's hidden state out of
/// `scratch.x`, which the next `produce` overwrites from the embedding, so a
/// draft has to be taken between the trunk's readback and the next token.
///
/// The other is that the head's KV has to COVER the span the draft will
/// attend over. The block attends `[0, position]`, so a draft at `steps - 1`
/// off an empty head reads `steps - 1` rows nobody wrote. Priming every
/// earlier position is what fills them, and it is free of any ordering
/// question because the token at `position + 1` is exactly what the trunk
/// just predicted.
fn trunk_then_draft(runner: &mut RealForwardRunner, steps: usize) -> Vec<f16> {
    runner.reset();
    let mut token = 5i32;
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    for position in 0..steps {
        runner
            .produce(token, position, &mut head)
            .expect("trunk produce succeeds");
        token = argmax(&head);
        // `token` now occupies `position + 1`, which is the head's input pair
        // at `position`. The LAST position's row is written by the draft
        // itself, so it is not primed here.
        if position + 1 < steps {
            runner
                .mtp_prime_step(token, position)
                .expect("prime step succeeds");
        }
    }
    let mut draft = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .mtp_draft_step(token, steps - 1, &mut draft)
        .expect("draft step succeeds");
    draft
}

fn argmax(logits: &[f16]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn digest(logits: &[f16]) -> String {
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    model_io::hash_data(&bytes)[..8].to_string()
}

/// Patches a resident tensor's bytes in place and re-opens.
///
/// In-place rather than rebuilding the fixture, for the reason
/// `gguf_qwen_convention_patch.rs` gives one crate over: tensors sit at fixed
/// offsets, the transform preserves length, and `open()` runs no receipt or
/// SHA-256 check, so a reachability probe costs milliseconds.
fn perturb(dir: &std::path::Path, tensor: &str) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("resident index");
    let e = index
        .entries
        .get(tensor)
        .unwrap_or_else(|| panic!("{tensor} is not resident"));
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("open weights");
    f.seek(SeekFrom::Start(e.file_offset)).unwrap();
    let mut buf = vec![0u8; e.size_bytes as usize];
    f.read_exact(&mut buf).unwrap();
    for b in buf.iter_mut() {
        *b ^= 0x5A;
    }
    f.seek(SeekFrom::Start(e.file_offset)).unwrap();
    f.write_all(&buf).unwrap();
}

// -- The step runs ---------------------------------------------------------

/// The headline: a head-carrying install drafts, and its logits are LOGITS.
#[test]
fn a_draft_step_runs_and_produces_finite_logits() {
    ask_for_drafts();
    let dir = temp_dir("runs");
    build_with_head(&dir);
    let mut runner = open(&dir);
    assert_eq!(
        runner.mtp_draft_depth(),
        DEPTH.parse::<usize>().unwrap(),
        "the resolved depth should be what the env asked for"
    );

    let draft = trunk_then_draft(&mut runner, 4);
    assert!(
        draft.iter().all(|v| v.to_f32().is_finite()),
        "non-finite draft logit"
    );
    // The producer contract, restated for the drafter: LOGITS, never
    // probabilities (crate Gotcha 1). A drafter feeds the same sampler.
    let sum: f32 = draft.iter().map(|v| v.to_f32()).sum();
    assert!(
        draft.iter().any(|v| v.to_f32() < 0.0) || (sum - 1.0).abs() > 1e-2,
        "the draft looks like a normalized distribution, sum {sum}"
    );
}

/// Depth beyond one is the CALLER's loop, and this is the shape of it: feed
/// the drafted token back at the next position. Asserts only that the chain
/// runs and stays finite -- untrained weights make the tokens meaningless.
#[test]
fn a_draft_chain_advances_the_heads_own_kv() {
    ask_for_drafts();
    let dir = temp_dir("chain");
    build_with_head(&dir);
    let mut runner = open(&dir);

    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut token = 5i32;
    for position in 0..3usize {
        runner.produce(token, position, &mut head).expect("produce");
        token = argmax(&head);
        if position + 1 < 3 {
            runner.mtp_prime_step(token, position).expect("prime");
        }
    }
    let mut draft = vec![f16::from_f32(0.0); VOCAB as usize];
    for depth in 0..DEPTH.parse::<usize>().unwrap() {
        runner
            .mtp_draft_step(token, 2 + depth, &mut draft)
            .unwrap_or_else(|e| panic!("draft {depth} failed: {e}"));
        assert!(draft.iter().all(|v| v.to_f32().is_finite()));
        token = argmax(&draft);
    }
    // Each step wrote exactly one row, so the cursor is the last drafted
    // position plus one. A chain that silently stopped advancing would still
    // return finite logits at every depth.
    assert_eq!(
        runner.mtp_kv_position(),
        2 + DEPTH.parse::<usize>().unwrap(),
        "the head's KV should advance once per draft step"
    );
}

// -- The change detector ---------------------------------------------------

/// **A FROZEN DIGEST over a deterministic synthetic install's draft logits.**
///
/// This is the only assertion in this file that compares against something
/// computed BEFORE a mutation, and it is why the file catches arithmetic
/// changes at all (Gotcha 51). It is not a correctness claim.
///
/// Re-freezing it needs a stated reason. Legitimate ones: the fixture's
/// weights change, or the draft step's dispatch order changes in a way whose
/// reduce order legitimately moves. "The test went red" is not one.
#[test]
fn the_draft_logits_have_a_frozen_digest() {
    ask_for_drafts();
    let dir = temp_dir("digest");
    build_with_head(&dir);
    let mut runner = open(&dir);
    let draft = trunk_then_draft(&mut runner, 4);

    // Reproducibility first: the same runner, the same walk, twice. A digest
    // frozen over a nondeterministic value is worse than no digest.
    let again = trunk_then_draft(&mut runner, 4);
    assert_eq!(
        digest(&draft),
        digest(&again),
        "the draft step is not deterministic across two walks in one process"
    );

    println!("draft digest = {}", digest(&draft));
    assert_eq!(
        digest(&draft),
        FROZEN_DRAFT_DIGEST,
        "the draft logits moved; see this test's doc comment before re-freezing"
    );
}

// -- Reachability, which the digest cannot localize -------------------------

/// The draft step reads the HEAD's block, not trunk layer 0's.
///
/// **This is the mutation the plan names** ("point the draft step at trunk
/// layer 0's tensors"), and it is stated as the discriminating pair rather
/// than as one movement: perturbing an `mtp.layers.0.*` tensor must move the
/// draft and must NOT move the trunk. A test that only asserted the first
/// half passes when the draft reads the trunk's layer 0 instead, because
/// that tensor is reachable either way.
#[test]
fn the_draft_reads_the_heads_block_and_the_trunk_does_not() {
    ask_for_drafts();
    let dir = temp_dir("block");
    build_with_head(&dir);

    let (trunk_before, draft_before) = {
        let mut runner = open(&dir);
        runner.reset();
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner.produce(5, 0, &mut head).expect("produce");
        let draft = trunk_then_draft(&mut runner, 4);
        (digest(&head), digest(&draft))
    };

    perturb(&dir, "mtp.layers.0.self_attn.q_proj.weight");

    let mut runner = open(&dir);
    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(5, 0, &mut head).expect("produce");
    let draft = trunk_then_draft(&mut runner, 4);

    assert_eq!(
        trunk_before,
        digest(&head),
        "perturbing an mtp tensor moved the TRUNK's logits; the trunk is \
         reading the head's block"
    );
    assert_ne!(
        draft_before,
        digest(&draft),
        "perturbing the head's q_proj did not move the draft; the draft is \
         not reading the head's block"
    );
}

/// Both pre-`fc` norms are reached, and they are DIFFERENT tensors.
///
/// The plan's third mutation is "drop `pre_fc_norm_*`", and dropping either
/// one alone has to be visible -- so each is perturbed independently. If the
/// step normalized both halves with one tensor, one of these two would pass
/// while the other failed.
#[test]
fn both_pre_fc_norms_are_reached_independently() {
    ask_for_drafts();
    for tensor in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ] {
        let dir = temp_dir("prefc");
        build_with_head(&dir);
        let before = {
            let mut runner = open(&dir);
            digest(&trunk_then_draft(&mut runner, 4))
        };
        perturb(&dir, tensor);
        let mut runner = open(&dir);
        let after = digest(&trunk_then_draft(&mut runner, 4));
        assert_ne!(before, after, "{tensor} is not reached by the draft step");
    }
}

/// `fc` itself is reached. Cheap, and it is the tensor the whole head is
/// built around: everything else could be right and a dropped `fc` would
/// still produce plausible logits, since its input halves are both
/// normalized hidden-width vectors.
#[test]
fn the_fc_projection_is_reached() {
    ask_for_drafts();
    let dir = temp_dir("fc");
    build_with_head(&dir);
    let before = {
        let mut runner = open(&dir);
        digest(&trunk_then_draft(&mut runner, 4))
    };
    perturb(&dir, "mtp.fc.weight");
    let mut runner = open(&dir);
    assert_ne!(
        before,
        digest(&trunk_then_draft(&mut runner, 4)),
        "mtp.fc.weight is not reached"
    );
}

// -- The refusals ----------------------------------------------------------

/// An install with NO head refuses a draft depth AT OPEN, by name.
///
/// Not at the first draft dispatch four layers down, and not as a silent
/// no-op: a caller that asked for speculation and quietly got none would
/// measure the non-speculative engine and report it as the speculative one.
/// That is the failure mode `crates/runtime` Gotcha 14 states for the chunk
/// driver, and it is the same argument here.
#[test]
fn an_install_without_a_head_refuses_a_draft_depth() {
    ask_for_drafts();
    let dir = temp_dir("headless");
    build_synthetic_qwen_gdn_dense_install(&dir, VOCAB, LAYERS, "no-head")
        .expect("a headless dense install builds");

    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("manifest peeks");
    // `err().expect()` rather than `expect_err`: the Ok type is a runner and
    // does not implement Debug.
    let Err(err) = RealForwardRunner::open(&dir, peeked) else {
        panic!("a headless install must refuse a draft depth");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("mtp.fc.weight"),
        "the refusal must name the missing tensor, got: {msg}"
    );
    assert!(
        msg.contains("MFERENCE_MTP_DRAFT"),
        "the refusal must name the knob that asked for it, got: {msg}"
    );
}

// -- The head's KV cursor --------------------------------------------------

/// A step off the head's cursor is REFUSED, in both directions.
///
/// This is the guard the accept-length probe is built on. The block attends
/// `[0, position]` and derives that span from its ARGUMENT, never from the
/// cursor, so a step past the cursor reads rows nobody wrote and a step
/// behind it silently re-drafts history. Both still return finite,
/// plausible-looking logits, which is why neither can be left to be noticed.
#[test]
fn a_step_off_the_heads_cursor_is_refused_in_both_directions() {
    ask_for_drafts();
    let dir = temp_dir("cursor");
    build_with_head(&dir);
    let mut runner = open(&dir);

    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(5, 0, &mut head).expect("produce");
    let token = argmax(&head);
    assert_eq!(runner.mtp_kv_position(), 0, "a reset head covers nothing");

    let mut draft = vec![f16::from_f32(0.0); VOCAB as usize];
    // AHEAD: position 3 against a cursor of 0 would attend over three
    // unwritten rows.
    let msg = runner
        .mtp_draft_step(token, 3, &mut draft)
        .expect_err("a step ahead of the cursor must be refused")
        .to_string();
    assert!(
        msg.contains('3') && msg.contains("0"),
        "the refusal must carry both the position and the cursor, got: {msg}"
    );

    // The legal step, which also moves the cursor.
    runner
        .mtp_draft_step(token, 0, &mut draft)
        .expect("a step AT the cursor is the legal one");
    assert_eq!(runner.mtp_kv_position(), 1);

    // BEHIND: position 0 again would rewrite a row the head has moved past.
    runner
        .mtp_draft_step(token, 0, &mut draft)
        .expect_err("a step behind the cursor must be refused");
}

/// The primed rows are READ by a later draft, which is what makes priming
/// worth its dispatches.
///
/// Asserted as a DISCRIMINATING pair rather than as "priming runs": prime the
/// same positions with two different tokens and require the draft to move. A
/// test that only checked the cursor advanced would pass against a step that
/// wrote no row at all.
#[test]
fn the_primed_rows_reach_a_later_draft() {
    ask_for_drafts();
    let dir = temp_dir("primed");
    build_with_head(&dir);
    let mut runner = open(&dir);

    let draft_after_priming_with = |runner: &mut RealForwardRunner, primer: i32| -> String {
        runner.reset();
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        let mut token = 5i32;
        for position in 0..4usize {
            runner.produce(token, position, &mut head).expect("produce");
            token = argmax(&head);
            if position + 1 < 4 {
                // The primer stands in for the token the trunk predicted.
                // Only the head's rows change; the trunk's walk is identical.
                runner.mtp_prime_step(primer, position).expect("prime");
            }
        }
        let mut draft = vec![f16::from_f32(0.0); VOCAB as usize];
        runner.mtp_draft_step(token, 3, &mut draft).expect("draft");
        digest(&draft)
    };

    let a = draft_after_priming_with(&mut runner, 1);
    let b = draft_after_priming_with(&mut runner, 7);
    assert_ne!(
        a, b,
        "the draft is blind to the head's own KV rows: priming wrote nothing a draft reads"
    );
}

/// Rewinding the head, PAIRED with the trunk's rollback, returns a whole
/// speculative round to where it started.
///
/// The two are exercised together because neither is sufficient, and the
/// first draft of this test got that wrong: rewinding the head alone does not
/// reproduce a draft, because `h_t` is not the head's state at all. It is read
/// out of the trunk's `scratch.x`, which the chained draft steps overwrite, so
/// restoring it means replaying the trunk. That asymmetry is the whole reason
/// `rollback` does not simply reach into the head -- they restore different
/// things, and only the caller knows both targets.
#[test]
fn a_rewound_head_and_a_rolled_back_trunk_redraft_the_same_logits() {
    ask_for_drafts();
    let dir = temp_dir("rewind");
    build_with_head(&dir);
    let mut runner = open(&dir);

    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut token = 5i32;
    // Positions 0 and 1, priming the head's row 0 as it goes.
    for position in 0..2usize {
        runner.produce(token, position, &mut head).expect("produce");
        token = argmax(&head);
        if position + 1 < 2 {
            runner.mtp_prime_step(token, position).expect("prime");
        }
    }
    // The block starts here: everything after this point is speculative.
    let point = runner.checkpoint();
    runner.mtp_prime_step(token, 1).expect("prime");
    runner.produce(token, 2, &mut head).expect("produce");
    let at_three = argmax(&head);

    let mut first = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .mtp_draft_step(at_three, 2, &mut first)
        .expect("draft");
    // Walk the head forward as a rejected draft would.
    let mut scratch = vec![f16::from_f32(0.0); VOCAB as usize];
    let mut chained = argmax(&first);
    for depth in 0..2usize {
        runner
            .mtp_draft_step(chained, 3 + depth, &mut scratch)
            .expect("chained draft");
        chained = argmax(&scratch);
    }
    assert_eq!(runner.mtp_kv_position(), 5);

    // Undo the round on BOTH sides and replay it.
    runner.rollback(&point);
    runner.mtp_rewind_to(2).expect("rewind");
    assert_eq!(runner.mtp_kv_position(), 2, "rewind must move the cursor");
    runner.produce(token, 2, &mut head).expect("replay");
    assert_eq!(
        at_three,
        argmax(&head),
        "the trunk did not replay to the same token"
    );

    let mut again = vec![f16::from_f32(0.0); VOCAB as usize];
    runner
        .mtp_draft_step(at_three, 2, &mut again)
        .expect("redraft");
    assert_eq!(
        digest(&first),
        digest(&again),
        "a rewound head did not reproduce the draft it had already taken"
    );

    // A target ahead of the cursor is a caller error, not a silent clamp.
    runner
        .mtp_rewind_to(9)
        .expect_err("rewinding forwards must be refused");
}
