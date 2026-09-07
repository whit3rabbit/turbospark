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
//! `TURBOSPARK_MTP_DRAFT` is read once at `open`, and integration tests in one
//! file share a process and its environment. So every test here asks for the
//! same depth rather than toggling it, and the OFF path is covered by the
//! open-time refusal below rather than by unsetting the var mid-run.

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_at_bits,
    build_synthetic_qwen_gdn_dense_install_with_mtp,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner, SpeculativeProducer};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
const BITS: u32 = 1;
/// Every test in this binary asks for the same depth. See the header.
const DEPTH: &str = "2";
/// The change detector. Taken from a run of the test that asserts it, over a
/// deterministic fixture, and frozen. See that test's doc comment before
/// touching this.
/// Re-frozen 2026-08-18 a FOURTH time, `4406a9e2` -> `fd43a56b`, finishing
/// what the third one started. The head's per-head `q_norm`/`k_norm` are
/// centered too and were still being read plainly, because
/// `encode_rms_norm_bf16w_perhead` had no centered sibling and
/// `encode_full_attention_block` resolves those two by NAME for the trunk and
/// the head alike. Now `gpu::encode_rms_norm_bf16w_perhead_centered` exists
/// and the convention is a parameter on that shared function
/// (`QkNormConvention`), so the trunk keeps the plain form and only the head
/// moved. MTPLX's `_RMSNORM_SUFFIXES` lists all seven norms, so this closes
/// the set rather than adding to it; the raw means here read 0.780 and 0.797
/// against its "healthy >= 1.74" threshold. Every reachability case in this
/// file stayed green again, which is the evidence the change is arithmetic
/// inside the head and reaches nothing else.
///
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
///
/// **DEVICE-BRANCHED, since 2026-09-04**; see `real_forward_muse.rs`'s
/// identical fix for the full account (CI's virtualized macOS runner does
/// not reproduce GPU floating-point arithmetic bit-for-bit against real
/// Apple Silicon, and a pure tolerance replacement was proven too weak
/// against `real_forward_qwen35_dflash.rs`'s documented
/// `DFLASH_RESIDUAL_EPS` bug). The final comparison below runs the
/// ORIGINAL exact `digest(&draft) == FROZEN_DRAFT_DIGEST` on real Apple
/// Silicon and falls back to a tolerant comparison against
/// `FROZEN_DRAFT_LOGITS` only on a virtualized device. The reproducibility
/// check two lines below (`digest(&draft) == digest(&again)`, same runner,
/// two walks in one process) stays EXACT unconditionally on purpose: both
/// sides run on the SAME machine in the SAME process, so there is no
/// cross-hardware axis for it to absorb, and weakening it would hide
/// genuine nondeterminism. `FROZEN_DRAFT_LOGITS` is recovered from the same
/// real-hardware run the exact hash (`fd43a56b`) was taken over.
#[rustfmt::skip]
const FROZEN_DRAFT_LOGITS: [f32; VOCAB as usize] = [
    3.5878906, -4.3046875, -2.5898438, 27.8125, -0.8105469, 2.6113281, 23.90625, -11.078125,
    -11.65625, -14.3515625, 1.6289063, -3.2480469, -1.3173828, 11.9140625, -5.0039063, 4.7070313,
    -8.796875, -10.734375, -14.9140625, 4.34375, 20.0, 15.109375, 9.2109375, -7.390625,
    7.6015625, 9.421875, 8.546875, -9.9765625, 7.2460938, 1.9804688, -0.06390381, 0.31689453,
    -1.6572266, -26.890625, 4.875, 1.6074219, -11.9765625, -16.390625, 1.7871094, -12.6640625,
    6.1210938, 18.390625, 14.1484375, -23.234375, -4.7070313, 1.3574219, 10.7734375, 5.890625,
    -0.5463867, -21.703125, 7.8671875, 3.5390625, 10.3203125, -11.6484375, -4.8164063, 2.2246094,
    18.171875, 10.703125, -5.890625, -8.5390625, -1.4609375, -6.046875, 14.09375, -7.296875,
    4.2617188, -10.515625, 5.828125, -6.8125, -0.61816406, 6.4414063, -5.4414063, 0.052215576,
    2.1425781, 15.59375, 6.1367188, -17.828125, 12.7734375, -1.4873047, -14.2109375, 5.4921875,
    0.24963379, -12.859375, 12.09375, 12.078125, -6.671875, -3.1074219, 13.9921875, 6.8828125,
    9.25, 6.671875, -9.734375, -3.7050781, -7.453125, -19.171875, -4.5039063, 7.3164063,
    -26.984375, 8.59375, -15.03125, 8.0234375, 2.890625, 12.125, 4.7070313, 4.0585938,
    -10.203125, -11.5546875, -20.890625, 11.9140625, 10.5390625, 3.7558594, -2.9316406, -11.7734375,
    -10.890625, 10.765625, 3.8085938, 10.328125, 16.015625, -5.34375, 0.5629883, 21.4375,
    11.625, -8.9296875, 1.5722656, 7.8046875, 2.2792969, 8.8671875, 7.625, 4.5,
];

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

/// Kept as a no-op so the call sites still read as "this test wants drafts".
///
/// The depth used to arrive through `TURBOSPARK_MTP_DRAFT` and now arrives as a
/// PARAMETER (`open` below). The env var still works and still means what it
/// did, but an UNSET one is now `Auto` rather than off, and `Auto` resolves to
/// a depth smaller than these tests need -- so passing it explicitly is both
/// clearer and immune to whatever another test in this binary set first.
fn ask_for_drafts() {}

fn build_with_head(dir: &std::path::Path) {
    build_synthetic_qwen_gdn_dense_install_with_mtp(dir, VOCAB, LAYERS, "mtp-toy", BITS)
        .expect("a dense qwen3_5 install WITH an MTP head builds");
}

fn open(dir: &std::path::Path) -> RealForwardRunner {
    let peeked = turbospark_repack::peek_manifest_arch(dir).expect("manifest peeks");
    open_at_depth(dir, peeked).expect("install opens")
}

/// The fallible form, for the cases that assert a REFUSAL.
fn open_at_depth(
    dir: &std::path::Path,
    peeked: model_io::ArchConfig,
) -> Result<RealForwardRunner, turbospark_runtime::RealForwardError> {
    RealForwardRunner::open_with_options_and_speculation(
        dir,
        peeked,
        4096,
        16,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(
            DEPTH.parse().expect("DEPTH parses"),
        )),
    )
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
    let context = gpu::MetalContext::new().expect("Metal device");
    let device_name = context.device().name().to_string();
    drop(context);
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized, not the real Apple Silicon this digest was \
             taken on; comparing against the frozen reference with a tolerance instead"
        );
        assert_eq!(draft.len(), FROZEN_DRAFT_LOGITS.len());
        for (i, (got, &want)) in draft.iter().zip(FROZEN_DRAFT_LOGITS.iter()).enumerate() {
            let got = got.to_f32();
            let diff = (got - want).abs();
            let tol = 0.02_f32.max(want.abs() * 0.02);
            assert!(
                diff <= tol,
                "logit {i}: the draft logits moved: got {got}, want {want} (diff {diff}, \
                 tolerance {tol}); see this test's doc comment before re-freezing"
            );
        }
        return;
    }
    assert_eq!(
        digest(&draft),
        FROZEN_DRAFT_DIGEST,
        "the draft step's arithmetic moved"
    );
}

/// The real-hardware branch's comparison target, taken over the same
/// real-hardware run `FROZEN_DRAFT_LOGITS` above was recovered from.
const FROZEN_DRAFT_DIGEST: &str = "fd43a56b";

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
    let Err(err) = open_at_depth(&dir, peeked) else {
        panic!("a headless install must refuse an EXPLICIT draft depth");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("mtp.fc.weight"),
        "the refusal must name the missing tensor, got: {msg}"
    );
    assert!(
        msg.contains("TURBOSPARK_MTP_DRAFT"),
        "the refusal must name the knob that asked for it, got: {msg}"
    );
}

/// The other half of the same rule, and the half that was missing while
/// detection was wired backwards.
///
/// An EXPLICIT depth on a headless install is an error (above): the caller
/// named something the install cannot do. `Auto` on the same install is NOT,
/// because nobody named anything -- it is the detection answering "no head
/// here" rather than a request failing. Getting these two the same way round
/// is the whole difference between a knob and a detected capability.
#[test]
fn auto_declines_a_headless_install_and_builds_a_head_where_there_is_one() {
    let headless = temp_dir("auto-headless");
    build_synthetic_qwen_gdn_dense_install(&headless, VOCAB, LAYERS, "no-head")
        .expect("a headless dense install builds");
    let peeked = turbospark_repack::peek_manifest_arch(&headless).expect("manifest peeks");
    let runner = RealForwardRunner::open_with_options_and_speculation(
        &headless,
        peeked,
        4096,
        16,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Auto),
    )
    .expect("Auto must not refuse an install that simply has no head");
    assert_eq!(
        runner.mtp_draft_depth(),
        0,
        "no head means no drafter, and no allocation"
    );

    let with_head = temp_dir("auto-head");
    build_with_head(&with_head);
    let peeked = turbospark_repack::peek_manifest_arch(&with_head).expect("manifest peeks");
    let runner = RealForwardRunner::open_with_options_and_speculation(
        &with_head,
        peeked,
        4096,
        16,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Auto),
    )
    .expect("install opens");
    // THE POINT OF THE CHANGE: an install carrying a head gets a drafter
    // without anyone having set an environment variable. Before this, the
    // presence check ran only to word the error above, and this install
    // decoded sequentially and silently.
    assert_eq!(
        runner.mtp_draft_depth(),
        turbospark_runtime::MtpDraftPolicy::AUTO_DEPTH,
        "a head in the resident index must be detected and built"
    );
}

/// `Off` is `Off` even where a head exists, which is what every measuring
/// caller relies on: `open_with_options` pins it, so no frozen footprint row
/// can acquire the head's KV and M-row scratch by detection.
#[test]
fn off_declines_an_install_that_does_carry_a_head() {
    let dir = temp_dir("explicit-off");
    build_with_head(&dir);
    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("manifest peeks");
    let runner = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        peeked,
        4096,
        16,
        turbospark_runtime::DraftPolicies::off(),
    )
    .expect("install opens");
    assert_eq!(runner.mtp_draft_depth(), 0);

    // And the measuring entry point pins it without being asked.
    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("manifest peeks");
    let pinned = RealForwardRunner::open_with_options(&dir, peeked, 4096, 16)
        .expect("install opens through the measuring form");
    assert_eq!(
        pinned.mtp_draft_depth(),
        0,
        "open_with_options is the measuring form and must never sense a head"
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

/// The retaining rollback's MTP half: after a partial round undone by
/// `rollback_retaining` (KV rows kept, GDN state replayed over the verify
/// tape) instead of `rollback` plus a shortened re-verify, the FIRST DRAFT
/// off the bonus token must be byte-identical to the old shape's. The draft
/// is the sharpest observable for this, because it reads two things the two
/// shapes could in principle disagree about: the trunk's GDN/KV state, AND
/// `h_t` out of scratch row 0 -- which the re-verify used to refresh and
/// which the retaining path refreshes with an explicit copy of the last
/// kept row.
///
/// TWO PARTIAL SHAPES, because they break differently. `keep == 2` accepts
/// the first proposal and rejects the second. `keep == 1` rejects the
/// first -- the most common partial outcome -- and is the only shape that
/// reads the row-0 stash: `produce_batched`'s trailing copy clobbers row 0
/// whenever the block is wider than one, so a single kept row cannot be
/// rebuilt from any other row. Its feed is also TWO rows against the
/// THREE-row scratch, so the tape's recorded stride differs from the
/// scratch capacity -- the shrink `speculative.rs`'s `round_block` produces
/// near a budget. The fixture alternates linear and full layers, so the
/// replay walks two slots, and the trunk pass after the rollback is what
/// actually reads the state that replay wrote: a replay using the
/// capacity-sized stride lands slot 1 on bytes nobody recorded, and it is
/// the `after` digest, never the draft, that moves.
///
/// INT4, like the batched-parity test below: the batched GEMM exists at 4
/// bits alone, so the verify this round is built on cannot run on the 1-bit
/// fixture the rest of this file uses.
#[test]
fn the_draft_after_a_retaining_rollback_matches_the_reverify_shape() {
    ask_for_drafts();
    let dir = temp_dir("retaining");
    build_synthetic_qwen_gdn_dense_install_with_mtp(&dir, VOCAB, LAYERS, "mtp-int4", 4)
        .expect("a 4-bit dense install with a head builds");
    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("manifest peeks");

    fn draft_after_rollback(runner: &mut RealForwardRunner, retain: bool, keep: usize) -> String {
        let vocab = VOCAB as usize;
        runner.reset();
        let mut row = vec![f16::from_f32(0.0); vocab];
        let mut token = 5i32;
        for position in 0..4usize {
            runner.produce(token, position, &mut row).expect("produce");
            token = argmax(&row);
            if position + 1 < 4 {
                runner.mtp_prime_step(token, position).expect("prime");
            }
        }
        let base = 4usize;

        // Learn the target's row argmaxes on a throwaway verify; a full
        // rollback undoes it (and clears the tape with it). Causality makes
        // rows 0 and 1 exactly what the real pass below will compute.
        let mut learn = vec![f16::from_f32(0.0); 3 * vocab];
        let point = runner.checkpoint();
        runner
            .verify(&[token, 13, 21], base, &mut learn)
            .expect("learn verify");
        let a0 = argmax(&learn[0..vocab]);
        let a1 = argmax(&learn[vocab..2 * vocab]);
        runner.rollback(&point);

        // The real feed, crafted to the keep. keep == 2 accepts the first
        // proposal and rejects the second; keep == 1 rejects the first, so
        // its feed is two rows and the round is partial at one kept row.
        let (feed, feed_rows) = if keep == 2 {
            let rejected = ((a1 as usize + 1) % vocab) as i32;
            (vec![token, a0, rejected], 3usize)
        } else {
            let rejected = ((a0 as usize + 1) % vocab) as i32;
            (vec![token, rejected], 2usize)
        };
        let mut batch = vec![f16::from_f32(0.0); feed_rows * vocab];
        let point = runner.checkpoint();
        runner.verify(&feed, base, &mut batch).expect("verify");
        assert_eq!(
            argmax(&batch[0..vocab]),
            a0,
            "the crafted round must agree with the learned row"
        );
        if keep == 2 {
            assert_ne!(
                argmax(&batch[vocab..2 * vocab]),
                feed[2],
                "the crafted second proposal must be rejected"
            );
        }

        if retain {
            runner
                .rollback_retaining(&point, keep)
                .expect("the retaining rollback replays the tape");
        } else {
            runner.rollback(&point);
            let mut replay = vec![f16::from_f32(0.0); keep * vocab];
            runner
                .verify(&feed[..keep], base, &mut replay)
                .expect("the shortened replay verify runs");
        }
        // The bonus comes from the last kept row. The head is untouched by
        // either rollback shape, so it still sits at its prefill cursor
        // covering `[0, base - 1)` -- draft there, where the step is legal,
        // and it reads `h_t` out of trunk scratch row 0: exactly the row
        // the two shapes could disagree about.
        let bonus = argmax(&batch[(keep - 1) * vocab..keep * vocab]);
        let mut draft = vec![f16::from_f32(0.0); vocab];
        runner
            .mtp_draft_step(bonus, base - 1, &mut draft)
            .expect("draft");
        // The draft reads the head and `h_t` but NOT the trunk's recurrent
        // state; the first trunk pass after the rollback does. Digesting
        // both is what makes the replay's GDN half observable here at all --
        // without it, a replay that read the tape at the wrong stride would
        // corrupt the trunk silently and this test would stay green.
        let mut after = vec![f16::from_f32(0.0); vocab];
        runner
            .produce(bonus, base + keep, &mut after)
            .expect("produce after the rollback");
        format!("{}:{}", digest(&draft), digest(&after))
    }

    for keep in [2usize, 1] {
        let mut legacy = open_at_depth(&dir, peeked.clone()).expect("open");
        let a = draft_after_rollback(&mut legacy, false, keep);
        let mut retained = open_at_depth(&dir, peeked.clone()).expect("open");
        let b = draft_after_rollback(&mut retained, true, keep);
        assert_eq!(
            a, b,
            "keep {keep}: the draft after the two rollback shapes diverged"
        );
    }
}

/// The headless-prefill gate's REASON, as a discriminating pair: the MTP
/// head reads the trunk's POST-FINAL-NORM hidden, which `produce_prefill`
/// skips -- so a headless-style prefill primes the head DIFFERENTLY from
/// the headful one. This is why `SpeculativeProducer::supports_headless_
/// prefill` is false for the MTP head while true for DFlash2: flipped, the
/// loop would still run losslessly and every acceptance number would
/// quietly sink, with nothing reddening but this.
#[test]
fn an_mtp_head_is_not_headless_prefill_safe() {
    ask_for_drafts();
    let dir = temp_dir("headless");
    build_with_head(&dir);
    let mut runner = open(&dir);

    fn draft_after(runner: &mut RealForwardRunner, headless: bool) -> String {
        let vocab = VOCAB as usize;
        runner.reset();
        let mut row = vec![f16::from_f32(0.0); vocab];
        let mut token = 5i32;
        for position in 0..4usize {
            let last = position + 1 == 4;
            if last || !headless {
                runner.produce(token, position, &mut row).expect("produce");
            } else {
                runner
                    .produce_prefill(token, position, &mut row)
                    .expect("produce_prefill");
            }
            token = argmax(&row);
            if position + 1 < 4 {
                runner.mtp_prime_step(token, position).expect("prime");
            }
        }
        let mut draft = vec![f16::from_f32(0.0); vocab];
        runner.mtp_draft_step(token, 3, &mut draft).expect("draft");
        digest(&draft)
    }

    let headful = draft_after(&mut runner, false);
    let headless = draft_after(&mut runner, true);
    assert_ne!(
        headful, headless,
        "the MTP draft is blind to the final norm: the headless gate would be pointless"
    );
}

/// STEP 4's WHOLE CONTRACT: one batched pass is BIT-IDENTICAL to the same
/// tokens run one at a time (`docs/MTP_SPECULATIVE.md`).
/// This is what makes speculative output provably identical to
/// non-speculative output, and it is asserted with `==` rather than a
/// tolerance because the batched kernel does not reassociate any sum -- each
/// output row keeps its own accumulator, exactly as B separate GEMV calls
/// would (`crates/gpu/tests/dequant_int4_gemm_parity.rs`).
///
/// A 4-BIT install, unlike every other case in this file. The batched GEMM
/// exists at INT4 alone, so a 1- or 2-bit fixture cannot reach this path at
/// all -- which is its own test, immediately below.
///
/// **IT RUNS TWO TOKENS PAST THE BATCH, and that is not thoroughness, it is
/// the only way half the state is observable.** A batch that starts at
/// position 0 begins from an empty gated-DeltaNet conv tail, and
/// `gdn_conv_mix_prefill` reconstructs every intra-batch row's history from
/// the batch itself -- so the batch's OWN logits are identical whether or
/// not the layer's tail is advanced afterwards. Dropping
/// `encode_gdn_conv_tail_update` reddens nothing until a token is produced
/// after the batch and reads the stale tail. Found by mutation, not review.
#[test]
fn a_batched_pass_is_bit_identical_to_the_same_tokens_run_sequentially() {
    ask_for_drafts();
    let dir = temp_dir("batched-parity");
    build_synthetic_qwen_gdn_dense_install_with_mtp(&dir, VOCAB, LAYERS, "mtp-int4", 4)
        .expect("a 4-bit dense install with a head builds");
    let mut runner = open(&dir);
    let vocab = VOCAB as usize;
    let tokens = [5i32, 9, 3, 7, 2];
    /// Rows the batched arm takes in one pass; the rest it produces one at a
    /// time, so the comparison covers the state the batch LEFT as well as
    /// the logits it produced.
    const BATCH: usize = 3;

    runner.reset();
    let mut sequential: Vec<f16> = Vec::with_capacity(tokens.len() * vocab);
    let mut row = vec![f16::from_f32(0.0); vocab];
    for (position, &token) in tokens.iter().enumerate() {
        runner
            .produce(token, position, &mut row)
            .expect("sequential produce");
        sequential.extend_from_slice(&row);
    }
    let sequential_cursor = runner.checkpoint().position();

    runner.reset();
    let mut batched = vec![f16::from_f32(0.0); BATCH * vocab];
    runner
        .produce_batched(&tokens[..BATCH], 0, &mut batched)
        .expect("batched pass");
    for (offset, &token) in tokens[BATCH..].iter().enumerate() {
        runner
            .produce(token, BATCH + offset, &mut row)
            .expect("produce after the batch");
        batched.extend_from_slice(&row);
    }

    assert_eq!(
        sequential, batched,
        "the batched pass diverged from sequential produce: a speculative \
         verify built on this would not be lossless"
    );
    // The cursor has to land where the same tokens run one at a time leave
    // it, or `rollback` would need a batched variant and every position
    // downstream would be off by the block size.
    assert_eq!(
        sequential_cursor,
        runner.checkpoint().position(),
        "the batched pass left the KV cursor somewhere else"
    );
}

/// The refusal that keeps the measurement honest (AGENTS.md Gotcha 35 one
/// layer down).
///
/// A sequential fallback here would be NUMERICALLY IDENTICAL, so it would
/// pass the parity case above and the end-to-end losslessness gate too --
/// and a "batched" verify would then measure the sequential engine and
/// report its cost as the batched one. The 1-bit and 2-bit checkpoints of
/// this same architecture have no batched kernel, so this is a live case and
/// not a hypothetical.
#[test]
fn a_sub_4_bit_install_is_refused_by_the_batched_path_rather_than_looped() {
    ask_for_drafts();
    let dir = temp_dir("batched-refuse-1bit");
    build_with_head(&dir);
    let mut runner = open(&dir);
    let vocab = VOCAB as usize;
    let tokens = [5i32, 9];

    runner.reset();
    let mut batched = vec![f16::from_f32(0.0); tokens.len() * vocab];
    let err = runner
        .produce_batched(&tokens, 0, &mut batched)
        .expect_err("a 1-bit install has no batched kernel");
    let msg = format!("{err}");
    assert!(
        msg.contains("BATCHED") && msg.contains("INT4-affine"),
        "the refusal must name the dtype and say it is not looped, got: {msg}"
    );
}

/// A block wider than the scratch is an error rather than an overrun, and a
/// block wider than the kernel's register file is an error here rather than
/// an `assert!` inside the dispatch.
#[test]
fn a_batch_beyond_the_scratch_is_refused() {
    ask_for_drafts();
    let dir = temp_dir("batched-too-wide");
    build_synthetic_qwen_gdn_dense_install_with_mtp(&dir, VOCAB, LAYERS, "mtp-int4-wide", 4)
        .expect("a 4-bit dense install with a head builds");
    let mut runner = open(&dir);
    let vocab = VOCAB as usize;

    // DEPTH is "2", so the scratch holds DEPTH + 1 = 3 rows.
    let tokens = [5i32, 9, 3, 7];
    runner.reset();
    let mut batched = vec![f16::from_f32(0.0); tokens.len() * vocab];
    let err = runner
        .produce_batched(&tokens, 0, &mut batched)
        .expect_err("4 rows against scratch sized for 3");
    assert!(
        format!("{err}").contains("scratch sized for 3"),
        "the refusal must name both widths, got: {err}"
    );
}

/// A batched pass has to leave the DRAFTER's input where a sequential run of
/// the same tokens leaves it.
///
/// `mtp_draft_step` reads `h_t` from `scratch.x` at offset 0, and a batched
/// pass writes `batch` rows there. Leaving row 0 as the FIRST token's
/// residual makes the head draft off the wrong hidden state -- and nothing
/// else can see it: the trunk is untouched, so the committed stream stays
/// byte-identical to a non-speculative run and the end-to-end losslessness
/// gate passes. Only the accept LENGTH moves, which reads as a verdict about
/// MTP rather than as a bug (measured on the real install: 1.84 accepted per
/// round to 1.10).
///
/// Both arms prime the head identically and draft at the SAME position with
/// the same head KV, so the only variable is what the trunk left in
/// `scratch.x`. The draft position does not correspond to the hidden state
/// being fed, deliberately: this is a test about buffer plumbing, and
/// requiring the two to agree would need a head cursor the batched arm
/// cannot reach.
#[test]
fn a_batched_pass_leaves_the_drafters_hidden_state_where_sequential_does() {
    ask_for_drafts();
    let dir = temp_dir("batched-draft-input");
    build_synthetic_qwen_gdn_dense_install_with_mtp(&dir, VOCAB, LAYERS, "mtp-int4-draft", 4)
        .expect("a 4-bit dense install with a head builds");
    let mut runner = open(&dir);
    let vocab = VOCAB as usize;
    let tokens = [5i32, 9, 3, 7];

    // Walk the first two tokens sequentially in BOTH arms, priming the head
    // to cursor 2 so a draft at position 2 is legal either way.
    let prime_prefix = |runner: &mut RealForwardRunner| {
        runner.reset();
        let mut row = vec![f16::from_f32(0.0); vocab];
        for position in 0..2 {
            runner
                .produce(tokens[position], position, &mut row)
                .expect("prefix produce");
            runner
                .mtp_prime_step(tokens[position + 1], position)
                .expect("prime");
        }
    };

    prime_prefix(&mut runner);
    let mut row = vec![f16::from_f32(0.0); vocab];
    for (position, &token) in tokens.iter().enumerate().skip(2) {
        runner
            .produce(token, position, &mut row)
            .expect("sequential tail");
    }
    let mut sequential_draft = vec![f16::from_f32(0.0); vocab];
    runner
        .mtp_draft_step(11, 2, &mut sequential_draft)
        .expect("draft after the sequential tail");

    prime_prefix(&mut runner);
    let mut batched = vec![f16::from_f32(0.0); 2 * vocab];
    runner
        .produce_batched(&tokens[2..], 2, &mut batched)
        .expect("batched tail");
    let mut batched_draft = vec![f16::from_f32(0.0); vocab];
    runner
        .mtp_draft_step(11, 2, &mut batched_draft)
        .expect("draft after the batched tail");

    assert_eq!(
        sequential_draft, batched_draft,
        "the head drafted off a different hidden state after the batched \
         pass: the batch's LAST row has to land at row 0 of scratch.x"
    );
}

/// The env mapping, as a pure function, because `from_env` reads a
/// process-global that every other test in this binary shares.
///
/// The load-bearing row is the FIRST one. Unset used to mean off, and that is
/// what made an install carrying a head decode sequentially unless somebody
/// knew to set a variable that is documented in one crate's gotcha list.
#[test]
fn the_env_mapping_treats_unset_as_auto_and_zero_as_off() {
    use turbospark_runtime::MtpDraftPolicy as P;
    assert_eq!(P::from_env_value(None), P::Auto);
    assert_eq!(P::from_env_value(Some("0")), P::Off);
    assert_eq!(P::from_env_value(Some("2")), P::Fixed(2));
    assert_eq!(P::from_env_value(Some("16")), P::Fixed(16));
    // A typo must not silently disable a feature the install can serve.
    assert_eq!(P::from_env_value(Some("yes")), P::Auto);
    assert_eq!(P::from_env_value(Some("")), P::Auto);
    assert_eq!(P::from_env_value(Some(" 4 ")), P::Fixed(4));
}

/// A HEAD IS NOT ENOUGH, and this is the case that says so.
///
/// The synthetic dense install is built at `BITS = 1`, so its projections are
/// dtype 15 and `encode_gemm_any` has no arm for them. It carries a real head,
/// so a head-presence check passes -- and speculation would then die at the
/// first batched verify, mid-generation. `speculation_blocker` is what turns
/// that into an answer available at open.
///
/// This is also the honest statement of what is and is not blocked: the same
/// install DECODES perfectly well below, which is the whole point. Only the
/// batched verify is INT4-only.
#[test]
fn a_sub_4_bit_install_with_a_head_reports_why_it_cannot_speculate() {
    ask_for_drafts();
    let dir = temp_dir("sub4bit-head");
    build_with_head(&dir);
    let runner = open(&dir);

    assert!(
        runner.mtp_draft_depth() > 0,
        "the fixture is built with a head; without one this test proves nothing"
    );
    let blocker = runner
        .speculation_blocker()
        .expect("a 1-bit install cannot run the batched verify");
    assert!(
        blocker.contains("INT4-only"),
        "the reason must name the real blocker rather than the head, got: {blocker}"
    );
    // And NOT the head, which this install has. The pair is what says the
    // ordering in `speculation_blocker` is doing work rather than the fixture
    // happening to have one obstacle.
    assert!(
        !blocker.contains("last shard"),
        "an install that HAS a head must not be sent after one: {blocker}"
    );

    // And the model still decodes. The blocker is about speculation alone.
    let mut runner = runner;
    let mut logits = vec![half::f16::from_f32(0.0); VOCAB as usize];
    runner
        .produce(1, 0, &mut logits)
        .expect("a 1-bit install decodes normally");
    assert!(logits.iter().all(|v| v.to_f32().is_finite()));
}

/// **THE NO-HEAD ARM, AND THE POINTER IT HAS TO CARRY.**
///
/// Dense and INT4, so both architectural checks pass and the head is the only
/// thing left: this is the one fixture in the repo that reaches
/// `speculation_blocker`'s last arm at all. Built at 4 bits rather than this
/// file's `BITS = 1` for exactly that reason, since a 1-bit install is stopped
/// by the INT4 arm above and can never see this one.
///
/// **THE POINTER IS THE POINT, and it went missing for a day without any test
/// noticing.** Before `draft_policies` grew its headless arm (2026-08-21) a
/// named block on this shape failed at OPEN, so the string a caller saw was
/// `MtpState::build`'s, which names the shard that fixes it. Afterwards the
/// open succeeds and THIS string is the only one they see, and it named the
/// obstacle without the remedy. `speculation_policy_tests.rs` could not catch
/// that: it feeds a `NO_HEAD` fixture into `resolve_speculation` and never
/// calls this function, so it pins the ROUTING and says nothing about the
/// text. Verified by mutation -- deleting the pointer here leaves that whole
/// module green and reddens this case alone.
#[test]
fn a_dense_int4_install_without_a_head_is_told_which_artifact_would_fix_it() {
    let dir = temp_dir("dense-int4-headless");
    build_synthetic_qwen_gdn_dense_install_at_bits(&dir, VOCAB, LAYERS, "headless-toy", 4)
        .expect("a dense INT4 qwen3_5 install builds");
    // `Auto` and not this file's `open`, which pins `Fixed(DEPTH)` and so
    // fails at the open on a headless install with `MtpState::build`'s error.
    // That refusal is the OLD path and is still correct where it fires; what
    // this test is about is the string a caller gets when the open SUCCEEDS,
    // which is what `draft_policies`' headless arm now arranges for a named
    // block. `Auto` reproduces that state without going through the CLI.
    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("manifest peeks");
    let runner = RealForwardRunner::open_with_options_and_speculation(
        &dir,
        peeked,
        4096,
        16,
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Auto),
    )
    .expect("a headless install opens under Auto");

    // ASSERT THE FIXTURE DISCRIMINATES: with a head, or at 1 bit, this arm is
    // unreachable and every assertion below would be testing another arm.
    assert_eq!(
        runner.mtp_draft_depth(),
        0,
        "the fixture must be headless or it cannot reach this arm"
    );
    let blocker = runner
        .speculation_blocker()
        .expect("a headless install cannot speculate");
    assert!(
        !blocker.contains("INT4-only") && !blocker.contains("experts"),
        "a dense INT4 install must reach past both architectural arms: {blocker}"
    );
    assert!(
        blocker.contains("multi-token-prediction head"),
        "on a dense install the head really is the obstacle: {blocker}"
    );
    assert!(
        blocker.contains("last shard") && blocker.contains("docs/MTP.md"),
        "naming the obstacle without the remedy is what this test exists to \
         prevent: {blocker}"
    );
}
