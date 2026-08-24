//! Does the batched verify apply the steering edit the way M sequential
//! `produce` calls do?
//!
//! This is the gate that let `real_forward_open.rs` stop refusing steering
//! and speculation together. That refusal was not conservatism: the verify
//! pass is what COMMITS a speculative token, so an unsteered batched forward
//! beside a steered sequential one emits a run of tokens drawn from two
//! different models -- coherent, wrong, and invisible in every downstream
//! number, because a losslessness check compares a speculative run against a
//! speculative reference and both would carry the same mixture.
//!
//! **THE HEADLINE ALONE WOULD BE GREEN IF STEERING DID NOTHING AT ALL.** An
//! `encode_steering` that never dispatched would make the batched and
//! sequential paths agree perfectly, which is exactly AGENTS.md Gotcha 48's
//! trap and the reason `a_steered_batch_is_not_the_unsteered_one` sits beside
//! it. The pair is the test; neither half is.
//!
//! Bit-identity is the right bar HERE and would not be on the real install:
//! `produce_batched` and `produce` part at span 3 on `qwen38-27b` and that is
//! this port's own shape floor (~1e-5 nats with the argmax agreeing), not a
//! defect. On this fixture the unsteered paths already agree to the bit
//! (`real_forward_qwen_moe_batched.rs`), so anything the edit breaks shows up
//! as an exact mismatch rather than as a tolerance judgement.

use half::f16;
use model_io::{LayerDirection, SteeringSet};
use turbospark_repack::build_synthetic_qwen_gdn_dense_install_with_mtp;
use turbospark_runtime::{LogitProducer, RealForwardRunner, SteeringPolicy};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;
/// INT4: `encode_gemm_any` is INT4-only, so the 1-bit and 2-bit widths this
/// builder also serves cannot reach the batched pass at all.
const BITS: u32 = 4;
const BATCH: usize = 3;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-steered-batched-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build(dir: &std::path::Path) {
    build_synthetic_qwen_gdn_dense_install_with_mtp(
        dir,
        VOCAB,
        LAYERS,
        "tiny-qwen35-steered",
        BITS,
    )
    .expect("a dense qwen3_5 install WITH an MTP head builds");
}

/// A direction on every layer, deterministic and NOT axis-aligned.
///
/// Varying with both the layer and the element matters: a constant direction
/// would make every layer's edit the same vector, so a bug that applied
/// layer 0's direction everywhere would still agree with itself. The values
/// are order-1 against a fixture residual of the same scale, which is what
/// makes the edit visible at a fractional alpha.
fn direction_set(hidden: usize, layers: usize) -> SteeringSet {
    let per_layer = (0..layers)
        .map(|l| {
            let values = (0..hidden)
                .map(|i| {
                    let a = (i as f32 * 0.37 + l as f32 * 1.13).sin();
                    let b = (i as f32 * 0.11).cos();
                    0.5 * a + 0.25 * b
                })
                .collect::<Vec<f32>>();
            Some(LayerDirection::new(values))
        })
        .collect();
    SteeringSet {
        layers: per_layer,
        hidden,
        declared_mode: None,
        declared_arch: None,
    }
}

/// The direction's width comes from the INSTALL, never from a literal.
///
/// `SteeringSet::validate` refuses a width that disagrees with the model, so
/// a hardcoded one is a test that fails for a reason unrelated to what it
/// asks -- which is how this file's first run went (64 against the fixture's
/// 128).
fn policy(arch: &model_io::ArchConfig, alpha: f32) -> SteeringPolicy {
    SteeringPolicy {
        set: Some(direction_set(
            arch.hidden_size as usize,
            arch.num_layers as usize,
        )),
        mode: foundation::SteeringMode::Ablate,
        alpha,
        target: 0.0,
        gate_threshold: 0.0,
    }
}

fn arch_of(dir: &std::path::Path) -> model_io::ArchConfig {
    turbospark_repack::peek_manifest_arch(dir).expect("manifest peeks")
}

fn open_steered(dir: &std::path::Path, steering: SteeringPolicy) -> RealForwardRunner {
    let peeked = arch_of(dir);
    RealForwardRunner::open_with_slot_policy_speculation_and_steering(
        dir,
        peeked,
        4096,
        turbospark_runtime::ExpertCacheSlots::Fixed(8),
        // The M-row scratch is allocated beside a drafter and `produce_batched`
        // refuses by name without it. Nothing here drafts a token: the point
        // is the VERIFY pass, which is the half that commits.
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(BATCH)),
        steering,
    )
    .expect("a steered install with a drafter opens")
}

/// Runs `tokens` one at a time and returns every logit.
fn sequential(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<f16> {
    let vocab = VOCAB as usize;
    let mut out: Vec<f16> = Vec::with_capacity(tokens.len() * vocab);
    let mut row = vec![f16::from_f32(0.0); vocab];
    for (position, &token) in tokens.iter().enumerate() {
        runner
            .produce(token, position, &mut row)
            .expect("sequential produce");
        out.extend_from_slice(&row);
    }
    out
}

/// Runs `tokens` as one block and returns every logit.
fn batched(runner: &mut RealForwardRunner, tokens: &[i32]) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); tokens.len() * VOCAB as usize];
    runner
        .produce_batched(tokens, 0, &mut out)
        .expect("batched pass");
    assert!(
        out.iter().all(|v| v.to_f32().is_finite()),
        "batched logits must be finite before any comparison means anything \
         (AGENTS.md Gotcha 59: NaN scores a perfect result on every rank instrument, \
         and two all-NaN vectors compare EQUAL under this file's assertions)"
    );
    out
}

/// The headline, and the assertion the lifted refusal rests on.
#[test]
fn the_batched_forward_steers_every_row_exactly_as_m_produce_calls_do() {
    let dir = temp_dir("parity");
    build(&dir);
    let mut runner = open_steered(&dir, policy(&arch_of(&dir), 0.35));
    let tokens = [5i32, 9, 2];

    let one_at_a_time = sequential(&mut runner, &tokens);
    let cursor = runner.checkpoint().position();

    runner.reset();
    let as_a_block = batched(&mut runner, &tokens);

    assert_eq!(
        one_at_a_time, as_a_block,
        "a steered batched verify diverged from steered sequential decode, so a \
         speculative run under --steering would commit tokens from a model that is \
         neither the steered one nor the unsteered one"
    );
    assert_eq!(
        cursor,
        runner.checkpoint().position(),
        "the steered batched pass left the KV cursor somewhere else"
    );
}

/// THE FIXTURE MUST BE ABLE TO SEE THE EDIT, or the case above is green for
/// the worst possible reason.
///
/// A steering hook that dispatched nothing -- a wrong `rows`, a coverage
/// table read as empty, an `inv_norm` of zero from a degenerate direction --
/// leaves the batched and sequential paths agreeing perfectly, because both
/// would be the unsteered model. Only a comparison against a KNOWN-different
/// engine can tell "agrees because both are steered" from "agrees because
/// neither is".
#[test]
fn a_steered_batch_is_not_the_unsteered_one() {
    let dir = temp_dir("discriminates");
    build(&dir);
    let tokens = [5i32, 9, 2];

    let mut off = open_steered(&dir, SteeringPolicy::off());
    let unsteered = batched(&mut off, &tokens);
    drop(off);

    let mut on = open_steered(&dir, policy(&arch_of(&dir), 0.35));
    let steered = batched(&mut on, &tokens);

    assert_ne!(
        unsteered, steered,
        "the batched pass produced identical logits steered and unsteered, so this \
         fixture cannot see the edit and the parity case above proves nothing"
    );
}

/// The null control, on the batched path: at alpha 0 the kernel runs at every
/// covered layer for every row and the result is bit-identical to steering
/// off.
///
/// This is the one case here that is about the KERNEL rather than about the
/// two paths agreeing. It covers a wrong reduction, a wrong direction offset,
/// a wrong row stride and a coefficient block that overruns into the next
/// layer's, all in one comparison -- the same job arm 1 of `steering_probe`
/// does per token, at M rows.
#[test]
fn the_batched_null_control_is_bit_identical_to_steering_off() {
    let dir = temp_dir("null");
    build(&dir);
    let tokens = [5i32, 9, 2];

    let mut off = open_steered(&dir, SteeringPolicy::off());
    let unsteered = batched(&mut off, &tokens);
    drop(off);

    let mut zero = open_steered(&dir, policy(&arch_of(&dir), 0.0));
    let at_zero = batched(&mut zero, &tokens);

    assert_eq!(
        unsteered, at_zero,
        "steering at alpha 0 moved the batched logits; the edit is the identity at \
         zero in every mode, so this is the dispatch itself being wrong rather than \
         the strength"
    );
}

/// A block too wide to steer is REFUSED somewhere, never truncated.
///
/// The kernel writes `coeff[row]` for every row it steers, so a block of M
/// rows writes M floats from the layer's base. Steering only the rows that
/// fit would leave the rest of the block drawn from the unsteered model,
/// which is the exact silent mixture the lifted refusal used to prevent
/// wholesale.
///
/// **THE STEERING GUARD CANNOT ACTUALLY FIRE, AND MEASURING THAT IS THE
/// POINT OF THIS CASE.** `MAX_STEER_ROWS` IS `gpu::MAX_BATCH_ROWS`, so the
/// batched INT4 GEMM refuses the same block one level down and one layer
/// earlier -- for its own reason (a per-thread register array), on the first
/// projection of layer 0, before anything is steered. So the assertion is
/// about the ORDER rather than about the message: some guard refuses, and the
/// one that does names what is too small. Asserting the steering wording here
/// would be asserting an unreachable branch.
#[test]
fn a_block_wider_than_the_widest_dispatch_is_refused_by_name() {
    let dir = temp_dir("too-wide");
    build(&dir);
    let wide = turbospark_runtime::MAX_STEER_ROWS + 1;
    let peeked = arch_of(&dir);
    let Ok(mut runner) = RealForwardRunner::open_with_slot_policy_speculation_and_steering(
        &dir,
        peeked,
        4096,
        turbospark_runtime::ExpertCacheSlots::Fixed(8),
        turbospark_runtime::DraftPolicies::mtp(turbospark_runtime::MtpDraftPolicy::Fixed(wide)),
        policy(&arch_of(&dir), 0.35),
    ) else {
        // The batched scratch may refuse a block this wide first, which is
        // its own correct answer; nothing here asserts WHICH guard fires,
        // only that no path steers part of a block.
        return;
    };
    let tokens: Vec<i32> = (0..wide).map(|i| (i % VOCAB as usize) as i32).collect();
    let mut out = vec![f16::from_f32(0.0); tokens.len() * VOCAB as usize];
    let err = runner
        .produce_batched(&tokens, 0, &mut out)
        .expect_err("a block wider than the coefficient buffer must be refused");
    let text = format!("{err:?}");
    assert!(
        text.contains(&format!("1..={}", turbospark_runtime::MAX_STEER_ROWS))
            || text.contains("coefficient buffer")
            || text.contains("scratch sized for"),
        "the refusal must NAME what is too small; got {text}"
    );
}

/// The steering guard and the kernel's cap are ONE number, not two that
/// agree.
///
/// If they ever parted, the coefficient buffer would be narrower than the
/// blocks the engine accepts and a batched steer at the last layer would
/// write past the end of it -- which is a GPU-side overrun, so the symptom
/// would be whatever happened to sit after that buffer rather than an error.
#[test]
fn the_steering_row_cap_is_the_batched_kernels_own() {
    assert_eq!(
        turbospark_runtime::MAX_STEER_ROWS,
        gpu::MAX_BATCH_ROWS,
        "the coefficient buffer is sized in MAX_STEER_ROWS and the widest block the \
         engine accepts is MAX_BATCH_ROWS; a gap between them is a buffer overrun"
    );
}
