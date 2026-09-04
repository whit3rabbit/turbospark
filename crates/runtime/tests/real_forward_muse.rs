//! The `muse_glimmer` decode flow, on a synthetic install.
//!
//! Shaped like `real_forward_gptoss.rs` and for its reason
//! (`crates/runtime/CLAUDE.md` Gotcha 11): a synthetic fixture's weights are
//! UNTRAINED, so "it decoded" is nearly worthless as evidence -- a wrong
//! norm, a dropped gate or a rotated NoPE layer all produce meaningless
//! output too, and `5279c88` shipped a Qwen at perplexity 255,409 with the
//! whole suite green.
//!
//! So the assertions are **DOES THIS INPUT REACH THE MATH**: for each tensor
//! that makes this a sixth flow, perturb it in the install and require the
//! logits to move. A flow that silently drops one is INVARIANT to its
//! perturbation, which a fixture CAN see even though it cannot judge text.
//!
//! Patching in place rather than rebuilding is AGENTS.md Gotcha 33's trick:
//! resident tensors sit at fixed offsets, every perturbation preserves
//! length, and `open()` runs no receipt check, so a hypothesis costs
//! milliseconds.
//!
//! **WHAT THIS FILE IS BLIND TO, MEASURED RATHER THAN ASSUMED.** Seven
//! mutations were run against it. Six redden (see
//! `the_synthetic_flows_arithmetic_is_frozen`). The one that SURVIVES is
//! collapsing the two RMS epsilons into one: at FP16 with a `mean_sq` near
//! 1, `rsqrt(1 + 1e-5)` and `rsqrt(1 + 1e-8)` differ by ~5e-6 relative,
//! which is an order of magnitude below FP16's resolution, so the digest
//! does not move. The VALUES are pinned offline against the real
//! `config.json` (`crates/repack/tests/museglimmer_config.rs`); which
//! epsilon reaches which of the six norm sites is a question only the
//! real-model quality gate can answer.
//!
//! This file also cannot judge whether the CENTERED norm is the right
//! convention for this checkpoint -- only that swapping it moves the
//! arithmetic. That too is the quality gate's question.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::build_synthetic_muse_glimmer_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 64;
/// A multiple of 4 so the `[0, 0, 0, 1]` window pattern is whole: layers 0-2
/// slide and layer 3 is FULL and NoPE. Eight gives two periods, so the
/// alternation itself is exercised rather than just its first instance.
const LAYERS: i64 = 8;

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("turbospark-muse-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = build_synthetic_muse_glimmer_install(&dir, VOCAB, LAYERS, "muse-toy")
        .expect("the install writes");
    (dir, arch)
}

/// Decodes `steps` greedy tokens and returns the last step's logits.
///
/// `steps` is deliberately larger than the fixture's 8-token sliding window,
/// so the SLIDING layers wrap their ring while the full ones do not. A window
/// bug that only bites past the window is invisible below it.
fn decode(dir: &std::path::Path, arch: &model_io::ArchConfig, steps: usize) -> Vec<f32> {
    let mut runner =
        RealForwardRunner::open(dir, arch.clone()).expect("the muse_glimmer install opens");
    let vocab = runner.vocab_size();
    let mut logits = vec![half::f16::ZERO; vocab];
    for step in 0..steps {
        runner
            .produce((step % VOCAB as usize) as i32, step, &mut logits)
            .expect("produce");
    }
    logits.iter().map(|v| v.to_f32()).collect()
}

/// 12 steps against an 8-token window: the ring wraps on the sliding layers.
const STEPS: usize = 12;

fn baseline() -> (std::path::PathBuf, model_io::ArchConfig, Vec<f32>) {
    let (dir, arch) = install();
    let logits = decode(&dir, &arch, STEPS);
    (dir, arch, logits)
}

/// Flips a few bytes of one resident tensor, in place.
///
/// Returns false when the tensor is absent, so a caller naming a tensor this
/// family does not have fails loudly rather than silently "passing".
fn perturb(dir: &std::path::Path, tensor: &str) -> bool {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index");
    let Some(entry) = index.entries.get(tensor) else {
        return false;
    };
    let mut bytes = std::fs::read(&path).expect("read");
    let start = entry.file_offset as usize;
    let end = start + entry.size_bytes as usize;
    for b in &mut bytes[start..end.min(start + 512)] {
        *b ^= 0x3C;
    }
    std::fs::write(&path, bytes).expect("write");
    true
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn a_muse_glimmer_install_decodes() {
    let (_dir, _arch, logits) = baseline();
    assert_eq!(logits.len(), VOCAB as usize);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "every logit must be finite"
    );
}

/// EVERY tensor this family carries must reach the logits.
///
/// `self_attn.gate_proj` is the one that earns this test. It is the
/// attention OUTPUT GATE, it is a separate projection unique to this family,
/// and its name collides with `mlp.gate_proj`, which every family has -- so
/// a flow that dropped it, or that read the MLP's gate by mistake, would
/// still decode. The four sandwich norms are here for the same reason: this
/// is the only family with all four, and a flow that applied two would be
/// invariant to the other two.
#[test]
fn every_layer_tensor_reaches_the_logits() {
    let (_base_dir, _base_arch, base) = baseline();
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.gate_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ] {
        let (dir, arch) = install();
        let name = format!("language_model.model.layers.0.{suffix}");
        assert!(perturb(&dir, &name), "{name} must exist in the install");
        let moved = decode(&dir, &arch, STEPS);
        assert!(
            max_abs_diff(&base, &moved) > 1e-3,
            "perturbing {suffix} did not move the logits: the flow is not reading it"
        );
    }
}

/// The two MODEL-level tensors, including the final norm -- which is the
/// PLAIN-convention one and therefore dispatched through a different kernel
/// than the four above.
#[test]
fn the_model_level_tensors_reach_the_logits() {
    let (_base_dir, _base_arch, base) = baseline();
    for name in [
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
    ] {
        let (dir, arch) = install();
        assert!(perturb(&dir, name), "{name} must exist");
        let moved = decode(&dir, &arch, STEPS);
        assert!(
            max_abs_diff(&base, &moved) > 1e-3,
            "perturbing {name} did not move the logits"
        );
    }
}

/// THE HEAD IS SOFTCAPPED, and the bound is the assertion.
///
/// `docs/NEW_MODEL.md` Phase 4 asks for exactly this where a family has a
/// softcap. `|logit| <= softcap` is a property of the transform alone, so it
/// holds on untrained weights, and a head that skipped the cap on a fixture
/// whose logits saturate would break it.
#[test]
fn the_head_is_softcapped() {
    let (_dir, arch, logits) = baseline();
    let cap = arch.final_logit_softcap as f32;
    assert!(cap > 0.0);
    for (i, v) in logits.iter().enumerate() {
        assert!(
            v.abs() <= cap + 1e-2,
            "logit {i} is {v}, outside the softcap {cap}"
        );
    }
}

/// AGENTS.md Gotcha 16: `produce` writes LOGITS, never probabilities.
///
/// This family stacks TWO post-head transforms (an output multiplier and a
/// softcap), which is exactly the shape that invites a third being added, so
/// the guard matters more here than in a family with none. Real logits are
/// not all non-negative and do not sum to 1.
#[test]
fn the_head_does_not_normalize() {
    let (_dir, _arch, logits) = baseline();
    let sum: f32 = logits.iter().sum();
    assert!(
        logits.iter().any(|v| *v < 0.0),
        "a softmaxed head would be all non-negative"
    );
    assert!(
        (sum - 1.0).abs() > 1e-2,
        "logits summing to 1.0 means the head normalized: {sum}"
    );
}

/// Prefill (which skips the head) must advance every other side effect
/// exactly as `produce` does.
#[test]
fn prefill_then_decode_matches_all_produce() {
    let (dir, arch) = install();
    let vocab = VOCAB as usize;
    let tokens: Vec<i32> = (0..STEPS).map(|i| (i % vocab) as i32).collect();

    let mut all = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_all = vec![half::f16::ZERO; vocab];
    for (i, &t) in tokens.iter().enumerate() {
        all.produce(t, i, &mut logits_all).expect("produce");
    }

    let mut split = RealForwardRunner::open(&dir, arch.clone()).expect("open");
    let mut logits_split = vec![half::f16::ZERO; vocab];
    for (i, &t) in tokens.iter().enumerate() {
        if i + 1 < tokens.len() {
            split
                .produce_prefill(t, i, &mut logits_split)
                .expect("prefill");
        } else {
            split.produce(t, i, &mut logits_split).expect("produce");
        }
    }
    assert_eq!(logits_all, logits_split);
}

/// The decode hot path allocates no Metal buffers.
#[test]
fn decode_hot_path_allocates_no_gpu_buffers() {
    let (dir, arch) = install();
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    let mut logits = vec![half::f16::ZERO; VOCAB as usize];
    runner.produce(1, 0, &mut logits).expect("warm");
    let before = runner.gpu_buffer_allocations();
    for step in 1..6 {
        runner.produce(2, step, &mut logits).expect("produce");
    }
    assert_eq!(before, runner.gpu_buffer_allocations());
}

/// Generation is DETERMINISTIC on one warm runner: the same prompt twice in
/// one process must produce the same logits.
///
/// AGENTS.md Gotcha 27's check, in miniature. This family streams no experts
/// so there is no slot cache to permute, which makes the property easy --
/// and is exactly why it is worth pinning now rather than assuming it stays
/// true if anything is ever cached here.
#[test]
fn a_replayed_generation_is_bit_identical() {
    let (dir, arch) = install();
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    let mut a = vec![half::f16::ZERO; VOCAB as usize];
    let mut b = vec![half::f16::ZERO; VOCAB as usize];
    for step in 0..STEPS {
        runner.produce((step % 7) as i32, step, &mut a).expect("a");
    }
    runner.reset();
    for step in 0..STEPS {
        runner.produce((step % 7) as i32, step, &mut b).expect("b");
    }
    assert_eq!(a, b);
}

/// Rewrites one `manifest.json -> arch` field and returns the install dir.
///
/// Every refusal below is a BACKSTOP behind `arch_validation`, which compares
/// the passed `ArchConfig` against the manifest field by field and fires
/// first -- so each case has to patch the manifest AND pass the matching
/// arch, or it tests the wrong gate. That is the same shape
/// `real_forward_gptoss.rs` records for its three.
fn install_with_arch_field(key: &str, value: serde_json::Value) -> (std::path::PathBuf, String) {
    let (dir, _) = install();
    let path = dir.join("manifest.json");
    let mut m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    m["arch"][key] = value;
    std::fs::write(&path, m.to_string()).unwrap();
    (dir, key.to_string())
}

/// **THE NoPE GUARD.** An install declaring a nonzero `fullRopeTheta` is
/// refused.
///
/// This is the most important refusal in the family and the one no
/// numerical test can replace. The full-attention layers rotate NOTHING --
/// `layer_rope_theta` is literally 0 there in the checkpoint -- and a flow
/// handed a nonzero theta would rotate them. That model is finite, fluent
/// and wrong, and on untrained weights it is indistinguishable from the
/// correct one, so the refusal IS the protection.
#[test]
fn a_nonzero_full_rope_theta_is_refused() {
    let (dir, _) = install_with_arch_field("fullRopeTheta", serde_json::json!(500000.0));
    let mut arch = tiny_arch();
    arch.full_rope_theta = 500_000.0;
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("a rotating full layer is refused"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(msg.contains("NoPE"), "{msg}");
}

/// An all-full mask is refused: it would size every KV buffer at
/// `max_context` and decode a different model past the window (AGENTS.md
/// Gotcha 18).
#[test]
fn a_mask_with_no_sliding_layer_is_refused() {
    let ones: Vec<u8> = vec![1; LAYERS as usize];
    let (dir, _) = install_with_arch_field("fullAttentionLayerMask", serde_json::json!(ones));
    let mut arch = tiny_arch();
    arch.full_attention_layer_mask = vec![1; LAYERS as usize];
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("an all-full mask is refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("sliding"), "{err}");
}

/// This family is DENSE. An install declaring experts is refused rather than
/// half-served.
#[test]
fn a_nonzero_expert_count_is_refused() {
    let (dir, _) = install_with_arch_field("numExperts", serde_json::json!(8));
    let mut arch = tiny_arch();
    arch.num_experts = 8;
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("experts are refused"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("dense"), "{err}");
}

/// The sandwich norms are not optional. An install declaring
/// `ffnSandwichNorms: false` would have the flow add raw residuals, which is
/// a different architecture.
#[test]
fn dropping_the_sandwich_norms_is_refused() {
    let (dir, _) = install_with_arch_field("ffnSandwichNorms", serde_json::json!(false));
    let mut arch = tiny_arch();
    arch.ffn_sandwich_norms = false;
    let err = match RealForwardRunner::open(&dir, arch) {
        Ok(_) => panic!("no sandwich norms is refused"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("sandwich") || err.to_string().contains("adding"),
        "{err}"
    );
}

/// The flow refuses another family outright, so flow selection can never
/// fall through to tensor naming (`crates/runtime` Gotcha 3).
#[test]
fn another_familys_arch_is_refused() {
    let (dir, _) = install();
    let mut arch = tiny_arch();
    arch.family = model_io::ModelFamily::Llama;
    // The manifest still says museGlimmer, so `arch_validation` fires first
    // -- which is itself the point: BOTH gates refuse, and neither is
    // load-bearing alone.
    assert!(RealForwardRunner::open(&dir, arch).is_err());
}

fn tiny_arch() -> model_io::ArchConfig {
    turbospark_repack::tiny_muse_glimmer_arch(VOCAB, LAYERS)
}

/// A FROZEN DIGEST of the synthetic install's logits.
///
/// **THIS TEST EXISTS BECAUSE EVERY OTHER TEST IN THIS FILE IS
/// SELF-RELATIVE, which was measured rather than assumed.** The perturbation
/// cases above rebuild their own baseline inside the same binary, so a
/// mutation that changes the MATH for every arm equally leaves them all
/// green. Six mutations were run against this file before this test existed
/// and only ONE reddened -- dropping the attention output gate, which makes
/// `gate_proj` unreachable and so breaks a reachability invariant. Five
/// survived, each a fluent wrong model.
///
/// With this check in place, these all redden:
///
///   - dropping the `qk_scale_factor` multiply on Q,
///   - rotating the FULL layers instead of leaving them NoPE,
///   - using the PLAIN norm where the CENTERED one belongs,
///   - using the CENTERED norm on the final norm, where PLAIN belongs,
///   - dropping the `output_multiplier` before the softcap.
///
/// One still survives and is recorded in the module header: collapsing the
/// two epsilons, which FP16 cannot resolve.
///
/// A frozen reference over a deterministic fixture catches five failure
/// modes for the price of one comparison, because it is the only assertion
/// here that compares against something computed BEFORE the mutation.
///
/// It is a CHANGE DETECTOR, not a correctness claim: the weights are
/// untrained, so this says the flow's arithmetic is what it was, never that
/// it is right. Whether it is right is the real-model quality gate's
/// question. If the fixture's constants change, re-freeze deliberately and
/// say why -- a reference updated reflexively protects nothing.
///
/// **DEVICE-BRANCHED, since 2026-09-04, and the reason is a real
/// measurement rather than a guess.** This used to hash the FP16 bit
/// pattern of every logit and compare against one frozen `u64` -- exact
/// down to the last mantissa bit, on the reasoning that a deterministic
/// synthetic fixture on deterministic GPU arithmetic reproduces exactly.
/// That holds on real Apple Silicon (this exact fixture reproduces the old
/// hash bit-for-bit on this machine, run after run) and does NOT hold on
/// CI's `macos-latest` runner, which is a VIRTUALIZED "Apple Paravirtual
/// device" -- a genuinely different Metal implementation, not just a
/// different real chip. Two OTHER things about that specific device were
/// independently confirmed the same session this was found:
/// `dequant_int4_gemm_parity.rs`'s `maxTotalThreadsPerThreadgroup` reflection
/// (a proxy for register allocation, which drives how a compiler schedules
/// and reassociates a reduction loop) varies with shape there where it is
/// constant on real hardware, and it exposes no GPU performance-counter
/// hardware at all. Metal's fast-math is free to reassociate any floating
/// sum (AGENTS.md Gotcha 27 and its citations across this codebase), and
/// this flow's EIGHT layers each carry a sandwich of norm reductions
/// (`crates/runtime/CLAUDE.md`'s own museGlimmer bullet: four CENTERED
/// per-layer norms plus a plain final one, per-head norms, TWO epsilons) --
/// more reduction-heavy than any other family's flow, which is plausibly
/// why this is the family where it first became visible rather than
/// evidence that only this family's arithmetic differs; every other
/// family's synthetic-fixture test uses the weaker REACHABILITY pattern
/// (perturb a tensor, require movement) instead of an exact frozen
/// comparison, and per this file's own header, that pattern is blind to
/// small drift by construction.
///
/// **A PURE TOLERANCE REPLACEMENT WAS TRIED FIRST AND WAS PROVEN TOO WEAK.**
/// A uniform `LOGIT_TOL` in place of the hash, with no device branch, passed
/// every mutation this file already lists as reddening the old hash -- but a
/// mutation-check against `real_forward_qwen35_dflash.rs`'s documented
/// `DFLASH_RESIDUAL_EPS` bug (a subtle epsilon-scaling error that the OLD
/// exact digest there was specifically calibrated to catch, and that no
/// reachability test can) moved 181 of 257 sampled logits by only
/// 0.002-0.012, all under a 0.02 floor. So a coarse tolerance is provably
/// not a safe universal substitute for the exact comparison: it protects
/// against the LARGE mutations this file's own list names and silently
/// loses the SMALL, deliberately-subtle ones the exact hash exists for.
///
/// The fix is therefore BOTH, chosen by which device is actually running:
/// on real Apple Silicon, the ORIGINAL exact `fnv1a` comparison runs
/// unchanged, with its full original sensitivity (this is the only branch a
/// developer's machine or a self-hosted real-hardware runner ever takes).
/// On a virtualized device the exact comparison is not meaningful (the
/// hardware itself moves individual FP16 roundings), so a tolerant
/// comparison against the SAME frozen reference values runs instead, wide
/// enough to absorb cross-hardware FP reassociation and narrow enough to
/// still catch the large mutations this file's own list names.
/// `LOGIT_TOL` matches this codebase's own existing cross-backend numerical
/// tolerance (`PERPLEXITY_REL_TOLERANCE` is 2%, `crates/repack/CLAUDE.md`
/// Gotcha 38) rather than an invented number.
#[test]
fn the_synthetic_flows_arithmetic_is_frozen() {
    let (_dir, _arch, logits) = baseline();
    let context = gpu::MetalContext::new().expect("Metal device");
    let device_name = context.device().name().to_string();
    drop(context);
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized, not the real Apple Silicon this hash was \
             taken on; comparing against the frozen reference with a tolerance instead"
        );
        assert_eq!(logits.len(), FROZEN_LOGITS.len());
        for (i, (&got, &want)) in logits.iter().zip(FROZEN_LOGITS.iter()).enumerate() {
            let diff = (got - want).abs();
            let tol = LOGIT_TOL.abs_floor.max(want.abs() * LOGIT_TOL.rel_fraction);
            assert!(
                diff <= tol,
                "logit {i}: the muse_glimmer flow's arithmetic moved: got {got}, want {want} \
                 (diff {diff}, tolerance {tol})"
            );
        }
        return;
    }
    assert_eq!(
        fnv1a(&logits),
        FROZEN_LOGIT_HASH,
        "the muse_glimmer flow's arithmetic moved"
    );
}

/// FNV-1a over the logits' FP16 bit patterns.
///
/// Hand-rolled rather than pulling `sha2` into this crate's dev-dependencies:
/// the requirement is "any deterministic function of every bit", not
/// cryptographic strength, and a change detector that costs a dependency is
/// a change detector that gets deleted.
fn fnv1a(logits: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in logits {
        for b in half::f16::from_f32(*v).to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// Frozen 2026-08-15 against the fixture at VOCAB 64 / LAYERS 8 / STEPS 12,
/// measured on real Apple Silicon (Apple M4 Max). The real-hardware branch
/// of `the_synthetic_flows_arithmetic_is_frozen` compares against this
/// exactly; see that test's doc for why a virtualized device takes a
/// different, tolerant path instead.
const FROZEN_LOGIT_HASH: u64 = 10_299_009_919_897_358_122;

struct LogitTolerance {
    abs_floor: f32,
    rel_fraction: f32,
}

/// See `the_synthetic_flows_arithmetic_is_frozen`'s doc for why this is a
/// tolerance rather than exact equality, and why 2% matches this
/// codebase's existing `PERPLEXITY_REL_TOLERANCE` precedent rather than
/// being invented for this file.
const LOGIT_TOL: LogitTolerance = LogitTolerance {
    abs_floor: 0.02,
    rel_fraction: 0.02,
};

/// Frozen 2026-08-15 against the fixture at VOCAB 64 / LAYERS 8 / STEPS 12,
/// measured on real Apple Silicon (Apple M4 Max). These are the exact
/// values the original `fnv1a` hash (`10_299_009_919_897_358_122`) was
/// taken over; recovered rather than recomputed, so this reference did not
/// change when the comparison strategy did.
#[rustfmt::skip]
const FROZEN_LOGITS: [f32; VOCAB as usize] = [
    1.5996094, 1.2109375, -1.1884766, 0.2746582, -1.9296875, -0.98535156,
    -0.5419922, 1.8330078, -1.0683594, 1.1074219, 1.3535156, -1.1933594,
    0.77734375, 1.1113281, 0.08850098, 0.81689453, 1.4755859, -0.10430908,
    -1.6953125, 1.0205078, 0.6791992, -0.54785156, -0.8339844, -1.2392578,
    0.41918945, 0.07281494, 0.16479492, 0.52978516, 1.4228516, -1.0576172,
    -0.07092285, -1.4511719, -0.03677368, -0.6875, 1.2021484, -0.49609375,
    -0.5024414, -1.2089844, 0.31347656, -0.5576172, 1.3671875, 0.5205078,
    1.0576172, -2.4101563, 0.7006836, 2.2695313, -1.5683594, 0.48535156,
    -1.0214844, -0.04888916, -2.0429688, -0.5888672, 0.5673828, 0.7089844,
    -2.359375, -0.1763916, 0.2529297, 1.2529297, 2.4257813, -1.0898438,
    2.0332031, 0.4140625, -1.0800781, 1.2109375,
];
