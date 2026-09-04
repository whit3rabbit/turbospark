#![cfg(target_os = "macos")]
//! End-to-end proof of the DENSE, SUB-4-BIT `qwen3_5` decode path (ROADMAP's
//! 1-bit entry step 4, and its ternary entry at two bits): builds a tiny
//! install through the REAL repack
//! pipeline, opens it with `RealForwardRunner` -- which selects
//! `families/qwen/`'s flow from `ArchConfig.family` and its dense half from
//! `num_experts` -- and drives real decode steps on real Metal.
//!
//! Two things are being asserted at once and they fail in different ways.
//! **The dense branch** is the flow question: the layer's FFN half is one
//! gated MLP where Qwen 3.6 runs a router, a shared expert and eight routed
//! ones, and a branch that encoded nothing would leave every finiteness
//! assertion green. **The 1-bit dispatch** is the dtype question: before this
//! step a dtype-15 tensor reached `encode_gemv_any`'s named catch-all and
//! failed by name, so merely opening and producing proves the two new arms
//! are reached.
//!
//! What NO test here can prove is that the numbers are right. The kernels are
//! held to `turbospark_compute::quant_1bit` by
//! `crates/gpu/tests/dequant_1bit_gemv_parity.rs`, and the group size the
//! dispatch derives is pinned by unit tests beside the derivation; whether the
//! whole forward pass agrees with MLX on the real checkpoint is step 5's
//! cross-engine KL and nothing cheaper. Weights here are deterministic but NOT
//! trained, so nothing asserts on generated TEXT (AGENTS.md Gotcha 12).

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install, build_synthetic_qwen_gdn_dense_install_at_bits,
};
use turbospark_runtime::{LogitProducer, RealForwardRunner};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen35-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn build(dir: &std::path::Path) {
    build_synthetic_qwen_gdn_dense_install(dir, VOCAB, LAYERS, "tiny-bonsai")
        .expect("dense 1-bit qwen3_5 install builds");
}

/// The same at TWO bits (ROADMAP's ternary entry). One architecture, two
/// quantizations, so the fixture takes the width rather than being forked.
fn build_2bit(dir: &std::path::Path) {
    build_synthetic_qwen_gdn_dense_install_at_bits(dir, VOCAB, LAYERS, "tiny-ternary", 2)
        .expect("dense 2-bit qwen3_5 install builds");
}

/// The step-4 headline: a dense 1-bit install OPENS, where until this step
/// `open()` refused the family by name, and then decodes.
///
/// It goes through `peek_manifest_arch` rather than the builder's own
/// `ArchConfig`, exactly as `crates/cli` does, so validation against
/// `qwen_gdn_dense_27b()` is really exercised.
#[test]
fn a_dense_one_bit_qwen35_install_opens_and_decodes() {
    let dir = temp_dir("decodes");
    build(&dir);

    let peeked = turbospark_repack::peek_manifest_arch(&dir)
        .expect("a dense 1-bit manifest peeks against the Bonsai baseline");
    assert_eq!(peeked.num_experts, 0, "dense: no routed experts");
    assert_eq!(peeked.top_k_experts, 0, "dense: nothing to route to");

    let mut runner = RealForwardRunner::open(&dir, peeked).expect("a qwen3_5 install opens");
    assert_eq!(runner.vocab_size(), VOCAB as usize);

    runner.reset();
    let mut token = 5i32;
    for position in 0..6usize {
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut head)
            .expect("dense 1-bit produce succeeds");
        assert!(
            head.iter().all(|v| v.to_f32().is_finite()),
            "non-finite logit at position {position}"
        );
        // The producer contract: LOGITS, never probabilities (crate Gotcha
        // 1). This family has no softcap, so a distribution is the only shape
        // to rule out.
        let sum: f32 = head.iter().map(|v| v.to_f32()).sum();
        assert!(
            head.iter().any(|v| v.to_f32() < 0.0) || (sum - 1.0).abs() > 1e-2,
            "position {position} looks like a normalized distribution, sum {sum}"
        );
        token = head
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
    }
}

/// The dense FFN is really running rather than the layer being a
/// pass-through, and the 1-bit GEMV is really reading the weights.
///
/// This is the mutation check the decode case cannot make on its own:
/// attention alone still produces finite, non-normalized logits and the
/// residual stream still reaches the head, so a branch that encoded NOTHING
/// would pass every assertion above.
#[test]
fn the_dense_ffn_weights_reach_the_logits() {
    let a = temp_dir("ffn-a");
    let b = temp_dir("ffn-b");
    build(&a);
    build(&b);

    let baseline = first_logits(&a);
    // Same builder, same seeds: the two installs are identical until this.
    assert_eq!(baseline, first_logits(&b), "the pair starts identical");

    let patched = patch_packed_bytes(
        &b,
        &[
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
    );
    assert_eq!(
        patched,
        3 * LAYERS as usize,
        "expected gate/up/down on every layer"
    );
    assert_ne!(
        baseline,
        first_logits(&b),
        "the dense FFN weights do not reach the logits, so the branch is not running"
    );
}

/// The 1-bit embedding LOOKUP is running, and its own kernel rather than the
/// head's GEMV.
///
/// `embed_tokens` and `lm_head` are separate 1-bit tensors in this checkpoint
/// (it is untied), so perturbing the table alone moves the logits only
/// through `embed_lookup_int1`. Without the dtype-15 arm in
/// `encode_embed_any` this install cannot even open, so what this case really
/// pins is that the arm reads the table it was handed.
#[test]
fn the_one_bit_embedding_table_reaches_the_logits() {
    let a = temp_dir("embed-a");
    let b = temp_dir("embed-b");
    build(&a);
    build(&b);
    assert_eq!(
        first_logits(&a),
        first_logits(&b),
        "the pair starts identical"
    );

    let baseline = first_logits(&a);
    let patched = patch_packed_bytes(&b, &["language_model.model.embed_tokens.weight"]);
    assert_eq!(patched, 1, "exactly one embedding table");
    assert_ne!(
        baseline,
        first_logits(&b),
        "the 1-bit embedding table does not reach the logits"
    );
}

/// The COMPANION planes are read, which is the axis nothing about a length
/// check can see: FP16 and BF16 are the same width, and a 1-bit tensor
/// decoded with the wrong companion dtype -- or with the scale plane
/// ignored -- produces finite logits of exactly the right shape.
///
/// Perturbing the SCALES of the dense FFN and nothing else separates "the
/// packed bits are read" from "the whole affine triple is read".
#[test]
fn the_one_bit_scale_planes_reach_the_logits() {
    let a = temp_dir("scale-a");
    let b = temp_dir("scale-b");
    build(&a);
    build(&b);

    let baseline = first_logits(&a);
    assert_eq!(baseline, first_logits(&b), "the pair starts identical");

    let patched = patch_scale_planes(&b, &["mlp.down_proj.weight"]);
    assert_eq!(patched, LAYERS as usize, "one down_proj per layer");
    assert_ne!(
        baseline,
        first_logits(&b),
        "the FP16 scale plane does not reach the logits"
    );
}

/// The ternary entry's headline: a dense TWO-BIT install opens and decodes.
///
/// The same install shape as the 1-bit case above with one constant moved, and
/// what it proves is the two new dtype-16 arms: before them a dtype-16 tensor
/// reached `encode_gemv_any`'s named catch-all and `open()`'s
/// `readable_resident_dtype` guard refused the install outright, so merely
/// producing a logit means both arms are reached.
///
/// Note it cannot be passing through the 1-bit arm by accident:
/// `affine_group_size` checks the packed run against `rows * cols / 4`, and a
/// 2-bit tensor read at one bit is off by a factor of two.
#[test]
fn a_dense_two_bit_qwen35_install_opens_and_decodes() {
    let dir = temp_dir("decodes-2bit");
    build_2bit(&dir);

    let peeked = turbospark_repack::peek_manifest_arch(&dir)
        .expect("a dense 2-bit manifest peeks against the qwen3_5 baseline");
    assert_eq!(peeked.num_experts, 0, "dense: no routed experts");

    let mut runner = RealForwardRunner::open(&dir, peeked).expect("a 2-bit qwen3_5 install opens");
    assert_eq!(runner.vocab_size(), VOCAB as usize);

    runner.reset();
    let mut token = 5i32;
    for position in 0..6usize {
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut head)
            .expect("dense 2-bit produce succeeds");
        assert!(
            head.iter().all(|v| v.to_f32().is_finite()),
            "non-finite logit at position {position}"
        );
        let sum: f32 = head.iter().map(|v| v.to_f32()).sum();
        assert!(
            head.iter().any(|v| v.to_f32() < 0.0) || (sum - 1.0).abs() > 1e-2,
            "position {position} looks like a normalized distribution, sum {sum}"
        );
        token = head
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
    }
}

/// The 2-bit dense FFN and its COMPANION planes both reach the logits.
///
/// Two perturbations in one case because they answer one question at this
/// width: the packed run says the GEMV runs at all, and the FP16 scale plane
/// says the whole affine triple is read rather than just the bits. The scale
/// axis is the one no length check can see, FP16 and BF16 being the same
/// width.
#[test]
fn the_two_bit_weights_and_scale_planes_reach_the_logits() {
    let a = temp_dir("2bit-a");
    let b = temp_dir("2bit-b");
    let c = temp_dir("2bit-c");
    build_2bit(&a);
    build_2bit(&b);
    build_2bit(&c);

    let baseline = first_logits(&a);
    assert_eq!(baseline, first_logits(&b), "the pair starts identical");

    let patched = patch_packed_bytes(
        &b,
        &[
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ],
    );
    assert_eq!(
        patched,
        3 * LAYERS as usize,
        "expected gate/up/down on every layer"
    );
    assert_ne!(
        baseline,
        first_logits(&b),
        "the 2-bit dense FFN weights do not reach the logits"
    );

    let patched = patch_scale_planes(&c, &["mlp.down_proj.weight"]);
    assert_eq!(patched, LAYERS as usize, "one down_proj per layer");
    assert_ne!(
        baseline,
        first_logits(&c),
        "the FP16 scale plane does not reach the logits at two bits"
    );
}

/// The embedding lookup dispatched for a 2-bit table is the 2-BIT one.
///
/// **This case exists because the obvious ones do not discriminate**, which
/// was checked rather than assumed: swapping `encode_embed_lookup_int2` for
/// its 1-bit sibling in the dispatch leaves every other test in this file
/// green. The wrong kernel still reads the table, still produces finite
/// logits, and still moves them when the table is perturbed -- it simply reads
/// the WRONG ROW, because it strides by `D / 8` bytes where a 2-bit table
/// strides by `D / 4`.
///
/// So the perturbation here is a BYTE RANGE rather than a whole tensor: token
/// 5's row under the 2-bit stride is `[5D/4, 6D/4)`, and under the 1-bit
/// stride the same token reads `[5D/8, 6D/8)` -- disjoint ranges. Patching
/// only the first moves the logits under the correct kernel and cannot move
/// them under the wrong one.
#[test]
fn the_embedding_lookup_strides_at_the_tables_own_width() {
    let a = temp_dir("embed-stride-a");
    let b = temp_dir("embed-stride-b");
    build_2bit(&a);
    build_2bit(&b);

    let baseline = first_logits(&a);
    assert_eq!(baseline, first_logits(&b), "the pair starts identical");

    // `first_logits` decodes token 5, and HIDDEN is 128 in this fixture.
    let d = 128usize;
    let row_bytes_2bit = d / 4;
    let row_bytes_1bit = d / 8;
    let token = 5usize;
    let (lo, hi) = (token * row_bytes_2bit, (token + 1) * row_bytes_2bit);
    // The discriminating half, asserted rather than reasoned about: the range
    // being patched is one the 1-bit stride does not read for this token.
    let (wrong_lo, wrong_hi) = (token * row_bytes_1bit, (token + 1) * row_bytes_1bit);
    assert!(
        wrong_hi <= lo || wrong_lo >= hi,
        "the two strides overlap for token {token}, so this case cannot discriminate"
    );

    patch_byte_range(
        &b,
        "language_model.model.embed_tokens.weight",
        lo as u64,
        hi as u64,
    );
    assert_ne!(
        baseline,
        first_logits(&b),
        "token {token}'s own 2-bit embedding row does not reach the logits; the lookup is \
         striding at the wrong width"
    );
}

/// Flips bits in a byte range of one tensor's PACKED region, offsets relative
/// to the start of that region.
fn patch_byte_range(dir: &std::path::Path, name: &str, lo: u64, hi: u64) {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index loads");
    let entry = &index.entries[name];
    assert!(
        hi <= entry.size_bytes,
        "{name}: range {lo}..{hi} is outside"
    );
    let mut bytes = std::fs::read(&path).unwrap();
    let start = (entry.file_offset + lo) as usize;
    let end = (entry.file_offset + hi) as usize;
    for b in &mut bytes[start..end] {
        *b ^= 0xFF;
    }
    std::fs::write(&path, bytes).unwrap();
}

/// Flips bits in the PACKED region of every entry whose name ends in one of
/// `suffixes`, and nothing else, in place. Returns how many tensors it
/// touched, because a patch that silently found none makes its assertion
/// vacuous.
fn patch_packed_bytes(dir: &std::path::Path, suffixes: &[&str]) -> usize {
    patch_region(dir, suffixes, |e| (e.file_offset, e.size_bytes))
}

/// The same for the FP16 scale plane. Flips the LOW mantissa bit rather than
/// a whole nibble: a scale is a magnitude, and clearing its exponent would
/// zero the tensor, which is a weaker statement than perturbing it.
fn patch_scale_planes(dir: &std::path::Path, suffixes: &[&str]) -> usize {
    patch_region(dir, suffixes, |e| (e.scale_offset, e.scale_size))
}

fn patch_region(
    dir: &std::path::Path,
    suffixes: &[&str],
    region: fn(&model_io::ResidentIndexEntry) -> (u64, u64),
) -> usize {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index loads");
    let mut bytes = std::fs::read(&path).unwrap();
    let mut touched = 0;
    for entry in index.entries.values() {
        if !suffixes.iter().any(|s| entry.name.ends_with(s)) {
            continue;
        }
        let (offset, size) = region(entry);
        assert!(size > 0, "{} has an empty region to patch", entry.name);
        let start = offset as usize;
        for b in &mut bytes[start..start + size as usize] {
            *b ^= 0x01;
        }
        touched += 1;
    }
    std::fs::write(&path, bytes).unwrap();
    touched
}

/// The dense trunk's change detector, over a deterministic fixture.
///
/// **PAST POSITION 0, and that is the whole design of it.** A softmax over one
/// key is exactly 1.0 whatever the score, so attention at position 0 returns V
/// alone and NO q/k transform is observable there -- which makes
/// [`first_logits`] structurally blind to the per-head `q_norm`/`k_norm`, the
/// same way `qwen3moe`'s divergence test was until it was moved off position
/// 0. This walks eight positions and digests the last.
///
/// MEASURED, not reasoned: cut to a single position, the correct model and
/// one with the trunk flipped to `Centered` both digest to `312e17a0`. The
/// same pair over eight positions differs. A change detector on this flow has
/// to attend over a real span or it silently detects nothing.
///
/// It exists because every other case in this file is self-relative:
/// each rebuilds its own baseline inside the mutated binary, so a change to
/// the MATH that applies equally to both arms leaves all of them green
/// (AGENTS.md Gotcha 51). Demonstrated rather than assumed: pointing
/// `families/qwen/mod.rs`'s `encode_full_attention_block` call at
/// `QkNormConvention::Centered` -- a wrong model that still decodes -- left
/// this file's seven cases and `real_forward_qwen.rs`'s eight ALL green, and
/// was caught only by `qwen38_quality_gate` on a real 14 GB install two
/// minutes away. This constant catches it in 0.2 seconds.
///
/// Re-freezing it needs a stated reason. It is a change DETECTOR and not a
/// correctness claim: the fixture's weights are untrained, so it can say the
/// arithmetic moved and never that it is right.
///
/// **DEVICE-BRANCHED, since 2026-09-04** -- same reason and same fix shape
/// as `real_forward_muse.rs`'s `the_synthetic_flows_arithmetic_is_frozen`
/// (read that one for the full account, including why a pure tolerance
/// replacement was tried first and proven too weak against
/// `real_forward_qwen35_dflash.rs`'s documented `DFLASH_RESIDUAL_EPS` bug).
/// A bit-exact hash over GPU-computed FP16 values reproduces on real Apple
/// Silicon (this fixture still hashes to `9ce8b693` on real hardware,
/// unmoved by this change) and does not reproduce on CI's virtualized
/// `macos-latest` runner, whose Metal implementation was independently
/// confirmed (same session) to differ from real hardware in ways that
/// plausibly reassociate a reduction differently
/// (`dequant_int4_gemm_parity.rs`'s register-pressure reflection varies
/// with shape there where it is constant on real hardware; Metal's
/// fast-math is free to reassociate any floating sum, AGENTS.md Gotcha 27).
/// So real hardware runs the ORIGINAL exact digest comparison, unchanged
/// from before this fix, and only a virtualized device falls back to the
/// tolerant comparison against these same frozen values (2%, matching this
/// codebase's existing `PERPLEXITY_REL_TOLERANCE` precedent) -- far tighter
/// than a real regression: the `Centered`-vs-`Plain` q/k norm mutation this
/// digest exists to catch is architecturally the same class of bug as
/// `real_forward_muse.rs`'s documented mutations, which moved logits by
/// orders of magnitude more than the tolerance floor there.
#[rustfmt::skip]
const FROZEN_HEAD_LOGITS: [f32; VOCAB as usize] = [
        0.20507813, -4.75, 10.65625, 6.3554688, -5.2890625, 2.5019531,
        -3.8261719, 13.4609375, 4.5117188, 3.5371094, -1.4404297, -2.7304688,
        -4.4414063, 1.4912109, -3.75, 4.8632813, -17.28125, -1.3681641,
        4.7851563, 2.2128906, -3.3671875, -0.3178711, 4.1171875, -0.34350586,
        -1.9580078, -0.8828125, -10.1484375, 0.49121094, 0.34277344, 6.9140625,
        -2.4492188, -12.875, -2.2246094, -3.0273438, -3.7949219, 3.5644531,
        6.296875, -1.1425781, 1.6142578, 5.1601563, 6.9257813, 8.2578125,
        5.921875, -0.42700195, -0.5644531, 2.7695313, -3.7402344, -0.6508789,
        -6.5078125, 0.75097656, 1.2646484, 0.29907227, -2.921875, 3.6992188,
        3.2636719, -10.0546875, -0.91845703, -3.2519531, -0.47973633,
        -5.9765625, -2.6816406, -5.0742188, -3.3886719, 4.4101563, -11.453125,
        -1.8408203, 9.328125, 6.7460938, 4.0898438, -6.8203125, -2.0703125,
        -1.6132813, -0.60791016, -0.5317383, 6.0742188, 2.7050781, -1.2714844,
        8.3203125, -4.2851563, -1.3886719, 6.8671875, 8.375, 6.7382813,
        -0.18029785, -0.2668457, -3.4960938, -4.6835938, -4.015625, 1.5732422,
        8.890625, 2.5917969, -2.2363281, -11.921875, 0.1529541, -5.5898438,
        3.6777344, -7.4804688, 4.4375, -3.0859375, 10.484375, 3.5292969,
        5.59375, 3.9160156, 8.28125, -0.3173828, -0.5708008, -1.1494141,
        -5.2148438, -4.921875, 6.5, 4.6445313, 7.6914063, 2.2011719, 1.0507813,
        7.078125, -3.1113281, 0.80078125, 0.05606079, 0.85498047, 2.3769531,
        -1.2246094, -5.2265625, -4.6289063, 7.3203125, 7.1757813, 3.0,
        0.7597656, 2.1992188,
];

#[test]
fn the_dense_trunk_logits_have_a_frozen_digest() {
    let dir = temp_dir("qwen35-dense-digest");
    build(&dir);
    let arch = turbospark_repack::peek_manifest_arch(&dir).expect("peeks");
    let mut runner = RealForwardRunner::open(&dir, arch).expect("opens");
    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    // Eight positions, so the full-attention layer attends over a real span
    // rather than over a single key. The fixture's mask is
    // `qwen_hybrid_layer_mask(4)`, so layer 3 is the full-attention one.
    for position in 0..8usize {
        runner
            .produce(
                ((position * 7) % VOCAB as usize) as i32,
                position,
                &mut head,
            )
            .expect("produces");
    }
    let _ = std::fs::remove_dir_all(&dir);
    let context = gpu::MetalContext::new().expect("Metal device");
    let device_name = context.device().name().to_string();
    drop(context);
    if device_name.contains("Paravirtual") {
        println!(
            "device {device_name:?} is virtualized, not the real Apple Silicon this digest was \
             taken on; comparing against the frozen reference with a tolerance instead"
        );
        for (i, (&got, &want)) in head.iter().zip(FROZEN_HEAD_LOGITS.iter()).enumerate() {
            let got = got.to_f32();
            let diff = (got - want).abs();
            let tol = 0.02_f32.max(want.abs() * 0.02);
            assert!(
                diff <= tol,
                "logit {i}: the dense trunk logits moved: got {got}, want {want} \
                 (diff {diff}, tolerance {tol}); see FROZEN_HEAD_LOGITS's doc before re-freezing"
            );
        }
        return;
    }
    assert_eq!(
        digest(&head),
        FROZEN_DENSE_TRUNK_DIGEST,
        "the dense trunk's arithmetic moved"
    );
}

fn digest(logits: &[f16]) -> String {
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    model_io::hash_data(&bytes)[..8].to_string()
}

/// See `the_dense_trunk_logits_have_a_frozen_digest`'s doc for why this is
/// the real-hardware branch's comparison target. Frozen 2026-08-15 on real
/// Apple Silicon and still reproducing after the device-branch fix.
const FROZEN_DENSE_TRUNK_DIGEST: &str = "9ce8b693";

fn first_logits(dir: &std::path::Path) -> Vec<u16> {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("peeks");
    let mut runner = RealForwardRunner::open(dir, arch).expect("opens");
    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(5, 0, &mut head).expect("produces");
    head.into_iter().map(|v| v.to_bits()).collect()
}
