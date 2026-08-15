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

fn first_logits(dir: &std::path::Path) -> Vec<u16> {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("peeks");
    let mut runner = RealForwardRunner::open(dir, arch).expect("opens");
    runner.reset();
    let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
    runner.produce(5, 0, &mut head).expect("produces");
    head.into_iter().map(|v| v.to_bits()).collect()
}
