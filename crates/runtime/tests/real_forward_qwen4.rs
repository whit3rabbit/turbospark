//! The `qwen4_exp` decode flow (Phase 3 of its bring-up), on a synthetic
//! install -- the FIRST TIME this flow has ever run.
//!
//! `crates/runtime/CLAUDE.md` Gotcha 11 and Gotcha 23 are why this file is
//! shaped the way it is. A synthetic fixture's weights are UNTRAINED, so
//! "it decoded" is nearly worthless as evidence on its own: a dropped
//! injection, a wrong norm convention, or a PLE table never reached would
//! all produce equally meaningless-but-finite output. So most of the
//! assertions here are DOES THIS INPUT REACH THE MATH -- perturb one
//! tensor, require the logits to move -- and the file closes with a FROZEN
//! DIGEST, because Gotcha 23's own lesson is that perturbation cases alone
//! rebuild their own baseline inside the same binary and can all stay
//! green under a mutation that moves the math for every arm equally.
//!
//! Every case decodes at least 8 positions before comparing, past the
//! single-token position where a softmax over one key is exactly 1.0 and a
//! q/k transform (or a PLE hash depending on token history) is invisible
//! (Gotcha 23's own trap).

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

fn tempdir() -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("turbospark-qwen4-{}-{unique}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

// Must exceed the fixture's own `NGRAM_EOS_TOKEN_ID` (999, used as the
// n-gram context's fill value before any real history exists): a real
// checkpoint's EOS id always sits inside its vocabulary, and the derived
// hash multipliers assume every context slot -- fill value included -- is
// bounded by `vocab_size - 1` (`model_io::ngram_hash`'s overflow guard).
const VOCAB: i64 = 2048;

fn qwen4_install() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = turbospark_repack::build_synthetic_qwen4_exp_decode_install(&dir, VOCAB, "qwen4-p3")
        .expect("write install");
    (dir, arch)
}

/// A context window comfortably under the fixture's QSA indexer budget
/// (2048) and comfortably over the handful of positions any test here
/// decodes -- `RealForwardRunner::open`'s own `Auto` default resolves above
/// the budget on this fixture (no trained context declared), which the
/// indexer refusal then correctly refuses.
const TEST_MAX_CONTEXT: usize = 512;

/// Decodes `steps` greedy tokens and returns the last step's logits.
fn decode(dir: &std::path::Path, arch: &model_io::ArchConfig, steps: usize) -> Vec<f32> {
    let vocab = arch.vocab_size as usize;
    let mut runner = RealForwardRunner::open_with_max_context(dir, arch.clone(), TEST_MAX_CONTEXT)
        .expect("a qwen4_exp install opens");
    runner.reset();
    let mut token = 5i32;
    let mut last = Vec::new();
    for position in 0..steps {
        let mut logits = vec![f16::from_f32(0.0); vocab];
        runner
            .produce(token, position, &mut logits)
            .unwrap_or_else(|e| panic!("produce failed at position {position}: {e}"));
        let bad: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.to_f32().is_finite())
            .map(|(i, _)| i)
            .collect();
        assert!(
            bad.is_empty(),
            "non-finite logit at position {position}: {} of {vocab}, first {:?}",
            bad.len(),
            &bad[..bad.len().min(8)]
        );
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        last = logits.iter().map(|v| v.to_f32()).collect();
    }
    last
}

/// Overwrites one resident tensor's bytes in place with a constant BF16
/// pattern, leaving its length and every offset around it untouched
/// (AGENTS.md Gotcha 33: `open()` runs no receipt or SHA-256 check, so a
/// hypothesis costs milliseconds against a rebuild).
fn patch(dir: &std::path::Path, name: &str, value: u16) {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index");
    let entry = index
        .entries
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not in the resident index"));
    let mut bytes = std::fs::read(&path).unwrap();
    let start = entry.file_offset as usize;
    let end = start + entry.size_bytes as usize;
    for chunk in bytes[start..end].chunks_exact_mut(2) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, bytes).unwrap();
}

/// A large BF16 value: `0x4380` is 256.0, well outside the range this
/// fixture's own deterministic weights occupy, so a perturbation cannot be
/// missed by coincidentally landing near the original value.
const PERTURB: u16 = 0x4380;

/// Overwrites one ROUTED (packed) expert sub-tensor's bytes in place, inside
/// `packed_experts/layer_NN.bin`. `.mlp.switch_mlp.` is `Qwen4Exp`'s routed
/// marker (`crates/repack/CLAUDE.md`'s `routed_marker`), so those tensors
/// are NOT in `model_weights.bin` and `patch()` cannot reach them; `role` is
/// one of the layout's own sub-tensor keys ("gate"/"up"/"down", never
/// "gate_proj" -- `expert_blobs.rs` strips the projection suffix).
fn patch_routed_expert(dir: &std::path::Path, layer: usize, expert: usize, role: &str) {
    let layout = model_io::load_packed_experts_layout(
        dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("packed_experts/layout.json reads");
    let layer_layout = layout
        .layers
        .iter()
        .find(|l| l.layer == layer)
        .unwrap_or_else(|| panic!("no packed layer {layer} in the layout"));
    let entry = layer_layout
        .experts
        .iter()
        .find(|e| e.expert == expert)
        .unwrap_or_else(|| panic!("no expert {expert} in packed layer {layer}"));
    let sub = entry
        .sub_tensors
        .get(role)
        .unwrap_or_else(|| panic!("no {role:?} sub-tensor for layer {layer} expert {expert}"));
    let blob_path = dir.join("packed_experts").join(&layer_layout.file);
    let mut bytes = std::fs::read(&blob_path).expect("expert blob reads");
    let start = (entry.offset + sub.offset) as usize;
    let end = start + sub.size as usize;
    for b in bytes[start..end].iter_mut() {
        *b ^= 0xFF;
    }
    std::fs::write(&blob_path, bytes).expect("expert blob rewrites");
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

#[test]
fn a_qwen4_exp_install_opens_and_decodes() {
    let (dir, arch) = qwen4_install();
    assert_eq!(arch.family, model_io::ModelFamily::Qwen4Exp);
    let logits = decode(&dir, &arch, 8);
    let first = logits[0];
    assert!(
        logits.iter().any(|v| *v != first),
        "every logit is {first}: the head produced nothing"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Context above `compressed_attention.index_budget` is refused, because
/// this cut has no QSA indexer (`docs/QWEN4_PHASE0.md` section 5).
#[test]
fn context_above_the_indexer_budget_is_refused() {
    let (dir, arch) = qwen4_install();
    let budget = arch.compressed_attention.index_budget as usize;
    let err = RealForwardRunner::open_with_max_context(&dir, arch, budget + 1)
        .err()
        .expect("must refuse context above the indexer budget");
    let msg = err.to_string();
    assert!(
        msg.contains("indexer") || msg.contains("QSA"),
        "refusal message should name the indexer budget, got: {msg}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Context AT the budget is accepted -- the refusal is `>`, not `>=`.
#[test]
fn context_at_the_indexer_budget_is_accepted() {
    let (dir, arch) = qwen4_install();
    let budget = arch.compressed_attention.index_budget as usize;
    RealForwardRunner::open_with_max_context(&dir, arch, budget)
        .expect("context exactly at the budget must open");
    std::fs::remove_dir_all(&dir).ok();
}

/// A cache too small to hold one token's top-k routing is refused at open,
/// with the arithmetic named (Phase 4, AGENTS.md Gotcha 64). This is
/// independent of that gotcha's `2 * top_k` pipelining margin: this flow has
/// no chunked-prefill driver, so the only hard requirement is that
/// `top_k_experts` distinct experts fit the cache at all.
#[test]
fn a_cache_below_top_k_is_refused() {
    let (dir, arch) = qwen4_install();
    assert!(
        (turbospark_repack::TOP_K as i64) > 1,
        "this test needs a fixture whose top_k allows a below-top_k slot count"
    );
    let below = turbospark_repack::TOP_K - 1;
    let err = RealForwardRunner::open_with_options(&dir, arch, TEST_MAX_CONTEXT, below)
        .err()
        .expect("a cache below top_k must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains(&turbospark_repack::TOP_K.to_string()) && msg.contains(&below.to_string()),
        "refusal message should name both top_k and the requested slot count, got: {msg}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A cache exactly at `top_k` is accepted -- the refusal is `<`, not `<=`.
#[test]
fn a_cache_at_top_k_is_accepted() {
    let (dir, arch) = qwen4_install();
    RealForwardRunner::open_with_options(&dir, arch, TEST_MAX_CONTEXT, turbospark_repack::TOP_K)
        .expect("a cache exactly at top_k must open");
    std::fs::remove_dir_all(&dir).ok();
}

/// `attn_hyper_connection`'s `hc_norm` reaches the math: without it, the
/// GDN/attention branch's input (`mixed`) is invariant to this tensor.
#[test]
fn attn_hyper_connection_norm_moves_the_output() {
    let (dir, arch) = qwen4_install();
    let before = decode(&dir, &arch, 8);
    patch(
        &dir,
        "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight",
        PERTURB,
    );
    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "perturbing attn_hyper_connection's hc_norm left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// `mlp_hyper_connection`'s `block_inject_weight` reaches the math: without
/// it, `moe(mixed)`'s contribution to the residual is invariant to the
/// inject gate.
#[test]
fn mlp_hyper_connection_inject_gate_moves_the_output() {
    let (dir, arch) = qwen4_install();
    let before = decode(&dir, &arch, 8);
    patch(
        &dir,
        "language_model.model.layers.0.mlp_hyper_connection.block_inject_weight.weight",
        PERTURB,
    );
    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "perturbing mlp_hyper_connection's block_inject_weight left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The GDN branch (layer 0, mask 2) reaches the math.
#[test]
fn the_gdn_branch_moves_the_output() {
    let (dir, arch) = qwen4_install();
    assert_eq!(
        arch.full_attention_layer_mask[0], 2,
        "layer 0 must be GDN for this case to mean anything"
    );
    let before = decode(&dir, &arch, 8);
    patch(
        &dir,
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        PERTURB,
    );
    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "perturbing the GDN chain's in_proj_qkv left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The QSA-as-dense-attention branch (layer 2, mask 1) reaches the math.
#[test]
fn the_attention_branch_moves_the_output() {
    let (dir, arch) = qwen4_install();
    assert_eq!(
        arch.full_attention_layer_mask[2], 1,
        "layer 2 must be full attention for this case to mean anything"
    );
    let before = decode(&dir, &arch, 8);
    patch(
        &dir,
        "language_model.model.layers.2.self_attn.q_proj.weight",
        PERTURB,
    );
    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "perturbing the attention branch's q_proj left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The gated shared expert reaches the math.
#[test]
fn the_shared_expert_moves_the_output() {
    let (dir, arch) = qwen4_install();
    let before = decode(&dir, &arch, 8);
    patch(
        &dir,
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
        PERTURB,
    );
    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "perturbing the shared expert's gate_proj left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The routed experts reach the math. `.mlp.switch_mlp.` is Qwen4Exp's
/// routed marker, so this tensor lives in `packed_experts/`, not
/// `model_weights.bin` -- `patch_routed_expert` reaches it there.
#[test]
fn the_routed_experts_move_the_output() {
    let (dir, arch) = qwen4_install();
    let before = decode(&dir, &arch, 8);
    patch_routed_expert(&dir, 0, 0, "gate");
    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "perturbing routed expert 0's gate at layer 0 left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// PLE's `key_proj` reaches the math: the host-side hash and dequant chain
/// feeds a GPU projection that must actually move the residual.
#[test]
fn ple_key_proj_moves_the_output() {
    let (dir, arch) = qwen4_install();
    let before = decode(&dir, &arch, 8);
    patch(
        &dir,
        "language_model.model.layers.1.ple.key_proj.weight",
        PERTURB,
    );
    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "perturbing PLE's key_proj left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The n-gram TABLE ITSELF reaches the math -- perturbing the table's bytes
/// (not a projection weight) must move the output, which is what says the
/// host-side hash-and-dequant chain is really reading rows out of it rather
/// than, say, a fixed placeholder.
#[test]
fn the_ngram_table_moves_the_output() {
    let (dir, arch) = qwen4_install();
    let before = decode(&dir, &arch, 8);

    let ngram_path = dir.join("ngram_table").join("rows.bin");
    let mut bytes = std::fs::read(&ngram_path).expect("ngram table bytes");
    for b in bytes.iter_mut() {
        *b ^= 0xFF;
    }
    std::fs::write(&ngram_path, bytes).unwrap();

    let after = decode(&dir, &arch, 8);
    assert!(
        distance(&before, &after) > 0.0,
        "flipping every byte of the n-gram table left the logits bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A FROZEN DIGEST of the synthetic install's logits.
///
/// The perturbation cases above are each SELF-RELATIVE (they rebuild their
/// own baseline inside the same binary), which is exactly the shape
/// `crates/runtime/CLAUDE.md` Gotcha 23 warns catches almost nothing: a
/// mutation that moves the underlying math for every arm equally (the wrong
/// hyper-connection formula, a dropped `/ hc_count`, `.sum()` where `.mean()`
/// belongs, the un-normed value read as normed, injecting into the NORMED
/// stream instead of `raw`) would leave every case above green, because the
/// "before" and "after" runs in each case share the same wrong arithmetic.
/// This digest is the one assertion here that compares against something
/// computed BEFORE any of that -- a value frozen once, off a run that was
/// itself never checked against anything but its own internal consistency,
/// exactly as `real_forward_muse.rs`'s own digest is.
///
/// It is a CHANGE DETECTOR, not a correctness claim: the weights are
/// untrained, so this says the flow's arithmetic is what it was, never that
/// it is right. Whether it is right needs a real checkpoint, which Phase 5
/// (blocked on disk space as of this writing) has not yet provided.
#[test]
fn the_synthetic_flows_arithmetic_is_frozen() {
    let (dir, arch) = qwen4_install();
    let logits = decode(&dir, &arch, 8);
    assert_eq!(
        fnv1a(&logits),
        FROZEN_LOGIT_HASH,
        "the qwen4_exp flow's arithmetic moved"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// FNV-1a over the logits' FP16 bit patterns, `real_forward_muse.rs`'s own
/// hand-rolled digest (no `sha2` dependency for a change detector that
/// needs no cryptographic strength).
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

/// Frozen on the real Metal run that first exercised this flow
/// (2026-09-02). A change detector, not a correctness claim: the weights
/// are untrained, so this pins the arithmetic as it stands, never that it
/// is right. Re-freeze only with a stated reason.
const FROZEN_LOGIT_HASH: u64 = 6153910550313830528;
