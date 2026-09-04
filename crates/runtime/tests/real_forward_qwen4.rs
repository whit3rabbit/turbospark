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

/// A FROZEN reference over the synthetic install's logits.
///
/// The perturbation cases above are each SELF-RELATIVE (they rebuild their
/// own baseline inside the same binary), which is exactly the shape
/// `crates/runtime/CLAUDE.md` Gotcha 23 warns catches almost nothing: a
/// mutation that moves the underlying math for every arm equally (the wrong
/// hyper-connection formula, a dropped `/ hc_count`, `.sum()` where `.mean()`
/// belongs, the un-normed value read as normed, injecting into the NORMED
/// stream instead of `raw`) would leave every case above green, because the
/// "before" and "after" runs in each case share the same wrong arithmetic.
/// This is the one assertion here that compares against something computed
/// BEFORE any of that -- values frozen once, off a run that was itself never
/// checked against anything but its own internal consistency, exactly as
/// `real_forward_muse.rs`'s own reference is.
///
/// It is a CHANGE DETECTOR, not a correctness claim: the weights are
/// untrained, so this says the flow's arithmetic is what it was, never that
/// it is right. Whether it is right needs a real checkpoint, which Phase 5
/// (blocked on disk space as of this writing) has not yet provided.
///
/// **DEVICE-BRANCHED, since 2026-09-04** -- this was a bit-exact `fnv1a`
/// hash over the FP16 bit pattern of every logit; see
/// `real_forward_muse.rs`'s identical fix for the full account of why a
/// pure tolerance replacement is not a safe universal substitute (proven
/// too weak against `real_forward_qwen35_dflash.rs`'s documented
/// `DFLASH_RESIDUAL_EPS` bug) and why the fix is instead to run the
/// ORIGINAL exact hash on real Apple Silicon, where it always reproduces,
/// and fall back to a tolerant comparison against the same frozen values
/// only on CI's virtualized `macos-latest` runner, whose Metal
/// implementation is genuinely different rather than just a different real
/// chip. `FROZEN_LOGITS` is recovered from the same real-hardware run the
/// old hash (`6153910550313830528`) was taken over, so this reference did
/// not change when the comparison strategy did.
#[test]
fn the_synthetic_flows_arithmetic_is_frozen() {
    let (dir, arch) = qwen4_install();
    let logits = decode(&dir, &arch, 8);
    std::fs::remove_dir_all(&dir).ok();
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
            let tol = 0.02_f32.max(want.abs() * 0.02);
            assert!(
                diff <= tol,
                "logit {i}: the qwen4_exp flow's arithmetic moved: got {got}, want {want} \
                 (diff {diff}, tolerance {tol})"
            );
        }
        return;
    }
    assert_eq!(
        fnv1a(&logits),
        FROZEN_LOGIT_HASH,
        "the qwen4_exp flow's arithmetic moved"
    );
}

/// FNV-1a over the logits' FP16 bit patterns. See
/// `real_forward_muse.rs`'s identical helper for why this is hand-rolled.
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

/// The real-hardware branch's comparison target, taken over the same
/// real-hardware run `FROZEN_LOGITS` below was recovered from.
const FROZEN_LOGIT_HASH: u64 = 6_153_910_550_313_830_528;

/// Frozen on the real Metal run that first exercised this flow
/// (2026-09-02). A change detector, not a correctness claim: the weights
/// are untrained, so this pins the arithmetic as it stands, never that it
/// is right. Re-freeze only with a stated reason.
#[rustfmt::skip]
const FROZEN_LOGITS: [f32; VOCAB as usize] = [
    3.3476563, -2.2207031, 0.16235352, 2.1425781, -5.4414063, 0.06640625, 1.8349609, -1.9785156,
    -2.734375, 1.3242188, -7.046875, 1.5078125, 6.2734375, -1.1201172, 1.3125, 0.020950317,
    -3.4804688, 0.19580078, 0.34204102, 2.7753906, 1.3183594, -1.7255859, 4.4492188, 5.1289063,
    -3.7832031, -3.2890625, 6.4296875, -1.4873047, -2.1289063, -2.2226563, 3.1523438, -7.2265625,
    6.28125, 3.390625, -0.1694336, -0.8154297, -2.0429688, -0.050842285, 4.2695313, 2.3125,
    -1.7910156, 0.9814453, -0.44384766, -0.375, -1.8408203, -7.78125, 4.4960938, -4.4335938,
    7.3867188, -3.0898438, 3.1875, 4.15625, 0.3762207, -2.7070313, 0.75341797, -7.2851563,
    1.6289063, -0.90478516, -0.21081543, 0.11791992, 2.1132813, -1.1289063, -0.90771484, -2.1777344,
    -2.5234375, -0.3239746, -1.53125, -0.042297363, 1.9248047, 1.8349609, 1.5595703, 3.7910156,
    2.5, -2.2011719, 1.0849609, -1.5224609, 1.2373047, 3.1054688, 2.0058594, 2.8203125,
    -0.67089844, -3.2597656, 1.3203125, -2.6875, 4.203125, 7.2070313, -5.5117188, 1.2314453,
    0.31420898, -0.9433594, 3.5136719, -2.7890625, 6.6601563, -2.9082031, -1.2236328, 0.47485352,
    -2.7226563, 1.0888672, -3.6660156, 1.1347656, -4.5546875, -2.7695313, -2.8046875, 7.0117188,
    2.7890625, 1.6591797, 5.46875, -2.5703125, -0.044311523, -6.5742188, 0.5810547, -1.5263672,
    -0.44848633, -1.1503906, 2.3320313, 2.2109375, 0.92578125, 1.0654297, 5.59375, -0.00945282,
    -4.9335938, -3.1367188, -6.9179688, 3.3632813, 2.0371094, -5.5351563, 6.0195313, 0.47485352,
    -2.6933594, 7.4726563, -8.7265625, 3.7363281, -4.8007813, 3.2890625, -1.4697266, 0.51953125,
    -0.23168945, 3.4921875, -1.3125, 5.6953125, 1.5556641, 6.4375, 6.296875, -0.13720703,
    -1.0136719, -3.1269531, -4.9023438, -0.64941406, 3.1171875, -6.6875, -2.9550781, -8.5859375,
    -4.265625, -6.1953125, 0.4814453, 3.3164063, 6.328125, -1.7939453, -4.171875, 3.0898438,
    -0.9506836, 0.31347656, 0.54833984, 6.390625, 1.3476563, 1.5839844, 2.4863281, -3.2226563,
    4.75, 3.671875, 1.4794922, 0.13305664, -0.5341797, -0.89160156, 2.7832031, 1.2451172,
    7.5664063, -3.9140625, 3.7109375, -0.5839844, -0.17175293, 7.1601563, 3.8320313, 2.8691406,
    2.9511719, -1.8476563, -3.703125, 4.8046875, -0.06097412, -3.375, 0.48291016, 6.6523438,
    -0.80908203, -0.3095703, -0.6899414, 1.8632813, -3.1484375, -1.0820313, 0.7807617, -1.9345703,
    7.1015625, -0.050933838, -3.1132813, -0.7211914, 1.4169922, 1.6582031, 4.7890625, 4.3828125,
    -0.51708984, 0.45751953, 4.8164063, -0.33911133, -2.6777344, -2.3222656, 0.8105469, 2.7851563,
    -3.125, 0.75146484, 9.8203125, -1.5830078, 1.0849609, 1.0039063, 1.1757813, -1.8486328,
    -6.7109375, 6.2578125, 1.9384766, -0.057525635, -6.3515625, -0.37524414, -1.2451172, 0.84716797,
    1.9033203, 2.265625, -3.2128906, -1.0927734, -1.1777344, -6.0664063, -1.6318359, 6.7226563,
    2.7460938, -1.2939453, -1.8681641, 1.5927734, -4.8515625, -3.7207031, -3.1523438, 2.8789063,
    0.6245117, -0.8754883, 1.0996094, 2.6347656, 1.8515625, -4.109375, -2.5449219, 4.4882813,
    1.7851563, 4.3515625, -6.1210938, -3.5996094, 3.1425781, 3.703125, -1.1855469, -5.109375,
    -5.7695313, 2.6191406, -3.2929688, 2.3808594, -5.140625, 0.82470703, 7.4726563, 0.34228516,
    -0.3305664, 2.6308594, -3.4511719, 5.7539063, -1.7041016, -7.3476563, -3.4921875, 4.2421875,
    1.8525391, -1.1826172, 0.5136719, -0.8071289, -0.37841797, 3.0136719, -4.7304688, -0.30004883,
    -8.3359375, 2.0214844, 0.640625, -2.1875, -6.90625, -1.7548828, -4.6835938, 3.6796875,
    1.0107422, -7.7539063, 1.7353516, 5.796875, -0.65478516, -3.8535156, 4.0859375, -3.5722656,
    4.5859375, -1.0273438, 1.3935547, 6.6796875, 2.6679688, 1.8056641, 0.4765625, -1.4257813,
    2.75, 0.1274414, 1.3212891, 1.5302734, 0.46972656, 5.0507813, 0.64746094, -4.1054688,
    -0.10430908, 1.2910156, 3.4023438, -0.39892578, -4.53125, 4.1132813, 4.8398438, -3.0859375,
    2.984375, 1.84375, 3.7167969, 0.2668457, -3.7226563, -5.4570313, 0.3232422, 1.0615234,
    -1.7089844, -0.044891357, -5.3359375, 2.265625, 5.8945313, 3.59375, 0.5258789, -1.8388672,
    7.4453125, 2.6621094, 2.4726563, -0.76220703, -2.859375, -0.8847656, -1.8603516, 1.3896484,
    -0.075805664, 5.8984375, 0.17810059, 0.088012695, -0.1381836, 1.2177734, 7.140625, 0.5371094,
    1.7636719, 3.9003906, 0.13745117, -4.4570313, -2.3691406, 7.8359375, -0.45922852, 3.0019531,
    4.4960938, 2.6308594, 3.4511719, -8.46875, 2.0644531, -5.0234375, -4.5117188, 5.7734375,
    -1.9365234, -2.8671875, 2.7988281, -7.4296875, 1.6767578, -0.7714844, 0.66308594, 0.69091797,
    3.2246094, 0.89160156, -4.1640625, -5.7851563, -2.234375, 3.9394531, 3.7929688, -6.28125,
    2.7890625, -1.6777344, 1.5917969, -0.45507813, -0.8125, 0.21508789, -1.3681641, -8.28125,
    -5.09375, 5.75, -1.6933594, -0.33789063, -3.3027344, 1.8544922, 2.1757813, -1.5273438,
    0.16418457, 2.5664063, 1.0253906, 2.4707031, 0.44189453, 4.6953125, 0.3388672, 1.2958984,
    1.9179688, -1.6865234, 2.1621094, 4.375, 1.5058594, 7.46875, -2.3378906, 6.0429688,
    -3.0878906, -0.26953125, 2.4414063, -2.109375, -7.3789063, 1.7070313, -0.8666992, 6.7421875,
    0.18493652, 2.2695313, 2.1972656, 2.90625, -2.3339844, -5.6289063, -1.578125, -2.1875,
    1.7929688, 1.2646484, 5.9453125, -2.3710938, -4.2851563, -1.109375, 3.1660156, -3.8359375,
    -0.7397461, -2.0488281, 5.0039063, -1.7197266, 1.4072266, -1.2636719, 1.9316406, 3.015625,
    -2.2402344, 2.5703125, -1.7773438, 2.9375, -2.7402344, -1.2597656, 2.3828125, 1.1259766,
    3.4257813, -1.7099609, -1.1923828, -0.7939453, -4.921875, 1.0253906, 2.2929688, -3.9121094,
    5.578125, 1.0654297, 2.6328125, -2.4101563, -4.8085938, 4.3632813, -2.6972656, -3.28125,
    -0.14099121, 1.5292969, -2.2617188, -0.70703125, -1.9716797, 3.2050781, 2.4511719, 1.0263672,
    -0.59716797, 1.3916016, 4.421875, -3.1171875, 0.31469727, 1.0585938, 2.9394531, -4.5664063,
    1.7353516, 1.8300781, -1.328125, -2.3574219, 2.0839844, 0.3942871, 4.2851563, 0.5410156,
    0.0703125, -3.7871094, -1.6611328, -0.5878906, -0.027450562, -5.15625, -0.99609375, -5.2539063,
    -0.019744873, -0.041656494, -0.4699707, 2.0644531, 2.9433594, 0.59033203, 3.4179688, 0.3449707,
    1.8955078, -2.8945313, 2.5859375, 0.77001953, 3.9394531, 6.1132813, -1.7265625, -6.7890625,
    2.9160156, 1.3154297, 0.8432617, -0.6274414, -0.7524414, 2.0488281, 0.10668945, 1.4345703,
    -0.47558594, -6.1796875, -0.5258789, 5.1289063, -0.81396484, 0.22473145, 4.1914063, -1.4853516,
    -4.1132813, 2.4589844, 6.8046875, 6.8398438, 1.9013672, -4.6953125, 2.3203125, -4.0039063,
    -0.27539063, -6.3945313, -0.6386719, -7.0351563, -0.31445313, 2.7421875, -2.9667969, -2.7988281,
    3.578125, 0.70703125, -1.8173828, -2.9316406, -2.7578125, -3.7226563, 5.9140625, 1.8476563,
    0.29492188, 3.6054688, 1.3144531, 3.5039063, -4.9296875, 3.2851563, -0.013160706, 0.18859863,
    0.27148438, -0.6904297, 0.9975586, 11.890625, 2.5761719, 2.3125, -1.8066406, -1.7460938,
    -4.8320313, 1.8857422, 1.8291016, -1.1044922, 1.7832031, -3.6875, -0.8129883, -1.7451172,
    -1.0009766, 3.5117188, -6.1171875, -0.69433594, 1.109375, -3.8046875, -1.4248047, -1.6523438,
    -4.1640625, 1.5966797, 2.1445313, -0.92822266, 0.17932129, -3.4609375, 0.7236328, -2.8925781,
    0.265625, -2.3925781, 5.8828125, 1.6533203, 0.98779297, 3.5996094, 2.3847656, -2.4375,
    5.6875, 1.1347656, 2.203125, 1.1210938, 0.71484375, 3.828125, 0.4765625, 1.9003906,
    3.9648438, -2.4785156, 4.1679688, 2.0761719, -3.5390625, 2.015625, 3.9375, -2.9082031,
    0.059539795, -4.9804688, -0.7260742, 0.7553711, -2.3027344, 3.4863281, -3.359375, 5.921875,
    2.8339844, -3.1191406, -1.5195313, -2.2714844, -0.9785156, -4.3242188, -0.5126953, -3.0292969,
    1.3320313, 0.36938477, 1.6044922, 2.5878906, -0.56884766, -4.9882813, 3.796875, 1.6513672,
    -2.5585938, -0.41137695, 1.2705078, 0.68408203, 5.0507813, 0.09313965, 2.3769531, 0.6791992,
    -0.94921875, 0.76416016, -3.0742188, 4.8203125, -1.0253906, -4.0195313, -8.3828125, -4.078125,
    0.40576172, -0.86083984, -5.3398438, 1.9736328, -0.042785645, 1.7792969, -3.6601563, -3.3085938,
    7.6445313, 1.0703125, -2.8242188, -3.2734375, 0.5185547, 1.171875, -2.9863281, 4.1484375,
    4.3398438, -3.6054688, 3.3554688, 2.3867188, 1.7480469, -7.0585938, -4.8203125, 0.5541992,
    2.8496094, -2.3867188, -1.3789063, -0.69091797, -3.96875, 0.43066406, -0.9746094, 2.7597656,
    -6.6210938, -5.8867188, 3.1113281, 4.0117188, -0.57373047, -3.4277344, 0.8803711, -0.6875,
    -1.21875, -3.6171875, 1.1181641, 2.7988281, -3.2207031, 5.7421875, -0.51416016, -1.2177734,
    -3.5273438, 0.013282776, -1.4746094, 1.0058594, -1.6474609, -4.0273438, -3.2070313, 0.27954102,
    3.3554688, 2.5566406, -5.3945313, -4.125, 0.49926758, 2.0273438, -2.1953125, 1.6474609,
    2.2402344, 4.8398438, 3.0429688, -1.4638672, 3.0175781, -0.89404297, -5.0742188, 0.21923828,
    0.47875977, 6.0078125, -1.8535156, 1.4023438, -0.25268555, 2.4394531, 2.328125, -3.2597656,
    -6.8359375, 5.046875, -0.5307617, 4.140625, 7.2148438, 1.6337891, -6.9453125, -1.9970703,
    2.1699219, 2.5859375, 1.359375, 0.16052246, 3.5078125, 1.5791016, 1.3730469, -1.4609375,
    -2.421875, 6.5, -6.4414063, -4.5546875, -0.85839844, -0.38867188, 0.43847656, -1.9970703,
    1.4130859, 1.0517578, -7.1992188, -0.29956055, -0.4296875, 2.5605469, -4.1992188, 2.2597656,
    -4.453125, 0.82373047, 2.9433594, -0.96484375, 1.03125, 6.6289063, 0.059631348, 2.0546875,
    -3.2246094, -2.0371094, -3.984375, 3.1054688, 4.78125, -1.2871094, -0.82910156, -0.30249023,
    -10.359375, -3.1972656, -3.9746094, 1.5, -2.1230469, -0.69091797, 0.2709961, -2.15625,
    3.8671875, 0.7519531, -3.3027344, -3.4804688, 3.6699219, 5.2421875, -6.0117188, -5.8398438,
    -0.12866211, 3.09375, -4.0546875, -1.1806641, -7.484375, -5.1289063, 0.34057617, -0.2052002,
    1.0888672, 3.8359375, 0.45507813, 3.734375, 4.6484375, 1.5097656, 1.6845703, 4.4765625,
    0.1385498, 1.9335938, -0.67285156, 3.3183594, -3.8144531, 3.3515625, -1.7099609, 2.3007813,
    1.5166016, -2.2636719, -0.05102539, -4.5390625, 3.0292969, 0.95654297, -6.0820313, 1.5341797,
    -2.9707031, 1.1328125, 2.3339844, -3.8554688, -5.046875, -2.9824219, 2.5214844, 1.9794922,
    6.4648438, -2.8828125, 3.9042969, 1.4003906, 0.028320313, -5.0820313, -0.36938477, -4.6835938,
    -5.53125, 0.9584961, 1.7509766, 5.5703125, -3.53125, -3.7089844, 4.1171875, 1.4931641,
    5.5703125, 0.2454834, 0.2758789, 1.3339844, -0.5229492, -0.87158203, 0.6411133, -7.796875,
    -3.4160156, 4.34375, -0.8623047, -0.55371094, -2.6132813, 0.11706543, -0.18847656, 3.6914063,
    -0.40600586, 3.0761719, 4.4765625, 4.015625, -0.15539551, 0.5810547, -2.2011719, 1.7412109,
    6.4648438, -1.5087891, 5.3125, -1.1513672, 0.5517578, -3.9960938, 0.6225586, -4.703125,
    5.59375, 2.8125, -2.7578125, -2.1796875, -0.22583008, 2.4160156, 2.1894531, 3.078125,
    2.1777344, -3.2480469, 3.2207031, 3.0292969, 0.35473633, 1.40625, 0.44677734, 5.46875,
    0.7885742, -1.1035156, 1.7216797, 3.5585938, -1.3613281, 8.7109375, 0.6201172, -4.4921875,
    -3.7910156, -6.6054688, 5.2460938, -4.5117188, 1.3632813, 2.3867188, -0.07305908, 3.4238281,
    2.7460938, -3.8378906, 7.53125, -5.8320313, -2.1699219, 0.11468506, 0.4177246, -2.65625,
    0.4038086, -3.8417969, -2.2773438, -1.2353516, -1.2460938, -0.15942383, 0.0054397583, 0.7807617,
    -0.8208008, -4.2148438, 0.34643555, 5.5117188, -3.0058594, 2.3300781, -0.61083984, -4.078125,
    -0.18432617, -8.2421875, -1.0107422, -2.6777344, 2.1953125, -1.8466797, -1.2529297, -0.94970703,
    -0.24938965, -0.3239746, 1.9277344, -2.09375, 0.6352539, -2.6894531, 3.2421875, 1.7402344,
    1.1445313, 2.0332031, 0.93896484, 0.30932617, -5.8515625, -4.515625, 1.5595703, 2.8925781,
    -0.44189453, 0.9326172, -2.53125, 0.46191406, 2.2675781, 2.484375, 0.5292969, -1.1503906,
    -8.46875, -0.7348633, -0.67626953, 7.0, -6.171875, 3.1503906, 2.9960938, -0.37939453,
    1.0214844, -5.2460938, 1.1425781, -2.5039063, 3.1757813, 2.4433594, -4.7226563, 2.4902344,
    1.6904297, -2.296875, 1.8603516, -0.7734375, 4.7539063, -3.984375, -1.9755859, 8.515625,
    5.625, 1.8369141, -6.046875, -1.4853516, -0.79052734, 4.6835938, -5.5273438, -5.5507813,
    -3.2910156, -2.6699219, 0.9399414, 1.1630859, -1.0673828, 1.8105469, -4.0429688, 6.4804688,
    6.1679688, -3.7558594, 4.9023438, -2.5371094, -5.0625, -0.29785156, -3.3125, 1.8564453,
    -2.6171875, -0.82421875, -3.3652344, -6.46875, -1.9287109, -2.7734375, -2.03125, -1.4707031,
    -4.1523438, 0.49682617, -5.5820313, 5.6484375, 0.31958008, 1.7080078, 2.0175781, -2.4707031,
    -4.8320313, 0.2705078, 3.875, -1.5019531, -0.8745117, -0.6279297, 1.1699219, 3.5175781,
    -1.5605469, -3.6132813, -5.953125, 3.8984375, -2.7226563, 0.10430908, -1.3154297, 1.2949219,
    -2.9648438, -0.3671875, -0.7973633, -5.6054688, -1.0859375, 3.1425781, 4.2421875, -5.6796875,
    3.1445313, 1.9423828, -6.8554688, 1.0859375, 5.0742188, -2.5976563, 2.2929688, 0.60498047,
    -0.14123535, 1.3056641, 2.328125, 3.5605469, -6.7226563, 0.6328125, 0.51904297, -0.79833984,
    5.1523438, -1.4404297, -2.0644531, 1.1269531, -2.828125, 0.328125, -1.40625, -1.9423828,
    -3.9199219, -1.3173828, -5.3242188, -0.103515625, -7.515625, 2.2519531, -0.32910156, -3.2734375,
    2.6640625, -0.4597168, 0.50341797, -0.53222656, 3.03125, -0.18334961, -0.50390625, 0.8022461,
    1.8769531, -4.6171875, -0.9663086, 7.1328125, 1.5009766, 5.40625, -0.4321289, -5.3828125,
    -3.7050781, 2.96875, 6.5078125, 6.875, -1.5351563, -2.1015625, 2.5976563, -0.50878906,
    2.0605469, -2.3183594, -2.3691406, 1.5087891, -1.109375, -5.7304688, 7.6914063, 0.24060059,
    1.6416016, 1.4560547, 0.28344727, -0.060668945, 0.13549805, -6.9453125, -3.5605469, 1.3603516,
    3.8222656, 4.5625, 2.296875, 3.6894531, 0.5317383, 3.8613281, -5.7265625, 3.6601563,
    1.8798828, 4.6132813, -0.24169922, -5.2851563, -0.47265625, -4.3242188, -3.9667969, 5.5429688,
    -1.0019531, 6.0742188, -3.0839844, 1.1201172, -0.10876465, -4.6992188, 0.8227539, 4.7851563,
    -7.2578125, 0.8852539, 0.015914917, -5.8789063, 3.1679688, -1.9931641, 0.6069336, 0.78027344,
    0.9628906, 1.4990234, -1.9541016, 6.8046875, 5.6171875, -1.2734375, -0.5966797, -2.7480469,
    2.0859375, -2.0429688, -0.87353516, -3.8046875, 3.0136719, 3.9511719, 0.5288086, -3.1699219,
    -3.2675781, -2.2675781, 1.6884766, 5.65625, -3.7539063, -1.7099609, 3.6914063, -1.7753906,
    1.5654297, -1.3085938, 3.7421875, -3.5996094, -2.9824219, 0.58984375, 2.171875, 3.3066406,
    -0.8310547, 3.5839844, 0.4038086, -0.4807129, -4.5742188, 2.0664063, -0.11004639, -4.4609375,
    -5.125, -0.24047852, -1.1191406, 4.9882813, 2.0273438, 3.8476563, -0.29663086, -0.8041992,
    -1.4960938, 1.5947266, 1.2871094, 0.7504883, 1.7490234, 2.9140625, -0.92626953, 1.5292969,
    3.5292969, 3.7578125, -0.008003235, -1.5712891, -1.8398438, -2.6738281, -1.3486328, -0.72998047,
    -4.4765625, 1.3154297, -1.4619141, -2.3339844, -4.8203125, -4.9882813, 0.9980469, 0.80371094,
    0.103881836, -3.5898438, -2.7539063, 1.6767578, -0.57421875, -2.5917969, 1.8603516, 1.2666016,
    -1.3544922, -1.1728516, -1.4316406, -4.1132813, -3.8105469, 1.7021484, -2.8828125, 6.5546875,
    2.5214844, -0.7553711, -3.0605469, 0.65478516, 1.3164063, -0.88720703, -4.2460938, -1.5361328,
    5.78125, -4.3945313, 1.2216797, 3.1992188, 1.2978516, -0.34033203, 1.3046875, 0.06088257,
    0.051116943, 0.38891602, 3.1113281, 2.8007813, -0.33007813, -1.9882813, -2.8339844, 4.4101563,
    6.03125, 2.5664063, -0.9082031, -4.0351563, -2.9277344, -0.45117188, -1.0292969, 7.6914063,
    -0.16357422, 6.2734375, 3.7597656, -5.0976563, -2.1855469, 1.3876953, 1.4248047, -2.5097656,
    -1.4716797, 0.97998047, -0.28222656, 1.3496094, 0.08081055, 2.7617188, 5.4257813, -7.2304688,
    -2.6542969, -1.2285156, -1.2558594, -1.1289063, 2.9140625, 5.6484375, 2.7539063, -3.9570313,
    -5.84375, 4.0625, 7.3125, -0.15539551, -2.2207031, -1.0908203, -0.052520752, 1.3496094,
    -7.3320313, 4.6640625, 2.2539063, 2.1972656, 3.7460938, 1.2851563, -4.4804688, -1.7802734,
    3.0996094, -3.9023438, 1.8310547, -6.1210938, -0.89453125, -2.2324219, 2.8066406, 6.796875,
    3.5703125, 0.07513428, -7.234375, -2.796875, 0.05581665, -1.6855469, 0.25512695, 5.1914063,
    -0.075927734, 2.8847656, -1.2402344, -5.0703125, 1.6201172, 4.6484375, 0.49951172, -5.6679688,
    -2.7539063, 2.6464844, 0.04901123, -1.7714844, 2.1210938, -3.6425781, 0.21691895, -1.4277344,
    -5.6796875, 1.7910156, -1.7294922, 1.59375, -1.2792969, -3.453125, -5.3632813, -1.2128906,
    1.2197266, 2.8164063, 1.1728516, 0.8300781, -3.4433594, -2.796875, -3.6113281, -5.9960938,
    -0.60791016, -0.57714844, 5.6875, -2.5976563, -5.5742188, 2.0976563, -1.7871094, -3.25,
    1.0625, 2.7539063, -6.3476563, -1.3115234, 0.49414063, 5.2226563, 4.671875, 1.1474609,
    -3.5605469, -1.6767578, 4.046875, -6.2382813, -0.5083008, 1.5371094, -1.5302734, 2.9394531,
    -4.0, 4.6757813, 3.5625, 1.3310547, 0.15148926, -2.3984375, 0.95996094, 2.0078125,
    2.2167969, -1.0185547, 4.703125, 5.2695313, 1.8720703, -4.6328125, 3.1386719, 3.8730469,
    -2.1015625, 3.7773438, 2.6542969, 2.2128906, 1.6210938, -2.5, -1.9072266, 2.6835938,
    -0.6772461, -2.0175781, -5.3046875, 2.4550781, 1.4863281, -8.9765625, -1.9882813, 0.2376709,
    5.8164063, 1.1054688, 1.1210938, 4.0546875, 7.1289063, 4.28125, -1.1669922, -1.7607422,
    -4.5859375, -3.171875, 0.029937744, -2.6621094, -2.078125, -6.5, 1.5126953, 0.31860352,
    1.671875, 1.7246094, 0.27661133, 0.76464844, -2.8691406, 2.9609375, -1.0390625, 0.08380127,
    2.7597656, 3.2285156, 4.1953125, -3.4941406, -2.0976563, 2.4882813, -2.734375, 1.7451172,
    -0.13842773, -3.7714844, -2.0800781, -5.0429688, 0.16540527, -1.5703125, -0.6669922, 6.296875,
    7.484375, 1.0507813, -1.0771484, 3.0585938, 7.3359375, 2.6796875, 3.1542969, 0.16589355,
    -0.050689697, -5.09375, 3.65625, -0.25708008, -1.4716797, -2.109375, -0.36108398, 8.0390625,
    -3.7285156, -6.2578125, 4.765625, -2.1445313, 0.23339844, 2.9121094, 0.09423828, -2.7539063,
    2.5234375, 4.4257813, 1.2548828, -2.3242188, 2.8574219, 0.10925293, -3.4316406, -1.2509766,
    2.2988281, 1.2294922, 2.765625, -0.24182129, -1.0273438, -4.2304688, -3.3867188, 3.6542969,
    0.034301758, 2.0488281, 5.46875, -2.9921875, 2.1914063, -0.46533203, -4.9453125, 0.6225586,
    0.123291016, -5.4882813, 1.6494141, -1.9980469, -0.8232422, 0.68408203, 4.1523438, -4.6953125,
    3.3320313, 0.09118652, -6.859375, 6.4257813, -3.2226563, -2.1191406, 3.0234375, -4.25,
    6.8984375, -1.2998047, 0.7192383, 1.6445313, -1.5869141, 0.7832031, -1.9677734, 0.11016846,
    -4.1757813, -1.7216797, -5.1835938, -0.15039063, 3.8242188, 1.0205078, 1.0595703, 4.3710938,
    5.4765625, -4.8085938, -4.0859375, -1.6464844, 4.9492188, 1.6708984, -2.1152344, -0.46679688,
    0.73779297, -5.96875, 3.546875, 5.3945313, 5.8515625, -2.0996094, -0.5151367, 2.8691406,
    -3.7324219, -3.7226563, 1.9980469, 1.8066406, -2.6035156, -4.2460938, -1.9423828, 7.3085938,
    2.2773438, -2.4023438, -0.9951172, -2.5566406, 3.9277344, -2.3886719, 0.5336914, 0.32421875,
    3.765625, -4.1875, 0.62939453, -4.0351563, -5.6914063, -2.4121094, -0.85791016, -1.7597656,
    3.8828125, 4.5742188, -1.6533203, 1.9638672, 0.9970703, -3.7011719, 0.14318848, 1.8046875,
    -0.16381836, 5.2070313, -1.5673828, 2.6796875, 6.7460938, 1.6044922, -2.4453125, 5.8671875,
    0.5830078, -2.2558594, -1.3125, 0.89453125, -0.5263672, -4.6210938, 5.78125, 7.625,
    -3.4882813, -3.8359375, 1.3916016, -1.4160156, 6.9296875, -4.40625, 2.3632813, 0.33081055,
    -0.050354004, 0.24255371, 1.5908203, 0.73876953, -3.1640625, -0.8378906, 1.4042969, 1.7822266,
    -1.8203125, 0.39453125, 3.203125, -0.6333008, -3.5097656, 0.43774414, -1.9121094, 0.022781372,
    3.0332031, 1.5664063, -6.71875, 0.16394043, -1.8144531, -0.90185547, -3.1113281, 3.234375,
    -2.3320313, -2.0488281, -3.7988281, -0.9316406, 1.1894531, 2.3769531, -1.5273438, -1.8076172,
    1.9970703, -4.1679688, 1.8232422, -4.84375, 4.984375, 2.53125, 5.0117188, -1.5097656,
    -1.2626953, -3.2910156, -2.1191406, -2.3554688, 3.5039063, 0.19470215, 1.4609375, 1.5126953,
    -7.0195313, -4.625, 1.5517578, -5.03125, -4.7070313, -1.5136719, 6.375, -8.828125,
    -3.0742188, 1.8515625, -1.6640625, 3.765625, 1.8955078, -2.1386719, 2.4609375, -1.8583984,
    -7.2070313, 3.0488281, -0.30078125, -4.3710938, 2.1875, -0.7402344, -4.9648438, 0.69189453,
    5.375, -2.0175781, 0.24169922, -0.035583496, -2.6152344, -8.609375, -3.7734375, -5.4960938,
    -3.2636719, -0.69628906, -0.37280273, -0.12634277, -1.0263672, 3.1425781, -0.03994751, -8.1171875,
    -1.1513672, -1.2402344, 5.2890625, -2.7167969, 5.6523438, -1.9873047, 6.875, -0.81884766,
    1.4511719, -2.6074219, 2.8339844, -0.11071777, 2.5625, -1.3730469, 5.8632813, -5.3632813,
    -2.0019531, 0.16577148, 3.5175781, -2.2148438, 4.84375, -2.9355469, 0.6196289, 4.4257813,
    -7.1835938, 6.5390625, -1.6005859, 0.5214844, -0.45214844, 7.7578125, 0.093444824, -4.546875,
    1.7089844, -0.55566406, -2.5976563, 1.4980469, 8.2265625, 1.9091797, 2.640625, -4.3945313,
    -2.5664063, 3.4394531, 1.3378906, -0.63916016, 0.8886719, 1.6611328, 1.2060547, -0.47045898,
    -2.2597656, 1.1357422, -1.0800781, 4.0898438, -3.6191406, 3.7753906, 4.2617188, 3.5683594,
    4.8125, -3.2890625, -0.7807617, -0.12731934, -2.5078125, -1.7939453, 6.1875, -3.3085938,
    -0.05001831, -4.9257813, -2.3476563, -1.7939453, 1.6757813, 4.9335938, -3.5039063, 7.1796875,
    3.2304688, -0.48388672, -1.6044922, -3.4355469, 0.5683594, -2.7597656, -0.33007813, 1.2792969,
    0.92333984, -7.4804688, 0.43408203, 2.46875, 2.9121094, -2.1464844, -1.9570313, -1.0556641,
    -3.2792969, -7.890625, 3.3027344, 1.8017578, -2.6132813, 3.9394531, -5.625, -1.4863281,
    3.3984375, 3.3554688, 1.8554688, -8.34375, -1.6025391, 1.4853516, 3.4394531, -3.4550781,
    -0.67626953, 1.234375, -4.0117188, 2.4082031, 0.7626953, -4.515625, -2.3886719, 3.1152344,
    3.7285156, -5.265625, 2.8222656, -7.2578125, -1.5185547, -0.18115234, 4.9101563, -0.5175781,
    1.59375, 0.6303711, -2.7988281, 3.0859375, 2.2421875, 0.037902832, -2.0039063, 3.9179688,
    6.7382813, -0.51953125, -1.1123047, 1.3876953, 2.5097656, 6.0351563, 1.9082031, 1.0566406,
    -2.1386719, 6.0429688, 3.2207031, -4.1484375, -5.4101563, -3.8457031, 3.9121094, -1.078125,
    6.3398438, -2.2734375, -1.4638672, 4.921875, -4.0039063, -7.75, 1.8085938, -2.1132813,
    -0.24572754, -1.4658203, -7.8632813, 5.9726563, 0.5029297, -1.5507813, -3.1699219, -0.69384766,
    -1.8388672, -1.7255859, -1.1679688, -4.4179688, -1.6845703, -1.0185547, -4.1757813, 1.4345703,
    1.6201172, 3.4492188, -0.41503906, -1.2910156, -0.46484375, -0.7314453, 1.8242188, 1.6650391,
    -6.578125, 3.9570313, 2.1699219, -3.6601563, -1.1230469, -1.1025391, 1.7392578, 1.5947266,
    -4.9453125, -5.28125, 0.4152832, -4.1484375, 3.1152344, -2.3476563, 2.6953125, -1.1074219,
    6.3476563, -1.9316406, 2.5566406, 3.6054688, -1.46875, -5.8203125, -2.4003906, -5.8007813,
    -0.45507813, 0.9555664, -4.0234375, -2.2148438, -2.2363281, 0.9013672, -2.4667969, -1.6289063,
    -2.390625, -3.5820313, -4.359375, -3.8847656, 0.74365234, -0.3540039, -0.8041992, -1.3720703,
    -1.8056641, -0.5136719, 3.1308594, -0.93408203, 2.5351563, -0.984375, 2.5585938, -4.3164063,
    -2.1152344, 3.6757813, -1.6884766, 0.8178711, 2.421875, 3.8242188, 0.8305664, 0.8256836,
    -6.2070313, -5.1484375, -6.6171875, 0.37890625, -3.453125, -0.9785156, 6.265625, 2.3222656,
    1.421875, 1.5888672, -0.54833984, -2.9804688, 1.6357422, 3.59375, 0.58447266, 2.4472656,
    -0.26782227, 1.78125, -2.5996094, -1.7666016, 2.0332031, 6.3164063, 1.3164063, 3.1347656,
    2.1992188, -0.66748047, -2.3554688, 0.80371094, 0.62939453, 0.8173828, 3.2714844, 3.4980469,
    1.3261719, 9.5390625, 7.0859375, 1.9160156, -4.5039063, -4.0507813, 4.0820313, -0.9926758,
    1.9658203, -4.3554688, -2.0722656, -2.5195313, 3.2109375, 3.1386719, -0.08856201, 1.7539063,
    -10.734375, 3.7988281, 5.9921875, 6.46875, -3.1289063, -0.8881836, -1.9111328, -0.23327637,
    -0.32104492, 3.5566406, -5.7734375, -2.5195313, -0.3798828, -1.7519531, 0.07879639, 3.8066406,
    -1.1103516, 4.5, -0.29125977, 0.43579102, 0.27954102, 0.3095703, -1.7470703, -1.71875,
    -5.2578125, 3.5527344, -2.4472656, -5.0195313, 4.7539063, 2.2402344, 3.1171875, 0.21252441,
    -3.2832031, 2.5664063, 0.33251953, 2.1679688, 0.20031738, 7.8867188, 2.7910156, -4.515625,
];
