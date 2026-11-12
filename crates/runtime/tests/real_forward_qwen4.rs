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

/// [`qwen4_install`], with the router shipped raw (unquantized BF16) rather
/// than pre-packed INT8 -- the shape the real REAP-288 checkpoint's router
/// actually takes, and the exact reproduction of the router-dtype bug this
/// covers (`crates/repack`'s `orchestrate.rs::read_resident_entries`, and
/// `families/qwen4/moe.rs:58`'s dtype-5 refusal on the runtime side). The
/// default fixture above ships the router pre-packed and cannot reach either
/// bug: this is the one that does.
fn qwen4_install_raw_router() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = turbospark_repack::build_synthetic_qwen4_exp_decode_install_raw_router(
        &dir,
        VOCAB,
        "qwen4-p3-raw",
    )
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

/// The router-dtype bug's real reproduction: before the fix,
/// `moe.rs:58`'s dtype check refused this exact install with "expected INT8
/// (dtype 5) ..., got dtype 1" (raw BF16), because
/// `read_resident_entries` only knew how to pass through an ALREADY-packed
/// `U32` router or narrow anything else straight to BF16. `qwen4_install()`
/// above cannot see this: its router is pre-packed by construction
/// (`int8_triple`) and so never exercises either the writer's new
/// `quantize_router_int8` branch or the runtime's dtype-5 requirement.
#[test]
fn a_qwen4_exp_install_with_a_raw_bf16_router_opens_and_decodes() {
    let (dir, arch) = qwen4_install_raw_router();
    assert_eq!(arch.family, model_io::ModelFamily::Qwen4Exp);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["quant"]["router"]["weightBits"], 8,
        "the manifest must state the router's forced INT8 width, not the \
         checkpoint's undeclared default"
    );

    let logits = decode(&dir, &arch, 8);
    let first = logits[0];
    assert!(
        logits.iter().any(|v| *v != first),
        "every logit is {first}: the head produced nothing"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A QSA `index_budget` small enough to cross in a handful of tokens:
/// 16 tokens is 4 blocks of `IDX_COMPRESS = 4`, so `index_top_k` is 4 and
/// block selection starts DROPPING blocks at the fifth complete one --
/// `visible >= 20`, i.e. position 19 onward.
const TINY_INDEXER_BUDGET: i64 = 16;
/// The first position at which [`TINY_INDEXER_BUDGET`] selection is not a
/// no-op: `(position + 1) / 4 > 4`.
const FIRST_SPARSE_POSITION: usize = 19;
/// Enough positions past [`FIRST_SPARSE_POSITION`] for the dropped block to
/// change more than one step, and for selection to have run at every
/// `visible % 4` phase (tail of 0, 1, 2 and 3 tokens).
const SPARSE_STEPS: usize = 28;

fn qwen4_install_tiny_budget() -> (std::path::PathBuf, model_io::ArchConfig) {
    let dir = tempdir();
    let arch = turbospark_repack::build_synthetic_qwen4_exp_decode_install_with_indexer_budget(
        &dir,
        VOCAB,
        "qwen4-p3-tiny-budget",
        TINY_INDEXER_BUDGET,
    )
    .expect("write install");
    (dir, arch)
}

/// Feeds a FIXED token sequence (not greedy, so both arms of a comparison
/// see identical inputs whatever their outputs) and returns every step's
/// logits, asserting each is finite. `force_dense` is the QSA diagnostic
/// arm (`RealForwardRunner::set_qsa_force_dense`).
fn decode_fixed_sequence(
    dir: &std::path::Path,
    arch: &model_io::ArchConfig,
    steps: usize,
    force_dense: bool,
) -> Vec<Vec<f32>> {
    let vocab = arch.vocab_size as usize;
    let mut runner = RealForwardRunner::open_with_max_context(dir, arch.clone(), TEST_MAX_CONTEXT)
        .expect("a qwen4_exp install opens above its indexer budget");
    runner.reset();
    runner.set_qsa_force_dense(force_dense);
    let mut all = Vec::with_capacity(steps);
    for position in 0..steps {
        let token = ((position * 37 + 11) % vocab) as i32;
        let mut logits = vec![f16::from_f32(0.0); vocab];
        runner
            .produce(token, position, &mut logits)
            .unwrap_or_else(|e| panic!("produce failed at position {position}: {e}"));
        assert!(
            logits.iter().all(|v| v.to_f32().is_finite()),
            "non-finite logit at position {position} (force_dense {force_dense})"
        );
        all.push(logits.iter().map(|v| v.to_f32()).collect());
    }
    all
}

/// Context above `compressed_attention.index_budget` OPENS now that the
/// indexer is wired: the refusal this test used to pin is gone, and the
/// three indexer tensors are what `open` requires instead.
#[test]
fn context_above_the_default_indexer_budget_opens() {
    let (dir, arch) = qwen4_install();
    let budget = arch.compressed_attention.index_budget as usize;
    RealForwardRunner::open_with_max_context(&dir, arch, budget + 1)
        .expect("context above the indexer budget must open now that QSA is wired");
    std::fs::remove_dir_all(&dir).ok();
}

/// The sparse path RUNS: a tiny budget is crossed at position 19 and every
/// later step scores blocks, reads them back, selects, and attends over the
/// selected positions -- through every `visible % 4` tail phase -- with
/// finite logits at every step. Greedy, so the sparse output feeds back
/// into later inputs the way a real generation's would.
#[test]
fn a_tiny_indexer_budget_decodes_sparsely_past_it() {
    let (dir, arch) = qwen4_install_tiny_budget();
    assert_eq!(
        arch.compressed_attention.index_top_k,
        TINY_INDEXER_BUDGET / 4
    );
    let logits = decode(&dir, &arch, SPARSE_STEPS);
    assert_eq!(logits.len(), VOCAB as usize);
    std::fs::remove_dir_all(&dir).ok();
}

/// THE BELOW-BUDGET EXACTNESS GUARD, and the proof that selection does
/// something above it. Two runners over one fixed token sequence, one with
/// the force-dense diagnostic arm on: their logits must be BITWISE equal at
/// every position where `select_blocks` keeps every block (positions 0
/// through 18 at this budget -- the indexer runs there too, and must not
/// touch the trunk), and must differ somewhere past position 19, where a
/// block is dropped from attention. A sparse path that silently attended
/// densely would pass the first half and fail the second; one that leaked
/// into the below-budget stream would fail the first.
#[test]
fn sparse_and_forced_dense_agree_below_budget_and_diverge_above() {
    let (dir, arch) = qwen4_install_tiny_budget();
    let sparse = decode_fixed_sequence(&dir, &arch, SPARSE_STEPS, false);
    let dense = decode_fixed_sequence(&dir, &arch, SPARSE_STEPS, true);
    std::fs::remove_dir_all(&dir).ok();

    for position in 0..FIRST_SPARSE_POSITION {
        let same = sparse[position]
            .iter()
            .zip(&dense[position])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same,
            "position {position} is at or below the budget, yet the sparse and forced-dense \
             arms differ: the indexer leaked into the trunk"
        );
    }
    let diverged: Vec<usize> = (FIRST_SPARSE_POSITION..SPARSE_STEPS)
        .filter(|&p| sparse[p].iter().zip(&dense[p]).any(|(a, b)| a != b))
        .collect();
    assert!(
        !diverged.is_empty(),
        "no position past {FIRST_SPARSE_POSITION} differs between the sparse and forced-dense \
         arms: block selection dropped nothing, or the dense kernel ran regardless"
    );
    println!("sparse arm diverges from forced-dense at positions {diverged:?}");
}

/// The indexer's projection reaches the output ONLY above budget: patched
/// to garbage, the tiny-budget install's greedy decode moves (scores, hence
/// selection, hence attention), while the default-budget install's 8-step
/// decode -- which runs the same projection and the same block pooling
/// every token -- is bit-identical, because nothing below budget consumes
/// what the indexer computes.
#[test]
fn the_indexer_projection_moves_the_output_only_above_budget() {
    let (dir, arch) = qwen4_install_tiny_budget();
    let clean = decode(&dir, &arch, SPARSE_STEPS);
    patch(
        &dir,
        "language_model.model.layers.2.self_attn.indexer.index_qk_proj.weight",
        0x3F80,
    );
    let patched = decode(&dir, &arch, SPARSE_STEPS);
    std::fs::remove_dir_all(&dir).ok();
    assert!(
        clean.iter().zip(&patched).any(|(a, b)| a != b),
        "patching the indexer projection above budget must move the logits"
    );

    let (dir, arch) = qwen4_install();
    let clean = decode(&dir, &arch, 8);
    patch(
        &dir,
        "language_model.model.layers.2.self_attn.indexer.index_qk_proj.weight",
        0x3F80,
    );
    let patched = decode(&dir, &arch, 8);
    std::fs::remove_dir_all(&dir).ok();
    assert!(
        clean
            .iter()
            .zip(&patched)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "below budget the indexer must not reach the trunk, yet patching it moved the logits"
    );
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
const FROZEN_LOGIT_HASH: u64 = 11_666_956_978_845_961_139;

/// Frozen on Apple M4 Max on 2026-09-26 after the SlotStream-guided
/// Qwen4 arithmetic corrections: GDN Q/K epsilon matches direct L2
/// normalization, PLE does not apply SiLU twice, and the shared MoE branch
/// joins after FP16 rounding. Re-frozen on 2026-10-31: that change also made
/// every RMS norm uncentered, which is right only for GGUF installs, and
/// decoded garbage on the real MLX REAP-288 install. The norm convention now
/// follows the install's quant scheme, and this fixture is an affine (MLX
/// convention) install, so its norms are centered again.
/// This is a change detector over untrained synthetic weights, not a
/// correctness claim. The real-device hash and this vector come from the
/// same run; the vector supports the existing virtualized-CI comparison.
/// Re-freeze only with a stated reason.
#[rustfmt::skip]
const FROZEN_LOGITS: [f32; VOCAB as usize] = [
    1.3417969f32, 2.9433594f32, -3.6132813f32, -0.9770508f32, 0.31884766f32, 1.7265625f32, 0.8432617f32, -4.5351563f32,
    0.36889648f32, 2.2714844f32, -6.1757813f32, 4.9375f32, -4.1445313f32, 4.90625f32, 10.7265625f32, 0.41357422f32,
    -1.4511719f32, 1.8642578f32, 3.1933594f32, -4.0703125f32, 2.3710938f32, -3.1738281f32, -2.1230469f32, -5.0390625f32,
    -7.7070313f32, 2.8183594f32, 2.9199219f32, 11.03125f32, 2.0253906f32, 0.42773438f32, 0.9394531f32, -1.3759766f32,
    0.14416504f32, -6.1601563f32, 2.0878906f32, 1.5429688f32, 4.7148438f32, 0.12213135f32, 2.8339844f32, -0.07244873f32,
    -3.6777344f32, 6.4414063f32, 4.7929688f32, 2.5449219f32, -4.453125f32, 0.12585449f32, 2.7695313f32, -2.5527344f32,
    -3.9589844f32, -6.859375f32, 0.46289063f32, -0.4543457f32, -0.36401367f32, -2.0039063f32, 7.71875f32, 7.5195313f32,
    2.4609375f32, -6.5507813f32, 11.5625f32, -0.5625f32, 3.1132813f32, 0.09234619f32, 2.5488281f32, -1.8984375f32,
    1.2246094f32, -9.9296875f32, 8.7421875f32, -0.12445068f32, 4.2734375f32, 3.2128906f32, 2.9882813f32, 1.9443359f32,
    -0.2668457f32, 5.1015625f32, -7.1796875f32, 3.9140625f32, -5.5664063f32, -1.4414063f32, -2.140625f32, 2.3886719f32,
    -6.84375f32, -3.78125f32, 0.30664063f32, -1.0166016f32, 1.0078125f32, -1.5009766f32, 3.15625f32, 2.0917969f32,
    0.60498047f32, -6.359375f32, 0.42138672f32, 3.8359375f32, -4.5507813f32, -1.4179688f32, 1.9453125f32, -2.4277344f32,
    4.5507813f32, 5.03125f32, 2.6621094f32, 4.78125f32, 6.375f32, 2.171875f32, 5.4375f32, 3.3769531f32,
    3.4316406f32, -3.4082031f32, 1.9111328f32, 0.061462402f32, 0.40063477f32, -1.8251953f32, -4.4375f32, -3.9941406f32,
    1.4111328f32, -4.3359375f32, -2.2714844f32, -1.3476563f32, -0.52978516f32, -0.6879883f32, -1.7255859f32, 1.0693359f32,
    -3.9765625f32, 5.4414063f32, -2.5664063f32, -0.50097656f32, -3.1132813f32, -6.5195313f32, -3.890625f32, 9.296875f32,
    -0.69921875f32, 1.1035156f32, 2.5390625f32, 0.42993164f32, -1.6884766f32, -3.9667969f32, 5.3828125f32, -4.4765625f32,
    5.5976563f32, -5.7734375f32, -5.390625f32, -2.3417969f32, 3.9355469f32, -0.33911133f32, 6.5351563f32, -8.3203125f32,
    6.1992188f32, -1.2919922f32, -3.4101563f32, -2.9453125f32, 3.6777344f32, -2.2636719f32, -5.5585938f32, 0.5463867f32,
    3.1445313f32, 3.1015625f32, -4.0898438f32, -0.70458984f32, 3.4082031f32, 6.515625f32, -3.3886719f32, 1.2714844f32,
    -3.6425781f32, 1.3583984f32, 1.4003906f32, -0.47631836f32, -7.6289063f32, 3.9804688f32, -2.515625f32, 0.9770508f32,
    -0.6870117f32, -0.43725586f32, -4.203125f32, 4.5625f32, 3.4414063f32, 7.4804688f32, 1.7978516f32, 2.7070313f32,
    -0.019577026f32, -4.5351563f32, -0.1385498f32, -5.1757813f32, 6.21875f32, -6.1484375f32, 4.8359375f32, 3.3691406f32,
    6.4296875f32, -5.328125f32, -1.8828125f32, 6.6328125f32, 6.7382813f32, -1.4746094f32, -5.2578125f32, 2.7363281f32,
    -0.068481445f32, -4.3398438f32, 1.1816406f32, 0.44384766f32, 6.5859375f32, -0.6821289f32, -5.7148438f32, 0.45385742f32,
    0.8486328f32, -6.3320313f32, 4.59375f32, -2.9765625f32, 1.5585938f32, 1.3466797f32, -1.8388672f32, -0.07824707f32,
    4.3867188f32, 0.46606445f32, 3.2636719f32, -4.140625f32, -0.859375f32, 0.37304688f32, 5.4257813f32, -6.1796875f32,
    -1.9824219f32, 1.1650391f32, -3.1777344f32, 4.6289063f32, -1.2734375f32, -1.3779297f32, -7.2695313f32, -3.4316406f32,
    -2.0488281f32, 0.7519531f32, -5.5742188f32, 3.5664063f32, 2.4003906f32, 1.3193359f32, -2.6914063f32, -3.3378906f32,
    4.8359375f32, 1.9042969f32, 4.3945313f32, -2.6796875f32, 8.359375f32, -2.3769531f32, -3.5527344f32, 0.7807617f32,
    -0.15063477f32, -2.8496094f32, 2.34375f32, -9.0078125f32, 1.6855469f32, 0.16931152f32, 1.9794922f32, 2.1894531f32,
    -8.8828125f32, 5.1601563f32, 1.4287109f32, 1.8027344f32, -0.60595703f32, -1.3232422f32, 1.0878906f32, 1.7744141f32,
    -5.8046875f32, -0.19299316f32, 6.3046875f32, 3.9394531f32, 9.3671875f32, 2.1679688f32, -0.30297852f32, 4.9804688f32,
    2.3417969f32, 2.9179688f32, -3.3886719f32, 4.296875f32, -4.4257813f32, -1.4472656f32, 3.8867188f32, -0.9042969f32,
    4.5195313f32, -0.41796875f32, -2.6796875f32, -4.4023438f32, -5.4960938f32, -2.78125f32, -2.4785156f32, 0.12036133f32,
    5.8554688f32, 2.8359375f32, -3.1171875f32, -0.6044922f32, 4.2617188f32, -7.9179688f32, -1.4912109f32, 4.2070313f32,
    -2.5371094f32, -0.28100586f32, -1.9472656f32, -4.9375f32, 6.640625f32, 1.9257813f32, -3.5234375f32, 1.8632813f32,
    0.43017578f32, -4.5390625f32, 0.86083984f32, -1.9492188f32, 3.8671875f32, -4.40625f32, 6.9492188f32, 6.0429688f32,
    1.4951172f32, 4.7929688f32, 3.9296875f32, -2.3144531f32, -5.125f32, -2.1875f32, 3.1953125f32, -0.73583984f32,
    -4.2773438f32, 2.578125f32, 1.1767578f32, -5.0039063f32, 1.2832031f32, 5.3242188f32, -0.48388672f32, -3.1542969f32,
    -1.4667969f32, -3.7246094f32, -5.3398438f32, 7.9960938f32, -7.3554688f32, -2.1933594f32, 0.66796875f32, -8.4140625f32,
    0.8442383f32, -4.4414063f32, 4.3945313f32, 5.1132813f32, -2.9902344f32, 6.6210938f32, -12.078125f32, 2.2460938f32,
    -1.8935547f32, -3.9238281f32, -2.5058594f32, -2.1171875f32, -3.8847656f32, -3.8886719f32, 0.9350586f32, -3.8261719f32,
    3.2578125f32, 3.3007813f32, 4.875f32, -0.15600586f32, 7.9960938f32, 2.0195313f32, -0.85058594f32, -2.1640625f32,
    -0.24108887f32, 0.6777344f32, 0.9404297f32, -0.5917969f32, 0.015808105f32, -7.8085938f32, 5.0859375f32, 5.1914063f32,
    -2.1074219f32, -3.53125f32, -1.2753906f32, -2.8925781f32, -9.0859375f32, -1.8710938f32, -2.0996094f32, 0.6538086f32,
    -3.0722656f32, 0.9121094f32, -6.96875f32, -5.1210938f32, -5.7265625f32, 3.546875f32, 0.057769775f32, 2.359375f32,
    0.27294922f32, -2.0878906f32, 3.7128906f32, 2.8261719f32, 1.9902344f32, -4.4570313f32, -1.4804688f32, 7.2734375f32,
    5.1835938f32, 2.3066406f32, -0.4260254f32, 0.8930664f32, -3.2246094f32, -0.48217773f32, -3.4121094f32, 1.4736328f32,
    -1.3203125f32, -1.6083984f32, -3.2988281f32, 0.010864258f32, 3.6679688f32, 4.3945313f32, -1.8076172f32, -2.9003906f32,
    -4.3710938f32, -0.61328125f32, -0.07116699f32, -1.9257813f32, 1.7314453f32, 5.28125f32, -0.018875122f32, -3.2441406f32,
    1.5693359f32, -3.7011719f32, -6.328125f32, -0.0496521f32, 4.328125f32, -0.12805176f32, 4.1679688f32, 2.7851563f32,
    -6.6757813f32, 2.0136719f32, -3.7636719f32, -2.6542969f32, 5.4609375f32, 0.9921875f32, 1.375f32, -1.7392578f32,
    0.080444336f32, -3.5839844f32, 2.5996094f32, -3.2207031f32, 6.2617188f32, -1.1435547f32, 0.48266602f32, 1.6357422f32,
    2.9042969f32, 2.5332031f32, -1.7480469f32, 3.4667969f32, 2.9960938f32, 1.3837891f32, 2.5527344f32, -2.171875f32,
    -2.5585938f32, 4.9453125f32, 2.7480469f32, 5.9453125f32, -1.3447266f32, -1.9658203f32, 2.7539063f32, 5.59375f32,
    -7.8828125f32, 3.5917969f32, -1.2685547f32, 10.40625f32, -5.1875f32, -4.7734375f32, -0.87109375f32, 4.9921875f32,
    -2.0605469f32, 1.8769531f32, 1.8212891f32, 2.9257813f32, 4.921875f32, -0.77490234f32, -6.921875f32, 0.44555664f32,
    -4.0429688f32, 5.4335938f32, -6.9960938f32, -6.9179688f32, 6.203125f32, 0.25805664f32, -5.1679688f32, -1.390625f32,
    3.7382813f32, -5.53125f32, 3.4296875f32, -3.2734375f32, 1.4736328f32, -3.3613281f32, -4.6289063f32, -3.1894531f32,
    5.0351563f32, -3.640625f32, 4.4453125f32, 0.3696289f32, -5.3476563f32, 0.73876953f32, 1.8183594f32, -1.1533203f32,
    3.1933594f32, 2.4746094f32, 1.3261719f32, 5.7890625f32, 5.6953125f32, 4.6601563f32, -4.5039063f32, -3.3222656f32,
    -5.6679688f32, 1.5439453f32, 5.1171875f32, 4.9492188f32, 0.19689941f32, -0.26171875f32, 1.9287109f32, -0.88378906f32,
    -0.3269043f32, -1.6806641f32, -1.2939453f32, -4.7734375f32, -0.8623047f32, -2.6132813f32, -4.9453125f32, -1.9433594f32,
    2.1074219f32, -0.42773438f32, 12.9140625f32, 7.0039063f32, 2.0585938f32, 2.1328125f32, 1.8037109f32, 2.5898438f32,
    0.55322266f32, -3.203125f32, 1.7294922f32, -0.024536133f32, 3.2910156f32, 3.3632813f32, 1.0253906f32, 6.7851563f32,
    9.7890625f32, -7.4921875f32, 1.2246094f32, -2.7753906f32, -3.0898438f32, -1.9658203f32, -7.328125f32, 3.1503906f32,
    -3.6230469f32, 1.0810547f32, 0.55859375f32, 1.7841797f32, 6.2617188f32, 0.9326172f32, 0.17822266f32, 0.27490234f32,
    -5.0898438f32, -2.3300781f32, 0.8540039f32, 1.2226563f32, 2.0351563f32, -0.38354492f32, 2.3652344f32, 5.9296875f32,
    2.4140625f32, -3.4746094f32, 1.2929688f32, 0.75146484f32, -0.37158203f32, 1.0009766f32, 1.4179688f32, 1.1005859f32,
    -2.8554688f32, -1.1992188f32, -0.9794922f32, 3.8535156f32, -2.5683594f32, -0.69433594f32, -6.0664063f32, 4.8789063f32,
    1.5078125f32, 5.8554688f32, 0.6347656f32, -5.0664063f32, -0.59765625f32, 1.8359375f32, -0.40429688f32, 0.18615723f32,
    -2.9726563f32, -6.703125f32, -3.609375f32, 2.4980469f32, -3.6015625f32, 3.7890625f32, 10.03125f32, -1.0595703f32,
    -4.1523438f32, -0.9921875f32, 3.0332031f32, 0.7714844f32, 4.2421875f32, -0.052001953f32, 0.78466797f32, -1.3427734f32,
    -6.6367188f32, -0.32202148f32, -1.7958984f32, 6.2734375f32, -4.6132813f32, -3.6621094f32, 1.8310547f32, 3.1132813f32,
    -3.6308594f32, -2.0683594f32, 0.7709961f32, 3.8632813f32, -4.8203125f32, -0.8666992f32, 4.7617188f32, 1.7109375f32,
    -4.8203125f32, -5.2148438f32, -5.0195313f32, 3.8886719f32, 0.9848633f32, -1.1845703f32, 5.6210938f32, -1.7138672f32,
    4.6289063f32, -3.2421875f32, -4.1328125f32, 1.8330078f32, 0.03829956f32, 1.8212891f32, 1.8701172f32, 5.609375f32,
    0.06640625f32, -4.0234375f32, 3.390625f32, 2.0859375f32, -4.75f32, -2.0546875f32, 1.6005859f32, 2.4765625f32,
    1.9941406f32, 0.86035156f32, 0.15783691f32, -6.6445313f32, 5.0898438f32, 2.0058594f32, 4.4648438f32, -5.4882813f32,
    2.2246094f32, -0.085510254f32, -3.3007813f32, -6.3203125f32, 0.77441406f32, -1.984375f32, -5.625f32, 5.2070313f32,
    3.9453125f32, 5.9570313f32, 0.5776367f32, 3.9960938f32, 0.9628906f32, 10.8359375f32, 3.609375f32, -2.9238281f32,
    3.4414063f32, 0.8925781f32, 2.9394531f32, 0.12988281f32, -4.484375f32, 5.8867188f32, 5.34375f32, -4.890625f32,
    -0.46166992f32, 4.0234375f32, -2.28125f32, -3.0332031f32, 1.7333984f32, 5.875f32, -2.1953125f32, 0.4345703f32,
    2.8847656f32, 0.40454102f32, 5.8242188f32, -4.671875f32, -6.125f32, -2.3652344f32, 1.6162109f32, 1.1708984f32,
    0.099731445f32, 2.9609375f32, 0.20458984f32, 0.37329102f32, 6.3164063f32, 0.9584961f32, 2.9394531f32, 3.8808594f32,
    7.953125f32, 5.15625f32, 1.2011719f32, 1.8945313f32, 3.9960938f32, -4.0390625f32, 5.578125f32, -3.6621094f32,
    1.2724609f32, -9.859375f32, -2.0273438f32, -4.4140625f32, 3.2753906f32, -0.043884277f32, 3.6054688f32, -1.2060547f32,
    -0.57958984f32, -2.8828125f32, 3.5253906f32, 2.6875f32, 1.1171875f32, -1.3007813f32, 4.5664063f32, -2.6015625f32,
    -7.015625f32, -0.74365234f32, -1.7480469f32, -0.57373047f32, -1.9619141f32, 1.7949219f32, 5.578125f32, 4.5039063f32,
    -3.9472656f32, 1.1494141f32, -5.1367188f32, -5.234375f32, -1.8291016f32, -2.7441406f32, 2.1035156f32, -8.421875f32,
    1.7255859f32, 2.8144531f32, 1.8691406f32, -1.9101563f32, -1.6035156f32, -1.765625f32, 3.8925781f32, -2.0488281f32,
    9.7734375f32, -4.3046875f32, -2.9238281f32, 0.5488281f32, -0.00050497055f32, -1.0009766f32, 2.5253906f32, 0.65527344f32,
    3.2851563f32, -0.008872986f32, -3.0039063f32, 1.875f32, -3.6875f32, 2.1328125f32, -1.9765625f32, 5.046875f32,
    0.0501709f32, 0.4350586f32, -3.2226563f32, -1.3574219f32, -4.890625f32, -1.3359375f32, 1.9433594f32, 2.296875f32,
    0.45361328f32, 6.7148438f32, -0.51464844f32, -3.0566406f32, 0.27856445f32, -5.1640625f32, -3.3183594f32, 0.94091797f32,
    5.7070313f32, 0.56591797f32, -5.3007813f32, -4.6757813f32, -4.53125f32, 4.5351563f32, 0.7480469f32, 0.30395508f32,
    1.0361328f32, -2.1230469f32, 3.3027344f32, -4.3476563f32, 4.9335938f32, -0.85595703f32, 0.054992676f32, -5.5976563f32,
    -4.6601563f32, 1.0439453f32, 3.1992188f32, 1.2539063f32, -4.921875f32, 5.90625f32, 1.1992188f32, 9.3515625f32,
    4.421875f32, -6.2695313f32, 5.0625f32, 4.2421875f32, 3.0839844f32, 5.1328125f32, 6.2695313f32, 5.1757813f32,
    4.4882813f32, 3.7714844f32, -2.296875f32, -3.1601563f32, 5.1523438f32, -0.21862793f32, -3.375f32, 3.0683594f32,
    1.1367188f32, 0.6147461f32, -11.7578125f32, 2.4804688f32, -9.2734375f32, 4.0820313f32, -2.0976563f32, -2.4667969f32,
    5.3359375f32, -5.890625f32, -0.7944336f32, 1.8330078f32, 5.0820313f32, -0.69873047f32, -1.8164063f32, 3.0703125f32,
    -4.4648438f32, 4.7109375f32, -4.7070313f32, 6.5546875f32, -3.2753906f32, -1.1904297f32, -5.7539063f32, 4.8007813f32,
    1.8808594f32, -3.4746094f32, 8.703125f32, 2.5390625f32, -0.24365234f32, 1.53125f32, 2.7636719f32, 6.734375f32,
    1.7490234f32, 0.25732422f32, -1.5878906f32, 1.2128906f32, -1.5478516f32, -4.1914063f32, -5.8242188f32, -0.5805664f32,
    1.9882813f32, -0.25952148f32, 0.10119629f32, 9.5f32, -1.3457031f32, -1.8505859f32, -2.6132813f32, 3.8339844f32,
    8.609375f32, 0.08148193f32, 1.5957031f32, 3.6132813f32, -2.1914063f32, 0.8378906f32, -1.2021484f32, 7.7148438f32,
    -6.046875f32, 3.1542969f32, 0.3828125f32, -5.640625f32, 1.8505859f32, -2.4394531f32, -2.8417969f32, 3.3769531f32,
    -2.6386719f32, 0.6738281f32, 5.1367188f32, -5.9570313f32, -3.3378906f32, 6.8867188f32, -3.2070313f32, 7.2265625f32,
    -2.2246094f32, -5.0976563f32, 4.8789063f32, 0.6064453f32, -3.5742188f32, -0.67578125f32, 2.0996094f32, -3.7421875f32,
    -5.3203125f32, -6.3359375f32, -0.7661133f32, -1.3828125f32, 0.67822266f32, -7.6796875f32, -0.48999023f32, 0.45214844f32,
    5.2109375f32, -0.20996094f32, 0.5366211f32, 5.6523438f32, 1.7529297f32, 2.1894531f32, -3.5019531f32, -8.8046875f32,
    -0.72314453f32, 4.140625f32, 1.4638672f32, -3.2382813f32, -1.4628906f32, 0.47680664f32, -0.52978516f32, -1.3427734f32,
    1.9951172f32, 4.5429688f32, 3.0566406f32, -2.2207031f32, 1.7070313f32, 5.8085938f32, 1.3330078f32, -3.1992188f32,
    -3.4667969f32, -9.078125f32, 0.15905762f32, -1.7363281f32, -0.062927246f32, 8.8125f32, 2.4941406f32, -1.0908203f32,
    -4.3125f32, 1.1777344f32, 4.890625f32, 1.8164063f32, -3.0761719f32, -3.8300781f32, 0.7236328f32, 5.734375f32,
    0.60595703f32, 0.50097656f32, -0.6972656f32, 2.9121094f32, 2.4902344f32, 1.6953125f32, -7.2460938f32, -0.22277832f32,
    3.9160156f32, 0.09649658f32, 5.2109375f32, 0.85302734f32, -3.1503906f32, -2.890625f32, 0.28808594f32, 0.10461426f32,
    -6.109375f32, -1.1640625f32, 6.1601563f32, -2.4453125f32, 0.7685547f32, 3.5996094f32, 3.1660156f32, 1.1005859f32,
    5.3554688f32, -3.1152344f32, -0.16674805f32, 1.4580078f32, 0.018753052f32, -0.9604492f32, -1.4580078f32, 7.8203125f32,
    5.9804688f32, 6.9492188f32, -0.41210938f32, -5.8046875f32, -3.7304688f32, -1.5214844f32, 4.4921875f32, -8.453125f32,
    4.53125f32, -5.4648438f32, -3.6035156f32, -0.70166016f32, 3.8242188f32, -0.53222656f32, -5.71875f32, 2.6484375f32,
    -0.18933105f32, -5.5f32, -0.13903809f32, 6.0546875f32, 2.7480469f32, -5.4414063f32, -2.5292969f32, 0.85302734f32,
    -0.60498047f32, -0.5878906f32, 4.1210938f32, -1.9541016f32, 1.2685547f32, 1.4550781f32, 6.1445313f32, -2.1582031f32,
    -10.5546875f32, -4.2695313f32, 1.8164063f32, -0.97021484f32, 3.7265625f32, -6.5429688f32, -0.3720703f32, 0.48095703f32,
    4.4726563f32, -3.0449219f32, -4.25f32, 1.0957031f32, 1.2167969f32, 0.052734375f32, -0.60595703f32, 3.0527344f32,
    -3.8203125f32, -2.7128906f32, -2.7089844f32, -2.9550781f32, -4.8515625f32, 1.0859375f32, 3.5371094f32, -3.2617188f32,
    -4.40625f32, 3.5332031f32, 0.60009766f32, -2.6738281f32, 0.2298584f32, -0.5283203f32, -3.0058594f32, 5.3320313f32,
    2.515625f32, 0.79541016f32, 4.8789063f32, -4.734375f32, 0.79296875f32, -1.4355469f32, -2.1171875f32, -0.87060547f32,
    0.22973633f32, -3.3242188f32, -8.6875f32, 1.0878906f32, 4.5859375f32, 3.9238281f32, 0.7817383f32, 0.80810547f32,
    -3.4179688f32, -1.9345703f32, -4.0742188f32, -1.4736328f32, -1.0722656f32, 7.6835938f32, 8.71875f32, 4.0507813f32,
    -3.5566406f32, -4.6445313f32, -2.6191406f32, 2.84375f32, 0.68115234f32, 3.9863281f32, 3.1152344f32, -1.0605469f32,
    -5.046875f32, -6.9023438f32, 3.4648438f32, 2.7929688f32, -4.3476563f32, 2.8496094f32, 2.8222656f32, 0.38842773f32,
    -5.1875f32, 1.9091797f32, -4.1171875f32, 0.70166016f32, 3.1210938f32, 3.2578125f32, -1.4541016f32, -1.1875f32,
    6.8046875f32, 3.7910156f32, -3.3242188f32, 0.064453125f32, 4.7265625f32, -0.13635254f32, -2.6542969f32, -5.4804688f32,
    7.7539063f32, 0.11541748f32, -2.3945313f32, 0.19433594f32, -2.0019531f32, 3.3007813f32, 4.4453125f32, -0.11236572f32,
    0.9013672f32, 2.6816406f32, 7.640625f32, 0.1315918f32, -3.8066406f32, 2.8125f32, 2.171875f32, -3.296875f32,
    0.20629883f32, 0.55029297f32, 4.8867188f32, 2.7304688f32, -4.7929688f32, 2.5566406f32, -2.5839844f32, -0.32226563f32,
    0.44921875f32, -0.55322266f32, -5.3320313f32, 2.7929688f32, -3.3886719f32, -6.6171875f32, 2.0878906f32, 7.7265625f32,
    3.28125f32, 4.0664063f32, -8.34375f32, 3.4804688f32, 4.3476563f32, -3.2871094f32, 6.0273438f32, -0.84814453f32,
    -2.1464844f32, 6.7890625f32, -3.8730469f32, -7.2617188f32, 5.4921875f32, 4.4179688f32, 2.6855469f32, 4.890625f32,
    4.6914063f32, -1.4990234f32, 2.734375f32, 1.9208984f32, -1.1923828f32, 4.9570313f32, 5.6757813f32, 3.328125f32,
    1.8291016f32, 4.53125f32, 0.13415527f32, -3.5820313f32, -3.7460938f32, -2.4121094f32, 2.7558594f32, 0.5258789f32,
    -5.03125f32, 2.1445313f32, 0.56591797f32, -2.7890625f32, 0.6660156f32, -2.5996094f32, -1.2148438f32, -1.8085938f32,
    0.32861328f32, -5.6445313f32, 6.171875f32, 1.2871094f32, -0.37475586f32, -2.4550781f32, 7.5234375f32, 1.609375f32,
    -0.9658203f32, 0.3005371f32, -4.3085938f32, 0.5859375f32, -5.0078125f32, 5.09375f32, 0.8803711f32, -2.7773438f32,
    3.078125f32, -1.2412109f32, 0.77441406f32, -8.0f32, -6.796875f32, -3.5683594f32, -2.8671875f32, -1.3691406f32,
    3.3320313f32, -1.7392578f32, 5.3203125f32, -0.25854492f32, 5.3867188f32, -0.8510742f32, -5.6835938f32, -3.9785156f32,
    -2.34375f32, -0.38378906f32, -1.9179688f32, -1.6611328f32, 10.8359375f32, -0.099243164f32, -3.2832031f32, 0.7910156f32,
    2.6914063f32, -0.2121582f32, 1.7353516f32, -3.2207031f32, 2.7597656f32, -0.7866211f32, 3.0449219f32, 3.9765625f32,
    2.0546875f32, 3.46875f32, 1.5576172f32, -1.2158203f32, -1.78125f32, 0.89404297f32, -4.0859375f32, -6.5703125f32,
    0.5395508f32, 1.15625f32, 2.4648438f32, 3.0234375f32, -1.7275391f32, 10.6015625f32, -1.9902344f32, -6.2304688f32,
    -6.3164063f32, -0.36938477f32, -2.4277344f32, -2.2421875f32, 0.98095703f32, -3.0351563f32, -7.0195313f32, -0.99121094f32,
    -0.64501953f32, -3.5273438f32, -6.1601563f32, 6.1640625f32, 0.7607422f32, 2.3203125f32, -3.0f32, -0.8515625f32,
    -2.6386719f32, -0.09753418f32, -9.25f32, 2.2128906f32, 2.8886719f32, 0.29077148f32, -7.4335938f32, 2.5371094f32,
    -1.7324219f32, -5.484375f32, -6.8554688f32, -0.9584961f32, -0.77490234f32, 3.5371094f32, 0.796875f32, 1.8261719f32,
    -1.3632813f32, -1.4052734f32, 2.5195313f32, 3.5644531f32, -2.9042969f32, 3.1796875f32, 0.41333008f32, -2.1933594f32,
    0.9379883f32, 8.6875f32, -2.0644531f32, -2.9941406f32, 0.030975342f32, -3.703125f32, 5.03125f32, 0.81640625f32,
    -1.1962891f32, -6.6796875f32, 2.4765625f32, 0.99072266f32, -2.3496094f32, -7.078125f32, 6.1796875f32, 1.8730469f32,
    -3.5390625f32, -1.7548828f32, -1.7851563f32, -2.078125f32, -5.0664063f32, 1.5351563f32, -2.3574219f32, -1.5214844f32,
    5.0f32, 0.5205078f32, -5.21875f32, 1.5693359f32, -1.5908203f32, -2.8105469f32, 1.8261719f32, -0.5654297f32,
    -6.5976563f32, 0.36865234f32, 2.4570313f32, 2.5917969f32, -3.3183594f32, 4.0195313f32, 6.125f32, -2.6933594f32,
    4.1445313f32, -5.3515625f32, -7.0351563f32, 1.8769531f32, -10.890625f32, -3.5449219f32, 0.625f32, -0.14819336f32,
    -1.6308594f32, 3.0039063f32, -4.375f32, 6.0664063f32, 3.1523438f32, -2.40625f32, -5.3203125f32, 2.6757813f32,
    3.0371094f32, 6.1914063f32, 3.0585938f32, -6.4882813f32, -1.6787109f32, 1.5283203f32, 2.6171875f32, 2.4375f32,
    -3.515625f32, -4.1914063f32, -1.4121094f32, 0.8642578f32, 2.2441406f32, 0.35498047f32, 2.0546875f32, 1.1240234f32,
    -0.72509766f32, -1.3769531f32, -3.0429688f32, -0.18566895f32, -1.8105469f32, 4.9726563f32, 7.734375f32, 4.1992188f32,
    2.0449219f32, 3.5703125f32, -5.8046875f32, 1.1455078f32, 4.5429688f32, 4.359375f32, 5.2578125f32, -5.109375f32,
    4.6289063f32, 2.8457031f32, 2.0820313f32, -1.3955078f32, -2.046875f32, -0.87597656f32, 0.53027344f32, -1.1826172f32,
    -4.8554688f32, -1.1552734f32, -3.2734375f32, 6.2226563f32, -5.3554688f32, -3.8164063f32, 0.9404297f32, 1.7587891f32,
    -0.8803711f32, -6.21875f32, -0.88378906f32, -2.1171875f32, -1.2402344f32, 12.5390625f32, 8.40625f32, -3.0917969f32,
    -1.3974609f32, -4.3515625f32, 2.9765625f32, -0.4482422f32, -3.1582031f32, -0.08129883f32, -0.6352539f32, 5.21875f32,
    1.9248047f32, -0.17224121f32, -0.9892578f32, -3.5019531f32, -2.7851563f32, -1.5947266f32, -4.8203125f32, -0.6958008f32,
    2.7363281f32, -3.4414063f32, -3.1523438f32, 0.54248047f32, 4.2539063f32, 6.9335938f32, -2.5898438f32, 0.59375f32,
    12.3671875f32, -0.65625f32, 5.7304688f32, -2.4609375f32, -8.171875f32, -4.6289063f32, -2.5195313f32, 5.8867188f32,
    4.328125f32, -8.625f32, -2.0820313f32, -0.109375f32, -2.5097656f32, 5.0078125f32, 4.1875f32, -7.4296875f32,
    0.69970703f32, 4.0195313f32, -4.4726563f32, -7.2734375f32, -10.703125f32, 2.8984375f32, -4.9921875f32, 0.13647461f32,
    4.3632813f32, -2.5527344f32, -7.5585938f32, 0.7050781f32, 3.0878906f32, -1.8251953f32, 0.36499023f32, -0.1538086f32,
    5.3203125f32, 1.9667969f32, -0.64208984f32, 1.6679688f32, -3.4414063f32, -1.8789063f32, 5.7304688f32, -5.90625f32,
    2.6972656f32, 3.2558594f32, 0.21643066f32, 0.70166016f32, -0.5463867f32, 1.9648438f32, 2.5332031f32, -0.18591309f32,
    5.5195313f32, -0.34765625f32, -0.5097656f32, 3.7304688f32, -1.3984375f32, 2.21875f32, -1.1435547f32, 1.7890625f32,
    3.5625f32, -10.75f32, -3.9863281f32, -4.4257813f32, 4.6640625f32, 0.83984375f32, -1.7216797f32, -9.828125f32,
    1.2529297f32, -0.5449219f32, 6.2734375f32, -4.9765625f32, -5.6328125f32, 3.078125f32, 1.4755859f32, 5.3203125f32,
    -5.7539063f32, -2.7460938f32, 1.1035156f32, 1.4609375f32, 0.9350586f32, -0.74609375f32, 8.0703125f32, 7.6445313f32,
    -2.7226563f32, 1.9072266f32, -2.5136719f32, -2.2011719f32, 3.9980469f32, -0.9868164f32, -1.6865234f32, 5.4296875f32,
    -3.5390625f32, 2.4277344f32, -0.6591797f32, -0.65966797f32, -0.115356445f32, 0.67871094f32, 2.3847656f32, 8.7265625f32,
    1.0820313f32, 0.47192383f32, -3.3808594f32, -0.5288086f32, 3.4882813f32, 1.9052734f32, -4.9023438f32, -1.1289063f32,
    -2.6347656f32, -0.22509766f32, 2.1425781f32, -4.0117188f32, -8.03125f32, 0.22363281f32, 3.5058594f32, 0.032318115f32,
    -4.9023438f32, -3.1464844f32, 0.49658203f32, -0.6791992f32, -4.8242188f32, -3.2089844f32, -6.234375f32, -0.60009766f32,
    -2.0019531f32, -1.7353516f32, 3.8886719f32, 5.328125f32, 0.81689453f32, -2.59375f32, 9.2734375f32, 9.1015625f32,
    -0.30981445f32, -9.390625f32, 3.6132813f32, -6.5234375f32, -0.076660156f32, -6.4023438f32, -0.2512207f32, -5.1835938f32,
    -2.3535156f32, -8.1875f32, -4.2851563f32, -4.375f32, -0.20996094f32, -0.3166504f32, 3.0566406f32, -0.3527832f32,
    1.6376953f32, -1.6396484f32, 5.0273438f32, 4.5390625f32, -1.0751953f32, -4.203125f32, -0.14526367f32, 3.4199219f32,
    -1.4414063f32, 6.6992188f32, 5.21875f32, -0.82958984f32, 3.6484375f32, 1.0537109f32, -1.4599609f32, -0.65283203f32,
    -0.73583984f32, -7.4375f32, -2.3613281f32, 2.1757813f32, -0.50097656f32, -2.8847656f32, 3.578125f32, -1.3662109f32,
    -0.54248047f32, -3.3847656f32, 1.2041016f32, -0.86328125f32, 7.171875f32, 2.3496094f32, -0.8105469f32, 2.6523438f32,
    -2.890625f32, 4.53125f32, -6.4960938f32, 4.0429688f32, 1.5810547f32, 3.7070313f32, 6.71875f32, 8.0546875f32,
    -2.703125f32, 1.6689453f32, -1.9638672f32, 1.4316406f32, 1.7910156f32, -4.8242188f32, -2.3242188f32, -2.6796875f32,
    -1.5771484f32, 3.96875f32, -3.5644531f32, -4.71875f32, 6.0742188f32, 3.390625f32, -0.06628418f32, 2.3945313f32,
    -0.734375f32, -0.67822266f32, 8.8671875f32, -0.16992188f32, 2.84375f32, 2.2285156f32, 2.953125f32, 1.9667969f32,
    -2.3320313f32, 2.4023438f32, -2.2636719f32, 0.46777344f32, -4.6875f32, 1.8261719f32, 3.6621094f32, 5.8632813f32,
    3.0527344f32, 1.2910156f32, 3.625f32, -2.75f32, 0.85839844f32, 0.26342773f32, -5.0546875f32, -3.0996094f32,
    3.5527344f32, -3.9296875f32, -2.6347656f32, -3.5390625f32, -7.578125f32, 6.6601563f32, -1.5849609f32, 0.87597656f32,
    1.4716797f32, -2.40625f32, -0.13989258f32, 4.3164063f32, 0.92871094f32, 3.9042969f32, 3.6015625f32, -0.7294922f32,
    -1.2167969f32, -1.3154297f32, -2.2226563f32, -0.5390625f32, -1.1259766f32, 6.6953125f32, -3.7089844f32, 2.1269531f32,
    -0.08642578f32, -1.0146484f32, -1.5478516f32, -0.9086914f32, 1.5058594f32, 0.80371094f32, 4.4492188f32, 2.3242188f32,
    5.5976563f32, 1.3134766f32, 8.2109375f32, 2.3984375f32, 5.796875f32, -5.7070313f32, 2.5703125f32, 4.859375f32,
    -6.9765625f32, 6.3242188f32, -2.5546875f32, -1.6005859f32, 0.7495117f32, 0.46679688f32, -5.4492188f32, -7.1210938f32,
    5.0117188f32, -1.8300781f32, 0.5439453f32, -1.9082031f32, -0.5371094f32, 4.7304688f32, 0.09869385f32, 5.3320313f32,
    7.2460938f32, 0.36035156f32, 3.5351563f32, -1.9853516f32, 0.13781738f32, -7.7460938f32, -4.0195313f32, -2.5859375f32,
    -5.453125f32, -2.7578125f32, 1.1279297f32, -3.2597656f32, 4.234375f32, 5.1992188f32, -8.265625f32, 1.4560547f32,
    -5.8007813f32, 8.515625f32, 3.3457031f32, 2.9746094f32, 3.90625f32, 1.5800781f32, 7.4335938f32, -4.1484375f32,
    -1.3837891f32, 1.3349609f32, -3.8457031f32, -0.4362793f32, 1.2246094f32, -5.4140625f32, 3.8476563f32, -4.1015625f32,
    4.1601563f32, 1.4570313f32, -1.3417969f32, 0.7973633f32, 0.21228027f32, -5.40625f32, -1.5087891f32, 2.6933594f32,
    0.72265625f32, -7.28125f32, -3.2050781f32, 1.1904297f32, 4.859375f32, 1.703125f32, 2.4726563f32, -8.1875f32,
    -2.8066406f32, 3.3085938f32, -0.025009155f32, -0.5493164f32, -2.578125f32, -1.9667969f32, 3.3125f32, 1.8144531f32,
    -2.3027344f32, 0.37280273f32, -3.8496094f32, -0.55566406f32, -7.4023438f32, 4.578125f32, 2.4785156f32, -0.9770508f32,
    -3.9433594f32, 2.9628906f32, 2.5644531f32, -8.171875f32, -3.1933594f32, -3.9003906f32, 4.859375f32, -2.3710938f32,
    2.7226563f32, 4.2382813f32, -6.6484375f32, 2.3964844f32, -6.328125f32, 1.7578125f32, -0.30078125f32, 2.0351563f32,
    6.125f32, 0.46069336f32, -0.31762695f32, 0.33642578f32, -1.765625f32, 0.5439453f32, 0.73828125f32, 4.125f32,
    1.0917969f32, -1.2792969f32, 2.390625f32, 1.6103516f32, -1.6259766f32, -0.43847656f32, -7.6367188f32, 7.90625f32,
    0.45825195f32, 0.3190918f32, 2.9257813f32, -2.1230469f32, -3.9257813f32, -5.890625f32, -1.5859375f32, 1.9472656f32,
    -1.859375f32, -1.8564453f32, -0.32714844f32, 5.4765625f32, 1.5175781f32, -0.58203125f32, -0.4638672f32, -2.1425781f32,
    4.5273438f32, 4.8710938f32, -1.7099609f32, -0.12976074f32, -1.8710938f32, 6.3671875f32, 3.7070313f32, 3.9726563f32,
    -5.7773438f32, -0.91503906f32, 0.70996094f32, -1.46875f32, 0.4008789f32, -0.3642578f32, -3.1425781f32, 3.1503906f32,
    -2.9199219f32, 1.2353516f32, 1.2978516f32, -2.0839844f32, 4.1445313f32, -5.3476563f32, -4.7148438f32, -9.703125f32,
    4.3867188f32, -0.48120117f32, 2.8066406f32, -2.5078125f32, 4.0625f32, -5.1914063f32, 2.4785156f32, 6.9765625f32,
    -1.7089844f32, -1.4941406f32, 9.3046875f32, -0.56103516f32, -3.0390625f32, 1.2851563f32, -0.6225586f32, -3.6875f32,
    1.6230469f32, 1.1132813f32, 1.5830078f32, -0.73339844f32, -2.0097656f32, 0.5288086f32, 0.76708984f32, -5.2226563f32,
    0.6567383f32, 0.1973877f32, -2.203125f32, -6.8203125f32, -1.7744141f32, 5.0390625f32, 3.453125f32, -0.9082031f32,
    0.49682617f32, -6.8984375f32, -7.0390625f32, -3.0234375f32, -3.2792969f32, 2.3125f32, -2.1386719f32, -0.30126953f32,
    -1.1328125f32, 5.3046875f32, 1.5292969f32, 3.5175781f32, -4.7460938f32, -8.484375f32, -0.66552734f32, 0.01725769f32,
    -1.5712891f32, -1.7578125f32, 6.3125f32, -2.4375f32, -2.5664063f32, 1.0244141f32, 2.7207031f32, -0.8618164f32,
    -0.5834961f32, 1.453125f32, -4.7773438f32, -1.2529297f32, 4.6757813f32, 8.7265625f32, 1.2441406f32, 2.5371094f32,
    -3.0f32, -1.1435547f32, 3.3027344f32, 0.10205078f32, -4.5976563f32, -2.4667969f32, -5.4609375f32, 7.2148438f32,
    7.0898438f32, -1.4238281f32, -4.1054688f32, -2.5371094f32, 2.734375f32, 0.6201172f32, -1.0771484f32, 3.8242188f32,
    4.015625f32, 3.015625f32, 1.59375f32, -3.3554688f32, -4.1601563f32, 3.7011719f32, -0.52734375f32, -2.7167969f32,
    0.6142578f32, 6.234375f32, 0.11193848f32, -1.890625f32, -1.3222656f32, 3.1035156f32, 2.1113281f32, -7.9257813f32,
    -2.65625f32, 1.7509766f32, 3.3183594f32, 0.89941406f32, 5.15625f32, -5.5390625f32, 7.2226563f32, -2.6171875f32,
    -1.4716797f32, 7.8007813f32, -2.9257813f32, -5.25f32, -0.1340332f32, -6.28125f32, 1.6044922f32, 4.8984375f32,
    -0.3059082f32, -4.7539063f32, -0.3623047f32, 0.5185547f32, 0.36987305f32, 3.3535156f32, -2.7578125f32, 4.8320313f32,
    8.984375f32, 2.7851563f32, 2.7753906f32, 3.75f32, 8.796875f32, -4.3320313f32, -2.1992188f32, -1.5986328f32,
    1.7402344f32, -1.1953125f32, 0.81396484f32, 0.7636719f32, 2.3027344f32, 5.2265625f32, 3.7988281f32, 4.6484375f32,
    -0.4440918f32, -6.671875f32, -6.9453125f32, 3.0546875f32, -1.7578125f32, 0.38110352f32, 6.3632813f32, -0.4501953f32,
    -3.9628906f32, 0.77685547f32, 1.9580078f32, -3.65625f32, -1.3847656f32, 1.2714844f32, -1.8496094f32, 0.57128906f32,
    -2.2421875f32, -0.6333008f32, -0.18078613f32, -3.0722656f32, -1.2119141f32, -4.1054688f32, 1.8154297f32, -2.4199219f32,
    1.4248047f32, 6.015625f32, 2.9375f32, 0.9946289f32, 1.3300781f32, 2.7773438f32, 0.95751953f32, 1.4785156f32,
    -0.33422852f32, 3.1679688f32, -2.2265625f32, 1.6728516f32, 0.8769531f32, -3.1933594f32, -5.6875f32, -1.0087891f32,
    -0.91796875f32, -10.40625f32, 1.9384766f32, -6.4335938f32, -2.6523438f32, -3.3691406f32, -0.1550293f32, -1.7841797f32,
    0.26220703f32, -2.6484375f32, -0.08728027f32, -1.4794922f32, -1.9345703f32, 2.9492188f32, 2.5175781f32, 0.43310547f32,
    -0.59472656f32, -3.1855469f32, 5.9648438f32, 2.0292969f32, 4.84375f32, -1.9277344f32, 2.4492188f32, 5.5390625f32,
    -2.3808594f32, -5.2539063f32, 2.1210938f32, 5.4101563f32, -6.3710938f32, -0.13671875f32, -0.8198242f32, -1.5175781f32,
    1.6279297f32, 10.7578125f32, 8.96875f32, 2.6347656f32, -2.046875f32, -5.5546875f32, -0.15991211f32, 1.9589844f32,
    -0.10089111f32, 3.7246094f32, -6.6445313f32, -2.3769531f32, 3.0859375f32, 6.5585938f32, -5.0664063f32, 1.1201172f32,
    -2.046875f32, -6.140625f32, 5.578125f32, -1.4384766f32, -2.4804688f32, -2.09375f32, 11.5625f32, -4.4726563f32,
];

#[test]
#[should_panic(expected = "checkpoint/rollback is not implemented for qwen4_exp")]
fn checkpoint_and_rollback_panic_on_qwen4_exp() {
    let (dir, arch) = qwen4_install();
    let runner = RealForwardRunner::open(&dir, arch).expect("open");
    let _ = runner.checkpoint();
}
