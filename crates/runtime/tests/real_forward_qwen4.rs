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
const FROZEN_LOGIT_HASH: u64 = 18_442_354_788_410_692_061;

/// Frozen on Apple M4 Max on 2026-09-26 after the SlotStream-guided
/// Qwen4 arithmetic corrections: GDN Q/K epsilon matches direct L2
/// normalization, PLE does not apply SiLU twice, Qwen4 RMS norms are
/// uncentered, and the shared MoE branch joins after FP16 rounding.
/// This is a change detector over untrained synthetic weights, not a
/// correctness claim. The real-device hash and this vector come from the
/// same run; the vector supports the existing virtualized-CI comparison.
/// Re-freeze only with a stated reason.
#[rustfmt::skip]
const FROZEN_LOGITS: [f32; VOCAB as usize] = [
    0.09484863f32, -0.013313293f32, -0.06854248f32, 0.08538818f32, 0.10900879f32, 0.03704834f32, -0.0022239685f32, -0.097229004f32,
    0.049957275f32, -0.07086182f32, -0.010475159f32, 0.080566406f32, 0.052825928f32, -0.005832672f32, -0.0090408325f32, -0.0055770874f32,
    0.05215454f32, 0.0055999756f32, -0.041381836f32, 0.02230835f32, 0.07836914f32, 0.046203613f32, 0.03677368f32, 0.00041890144f32,
    0.048431396f32, 0.044433594f32, 0.036865234f32, -0.050598145f32, 0.04534912f32, -0.02357483f32, -0.101745605f32, 0.018493652f32,
    -0.013641357f32, -0.055419922f32, 0.093688965f32, 0.037261963f32, 0.02053833f32, 0.0703125f32, -0.039520264f32, -0.11199951f32,
    -0.064941406f32, 0.033477783f32, -0.121032715f32, -0.06958008f32, 0.08630371f32, -0.0602417f32, -0.020950317f32, 0.026977539f32,
    -0.029342651f32, -0.13146973f32, -0.08050537f32, 0.17822266f32, 0.048614502f32, -0.051452637f32, -0.005050659f32, 0.17407227f32,
    0.09466553f32, 0.16040039f32, 0.045562744f32, -0.030929565f32, -0.018981934f32, 0.08673096f32, 0.017242432f32, -0.07080078f32,
    -0.0748291f32, -0.09082031f32, 0.0637207f32, -0.046813965f32, 0.048858643f32, -0.01335907f32, -0.023468018f32, -0.02670288f32,
    -0.024963379f32, -0.075683594f32, -0.067871094f32, -0.04385376f32, 0.04864502f32, 0.014877319f32, -0.0960083f32, 0.052978516f32,
    0.06945801f32, 0.028381348f32, 0.01576233f32, -0.027755737f32, 0.02720642f32, 0.062438965f32, 0.07104492f32, 0.19262695f32,
    0.08380127f32, 0.10229492f32, -0.082214355f32, 0.06201172f32, 0.032714844f32, -0.13879395f32, -0.1484375f32, -0.0053749084f32,
    0.013427734f32, 0.070495605f32, 0.18713379f32, -0.032165527f32, 0.06347656f32, -0.15930176f32, 0.030807495f32, 0.0024375916f32,
    0.16333008f32, -0.0018577576f32, -0.02305603f32, -0.034362793f32, 0.05444336f32, 0.04147339f32, -0.05886841f32, 0.036468506f32,
    0.08282471f32, -0.0692749f32, -0.13574219f32, -0.0077438354f32, -0.037353516f32, 0.03225708f32, 0.020645142f32, -0.04486084f32,
    0.094177246f32, -0.08093262f32, -0.01979065f32, 0.021240234f32, -0.02949524f32, 0.0057525635f32, -0.07763672f32, -0.058288574f32,
    0.070007324f32, 0.06188965f32, 0.14111328f32, -0.1517334f32, 0.01727295f32, -0.11956787f32, -0.025878906f32, -0.019500732f32,
    -0.036956787f32, 0.06085205f32, 0.079589844f32, 0.007457733f32, -0.032348633f32, 0.06225586f32, -0.012062073f32, 0.020248413f32,
    -0.048980713f32, 0.02708435f32, 0.07470703f32, 0.081848145f32, 0.03805542f32, -0.08544922f32, 0.04168701f32, -0.022338867f32,
    0.008552551f32, -0.023391724f32, 0.07885742f32, 0.03253174f32, -0.016479492f32, -0.048614502f32, 0.07904053f32, -0.16455078f32,
    -0.09667969f32, 0.053955078f32, 0.12438965f32, -0.074279785f32, 0.14355469f32, -0.08227539f32, -0.10626221f32, 0.009765625f32,
    -0.028121948f32, 0.039611816f32, -0.15063477f32, 0.15307617f32, 0.037506104f32, -0.0637207f32, -0.012458801f32, -0.015350342f32,
    0.00982666f32, -0.07684326f32, -0.018981934f32, 0.0736084f32, -0.05657959f32, -0.08782959f32, -0.052368164f32, -0.09197998f32,
    -0.11077881f32, 0.038604736f32, 0.08673096f32, -0.035491943f32, -0.04547119f32, 0.12988281f32, -0.11804199f32, -0.043273926f32,
    0.05682373f32, 0.003643036f32, 0.024841309f32, -0.050994873f32, -0.17285156f32, -0.027420044f32, -0.010604858f32, 0.020858765f32,
    0.053833008f32, -0.0030994415f32, -0.020248413f32, -0.06591797f32, 0.044799805f32, 0.012580872f32, -0.109375f32, -0.113098145f32,
    -0.027755737f32, 0.057037354f32, -0.053222656f32, -0.0024719238f32, 0.08215332f32, -0.0019407272f32, -0.0011701584f32, -0.14404297f32,
    0.001958847f32, 0.017196655f32, -0.017837524f32, -0.038116455f32, -0.10229492f32, -0.03829956f32, 0.01600647f32, 0.051086426f32,
    -0.059020996f32, 0.03060913f32, 0.08868408f32, 0.022598267f32, -0.011734009f32, -0.0418396f32, -0.047180176f32, -0.08935547f32,
    -0.043914795f32, -0.024932861f32, 0.05505371f32, 0.069885254f32, -0.0027923584f32, -0.07458496f32, -0.0881958f32, -0.01651001f32,
    0.03552246f32, -0.027389526f32, 0.101745605f32, 0.10028076f32, -0.09088135f32, -0.054138184f32, 0.040863037f32, 0.16394043f32,
    -0.021636963f32, 0.030197144f32, 0.07989502f32, 0.09399414f32, -0.019805908f32, 0.030197144f32, -0.01727295f32, -0.022598267f32,
    0.13183594f32, 0.090148926f32, -0.049041748f32, 0.101989746f32, -0.067871094f32, -0.013519287f32, -0.034729004f32, 0.07409668f32,
    -0.020202637f32, 0.03161621f32, 0.068603516f32, 0.08306885f32, 0.08099365f32, -0.023101807f32, -0.12866211f32, 0.14050293f32,
    0.009857178f32, 0.00579834f32, 0.08093262f32, 0.11657715f32, -0.031921387f32, 0.0009303093f32, 0.0016527176f32, 0.024932861f32,
    0.06793213f32, 0.0034236908f32, -0.0013446808f32, 0.02067566f32, 0.0077705383f32, 0.07513428f32, -0.06112671f32, 0.006652832f32,
    -0.109436035f32, 0.12475586f32, 0.015808105f32, 0.038330078f32, -0.09692383f32, -0.08312988f32, 0.004081726f32, -0.019897461f32,
    0.16186523f32, 0.01612854f32, 0.11431885f32, -0.06933594f32, -0.10845947f32, 0.07312012f32, 0.0592041f32, -0.01209259f32,
    -0.07489014f32, 0.05239868f32, -0.005115509f32, -0.06854248f32, 0.04458618f32, -0.0067329407f32, 0.05908203f32, 0.040039063f32,
    -0.08508301f32, -0.03479004f32, 0.123168945f32, 0.024032593f32, 0.064208984f32, -0.029205322f32, -0.10534668f32, -0.054016113f32,
    -0.011932373f32, -0.053466797f32, -0.061065674f32, 0.12207031f32, -0.04675293f32, -0.09362793f32, -0.0049858093f32, -0.023284912f32,
    0.11834717f32, 0.014022827f32, 0.12463379f32, -0.10205078f32, -0.044433594f32, -0.07366943f32, 0.0064888f32, 0.053466797f32,
    0.1538086f32, 0.021743774f32, -0.1005249f32, 0.073791504f32, 0.10229492f32, 0.03161621f32, -0.03845215f32, 0.0049362183f32,
    -0.1508789f32, -0.061676025f32, -0.012786865f32, -0.086242676f32, 0.0043029785f32, -0.01234436f32, 0.056549072f32, -0.059753418f32,
    -0.117370605f32, -0.023956299f32, 0.058776855f32, -0.043945313f32, -0.049804688f32, 0.0927124f32, 0.041900635f32, 0.12402344f32,
    0.0021133423f32, 0.09063721f32, 0.013389587f32, 0.049804688f32, -0.06933594f32, -0.08099365f32, 0.015487671f32, -0.055023193f32,
    0.13708496f32, -0.011177063f32, -0.07598877f32, 0.01928711f32, -0.07080078f32, -0.012565613f32, -0.10809326f32, -0.15979004f32,
    0.0018291473f32, -0.028259277f32, -0.04309082f32, 0.005645752f32, 0.031280518f32, 0.08630371f32, 0.035125732f32, -0.024002075f32,
    -0.0637207f32, -0.009239197f32, -0.040283203f32, 0.00207901f32, -0.03817749f32, 0.018081665f32, -0.019256592f32, 0.0135650635f32,
    -0.039886475f32, 0.16113281f32, 0.16308594f32, 0.021331787f32, 0.11102295f32, 0.17651367f32, -0.040374756f32, -0.012016296f32,
    0.078552246f32, -0.00756073f32, 0.09881592f32, -0.008995056f32, -0.09338379f32, -0.042633057f32, -0.017440796f32, 0.011314392f32,
    0.011810303f32, 0.048217773f32, 0.0020580292f32, -0.0014953613f32, -0.037597656f32, -0.027420044f32, -0.022155762f32, 0.078063965f32,
    -0.06689453f32, -0.10028076f32, 0.03543091f32, -0.009986877f32, -0.10986328f32, 0.10192871f32, 0.19494629f32, 0.08972168f32,
    -0.094055176f32, -0.12792969f32, -0.031433105f32, 0.04071045f32, -0.023635864f32, 0.038146973f32, -0.10131836f32, -0.045318604f32,
    0.06298828f32, 0.01637268f32, 0.062347412f32, 0.08001709f32, 0.013046265f32, 0.010643005f32, -0.036987305f32, 0.042938232f32,
    0.011238098f32, -0.09118652f32, 0.08135986f32, -0.099731445f32, 0.10852051f32, -0.13952637f32, -0.012397766f32, -0.014564514f32,
    -0.08343506f32, -0.052093506f32, -0.06304932f32, -0.0826416f32, 0.0054779053f32, -0.14074707f32, 0.016845703f32, 0.013168335f32,
    0.014015198f32, -0.025314331f32, -0.07324219f32, -0.06378174f32, 0.1149292f32, -0.007232666f32, -0.074279785f32, 0.093566895f32,
    0.007286072f32, 0.003074646f32, 0.078125f32, 0.032196045f32, -0.12145996f32, 0.011627197f32, 0.07788086f32, 0.023025513f32,
    -0.060058594f32, -0.035095215f32, -0.028884888f32, -0.0770874f32, -0.11553955f32, -0.07397461f32, 0.037963867f32, 0.023330688f32,
    -0.0513916f32, 0.00881958f32, -0.09649658f32, -0.09460449f32, 0.03289795f32, 0.074035645f32, 0.018005371f32, -0.008529663f32,
    0.026275635f32, -0.0017795563f32, 0.042663574f32, 0.07342529f32, -0.013374329f32, 0.09790039f32, -0.1348877f32, -0.076049805f32,
    -0.03604126f32, 0.0021305084f32, -0.06915283f32, 0.07269287f32, -0.045715332f32, -0.0046043396f32, -0.00573349f32, 0.15966797f32,
    0.09869385f32, 0.013023376f32, -0.030593872f32, -0.02532959f32, -0.07550049f32, 0.051116943f32, 0.054382324f32, -0.066345215f32,
    0.044891357f32, 0.0058288574f32, -0.10003662f32, 0.1640625f32, 0.07611084f32, -0.093566895f32, -0.03414917f32, 0.09564209f32,
    -0.03149414f32, -0.026885986f32, 0.010559082f32, -0.0440979f32, 0.07086182f32, 0.027252197f32, -0.040893555f32, 0.057800293f32,
    0.04510498f32, 0.08331299f32, -0.01687622f32, -0.037017822f32, -0.09173584f32, 0.017089844f32, 0.17883301f32, -0.066223145f32,
    -0.01902771f32, -0.0552063f32, -0.035736084f32, 0.04598999f32, -0.0949707f32, -0.0725708f32, -0.047943115f32, -0.15844727f32,
    -0.058898926f32, -0.115356445f32, -0.10168457f32, 0.12878418f32, 0.032073975f32, -0.078186035f32, -0.1583252f32, -0.20544434f32,
    0.030014038f32, 0.16418457f32, -0.1640625f32, 0.04736328f32, -0.06555176f32, -0.040252686f32, -0.17260742f32, 0.03125f32,
    0.021652222f32, 0.008354187f32, 0.0061569214f32, -0.021759033f32, -0.1204834f32, 0.008834839f32, -0.06500244f32, 0.024139404f32,
    -0.056610107f32, 0.0004041195f32, 0.03286743f32, 0.064819336f32, 0.035491943f32, -0.04537964f32, -0.038024902f32, -0.032684326f32,
    0.039398193f32, -0.059661865f32, -0.11425781f32, -0.08868408f32, -0.02720642f32, 0.024047852f32, -0.040618896f32, 0.0038146973f32,
    0.042297363f32, -0.028335571f32, -0.035858154f32, 0.084472656f32, 0.06726074f32, 0.13745117f32, -0.005897522f32, 0.06964111f32,
    -0.09857178f32, -0.019485474f32, 0.02961731f32, -0.049468994f32, -0.044036865f32, 0.058502197f32, -0.00084733963f32, -0.016021729f32,
    -0.02142334f32, 0.089538574f32, -0.07897949f32, 0.024215698f32, -0.15734863f32, -0.048217773f32, 0.03704834f32, -0.04977417f32,
    -0.010261536f32, -0.018447876f32, 0.029754639f32, 0.015556335f32, 0.113098145f32, 0.018676758f32, -0.015701294f32, -0.013412476f32,
    0.018966675f32, -0.09753418f32, 0.024734497f32, 0.01802063f32, -0.049621582f32, 0.0026245117f32, 0.034332275f32, 0.032348633f32,
    -0.034973145f32, 0.0056114197f32, -0.05493164f32, -0.037963867f32, -0.057006836f32, -0.009536743f32, -0.030090332f32, -0.03111267f32,
    -0.13684082f32, 0.05343628f32, -0.05404663f32, -0.029052734f32, -0.01146698f32, -0.060546875f32, 0.01612854f32, 0.027420044f32,
    -0.00541687f32, -0.08880615f32, 0.014183044f32, 0.0119018555f32, -0.08312988f32, 0.076049805f32, -0.011253357f32, -0.09613037f32,
    0.10369873f32, -0.10229492f32, -0.06744385f32, -0.046936035f32, 0.04321289f32, -0.12719727f32, -0.09338379f32, -0.060577393f32,
    0.058502197f32, 0.0022087097f32, 0.047943115f32, 0.033813477f32, -0.009315491f32, 0.013206482f32, -0.024414063f32, 0.021652222f32,
    -0.0914917f32, -0.029754639f32, 0.0960083f32, 0.06402588f32, -0.093444824f32, 0.038726807f32, -0.029754639f32, -0.0029067993f32,
    -0.00434494f32, 0.055114746f32, 0.058685303f32, 0.0473938f32, -0.019058228f32, 0.05682373f32, 0.024230957f32, 0.14038086f32,
    0.031311035f32, -0.0055732727f32, 0.119628906f32, -0.09576416f32, -0.013252258f32, 0.066101074f32, 0.037750244f32, -0.019836426f32,
    -0.039001465f32, -0.10333252f32, -0.028945923f32, -0.051605225f32, -0.046142578f32, -0.11029053f32, -0.011917114f32, -0.006931305f32,
    0.095825195f32, 0.0418396f32, 0.06591797f32, 0.04711914f32, 0.112976074f32, 0.046875f32, 0.02935791f32, 0.00969696f32,
    -0.010543823f32, -0.13122559f32, 0.1517334f32, 0.00037837029f32, 0.08319092f32, 0.007411957f32, -0.038146973f32, 0.10900879f32,
    -0.11340332f32, 0.041931152f32, -0.12854004f32, -0.024841309f32, -0.052520752f32, 0.09100342f32, 0.026947021f32, -0.09466553f32,
    -0.0068206787f32, 0.064208984f32, -0.0435791f32, -0.03314209f32, 0.08282471f32, 0.0005249977f32, -0.061187744f32, 0.13049316f32,
    0.027328491f32, 0.0062561035f32, -0.04055786f32, 0.007701874f32, -0.0045166016f32, 0.060058594f32, 0.030960083f32, 0.01928711f32,
    0.037322998f32, 0.025146484f32, -0.076416016f32, -0.05984497f32, -0.10491943f32, 0.0010375977f32, -0.064941406f32, -0.015037537f32,
    -0.038604736f32, 0.13928223f32, -0.08062744f32, 0.039886475f32, -0.047027588f32, -0.031951904f32, -0.06744385f32, -0.034362793f32,
    0.072631836f32, -0.08880615f32, -0.12646484f32, -0.00073099136f32, -0.0524292f32, -0.10461426f32, 0.07623291f32, -0.04776001f32,
    0.022216797f32, -0.010757446f32, 0.03555298f32, -0.060028076f32, -0.015930176f32, 0.004425049f32, 0.06976318f32, -0.085510254f32,
    -0.13293457f32, 0.088134766f32, -0.16662598f32, 0.005027771f32, 0.043518066f32, -0.00016283989f32, 0.00065135956f32, 0.03591919f32,
    0.052978516f32, -0.08282471f32, -0.09887695f32, 0.028945923f32, 0.051086426f32, 0.036376953f32, 0.121154785f32, 0.027435303f32,
    -0.032104492f32, -0.03225708f32, -0.014427185f32, -0.059509277f32, 0.09466553f32, -0.07470703f32, 0.017837524f32, 0.08508301f32,
    -0.050689697f32, -0.10803223f32, -0.13879395f32, -0.10839844f32, -0.08996582f32, -0.05319214f32, 0.03201294f32, 0.087646484f32,
    0.09112549f32, -0.11035156f32, -0.08892822f32, 0.09625244f32, -0.052825928f32, 0.08807373f32, 0.15014648f32, 0.0059509277f32,
    -0.035003662f32, -4.9114227e-05f32, -0.0814209f32, 0.014717102f32, 0.039367676f32, 0.015266418f32, -0.13427734f32, -0.06573486f32,
    0.016906738f32, 0.07489014f32, -0.005519867f32, -0.043640137f32, -0.007484436f32, -0.00031590462f32, -0.082458496f32, -0.14526367f32,
    0.07342529f32, 0.034423828f32, -0.00856781f32, -0.015472412f32, 0.044708252f32, 0.0592041f32, -0.04638672f32, -0.03152466f32,
    -0.09118652f32, -0.019577026f32, 0.092163086f32, 0.05480957f32, -0.08105469f32, 0.099853516f32, 0.009490967f32, 0.053710938f32,
    0.062561035f32, -0.10986328f32, -0.0034122467f32, 0.05340576f32, 0.1538086f32, -0.06286621f32, 0.05871582f32, -0.029174805f32,
    0.06774902f32, 0.13134766f32, -0.046203613f32, -0.0017757416f32, 0.018447876f32, 0.06427002f32, 0.13342285f32, 0.05291748f32,
    -0.027023315f32, 0.08477783f32, -0.0284729f32, 0.03201294f32, 0.06323242f32, 0.120788574f32, -0.022735596f32, -0.054138184f32,
    -0.055267334f32, -0.06890869f32, -0.06359863f32, -0.06561279f32, -0.04498291f32, 0.051605225f32, -0.04071045f32, 0.013542175f32,
    -0.011009216f32, -0.023864746f32, 0.0014734268f32, -0.037078857f32, -0.103149414f32, -0.12384033f32, -0.021469116f32, -0.038879395f32,
    0.049835205f32, 0.09893799f32, -0.07458496f32, -0.012069702f32, 0.008911133f32, -0.013748169f32, 0.04309082f32, 0.027816772f32,
    0.056396484f32, 0.03491211f32, -0.047912598f32, 0.06124878f32, -0.09729004f32, 0.02696228f32, 0.05368042f32, -0.015022278f32,
    -0.05795288f32, 0.010681152f32, -0.010848999f32, -0.06060791f32, 0.035247803f32, -0.05206299f32, 0.00233078f32, -0.01423645f32,
    -0.06982422f32, 0.107543945f32, 0.19580078f32, 0.030914307f32, 0.08874512f32, 0.023544312f32, -0.05255127f32, 0.037384033f32,
    -0.07232666f32, 0.025100708f32, 0.058502197f32, 0.05239868f32, 0.012214661f32, 0.05130005f32, -0.1940918f32, 0.0002245903f32,
    -0.0020523071f32, -0.14257813f32, 0.010643005f32, -0.066833496f32, 0.04083252f32, -0.04244995f32, 0.04324341f32, 0.081604004f32,
    -0.03274536f32, 0.21862793f32, 0.038726807f32, 0.009384155f32, 0.11413574f32, 0.051971436f32, -0.029052734f32, 0.06628418f32,
    -0.07067871f32, 0.06768799f32, 0.07775879f32, 0.010765076f32, -0.032226563f32, -0.0067443848f32, -0.07537842f32, -0.03390503f32,
    0.01474762f32, -0.013633728f32, 0.020614624f32, -0.04421997f32, -0.1385498f32, -0.043548584f32, 0.011184692f32, 0.11212158f32,
    -0.11419678f32, 0.0064735413f32, -0.06213379f32, 0.1809082f32, 0.04043579f32, -0.039794922f32, -0.09674072f32, 0.021606445f32,
    0.028656006f32, -0.040771484f32, 0.093322754f32, -0.028137207f32, -0.075805664f32, -0.061035156f32, 0.07244873f32, -0.055908203f32,
    -0.0340271f32, 0.05697632f32, 0.080566406f32, -0.062347412f32, -0.095214844f32, -0.017471313f32, -0.0010519028f32, -0.0033931732f32,
    -0.06262207f32, -0.07946777f32, -0.07019043f32, 0.041992188f32, 0.01676941f32, 0.028656006f32, -0.109680176f32, -0.059753418f32,
    0.051818848f32, -0.101135254f32, -0.01524353f32, 0.022247314f32, -0.006542206f32, 0.11816406f32, -0.022994995f32, 0.07525635f32,
    -0.04031372f32, 0.09289551f32, 0.02709961f32, 0.0033187866f32, -0.031036377f32, 0.01914978f32, -0.0982666f32, -0.042419434f32,
    0.041748047f32, -0.08026123f32, -0.095458984f32, 0.062927246f32, -0.03277588f32, -0.061798096f32, -0.0024776459f32, -0.10217285f32,
    0.01625061f32, 0.047332764f32, 0.05053711f32, -0.117004395f32, 0.04006958f32, 0.05090332f32, -0.014923096f32, -0.02935791f32,
    -0.032043457f32, -0.003250122f32, -0.08868408f32, 0.011528015f32, 0.052459717f32, -0.16064453f32, 0.050720215f32, 0.0090408325f32,
    0.12866211f32, 0.011390686f32, -0.0002925396f32, 0.047851563f32, -0.070739746f32, 0.080566406f32, -0.019104004f32, 0.008659363f32,
    -0.13330078f32, 0.11853027f32, 0.07470703f32, -0.08569336f32, 0.015853882f32, -0.1385498f32, 0.105773926f32, 0.11645508f32,
    0.03652954f32, -0.036743164f32, -0.07507324f32, -0.07104492f32, -0.0181427f32, 0.095825195f32, 0.028961182f32, -0.04296875f32,
    0.015838623f32, 0.13684082f32, 0.06237793f32, -0.15759277f32, 0.0036582947f32, 0.01033783f32, -0.056030273f32, -0.10473633f32,
    0.038604736f32, 0.009521484f32, 0.10668945f32, 0.0018043518f32, 0.021621704f32, 0.06573486f32, 0.018051147f32, -0.058380127f32,
    -0.08746338f32, -0.01210022f32, -0.008033752f32, 0.04486084f32, -0.014328003f32, -0.006706238f32, -0.025924683f32, 0.022598267f32,
    0.066833496f32, -0.071777344f32, -0.022506714f32, 0.02746582f32, 0.0067481995f32, 0.053649902f32, -0.00024020672f32, 0.00024604797f32,
    -0.09423828f32, 0.07342529f32, 0.010040283f32, 0.048919678f32, 0.12310791f32, -0.06124878f32, 0.048797607f32, 0.027908325f32,
    -0.06573486f32, 0.1027832f32, -0.059295654f32, 0.031433105f32, 0.09375f32, -0.08618164f32, 0.056610107f32, -0.16394043f32,
    -0.040039063f32, -0.030334473f32, -0.11395264f32, -0.04864502f32, 0.071777344f32, 0.10418701f32, -0.06524658f32, 0.09503174f32,
    -0.20178223f32, -0.16027832f32, 0.036010742f32, -0.07501221f32, -0.011245728f32, -0.011161804f32, 0.030029297f32, -0.008483887f32,
    -0.03213501f32, 0.017196655f32, 0.08874512f32, 0.10870361f32, -0.012962341f32, -0.029006958f32, -0.0029277802f32, -0.0826416f32,
    -0.05279541f32, -0.11462402f32, 0.04852295f32, 0.1270752f32, -0.008087158f32, -0.023803711f32, 0.12988281f32, -0.036834717f32,
    0.09649658f32, -0.09777832f32, -0.05303955f32, -0.02229309f32, -0.018798828f32, -0.16577148f32, -0.013442993f32, 0.010795593f32,
    -0.07672119f32, 0.021514893f32, -0.05444336f32, -0.05114746f32, -0.0038814545f32, -0.10089111f32, 0.03375244f32, -0.010551453f32,
    -0.1270752f32, -0.08343506f32, -0.10675049f32, -0.09320068f32, -0.020233154f32, -0.013381958f32, -0.07507324f32, -0.07556152f32,
    0.10681152f32, -0.007003784f32, 0.07397461f32, -0.101989746f32, 0.062438965f32, -0.059814453f32, 0.11663818f32, 0.034851074f32,
    -0.09686279f32, 0.017501831f32, -0.12841797f32, 0.07122803f32, -0.0446167f32, 0.049987793f32, -0.022842407f32, -0.046905518f32,
    -0.012542725f32, 0.005004883f32, 0.06274414f32, 0.029632568f32, 0.072509766f32, 0.04345703f32, 0.036499023f32, 0.028167725f32,
    -0.058898926f32, -0.040374756f32, -0.03439331f32, 0.044006348f32, 0.07232666f32, 0.04244995f32, -0.045440674f32, -0.04107666f32,
    -0.0013713837f32, -0.0064888f32, 0.05609131f32, -0.15942383f32, 0.010536194f32, 0.07733154f32, 0.069885254f32, -0.0015525818f32,
    -0.04421997f32, 0.07836914f32, 0.047668457f32, 0.06213379f32, -0.10284424f32, 0.012207031f32, -0.10620117f32, 0.020828247f32,
    0.028045654f32, -0.0395813f32, 0.024261475f32, -0.04019165f32, 0.08917236f32, -0.035186768f32, 0.05508423f32, -0.105041504f32,
    -0.016479492f32, -0.08203125f32, -0.09869385f32, -0.07495117f32, 0.0635376f32, -0.01411438f32, -0.012718201f32, -0.06915283f32,
    0.08520508f32, 0.058288574f32, -0.04714966f32, -0.1583252f32, -0.081970215f32, -0.11016846f32, -0.11853027f32, -0.02998352f32,
    0.04248047f32, 0.03945923f32, 0.044952393f32, 0.039123535f32, -0.08135986f32, -0.0008497238f32, 0.007789612f32, -0.037902832f32,
    0.021881104f32, 0.08654785f32, 0.019485474f32, 0.037384033f32, 0.105651855f32, -0.07543945f32, 0.059570313f32, -0.037963867f32,
    -0.009414673f32, -0.045440674f32, -0.089904785f32, -0.018463135f32, -0.053619385f32, -0.012229919f32, -0.03555298f32, -0.05795288f32,
    -0.017105103f32, -0.07312012f32, -0.042144775f32, 0.10632324f32, 0.017532349f32, 0.07373047f32, 0.05618286f32, 0.019256592f32,
    0.072021484f32, -0.025039673f32, 0.0836792f32, 0.06427002f32, 0.10089111f32, 0.08050537f32, 0.037506104f32, -0.07336426f32,
    0.09051514f32, 0.06854248f32, 0.026824951f32, -0.07159424f32, -0.015205383f32, -0.0021915436f32, 0.050994873f32, 0.09490967f32,
    0.18237305f32, 0.08514404f32, 0.06173706f32, 0.0032196045f32, 0.07342529f32, -0.057434082f32, -0.051696777f32, -0.1217041f32,
    0.005191803f32, 0.06530762f32, -0.007858276f32, -0.029968262f32, -0.005962372f32, -0.010017395f32, 0.07147217f32, -0.071899414f32,
    -0.06384277f32, 0.11065674f32, 0.011985779f32, 0.07446289f32, 0.08648682f32, 0.035614014f32, -0.03640747f32, 0.07330322f32,
    0.0015640259f32, 0.042022705f32, -0.010536194f32, 0.0066871643f32, 0.029693604f32, -0.01828003f32, -0.017364502f32, -0.08569336f32,
    -0.117248535f32, -0.062072754f32, 0.009529114f32, -0.12573242f32, -0.043884277f32, 0.0892334f32, -0.03286743f32, 0.092163086f32,
    0.070739746f32, -5.954504e-05f32, -0.034942627f32, -0.027786255f32, 0.030670166f32, 0.15026855f32, 0.0048332214f32, -0.020721436f32,
    0.0016403198f32, -0.0012731552f32, 0.021514893f32, -0.03845215f32, 0.043548584f32, 0.010429382f32, -0.04876709f32, -0.02444458f32,
    -0.08911133f32, -0.06829834f32, -0.051116943f32, -0.07336426f32, -0.034454346f32, 0.06549072f32, -0.037719727f32, -0.00038790703f32,
    0.06109619f32, -0.043701172f32, 0.02885437f32, 0.13415527f32, 0.15197754f32, 0.11895752f32, -0.18688965f32, 0.038848877f32,
    -0.014305115f32, -0.1451416f32, -0.03741455f32, 0.01651001f32, -0.0085372925f32, 0.044281006f32, -0.02420044f32, -0.023986816f32,
    -0.024368286f32, 0.1159668f32, 0.123291016f32, -0.0021858215f32, -0.035369873f32, -0.047576904f32, -0.1430664f32, -0.07293701f32,
    -0.0463562f32, 0.083984375f32, -0.13879395f32, 0.062408447f32, 0.012374878f32, 0.09118652f32, -0.018417358f32, 0.034332275f32,
    -0.03289795f32, -0.0032424927f32, -0.031234741f32, 0.12011719f32, -0.17736816f32, -0.066223145f32, 0.030929565f32, 0.0026721954f32,
    0.05001831f32, 0.0055999756f32, 0.058563232f32, -0.061828613f32, 0.019165039f32, -0.026153564f32, 0.12017822f32, -0.068603516f32,
    -0.066223145f32, -0.07110596f32, -0.013702393f32, -0.032592773f32, 0.14111328f32, 0.037353516f32, 0.11206055f32, -0.12512207f32,
    0.008613586f32, -0.018356323f32, 0.020965576f32, 0.02458191f32, -0.021392822f32, -0.01739502f32, -0.027557373f32, -0.008056641f32,
    -0.028366089f32, 0.045532227f32, -0.11273193f32, 0.0011911392f32, -0.047424316f32, 0.07055664f32, -0.090270996f32, -0.0027561188f32,
    -0.054901123f32, 0.078186035f32, 0.06719971f32, 0.0022468567f32, 0.033477783f32, -0.026947021f32, 0.014625549f32, 0.020950317f32,
    0.095336914f32, 0.042175293f32, -0.14013672f32, -0.015197754f32, 0.21044922f32, -0.032470703f32, 0.031311035f32, -0.07196045f32,
    -0.009483337f32, 0.056549072f32, 0.07122803f32, -0.007030487f32, -0.02192688f32, 0.09863281f32, 0.031555176f32, -0.0670166f32,
    -0.070617676f32, 0.017929077f32, -0.08428955f32, -0.041748047f32, -0.0032176971f32, 0.066467285f32, -0.0068855286f32, 0.13061523f32,
    0.0085372925f32, 0.11303711f32, 0.07519531f32, 0.027999878f32, 0.030944824f32, -0.040649414f32, 0.040496826f32, 0.09442139f32,
    0.014595032f32, -0.10852051f32, -0.0096588135f32, -0.021530151f32, 0.048614502f32, 0.03866577f32, -0.04824829f32, 0.027832031f32,
    -0.035461426f32, -0.01576233f32, 0.066223145f32, 0.016555786f32, -0.07220459f32, -0.076293945f32, -0.030792236f32, -0.020935059f32,
    0.0927124f32, -0.099121094f32, -0.0067367554f32, -0.056732178f32, -0.006626129f32, 0.058013916f32, 0.048339844f32, -0.10101318f32,
    -0.0023345947f32, 0.1484375f32, -0.022888184f32, -0.10040283f32, 0.019470215f32, -0.103027344f32, 0.13098145f32, 0.0047836304f32,
    -0.006122589f32, -0.041259766f32, -0.025634766f32, 0.0060195923f32, 0.0051956177f32, 0.21838379f32, -0.021408081f32, 0.09887695f32,
    -0.015945435f32, -0.02067566f32, -0.13183594f32, 0.009223938f32, 0.061676025f32, 0.0357666f32, 0.029342651f32, -0.021743774f32,
    0.09283447f32, 0.028884888f32, -0.10272217f32, 0.11199951f32, 0.00945282f32, 0.06500244f32, 0.040374756f32, 0.04574585f32,
    -0.0146484375f32, -0.11413574f32, -0.021224976f32, 0.028335571f32, -0.08660889f32, 0.018661499f32, -0.16125488f32, -0.01133728f32,
    -0.13464355f32, -0.013114929f32, -0.022903442f32, 0.0075416565f32, 0.03237915f32, -0.011268616f32, 0.015777588f32, -0.031097412f32,
    0.047912598f32, 0.1003418f32, -0.17687988f32, -0.0052223206f32, 0.062194824f32, 0.003479004f32, -0.036376953f32, -0.13586426f32,
    -0.12878418f32, -0.014472961f32, 0.0067749023f32, -0.10675049f32, -0.0423584f32, -0.034057617f32, -0.068237305f32, -0.040527344f32,
    0.066467285f32, 0.004310608f32, -0.015075684f32, 0.01878357f32, -0.036712646f32, -0.15283203f32, -0.0116119385f32, -0.08734131f32,
    -0.013549805f32, 0.03048706f32, 0.066223145f32, -0.089538574f32, 0.020599365f32, -0.0017318726f32, 0.059631348f32, 0.01878357f32,
    0.06173706f32, 0.05960083f32, -0.06262207f32, 0.008346558f32, 0.10656738f32, -0.12261963f32, -0.0317688f32, -0.11352539f32,
    -0.0052757263f32, -0.09906006f32, -0.12561035f32, -0.022033691f32, -0.023986816f32, 0.03677368f32, -0.033416748f32, -0.048614502f32,
    0.0625f32, -0.020309448f32, 0.03427124f32, 0.050445557f32, 0.0012588501f32, 0.03186035f32, -0.00573349f32, 0.025375366f32,
    -0.04260254f32, 0.01448822f32, 0.022659302f32, 0.08105469f32, -0.039367676f32, 0.049957275f32, -0.15490723f32, 0.04272461f32,
    -0.06665039f32, -0.022827148f32, -0.044006348f32, -0.078308105f32, 0.011672974f32, -0.061676025f32, 0.0012626648f32, 0.037109375f32,
    0.02468872f32, -0.030731201f32, -0.17614746f32, 0.031188965f32, -0.055114746f32, -0.06536865f32, -0.12866211f32, -0.0149002075f32,
    0.032470703f32, -0.0052108765f32, 0.059692383f32, -0.035369873f32, -0.07092285f32, -0.053588867f32, -0.039855957f32, -0.07672119f32,
    -0.013252258f32, -0.029953003f32, 0.02017212f32, 0.053344727f32, 0.032836914f32, -0.019439697f32, -0.023880005f32, -0.007411957f32,
    0.022003174f32, -0.066223145f32, 0.0060043335f32, -0.07342529f32, -0.044647217f32, -0.1081543f32, 0.020767212f32, 0.033172607f32,
    -0.044952393f32, 0.015388489f32, 0.043060303f32, -0.087402344f32, 0.022125244f32, -0.016601563f32, -0.066589355f32, 0.048187256f32,
    -0.034820557f32, -0.109436035f32, 0.052764893f32, 0.033172607f32, -0.012130737f32, -0.023635864f32, -0.016586304f32, -0.03640747f32,
    0.092285156f32, -0.079956055f32, -0.014320374f32, 0.022964478f32, -0.0010471344f32, -0.061035156f32, 0.10821533f32, 0.0049209595f32,
    -0.05645752f32, 0.046325684f32, -0.03753662f32, 0.009384155f32, 0.070617676f32, -0.054260254f32, 0.022354126f32, 0.033966064f32,
    -0.058563232f32, 0.08093262f32, 0.04034424f32, 0.046417236f32, 0.00440979f32, 0.09136963f32, 0.03010559f32, -0.028747559f32,
    0.045013428f32, 0.004196167f32, -0.038482666f32, 0.019866943f32, -0.09222412f32, -0.12988281f32, 0.049102783f32, 0.056732178f32,
    -0.0018472672f32, 0.010253906f32, -0.015686035f32, -0.055511475f32, 0.05758667f32, 0.07788086f32, -0.051818848f32, -0.03390503f32,
    -0.035614014f32, -0.01965332f32, -0.01197052f32, 0.08569336f32, 0.003824234f32, 0.014091492f32, -0.019241333f32, 0.018081665f32,
    0.009048462f32, -0.05340576f32, 0.07659912f32, 0.05831909f32, -0.11505127f32, -0.09716797f32, 0.036132813f32, -0.055236816f32,
    0.08093262f32, -0.0067443848f32, 0.10192871f32, -0.050476074f32, 0.07928467f32, -0.046844482f32, 0.005332947f32, 0.013999939f32,
    0.035003662f32, 0.005695343f32, -0.036102295f32, -0.11431885f32, -0.08850098f32, -0.09063721f32, -0.035186768f32, -0.087768555f32,
    -0.07122803f32, -0.08319092f32, 0.022537231f32, -0.018051147f32, 0.05935669f32, -0.013244629f32, -0.014564514f32, -0.023101807f32,
    0.053894043f32, -0.009384155f32, -0.049713135f32, -0.067993164f32, -0.026565552f32, -0.104003906f32, 0.04083252f32, 0.03805542f32,
    0.012046814f32, -0.0541687f32, -0.03704834f32, -0.028640747f32, -0.0056877136f32, -0.07476807f32, -0.0178833f32, -0.0211792f32,
    -0.038085938f32, 0.008255005f32, 0.055145264f32, -0.028457642f32, -0.08703613f32, 0.035217285f32, -0.042755127f32, 0.0055885315f32,
    -0.03656006f32, -0.10974121f32, 0.07702637f32, 0.0574646f32, 0.0042648315f32, 0.036834717f32, 0.014549255f32, 0.03881836f32,
    0.0074157715f32, 0.05041504f32, -0.076538086f32, -0.07373047f32, 0.10321045f32, -0.14208984f32, 0.08886719f32, -0.025634766f32,
    -0.099365234f32, 0.03930664f32, -0.21191406f32, 0.008300781f32, -0.060394287f32, -0.024810791f32, -0.045837402f32, -0.026519775f32,
    0.0075950623f32, 0.10681152f32, -0.14147949f32, -0.061645508f32, 0.043914795f32, -0.021469116f32, 0.035186768f32, -0.06958008f32,
    -0.066467285f32, -0.013153076f32, -0.10662842f32, 0.059051514f32, 0.09100342f32, -0.09362793f32, -0.014915466f32, 0.0055007935f32,
    -0.024505615f32, 0.019744873f32, -0.10443115f32, -0.17700195f32, 0.09399414f32, 0.09893799f32, -0.09460449f32, 0.0033931732f32,
    -0.03353882f32, 0.056152344f32, -0.045776367f32, -0.033813477f32, -0.027694702f32, -0.03805542f32, 0.015808105f32, 0.033081055f32,
    -0.050201416f32, -0.008903503f32, -0.09509277f32, -0.027130127f32, -0.052490234f32, -0.07720947f32, -0.022567749f32, 0.02279663f32,
    0.07946777f32, 0.03100586f32, 0.025848389f32, 0.046569824f32, -0.022384644f32, 0.13635254f32, -0.050476074f32, -0.00044322014f32,
    0.04977417f32, -0.027832031f32, -0.06616211f32, -0.003753662f32, 0.042785645f32, -0.030899048f32, 0.12432861f32, -0.0635376f32,
    -0.15979004f32, -0.07086182f32, -0.018447876f32, -0.076538086f32, -0.0552063f32, -0.01374054f32, 0.061523438f32, -0.066101074f32,
    0.03878784f32, 0.08099365f32, -0.024490356f32, 0.037322998f32, -0.040039063f32, -0.068725586f32, -0.06210327f32, -0.042755127f32,
    0.04031372f32, 0.07562256f32, -0.04397583f32, -0.042877197f32, 0.09881592f32, -0.017852783f32, 0.099121094f32, -0.015335083f32,
    0.09399414f32, -0.12005615f32, 0.047729492f32, -0.0340271f32, -0.047210693f32, 0.040222168f32, 0.0072364807f32, 0.064208984f32,
    -0.060333252f32, -0.06585693f32, 0.015533447f32, -0.026290894f32, 0.007972717f32, 0.020401001f32, 0.07507324f32, 0.0016527176f32,
    -0.08581543f32, 0.12780762f32, 0.11468506f32, 0.03265381f32, 0.083740234f32, 0.10827637f32, -0.09503174f32, 0.020996094f32,
    -0.026031494f32, -0.04296875f32, 0.0014371872f32, 0.0758667f32, -0.03756714f32, 0.0067825317f32, 0.01158905f32, -0.10723877f32,
    0.0132751465f32, 0.018585205f32, -0.007820129f32, -0.06738281f32, -0.09362793f32, 0.046539307f32, -0.0960083f32, -0.041748047f32,
    0.08569336f32, 0.0925293f32, -0.12243652f32, -0.009407043f32, -0.10852051f32, -0.1842041f32, 0.0574646f32, 0.06890869f32,
    0.021209717f32, -0.08996582f32, -0.048614502f32, -0.0032100677f32, 0.05255127f32, -0.09185791f32, 0.027328491f32, -0.008430481f32,
    0.01737976f32, -0.02178955f32, 0.047210693f32, 0.084106445f32, 0.022949219f32, -0.13623047f32, -0.035583496f32, 0.013031006f32,
    -0.021469116f32, -0.08996582f32, 0.031677246f32, -0.027114868f32, 0.027328491f32, -0.0090408325f32, -0.026809692f32, 0.04324341f32,
    -0.095581055f32, -0.105895996f32, 0.0039405823f32, 0.05154419f32, -0.017364502f32, 0.03729248f32, -0.009742737f32, 0.022521973f32,
    0.018600464f32, -0.031234741f32, 0.068847656f32, 0.03265381f32, 0.07287598f32, -0.06311035f32, 0.090148926f32, 0.0035247803f32,
    -0.023956299f32, -0.011299133f32, -0.012039185f32, 0.049682617f32, -0.004055023f32, 0.03793335f32, 0.0033187866f32, 0.058898926f32,
    0.019577026f32, 0.017700195f32, 0.060913086f32, 0.052368164f32, 0.034088135f32, -0.072753906f32, 0.001578331f32, -0.0018777847f32,
    0.061462402f32, -0.0758667f32, -0.03604126f32, 0.050811768f32, 0.088378906f32, -0.08068848f32, 0.085632324f32, 0.09094238f32,
    0.006526947f32, -0.03918457f32, -0.032348633f32, -0.015029907f32, 0.09100342f32, -0.04638672f32, 0.058624268f32, 0.0011119843f32,
    0.044158936f32, 0.060760498f32, 0.06652832f32, -0.031341553f32, 0.08648682f32, -0.019088745f32, -0.072631836f32, 0.16870117f32,
    0.088134766f32, -0.053100586f32, -0.0030403137f32, 0.025558472f32, -0.01234436f32, -0.061798096f32, 0.09289551f32, 0.009666443f32,
    0.0011949539f32, -0.11828613f32, -0.020339966f32, 0.02154541f32, 0.017959595f32, 0.03112793f32, -0.055664063f32, -0.039154053f32,
    -0.09063721f32, -0.0011301041f32, -0.0032634735f32, 0.0028686523f32, 0.022567749f32, 0.07354736f32, -0.078063965f32, 0.06652832f32,
    -0.061798096f32, 0.044311523f32, -0.0063972473f32, -0.0048179626f32, 0.017456055f32, 0.02609253f32, 0.105773926f32, -0.09118652f32,
    0.049438477f32, 0.027160645f32, -0.042907715f32, -0.050323486f32, 0.027420044f32, -0.039886475f32, 0.061187744f32, -0.13342285f32,
    -0.09729004f32, 0.07006836f32, -0.080322266f32, -0.03942871f32, -0.03842163f32, 0.13085938f32, 0.011451721f32, 0.11566162f32,
    0.1071167f32, -0.021331787f32, 0.016067505f32, 0.016571045f32, 0.026565552f32, -0.024017334f32, 0.05230713f32, -0.07080078f32,
    -0.08770752f32, -0.022705078f32, 0.05331421f32, 0.03768921f32, 0.004295349f32, -0.028015137f32, -0.0043945313f32, 0.076660156f32,
    0.0009841919f32, 0.05166626f32, -0.048858643f32, -0.04736328f32, -0.0009288788f32, 0.013183594f32, 0.068115234f32, -0.04434204f32,
];

#[test]
#[should_panic(expected = "checkpoint/rollback is not implemented for qwen4_exp")]
fn checkpoint_and_rollback_panic_on_qwen4_exp() {
    let (dir, arch) = qwen4_install();
    let runner = RealForwardRunner::open(&dir, arch).expect("open");
    let _ = runner.checkpoint();
}
