#![cfg(target_os = "macos")]
//! Synthetic `qwen3_vl` coverage for the shared Llama flow's per-head-qk-norm
//! dense arm (the `Qwen3Dense` switch combination plus a TIED head).
//!
//! Weights are untrained, so assertions are structural: finite logits, the
//! tied head actually reading the embedding table, the per-head q/k norms
//! actually reaching the logits, and prefill-chunk equals sequential-prefill.
//!
//! KNOWN NON-COVERAGE, mutation-checked: flipping the family's `rms_eps`
//! arm (1e-6 against the flow's 1e-5 default) leaves every case here green
//! -- the difference is below what untrained weights can produce, which is
//! the invariant-doing-the-work shape AGENTS.md's mutation rule names. That
//! arm's real guard is the family's own quality gate on the pinned install,
//! the same place `qwen3`'s epsilon is held.

use half::f16;
use turbospark_repack::{build_synthetic_qwen3_vl_install, tiny_qwen3_vl_arch};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("turbospark-qwen3vl-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn open(dir: &std::path::Path) -> RealForwardRunner {
    let arch = turbospark_repack::peek_manifest_arch(dir).expect("manifest arch peeks");
    RealForwardRunner::open(dir, arch).expect("qwen3_vl install opens")
}

fn logits_after_prompt(dir: &std::path::Path, tokens: &[i32]) -> Vec<f16> {
    let mut runner = open(dir);
    let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in tokens.iter().enumerate() {
        if position + 1 == tokens.len() {
            runner
                .produce(token, position, &mut logits)
                .expect("produces");
        } else {
            runner
                .produce_prefill(token, position, &mut logits)
                .expect("prefill produces");
        }
    }
    logits
}

#[test]
fn qwen3vl_install_opens_and_produces_finite_logits() {
    let dir = temp_dir("decode");
    let arch = build_synthetic_qwen3_vl_install(&dir, VOCAB, LAYERS, "tiny-qwen3vl")
        .expect("qwen3_vl fixture builds");
    assert_eq!(arch.family, model_io::ModelFamily::Qwen3Vl);
    assert_eq!(arch.num_experts, 0);
    assert!(arch.tie_word_embeddings, "the family's head is tied");

    let logits = logits_after_prompt(&dir, &[5]);
    assert!(
        logits.iter().all(|v| v.to_f32().is_finite()),
        "qwen3_vl logits must remain finite"
    );
}

#[test]
fn qwen3vl_manifest_round_trips_the_family_string() {
    // The canonical persisted string is a format constant: this is the one
    // assertion that pins what the manifest really writes, so a rename that
    // strands every install on disk reddens here rather than at open.
    let dir = temp_dir("manifest");
    build_synthetic_qwen3_vl_install(&dir, VOCAB, LAYERS, "tiny-qwen3vl")
        .expect("qwen3_vl fixture builds");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("manifest.json")).expect("manifest exists"),
    )
    .unwrap();
    assert_eq!(manifest["arch"]["family"], "qwen3_vl");
    assert_eq!(manifest["arch"]["tieWordEmbeddings"], true);
}

/// Overwrites the first two bytes of one resident tensor in a built install,
/// in place. `open()` runs no checksum, so an in-place patch is the
/// sanctioned way to perturb a built install (`docs/NEW_MODEL.md` Phase 1).
fn patch_tensor_head(dir: &std::path::Path, name: &str, bytes: [u8; 2]) {
    let bin = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&bin).expect("resident index loads");
    let entry = index
        .entries
        .get(name)
        .unwrap_or_else(|| panic!("{name} exists in the resident index"));
    let mut weights = std::fs::read(&bin).unwrap();
    let at = entry.file_offset as usize;
    weights[at..at + 2].copy_from_slice(&bytes);
    std::fs::write(&bin, weights).expect("patch lands");
}

#[test]
fn qwen3vl_tied_head_reads_the_embedding_table() {
    // Perturb the embedding table (which IS the head) and the logits must
    // move. On a mis-wired untied head the open path would have refused for
    // a missing `lm_head` tensor, so this guards the subtler failure: the
    // head silently reading a zeroed or stale row.
    let base = temp_dir("tied-base");
    let perturbed = temp_dir("tied-perturbed");
    build_synthetic_qwen3_vl_install(&base, VOCAB, LAYERS, "tiny-qwen3vl")
        .expect("base fixture builds");
    build_synthetic_qwen3_vl_install(&perturbed, VOCAB, LAYERS, "tiny-qwen3vl")
        .expect("perturbed fixture builds");

    patch_tensor_head(
        &perturbed,
        "language_model.model.embed_tokens.weight",
        0x5A5Au16.to_le_bytes(),
    );

    assert_ne!(
        logits_after_prompt(&base, &[5]),
        logits_after_prompt(&perturbed, &[5]),
        "perturbing the tied embedding/head row must move the logits"
    );
}

#[test]
fn qwen3vl_per_head_qk_norms_reach_the_logits() {
    // Two installs identical except for one q_norm row; the norm scales the
    // query BEFORE RoPE and attention, so the logits must differ. A flow
    // that skipped the norm (the `QkNorm::None` arm) would read the weight
    // as dead bytes and the two would come out equal.
    let base = temp_dir("qk-base");
    let perturbed = temp_dir("qk-perturbed");
    build_synthetic_qwen3_vl_install(&base, VOCAB, LAYERS, "tiny-qwen3vl")
        .expect("base fixture builds");
    build_synthetic_qwen3_vl_install(&perturbed, VOCAB, LAYERS, "tiny-qwen3vl")
        .expect("perturbed fixture builds");

    patch_tensor_head(
        &perturbed,
        "language_model.model.layers.0.self_attn.q_norm.weight",
        0x4248u16.to_le_bytes(), // ~3.06, not the fixture's ~1.0
    );

    // THREE tokens, not one: at position 0 the attention is a softmax over a
    // single key, whose weight is 1.0 whatever the query is, so NO q-side
    // patch (q_proj, q_norm, rope) can move a one-token prompt's logits. The
    // final position of a three-token prompt attends over three keys and is
    // where the query first matters.
    assert_ne!(
        logits_after_prompt(&base, &[5, 7, 11]),
        logits_after_prompt(&perturbed, &[5, 7, 11]),
        "the per-head q_norm weight must affect the logits"
    );
}

#[test]
fn qwen3vl_chunked_prefill_matches_sequential_prefill() {
    let dir = temp_dir("prefill");
    build_synthetic_qwen3_vl_install(&dir, VOCAB, LAYERS, "tiny-qwen3vl")
        .expect("qwen3_vl fixture builds");
    let tokens = [5, 7, 11, 13, 17];

    let mut sequential = open(&dir);
    sequential.reset();
    let mut expected = vec![f16::from_f32(0.0); VOCAB as usize];
    for (position, &token) in tokens.iter().enumerate() {
        if position + 1 == tokens.len() {
            sequential
                .produce(token, position, &mut expected)
                .expect("sequential final token produces");
        } else {
            sequential
                .produce_prefill(token, position, &mut expected)
                .expect("sequential prefill token produces");
        }
    }

    let mut chunked = open(&dir);
    chunked.reset();
    let mut actual = vec![f16::from_f32(0.0); VOCAB as usize];
    chunked
        .prefill_chunk(&tokens, 0, &mut actual)
        .expect("chunked prefill succeeds");

    assert_eq!(actual, expected, "chunked prefill must be bit-identical");
}

#[test]
fn the_qwen3vl_tiny_arch_validates_against_the_baseline_extension_fields() {
    // The tiny fixture's behavioural fields must agree with the real
    // baseline's wherever they are the same KIND of answer: the fixture
    // exists to exercise the flow, not to bless different answers.
    let tiny = tiny_qwen3_vl_arch(VOCAB, LAYERS);
    let real = model_io::qwen3_vl_4b();
    assert_eq!(tiny.attention_scale, 0.25); // binary fraction; see the builder
    assert_eq!(tiny.partial_rotary_factor, real.partial_rotary_factor);
    assert_eq!(tiny.rope_neox_subdim, real.rope_neox_subdim);
    assert_eq!(tiny.attn_output_gate, real.attn_output_gate);
    assert_eq!(tiny.tie_word_embeddings, real.tie_word_embeddings);
    assert_eq!(tiny.full_attention_layer_mask, vec![1u8; LAYERS as usize]);
}
