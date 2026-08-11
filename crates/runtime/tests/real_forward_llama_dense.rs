#![cfg(target_os = "macos")]
//! ROADMAP M4 step 1, the DISCRIMINATING TEST: does the DENSE half of the
//! `llama` architecture need its own `ModelFamily` variant?
//!
//! The question is not cosmetic. `known_architecture(Llama)` returns
//! `mixtral_8x7b()` and `arch_validation` compares a manifest against that
//! baseline field by field, so a dense install -- `numExperts` 0 against
//! Mixtral's 8 -- looks like it must be rejected before any decode code sees
//! it. If so, the dense half needs a fifth variant, a fifth baseline, and a
//! fifth row everywhere `ModelFamily` is matched exhaustively.
//!
//! It does not, and this test is the evidence. `open_model_runner` goes
//! through `repack::peek_manifest_arch`, which OVERWRITES every shape field
//! from the manifest before validation runs, so the shapes are compared
//! against themselves and only the family-EXTENSION fields really bind to the
//! baseline -- and a dense Llama's extensions are Mixtral's exactly.
//!
//! So a dense install reaches the decode flow, and M4's remaining work was
//! the FFN itself. It now runs, and the cases below drive it.

use half::f16;
use turbospark_repack::build_synthetic_dense_llama_install;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

const VOCAB: i64 = 128;
const LAYERS: i64 = 4;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-llama-dense-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The whole point: a dense `llama` install BUILDS, and it builds with zero
/// packed-expert layer files.
///
/// The install format was the thing M4's scoping notes expected to block
/// here, on the reading that `manifest.rs` requires a
/// `packed_experts/layer_NN.bin` per layer. It requires one per PACKED-EXPERT
/// layer, and a dense install declares zero of those while still declaring
/// its four real `arch.numLayers` -- two different `numLayers` fields, which
/// is what makes the empty case already legal.
#[test]
fn a_dense_llama_install_writes_with_no_packed_expert_files() {
    let dir = temp_dir("writes");
    let arch = build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("dense llama install builds");

    assert_eq!(arch.num_experts, 0, "dense: no routed experts");
    assert_eq!(arch.top_k_experts, 0, "dense: nothing to route to");

    let packed = dir.join("packed_experts");
    assert!(
        packed.join("layout.json").exists(),
        "layout.json is a REQUIRED_FILE even when it describes nothing"
    );
    let layer_files: Vec<_> = std::fs::read_dir(&packed)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("layer_"))
        .collect();
    assert!(
        layer_files.is_empty(),
        "a dense install has no expert blobs to pack, got {layer_files:?}"
    );
}

/// The discriminating half, and now also the end-to-end one: a dense install
/// must survive validation against the MIXTRAL baseline and decode.
///
/// A failure here that mentions the manifest, the baseline, or `numExperts`
/// is the signal that M4 needs a fifth `ModelFamily` after all.
#[test]
fn a_dense_llama_install_validates_against_the_mixtral_baseline_and_decodes() {
    let dir = temp_dir("decodes");
    let arch = build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("dense llama install builds");

    // The real open path resolves the arch from the manifest first, exactly
    // as `crates/cli` does. Doing it this way rather than passing the
    // builder's `arch` straight through is the part that actually exercises
    // validation against `mixtral_8x7b()`.
    let peeked = turbospark_repack::peek_manifest_arch(&dir)
        .expect("a dense manifest peeks against the Mixtral baseline");
    assert_eq!(peeked.num_experts, 0);
    assert_eq!(peeked.family, arch.family);

    let mut runner = RealForwardRunner::open(&dir, peeked).expect("a dense llama install opens");
    assert_eq!(runner.vocab_size(), VOCAB as usize);

    // Weights are deterministic but untrained, so nothing here asserts on
    // the TOKENS (AGENTS.md Gotcha 12): only the logits contract.
    runner.reset();
    let mut token = 5i32;
    for position in 0..6usize {
        let mut head = vec![f16::from_f32(0.0); VOCAB as usize];
        runner
            .produce(token, position, &mut head)
            .expect("dense produce succeeds");
        assert!(
            head.iter().all(|v| v.to_f32().is_finite()),
            "non-finite logit at position {position}"
        );
        // The producer contract: LOGITS, never probabilities (crate Gotcha
        // 1). This architecture has no softcap, so a distribution is the
        // only shape to rule out.
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

/// The FFN is really running, rather than the layer being a pass-through.
///
/// This is the mutation check the decode case cannot make on its own: a
/// dense branch that encoded NOTHING would leave every assertion above
/// green, because attention alone still produces finite, non-normalized
/// logits and the residual stream still reaches the head. So perturb exactly
/// the three FFN tensors -- located through the resident index, so nothing
/// else in the file moves -- and require the logits to follow.
#[test]
fn the_dense_ffn_weights_reach_the_logits() {
    let a = temp_dir("ffn-a");
    let b = temp_dir("ffn-b");
    build_synthetic_dense_llama_install(&a, VOCAB, LAYERS, "tiny-mistral").expect("builds");
    build_synthetic_dense_llama_install(&b, VOCAB, LAYERS, "tiny-mistral").expect("builds");

    let baseline = first_logits(&a);
    // Same builder, same seeds: the two installs are identical until this.
    assert_eq!(baseline, first_logits(&b), "the pair starts identical");

    let patched = patch_dense_ffn_only(&b);
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

/// Flips one bit in the packed data of every `mlp.{gate,up,down}_proj.weight`
/// and NOTHING else, in place. Returns how many tensors it touched, because
/// a patch that silently found none would make the assertion above vacuous.
fn patch_dense_ffn_only(dir: &std::path::Path) -> usize {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("index loads");
    let mut bytes = std::fs::read(&path).unwrap();
    let mut touched = 0;
    for entry in index.entries.values() {
        if ![
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        .iter()
        .any(|s| entry.name.ends_with(s))
        {
            continue;
        }
        let start = entry.file_offset as usize;
        let end = start + entry.size_bytes as usize;
        for b in &mut bytes[start..end] {
            *b ^= 0x11;
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

/// The MUTATION CHECK for the test above, and the statement of what actually
/// binds a dense install to the Mixtral baseline.
///
/// The test above passing could mean either "validation accepted a dense
/// install on its merits" or "validation never ran". This case settles it by
/// corrupting a field `peek_manifest_arch` does NOT overwrite from the
/// manifest -- a family-EXTENSION field, which is the only class that really
/// compares against `mixtral_8x7b()` -- and asserting the open dies there
/// instead, upstream of the dense refusal.
///
/// So `numExperts` is free because both sides of its comparison come from the
/// same file, and `ffnSandwichNorms` is not. That asymmetry is the whole
/// reason the dense half needs no fifth `ModelFamily`.
#[test]
fn a_corrupted_family_extension_field_still_fails_validation_first() {
    let dir = temp_dir("mutation");
    build_synthetic_dense_llama_install(&dir, VOCAB, LAYERS, "tiny-mistral")
        .expect("dense llama install builds");

    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    // Mixtral has none; claiming them must be rejected.
    manifest["arch"]["ffnSandwichNorms"] = serde_json::Value::Bool(true);
    std::fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("still peeks");
    let text = RealForwardRunner::open(&dir, peeked)
        .err()
        .expect("a mismatched extension field must be refused")
        .to_string();
    assert!(
        !text.contains("DENSE llama install"),
        "validation must reject this BEFORE the decode flow sees it, got: {text}"
    );
}
