//! Tests for manifest.json decode + validation against a resolved
//! `ArchConfig`, using a synthetic (non-Gemma-sized) toy architecture so no
//! `quant` block is required.

use std::io::Write;

use turbospark_model_io::{
    load_manifest, peek_family, LinearAttentionConfig, ModelError, ModelFamily,
};

fn toy_arch() -> turbospark_model_io::ArchConfig {
    turbospark_model_io::ArchConfig {
        hidden_size: 64,
        intermediate_size: 128,
        moe_intermediate_size: 64,
        num_heads: 2,
        num_kv_heads: 1,
        num_full_kv_heads: 1,
        head_dim: 32,
        full_head_dim: 32,
        vocab_size: 100,
        sliding_window: 16,
        final_logit_softcap: 0.0,
        rope_theta: 10000.0,
        full_rope_theta: 10000.0,
        partial_rotary_factor: 1.0,
        num_layers: 2,
        num_experts: 4,
        top_k_experts: 2,
        tie_word_embeddings: true,
        attention_k_eq_v: true,
        full_attention_layer_mask: vec![1, 1],
        hidden_activation: "gelu_pytorch_tanh".to_string(),
        family: ModelFamily::Gemma4,
        attn_output_gate: false,
        attention_scale: 1.0,
        embedding_scaled_by_sqrt_hidden: true,
        router_scaled: true,
        ffn_sandwich_norms: true,
        shared_expert_gated: false,
        rope_neox_subdim: false,
        linear_attention: LinearAttentionConfig::NONE,
        compressed_attention: turbospark_model_io::CompressedAttentionConfig::NONE,
        hyper_connections: turbospark_model_io::HyperConnectionConfig::NONE,
        num_hash_routed_layers: 0,
        router_scoring_func: "softmax".to_string(),
        routed_scaling_factor: 1.0,
        swiglu_limit: 0.0,
        rope_scaling: turbospark_model_io::RopeScalingConfig::NONE,
        vision: turbospark_model_io::VisionConfig::NONE,
        ple: turbospark_model_io::PleConfig::NONE,
    }
}

fn write_manifest(dir: &std::path::Path, json: &str) {
    let mut f = std::fs::File::create(dir.join("manifest.json")).unwrap();
    f.write_all(json.as_bytes()).unwrap();
}

fn toy_manifest_json() -> String {
    r#"{
        "magic": "GTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": {},
        "modelID": "toy",
        "sourceSnapshotHash": null,
        "arch": {
            "hiddenSize": 64,
            "ffnIntermediate": 128,
            "moeIntermediateSize": 64,
            "numHeads": 2,
            "numKVHeads": 1,
            "numFullKVHeads": 1,
            "headDim": 32,
            "fullHeadDim": 32,
            "vocabSize": 100,
            "slidingWindow": 16,
            "finalLogitSoftcap": 0.0,
            "ropeTheta": 10000.0,
            "fullRopeTheta": 10000.0,
            "partialRotaryFactor": 1.0,
            "numLayers": 2,
            "numExperts": 4,
            "topKExperts": 2,
            "tieWordEmbeddings": true,
            "attentionKEqV": true,
            "hiddenActivation": "gelu_pytorch_tanh",
            "fullAttentionLayerMask": [1, 1]
        },
        "quant": null,
        "files": {
            "model_weights.bin": {"size": 1, "sha256": "a"},
            "packed_experts/layout.json": {"size": 1, "sha256": "b"},
            "packed_experts/layer_00.bin": {"size": 1, "sha256": "c"},
            "packed_experts/layer_01.bin": {"size": 1, "sha256": "d"}
        },
        "expertsPerLayer": 4,
        "numLayers": 2,
        "expertStride": 4096
    }"#
    .to_string()
}

#[test]
fn load_succeeds_for_a_matching_toy_manifest() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &toy_manifest_json());
    let manifest = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap();
    assert_eq!(manifest.magic, "GTURBO");
    assert_eq!(manifest.num_layers, 2);
}

#[test]
fn text_loader_rejects_an_image_capability_with_a_clear_error() {
    let dir = tempfile_dir();
    write_manifest(
        dir.path(),
        r#"{"magic":"GTURBO","version":1,"capability":"image-generation"}"#,
    );
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    assert_eq!(
        err,
        ModelError::UnsupportedCapability {
            capability: "image-generation".to_string()
        }
    );
    assert!(err.to_string().contains("text-model loader"));
}

#[test]
fn load_rejects_hidden_size_mismatch() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &toy_manifest_json());
    let mut arch = toy_arch();
    arch.hidden_size = 999;
    let err = load_manifest(dir.path(), &arch, 4 * 1024 * 1024).unwrap_err();
    match err {
        ModelError::ArchMismatch { field, .. } => assert_eq!(field, "hiddenSize"),
        other => panic!("expected ArchMismatch, got {other:?}"),
    }
}

/// `as u8` used to narrow silently: 257 would read as 1 and pass the mask
/// check by coincidence. A value outside `0..=255` must be refused by name
/// instead.
#[test]
fn load_rejects_a_mask_entry_that_overflows_a_byte() {
    let dir = tempfile_dir();
    let json = toy_manifest_json().replace(
        "\"fullAttentionLayerMask\": [1, 1]",
        "\"fullAttentionLayerMask\": [257, 1]",
    );
    write_manifest(dir.path(), &json);
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    match err {
        ModelError::ArchMismatch { field, .. } => assert_eq!(field, "fullAttentionLayerMask"),
        other => panic!("expected ArchMismatch, got {other:?}"),
    }
}

#[test]
fn load_rejects_missing_manifest_file() {
    let dir = tempfile_dir();
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::PartialInstall { .. }));
}

#[test]
fn load_rejects_unknown_flag() {
    let dir = tempfile_dir();
    let json = toy_manifest_json().replacen("\"flags\": {}", "\"flags\": {\"bogus\": true}", 1);
    write_manifest(dir.path(), &json);
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::UnknownFlag { .. }));
}

#[test]
fn load_rejects_missing_layer_file_entry() {
    let dir = tempfile_dir();
    let json = toy_manifest_json().replace(
        "\"packed_experts/layer_01.bin\": {\"size\": 1, \"sha256\": \"d\"}",
        "\"unused\": {\"size\": 1, \"sha256\": \"d\"}",
    );
    write_manifest(dir.path(), &json);
    let err = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err();
    assert!(matches!(err, ModelError::MissingFile { .. }));
}

#[test]
fn peek_family_defaults_to_gemma_when_family_field_is_absent() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &toy_manifest_json());
    let family = peek_family(dir.path(), 4 * 1024 * 1024).unwrap();
    assert_eq!(family, ModelFamily::Gemma4);
}

// ---------------------------------------------------------------------------
// `validate_quant`
//
// The toy manifest above carries `"quant": null`, which is why none of the
// cases before this point reach the gate at all. These build a quant block
// explicitly. They exist because the gate had no direct test: it was only
// ever exercised through `crates/repack`'s manifests, where a wrong
// acceptance reads as a repack bug three crates away.
// ---------------------------------------------------------------------------

/// The INT4 install's per-slot bit widths. The router is 8 and always has
/// been; a uniform 4 across all five is not a shape any install has, which
/// the first draft of these tests got wrong and the gate caught.
const INT4_BITS: [i64; 5] = [4, 4, 8, 4, 4];
/// The 1-bit install's. Uniform, because the checkpoint quantizes everything
/// it quantizes at one bit and the three slots it has no component for fall
/// back to the same value.
const INT1_BITS: [i64; 5] = [1, 1, 1, 1, 1];
/// The ternary install's, uniform for the 1-bit install's reason. Note the
/// value 2 also appears in the INT4 arm's `routedExpert` bit list at BF16 and
/// group 64 (DeepSeek-V4-Flash's dynamic quant), which is a DIFFERENT shape:
/// the cases below vary the companions and group away from this one to keep
/// the two from being read as one.
const INT2_BITS: [i64; 5] = [2, 2, 2, 2, 2];

/// A quant block with per-slot bit widths, one companion dtype and one group
/// size, so a case can vary exactly one axis away from a valid shape.
fn quant_block(bits: [i64; 5], companions: &str, group: i64) -> String {
    let names = [
        "embedding",
        "attention",
        "router",
        "sharedExpert",
        "routedExpert",
    ];
    let slots: Vec<String> = names
        .iter()
        .zip(bits.iter())
        .map(|(name, b)| {
            format!(
                r#""{name}": {{"weightBits": {b}, "scheme": "affine",
                    "scaleType": "{companions}", "biasType": "{companions}",
                    "groupSize": {group}}}"#
            )
        })
        .collect();
    format!(r#""quant": {{{}}}"#, slots.join(", "))
}

fn manifest_with_quant(bits: [i64; 5], companions: &str, group: i64) -> String {
    toy_manifest_json().replace("\"quant\": null", &quant_block(bits, companions, group))
}

fn quant_error(bits: [i64; 5], companions: &str, group: i64) -> String {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &manifest_with_quant(bits, companions, group));
    match load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap_err() {
        ModelError::IndexCorrupt { detail } => detail,
        other => panic!("expected IndexCorrupt, got {other:?}"),
    }
}

/// The shape the one published 1-bit checkpoint declares: FP16 companions at
/// group 128, on every slot.
///
/// Every slot, including `routedExpert`, which has no 1-bit kernel. That is
/// deliberate and `validate_quant`'s doc says why: the checkpoint is dense,
/// so three of the five slots describe components it does not have and fall
/// back to the type the rest of the model uses.
#[test]
fn a_one_bit_affine_quant_block_at_group_128_is_accepted() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &manifest_with_quant(INT1_BITS, "fp16", 128));
    let manifest = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap();
    let quant = manifest.quant.expect("the quant block decoded");
    assert_eq!(quant.embedding.weight_bits, 1);
    assert_eq!(quant.embedding.group_size, 128);
    assert_eq!(quant.routed_expert.scale_type, "fp16");
}

/// FP16 companions on a 1-bit slot are required, and BF16 ones are refused.
///
/// This is the axis that cannot fail any other way: the two planes are the
/// same width, so a wrong reading passes every length and offset check in
/// the install and decodes 0.0271 as 1.7e-16. The message is asserted to
/// name the dtype, because a bare "unsupported quantization" would send the
/// reader looking at the bit width.
#[test]
fn a_one_bit_slot_with_bf16_companions_is_refused_and_the_dtype_is_named() {
    let detail = quant_error(INT1_BITS, "bf16", 128);
    assert!(detail.contains("bf16"), "{detail}");
    assert!(detail.contains("fp16"), "{detail}");
}

/// The shape the published ternary checkpoint declares: FP16 companions at
/// group 128 on every slot, at TWO bits.
///
/// Every slot again, for the 1-bit case's reason: that checkpoint is dense
/// too, so three of the five slots are defaulted statements.
#[test]
fn a_two_bit_affine_quant_block_at_group_128_is_accepted() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &manifest_with_quant(INT2_BITS, "fp16", 128));
    let manifest = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap();
    let quant = manifest.quant.expect("the quant block decoded");
    assert_eq!(quant.embedding.weight_bits, 2);
    assert_eq!(quant.embedding.group_size, 128);
    assert_eq!(quant.routed_expert.scale_type, "fp16");
}

/// The 2-bit shape is its own conjunction and does NOT inherit the affine
/// arm's `routedExpert` 2-bit permission.
///
/// That arm allows `weightBits: 2` at BF16 and group 64 on the routed slot
/// alone, for the DeepSeek-V4-Flash dynamic-quant checkpoint. This install
/// declares 2 bits at FP16 and group 128 on EVERY slot, so the two overlap
/// only in the width -- and a gate that collapsed them into one bit list
/// would accept the four cases below.
#[test]
fn the_cross_products_of_the_two_bit_shape_are_refused() {
    let cases = [
        (INT2_BITS, "bf16", 128, "2-bit with the INT4 companions"),
        (INT2_BITS, "fp16", 64, "2-bit at the INT4 group size"),
        (INT2_BITS, "bf16", 64, "the DSV4 routed shape on every slot"),
        (INT4_BITS, "fp16", 128, "INT4 bits under the 2-bit shape"),
    ];
    for (bits, companions, group, what) in cases {
        let detail = quant_error(bits, companions, group);
        assert!(
            detail.contains("unsupported quantization"),
            "{what} was accepted: {detail}"
        );
    }
}

/// The bit width, the group size and the companion dtype are ONE shape, not
/// three independent axes.
///
/// Every case here starts from an ACCEPTED shape and moves exactly one axis,
/// so what it proves is that the axis is load-bearing rather than that some
/// field somewhere was wrong. A gate written as three independent widenings
/// (`1` added to the bit lists, `128` to the group sizes, `fp16` to the
/// companion types) accepts all four of these, and no kernel implements any
/// of them.
#[test]
fn the_cross_products_of_the_two_affine_shapes_are_refused() {
    let cases = [
        (INT4_BITS, "fp16", 64, "INT4 with the 1-bit companions"),
        (INT4_BITS, "bf16", 128, "INT4 at the 1-bit group size"),
        (INT1_BITS, "bf16", 128, "1-bit with the INT4 companions"),
        (INT1_BITS, "fp16", 64, "1-bit at the INT4 group size"),
        // The one case that moves TWO axes, and it has to be here: it is the
        // only shape the bit-width conjunct alone refuses. Without it,
        // deleting `weight_bits == 1` from the 1-bit predicate leaves this
        // whole file green -- checked, not assumed.
        (
            INT4_BITS,
            "fp16",
            128,
            "INT4 bits under the 1-bit companion shape",
        ),
    ];
    for (bits, companions, group, what) in cases {
        let detail = quant_error(bits, companions, group);
        assert!(
            detail.contains("unsupported quantization"),
            "{what} was accepted: {detail}"
        );
    }
}

/// The INT4 shape is unmoved. A regression guard, since the 1-bit arm was
/// added beside it rather than by widening it.
#[test]
fn the_four_bit_affine_shape_still_loads() {
    let dir = tempfile_dir();
    write_manifest(dir.path(), &manifest_with_quant(INT4_BITS, "bf16", 64));
    let manifest = load_manifest(dir.path(), &toy_arch(), 4 * 1024 * 1024).unwrap();
    let quant = manifest.quant.unwrap();
    assert_eq!(quant.attention.weight_bits, 4);
    assert_eq!(quant.router.weight_bits, 8);
}

fn tempfile_dir() -> TempDir {
    TempDir::new()
}

/// Minimal owned-directory helper so this test crate doesn't need an extra
/// `tempfile` dependency for a handful of throwaway directories.
struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let unique_counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        let unique = format!(
            "turbospark-model-io-test-{}-{}-{unique_counter}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
