//! `parse_qwen_gdn_dense_config` against the PRODUCTION field values of BOTH
//! published `qwen3_5` checkpoints: `prism-ml/Bonsai-27B-mlx-1bit` (ROADMAP's
//! 1-bit entry), `Qwen/Qwen3.8-27B` (2026-08-14) and
//! `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (its ternary entry).
//!
//! Its own target rather than a case in `qwen36_config.rs`, and the fixture
//! is the real `text_config` verbatim for that file's reason: the whole test
//! reduces to one assertion, that the parse equals
//! `model_io::qwen_gdn_dense_27b()` field for field. A shape-only fixture cannot
//! catch a key mapped to the wrong field, because every dimension would be
//! some other made-up number either way.
//!
//! **THE ASSERTION IS SHARPER HERE THAN FOR ANY OTHER FAMILY, and that is
//! the point of the file.** This config and Qwen 3.6's differ in exactly
//! four fields, all of them about the FFN, so a parser that silently fell
//! through to the MoE branch would produce something that still validates
//! structurally. What catches it is the baseline comparison plus the
//! `num_experts` cases below.
//!
//! **ONE `text_config()` SERVES ALL THREE CHECKPOINTS, WHICH IS THE FINDING
//! AND NOT A SHORTCUT.** Read side by side off the published files, their
//! `text_config`s agree on 33 of 35 keys; the two that differ are
//! `eos_token_id` (248046 for Bonsai and the ternary checkpoint, 248044 for
//! Qwen3.8) and the `quantization` object, and NEITHER reaches an
//! `ArchConfig` field -- the first is the tokenizer's business and the second
//! `parse_gemma4_quantization`'s. The ternary file goes further and shares
//! Bonsai's `eos_token_id` too, so those two differ in the quantization
//! object ALONE. So the fixture carries the shared body once and forks only
//! where it must, and
//! `every_published_checkpoint_parses_to_one_baseline` is what turns that
//! reading into an assertion. If a future point release of any of the three
//! moves a shape key, THAT is the test which reddens, rather than a multi-GB
//! stream failing at some tensor offset.

use turbospark_repack::{
    parse_gemma4_quantization, parse_prism_hadamard, parse_qwen_gdn_dense_config,
    parse_qwen_gdn_moe_config,
};

/// 64 layers, gated-DeltaNet everywhere except every 4th
/// (`full_attention_interval = 4`), which reproduces the checkpoint's own
/// `layer_types` list.
fn layer_types() -> Vec<&'static str> {
    (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}

/// The production `text_config`, trimmed to the keys the parser reads plus
/// the mrope fields it must ignore and the `mtp_*` ones nothing implements.
///
/// `eos` is the one key that differs across the published checkpoints (Bonsai
/// and the ternary file 248046, Qwen3.8 248044). It is a parameter rather
/// than a constant so the difference is stated, and it must reach NO field of
/// the result --
/// which `every_published_checkpoint_parses_to_one_baseline` is what checks.
fn text_config_with_eos(eos: u32) -> serde_json::Value {
    serde_json::json!({
        "eos_token_id": eos,
        "attn_output_gate": true,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 5120,
        // The DENSE FFN width. Qwen 3.6's text_config does not carry this
        // key at all; its FFN width lives in
        // `shared_expert_intermediate_size`.
        "intermediate_size": 17408,
        "layer_types": layer_types(),
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262_144,
        "model_type": "qwen3_5_text",
        // Declared and unimplemented; no `mtp` tensor ships in the file.
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "output_gate_type": "swish",
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        },
        "tie_word_embeddings": false,
        "vocab_size": 248_320
    })
}

/// `prism-ml/Bonsai-27B-mlx-1bit`'s shared `text_config` body.
fn text_config() -> serde_json::Value {
    text_config_with_eos(248_046)
}

/// The Bonsai-27B config: 1-bit at group 128.
fn config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": false,
        "model_type": "qwen3_5",
        "text_config": text_config(),
        "vision_config": {"depth": 27},
        // The real object, verbatim: two keys and no per-tensor overrides.
        "quantization": {"group_size": 128, "bits": 1}
    })
    .to_string()
}

/// The `prism-ml/Ternary-Bonsai-27B-mlx-2bit` config: the SAME architecture
/// again, at 2-bit group 128 (ROADMAP's ternary entry).
///
/// Its `text_config` is Bonsai's to the KEY -- same `eos_token_id` 248046 and
/// all -- so the two differ in the `quantization` object alone. That is a
/// stronger statement than the Qwen3.8 pair makes, and it is why the ternary
/// checkpoint needed no parser, baseline or flow work at all.
fn ternary_config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": false,
        "model_type": "qwen3_5",
        "text_config": text_config(),
        "vision_config": {"depth": 27},
        // The real object, verbatim: two keys and no per-tensor overrides.
        "quantization": {"group_size": 128, "bits": 2}
    })
    .to_string()
}

/// The `mlx-community/Qwen3.8-27B-4bit` config: the SAME architecture at INT4
/// group 64.
///
/// Its `quantization` object is verbatim, three keys and no per-tensor
/// overrides, and the `mode` key is the one Bonsai's does not carry --
/// `parse_gemma4_quantization` skips it by name, so it must not become a
/// per-tensor override entry.
fn qwen38_config_json() -> String {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": false,
        "model_type": "qwen3_5",
        "text_config": text_config_with_eos(248_044),
        "vision_config": {"depth": 27},
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
    })
    .to_string()
}

/// The `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` config (the HADAMARD line):
/// the SAME architecture a fourth time, in a rotated weight basis.
///
/// Three real differences from the ternary file above, and none reaches an
/// `ArchConfig` field. The root `model_type` renames to
/// `prism_hadamard_qwen35` -- unknown to `SUPPORTED_HF`, so resolution falls
/// through to `text_config.model_type`, which is still `qwen3_5_text`; that
/// fallthrough is exactly what `config_json_resolution`'s two-step probe is
/// for. The `quantization` object MOVES from `text_config` to the root and
/// gains `mode: "affine"`. And `eos_token_id` joins Qwen3.8's 248044 while
/// `mtp_num_hidden_layers` drops to 0 -- both inert, which the baseline
/// assertion below proves rather than assumes.
fn bonsai2_config_json() -> String {
    serde_json::json!({
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "text_config": text_config_with_eos(248_044),
        "vision_config": {"depth": 27},
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "hadamard_config": "hadamard.json",
        "requires_runtime": "runtime/artifact.py",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "base_model_type": "qwen3_5",
        "components": {"text": true, "vision": true, "mtp": false}
    })
    .to_string()
}

#[test]
fn parses_the_production_config_into_the_pinned_baseline() {
    let arch = parse_qwen_gdn_dense_config(&config_json()).expect("config parses");
    assert_eq!(
        arch,
        model_io::qwen_gdn_dense_27b(),
        "parsed config does not match the pinned qwen3_5 baseline"
    );
}

/// **ALL THREE published `qwen3_5` checkpoints parse to ONE `ArchConfig`, and
/// that is what makes each of the later two a checkpoint rather than a new
/// family.**
///
/// The equality is with `qwen_gdn_dense_27b()` on every side AND between them,
/// so it fails whichever way a future divergence arrives: a point release that
/// moves a shape key, or an edit to the baseline that suits one checkpoint and
/// not the others.
///
/// It also pins the two differences as INERT. `eos_token_id` and the
/// quantization block are the only keys the two files disagree on, and
/// neither is an `ArchConfig` field -- so a parser that started reading
/// either into the arch would be caught here rather than by an install that
/// validates and then decodes wrongly.
#[test]
fn every_published_checkpoint_parses_to_one_baseline() {
    let bonsai = parse_qwen_gdn_dense_config(&config_json()).expect("bonsai config parses");
    let qwen38 = parse_qwen_gdn_dense_config(&qwen38_config_json()).expect("qwen3.8 config parses");
    let ternary = parse_qwen_gdn_dense_config(&ternary_config_json()).expect("ternary parses");
    let bonsai2 = parse_qwen_gdn_dense_config(&bonsai2_config_json()).expect("bonsai2 parses");

    assert_eq!(bonsai, model_io::qwen_gdn_dense_27b());
    assert_eq!(qwen38, model_io::qwen_gdn_dense_27b());
    assert_eq!(ternary, model_io::qwen_gdn_dense_27b());
    assert_eq!(
        bonsai, qwen38,
        "the published qwen3_5 checkpoints must share one architecture"
    );
    assert_eq!(bonsai, ternary, "ternary is Bonsai at a different width");
    assert_eq!(
        bonsai, bonsai2,
        "bonsai2 is the same architecture in a rotated basis; no shape key moved"
    );
}

/// The FOURTH checkpoint's root `model_type` is unknown to `SUPPORTED_HF`
/// (`prism_hadamard_qwen35`), so family resolution rides on the two-step
/// probe falling through to `text_config.model_type`. Mutating the INNER
/// string away must leave the file resolving to NOTHING -- which is what
/// proves the fallthrough is load-bearing for this family, and that a future
/// publisher renaming the inner string reddens here rather than in a
/// multi-GB stream's open error.
#[test]
fn bonsai2_resolves_only_through_the_text_config() {
    let root: serde_json::Value =
        serde_json::from_str(&bonsai2_config_json()).expect("fixture parses");
    assert_eq!(
        turbospark_repack::config_json_family(&root),
        Some(model_io::ModelFamily::QwenGdnDense),
        "the hadamard root string must resolve through text_config"
    );

    let mut mutated = root.clone();
    mutated["text_config"]["model_type"] = serde_json::json!("qwen3_5_text_renamed");
    assert_eq!(
        turbospark_repack::config_json_family(&mutated),
        None,
        "with the inner string renamed nothing vouches for this file any more"
    );
}

/// The quantization SPEC is where the checkpoints really differ, and each is a
/// shape this port has kernels for.
///
/// Bonsai is `(1, 128)` with FP16 companions, the ternary file `(2, 128)` with
/// FP16 ones, and Qwen3.8's mlx-community artifact `(4, 64)` with BF16 --
/// the three arms of `is_supported_affine_shape`'s conjunction. Asserted here because it is the
/// one axis the shared baseline above deliberately says nothing about, and
/// because a `mode` key mis-read as a per-tensor override would surface as a
/// bogus bits entry rather than as an error.
#[test]
fn the_three_checkpoints_declare_three_supported_affine_shapes() {
    let bonsai = parse_gemma4_quantization(&config_json()).expect("bonsai quantization parses");
    assert_eq!((bonsai.default_bits, bonsai.group_size), (1, 128));

    let qwen38 =
        parse_gemma4_quantization(&qwen38_config_json()).expect("qwen3.8 quantization parses");
    assert_eq!((qwen38.default_bits, qwen38.group_size), (4, 64));

    // The third: ROADMAP's ternary entry. It shares Bonsai's group size and
    // companion dtype and differs in the width alone, which is exactly the
    // pair `is_supported_affine_shape` had to grow.
    let ternary =
        parse_gemma4_quantization(&ternary_config_json()).expect("ternary quantization parses");
    assert_eq!((ternary.default_bits, ternary.group_size), (2, 128));
}

/// The dense FFN width comes from `intermediate_size`, and the three MoE
/// fields are zero rather than inherited.
///
/// Stated separately from the baseline comparison because it is the one
/// place the two Qwen parsers part company, and because a `0` that came from
/// a missing key would look identical to a `0` that was decided.
#[test]
fn the_dense_ffn_width_is_read_and_the_moe_fields_are_zero() {
    let arch = parse_qwen_gdn_dense_config(&config_json()).expect("config parses");
    assert_eq!(arch.intermediate_size, 17408);
    assert_eq!(arch.moe_intermediate_size, 0);
    assert_eq!(arch.num_experts, 0);
    assert_eq!(arch.top_k_experts, 0);
    assert!(!arch.shared_expert_gated);
}

/// A config that declares BOTH a dense FFN and experts is refused rather
/// than half-read.
///
/// Half-reading it yields an `ArchConfig` whose FFN width and expert count
/// disagree, which validates structurally and then dispatches the wrong
/// branch -- the failure mode this whole family split exists to avoid, since
/// `qwen3_5` and `qwen3_5_moe` are one suffix apart.
#[test]
fn a_dense_config_that_also_declares_experts_is_refused() {
    let mut root: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    root["text_config"]["num_experts"] = serde_json::json!(256);
    let err = parse_qwen_gdn_dense_config(&root.to_string()).expect_err("must be refused");
    let text = format!("{err:?}");
    assert!(text.contains("num_experts"), "{text}");
    assert!(text.contains("qwen3_5_moe"), "{text}");

    // `num_experts: 0` is not a contradiction and still parses: it is what
    // a dense config would say if it said anything.
    root["text_config"]["num_experts"] = serde_json::json!(0);
    assert!(parse_qwen_gdn_dense_config(&root.to_string()).is_ok());
}

/// The two parsers refuse each other's configs, which is what
/// `refuse_foreign_config` is for.
///
/// Without it every parser hardcodes `family:` on the way out, so a Bonsai
/// config parsed by the MoE parser would yield an `ArchConfig` labelled
/// `qwen36` -- with a 5120 hidden size it never had and a shared-expert
/// width read from a key that is not there.
#[test]
fn the_two_qwen_parsers_refuse_each_others_configs() {
    assert!(parse_qwen_gdn_moe_config(&config_json()).is_err());

    let moe = serde_json::json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": text_config(),
    })
    .to_string();
    assert!(parse_qwen_gdn_dense_config(&moe).is_err());
}

/// The layer graph: 64 entries, 48 linear to 16 full, layer 0 linear.
#[test]
fn the_layer_mask_is_48_linear_to_16_full() {
    let arch = parse_qwen_gdn_dense_config(&config_json()).expect("config parses");
    assert_eq!(arch.full_attention_layer_mask.len(), 64);
    assert_eq!(
        arch.full_attention_layer_mask
            .iter()
            .filter(|&&k| k == 2)
            .count(),
        48
    );
    assert!(arch.layer_is_linear(0));
    assert!(arch.layer_is_full(3));
}

/// `attention_scale` is `head_dim ** -0.5` and, at 256, exactly 1/16 -- a
/// binary fraction, so it survives the serde_json round trip AGENTS.md
/// Gotcha 24 warns about.
#[test]
fn attention_scale_is_the_reference_head_dim_power() {
    let arch = parse_qwen_gdn_dense_config(&config_json()).expect("config parses");
    assert_eq!(arch.attention_scale, 0.0625);
    let back: f64 = serde_json::from_str(&serde_json::to_string(&arch.attention_scale).unwrap())
        .expect("round trips");
    assert_eq!(back, arch.attention_scale);
}

/// The Hadamard contract parses out of the Bonsai-2 config and translates
/// module paths to ENGINE names -- `lm_head` to
/// `language_model.lm_head.weight`, `model.embed_tokens` (the one
/// `embedding: true` module) into the INVERSE list, everything else folded.
/// The module list here is the real one's SHAPE trimmed to one of each kind.
#[test]
fn the_hadamard_contract_parses_to_engine_names() {
    let config = serde_json::json!({
        "model_type": "prism_hadamard_qwen35",
        "text_config": text_config(),
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "hadamard_config": "hadamard.json",
        "modules": [
            {"path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"},
            {"path": "model.embed_tokens", "block": 1024, "embedding": true, "dtype": "float16"},
            {"path": "model.layers.0.mlp.gate_proj", "block": 1024, "embedding": false, "dtype": "float16"}
        ]
    })
    .to_string();
    let contract = parse_prism_hadamard(&config)
        .expect("contract parses")
        .expect("present");
    assert_eq!(contract.block, 1024);
    assert_eq!(
        contract.folded,
        vec![
            "language_model.lm_head.weight".to_string(),
            "language_model.model.layers.0.mlp.gate_proj.weight".to_string(),
        ]
    );
    assert_eq!(
        contract.inverse,
        vec!["language_model.model.embed_tokens.weight".to_string()]
    );
}

/// A config with no packed modules is not a folded checkpoint, and the
/// parser answers `None` rather than an empty contract.
#[test]
fn a_config_without_modules_is_not_folded() {
    assert_eq!(parse_prism_hadamard(&config_json()).expect("parses"), None);
}

/// The all-or-nothing rule: one packed module at block 0 amid transformed
/// ones is refused, because a module reading the RAW activation while its
/// siblings read the rotated one is exactly the shape the shared-transform
/// wiring cannot express.
#[test]
fn a_block_zero_module_amid_folded_ones_is_refused() {
    let config = serde_json::json!({
        "model_type": "prism_hadamard_qwen35",
        "modules": [
            {"path": "lm_head", "block": 1024, "embedding": false},
            {"path": "model.layers.0.mlp.down_proj", "block": 0, "embedding": false}
        ]
    })
    .to_string();
    // The refusal must come from the ALL-OR-NOTHING rule itself, not from the
    // width-set check downstream (which also rejects block 0, and would have
    // made this test a survivor when the arm above was mutated off). The
    // message names the contract; assert it, so this case pins WHICH guard
    // fired.
    let err = parse_prism_hadamard(&config).expect_err("must be refused");
    let text = format!("{err:?}");
    assert!(text.contains("all-or-nothing"), "{text}");
}

/// One block per checkpoint: a module declaring a different width against
/// the rest is refused, as is a width outside the bundled runtime's set.
#[test]
fn mismatched_and_unknown_block_widths_are_refused() {
    let two_blocks = serde_json::json!({
        "model_type": "prism_hadamard_qwen35",
        "modules": [
            {"path": "lm_head", "block": 1024, "embedding": false},
            {"path": "model.layers.0.mlp.gate_proj", "block": 512, "embedding": false}
        ]
    })
    .to_string();
    assert!(parse_prism_hadamard(&two_blocks).is_err());

    let odd_block = serde_json::json!({
        "model_type": "prism_hadamard_qwen35",
        "modules": [
            {"path": "lm_head", "block": 1000, "embedding": false},
        ]
    })
    .to_string();
    assert!(parse_prism_hadamard(&odd_block).is_err());
}
