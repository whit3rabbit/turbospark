//! The probe's gates, with no network.
//!
//! **The refusal paths are the point.** A live probe of a curated row takes
//! the accepting path every time, so the branches that decide a model is
//! unusable -- and the wording that tells a reader why -- would otherwise
//! only ever be exercised by pointing the command at something broken and
//! reading the output by eye.
//!
//! `evaluate_gguf` and `evaluate_config` exist as separate entry points from
//! the fetching wrappers for exactly this.

use turbospark_catalog::{evaluate_config, evaluate_gguf, RepoRef, Verdict};

use repack::{
    build_synthetic_gemma4_gguf, parse_gguf_header, GgufBuilder, SyntheticGgufShape,
    GGUF_DEFAULT_MAX_HEADER_BYTES,
};

fn repo() -> RepoRef {
    RepoRef::new("owner/name", "main")
}

fn header_of(bytes: &[u8]) -> repack::GgufHeader {
    parse_gguf_header(bytes, GGUF_DEFAULT_MAX_HEADER_BYTES).expect("fixture parses")
}

// -- GGUF ------------------------------------------------------------------

#[test]
fn a_q8_0_gemma_fixture_is_runnable_and_reports_its_shape() {
    let shape = SyntheticGgufShape::default();
    let (bytes, _) = build_synthetic_gemma4_gguf(shape);
    let report = evaluate_gguf(&header_of(&bytes), &repo(), "f.gguf", Some(4242));

    assert_eq!(report.verdict, Verdict::Runnable, "{:?}", report.verdict);
    assert_eq!(report.architecture.as_deref(), Some("gemma4"));
    assert_eq!(report.family, Some(model_io::ModelFamily::Gemma4));
    assert_eq!(report.download_bytes, Some(4242));
    let arch = report.arch.expect("an ArchConfig is derived");
    assert_eq!(arch.num_layers, shape.num_layers as i64);
    assert_eq!(arch.num_experts, shape.num_experts as i64);
    assert!(
        report
            .types
            .iter()
            .any(|t| t.name == "Q8_0" && t.executable),
        "Q8_0 should be reported as executable: {:?}",
        report.types
    );
}

/// **F32 must not read as blocked, and this is not hypothetical**: checking
/// the transcoded types against `EXECUTABLE_GGUF_TYPES` is exactly what
/// `scopes_the_dense_llama_candidates`' first run did, and it marked every
/// real candidate unusable (`crates/repack/CLAUDE.md` Gotcha 5).
#[test]
fn the_transcoded_types_do_not_block_an_install_and_are_labelled_as_such() {
    let (bytes, _) = build_synthetic_gemma4_gguf(SyntheticGgufShape::default());
    let report = evaluate_gguf(&header_of(&bytes), &repo(), "f.gguf", None);
    let f32 = report
        .types
        .iter()
        .find(|t| t.name == "F32")
        .expect("the fixture carries F32 norms");
    assert!(f32.executable, "F32 must not block an install");
    assert!(
        f32.transcoded,
        "F32 is executable because it is TRANSCODED, not because it has a kernel; \
         printing 'has kernels' beside it states something false about this port"
    );
    assert_eq!(report.verdict, Verdict::Runnable);
}

/// A mixed K-quant file is the normal case, not the exotic one.
#[test]
fn a_mixed_k_quant_fixture_reports_every_type_it_carries() {
    let (bytes, _) = build_synthetic_gemma4_gguf(SyntheticGgufShape::k_quant());
    let report = evaluate_gguf(&header_of(&bytes), &repo(), "f.gguf", None);
    assert_eq!(report.verdict, Verdict::Runnable, "{:?}", report.verdict);
    let names: Vec<&str> = report.types.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"Q4_K"), "{names:?}");
    assert!(names.contains(&"Q6_K"), "{names:?}");
    assert!(
        report.types.iter().all(|t| t.executable),
        "a Q4_K_M-shaped file has kernels for every type: {:?}",
        report.types
    );
}

/// The expert-granularity arithmetic, which is the number that decides
/// whether a model FITS here (AGENTS.md Gotcha 36). It gates nothing, so
/// without a test it could silently become zero.
#[test]
fn an_moe_fixture_reports_an_expert_stride_and_a_slot_cache_that_scales() {
    let (bytes, _) = build_synthetic_gemma4_gguf(SyntheticGgufShape::default());
    let report = evaluate_gguf(&header_of(&bytes), &repo(), "f.gguf", None);
    let stride = report.expert_stride.expect("an MoE fixture has a stride");
    assert!(stride > 0);

    let cache = report.slot_cache_bytes();
    assert_eq!(cache.len(), 3, "8/16/32 slots");
    let layers = report.arch.as_ref().unwrap().num_layers as u64;
    for (slots, bytes) in &cache {
        assert_eq!(
            *bytes,
            slots * layers * stride,
            "the slot cache is slots x layers x expert stride"
        );
    }
    // Doubling the slots doubles the cache. Stated because the whole reason
    // this is printed is that a reader can extrapolate it.
    assert_eq!(cache[1].1, cache[0].1 * 2);
    assert_eq!(cache[2].1, cache[1].1 * 2);
}

/// A GGUF whose architecture this port recognizes and cannot run gets the
/// registry's own `needs` clause, which is far more useful than "unknown".
#[test]
fn a_planned_architecture_is_refused_with_what_it_would_need() {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "phi3")
        .build();
    let report = evaluate_gguf(&header_of(&bytes), &repo(), "f.gguf", None);
    match &report.verdict {
        Verdict::Refused(why) => {
            assert!(why.contains("phi3"), "{why}");
            assert!(
                why.contains("recognized but has no decode flow"),
                "a PLANNED architecture must not read as an unknown one: {why}"
            );
            assert!(why.contains("docs/NEW_MODEL.md"), "{why}");
        }
        other => panic!("phi3 should be refused, got {other:?}"),
    }
}

#[test]
fn an_unknown_architecture_is_refused_differently_from_a_planned_one() {
    let (bytes, _) = GgufBuilder::new()
        .metadata_str("general.architecture", "not-a-real-architecture")
        .build();
    let report = evaluate_gguf(&header_of(&bytes), &repo(), "f.gguf", None);
    match &report.verdict {
        Verdict::Refused(why) => {
            assert!(why.contains("not in this port's registry"), "{why}");
            assert!(
                !why.contains("recognized but"),
                "an unknown architecture must not claim to be recognized: {why}"
            );
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_gguf_declaring_no_architecture_at_all_is_refused() {
    let (bytes, _) = GgufBuilder::new().build();
    let report = evaluate_gguf(&header_of(&bytes), &repo(), "f.gguf", None);
    assert!(!report.verdict.is_runnable());
}

// -- safetensors -----------------------------------------------------------

/// The minimum a `config.json` needs to reach the quantization gate. Deleting
/// or overriding keys from this is how the refusal cases below are built.
fn qwen35_config(bits: u32, group: u32) -> String {
    // Every fourth layer is full attention, which is `full_attention_interval
    // = 4` written out. The parser derives the layer mask from this list
    // rather than from the interval, so it cannot be omitted.
    let layer_types: Vec<&str> = (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "\"full_attention\""
            } else {
                "\"linear_attention\""
            }
        })
        .collect();
    let layer_types = layer_types.join(", ");
    format!(
        r#"{{
        "model_type": "qwen3_5",
        "quantization": {{"bits": {bits}, "group_size": {group}}},
        "text_config": {{
            "model_type": "qwen3_5_text",
            "layer_types": [{layer_types}],
            "hidden_size": 5120, "intermediate_size": 20480,
            "num_hidden_layers": 64, "num_attention_heads": 40,
            "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 248320, "rms_norm_eps": 1e-05,
            "rope_parameters": {{"rope_theta": 5000000.0}},
            "tie_word_embeddings": false,
            "full_attention_interval": 4, "partial_rotary_factor": 0.5,
            "hidden_act": "silu",
            "linear_num_value_heads": 48, "linear_num_key_heads": 24,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4
        }}
    }}"#
    )
}

#[test]
fn a_two_bit_group_128_config_is_runnable_and_reports_its_width() {
    let report =
        evaluate_config(&qwen35_config(2, 128), &repo(), Some(99)).expect("the config parses");
    assert_eq!(report.verdict, Verdict::Runnable, "{:?}", report.verdict);
    assert_eq!(report.affine, Some((2, 128)));
    assert_eq!(report.family, Some(model_io::ModelFamily::QwenGdnDense));
    assert_eq!(report.download_bytes, Some(99));
    assert_eq!(
        report.expert_stride, None,
        "this family is dense, so nothing streams"
    );
}

/// **The affine shape is a CONJUNCTION, not two independent lists.** 2-bit
/// exists and group 64 exists, and `(2, 64)` is a pair no kernel implements.
/// A probe that checked the two axes separately would wave it through.
#[test]
fn an_affine_shape_no_kernel_implements_is_refused_even_though_both_axes_exist() {
    for (bits, group) in [(2, 64), (4, 128), (1, 64), (3, 128)] {
        let report =
            evaluate_config(&qwen35_config(bits, group), &repo(), None).expect("the config parses");
        match &report.verdict {
            Verdict::Refused(why) => assert!(
                why.contains(&format!("{bits}-bit at group {group}")),
                "the refusal should name the shape: {why}"
            ),
            other => panic!("({bits}, {group}) should be refused, got {other:?}"),
        }
    }
    // The controls: every shape that DOES have kernels passes.
    for (bits, group) in [(4, 64), (8, 64), (1, 128), (2, 128)] {
        let report = evaluate_config(&qwen35_config(bits, group), &repo(), None).unwrap();
        assert_eq!(
            report.verdict,
            Verdict::Runnable,
            "({bits}, {group}) has kernels and should pass"
        );
    }
}

#[test]
fn an_unregistered_model_type_is_refused_and_the_string_is_still_reported() {
    // BOTH spellings, because `config_json_family` reads the root
    // `model_type` and then `text_config.model_type`: replacing only the
    // first still resolves through the second, which is the multimodal
    // checkpoints' convention working as designed.
    let config = qwen35_config(4, 64)
        .replace("\"qwen3_5\"", "\"mistral\"")
        .replace("\"qwen3_5_text\"", "\"mistral_text\"");
    let report = evaluate_config(&config, &repo(), None).expect("the config parses");
    match &report.verdict {
        Verdict::Refused(why) => {
            assert!(why.contains("mistral"), "{why}");
            assert!(why.contains("arch_registry.rs"), "{why}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(
        report.architecture.as_deref(),
        Some("mistral"),
        "an unrecognized model_type is more useful printed than elided"
    );
}

/// **An unquantized checkpoint must be refused, and the parser's own default
/// says the opposite.** `parse_gemma4_quantization` answers 4-bit group 64
/// when the block is absent, which is correct for callers that have already
/// established the checkpoint is MLX-quantized and catastrophic for a probe
/// whose whole job is asking. Without the presence check this test drives,
/// every BF16 checkpoint on Hugging Face reports as runnable INT4.
#[test]
fn an_unquantized_checkpoint_is_refused_by_name() {
    let config =
        qwen35_config(4, 64).replace(r#""quantization": {"bits": 4, "group_size": 64},"#, "");
    let report = evaluate_config(&config, &repo(), None).expect("the config parses");
    match &report.verdict {
        Verdict::Refused(why) => assert!(
            why.contains("quantization"),
            "the refusal should name the missing quantization: {why}"
        ),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_config_that_is_not_json_is_an_error_rather_than_a_refusal() {
    // A parse failure is a broken input, not a model verdict, and collapsing
    // the two would report "this model will not run" for a truncated download.
    assert!(evaluate_config("{not json", &repo(), None).is_err());
}

// -- RepoRef ---------------------------------------------------------------

#[test]
fn a_repo_reference_parses_with_and_without_a_revision() {
    let plain = RepoRef::parse("owner/name").unwrap();
    assert_eq!(plain.revision, "main", "an omitted revision means main");
    let pinned = RepoRef::parse("owner/name@abc123").unwrap();
    assert_eq!(pinned.repo, "owner/name");
    assert_eq!(pinned.revision, "abc123");
    assert_eq!(
        pinned.file_url("config.json"),
        "https://huggingface.co/owner/name/resolve/abc123/config.json"
    );

    for bad in ["name", "owner/name/extra", "/name", "owner/", "owner/name@"] {
        assert!(RepoRef::parse(bad).is_err(), "{bad:?} should not parse");
    }
}

#[test]
fn minimax_safetensors_remain_explicitly_refused() {
    let report = evaluate_config(r#"{"model_type":"minimax_m2"}"#, &repo(), None).unwrap();
    assert_eq!(report.family, Some(model_io::ModelFamily::MiniMaxM2));
    let Verdict::Refused(reason) = report.verdict else {
        panic!("MiniMax safetensors must not be admitted")
    };
    assert!(
        reason.contains("safetensors intake is not wired"),
        "{reason}"
    );
}
