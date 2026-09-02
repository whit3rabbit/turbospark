//! The THREE consumers of `qwen4_exp`'s new manifest fields: the writer, the
//! validator and the peeker (`crates/repack` Gotcha 18).
//!
//! **ONLY THE FIRST TWO FAIL LOUDLY WHEN ONE IS MISSED, WHICH IS THE WHOLE
//! REASON THIS FILE EXISTS.** M-V3 taught `build_manifest_json` to write the
//! vision tower and `arch_validation` to check it and stopped there, so every
//! caller of `peek_manifest_arch` resolved a vision install to
//! `VisionConfig::NONE`: the install opened, decoded text correctly, and
//! refused an image as though it were headless. No error, no warning, and a
//! manifest declaring `visionDepth: 27` two feet away. It took M-V4's real
//! parity gate on a 15 GB install to find.
//!
//! The same hole is worse here. `hyper_connections.mult` decides how WIDE the
//! residual stream is, so a peeker resolving it to zero hands a caller an
//! `ArchConfig` saying one stream for a model that has four -- and unlike a
//! missing tower, nothing about that refuses loudly at the edge.
//!
//! **AND THESE FIELDS WERE ALREADY HALF-WIRED BEFORE `qwen4_exp` ARRIVED.**
//! Every `ca*` and `hc*` field was VALIDATED and written by nothing, which
//! `build_manifest_json`'s own comment recorded as unreachable "until the day
//! a DSV4 install can be written". `qwen4_exp` reaches it by a different door,
//! since it declares both blocks and has a repack path. Left unwritten,
//! `arch_validation` would resolve `hcMult` to 0, compare it against 4, and
//! refuse every install this walk produced.

use std::path::{Path, PathBuf};

const VOCAB: i64 = 64;
const LAYERS: i64 = 4;

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "turbospark-qwen4-manifest-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn read_manifest(dir: &Path) -> serde_json::Value {
    let raw = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest.json");
    serde_json::from_str(&raw).expect("valid manifest json")
}

/// A real install of an existing family, whose manifest the walk just wrote.
fn build_install() -> PathBuf {
    let dir = temp_dir();
    turbospark_repack::build_synthetic_qwen_gdn_dense_install(
        &dir,
        VOCAB,
        LAYERS,
        "qwen4-mroundtrip",
    )
    .expect("synthetic install");
    dir
}

/// THE WRITER. Every new key must be PRESENT in a manifest the walk wrote,
/// including on a family that has none of the components.
///
/// Presence rather than value, because the value on this family is zero and a
/// zero is what an ABSENT key also resolves to -- so a value assertion here
/// would pass against a writer that emits nothing, which is exactly the bug.
/// The keys are checked by name against the deserialized struct's spelling.
#[test]
fn the_writer_emits_every_new_field_unconditionally() {
    let dir = build_install();
    let manifest = read_manifest(&dir);
    let arch = manifest.get("arch").expect("arch block");

    for key in [
        // Previously validated and written by NOTHING.
        "hcMult",
        "hcSinkhornIters",
        "hcEps",
        "caIndexNHeads",
        "caIndexHeadDim",
        "caIndexTopK",
        "caCSACompressRate",
        "caHCACompressRate",
        "caQLoraRank",
        "caCompressRopeTheta",
        // New with qwen4_exp.
        "hcLowrank",
        "caIndexKvHeads",
        "caIndexBudget",
        "linearOutputGateSigmoid",
        "pleNgramSize",
        "pleHeadsPerNgram",
        "pleNgramVocabSizeBase",
        "pleMakeDivisibleBy",
        "pleSplitNgramParts",
        "pleEmbedDim",
        "pleConvKernelSize",
        "pleLayerIds",
        "pleSeed",
    ] {
        assert!(
            arch.get(key).is_some(),
            "manifest.arch is missing {key}; an omitted field is validated against a \
             BASELINE, so a family that has the component and says nothing cannot load"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// **THE TWO `ca*` COMPRESS-RATE KEYS CARRY EXPLICIT SERDE RENAMES, AND
/// WRITING THE CAMEL-CASE SPELLING WOULD BE SILENT.**
///
/// `ManifestArch` is `rename_all = "camelCase"`, which would derive
/// `caCsaCompressRate`, but both fields override it to `caCSACompressRate` and
/// `caHCACompressRate`. A writer emitting the derived spelling produces a key
/// nothing deserializes: `Option` resolves to `None`, `unwrap_or(0)` makes it
/// zero, and the install validates -- against zero, for a family whose real
/// value is not zero.
#[test]
fn the_two_renamed_compress_rate_keys_use_the_spelling_serde_reads() {
    let dir = build_install();
    let arch = read_manifest(&dir);
    let arch = arch.get("arch").expect("arch block");

    assert!(arch.get("caCSACompressRate").is_some());
    assert!(arch.get("caHCACompressRate").is_some());
    assert!(
        arch.get("caCsaCompressRate").is_none(),
        "the camelCase-derived spelling is not what serde reads; writing it is silent"
    );
    assert!(arch.get("caHcaCompressRate").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// THE PEEKER, absent half: a family with none of the components must read
/// back as having none, and NOT as inheriting another family's.
#[test]
fn an_install_without_the_components_peeks_back_as_none() {
    let dir = build_install();
    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("peek");

    assert_eq!(
        peeked.hyper_connections,
        model_io::HyperConnectionConfig::NONE
    );
    assert_eq!(
        peeked.compressed_attention,
        model_io::CompressedAttentionConfig::NONE
    );
    assert_eq!(peeked.ple, model_io::PleConfig::NONE);
    assert!(!peeked.hyper_connections.is_active());
    assert!(!peeked.ple.is_active());
    assert!(
        !peeked.linear_attention.output_gate_sigmoid,
        "absent means SILU, which is what the kernel did unconditionally"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// THE PEEKER, present half: the values a `qwen4_exp` install declares must
/// come back exactly.
///
/// Patches an existing install's manifest rather than building a `qwen4_exp`
/// one, because this family has no synthetic builder yet -- that is Phase 1's
/// work. What this can still prove is the whole of the peeker's mapping, which
/// is the consumer that fails silently; the writer is covered by the two cases
/// above and the validator by the workspace suite, which would redden on any
/// install whose written and expected values disagreed.
#[test]
fn the_peeker_reads_every_qwen4_block_back() {
    let dir = build_install();
    let mut manifest = read_manifest(&dir);
    let want = model_io::qwen4_exp_125b_a6b();

    {
        let arch = manifest
            .get_mut("arch")
            .and_then(serde_json::Value::as_object_mut)
            .expect("arch object");
        for (key, value) in [
            ("hcMult", serde_json::json!(want.hyper_connections.mult)),
            (
                "hcLowrank",
                serde_json::json!(want.hyper_connections.lowrank),
            ),
            (
                "caIndexNHeads",
                serde_json::json!(want.compressed_attention.index_n_heads),
            ),
            (
                "caIndexKvHeads",
                serde_json::json!(want.compressed_attention.index_kv_heads),
            ),
            (
                "caIndexHeadDim",
                serde_json::json!(want.compressed_attention.index_head_dim),
            ),
            (
                "caIndexTopK",
                serde_json::json!(want.compressed_attention.index_top_k),
            ),
            (
                "caIndexBudget",
                serde_json::json!(want.compressed_attention.index_budget),
            ),
            (
                "caCSACompressRate",
                serde_json::json!(want.compressed_attention.csa_compress_rate),
            ),
            ("linearOutputGateSigmoid", serde_json::json!(true)),
            ("pleNgramSize", serde_json::json!(want.ple.ngram_size)),
            (
                "pleHeadsPerNgram",
                serde_json::json!(want.ple.heads_per_ngram),
            ),
            (
                "pleNgramVocabSizeBase",
                serde_json::json!(want.ple.ngram_vocab_size_base),
            ),
            (
                "pleMakeDivisibleBy",
                serde_json::json!(want.ple.make_divisible_by),
            ),
            (
                "pleSplitNgramParts",
                serde_json::json!(want.ple.split_ngram_parts),
            ),
            ("pleEmbedDim", serde_json::json!(want.ple.ple_embed_dim)),
            (
                "pleConvKernelSize",
                serde_json::json!(want.ple.conv_kernel_size),
            ),
            ("pleLayerIds", serde_json::json!(want.ple.layer_ids)),
            ("pleSeed", serde_json::json!(want.ple.seed)),
        ] {
            arch.insert(key.to_string(), value);
        }
    }
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("serialize"),
    )
    .expect("write manifest");

    let peeked = turbospark_repack::peek_manifest_arch(&dir).expect("peek");

    assert_eq!(
        peeked.hyper_connections.mult, 4,
        "the residual stream's WIDTH; zero here is a four-stream model read as one"
    );
    assert_eq!(peeked.hyper_connections.lowrank, 320);
    assert!(peeked.hyper_connections.is_active());

    assert_eq!(peeked.compressed_attention.index_budget, 2048);
    assert_eq!(peeked.compressed_attention.sparse_below(), 2048);
    assert_eq!(peeked.compressed_attention.index_kv_heads, 1);
    assert_eq!(peeked.compressed_attention.index_top_k, 512);
    assert_eq!(peeked.compressed_attention.csa_compress_rate, 4);

    assert_eq!(peeked.ple, want.ple, "the whole PLE block, field for field");
    assert!(peeked.ple.is_active());
    assert_eq!(peeked.ple.layer_indices(), vec![1]);

    assert!(
        peeked.linear_attention.output_gate_sigmoid,
        "the GDN output gate's activation; reading it wrong is fluent wrong output \
         (crates/gpu Gotcha 12)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
