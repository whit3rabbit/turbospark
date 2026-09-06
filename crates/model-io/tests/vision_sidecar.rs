//! Tests for the vision-tower sidecar format (vision memory sidecar, part
//! A1): `ManifestArch::vision_config()`'s fallback rule, `SidecarRecord`'s
//! JSON round trip, and `is_sidecar_dir`.
//!
//! The full `load()` round trip through a real writer-produced directory is
//! covered in `crates/repack/tests/vision_sidecar.rs`, which has
//! `gturbo_writer` and the synthetic fixture builders available to produce a
//! structurally complete manifest; this crate cannot depend on `repack` (the
//! dependency runs the other way), so hand-writing a full `arch` object here
//! would just be a second, driftable copy of `build_manifest_json`'s ~90
//! keys.

use turbospark_model_io::{ModelFamily, VisionConfig};

fn tiny_vision() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden_size: 64,
        intermediate_size: 96,
        num_heads: 4,
        patch_size: 4,
        temporal_patch_size: 2,
        in_channels: 3,
        spatial_merge_size: 2,
        num_position_embeddings: 16,
        out_hidden_size: 5120,
        mrope_section: [11, 11, 10],
        vision_start_token_id: 248_053,
        vision_end_token_id: 248_054,
        image_token_id: 248_056,
        video_token_id: 248_057,
    }
}

/// A `ManifestArch`-shaped JSON string with every REQUIRED field filled and
/// the fifteen optional `vision*` fields taken verbatim from `vision_json`
/// (either omitted entirely, to test the fallback, or filled in to test the
/// round trip).
fn manifest_arch_json(vision_json: &str) -> String {
    format!(
        r#"{{
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
            {vision_json}
        }}"#
    )
}

/// Every `vision*` field absent means `VisionConfig::NONE`, never a
/// baseline's tower -- the same fallback `arch_validation` applies inline,
/// codified once so a reader (`crates/repack`'s `manifest_peek`, and
/// `model_io::vision_sidecar::load`) cannot disagree with the loader about
/// what silence means (AGENTS.md Gotcha 24).
#[test]
fn vision_config_defaults_to_none_when_fields_are_absent() {
    let arch: turbospark_model_io::ManifestArch =
        serde_json::from_str(&manifest_arch_json("")).expect("a manifest arch with no vision key");
    assert_eq!(arch.vision_config(), VisionConfig::NONE);
}

/// Every declared `vision*` field reaches `VisionConfig` unchanged.
#[test]
fn vision_config_round_trips_declared_fields() {
    let vision_json = r#",
        "visionDepth": 2,
        "visionHiddenSize": 64,
        "visionIntermediateSize": 96,
        "visionNumHeads": 4,
        "visionPatchSize": 4,
        "visionTemporalPatchSize": 2,
        "visionInChannels": 3,
        "visionSpatialMergeSize": 2,
        "visionNumPositionEmbeddings": 16,
        "visionOutHiddenSize": 5120,
        "visionMropeSection": [11, 11, 10],
        "visionStartTokenId": 248053,
        "visionEndTokenId": 248054,
        "visionImageTokenId": 248056,
        "visionVideoTokenId": 248057"#;
    let arch: turbospark_model_io::ManifestArch =
        serde_json::from_str(&manifest_arch_json(vision_json))
            .expect("a manifest arch with every vision key declared");
    assert_eq!(arch.vision_config(), tiny_vision());
}

/// A PARTIAL declaration is not silently completed from a baseline: only the
/// fields actually present move, and everything else still falls back to
/// zero rather than to Gemma's (or any other family's) values.
#[test]
fn vision_config_treats_a_partial_declaration_as_partial() {
    let vision_json = r#",
        "visionDepth": 2,
        "visionHiddenSize": 64"#;
    let arch: turbospark_model_io::ManifestArch =
        serde_json::from_str(&manifest_arch_json(vision_json))
            .expect("a manifest arch with only two vision keys declared");
    let vision = arch.vision_config();
    assert_eq!(vision.depth, 2);
    assert_eq!(vision.hidden_size, 64);
    assert_eq!(vision.intermediate_size, 0, "an undeclared field must be 0");
    assert_eq!(vision.out_hidden_size, 0, "an undeclared field must be 0");
}

fn tempfile_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "turbospark-model-io-vision-sidecar-test-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// `SidecarRecord` round-trips through its own file format with no manifest
/// involved.
#[test]
fn sidecar_record_write_and_read_round_trip() {
    let dir = tempfile_dir();
    let record = turbospark_model_io::SidecarRecord {
        kind: turbospark_model_io::SIDECAR_KIND.to_string(),
        pairs_with: turbospark_model_io::PairsWith {
            family: ModelFamily::QwenGdnDense.as_str().to_string(),
            hidden_size: 5120,
        },
        source: turbospark_model_io::SidecarSource {
            repo: "mlx-community/Qwen3.8-27B-4bit".to_string(),
            revision: "3e6447f082e89cc7f0bc6e5441afd38dfce760ff".to_string(),
            prefix: "vision_tower.".to_string(),
            file: "model.safetensors".to_string(),
        },
        tower_blocks: 2,
        block_stride: 65536,
    };
    record.write(&dir).expect("write vision_sidecar.json");
    let back = turbospark_model_io::SidecarRecord::read(&dir).expect("read vision_sidecar.json");
    assert_eq!(back, record);
}

/// `is_sidecar_dir` is true only once a record naming [`SIDECAR_KIND`] is on
/// disk, and false for an ordinary directory (a plain trunk install has no
/// `vision_sidecar.json` at all).
#[test]
fn is_sidecar_dir_true_only_after_a_matching_record_is_written() {
    let dir = tempfile_dir();
    assert!(
        !turbospark_model_io::is_sidecar_dir(&dir),
        "an empty directory must not read as a sidecar"
    );

    let record = turbospark_model_io::SidecarRecord {
        kind: turbospark_model_io::SIDECAR_KIND.to_string(),
        pairs_with: turbospark_model_io::PairsWith {
            family: ModelFamily::QwenGdnMoe.as_str().to_string(),
            hidden_size: 5120,
        },
        source: turbospark_model_io::SidecarSource {
            repo: "test/repo".to_string(),
            revision: "deadbeef".to_string(),
            prefix: "vision_tower.".to_string(),
            file: "model.safetensors".to_string(),
        },
        tower_blocks: 2,
        block_stride: 65536,
    };
    record.write(&dir).expect("write vision_sidecar.json");
    assert!(turbospark_model_io::is_sidecar_dir(&dir));
}

/// A record naming a DIFFERENT kind is not a sidecar, even though the file
/// parses. `is_sidecar_dir` reads `kind` as plain data rather than assuming
/// any file named `vision_sidecar.json` is one of these.
#[test]
fn is_sidecar_dir_false_for_a_record_with_a_different_kind() {
    let dir = tempfile_dir();
    std::fs::write(
        dir.join(turbospark_model_io::SIDECAR_RECORD_FILE),
        r#"{"kind":"something-else","pairsWith":{"family":"qwen35","hiddenSize":1},"source":{"repo":"r","revision":"v","prefix":"p","file":"f"},"towerBlocks":1,"blockStride":1}"#,
    )
    .unwrap();
    assert!(!turbospark_model_io::is_sidecar_dir(&dir));
}
