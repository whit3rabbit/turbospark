//! The `gemma4_unified` encoder-free image embedder's config and ingest
//! (`docs/GEMMA4_UNIFIED_VISION_PHASE0.md`).
//!
//! There are no transformer blocks. The whole embedder (patch LayerNorm,
//! dense, LayerNorm, 2-D position table, LayerNorm) and the `embed_vision`
//! projector are written as ONE blob of `packed_vision/`, so the manifest, the
//! writers and the open-time file checks need nothing new, and a text turn
//! never reads the roughly 100 MiB.
//!
//! The two matrices ship MLX affine on the published conversions and are
//! dequantized here to FP16, as the Gemma 4 projector is.
//!
//! The checkpoint also carries `embed_audio.`. Audio is not ported;
//! [`super::drop_gemma4_unified_audio`] removes it before the walk, by name.

use model_io::{VisionConfig, VisionTowerKind};

use super::config::{Gemma4Error, Gemma4Quant};
use super::shards::{Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{read_tower_tensor, VisionRead};
use super::vision_gemma4::EMBED_VISION_PREFIX;
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};

/// The wrapper's root `model_type`.
pub const GEMMA4_UNIFIED_MODEL_TYPE: &str = "gemma4_unified";
/// The embedder's checkpoint prefix (the MLX conversions' spelling).
pub const GEMMA4_UNIFIED_EMBEDDER_PREFIX: &str = "vision_embedder.";

/// `(role in the blob, checkpoint name)`, in forward order.
pub const GEMMA4_UNIFIED_ROLES: [(&str, &str); 10] = [
    ("ln1_w", "vision_embedder.patch_ln1.weight"),
    ("ln1_b", "vision_embedder.patch_ln1.bias"),
    ("dense_w", "vision_embedder.patch_dense.weight"),
    ("dense_b", "vision_embedder.patch_dense.bias"),
    ("ln2_w", "vision_embedder.patch_ln2.weight"),
    ("ln2_b", "vision_embedder.patch_ln2.bias"),
    ("pos", "vision_embedder.pos_embedding"),
    ("posln_w", "vision_embedder.pos_norm.weight"),
    ("posln_b", "vision_embedder.pos_norm.bias"),
    ("proj_w", "embed_vision.embedding_projection.weight"),
];

/// The soft-token budget the runtime serves (`vision-io`'s
/// `GEMMA4_DEFAULT_SOFT_TOKENS`; this crate does not depend on it).
const SOFT_TOKENS: i64 = 280;

/// Bound on every dimension read off the config, so shape arithmetic on
/// untrusted numbers cannot overflow. Published values are at most 3840.
const MAX_DIM: i64 = 1 << 16;

/// Parse a `gemma4_unified` `config.json`'s vision half.
///
/// Absent `vision_config` is [`VisionConfig::NONE`]. What the runtime does not
/// store (the pooling and patch sizes behind the 48 px model patch, the image
/// mask, the projector epsilon, the 280 token budget) is checked here and
/// refused when it differs.
///
/// Field meanings for [`VisionTowerKind::Gemma4Unified`]: `depth` is 1 (the one
/// embedder stage, so the tower reads as active and owns one blob),
/// `hidden_size` is `mm_embed_dim`, `spatial_merge_size` is the pooling kernel
/// (a model patch is `patch_size * spatial_merge_size` pixels),
/// `num_position_embeddings` is `mm_posemb_size`, and `intermediate_size` and
/// `num_heads` are 0 because there is no encoder.
pub fn parse_gemma4_unified_vision_config(json: &str) -> Result<VisionConfig, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let bad = Gemma4Error::Config;
    let vc = match root.get("vision_config") {
        None | Some(serde_json::Value::Null) => return Ok(VisionConfig::NONE),
        Some(v) => v,
    };
    let text = |v: &serde_json::Value, key: &str| {
        v.get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    };
    if text(&root, "model_type").as_deref() != Some(GEMMA4_UNIFIED_MODEL_TYPE)
        || text(vc, "model_type").as_deref() != Some("gemma4_unified_vision")
    {
        return Err(bad(
            "expected model_type gemma4_unified with a gemma4_unified_vision vision_config".into(),
        ));
    }
    let tc = root
        .get("text_config")
        .ok_or_else(|| bad("missing text_config".into()))?;
    if text(tc, "use_bidirectional_attention").as_deref() != Some("vision") {
        return Err(bad(
            "text_config.use_bidirectional_attention is not \"vision\", the implemented image mask"
                .into(),
        ));
    }
    let int = |v: &serde_json::Value, key: &str| -> Result<i64, Gemma4Error> {
        v.get(key)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| bad(format!("missing {key}")))
    };
    let dim = |v: &serde_json::Value, key: &str| -> Result<i64, Gemma4Error> {
        let n = int(v, key)?;
        if !(1..=MAX_DIM).contains(&n) {
            return Err(bad(format!("{key} {n} is outside 1..={MAX_DIM}")));
        }
        Ok(n)
    };
    let (patch, pool, model_patch) = (
        int(vc, "patch_size")?,
        int(vc, "pooling_kernel_size")?,
        int(vc, "model_patch_size")?,
    );
    if (patch, pool, model_patch) != (16, 3, 48) {
        return Err(bad(format!(
            "patch {patch}, pool {pool}, model patch {model_patch}; the implemented \
             preprocessing is 16, 3, 48"
        )));
    }
    let soft = int(vc, "num_soft_tokens")?;
    if soft != SOFT_TOKENS {
        return Err(bad(format!(
            "num_soft_tokens {soft}; the runtime serves the 280 token budget"
        )));
    }
    // Only the projector's RMSNorm epsilon is in the config. The three
    // LayerNorms use the reference's unconfigured default.
    if vc.get("rms_norm_eps").and_then(serde_json::Value::as_f64) != Some(1e-6) {
        return Err(bad("vision rms_norm_eps must be 1e-6".into()));
    }
    let embed = dim(vc, "mm_embed_dim")?;
    if int(vc, "output_proj_dims")? != embed {
        return Err(bad(
            "output_proj_dims differs from mm_embed_dim; the projector reads the embedder's rows"
                .into(),
        ));
    }
    Ok(VisionConfig {
        depth: 1,
        hidden_size: embed,
        intermediate_size: 0,
        num_heads: 0,
        patch_size: patch,
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: pool,
        num_position_embeddings: dim(vc, "mm_posemb_size")?,
        out_hidden_size: dim(tc, "hidden_size")?,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: int(&root, "boi_token_id")?,
        vision_end_token_id: int(&root, "eoi_token_id")?,
        image_token_id: int(&root, "image_token_id")?,
        video_token_id: int(&root, "video_token_id")?,
        tower: VisionTowerKind::Gemma4Unified,
    })
}

/// The logical shape each role must have.
fn role_shape(role: &str, v: &VisionConfig) -> Vec<u64> {
    let d = v.hidden_size as u64;
    let edge = (v.patch_size * v.spatial_merge_size) as u64;
    let patch = edge * edge * v.in_channels as u64;
    match role {
        "ln1_w" | "ln1_b" => vec![patch],
        "dense_w" => vec![d, patch],
        "pos" => vec![v.num_position_embeddings as u64, 2, d],
        "proj_w" => vec![v.out_hidden_size as u64, d],
        _ => vec![d],
    }
}

/// Reads the embedder and projector into one FP16 blob.
///
/// Every name must be one of [`GEMMA4_UNIFIED_ROLES`] and every role must be
/// present: an unknown or missing tensor is refused by name.
pub(super) fn read_gemma4_unified_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
    quant: Option<&Gemma4Quant>,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::Gemma4Unified || vision.depth != 1 {
        return Err(Gemma4Error::Config(
            "not a gemma4_unified vision config".into(),
        ));
    }
    for &name in vision_bases {
        let ours = name.starts_with(GEMMA4_UNIFIED_EMBEDDER_PREFIX)
            || name.starts_with(EMBED_VISION_PREFIX);
        // A caller may pass the quantization companions; the walk does not.
        let companion = name
            .strip_suffix(".scales")
            .or_else(|| name.strip_suffix(".biases"))
            .is_some_and(|base| {
                GEMMA4_UNIFIED_ROLES
                    .iter()
                    .any(|(_, n)| n.strip_suffix(".weight") == Some(base))
            });
        if !ours || (!companion && !GEMMA4_UNIFIED_ROLES.iter().any(|(_, n)| *n == name)) {
            return Err(Gemma4Error::UnknownTensor(format!(
                "{name}: not a gemma4_unified vision tensor"
            )));
        }
    }
    let mut sub_tensors = Vec::with_capacity(GEMMA4_UNIFIED_ROLES.len());
    let mut lossy_conversion = Vec::new();
    let mut used = 0u64;
    for (role, name) in GEMMA4_UNIFIED_ROLES {
        if !vision_bases.contains(&name) {
            return Err(Gemma4Error::MissingTensor(name.to_string()));
        }
        let (converted, shape) = read_tower_tensor(shards, name, quant)?;
        let want = role_shape(role, vision);
        let bytes = want
            .iter()
            .try_fold(2u64, |a, &d| a.checked_mul(d))
            .ok_or_else(|| Gemma4Error::Config("vision shape overflows".into()))?;
        // The logical shape AND the byte count: a quantized plane's stored
        // shape is `[rows, words]`, and an undeclared width unpacks wrong.
        if shape != want || converted.bytes.len() as u64 != bytes {
            return Err(Gemma4Error::ShapeMismatch {
                tensor: name.to_string(),
                detail: format!(
                    "logical shape {shape:?} ({} bytes of FP16), expected {want:?}",
                    converted.bytes.len()
                ),
            });
        }
        if converted.lossy > 0 {
            lossy_conversion.push((name.to_string(), converted.lossy));
        }
        used += bytes;
        sub_tensors.push(SubTensor {
            role: role.to_string(),
            bytes: converted.bytes,
            dtype: "fp16".to_string(),
            shape,
        });
    }
    Ok(VisionRead {
        entries: Vec::new(),
        blocks: LayerBlobs {
            layer: 0,
            experts: vec![ExpertBlob {
                expert: 0,
                sub_tensors,
            }],
        },
        block_stride: used.div_ceil(GTURBO_PAGE_BYTES) * GTURBO_PAGE_BYTES,
        lossy_conversion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mlx-community/gemma-4-12B-it-4bit@73bcf090`, trimmed to what is read.
    fn real_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gemma4_unified",
            "boi_token_id": 255999, "eoi_token_id": 258882,
            "image_token_id": 258880, "video_token_id": 258884,
            "text_config": {"hidden_size": 3840, "use_bidirectional_attention": "vision"},
            "vision_config": {
                "model_type": "gemma4_unified_vision", "mm_embed_dim": 3840,
                "mm_posemb_size": 1120, "model_patch_size": 48, "num_soft_tokens": 280,
                "output_proj_dims": 3840, "patch_size": 16, "pooling_kernel_size": 3,
                "rms_norm_eps": 1e-6
            }
        })
    }

    #[test]
    fn the_published_config_parses() {
        let v = parse_gemma4_unified_vision_config(&real_config().to_string()).unwrap();
        assert_eq!(v.tower, VisionTowerKind::Gemma4Unified);
        assert_eq!(
            (v.depth, v.hidden_size, v.num_position_embeddings),
            (1, 3840, 1120)
        );
        assert_eq!((v.out_hidden_size, v.image_token_id), (3840, 258880));
        assert_eq!(
            (v.vision_start_token_id, v.vision_end_token_id),
            (255999, 258882)
        );
        assert_eq!(role_shape("dense_w", &v), vec![3840, 6912]);
        assert_eq!(role_shape("pos", &v), vec![1120, 2, 3840]);
    }

    #[test]
    fn each_changed_value_is_refused() {
        for (path, value) in [
            ("/vision_config/patch_size", serde_json::json!(14)),
            ("/vision_config/pooling_kernel_size", serde_json::json!(2)),
            ("/vision_config/model_patch_size", serde_json::json!(32)),
            ("/vision_config/num_soft_tokens", serde_json::json!(560)),
            ("/vision_config/rms_norm_eps", serde_json::json!(1e-5)),
            ("/vision_config/output_proj_dims", serde_json::json!(1920)),
            ("/vision_config/mm_embed_dim", serde_json::json!(0)),
            (
                "/vision_config/model_type",
                serde_json::json!("gemma4_vision"),
            ),
            ("/model_type", serde_json::json!("gemma4")),
            (
                "/text_config/use_bidirectional_attention",
                serde_json::json!("all"),
            ),
        ] {
            let mut c = real_config();
            *c.pointer_mut(path).unwrap() = value;
            assert!(
                parse_gemma4_unified_vision_config(&c.to_string()).is_err(),
                "{path}"
            );
        }
        let mut c = real_config();
        c["vision_config"] = serde_json::Value::Null;
        assert_eq!(
            parse_gemma4_unified_vision_config(&c.to_string()).unwrap(),
            VisionConfig::NONE
        );
    }
}
