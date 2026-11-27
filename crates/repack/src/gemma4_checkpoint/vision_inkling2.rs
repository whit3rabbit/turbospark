//! The Inkling 2 Mini vision tower's ingest (`docs/INKLING2_VISION_PHASE0.md`).
//!
//! The source is the HF original's `model.visual.*` (canonicalized onto
//! `vision_tower.` by [`super::classify::canonicalize_vision_header`], so the
//! names arrive as `vision_tower.backbone.*` and `vision_tower.proj_{in,out}`)
//! or an mlx-vlm export's `vision_tower.*` with the module renames. Every
//! tensor lands as FP16.
//!
//! Three transforms happen here, each exact in f32 and rounded to FP16 ONCE,
//! so the runtime block is the Muse block (`BLOCK_ROLES`, half-split rope)
//! with no Inkling-specific kernel:
//!
//! 1. **q, k and v are fused** into one `[3 * hidden, hidden]` weight and one
//!    bias, q then k then v (the Muse layout).
//! 2. **q and k are permuted from interleaved to half-split rope pairs.** The
//!    reference rotates channel pairs `(2i, 2i + 1)` of each head by angle
//!    `i`; the runtime's `encode_vision_rope_2d` rotates `(i, i + hd/2)`.
//!    Reordering each head's q and k rows to `[0, 2, 4, .., 1, 3, 5, ..]`
//!    turns one into the other, and q.k is unchanged because both sides get
//!    the same permutation. v and the output projection are untouched.
//! 3. **The layer scale is folded** into the branch's last linear:
//!    `x + g * (W a + b) == x + (diag(g) W) a + g * b`, so `ls_1.gamma` scales
//!    `out_proj`'s rows and bias and `ls_2.gamma` scales `c_proj`'s.
//!
//! The class row is stored as one tensor, `class_embedding +
//! cls_position_embedding` (the reference only ever uses the sum).

use std::collections::BTreeSet;

use model_io::{VisionConfig, VisionTowerKind};

use super::classify::{VISION_INSTALL_PREFIX, VISION_PREFIX};
use super::config::Gemma4Error;
use super::narrow::convert_raw_to_fp16;
use super::shards::{Gemma4Shards, GTURBO_PAGE_BYTES};
use super::vision::{VisionRead, BLOCK_ROLES};
use crate::gturbo_writer::{ExpertBlob, LayerBlobs, SubTensor};
use crate::resident_writer::{RawTensorSpec, ResidentEntrySpec};

/// `model_type` of the checkpoint root.
pub const INKLING2_MODEL_TYPE: &str = "inkling2_loop";
/// `<|unused_200054|>`, the image placeholder (`config.image_token_id`
/// defaults to it in transformers; the pinned `config.json` omits it).
pub const INKLING2_IMAGE_TOKEN_ID: i64 = 200_054;
/// `<|content_image|>`, rendered right before each placeholder.
pub const INKLING2_CONTENT_IMAGE_ID: i64 = 200_005;

const DEPTH: i64 = 23;
const HIDDEN: i64 = 1024;
const HEADS: i64 = 16;
const PATCH: i64 = 20;
const MERGE: i64 = 2;
const GRID_SIDE: i64 = 32;
const PROJECTION_HIDDEN: i64 = 4096;

/// The resident tensors: `(HF tail, mlx-vlm tail, install name)`, tails
/// relative to `vision_tower.` after canonicalization. The class row's two
/// sources are summed into `cls_embed.weight` and are not listed here.
pub const INKLING2_RESIDENT_TENSORS: [(&str, &str, &str); 5] = [
    (
        "backbone.embeddings.patch_embedding.weight",
        "embeddings.patch_embedding.weight",
        "patch_embed.proj.weight",
    ),
    (
        "backbone.embeddings.position_embedding",
        "embeddings.position_embedding",
        "pos_embed.weight",
    ),
    (
        "backbone.ln_pre.weight",
        "pre_layernorm.weight",
        "ln_pre.weight",
    ),
    ("backbone.ln_pre.bias", "pre_layernorm.bias", "ln_pre.bias"),
    ("proj_in.weight", "proj_in.weight", "merger.fc1.weight"),
];

/// `proj_out`, beside the residents because its shape is checked against
/// the trunk width.
const PROJ_OUT: (&str, &str, &str) = ("proj_out.weight", "proj_out.weight", "merger.fc2.weight");

const CLASS: [(&str, &str); 2] = [
    (
        "backbone.embeddings.class_embedding",
        "embeddings.class_embedding",
    ),
    (
        "backbone.embeddings.cls_position_embedding",
        "embeddings.cls_position_embedding",
    ),
];

/// Per-block source tails `(HF, mlx-vlm)`, relative to the block prefix.
const BLOCK_SOURCES: [(&str, &str); 18] = [
    ("ln_1.weight", "layernorm_before.weight"),
    ("ln_1.bias", "layernorm_before.bias"),
    ("ls_1.gamma", "lambda_1"),
    ("self_attn.q_proj.weight", "self_attn.q_proj.weight"),
    ("self_attn.q_proj.bias", "self_attn.q_proj.bias"),
    ("self_attn.k_proj.weight", "self_attn.k_proj.weight"),
    ("self_attn.k_proj.bias", "self_attn.k_proj.bias"),
    ("self_attn.v_proj.weight", "self_attn.v_proj.weight"),
    ("self_attn.v_proj.bias", "self_attn.v_proj.bias"),
    ("self_attn.out_proj.weight", "self_attn.out_proj.weight"),
    ("self_attn.out_proj.bias", "self_attn.out_proj.bias"),
    ("ln_2.weight", "layernorm_after.weight"),
    ("ln_2.bias", "layernorm_after.bias"),
    ("ls_2.gamma", "lambda_2"),
    ("mlp.c_fc.weight", "mlp.fc1.weight"),
    ("mlp.c_fc.bias", "mlp.fc1.bias"),
    ("mlp.c_proj.weight", "mlp.fc2.weight"),
    ("mlp.c_proj.bias", "mlp.fc2.bias"),
];

/// Which of the two source spellings a header carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Spelling {
    Hf,
    MlxVlm,
}

impl Spelling {
    fn pick<'a>(self, pair: (&'a str, &'a str)) -> &'a str {
        match self {
            Spelling::Hf => pair.0,
            Spelling::MlxVlm => pair.1,
        }
    }

    fn block_prefix(self) -> &'static str {
        match self {
            Spelling::Hf => "backbone.layers.",
            Spelling::MlxVlm => "layers.",
        }
    }
}

/// `vision_config` of an `inkling2_loop` checkpoint (read off
/// `thinkingmachines/Inkling-2-Mini` @ `ed40e7da`).
///
/// The kind's constants are checked, not stored: 23 blocks of 1024 with 16
/// heads, MLP ratio 4, patch 20, merge 2, the 32x32 table, the 4096-wide
/// bias-free projector, LayerNorm eps 1e-5. The rope (axial, theta 10000) and
/// the exact GELU are transformers defaults the file does not restate, so a
/// `vision_config` that does state a different one is refused.
pub fn parse_inkling2_vision_config(
    root: &serde_json::Value,
    vc: &serde_json::Value,
) -> Result<VisionConfig, Gemma4Error> {
    let cfg = |why: String| Gemma4Error::Config(format!("inkling2 vision_config: {why}"));
    let i = |k: &str| -> Result<i64, Gemma4Error> {
        vc.get(k)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| cfg(format!("missing {k}")))
    };
    if vc.get("model_type").and_then(serde_json::Value::as_str) != Some("inkling2_vision") {
        return Err(cfg(format!(
            "model_type {:?} is not inkling2_vision",
            vc.get("model_type")
        )));
    }
    let text_hidden = root
        .get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| cfg("missing text_config.hidden_size".into()))?;
    if i("text_hidden_size")? != text_hidden {
        return Err(cfg(format!(
            "text_hidden_size {} differs from text_config.hidden_size {text_hidden}",
            i("text_hidden_size")?
        )));
    }
    let mlp_ratio = vc
        .get("mlp_ratio")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(4.0);
    let eps = vc
        .get("layer_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| cfg("missing layer_norm_eps".into()))?;
    let checks = [
        ("num_hidden_layers", i("num_hidden_layers")?, DEPTH),
        ("hidden_size", i("hidden_size")?, HIDDEN),
        ("num_attention_heads", i("num_attention_heads")?, HEADS),
        ("native_patch_size", i("native_patch_size")?, PATCH),
        ("adapter_merge_size", i("adapter_merge_size")?, MERGE),
        ("posemb_grid_size", i("posemb_grid_size")?, GRID_SIDE),
        (
            "projection_hidden_dim",
            i("projection_hidden_dim")?,
            PROJECTION_HIDDEN,
        ),
    ];
    for (k, got, want) in checks {
        if got != want {
            return Err(cfg(format!("{k} {got}; only {want} is implemented")));
        }
    }
    if (mlp_ratio - 4.0).abs() > 0.0 || (eps - 1e-5).abs() > 1e-12 {
        return Err(cfg(format!(
            "mlp_ratio {mlp_ratio} / layer_norm_eps {eps}; only 4.0 / 1e-5 is implemented"
        )));
    }
    if vc
        .get("projector_bias")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
    {
        return Err(cfg("projector_bias must be false".into()));
    }
    if let Some(act) = vc.get("hidden_act").and_then(serde_json::Value::as_str) {
        if act != "gelu" {
            return Err(cfg(format!(
                "hidden_act {act}; only exact gelu is implemented"
            )));
        }
    }
    if let Some(rope) = vc.get("rope_parameters") {
        let ty = rope.get("rope_type").and_then(serde_json::Value::as_str);
        let theta = rope.get("rope_theta").and_then(serde_json::Value::as_f64);
        if ty.is_some_and(|t| t != "axial") || theta.is_some_and(|t| t != 10_000.0) {
            return Err(cfg(format!(
                "rope_parameters {rope}; only axial at theta 10000 is implemented"
            )));
        }
    }
    let image_token_id = root
        .get("image_token_id")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(INKLING2_IMAGE_TOKEN_ID);
    if image_token_id != INKLING2_IMAGE_TOKEN_ID {
        return Err(cfg(format!(
            "image_token_id {image_token_id}; the template renders {INKLING2_IMAGE_TOKEN_ID}"
        )));
    }
    Ok(VisionConfig {
        depth: DEPTH,
        hidden_size: HIDDEN,
        intermediate_size: HIDDEN * 4,
        num_heads: HEADS,
        patch_size: PATCH,
        // No temporal axis: the patch embedding is [1024, 20 * 20 * 3].
        temporal_patch_size: 1,
        in_channels: 3,
        spatial_merge_size: MERGE,
        num_position_embeddings: GRID_SIDE * GRID_SIDE,
        out_hidden_size: text_hidden,
        mrope_section: [0, 0, 0],
        deepstack_visual_indexes: Vec::new(),
        vision_start_token_id: INKLING2_CONTENT_IMAGE_ID,
        vision_end_token_id: 0,
        image_token_id,
        video_token_id: 0,
        tower: VisionTowerKind::Inkling2,
    })
}

/// Read one tensor as f32, whatever its source float dtype.
fn read_f32(shards: &Gemma4Shards<'_>, name: &str) -> Result<(Vec<f32>, Vec<u64>), Gemma4Error> {
    let info = shards.info(name)?;
    let bytes = shards.read(name)?;
    let values: Vec<f32> = match info.dtype.as_str() {
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| compute::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => {
            return Err(Gemma4Error::UnsupportedDtype {
                tensor: name.to_string(),
                dtype: format!("{other} in the inkling2 vision tower (expected BF16, F16 or F32)"),
            })
        }
    };
    let count: u64 = info.shape.iter().product();
    if values.len() as u64 != count {
        return Err(Gemma4Error::ShapeMismatch {
            tensor: name.to_string(),
            detail: format!("{} values for shape {:?}", values.len(), info.shape),
        });
    }
    Ok((values, info.shape.clone()))
}

/// f32 values to FP16 bytes, refusing an overflow by name.
fn to_fp16(name: &str, values: &[f32]) -> Result<(Vec<u8>, usize), Gemma4Error> {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c = convert_raw_to_fp16(name, "F32", raw)?;
    Ok((c.bytes, c.lossy))
}

/// The row order that turns interleaved rope pairs into half-split ones,
/// per head: new row `j` takes old row `2j` for `j < hd/2`, else
/// `2(j - hd/2) + 1`.
pub fn half_split_order(head_dim: usize) -> Vec<usize> {
    let half = head_dim / 2;
    (0..head_dim)
        .map(|j| if j < half { 2 * j } else { 2 * (j - half) + 1 })
        .collect()
}

/// Permute the rows of a `[heads * head_dim, cols]` matrix (or a
/// `[heads * head_dim]` bias, `cols` 1) within each head.
pub fn permute_head_rows(values: &[f32], heads: usize, head_dim: usize, cols: usize) -> Vec<f32> {
    let order = half_split_order(head_dim);
    let mut out = vec![0.0f32; values.len()];
    for h in 0..heads {
        for (j, &src) in order.iter().enumerate() {
            let to = (h * head_dim + j) * cols;
            let from = (h * head_dim + src) * cols;
            out[to..to + cols].copy_from_slice(&values[from..from + cols]);
        }
    }
    out
}

/// Scale row `r` of a `[rows, cols]` matrix by `scale[r]`.
pub fn scale_rows(values: &mut [f32], scale: &[f32], cols: usize) {
    for (r, row) in values.chunks_exact_mut(cols).enumerate() {
        for v in row {
            *v *= scale[r];
        }
    }
}

/// Reads the whole tower. `vision_bases` is every name under
/// `vision_tower.` after canonicalization.
pub fn read_inkling2_vision_entries(
    shards: &Gemma4Shards<'_>,
    vision_bases: &[&str],
    vision: &VisionConfig,
) -> Result<VisionRead, Gemma4Error> {
    if vision.tower != VisionTowerKind::Inkling2
        || vision.depth != DEPTH
        || vision.hidden_size != HIDDEN
        || vision.num_heads != HEADS
    {
        return Err(Gemma4Error::Config(format!(
            "the inkling2 reader was handed a {} tower of depth {} x {} with {} heads",
            vision.tower.as_str(),
            vision.depth,
            vision.hidden_size,
            vision.num_heads
        )));
    }
    let depth = DEPTH as usize;
    let hidden = HIDDEN as usize;
    let heads = HEADS as usize;
    let head_dim = hidden / heads;
    let present: BTreeSet<&str> = vision_bases.iter().copied().collect();
    let spelling = if present
        .iter()
        .any(|n| n.starts_with(&format!("{VISION_PREFIX}backbone.")))
    {
        Spelling::Hf
    } else {
        Spelling::MlxVlm
    };
    let full = |tail: &str| format!("{VISION_PREFIX}{tail}");
    let block = |layer: usize, pair: (&str, &str)| {
        full(&format!(
            "{}{layer}.{}",
            spelling.block_prefix(),
            spelling.pick(pair)
        ))
    };

    // The exact set of names, refused by name before any bytes move.
    let mut expected: BTreeSet<String> = BTreeSet::new();
    for (hf, mlx, _) in INKLING2_RESIDENT_TENSORS.iter().chain([&PROJ_OUT]) {
        expected.insert(full(spelling.pick((*hf, *mlx))));
    }
    for pair in CLASS {
        expected.insert(full(spelling.pick(pair)));
    }
    for layer in 0..depth {
        for pair in BLOCK_SOURCES {
            expected.insert(block(layer, pair));
        }
    }
    if let Some(name) = vision_bases.iter().find(|n| !expected.contains(**n)) {
        return Err(Gemma4Error::UnknownTensor(format!(
            "{name}: not one of the inkling2 vision tower's tensors"
        )));
    }
    if let Some(name) = expected.iter().find(|n| !present.contains(n.as_str())) {
        return Err(Gemma4Error::MissingTensor(name.clone()));
    }

    let mut lossy_conversion = Vec::new();
    let mut note = |name: &str, lossy: usize| {
        if lossy > 0 {
            lossy_conversion.push((name.to_string(), lossy));
        }
    };
    let mismatch = |name: &str, want: &[u64], got: &[u64]| Gemma4Error::ShapeMismatch {
        tensor: name.to_string(),
        detail: format!("expected {want:?}, got {got:?}"),
    };
    let h = hidden as u64;
    let patch_dim = (PATCH * PATCH * 3) as u64;
    let proj = PROJECTION_HIDDEN as u64;
    let out_hidden = vision.out_hidden_size as u64;

    let mut entries = Vec::with_capacity(INKLING2_RESIDENT_TENSORS.len() + 2);
    let mut push = |install: &str, bytes: Vec<u8>, shape: &[u64]| {
        entries.push(ResidentEntrySpec::Raw(RawTensorSpec {
            name: format!("{VISION_INSTALL_PREFIX}{install}"),
            dtype: crate::DTYPE_FP16,
            bytes,
            shape: (
                shape[0] as u32,
                shape.get(1).copied().unwrap_or(0) as u32,
                0,
                0,
            ),
        }));
    };
    for (hf, mlx, install) in INKLING2_RESIDENT_TENSORS.iter().chain([&PROJ_OUT]) {
        let install: &str = install;
        let name = full(spelling.pick((*hf, *mlx)));
        let want: Vec<u64> = match install {
            "patch_embed.proj.weight" => vec![h, patch_dim],
            "pos_embed.weight" => vec![(GRID_SIDE * GRID_SIDE) as u64, h],
            "merger.fc1.weight" => vec![proj, h * (MERGE * MERGE) as u64],
            "merger.fc2.weight" => vec![out_hidden, proj],
            _ => vec![h],
        };
        let (values, shape) = read_f32(shards, &name)?;
        if shape != want {
            return Err(mismatch(&name, &want, &shape));
        }
        let (bytes, lossy) = to_fp16(&name, &values)?;
        note(&name, lossy);
        push(install, bytes, &shape);
    }
    let (mut cls, cls_shape) = read_f32(shards, &full(spelling.pick(CLASS[0])))?;
    let (cls_pos, pos_shape) = read_f32(shards, &full(spelling.pick(CLASS[1])))?;
    if cls_shape != [h] || pos_shape != [h] {
        return Err(mismatch(&full(spelling.pick(CLASS[0])), &[h], &cls_shape));
    }
    for (a, b) in cls.iter_mut().zip(&cls_pos) {
        *a += b;
    }
    // An arithmetic result rounds to FP16 by construction; only a plain
    // conversion's loss is reported (see the block loop).
    let (bytes, _) = to_fp16("cls_embed.weight", &cls)?;
    push("cls_embed.weight", bytes, &[h]);

    // The blocks: one blob each, in BLOCK_ROLES order.
    let mut experts = Vec::with_capacity(depth);
    let mut widest = 0u64;
    for layer in 0..depth {
        let get = |pair: (&str, &str), want: &[u64]| -> Result<Vec<f32>, Gemma4Error> {
            let name = block(layer, pair);
            let (values, shape) = read_f32(shards, &name)?;
            if shape != want {
                return Err(mismatch(&name, want, &shape));
            }
            Ok(values)
        };
        let src = |k: usize| BLOCK_SOURCES[k];
        let ln1_w = get(src(0), &[h])?;
        let ln1_b = get(src(1), &[h])?;
        let ls1 = get(src(2), &[h])?;
        let q_w = permute_head_rows(&get(src(3), &[h, h])?, heads, head_dim, hidden);
        let q_b = permute_head_rows(&get(src(4), &[h])?, heads, head_dim, 1);
        let k_w = permute_head_rows(&get(src(5), &[h, h])?, heads, head_dim, hidden);
        let k_b = permute_head_rows(&get(src(6), &[h])?, heads, head_dim, 1);
        let v_w = get(src(7), &[h, h])?;
        let v_b = get(src(8), &[h])?;
        let mut o_w = get(src(9), &[h, h])?;
        let mut o_b = get(src(10), &[h])?;
        let ln2_w = get(src(11), &[h])?;
        let ln2_b = get(src(12), &[h])?;
        let ls2 = get(src(13), &[h])?;
        let fc1_w = get(src(14), &[h * 4, h])?;
        let fc1_b = get(src(15), &[h * 4])?;
        let mut fc2_w = get(src(16), &[h, h * 4])?;
        let mut fc2_b = get(src(17), &[h])?;
        scale_rows(&mut o_w, &ls1, hidden);
        scale_rows(&mut o_b, &ls1, 1);
        scale_rows(&mut fc2_w, &ls2, hidden * 4);
        scale_rows(&mut fc2_b, &ls2, 1);
        let qkv_w: Vec<f32> = [q_w, k_w, v_w].concat();
        let qkv_b: Vec<f32> = [q_b, k_b, v_b].concat();

        let parts: [(&[f32], Vec<u64>); 12] = [
            (&ln1_w, vec![h]),
            (&ln1_b, vec![h]),
            (&qkv_w, vec![3 * h, h]),
            (&qkv_b, vec![3 * h]),
            (&o_w, vec![h, h]),
            (&o_b, vec![h]),
            (&ln2_w, vec![h]),
            (&ln2_b, vec![h]),
            (&fc1_w, vec![4 * h, h]),
            (&fc1_b, vec![4 * h]),
            (&fc2_w, vec![h, 4 * h]),
            (&fc2_b, vec![h]),
        ];
        let mut sub_tensors = Vec::with_capacity(BLOCK_ROLES.len());
        let mut used = 0u64;
        for ((role, _), (values, shape)) in BLOCK_ROLES.iter().zip(parts) {
            let label = format!("{VISION_PREFIX}block.{layer}.{role}");
            let (bytes, lossy) = to_fp16(&label, values)?;
            // `lossy_conversion` means a BF16 value FP16 cannot hold (the
            // subnormal range). The folded roles are f32 products, which
            // round once to FP16 like any computed value; counting that
            // rounding here would report the fold as a precision loss of
            // the source.
            if !matches!(*role, "proj_w" | "proj_b" | "fc2_w" | "fc2_b") {
                note(&label, lossy);
            }
            used += bytes.len() as u64;
            sub_tensors.push(SubTensor {
                role: role.to_string(),
                bytes,
                dtype: "fp16".to_string(),
                shape,
            });
        }
        widest = widest.max(used);
        experts.push(ExpertBlob {
            expert: layer,
            sub_tensors,
        });
    }

    Ok(VisionRead {
        entries,
        blocks: LayerBlobs { layer: 0, experts },
        block_stride: widest.div_ceil(GTURBO_PAGE_BYTES) * GTURBO_PAGE_BYTES,
        lossy_conversion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "inkling2_loop",
            "text_config": {"hidden_size": 2304},
            "vision_config": {
                "model_type": "inkling2_vision", "text_hidden_size": 2304,
                "temporal_patch_size": 2, "native_patch_size": 20, "adapter_merge_size": 2,
                "projection_hidden_dim": 4096, "projector_bias": false, "hidden_size": 1024,
                "num_hidden_layers": 23, "num_attention_heads": 16, "mlp_ratio": 4.0,
                "posemb_grid_size": 32, "layer_norm_eps": 1e-05
            }
        })
    }

    #[test]
    fn the_pinned_vision_config_parses() {
        let root = pinned_config();
        let v = parse_inkling2_vision_config(&root, &root["vision_config"]).unwrap();
        assert_eq!(v.tower, VisionTowerKind::Inkling2);
        assert_eq!((v.depth, v.hidden_size, v.num_heads), (23, 1024, 16));
        assert_eq!(
            (v.patch_size, v.spatial_merge_size, v.temporal_patch_size),
            (20, 2, 1)
        );
        assert_eq!(v.num_position_embeddings, 1024);
        assert_eq!(v.out_hidden_size, 2304);
        assert_eq!(v.image_token_id, 200_054);
        assert_eq!(v.vision_start_token_id, 200_005);
        assert!(v.is_active());
    }

    #[test]
    fn a_tower_this_port_does_not_implement_is_refused_by_name() {
        for (k, bad) in [
            ("num_hidden_layers", serde_json::json!(24)),
            ("native_patch_size", serde_json::json!(16)),
            ("projector_bias", serde_json::json!(true)),
            ("text_hidden_size", serde_json::json!(4608)),
            ("model_type", serde_json::json!("siglip_vision_model")),
            ("hidden_act", serde_json::json!("gelu_pytorch_tanh")),
        ] {
            let mut root = pinned_config();
            root["vision_config"][k] = bad;
            let e = parse_inkling2_vision_config(&root, &root["vision_config"]).unwrap_err();
            assert!(
                e.to_string().contains(k) || k == "text_hidden_size",
                "{k}: {e}"
            );
        }
        let mut root = pinned_config();
        root["image_token_id"] = serde_json::json!(5);
        assert!(parse_inkling2_vision_config(&root, &root["vision_config"]).is_err());
    }

    /// Interleaved-pair rope on `x` with angles `ang[i]` for pair `i`.
    fn rope_interleaved(x: &[f32], ang: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        for (i, a) in ang.iter().enumerate() {
            let (c, s) = (a.cos(), a.sin());
            let (x1, x2) = (x[2 * i], x[2 * i + 1]);
            out[2 * i] = x1 * c - x2 * s;
            out[2 * i + 1] = x2 * c + x1 * s;
        }
        out
    }

    /// Half-split rope: element `i` pairs with `i + hd/2`.
    fn rope_half(x: &[f32], ang: &[f32]) -> Vec<f32> {
        let half = x.len() / 2;
        let mut out = x.to_vec();
        for (i, a) in ang.iter().enumerate() {
            let (c, s) = (a.cos(), a.sin());
            let (x1, x2) = (x[i], x[i + half]);
            out[i] = x1 * c - x2 * s;
            out[i + half] = x2 * c + x1 * s;
        }
        out
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn permuted_rows_under_half_split_rope_give_the_interleaved_scores() {
        let hd = 8;
        let q: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.7).sin() + 0.1).collect();
        let k: Vec<f32> = (0..hd).map(|i| (i as f32 * 1.3).cos() - 0.2).collect();
        let aq = [0.3f32, 1.1, 2.0, 0.05];
        let ak = [1.7f32, 0.2, 0.9, 3.1];
        let want = dot(&rope_interleaved(&q, &aq), &rope_interleaved(&k, &ak));
        let qp = permute_head_rows(&q, 1, hd, 1);
        let kp = permute_head_rows(&k, 1, hd, 1);
        let got = dot(&rope_half(&qp, &aq), &rope_half(&kp, &ak));
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        // And the unpermuted vectors under the half-split kernel do NOT
        // match, so this test can tell the two pairings apart.
        let wrong = dot(&rope_half(&q, &aq), &rope_half(&k, &ak));
        assert!((wrong - want).abs() > 1e-3, "{wrong} vs {want}");
    }

    #[test]
    fn the_permutation_is_per_head() {
        // Two heads of four rows, one column: head 1 must stay in rows 4..8.
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert_eq!(
            permute_head_rows(&v, 2, 4, 1),
            vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]
        );
        assert_eq!(half_split_order(4), vec![0, 2, 1, 3]);
    }

    #[test]
    fn folding_the_layer_scale_matches_scaling_the_branch() {
        // [2 rows, 3 cols] weight, bias, scale g; input a.
        let w = vec![1.0f32, -2.0, 0.5, 3.0, 0.25, -1.0];
        let b = vec![0.1f32, -0.3];
        let g = vec![0.5f32, -2.0];
        let a = [0.7f32, -0.4, 1.9];
        let branch: Vec<f32> = (0..2)
            .map(|r| g[r] * (dot(&w[r * 3..r * 3 + 3], &a) + b[r]))
            .collect();
        let (mut wf, mut bf) = (w.clone(), b.clone());
        scale_rows(&mut wf, &g, 3);
        scale_rows(&mut bf, &g, 1);
        let folded: Vec<f32> = (0..2)
            .map(|r| dot(&wf[r * 3..r * 3 + 3], &a) + bf[r])
            .collect();
        for (x, y) in branch.iter().zip(&folded) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn the_source_tables_cover_the_published_tower() {
        // 18 per block x 23 + 5 residents + proj_out + 2 class rows = 422,
        // the HF original's `model.visual.*` count.
        assert_eq!(
            BLOCK_SOURCES.len() * 23 + INKLING2_RESIDENT_TENSORS.len() + 1 + 2,
            422
        );
    }
}
