//! `sparse_attention_config`: the DASH constants an `unlimited_ocr2` install
//! carries beside its manifest.
//!
//! A sidecar rather than an `ArchConfig` field, because these are per-layer
//! router constants one family reads and `ArchConfig` is a struct literal in
//! every baseline and fixture. Written at repack, and read back through this
//! same parser at open, so the repack and the runtime cannot disagree about
//! what a key means.
//!
//! Every refusal is a behaviour the flow does not implement, stated at the
//! door: each would otherwise decode fluently and read the wrong page region.

use std::path::Path;

/// Install-relative file holding `{"sparse_attention_config": {...}}`.
pub const DASH_SIDECAR: &str = "unlimited_ocr2_dash.json";

/// Upstream's `DASH_INFER_TOPM`: scores entering the entmax prior. A constant
/// of the inference path, not a key in the checkpoint.
pub const DASH_INFER_TOPM: usize = 16;

/// The validated DASH constants.
#[derive(Debug, Clone, PartialEq)]
pub struct DashConfig {
    pub block_size: usize,
    pub local_blocks: usize,
    pub index_dim: usize,
    pub topk: usize,
    pub sigma: f32,
    pub image_token_id: i32,
    pub page_token_id: i32,
    pub per_page_image_tokens: usize,
    pub rope_tok_theta: f32,
    pub rope_page_theta: f32,
    pub rope_page_pairs: usize,
    pub rope_page_mod: usize,
    /// Per layer, `dash_scale_per_layer / sqrt(index_dim)`.
    pub idx_scale: Vec<f32>,
    /// Per layer, the prior floor.
    pub floor: Vec<f32>,
}

fn usize_key(sp: &serde_json::Value, key: &str) -> Result<usize, String> {
    sp.get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("sparse_attention_config.{key} is missing or not an integer"))
}

fn f64_key(sp: &serde_json::Value, key: &str) -> Result<f64, String> {
    sp.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("sparse_attention_config.{key} is missing or not a number"))
}

fn str_key<'a>(sp: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    sp.get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("sparse_attention_config.{key} is missing"))
}

fn per_layer(sp: &serde_json::Value, key: &str, layers: usize) -> Result<Vec<f64>, String> {
    let values = sp
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("sparse_attention_config.{key} is missing"))?;
    if values.len() != layers {
        return Err(format!(
            "sparse_attention_config.{key} has {} entries for {layers} layers",
            values.len()
        ));
    }
    values
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| format!("sparse_attention_config.{key} holds a non-number"))
        })
        .collect()
}

impl DashConfig {
    /// Parse and validate the `sparse_attention_config` object.
    pub fn parse(sp: &serde_json::Value, layers: usize, head_dim: usize) -> Result<Self, String> {
        let stage = str_key(sp, "training_stage")?;
        if stage != "dash" || sp.get("use_sparse_attention") != Some(&true.into()) {
            return Err(format!(
                "training_stage {stage:?} with use_sparse_attention off is not DASH inference"
            ));
        }
        for (key, want) in [
            ("dash_stage0", "head_cls"),
            ("sparse_main_rope_mode", "page2d"),
            ("sparse_main_rope_tok_freq_mode", "resample"),
        ] {
            let got = str_key(sp, key)?;
            if got != want {
                return Err(format!(
                    "sparse_attention_config.{key} is {got:?}; only {want:?} is wired"
                ));
            }
        }
        if sp.get("dash_image_only") != Some(&true.into()) {
            return Err("dash_image_only must be true; open text routing is not wired".into());
        }
        let freq = sp
            .get("sparse_attention_freq")
            .and_then(serde_json::Value::as_array)
            .ok_or("sparse_attention_config.sparse_attention_freq is missing")?;
        if freq.len() != layers || freq.iter().any(|f| f.as_i64() != Some(1)) {
            return Err("every layer must be a sparse layer (sparse_attention_freq all 1)".into());
        }
        let index_dim = usize_key(sp, "sparse_index_dim")?;
        if index_dim != head_dim {
            return Err(format!(
                "sparse_index_dim {index_dim} differs from head_dim {head_dim}; the router \
                 shares the trunk's rope table"
            ));
        }
        let block_size = usize_key(sp, "sparse_block_size")?;
        let local_blocks = usize_key(sp, "sparse_local_block")?;
        let topk = usize_key(sp, "dash_topk_blocks")?;
        let rope_page_pairs = usize_key(sp, "sparse_rope_page_pairs")?;
        if block_size == 0 || local_blocks == 0 || topk == 0 || topk > DASH_INFER_TOPM {
            return Err(format!(
                "block size {block_size}, local blocks {local_blocks} and top-k {topk} must be \
                 positive, with top-k at most {DASH_INFER_TOPM}"
            ));
        }
        if rope_page_pairs == 0 || rope_page_pairs >= head_dim / 2 {
            return Err(format!(
                "sparse_rope_page_pairs {rope_page_pairs} leaves no token pairs in head_dim \
                 {head_dim}"
            ));
        }
        let sigma = f64_key(sp, "dash_sigma")?;
        if sigma <= 0.0 {
            return Err(format!("dash_sigma {sigma} must be positive"));
        }
        let scale = 1.0 / (index_dim as f64).sqrt();
        let idx_scale = per_layer(sp, "dash_scale_per_layer", layers)?
            .into_iter()
            .map(|s| (s * scale) as f32)
            .collect();
        let floor = per_layer(sp, "dash_floor_per_layer", layers)?
            .into_iter()
            .map(|f| f as f32)
            .collect();
        Ok(Self {
            block_size,
            local_blocks,
            index_dim,
            topk,
            sigma: sigma as f32,
            image_token_id: usize_key(sp, "image_token_id")? as i32,
            page_token_id: usize_key(sp, "page_token_id")? as i32,
            per_page_image_tokens: usize_key(sp, "per_page_image_tokens")?,
            rope_tok_theta: f64_key(sp, "sparse_rope_tok_theta")? as f32,
            rope_page_theta: f64_key(sp, "sparse_rope_page_theta")? as f32,
            rope_page_pairs,
            rope_page_mod: usize_key(sp, "sparse_rope_page_mod")?,
            idx_scale,
            floor,
        })
    }

    /// Write the source object into an install, unmodified.
    pub fn write_sidecar(dir: &Path, sparse: &serde_json::Value) -> Result<(), String> {
        let body = serde_json::json!({ "sparse_attention_config": sparse });
        std::fs::write(
            dir.join(DASH_SIDECAR),
            serde_json::to_vec_pretty(&body).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("writing {DASH_SIDECAR}: {e}"))
    }

    /// Read an install's sidecar. A missing one is an error rather than a
    /// default: nothing about the weights says how to route.
    pub fn load(dir: &Path, layers: usize, head_dim: usize) -> Result<Self, String> {
        let bytes = std::fs::read(dir.join(DASH_SIDECAR))
            .map_err(|e| format!("reading {DASH_SIDECAR}: {e}"))?;
        let root: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("parsing {DASH_SIDECAR}: {e}"))?;
        let sparse = root
            .get("sparse_attention_config")
            .ok_or_else(|| format!("{DASH_SIDECAR} has no sparse_attention_config"))?;
        Self::parse(sparse, layers, head_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped() -> serde_json::Value {
        serde_json::json!({
            "training_stage": "dash",
            "use_sparse_attention": true,
            "dash_stage0": "head_cls",
            "dash_image_only": true,
            "dash_sigma": 1.0,
            "dash_topk_blocks": 12,
            "dash_scale_per_layer": [128.0, 128.0],
            "dash_floor_per_layer": [0.0001, 0.0001],
            "image_token_id": 128815,
            "page_token_id": 128827,
            "per_page_image_tokens": 273,
            "sparse_attention_freq": [1, 1],
            "sparse_block_size": 32,
            "sparse_index_dim": 128,
            "sparse_local_block": 4,
            "sparse_main_rope_mode": "page2d",
            "sparse_main_rope_tok_freq_mode": "resample",
            "sparse_rope_page_mod": 2,
            "sparse_rope_page_pairs": 1,
            "sparse_rope_page_theta": 256.0,
            "sparse_rope_tok_theta": 10000.0,
        })
    }

    #[test]
    fn parses_the_shipped_shape() {
        let c = DashConfig::parse(&shipped(), 2, 128).unwrap();
        assert_eq!(c.block_size * c.local_blocks, 128);
        assert_eq!(c.idx_scale.len(), 2);
        assert!((c.idx_scale[0] - 128.0 / 128f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn refuses_what_the_flow_does_not_implement() {
        for (key, value) in [
            ("training_stage", serde_json::json!("warmup")),
            ("dash_stage0", serde_json::json!("indexer")),
            ("sparse_main_rope_mode", serde_json::json!("1d")),
            (
                "sparse_main_rope_tok_freq_mode",
                serde_json::json!("truncate"),
            ),
            ("dash_image_only", serde_json::json!(false)),
            ("sparse_attention_freq", serde_json::json!([1, 0])),
            ("sparse_index_dim", serde_json::json!(64)),
            ("dash_topk_blocks", serde_json::json!(17)),
            ("sparse_rope_page_pairs", serde_json::json!(64)),
            ("dash_scale_per_layer", serde_json::json!([128.0])),
        ] {
            let mut sp = shipped();
            sp[key] = value;
            assert!(
                DashConfig::parse(&sp, 2, 128).is_err(),
                "{key} was accepted"
            );
        }
    }
}
