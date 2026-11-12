//! Fish Audio Drama 3 intake: a headless `QwenGdnMoe` trunk plus the audio
//! side tensors, from `mlx-community/fish-audio-drama-3-4bit`.
//!
//! The checkpoint is one safetensors set holding three models. The text
//! backbone (`backbone.model.*`) is the Qwen 3.6 35B-A3B architecture field
//! for field, so it goes through the existing streamed `qwen36` writer into
//! `<install>/trunk/`, marked headless (`model_io::trunk_role`). The audio
//! tower (`tower.*`) and the semantic head (`semantic_head.*`) belong to the
//! TTS model, not the trunk, and are copied verbatim, under their original
//! names, into `<install>/tts_side.safetensors`. The codec ships as its own
//! file and is not touched here. Facts and gates: `docs/FISH_DRAMA3_PHASE0.md`.
//!
//! The walk sees one renamed namespace: trunk names become the canonical
//! `language_model.model.*`, and the side tensors move under
//! [`DRAMA3_SIDE_PREFIX`], an `audio_tower.` prefix `classify_for_family`
//! already excludes for every family. So the trunk writer drops them without
//! a new bucket, and [`write_drama3_side_safetensors`] reads them back out
//! of the same shards.

use std::io::Write;
use std::path::Path;

use model_io::{ArchConfig, ModelFamily};

use crate::gemma4_checkpoint::{Gemma4Error, Gemma4Shards};
use crate::safetensors_header::SafetensorsHeader;

/// The root `config.json` `model_type` of a Drama 3 checkpoint.
pub const DRAMA3_MODEL_TYPE: &str = "fish_drama3";

/// Where the side tensors live in the renamed walk namespace. Under
/// `audio_tower.`, which `classify_for_family` files as
/// `ExcludedMultimodal` for every family, so the trunk writer skips them.
pub const DRAMA3_SIDE_PREFIX: &str = "audio_tower.fish_drama3.";

/// The trunk install's directory under the Drama install root.
pub const DRAMA3_TRUNK_DIR: &str = "trunk";

/// The side tensors' file under the Drama install root.
pub const DRAMA3_SIDE_FILE: &str = "tts_side.safetensors";

/// `text_config` settings the `qwen36` decode flow hardcodes. A checkpoint
/// that differs would load and run, but compute something else, so each is
/// refused by name. Mirrors mlx-audio `fish_drama3/backbone.py`'s
/// `FIXED_TEXT_SETTINGS`.
const FIXED_TEXT_SETTINGS: [(&str, FixedValue); 6] = [
    ("attn_output_gate", FixedValue::Bool(true)),
    ("hidden_act", FixedValue::Str("silu")),
    ("layernorm_zero_centered_gamma", FixedValue::Bool(true)),
    ("rotary_interleaved", FixedValue::Bool(false)),
    ("shared_expert_gate", FixedValue::Bool(true)),
    ("use_qk_norm", FixedValue::Bool(true)),
];

enum FixedValue {
    Bool(bool),
    Str(&'static str),
}

impl FixedValue {
    fn matches(&self, value: Option<&serde_json::Value>) -> bool {
        match self {
            FixedValue::Bool(b) => value.and_then(serde_json::Value::as_bool) == Some(*b),
            FixedValue::Str(s) => value.and_then(serde_json::Value::as_str) == Some(*s),
        }
    }
}

/// The qwen flow's RMSNorm epsilon (`crates/runtime/src/families/qwen/mod.rs`
/// hardcodes it rather than reading the config).
const QWEN_FLOW_RMS_EPS: f64 = 1e-6;

/// The lowest aggregate mean the trunk's layer and final norms may have.
///
/// The checkpoint's norms are zero-centered (`w`) and the converter stores
/// them shifted (`1 + w`), which is what the qwen flow multiplies by. Measured
/// on the pinned checkpoint over all 81 input, post-attention and final norm
/// gammas: aggregate mean 0.984, no negative values. The same values unshifted
/// would average -0.016 with 55% negative. Single tensors are not separable
/// (one shifted post-attention norm averages 0.185), the aggregate is. 0.5
/// sits halfway, so an export that skipped the shift is refused instead of
/// running every norm about 1 too low.
const MIN_SHIFTED_NORM_MEAN: f64 = 0.5;

/// Validates a Drama 3 `config.json` for the trunk and returns the config
/// the generic `qwen36` parsers read: `text_config` unchanged, and the
/// `quantization` block with its per-path keys renamed onto the walk's
/// namespace.
///
/// The rename is load-bearing: the routed experts' 4-bit overrides are keyed
/// `backbone.model.layers.N.mlp.switch_mlp.*`, and after the header rename
/// no tensor carries those names, so without it every expert would fall back
/// to the 8-bit global default.
pub fn drama3_trunk_config(config_text: &str) -> Result<String, Gemma4Error> {
    let root: serde_json::Value =
        serde_json::from_str(config_text).map_err(|e| Gemma4Error::Config(e.to_string()))?;
    let model_type = root.get("model_type").and_then(serde_json::Value::as_str);
    if model_type != Some(DRAMA3_MODEL_TYPE) {
        return Err(Gemma4Error::Config(format!(
            "model_type {model_type:?} is not {DRAMA3_MODEL_TYPE:?}"
        )));
    }
    let tc = root
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Gemma4Error::Config("missing text_config".to_string()))?;

    for (key, want) in &FIXED_TEXT_SETTINGS {
        if !want.matches(tc.get(*key)) {
            return Err(Gemma4Error::Config(format!(
                "unsupported text_config {key}={:?}",
                tc.get(*key)
            )));
        }
    }
    let eps = tc.get("rms_norm_eps").and_then(serde_json::Value::as_f64);
    if eps != Some(QWEN_FLOW_RMS_EPS) {
        return Err(Gemma4Error::Config(format!(
            "text_config rms_norm_eps {eps:?}; the qwen36 flow hardcodes {QWEN_FLOW_RMS_EPS}"
        )));
    }
    if tc
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        // A tied manifest would let a binary that ignores the trunk-role
        // marker decode with the embedding table as its head: fluent output
        // from the wrong vocabulary instead of a missing-tensor refusal.
        return Err(Gemma4Error::Config(
            "a Drama 3 trunk must not tie its embeddings".to_string(),
        ));
    }
    let layers = tc
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Gemma4Error::Config("missing num_hidden_layers".to_string()))?;
    let full_every_fourth: Vec<serde_json::Value> = (0..layers)
        .map(|i| {
            let kind = if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            };
            serde_json::Value::from(kind)
        })
        .collect();
    if tc.get("layer_types").and_then(serde_json::Value::as_array) != Some(&full_every_fourth) {
        return Err(Gemma4Error::Config(
            "expected three linear-attention layers per full-attention layer".to_string(),
        ));
    }
    if let Some(types) = tc
        .get("mlp_layer_types")
        .and_then(serde_json::Value::as_array)
    {
        if types.len() as u64 != layers || types.iter().any(|t| t.as_str() != Some("moe")) {
            return Err(Gemma4Error::Config(
                "expected a mixture-of-experts MLP in every layer".to_string(),
            ));
        }
    }

    let mut quantization = serde_json::Map::new();
    if let Some(q) = root
        .get("quantization")
        .and_then(serde_json::Value::as_object)
    {
        for (key, value) in q {
            let canonical = match canonical_name(key) {
                Some(name) => name,
                // `group_size`, `bits` and `mode` are the block's globals.
                None if !key.contains('.') => key.clone(),
                None => {
                    return Err(Gemma4Error::Config(format!(
                        "quantization key {key} names no tensor of this checkpoint"
                    )))
                }
            };
            if quantization
                .insert(canonical.clone(), value.clone())
                .is_some()
            {
                return Err(Gemma4Error::Config(format!(
                    "quantization key {canonical} is declared twice"
                )));
            }
        }
    }
    let derived = serde_json::json!({
        "text_config": serde_json::Value::Object(tc.clone()),
        "quantization": serde_json::Value::Object(quantization),
    });
    Ok(derived.to_string())
}

/// The walk name of a source tensor (or quantization path), or `None` when
/// it is outside every namespace this checkpoint uses.
fn canonical_name(name: &str) -> Option<String> {
    const TRUNK: [&str; 3] = [
        "backbone.model.embed_tokens",
        "backbone.model.layers.",
        "backbone.model.norm",
    ];
    if TRUNK.iter().any(|p| name.starts_with(p)) {
        let tail = name.strip_prefix("backbone.model.")?;
        return Some(format!("language_model.model.{tail}"));
    }
    if name.starts_with("tower.") || name == "semantic_head" || name.starts_with("semantic_head.") {
        return Some(format!("{DRAMA3_SIDE_PREFIX}{name}"));
    }
    None
}

/// Renames a Drama 3 shard header onto the walk namespace (see the module
/// doc). Any name outside the three source namespaces is refused, including
/// a `language_model.*` name already present (a mixed namespace) and a
/// `backbone.lm_head` (Drama 3 has none, and a head here would be a
/// checkpoint this intake has never seen).
pub fn canonicalize_drama3_header(header: &mut SafetensorsHeader) -> Result<(), Gemma4Error> {
    let mut renamed = std::collections::BTreeMap::new();
    for (name, info) in std::mem::take(&mut header.tensors) {
        let canonical =
            canonical_name(&name).ok_or_else(|| Gemma4Error::UnknownTensor(name.clone()))?;
        if renamed.insert(canonical.clone(), info).is_some() {
            return Err(Gemma4Error::Config(format!(
                "Drama 3 tensor rename creates duplicate tensor {canonical}"
            )));
        }
    }
    header.tensors = renamed;
    Ok(())
}

fn decode_f32(name: &str, dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, Gemma4Error> {
    match dtype {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        other => Err(Gemma4Error::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// Refuses a trunk whose zero-centered norms were not stored shifted to
/// `1 + w` ([`MIN_SHIFTED_NORM_MEAN`]). Returns the aggregate mean.
pub fn check_drama3_norm_shift(shards: &Gemma4Shards<'_>) -> Result<f64, Gemma4Error> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for name in shards.names() {
        let layer_norm = name.starts_with("language_model.model.layers.")
            && (name.ends_with(".input_layernorm.weight")
                || name.ends_with(".post_attention_layernorm.weight"));
        if !(layer_norm || name.as_str() == "language_model.model.norm.weight") {
            continue;
        }
        let info = shards.info(name)?;
        for v in decode_f32(name, &info.dtype, &shards.read(name)?)? {
            sum += f64::from(v);
            count += 1;
        }
    }
    if count == 0 {
        return Err(Gemma4Error::MissingTensor(
            "language_model.model.norm.weight".to_string(),
        ));
    }
    let mean = sum / count as f64;
    // NaN is refused too: a non-finite gamma is not a shifted norm.
    if mean.is_nan() || mean < MIN_SHIFTED_NORM_MEAN {
        return Err(Gemma4Error::Config(format!(
            "trunk norm gammas average {mean:.3}; a Drama 3 trunk stores its \
             zero-centered norms shifted to 1 + w (aggregate mean near 1)"
        )));
    }
    Ok(mean)
}

/// Writes every side tensor (under [`DRAMA3_SIDE_PREFIX`] in the walk) to
/// `path` as a plain safetensors file under its ORIGINAL name, bytes
/// verbatim. Streams one tensor at a time, through a temporary file and a
/// rename. Returns the tensor count.
pub fn write_drama3_side_safetensors(
    path: &Path,
    shards: &Gemma4Shards<'_>,
) -> Result<usize, Gemma4Error> {
    let mut names: Vec<(&str, &str)> = shards
        .names()
        .filter_map(|n| n.strip_prefix(DRAMA3_SIDE_PREFIX).map(|o| (o, n.as_str())))
        .collect();
    names.sort_unstable();
    if names.is_empty() {
        return Err(Gemma4Error::MissingTensor(format!(
            "{DRAMA3_SIDE_PREFIX}tower.*"
        )));
    }

    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (original, walk) in &names {
        let info = shards.info(walk)?;
        let len = info.data_offsets.1 - info.data_offsets.0;
        header.insert(
            (*original).to_string(),
            serde_json::json!({
                "dtype": info.dtype,
                "shape": info.shape,
                "data_offsets": [offset, offset + len],
            }),
        );
        offset += len;
    }
    let mut json = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| Gemma4Error::Config(e.to_string()))?;
    // The format pads the header with spaces to an 8-byte boundary so the
    // data region is aligned for readers that map it.
    while json.len() % 8 != 0 {
        json.push(b' ');
    }

    let io = |e: std::io::Error| Gemma4Error::Config(format!("{}: {e}", path.display()));
    let tmp = path.with_extension("safetensors.tmp");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(io)?);
    let written = (|| {
        file.write_all(&(json.len() as u64).to_le_bytes())
            .map_err(io)?;
        file.write_all(&json).map_err(io)?;
        for (_, walk) in &names {
            let info = shards.info(walk)?;
            let bytes = shards.read(walk)?;
            if bytes.len() as u64 != info.data_offsets.1 - info.data_offsets.0 {
                return Err(Gemma4Error::ShapeMismatch {
                    tensor: (*walk).to_string(),
                    detail: format!("read {} bytes for a declared range", bytes.len()),
                });
            }
            file.write_all(&bytes).map_err(io)?;
        }
        file.flush().map_err(io)?;
        file.get_ref().sync_all().map_err(io)
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    std::fs::rename(&tmp, path).map_err(io)?;
    Ok(names.len())
}

/// Streams a Drama 3 checkpoint into `dir`: the headless trunk under
/// [`DRAMA3_TRUNK_DIR`] and the side tensors in [`DRAMA3_SIDE_FILE`].
///
/// `shards` must already be renamed by [`canonicalize_drama3_header`]. The
/// root files (`config.json`, tokenizer, `codec.safetensors`, `LICENSE`,
/// `NOTICE`) are the caller's: they are verbatim downloads with pinned
/// hashes, not repack output.
///
/// Numerics this accepts, recorded here because the trunk writer applies
/// them silently: the F32 `1 + w` norms narrow to BF16 (the lossy count is
/// reported), and the BF16 router is quantized to INT8 because the qwen
/// flow's router kernel is INT8-only. The manifest's affine `embedding` slot
/// states the 8-bit global default although the table is raw BF16. The
/// runtime gathers by each tensor's own dtype, so the table reads correctly.
pub fn write_fish_drama3_install_streamed(
    dir: &Path,
    config_text: &str,
    model_id: &str,
    shards: &Gemma4Shards<'_>,
    mut progress: impl FnMut(&str),
) -> Result<ArchConfig, Box<dyn std::error::Error>> {
    let trunk_config = drama3_trunk_config(config_text)?;
    let arch = crate::qwen36_config::parse_qwen_gdn_moe_config(&trunk_config)?;
    if arch.family != ModelFamily::QwenGdnMoe || arch.tie_word_embeddings {
        return Err(Box::new(Gemma4Error::Config(
            "the Drama 3 trunk must parse as an untied qwen36 trunk".to_string(),
        )));
    }
    let quant = crate::gemma4_checkpoint::parse_gemma4_quantization(&trunk_config)?;
    if shards.contains("language_model.lm_head.weight") {
        return Err(Box::new(Gemma4Error::UnsupportedTensor {
            tensor: "language_model.lm_head.weight".to_string(),
            detail: "a Drama 3 trunk is headless; its head is semantic_head".to_string(),
        }));
    }
    let mean = check_drama3_norm_shift(shards)?;
    progress(&format!("trunk norms shifted (aggregate mean {mean:.3})"));

    let trunk = dir.join(DRAMA3_TRUNK_DIR);
    std::fs::create_dir_all(&trunk)?;
    // Before the trunk: the streamed writer copies existing files into its
    // staging directory, so the marker is published in the same rename as
    // the manifest and no reader ever sees an unmarked headless trunk.
    model_io::trunk_role::write_headless(&trunk, DRAMA3_MODEL_TYPE)?;
    crate::gemma4_checkpoint::write_qwen_gdn_moe_install_streamed(
        &trunk,
        &arch,
        model_id,
        shards,
        &quant,
        &mut progress,
    )?;
    let side = write_drama3_side_safetensors(&dir.join(DRAMA3_SIDE_FILE), shards)?;
    progress(&format!("wrote {side} side tensors to {DRAMA3_SIDE_FILE}"));
    Ok(arch)
}
