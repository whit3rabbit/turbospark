//! The safetensors half of the probe: `model_type`, the affine width, and
//! the routed-expert stride.

use model_io::{ArchConfig, ModelFamily};

use super::{ProbeReport, Verdict};
use crate::hf::{Client, RepoRef};

pub(super) fn probe_safetensors(
    client: &Client,
    repo: &RepoRef,
    files: &[String],
) -> Result<ProbeReport, String> {
    let config_bytes = client.get(&repo.file_url("config.json"))?;
    let config_text = String::from_utf8(config_bytes)
        .map_err(|e| format!("config.json of {repo} is not UTF-8: {e}"))?;
    let shard_bytes: u64 = files
        .iter()
        .filter(|f| f.ends_with(".safetensors"))
        .filter_map(|f| client.content_length(&repo.file_url(f)).ok().flatten())
        .sum();

    let mut report = evaluate_config(&config_text, repo, (shard_bytes > 0).then_some(shard_bytes))?;
    if let Some(arch) = &report.arch {
        report.expert_stride = mlx_expert_stride(client.token(), repo, files, arch);
    }
    Ok(report)
}

/// Every safetensors gate, against a `config.json` that is already in hand.
///
/// The network half of a safetensors probe is two GETs and a HEAD per shard;
/// this is the part that decides anything, and splitting it out is what lets
/// the refusal paths -- an unregistered `model_type`, an affine shape with no
/// kernels, an unquantized checkpoint -- be tested with no network.
pub fn evaluate_config(
    config_text: &str,
    repo: &RepoRef,
    download_bytes: Option<u64>,
) -> Result<ProbeReport, String> {
    let config: serde_json::Value = serde_json::from_str(config_text)
        .map_err(|e| format!("parsing config.json of {repo}: {e}"))?;

    // Read the string for the report even when it resolves to nothing: an
    // unrecognized `model_type` is far more useful printed than elided.
    let architecture = ["model_type"]
        .iter()
        .find_map(|k| config.get(*k).and_then(|v| v.as_str()))
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|t| t.get("model_type"))
                .and_then(|v| v.as_str())
        })
        .map(str::to_string);

    let mut report = ProbeReport {
        repo: repo.clone(),
        kind: crate::SourceKind::Mlx,
        file: None,
        download_bytes,
        architecture: architecture.clone(),
        family: None,
        arch: None,
        types: Vec::new(),
        affine: None,
        expert_stride: None,
        trained_context: repack::trained_context_meta::from_config_json(config_text),
        sidecars_present: Vec::new(),
        sidecars_missing: Vec::new(),
        chat_template: None,
        verdict: Verdict::Runnable,
        warnings: Vec::new(),
    };

    let Some(family) = repack::config_json_family(&config) else {
        report.refuse(format!(
            "config.json declares model_type {:?}, which is not in this port's registry \
             (crates/repack/src/arch_registry.rs). Bring-up checklist: docs/NEW_MODEL.md",
            architecture.as_deref().unwrap_or("<absent>")
        ));
        return Ok(report);
    };
    report.family = Some(family);

    // The family picks the parser. Each parser refuses a config that
    // positively claims another family, so this cannot silently mis-parse.
    let parsed = match family {
        ModelFamily::Gemma4 => repack::parse_gemma4_config(config_text).map_err(|e| e.to_string()),
        ModelFamily::QwenGdnMoe => {
            repack::parse_qwen_gdn_moe_config(config_text).map_err(|e| e.to_string())
        }
        ModelFamily::QwenGdnDense => {
            repack::parse_qwen_gdn_dense_config(config_text).map_err(|e| e.to_string())
        }
        ModelFamily::MuseGlimmer => {
            repack::parse_muse_glimmer_config(config_text).map_err(|e| e.to_string())
        }
        ModelFamily::Qwen4Exp => {
            repack::parse_qwen4_exp_config(config_text).map_err(|e| e.to_string())
        }
        other => Err(format!(
            "model_type resolves to {}, whose safetensors intake is not wired here; \
             this port installs that family from GGUF instead",
            other.as_str()
        )),
    };
    match parsed {
        Ok(arch) => report.arch = Some(arch),
        Err(e) => {
            report.refuse(e);
            return Ok(report);
        }
    }

    // The quantization shape, which for this intake is the whole question:
    // an unquantized checkpoint has no kernels here at all.
    //
    // **The PRESENCE of the block is checked separately from parsing it, and
    // that is not belt-and-braces.** `parse_gemma4_quantization` answers
    // `Gemma4Quant::default()` -- 4-bit, group 64 -- when the key is absent,
    // which is right for its own callers (they have already established the
    // checkpoint is MLX-quantized) and wrong here, where the caller is asking
    // whether it is. Inheriting that default makes every BF16 checkpoint on
    // Hugging Face probe as a runnable INT4 one. A default is a claim about
    // what silence means, and this module's question makes silence mean
    // something else (AGENTS.md Gotcha 39).
    if config
        .get("quantization")
        .and_then(|v| v.as_object())
        .is_none()
    {
        report.refuse(
            "config.json declares no `quantization` block, so this is an unquantized \
             checkpoint. This intake installs MLX-quantized checkpoints only (4- or \
             8-bit group 64, 1- or 2-bit group 128); there is no kernel here for BF16 \
             or FP16 weight matrices. Look for an mlx-community conversion of this \
             model, or a GGUF of it."
                .to_string(),
        );
        return Ok(report);
    }
    match repack::parse_gemma4_quantization(config_text) {
        Ok(quant) => {
            report.affine = Some((quant.default_bits, quant.group_size));
            let mut bad: Vec<(u32, u32)> = Vec::new();
            if !repack::is_supported_affine_shape(quant.default_bits, quant.group_size) {
                bad.push((quant.default_bits, quant.group_size));
            }
            for bits in quant.bits_overrides.values().copied() {
                if !repack::is_supported_affine_shape(bits, quant.group_size)
                    && !bad.contains(&(bits, quant.group_size))
                {
                    bad.push((bits, quant.group_size));
                }
            }
            if !bad.is_empty() {
                let shapes: Vec<String> = bad
                    .iter()
                    .map(|(b, g)| format!("{b}-bit at group {g}"))
                    .collect();
                report.refuse(format!(
                    "no kernels for {}. This port dispatches MLX affine at 4- or 8-bit \
                     group 64 and at 1- or 2-bit group 128, and the pairs are checked as \
                     whole conjunctions rather than as independent lists.",
                    shapes.join(", ")
                ));
            }
        }
        Err(e) => {
            report.refuse(format!(
                "config.json declares no usable quantization block ({e}). This intake \
                 installs MLX-quantized checkpoints only; an unquantized checkpoint has \
                 no kernels here."
            ));
            return Ok(report);
        }
    }

    Ok(report)
}

/// Bytes of ONE routed expert in an MLX install, off the first shard's
/// header.
///
/// The two routed markers differ per family (`.mlp.switch_mlp.` for Qwen,
/// `.experts.switch_glu.` for Gemma) and an unrecognized one is NOT an error
/// here: it means the model is dense, or that its experts live somewhere this
/// probe does not look, and either way the answer is "no stride reported"
/// rather than a wrong number.
fn mlx_expert_stride(
    token: Option<&str>,
    repo: &RepoRef,
    files: &[String],
    arch: &ArchConfig,
) -> Option<u64> {
    if arch.num_experts <= 0 {
        return None;
    }
    let first = files.iter().find(|f| f.ends_with(".safetensors"))?;
    let source = repack::HttpRangeSource::new(repo.file_url(first)).with_optional_token(token);
    let header = repack::fetch_safetensors_header(&source).ok()?;
    let mut total = 0u64;
    for (name, info) in &header.tensors {
        let routed = name.contains(".mlp.switch_mlp.") || name.contains(".experts.switch_glu.");
        if routed && (name.contains(".layers.0.") || name.contains(".0.")) {
            total += info.data_offsets.1.saturating_sub(info.data_offsets.0);
        }
    }
    (total > 0).then(|| total / arch.num_experts as u64)
}
