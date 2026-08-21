use std::path::Path;
use std::sync::Arc;

use model_io::{ArchConfig, ModelFamily};
use repack::{ByteProgressCallback, Gemma4Shards, HttpRangeSource, RangeSource};

use crate::hf::Client;
use crate::install::InstallPlan;

pub(crate) fn stream_gguf(
    plan: &InstallPlan,
    dir: &Path,
    progress: &mut impl FnMut(&str),
    byte_progress: Option<&ByteProgressCallback>,
) -> Result<ArchConfig, String> {
    let file = plan
        .file
        .as_deref()
        .ok_or_else(|| "a gguf install needs a filename".to_string())?;
    let url = plan.weights.file_url(file);
    let source = match byte_progress {
        Some(cb) => HttpRangeSource::with_progress(url, Arc::clone(cb)),
        None => HttpRangeSource::new(url),
    };
    let header = repack::fetch_gguf_header(&source)
        .map_err(|e| format!("reading the GGUF header of {file}: {e}"))?;
    let model_id = plan.weights.repo.clone();
    let arch = repack::write_gguf_install_streamed(dir, &header, &source, &model_id, |stage| {
        progress(&format!("[repack] {stage}"))
    })
    .map_err(|e| format!("streaming {file}: {e}"))?;
    record_trained_context(
        dir,
        repack::trained_context_meta::from_gguf(&header),
        progress,
    );
    Ok(arch)
}

/// Annotate the freshly written install with the checkpoint's own context
/// length, so `--max-context auto` has a ceiling to resolve against.
///
/// **Never fatal.** The install is complete and correct without it; all
/// that is lost is the model-side half of the context policy, which falls
/// back to the documented default exactly as it does for every install
/// written before the field existed. Failing a 20-minute stream over an
/// advisory number would be the wrong trade.
pub(crate) fn record_trained_context(
    dir: &Path,
    trained: Option<u32>,
    progress: &mut impl FnMut(&str),
) {
    match trained {
        Some(n) => match repack::trained_context_meta::record(dir, n) {
            Ok(()) => progress(&format!("trained context {n} recorded")),
            Err(e) => progress(&format!("could not record the trained context: {e}")),
        },
        None => progress("the checkpoint declares no trained context"),
    }
}

pub(crate) fn stream_mlx(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    progress: &mut impl FnMut(&str),
    byte_progress: Option<&ByteProgressCallback>,
) -> Result<ArchConfig, String> {
    let config_text = String::from_utf8(client.get(&plan.weights.file_url("config.json"))?)
        .map_err(|e| format!("config.json is not UTF-8: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&config_text).map_err(|e| format!("parsing config.json: {e}"))?;
    let family = repack::config_json_family(&config)
        .ok_or_else(|| "config.json declares no model_type this port recognizes".to_string())?;

    let arch = match family {
        ModelFamily::Gemma4 => repack::parse_gemma4_config(&config_text).map_err(|e| e.to_string()),
        ModelFamily::QwenGdnMoe => {
            repack::parse_qwen_gdn_moe_config(&config_text).map_err(|e| e.to_string())
        }
        ModelFamily::QwenGdnDense => {
            repack::parse_qwen_gdn_dense_config(&config_text).map_err(|e| e.to_string())
        }
        ModelFamily::MuseGlimmer => {
            repack::parse_muse_glimmer_config(&config_text).map_err(|e| e.to_string())
        }
        other => Err(format!("{} has no safetensors intake here", other.as_str())),
    }?;
    let quant = repack::parse_gemma4_quantization(&config_text)
        .map_err(|e| format!("parsing the quantization block: {e}"))?;

    // The shard list. A single-file checkpoint may still ship an index, so
    // the index is consulted first and `model.safetensors` is the fallback
    // rather than the other way round.
    let shard_names = shard_names(plan, client)?;
    progress(&format!(
        "{} shard(s), family {}, {}-bit affine at group {}",
        shard_names.len(),
        arch.family.as_str(),
        quant.default_bits,
        quant.group_size
    ));

    let sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| {
            let url = plan.weights.file_url(name);
            match byte_progress {
                Some(cb) => HttpRangeSource::with_progress(url, Arc::clone(cb)),
                None => HttpRangeSource::new(url),
            }
        })
        .collect();
    let headers = sources
        .iter()
        .map(|s| repack::fetch_safetensors_header(s).map_err(|e| format!("shard header: {e}")))
        .collect::<Result<Vec<_>, _>>()?;
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    );

    let model_id = plan.weights.repo.clone();
    let report = |stage: &str| progress(&format!("[repack] {stage}"));
    match family {
        ModelFamily::Gemma4 => {
            repack::write_gemma4_install_streamed(dir, &arch, &model_id, &shards, &quant, report)
        }
        ModelFamily::QwenGdnMoe => repack::write_qwen_gdn_moe_install_streamed(
            dir, &arch, &model_id, &shards, &quant, report,
        ),
        ModelFamily::QwenGdnDense => repack::write_qwen_gdn_dense_install_streamed(
            dir, &arch, &model_id, &shards, &quant, report,
        ),
        // MISSING THIS ARM IS THE WORST OF THE THREE (`docs/NEW_MODEL.md`
        // Phase 7): the probe would say RUNNABLE, the sidecars would fetch
        // and verify, and it would die at the top of the stream.
        ModelFamily::MuseGlimmer => repack::write_muse_glimmer_install_streamed(
            dir, &arch, &model_id, &shards, &quant, report,
        ),
        other => Err(Box::<dyn std::error::Error>::from(format!(
            "{} has no safetensors writer here",
            other.as_str()
        ))),
    }
    .map_err(|e| format!("streaming {}: {e}", plan.weights))?;
    record_trained_context(
        dir,
        repack::trained_context_meta::from_config_json(&config_text),
        progress,
    );
    Ok(arch)
}

/// Shard filenames, from the index where there is one.
pub(crate) fn shard_names(plan: &InstallPlan, client: &Client) -> Result<Vec<String>, String> {
    let index_url = plan.weights.file_url("model.safetensors.index.json");
    if let Some(bytes) = client.get_optional(&index_url)? {
        let index: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("parsing the shard index: {e}"))?;
        let map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| "the shard index has no weight_map".to_string())?;
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for value in map.values() {
            if let Some(name) = value.as_str() {
                names.insert(name.to_string());
            }
        }
        if !names.is_empty() {
            return Ok(names.into_iter().collect());
        }
    }
    Ok(vec!["model.safetensors".to_string()])
}
