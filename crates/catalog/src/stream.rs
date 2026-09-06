use std::path::Path;
use std::sync::Arc;

use model_io::{ArchConfig, ModelFamily};
use repack::{ByteProgressCallback, Gemma4Shards, HttpRangeSource, RangeSource, SafetensorsHeader};

use crate::hf::{Client, RepoRef};
use crate::install::InstallPlan;

pub(crate) fn stream_gguf(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
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
    }
    .with_optional_token(client.token());
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
        ModelFamily::Qwen4Exp => {
            repack::parse_qwen4_exp_config(&config_text).map_err(|e| e.to_string())
        }
        other => Err(format!("{} has no safetensors intake here", other.as_str())),
    }?;
    let quant = repack::parse_gemma4_quantization(&config_text)
        .map_err(|e| format!("parsing the quantization block: {e}"))?;

    // Reuse an EXISTING install's trunk bytes rather than re-streaming them:
    // only the head crosses the network here. `--reuse-trunk-from` (the CLI
    // layer) has already checked the named install's recorded repo and
    // revision match `plan.weights` exactly before setting this.
    if let Some(existing_dir) = &plan.reuse_trunk_from {
        let mtp = plan.mtp.as_ref().ok_or_else(|| {
            "reuse_trunk_from is set but this row names no mtp source to graft".to_string()
        })?;
        if family != ModelFamily::QwenGdnDense {
            return Err(format!(
                "{}: reusing an existing trunk is only wired for the qwen35 dense family, \
                 got {}",
                plan.weights,
                family.as_str()
            ));
        }
        progress(&format!(
            "reusing the trunk already installed at {}",
            existing_dir.display()
        ));
        let head_pairs = fetch_mtp_shards(mtp, client, byte_progress)?;
        let mtp_base_names: Vec<String> = head_pairs
            .iter()
            .flat_map(|(h, _)| h.tensors.keys().cloned())
            .collect();
        let mtp_bases: Vec<&str> = mtp_base_names.iter().map(String::as_str).collect();
        let head_shards_input: Vec<(&SafetensorsHeader, &dyn RangeSource)> = head_pairs
            .iter()
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect();
        let head_shards = Gemma4Shards::new(head_shards_input);
        let model_id = plan.weights.repo.clone();
        repack::graft_qwen_gdn_dense_mtp_head(
            existing_dir,
            dir,
            &arch,
            &model_id,
            &head_shards,
            &mtp_bases,
            &quant,
            |stage: &str| progress(&format!("[repack] {stage}")),
        )
        .map_err(|e| format!("grafting onto {}: {e}", existing_dir.display()))?;
        record_trained_context(
            dir,
            repack::trained_context_meta::from_config_json(&config_text),
            progress,
        );
        return Ok(arch);
    }

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

    let mut sources: Vec<HttpRangeSource> = shard_names
        .iter()
        .map(|name| {
            let url = plan.weights.file_url(name);
            match byte_progress {
                Some(cb) => HttpRangeSource::with_progress(url, Arc::clone(cb)),
                None => HttpRangeSource::new(url),
            }
            .with_optional_token(client.token())
        })
        .collect();
    let mut headers = sources
        .iter()
        .map(|s| repack::fetch_safetensors_header(s).map_err(|e| format!("shard header: {e}")))
        .collect::<Result<Vec<_>, _>>()?;

    // A separate repository carrying a multi-token-prediction head this
    // artifact's own conversion drops (`docs/MTP_SPECULATIVE.md` step 1).
    // Its shard(s) join the same multi-shard registry the trunk uses, so the
    // rest of the walk -- classification, quantization, the writer -- sees
    // one merged tensor namespace and needs no changes.
    if let Some(mtp) = &plan.mtp {
        if headers
            .iter()
            .any(|h| h.tensors.keys().any(|k| k.starts_with(repack::MTP_PREFIX)))
        {
            return Err(format!(
                "{}: the trunk already carries an MTP head; remove the catalog \
                 row's mtp source",
                plan.weights
            ));
        }
        progress(&format!(
            "fetching the multi-token-prediction head from {mtp}"
        ));
        for (header, source) in fetch_mtp_shards(mtp, client, byte_progress)? {
            headers.push(header);
            sources.push(source);
        }
    }

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
        ModelFamily::Qwen4Exp => {
            repack::write_qwen4_exp_install_streamed(dir, &arch, &model_id, &shards, &quant, report)
        }
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

/// The multi-token-prediction head's shard(s), read from a repository
/// SEPARATE from the trunk and filtered down to its `mtp.`-prefixed tensors
/// alone.
///
/// The filter matters: `Qwen/Qwen3.8-27B`'s last shard also carries a bare
/// BF16 `lm_head.weight` this port already has, quantized, under the trunk's
/// own `language_model.lm_head.weight` name. Left in, `Gemma4Shards`' merged
/// registry would offer it under a name `classify_for_family` does not
/// recognize and the walk would refuse the whole install by name, most of an
/// hour into the trunk's own stream.
fn fetch_mtp_shards(
    mtp: &RepoRef,
    client: &Client,
    byte_progress: Option<&ByteProgressCallback>,
) -> Result<Vec<(SafetensorsHeader, HttpRangeSource)>, String> {
    let index_url = mtp.file_url("model.safetensors.index.json");
    let index_bytes = client
        .get(&index_url)
        .map_err(|e| format!("fetching {mtp}'s shard index: {e}"))?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes)
        .map_err(|e| format!("parsing {mtp}'s shard index: {e}"))?;
    let map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| format!("{mtp}'s shard index has no weight_map"))?;

    let mut mtp_shard_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (name, shard) in map {
        if name.starts_with(repack::MTP_PREFIX) {
            if let Some(shard) = shard.as_str() {
                mtp_shard_names.insert(shard.to_string());
            }
        }
    }
    if mtp_shard_names.is_empty() {
        return Err(format!(
            "{mtp} declares no mtp.* tensor; the catalog row's mtp source is wrong"
        ));
    }

    let mut out = Vec::with_capacity(mtp_shard_names.len());
    for shard_name in mtp_shard_names {
        let url = mtp.file_url(&shard_name);
        let source = match byte_progress {
            Some(cb) => HttpRangeSource::with_progress(url, Arc::clone(cb)),
            None => HttpRangeSource::new(url),
        }
        .with_optional_token(client.token());
        let mut header = repack::fetch_safetensors_header(&source)
            .map_err(|e| format!("{mtp}/{shard_name} header: {e}"))?;
        header
            .tensors
            .retain(|name, _| name.starts_with(repack::MTP_PREFIX));
        out.push((header, source));
    }
    Ok(out)
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
