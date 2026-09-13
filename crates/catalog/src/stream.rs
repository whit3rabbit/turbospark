use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use model_io::{ArchConfig, ModelFamily, VisionConfig};
use repack::{
    ByteProgressCallback, CancelFlag, Gemma4Shards, HttpRangeSource, RangeSource, SafetensorsHeader,
};

use crate::hf::{Client, RepoRef};
use crate::install::{InstallPlan, INSTALL_CANCELLED};

/// The one place every `HttpRangeSource` of a walk is built, so the cancel
/// flag is attached to each of them exactly once. `None` is the CLI's
/// unstoppable shape, byte-for-byte as before.
pub(crate) fn range_source(
    url: String,
    byte_progress: Option<&ByteProgressCallback>,
    client: &Client,
    cancel: Option<&CancelFlag>,
) -> HttpRangeSource {
    let source = match byte_progress {
        Some(cb) => HttpRangeSource::with_progress(url, Arc::clone(cb)),
        None => HttpRangeSource::new(url),
    }
    .with_optional_token(client.token());
    match cancel {
        Some(flag) => source.with_cancel(flag.clone()),
        None => source,
    }
}

/// Step-boundary cancel check for the whole-file GETs (`config.json`, a
/// shard index) that bypass `HttpRangeSource` and so cannot see the flag
/// mid-download. Those files are KB-scale, so a boundary check is the
/// right granularity for them.
fn check_cancelled(cancel: Option<&CancelFlag>) -> Result<(), String> {
    match cancel {
        Some(flag) if flag.is_cancelled() => Err(INSTALL_CANCELLED.to_string()),
        _ => Ok(()),
    }
}

pub(crate) fn stream_gguf(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
    progress: &mut impl FnMut(&str),
    byte_progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
) -> Result<ArchConfig, String> {
    let file = plan
        .file
        .as_deref()
        .ok_or_else(|| "a gguf install needs a filename".to_string())?;
    let source = crate::gguf_source::load(client, &plan.weights, file, byte_progress, cancel)?;
    let header = &source.header;
    if header.architecture() == Some("minimax-m2") {
        let size = repack::minimax_gguf_sizing(header).map_err(|e| e.to_string())?;
        progress(&format!("MiniMax storage preflight: download {}, install allowance {}, resident {}, eight slots {}, FP16 KV at 8192 {} bytes", source.bytes, size.install_bytes(), size.resident_bytes, size.eight_slot_bytes, size.kv_8192_bytes));
    }
    let model_id = plan.weights.repo.clone();
    let arch = repack::write_gguf_install_streamed(dir, header, &source, &model_id, |stage| {
        progress(&format!("[repack] {stage}"))
    })
    .map_err(|e| format!("streaming {file}: {e}"))?;
    record_trained_context(
        dir,
        repack::trained_context_meta::from_gguf(header),
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
    cancel: Option<&CancelFlag>,
) -> Result<ArchConfig, String> {
    check_cancelled(cancel)?;
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
        ModelFamily::Qwen2Dense => {
            repack::parse_qwen2_config(&config_text).map_err(|e| e.to_string())
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
        let head_pairs = fetch_mtp_shards(mtp, client, byte_progress, cancel)?;
        let mtp_base_names: Vec<String> = head_pairs
            .iter()
            .flat_map(|(h, _)| h.tensors.keys().cloned())
            .collect();
        let mtp_bases: Vec<&str> = mtp_base_names.iter().map(String::as_str).collect();
        let head_shards_input: Vec<(&SafetensorsHeader, &dyn RangeSource)> = head_pairs
            .iter()
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect();
        let head_shards = Gemma4Shards::new(head_shards_input)
            .map_err(|e| format!("multi-token-prediction head shards: {e}"))?;
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
            range_source(url, byte_progress, client, cancel)
        })
        .collect();
    let mut headers = sources
        .iter()
        .map(|s| repack::fetch_safetensors_header(s).map_err(|e| format!("shard header: {e}")))
        .collect::<Result<Vec<_>, _>>()?;

    if family == ModelFamily::Qwen2Dense {
        for header in &mut headers {
            repack::canonicalize_qwen2_header(header).map_err(|e| e.to_string())?;
        }
    }

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
        for (header, source) in fetch_mtp_shards(mtp, client, byte_progress, cancel)? {
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
    )
    .map_err(|e| format!("checkpoint shards: {e}"))?;

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
        ModelFamily::Qwen2Dense => repack::write_qwen2_dense_install_streamed(
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
///
/// A thin wrapper over [`fetch_prefixed_shards`], so every existing caller
/// and test keeps this exact return shape (no shard filename, which nothing
/// here has ever needed) while the general form gains a second caller in
/// [`stream_vision_sidecar`].
fn fetch_mtp_shards(
    mtp: &RepoRef,
    client: &Client,
    byte_progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
) -> Result<Vec<(SafetensorsHeader, HttpRangeSource)>, String> {
    let shards = fetch_prefixed_shards(
        mtp,
        &[repack::MTP_PREFIX],
        None,
        client,
        byte_progress,
        cancel,
    )
    .map_err(|e| {
        // Preserve the original wording for the "nothing matched" case,
        // which named the head explicitly rather than a generic prefix
        // list -- nothing tests the string today, but a caller reading a
        // failed `pull --reuse-trunk-from` error deserves the specific one.
        if e.contains("declares no tensor under") {
            format!("{mtp} declares no mtp.* tensor; the catalog row's mtp source is wrong")
        } else {
            e
        }
    })?;
    Ok(shards.into_iter().map(|(_name, h, s)| (h, s)).collect())
}

/// Which shard file(s) in a safetensors index carry at least one tensor whose
/// name starts with one of `prefixes`.
///
/// Pure and split out from the network fetch so it is testable with a
/// constructed JSON value and no HTTP round trip -- the same split
/// `crate::probe`'s `evaluate_gguf`/`evaluate_config` already make between a
/// gate's decision and its fetch.
fn shard_names_for_prefixes(
    index: &serde_json::Value,
    prefixes: &[&str],
) -> Result<BTreeSet<String>, String> {
    let map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| "shard index has no weight_map".to_string())?;
    let mut names = BTreeSet::new();
    for (tensor, shard) in map {
        if prefixes.iter().any(|p| tensor.starts_with(p)) {
            if let Some(shard) = shard.as_str() {
                names.insert(shard.to_string());
            }
        }
    }
    Ok(names)
}

/// Shard(s) of `repo` carrying at least one tensor under any of `prefixes`,
/// each header filtered down to just those tensors.
///
/// Reads `repo`'s safetensors shard index when it has one (a repository too
/// small to shard, or one this port reads only for a sub-component such as a
/// vision tower, may have none at all) and falls back to `explicit_file` when
/// the index is absent OR present but names nothing under `prefixes` -- the
/// same "index first, explicit name as the fallback" order [`shard_names`]
/// already uses for the trunk's own shard list. Refuses only when NEITHER
/// source names a candidate, so a caller with no index and no `--file` gets a
/// message naming the gap rather than an empty result.
pub(crate) fn fetch_prefixed_shards(
    repo: &RepoRef,
    prefixes: &[&str],
    explicit_file: Option<&str>,
    client: &Client,
    byte_progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
) -> Result<Vec<(String, SafetensorsHeader, HttpRangeSource)>, String> {
    check_cancelled(cancel)?;
    let index_url = repo.file_url("model.safetensors.index.json");
    let indexed = client.get_optional(&index_url)?;
    let mut shard_names: BTreeSet<String> = match &indexed {
        Some(bytes) => {
            let index: serde_json::Value = serde_json::from_slice(bytes)
                .map_err(|e| format!("parsing {repo}'s shard index: {e}"))?;
            shard_names_for_prefixes(&index, prefixes).map_err(|e| format!("{repo}: {e}"))?
        }
        None => BTreeSet::new(),
    };
    if shard_names.is_empty() {
        match explicit_file {
            Some(file) => {
                shard_names.insert(file.to_string());
            }
            None => {
                return Err(format!(
                    "{repo} declares no tensor under {prefixes:?} {}; pass an explicit file",
                    if indexed.is_some() {
                        "in its shard index"
                    } else {
                        "and has no shard index"
                    }
                ));
            }
        }
    }

    let mut out = Vec::with_capacity(shard_names.len());
    for shard_name in shard_names {
        check_cancelled(cancel)?;
        let url = repo.file_url(&shard_name);
        let source = range_source(url, byte_progress, client, cancel);
        let mut header = repack::fetch_safetensors_header(&source)
            .map_err(|e| format!("{repo}/{shard_name} header: {e}"))?;
        header
            .tensors
            .retain(|name, _| prefixes.iter().any(|p| name.starts_with(p)));
        if header.tensors.is_empty() {
            return Err(format!(
                "{repo}/{shard_name} carries no tensor under {prefixes:?} after filtering; \
                 wrong shard or wrong prefix list"
            ));
        }
        out.push((shard_name, header, source));
    }
    Ok(out)
}

/// Streams a vision-tower SIDECAR install (vision memory sidecar, part A5):
/// `weights`'s `config.json` decides the family and the tower's own shape,
/// its shard(s) are fetched and canonicalized, and the result is written
/// through [`repack::write_vision_sidecar`] into `out_dir`.
///
/// **Fetches its own `config.json`, independent of whatever sidecar files
/// `install()` wrote into `out_dir`**, the same way [`stream_mlx`] fetches
/// its own copy rather than reading one a caller may have written first: the
/// two `stream_*` functions are meant to be usable on their own, and a
/// vision-only install additionally keeps `config.json` on disk afterwards
/// (`crate::install::InstallPlan::vision_only`), which is a property of the
/// INSTALL rather than of this function's own needs.
///
/// Refuses a repository declaring no vision tower at all, and refuses one
/// whose trunk `hidden_size` disagrees with its own `vision_config`'s
/// `out_hidden_size` -- architecturally the two are the same number (the
/// merger writes straight into the trunk's residual stream), so a mismatch
/// is a checkpoint this walk has never seen rather than a value to silently
/// prefer one side of.
pub(crate) fn stream_vision_sidecar(
    weights: &RepoRef,
    out_dir: &Path,
    explicit_file: Option<&str>,
    client: &Client,
    progress: &mut impl FnMut(&str),
    byte_progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
) -> Result<VisionConfig, String> {
    check_cancelled(cancel)?;
    let config_text = String::from_utf8(client.get(&weights.file_url("config.json"))?)
        .map_err(|e| format!("config.json is not UTF-8: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&config_text).map_err(|e| format!("parsing config.json: {e}"))?;

    // Family and hidden_size, off the SAME text-config-unwrapping logic every
    // family parser already carries -- reused rather than re-derived, so a
    // future family's `text_config` convention change cannot drift the two
    // apart. Only `hidden_size` is used, as a cross-check on the tower's own
    // declared output width; the rest of the arch this returns describes the
    // TRUNK, which this walk does not install.
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
        other => Err(format!(
            "{} has no safetensors intake here, so its vision tower cannot be read either",
            other.as_str()
        )),
    }?;

    let vision = repack::parse_vision_config(&config_text).map_err(|e| e.to_string())?;
    if !vision.is_active() {
        return Err(format!(
            "{weights} declares no vision_config; there is no tower here to install"
        ));
    }
    if arch.hidden_size != vision.out_hidden_size {
        return Err(format!(
            "{weights}: trunk hidden_size {} does not match vision_config.out_hidden_size {}; \
             refusing an inconsistent checkpoint",
            arch.hidden_size, vision.out_hidden_size
        ));
    }

    progress(&format!(
        "{weights}: family {}, tower depth {}, out_hidden_size {}",
        family.as_str(),
        vision.depth,
        vision.out_hidden_size
    ));

    let fetched = fetch_prefixed_shards(
        weights,
        &repack::VISION_SOURCE_PREFIXES,
        explicit_file,
        client,
        byte_progress,
        cancel,
    )?;
    progress(&format!("{} vision shard(s) fetched", fetched.len()));

    let mut shard_names = Vec::with_capacity(fetched.len());
    let mut headers = Vec::with_capacity(fetched.len());
    let mut sources = Vec::with_capacity(fetched.len());
    for (name, mut header, source) in fetched {
        repack::canonicalize_vision_header(&mut header).map_err(|e| e.to_string())?;
        shard_names.push(name);
        headers.push(header);
        sources.push(source);
    }

    let vision_bases: Vec<&str> = headers
        .iter()
        .flat_map(|h| h.tensors.keys())
        .filter(|k| k.starts_with(repack::VISION_PREFIX))
        .map(String::as_str)
        .collect();
    if vision_bases.is_empty() {
        return Err(format!(
            "{weights}: fetched shard(s) carry no {:?}-prefixed tensor after canonicalization",
            repack::VISION_PREFIX
        ));
    }

    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    )
    .map_err(|e| format!("vision tower shards: {e}"))?;
    let read = repack::read_vision_entries(&shards, &vision_bases, &vision)
        .map_err(|e| format!("reading the vision tower: {e}"))?;

    // Reported the same way `write_gemma4_install_streamed` reports the
    // combined-install tower's cost (`crates/repack` Gotcha 13): a nonzero
    // count here means values in FP16's subnormal range, never a wholesale
    // precision loss.
    let lossy: usize = read.lossy_conversion.iter().map(|(_, n)| n).sum();
    if lossy > 0 {
        progress(&format!(
            "converted {} vision tensors to FP16 with {lossy} values losing bits \
             (subnormals; the normal range is exact)",
            read.lossy_conversion.len()
        ));
    }

    let record = model_io::SidecarRecord {
        kind: model_io::SIDECAR_KIND.to_string(),
        pairs_with: model_io::PairsWith {
            family: family.as_str().to_string(),
            hidden_size: vision.out_hidden_size,
        },
        source: model_io::SidecarSource {
            repo: weights.repo.clone(),
            revision: weights.revision.clone(),
            prefix: repack::VISION_PREFIX.to_string(),
            file: shard_names.join(", "),
        },
        tower_blocks: vision.depth,
        block_stride: read.block_stride,
    };

    repack::write_vision_sidecar(out_dir, family, &vision, &weights.repo, &read, record)
        .map_err(|e| format!("writing the vision sidecar: {e}"))?;
    progress(&format!(
        "vision sidecar written: {} block(s), {} bytes/block",
        vision.depth, read.block_stride
    ));

    Ok(vision)
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

#[cfg(test)]
mod tests {
    use super::shard_names_for_prefixes;

    /// The pure half of [`super::fetch_prefixed_shards`]: given an index, find
    /// the shard(s) carrying at least one tensor under any of the prefixes,
    /// with no HTTP round trip.
    #[test]
    fn shard_names_for_prefixes_finds_matching_shards_across_the_map() {
        let index = serde_json::json!({
            "weight_map": {
                "language_model.model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "vision_tower.blocks.0.norm1.weight": "model-00002-of-00003.safetensors",
                "vision_tower.merger.norm.weight": "model-00003-of-00003.safetensors",
                "mtp.fc.weight": "model-00003-of-00003.safetensors",
            }
        });
        let names = shard_names_for_prefixes(&index, &["vision_tower.", "model.visual."]).unwrap();
        assert_eq!(
            names,
            [
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors"
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        );
    }

    #[test]
    fn shard_names_for_prefixes_matches_the_hf_native_spelling_too() {
        let index = serde_json::json!({
            "weight_map": {
                "model.visual.blocks.0.norm1.weight": "shard-a.safetensors",
                "model.language_model.embed_tokens.weight": "shard-b.safetensors",
            }
        });
        let names = shard_names_for_prefixes(&index, &["vision_tower.", "model.visual."]).unwrap();
        assert_eq!(
            names,
            ["shard-a.safetensors"]
                .into_iter()
                .map(str::to_string)
                .collect()
        );
    }

    #[test]
    fn shard_names_for_prefixes_is_empty_when_nothing_matches() {
        let index = serde_json::json!({
            "weight_map": {
                "language_model.model.embed_tokens.weight": "shard-a.safetensors",
            }
        });
        let names = shard_names_for_prefixes(&index, &["vision_tower.", "model.visual."]).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn shard_names_for_prefixes_refuses_a_map_with_no_weight_map() {
        let index = serde_json::json!({"not_weight_map": {}});
        let err = shard_names_for_prefixes(&index, &["vision_tower."]).unwrap_err();
        assert!(err.contains("weight_map"), "{err}");
    }
}
