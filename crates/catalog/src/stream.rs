use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
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
    cache_dir: Option<&Path>,
) -> HttpRangeSource {
    let mut source = match byte_progress {
        Some(cb) => HttpRangeSource::with_progress(url, Arc::clone(cb)),
        None => HttpRangeSource::new(url),
    }
    .with_optional_token(client.token());
    if let Some(cache_dir) = cache_dir {
        source = source.with_cache_dir(cache_dir);
    }
    match cancel {
        Some(flag) => source.with_cancel(flag.clone()),
        None => source,
    }
}

/// Immutable repository URLs can safely retain completed network ranges
/// between failed walks. Floating branches must never reuse cached bytes.
pub(crate) fn immutable_download_cache(dir: &Path, repo: &RepoRef) -> Option<PathBuf> {
    (repo.revision.len() == 40 && repo.revision.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| dir.join(".download-cache"))
}

fn download_cache_for_install(dir: &Path, repo: &RepoRef, disabled: bool) -> Option<PathBuf> {
    if disabled {
        None
    } else {
        immutable_download_cache(dir, repo)
    }
}

fn download_cache_disabled() -> bool {
    std::env::var("TURBOSPARK_DISABLE_DOWNLOAD_CACHE").as_deref() == Ok("1")
}

/// Step-boundary cancel check for the whole-file GETs (`config.json`, a
/// shard index) that bypass `HttpRangeSource` and so cannot see the flag
/// mid-download. Those files are KB-scale, so a boundary check is the
/// right granularity for them.
fn check_cancelled(cancel: Option<&CancelFlag>) -> Result<(), String> {
    match cancel {
        Some(flag) if flag.checkpoint().is_err() => Err(INSTALL_CANCELLED.to_string()),
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
    let cache = download_cache_for_install(dir, &plan.weights, download_cache_disabled());
    if download_cache_disabled() {
        progress(
            "persistent range cache disabled; an interrupted download will fetch the source again",
        );
    }
    let source = crate::gguf_source::load(
        client,
        &plan.weights,
        file,
        byte_progress,
        cancel,
        cache.as_deref(),
    )?;
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
    let cache = download_cache_for_install(dir, &plan.weights, download_cache_disabled());
    check_cancelled(cancel)?;
    let config_text = String::from_utf8(client.get(&plan.weights.file_url("config.json"))?)
        .map_err(|e| format!("config.json is not UTF-8: {e}"))?;
    let config: serde_json::Value =
        serde_json::from_str(&config_text).map_err(|e| format!("parsing config.json: {e}"))?;
    let family = repack::config_json_family(&config)
        .ok_or_else(|| "config.json declares no model_type this port recognizes".to_string())?;

    let mut arch = match family {
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
        ModelFamily::Qwen3Vl => {
            repack::parse_qwen3_vl_config(&config_text).map_err(|e| e.to_string())
        }
        other => Err(format!("{} has no safetensors intake here", other.as_str())),
    }?;
    enable_requested_vision(plan, family, &mut arch, &config_text)?;
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
        let head_pairs = fetch_mtp_shards(mtp, client, byte_progress, cancel, cache.as_deref())?;
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
            range_source(url, byte_progress, client, cancel, cache.as_deref())
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

    // The HF-native `model.visual.` spelling renames onto `vision_tower.`
    // here, for the same three families `classify_for_family` buckets as
    // `VisionTower` -- the SAME canonicalization `stream_vision_sidecar`
    // already runs, so a combined walk and a sidecar walk cannot disagree
    // about which spellings name a tower. A header carrying BOTH spellings
    // is refused by `canonicalize_vision_header` itself; a text-only
    // checkpoint (Ornith's declare-but-not-ship rule) carries neither and
    // passes through untouched.
    if matches!(
        family,
        ModelFamily::QwenGdnDense | ModelFamily::QwenGdnMoe | ModelFamily::Qwen3Vl
    ) {
        for header in &mut headers {
            repack::canonicalize_vision_header(header).map_err(|e| e.to_string())?;
        }
    }

    // ROADMAP P2.9: a probe-driven pull of a combined VLM checkpoint keeps its
    // tower. `enable_requested_vision` above only fires on catalog-row intent,
    // which a `--repo` pull never has; the shard headers are the other half of
    // the answer and they are in hand by this line.
    enable_bytes_detected_vision(plan, family, &mut arch, &headers, &config_text)?;
    if arch.vision.is_active() {
        install_vision_preprocessor(plan, dir, client)?;
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
        for (header, source) in
            fetch_mtp_shards(mtp, client, byte_progress, cancel, cache.as_deref())?
        {
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
        ModelFamily::QwenGdnDense => {
            // The Hadamard-folded contract (the Bonsai-2 line) rides the same
            // config.json: a config whose modules declare transformed bases
            // routes to the contract-carrying walk, and the folded walk
            // itself refuses a checkpoint with `.signs` tensors that arrived
            // here without one -- so the two entry points cannot be confused
            // silently, whichever way a future row is misdeclared.
            let hadamard = repack::parse_prism_hadamard(&config_text)
                .map_err(|e| format!("parsing the hadamard contract: {e}"))?;
            match hadamard {
                Some(contract) => repack::write_qwen_gdn_dense_install_streamed_with_hadamard(
                    dir, &arch, &model_id, &shards, &quant, &contract, report,
                ),
                None => repack::write_qwen_gdn_dense_install_streamed(
                    dir, &arch, &model_id, &shards, &quant, report,
                ),
            }
        }
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
        ModelFamily::Qwen3Vl => {
            repack::write_qwen3_vl_install_streamed(dir, &arch, &model_id, &shards, &quant, report)
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

/// Apply a catalog row's explicit combined-vision intent to the architecture
/// passed to the streamed writer. The trunk parsers deliberately return a
/// text-only architecture, so this cannot be inferred from `config.json`.
fn enable_requested_vision(
    plan: &InstallPlan,
    family: ModelFamily,
    arch: &mut ArchConfig,
    config_text: &str,
) -> Result<(), String> {
    if !plan.include_vision {
        return Ok(());
    }
    if !matches!(
        family,
        ModelFamily::QwenGdnDense | ModelFamily::QwenGdnMoe | ModelFamily::Qwen3Vl
    ) {
        return Err(format!(
            "{}: combined vision ingestion is only wired for the qwen35 dense, qwen36 MoE              and qwen3_vl trunks",
            plan.alias
        ));
    }
    let vision = repack::parse_vision_config(config_text).map_err(|e| e.to_string())?;
    if !vision.is_active() {
        return Err(format!(
            "{}: catalog row requests vision, but config.json declares no vision_config",
            plan.alias
        ));
    }
    if vision.out_hidden_size != arch.hidden_size {
        return Err(format!(
            "{}: vision output width {} does not match trunk hidden size {}",
            plan.alias, vision.out_hidden_size, arch.hidden_size
        ));
    }
    arch.vision = vision;
    Ok(())
}

/// Detect a combined vision tower from the checkpoint's BYTES and enable it
/// on the architecture passed to the streamed writer (ROADMAP P2.9).
///
/// [`enable_requested_vision`] answers a catalog row's INTENT; this answers
/// what the artifact actually ships. The trunk parsers deliberately return a
/// text-only architecture because `config.json` alone cannot be trusted to:
/// `ornith-ai/Ornith-1.5-35B-A3B` declares a tower in its config and ships no
/// `vision_tower.` tensors, so the config gate must stay insufficient. The
/// shard headers, fetched by the time [`stream_mlx`] reaches this call, are
/// the bytes half of that rule -- and a checkpoint that ships tower tensors
/// has a tower, whatever a sibling repo's config does.
///
/// Detection matches [`repack::VISION_PREFIX`] alone, the exact string
/// `classify_for_family` buckets as `VisionTower` for these families. The
/// HF-native `model.visual.` spelling stays a sidecar-path concern, as its
/// classify comment records: the combined walk has never read one, and
/// enabling a tower the walk would then drop (or refuse on) helps nobody.
fn enable_bytes_detected_vision(
    plan: &InstallPlan,
    family: ModelFamily,
    arch: &mut ArchConfig,
    headers: &[SafetensorsHeader],
    config_text: &str,
) -> Result<(), String> {
    if arch.vision.is_active() {
        // Catalog intent already decided, and its parse + width check ran.
        return Ok(());
    }
    // The same family gate `classify_for_family` applies to `VisionTower`:
    // both halves of the shared qwen3_5 architecture and nothing else (an
    // enabled tower under a family that buckets the prefix as
    // `ExcludedMultimodal` would write a manifest declaring tensors the
    // install does not carry).
    if !matches!(
        family,
        ModelFamily::QwenGdnDense | ModelFamily::QwenGdnMoe | ModelFamily::Qwen3Vl
    ) {
        return Ok(());
    }
    let carries_tower = headers.iter().any(|h| {
        h.tensors
            .keys()
            .any(|k| k.starts_with(repack::VISION_PREFIX))
    });
    if !carries_tower {
        return Ok(());
    }
    // The bytes are present, so a config this port cannot pair with them is
    // an inconsistency in the checkpoint itself, and the walk refuses by name
    // rather than silently dropping a third of the tensors -- which is
    // precisely the behavior this function exists to replace.
    let vision = repack::parse_vision_config(config_text).map_err(|e| {
        format!(
            "{}: the checkpoint ships {} vision_tower tensors, but its config.json does not \
             describe a tower this port supports: {e}",
            plan.weights,
            headers
                .iter()
                .map(|h| h
                    .tensors
                    .keys()
                    .filter(|k| k.starts_with(repack::VISION_PREFIX))
                    .count())
                .sum::<usize>()
        )
    })?;
    if !vision.is_active() {
        return Err(format!(
            "{}: the checkpoint ships vision_tower tensors, but its config.json declares no \
             vision_config they could belong to",
            plan.weights
        ));
    }
    if vision.out_hidden_size != arch.hidden_size {
        return Err(format!(
            "{}: vision output width {} does not match trunk hidden size {}",
            plan.alias, vision.out_hidden_size, arch.hidden_size
        ));
    }
    arch.vision = vision;
    Ok(())
}

/// Fetch the checkpoint's `preprocessor_config.json` into an install whose
/// tower was enabled from BYTES (ROADMAP P2.9 residual).
///
/// A catalog row's `include_vision` intent carries the file in its own
/// sidecar list, fetched before the walk runs; a probe-driven `--repo` pull
/// has no row, and `KNOWN_SIDECARS` is a tokenizer list that has never named
/// the preprocessor config. Without this the install opens, decodes text,
/// and refuses its first image with "this install declares no image
/// preprocessing config" -- a combined install that cannot see was exactly
/// the silent-drop class the bytes detection replaced, one file over.
///
/// **A 404 REFUSES RATHER THAN WARNS.** The bytes detection has already
/// established the tower is real, and `vision_dir()` names this file as the
/// one place every image consumer reads, so an install without it is a
/// booby trap: refuse by name and let the walk's staging directory go away.
/// Any other transport failure says so (an unauthenticated gated repo is the
/// `probe`'s own 401 case, not a missing file).
fn install_vision_preprocessor(
    plan: &InstallPlan,
    dir: &Path,
    client: &Client,
) -> Result<(), String> {
    let fetched = client
        .get_optional(&plan.weights.file_url("preprocessor_config.json"))
        .map_err(|e| {
            format!(
                "fetching preprocessor_config.json from {}: {e}. If this is a gated \
                 repository refusing an unauthenticated request rather than a missing \
                 file, export HF_TOKEN and re-run.",
                plan.weights
            )
        });
    write_vision_preprocessor(
        &plan.weights,
        &dir.join("preprocessor_config.json"),
        fetched,
    )
}

/// The decision half of [`install_vision_preprocessor`], split so the
/// keep-existing, 404-refusal and write paths are testable without a
/// transport. The fetch itself happens unconditionally in the caller: the
/// file is KB-scale, and deciding from the fetch result keeps the
/// exists-check and the bytes in one place.
fn write_vision_preprocessor(
    weights: &crate::hf::RepoRef,
    dest: &Path,
    fetched: Result<Option<Vec<u8>>, String>,
) -> Result<(), String> {
    if dest.is_file() {
        // The catalog-intent path fetched it with the rest of the row's
        // sidecars; keep those bytes rather than second-guessing them.
        return Ok(());
    }
    match fetched {
        Ok(Some(bytes)) => {
            std::fs::write(dest, bytes).map_err(|e| format!("writing {}: {e}", dest.display()))
        }
        Ok(None) => Err(format!(
            "{weights} ships a vision tower but no preprocessor_config.json; this port \
             cannot preprocess images for it, so the combined install is refused"
        )),
        Err(e) => Err(e),
    }
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
    cache_dir: Option<&Path>,
) -> Result<Vec<(SafetensorsHeader, HttpRangeSource)>, String> {
    let shards = fetch_prefixed_shards(
        mtp,
        &[repack::MTP_PREFIX],
        None,
        client,
        byte_progress,
        cancel,
        cache_dir,
    )
    .map_err(|e| {
        // Preserve the head-specific wording for the "nothing matched"
        // cases, which name the head rather than a generic prefix list --
        // nothing tests the string today, but a caller reading a failed
        // `pull --reuse-trunk-from` error deserves the specific one. The
        // second form is the single-file fallback's ("model.safetensors
        // carries no tensor under ... after filtering"), reached when the
        // index named nothing that survived `retain_existing_shard_names`
        // and the consolidated file carries no head either.
        if e.contains("declares no tensor under") || e.contains("carries no tensor under") {
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
/// Keeps the shard names from a repo's index only when the repository still
/// lists ALL of them.
///
/// **A REPO'S SHARD INDEX IS A CLAIM, NOT A FACT** (`docs/QWEN3VL_PHASE0.md`
/// section 0): `mlx-community/Qwen3-VL-4B-Instruct-4bit` ships an index
/// naming two shards from an earlier two-shard upload while the repo as it
/// stands carries one consolidated `model.safetensors` -- so every
/// index-derived name 404s at the first header fetch and the whole pull
/// dies on a file the publisher deleted. The repository's own file listing
/// is what the download URLs are built from, so it is the authority; an
/// index name not in the listing is stale by definition. The index is atomic:
/// if any referenced shard is stale, the whole set is discarded so the caller
/// falls back to its single-file convention rather than installing a partial
/// model from the surviving shards.
fn retain_existing_shard_names(
    repo: &RepoRef,
    client: &Client,
    names: BTreeSet<String>,
) -> Result<BTreeSet<String>, String> {
    if names.is_empty() {
        return Ok(names);
    }
    let files = client.file_list(repo)?;
    Ok(retain_complete_shard_set(names, &files))
}

fn retain_complete_shard_set(names: BTreeSet<String>, files: &[String]) -> BTreeSet<String> {
    let listed: std::collections::HashSet<&str> = files.iter().map(String::as_str).collect();
    if names.iter().all(|name| listed.contains(name.as_str())) {
        names
    } else {
        BTreeSet::new()
    }
}

pub(crate) fn fetch_prefixed_shards(
    repo: &RepoRef,
    prefixes: &[&str],
    explicit_file: Option<&str>,
    client: &Client,
    byte_progress: Option<&ByteProgressCallback>,
    cancel: Option<&CancelFlag>,
    cache_dir: Option<&Path>,
) -> Result<Vec<(String, SafetensorsHeader, HttpRangeSource)>, String> {
    check_cancelled(cancel)?;
    let index_url = repo.file_url("model.safetensors.index.json");
    let indexed = client.get_optional(&index_url)?;
    let mut shard_names: BTreeSet<String> = match &indexed {
        Some(bytes) => {
            let index: serde_json::Value = serde_json::from_slice(bytes)
                .map_err(|e| format!("parsing {repo}'s shard index: {e}"))?;
            let names =
                shard_names_for_prefixes(&index, prefixes).map_err(|e| format!("{repo}: {e}"))?;
            retain_existing_shard_names(repo, client, names)?
        }
        None => BTreeSet::new(),
    };
    if shard_names.is_empty() {
        match explicit_file {
            Some(file) => {
                shard_names.insert(file.to_string());
            }
            None => {
                // The single-file convention, the same fallback
                // [`shard_names`] makes for the trunk: the repo may carry
                // one consolidated `model.safetensors` that NO index names
                // -- because it ships none (`Qwen/Qwen3-VL-4B-Instruct`) or
                // because its index describes an earlier sharding pass and
                // every name it lists was just dropped as stale
                // (`mlx-community/Qwen3-VL-4B-Instruct-4bit`, the same repo
                // `retain_existing_shard_names` was written for). The
                // prefix filter cannot consult the single file's header
                // before this point, so the caller's own
                // "no `VISION_PREFIX` tensor after canonicalization"
                // refusal is what catches a repo whose one file carries no
                // tower.
                shard_names.insert("model.safetensors".to_string());
            }
        }
    }

    let mut out = Vec::with_capacity(shard_names.len());
    for shard_name in shard_names {
        check_cancelled(cancel)?;
        let url = repo.file_url(&shard_name);
        let source = range_source(url, byte_progress, client, cancel, cache_dir);
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
    let cache = download_cache_for_install(out_dir, weights, download_cache_disabled());
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
        ModelFamily::Qwen3Vl => {
            repack::parse_qwen3_vl_config(&config_text).map_err(|e| e.to_string())
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
        cache.as_deref(),
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
        // The index is cross-checked against the repo's real file list
        // before any of its names is trusted -- the stale-index 404 this
        // prevents is `retain_existing_shard_names`'s own doc.
        let names = retain_existing_shard_names(&plan.weights, client, names)?;
        if !names.is_empty() {
            return Ok(names.into_iter().collect());
        }
    }
    Ok(vec!["model.safetensors".to_string()])
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        download_cache_for_install, enable_bytes_detected_vision, enable_requested_vision,
        immutable_download_cache, retain_complete_shard_set, shard_names_for_prefixes,
        write_vision_preprocessor,
    };
    use crate::{Catalog, InstallPlan};
    use model_io::ModelFamily;
    use repack::{SafetensorsHeader, TensorInfo, VISION_PREFIX};

    /// The 27-depth tower config every qwen3_5 vision checkpoint carries,
    /// trued up to `qwen_gdn_dense_27b`'s hidden size.
    fn vision_config_text() -> String {
        serde_json::json!({
            "text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}},
            "vision_config": {
                "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
                "num_heads": 16, "patch_size": 16, "temporal_patch_size": 2,
                "in_channels": 3, "spatial_merge_size": 2,
                "num_position_embeddings": 2304, "out_hidden_size": 5120
            },
            "vision_start_token_id": 1, "vision_end_token_id": 2,
            "image_token_id": 3, "video_token_id": 4
        })
        .to_string()
    }

    fn header_with(tensor_names: &[&str]) -> SafetensorsHeader {
        SafetensorsHeader {
            tensors: tensor_names
                .iter()
                .map(|name| {
                    (
                        (*name).to_string(),
                        TensorInfo {
                            dtype: "BF16".to_string(),
                            shape: vec![4, 4],
                            data_offsets: (0, 32),
                        },
                    )
                })
                .collect(),
            metadata: None,
            header_len: 8,
        }
    }

    fn probe_plan() -> InstallPlan {
        // A probe-driven plan is the shape that has no catalog intent to lean
        // on: `include_vision` is false by construction, exactly as
        // `InstallPlan::from_probe` builds it.
        InstallPlan {
            alias: "probe-vlm".to_string(),
            weights: crate::hf::RepoRef::new("example/vlm", "main"),
            file: None,
            kind: crate::entry::SourceKind::Mlx,
            sidecars: crate::hf::RepoRef::new("example/vlm", "main"),
            sidecar_files: Vec::new(),
            include_vision: false,
            install_bytes: 0,
            status: "unlisted".to_string(),
            mtp: None,
            reuse_trunk_from: None,
            vision_only: false,
            vision_file: None,
        }
    }

    #[test]
    fn only_immutable_revisions_receive_a_durable_range_cache() {
        let root = std::path::Path::new("/tmp/install");
        let pinned =
            crate::hf::RepoRef::new("example/model", "0123456789abcdef0123456789abcdef01234567");
        assert_eq!(
            immutable_download_cache(root, &pinned),
            Some(root.join(".download-cache"))
        );
        let floating = crate::hf::RepoRef::new("example/model", "main");
        assert_eq!(immutable_download_cache(root, &floating), None);
    }

    #[test]
    fn disk_constrained_install_can_skip_the_persistent_range_cache() {
        let root = std::path::Path::new("/tmp/install");
        let pinned =
            crate::hf::RepoRef::new("example/model", "0123456789abcdef0123456789abcdef01234567");
        assert_eq!(
            download_cache_for_install(root, &pinned, false),
            Some(root.join(".download-cache"))
        );
        assert_eq!(download_cache_for_install(root, &pinned, true), None);
    }

    #[test]
    fn probe_driven_bytes_carrying_the_tower_enable_combined_vision() {
        let plan = probe_plan();
        let mut arch = model_io::qwen_gdn_dense_27b();
        assert!(!arch.vision.is_active());
        let headers = [header_with(&[
            "language_model.model.embed_tokens.weight",
            "vision_tower.blocks.0.norm1.weight",
        ])];
        enable_bytes_detected_vision(
            &plan,
            ModelFamily::QwenGdnDense,
            &mut arch,
            &headers,
            &vision_config_text(),
        )
        .unwrap();
        assert!(arch.vision.is_active(), "tower bytes must enable the tower");
        assert_eq!(arch.vision.depth, 27);
    }

    /// The Ornith rule, held at the new layer: a config that DECLARES a tower
    /// is not a checkpoint that SHIPS one, and the gate answers to bytes.
    #[test]
    fn a_declared_config_without_tower_bytes_stays_text_only() {
        let plan = probe_plan();
        let mut arch = model_io::qwen_gdn_dense_27b();
        let headers = [header_with(&["language_model.model.embed_tokens.weight"])];
        enable_bytes_detected_vision(
            &plan,
            ModelFamily::QwenGdnDense,
            &mut arch,
            &headers,
            &vision_config_text(),
        )
        .unwrap();
        assert!(!arch.vision.is_active());
    }

    #[test]
    fn tower_bytes_without_a_parseable_tower_config_refuse_by_name() {
        let plan = probe_plan();
        let mut arch = model_io::qwen_gdn_dense_27b();
        let headers = [header_with(&[&format!(
            "{VISION_PREFIX}blocks.0.norm1.weight"
        )])];
        let err = enable_bytes_detected_vision(
            &plan,
            ModelFamily::QwenGdnDense,
            &mut arch,
            &headers,
            "{\"text_config\": {}}",
        )
        .unwrap_err();
        assert!(err.contains("vision_tower"), "{err}");
        assert!(!arch.vision.is_active());
    }

    #[test]
    fn tower_bytes_with_a_mismatched_hidden_size_refuse_by_name() {
        let plan = probe_plan();
        let mut arch = model_io::qwen_gdn_dense_27b();
        let headers = [header_with(&[&format!(
            "{VISION_PREFIX}blocks.0.norm1.weight"
        )])];
        let mismatched = serde_json::json!({
            "text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}},
            "vision_config": {
                "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
                "num_heads": 16, "patch_size": 16, "temporal_patch_size": 2,
                "in_channels": 3, "spatial_merge_size": 2,
                "num_position_embeddings": 2304, "out_hidden_size": 3584
            },
            "vision_start_token_id": 1, "vision_end_token_id": 2,
            "image_token_id": 3, "video_token_id": 4
        })
        .to_string();
        let err = enable_bytes_detected_vision(
            &plan,
            ModelFamily::QwenGdnDense,
            &mut arch,
            &headers,
            &mismatched,
        )
        .unwrap_err();
        assert!(err.contains("does not match trunk hidden size"), "{err}");
    }

    /// Catalog intent keeps its precedence: the row's own parse + width check
    /// already ran by the time bytes detection is reached, and this function
    /// must not second-guess or double-apply it.
    #[test]
    fn bytes_detection_is_a_no_op_when_catalog_intent_already_set_the_tower() {
        let catalog = Catalog::embedded().unwrap();
        let entry = catalog.get("qwen38-27b-vision").unwrap();
        let plan = InstallPlan::from_entry(entry);
        assert!(plan.include_vision);
        let mut arch = model_io::qwen_gdn_dense_27b();
        enable_requested_vision(
            &plan,
            ModelFamily::QwenGdnDense,
            &mut arch,
            &vision_config_text(),
        )
        .unwrap();
        let headers = [header_with(&[
            "language_model.model.embed_tokens.weight",
            "vision_tower.blocks.0.norm1.weight",
        ])];
        enable_bytes_detected_vision(
            &plan,
            ModelFamily::QwenGdnDense,
            &mut arch,
            &headers,
            // A config that could not back the tower the intent path already
            // enabled must not be consulted at all.
            "{\"text_config\": {}}",
        )
        .unwrap();
        assert!(arch.vision.is_active());
    }

    /// The HF-native `model.visual.` spelling reaches the same combined
    /// enablement, through the same canonicalization the sidecar walk runs.
    #[test]
    fn the_hf_native_spelling_enables_combined_vision_after_canonicalization() {
        let plan = probe_plan();
        let mut arch = model_io::qwen_gdn_dense_27b();
        let mut headers = [header_with(&["model.visual.blocks.0.norm1.weight"])];
        for header in &mut headers {
            repack::canonicalize_vision_header(header).unwrap();
        }
        enable_bytes_detected_vision(
            &plan,
            ModelFamily::QwenGdnDense,
            &mut arch,
            &headers,
            &vision_config_text(),
        )
        .unwrap();
        assert!(arch.vision.is_active());
    }

    /// The family gate mirrors `classify_for_family`'s `VisionTower` arm
    /// exactly: both halves of qwen3_5, nothing else (AGENTS.md Gotcha 61).
    #[test]
    fn bytes_detection_ignores_families_that_do_not_classify_the_prefix() {
        let plan = probe_plan();
        for family in [ModelFamily::Gemma4, ModelFamily::Qwen2Dense] {
            let mut arch = model_io::known_architecture(family);
            let headers = [header_with(&[&format!(
                "{VISION_PREFIX}blocks.0.norm1.weight"
            )])];
            enable_bytes_detected_vision(&plan, family, &mut arch, &headers, &vision_config_text())
                .unwrap();
            assert!(
                !arch.vision.is_active(),
                "{} must not enable a tower its walk cannot classify",
                arch.family.as_str()
            );
        }
    }

    /// A bytes-enabled tower lands with its preprocessor config beside the
    /// install, the file `vision_dir()` names as the one place every image
    /// consumer reads.
    #[test]
    fn a_bytes_enabled_tower_writes_the_preprocessor_config() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-preprocessor-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("preprocessor_config.json");
        write_vision_preprocessor(
            &crate::hf::RepoRef::new("example/vlm", "main"),
            &dest,
            Ok(Some(b"{\"patch_size\": 16}".to_vec())),
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"{\"patch_size\": 16}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tower_without_a_preprocessor_config_refuses_by_name() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-preprocessor-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("preprocessor_config.json");
        let err = write_vision_preprocessor(
            &crate::hf::RepoRef::new("example/vlm", "main"),
            &dest,
            Ok(None),
        )
        .unwrap_err();
        assert!(err.contains("preprocessor_config.json"), "{err}");
        assert!(!dest.is_file(), "a refusal must not leave a partial file");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The catalog-intent path fetches the file with the row's own sidecars;
    /// the bytes-path helper must keep those bytes rather than overwrite.
    #[test]
    fn an_existing_preprocessor_config_is_kept() {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-preprocessor-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("preprocessor_config.json");
        std::fs::write(&dest, b"row-fetched").unwrap();
        write_vision_preprocessor(
            &crate::hf::RepoRef::new("example/vlm", "main"),
            &dest,
            Ok(Some(b"bytes-path".to_vec())),
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"row-fetched");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_vision_catalog_row_enables_the_combined_tower() {
        let catalog = Catalog::embedded().unwrap();
        let entry = catalog.get("qwen38-27b-vision").unwrap();
        let plan = InstallPlan::from_entry(entry);
        assert!(plan.include_vision);

        let mut arch = model_io::qwen_gdn_dense_27b();
        assert!(!arch.vision.is_active());
        let config = serde_json::json!({
            "text_config": {"rope_parameters": {"mrope_section": [11, 11, 10]}},
            "vision_config": {
                "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
                "num_heads": 16, "patch_size": 16, "temporal_patch_size": 2,
                "in_channels": 3, "spatial_merge_size": 2,
                "num_position_embeddings": 2304, "out_hidden_size": 5120
            },
            "vision_start_token_id": 1, "vision_end_token_id": 2,
            "image_token_id": 3, "video_token_id": 4
        })
        .to_string();
        enable_requested_vision(&plan, ModelFamily::QwenGdnDense, &mut arch, &config).unwrap();
        assert!(arch.vision.is_active());
        assert_eq!(arch.vision.depth, 27);
    }

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

    #[test]
    fn a_partially_stale_index_discards_every_indexed_shard() {
        let names = ["model-00001.safetensors", "model-00002.safetensors"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let files = vec!["model-00001.safetensors".to_string()];

        assert!(retain_complete_shard_set(names, &files).is_empty());
    }

    #[test]
    fn a_fully_listed_index_keeps_every_indexed_shard() {
        let names: BTreeSet<String> = ["model-00001.safetensors", "model-00002.safetensors"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let files = vec![
            "model-00002.safetensors".to_string(),
            "model-00001.safetensors".to_string(),
        ];

        assert_eq!(retain_complete_shard_set(names.clone(), &files), names);
    }
}
