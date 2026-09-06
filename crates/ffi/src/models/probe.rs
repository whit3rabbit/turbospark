//! Hugging Face repository probing and GGUF variant discovery.

use catalog::{Client, RepoRef, Verdict};
use serde_json::json;

use super::fit::probe_fit;

/// `owner/name` or `owner/name@revision`, defaulting to `main`.
pub(super) fn parse_repo(spec: &str) -> Result<RepoRef, String> {
    let (repo, revision) = match spec.split_once('@') {
        Some((r, rev)) => (r, rev),
        None => (spec, "main"),
    };
    if repo.split('/').count() != 2 || repo.starts_with('/') || repo.ends_with('/') {
        return Err(format!(
            "repository must be owner/name (optionally @revision), got {spec:?}"
        ));
    }
    Ok(RepoRef::new(repo, revision))
}

/// The header-only probe: what this engine makes of an arbitrary repository,
/// reading kilobytes rather than gigabytes.
///
/// `repo` and `sidecarRepo` accept `owner/name` or `owner/name@revision`.
pub(crate) fn probe_json(
    repo: &str,
    file: Option<&str>,
    sidecar_repo: Option<&str>,
    context: Option<u32>,
    slots: model_io::ExpertCacheSlots,
    guard: model_io::LoadGuard,
) -> Result<String, String> {
    let repo = parse_repo(repo)?;
    let sidecars = sidecar_repo.map(parse_repo).transpose()?;
    let report = catalog::probe(&Client::new(), &repo, file, sidecars.as_ref())?;
    // The checkpoint's own trained window, read from the same header the
    // probe already fetched. It is NOT an `ArchConfig` field (Gotcha 55: it
    // is per-checkpoint where that struct is per-architecture), so it is
    // carried beside the fit rather than inside it.
    let trained = report.trained_context;
    let fit = probe_fit(&report, context.unwrap_or(4096), slots, guard, trained);

    // A hand-built projection rather than a derived one. The two omissions
    // are deliberate: the full `ArchConfig` is engine internals a GUI has no
    // use for, and `slot_cache_bytes` is included BECAUSE it is the number
    // that decides whether a model fits at all -- `slots x layers x stride`,
    // not the model's size.
    Ok(json!({
        "repo": repo.repo,
        "revision": repo.revision,
        "file": report.file,
        "downloadBytes": report.download_bytes,
        "architecture": report.architecture,
        "family": report.family.map(|f| f.as_str()),
        "runnable": report.verdict.is_runnable(),
        "refusedBecause": match &report.verdict {
            Verdict::Runnable => None,
            Verdict::Refused(why) => Some(why.clone()),
        },
        "types": report.types.iter().map(|t| json!({
            "name": t.name,
            "tensors": t.tensors,
            // Null rather than zero when this port cannot size the type,
            // which is a much louder statement: an unsized type is usually
            // the one carrying the model.
            "bytes": t.bytes,
            "executable": t.executable,
        })).collect::<Vec<_>>(),
        "affine": report.affine.map(|(bits, group)| json!({"bits": bits, "groupSize": group})),
        "expertStride": report.expert_stride,
        // Null when the file declares none, never 0: every install written
        // before this field existed declares nothing, and a zero would read
        // as a window of zero rather than as an unknown one (Gotcha 55).
        "trainedContext": report.trained_context,
        "slotCacheBytes": report.slot_cache_bytes().iter()
            .map(|(slots, bytes)| json!({"slots": slots, "bytes": bytes}))
            .collect::<Vec<_>>(),
        "sidecarsPresent": report.sidecars_present,
        "sidecarsMissing": report.sidecars_missing,
        "chatTemplate": report.chat_template,
        "warnings": report.warnings,
        // Null when nothing could be read off the header, never a zeroed
        // object: an absent measurement is not a measurement of zero, and a
        // fit of 0 bytes reads as "it fits easily" -- the exact inverse.
        "fit": fit,
    })
    .to_string())
}

/// The `.gguf` files a repository publishes, as JSON, best quality first.
///
/// One Hugging Face API call and no header reads, so a GUI can fill a
/// quantization picker on open. **It carries no fit**, deliberately: a fit
/// needs an `ArchConfig`, which needs a header read PER FILE, so the flow is
/// this listing for the menu and one `ts_probe_json` for the file the user
/// picks.
pub(crate) fn repo_variants_json(repo: &str) -> Result<String, String> {
    let repo = parse_repo(repo)?;
    let found = catalog::gguf_variants(&Client::new(), &repo)?;
    Ok(json!({
        "repo": repo.repo,
        "revision": repo.revision,
        "variants": found.variants.iter().map(|v| json!({
            "file": v.file,
            // Null rather than 0 when Hugging Face reported no length: a
            // 0-byte row sorts and reads as a tiny file, which is the
            // opposite of "unknown".
            "bytes": v.bytes,
            "quantLabel": v.quant_label,
            "ladderRank": v.ladder_rank,
            "executable": v.executable,
        })).collect::<Vec<_>>(),
        // Nonzero here is why a picker can be short or empty; say so rather
        // than letting it read as "this repository publishes no GGUF".
        "shardedSkipped": found.sharded_skipped,
    })
    .to_string())
}
