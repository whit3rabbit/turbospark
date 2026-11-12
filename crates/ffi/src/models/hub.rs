//! Hugging Face search and the per-repository compatibility check.
//!
//! Two calls, because they cost differently and a host should not blur them:
//! [`hub_search_json`] is one list request carrying ADVISORY metadata (format
//! tags, the architecture string the Hub already read), and
//! [`hub_check_json`] is the real verdict for ONE repository the user chose.
//! A search row never claims a model runs; only a check does.

use catalog::{
    arch_hint, ArchHint, Catalog, CheckState, Client, HubClient, HubError, HubFormat, HubPage,
    HubQuery, HubSort, Refresh,
};
use serde_json::json;

use super::fit::{local_machine, probe_fit};
use super::probe::{parse_repo, variant_json};
use crate::wire::HubSearchOptions;

const DEFAULT_LIMIT: u32 = 20;
/// Well under the Hub's own cap: a result page is read by a person.
const MAX_LIMIT: u32 = 50;
/// Shorter queries match most of the Hub and say nothing about intent.
const MIN_QUERY_CHARS: usize = 2;

fn parse_sort(value: Option<&str>) -> Result<HubSort, String> {
    match value {
        None | Some("downloads") => Ok(HubSort::Downloads),
        Some("trending") => Ok(HubSort::Trending),
        Some("likes") => Ok(HubSort::Likes),
        Some(other) => Err(format!(
            "sort must be trending, downloads or likes, got {other:?}"
        )),
    }
}

fn parse_format(value: Option<&str>) -> Result<HubFormat, String> {
    match value {
        None | Some("any") => Ok(HubFormat::Any),
        Some("gguf") => Ok(HubFormat::Gguf),
        Some("mlx") => Ok(HubFormat::Mlx),
        Some(other) => Err(format!("format must be any, gguf or mlx, got {other:?}")),
    }
}

/// A failed list request as a STATUS rather than an error code, so a host can
/// say "you are offline" and "slow down" differently from "something broke".
fn failure_json(error: &HubError) -> serde_json::Value {
    let (status, retry_after) = match error {
        HubError::Offline => ("offline", None),
        HubError::RateLimited { retry_after_secs } => ("rateLimited", *retry_after_secs),
        HubError::Network(message)
            if message.contains("HTTP 401") || message.contains("HTTP 403") =>
        {
            ("needsAuth", None)
        }
        _ => ("error", None),
    };
    json!({
        "status": status,
        "message": error.to_string(),
        "retryAfterSecs": retry_after,
        "results": [],
    })
}

/// The search rows of one composed page. Pure, so the shape is testable
/// without a network.
///
/// Bundled-only rows are dropped: the host already shows the curated catalog,
/// and listing it again under "Hugging Face" is the duplicate this avoids. A
/// live row that ALSO matches a curated alias keeps `bundledAliases`, so the
/// host can send the user to the verified install instead of a probed one.
fn page_json(page: &HubPage) -> serde_json::Value {
    let results: Vec<_> = page
        .entries
        .iter()
        .filter_map(|entry| {
            let live = entry.live_target.as_ref()?;
            let meta = &live.meta;
            let (hint, needs) = match arch_hint(meta) {
                ArchHint::Recognized => ("recognized", None),
                ArchHint::Planned { needs } => ("planned", Some(needs)),
                ArchHint::Unrecognized => ("unrecognized", None),
                ArchHint::Unknown => ("unknown", None),
            };
            Some(json!({
                "repoId": entry.repo_id,
                "revision": live.repo.revision,
                "downloads": entry.downloads,
                "likes": entry.likes,
                "hasGguf": meta.gguf,
                "hasMlx": meta.mlx,
                "gated": meta.gated,
                "baseModel": meta.base_model,
                "architecture": meta.gguf_architecture.as_ref().or(meta.model_type.as_ref()),
                "archHint": hint,
                "archNeeds": needs,
                "bundledAliases": entry
                    .bundled_targets
                    .iter()
                    .map(|target| target.alias.as_str())
                    .collect::<Vec<_>>(),
            }))
        })
        .collect();
    let fetched_at = page
        .fetched_at
        .duration_since(std::time::UNIX_EPOCH)
        .map(|age| age.as_secs())
        .unwrap_or(0);
    json!({
        "status": "ok",
        "message": null,
        "retryAfterSecs": null,
        "fetchedAt": fetched_at,
        "fromCache": page.from_cache,
        "rejected": page.rejected.len(),
        "results": results,
    })
}

/// One live Hub search. Text models only; see `catalog::HubFormat`.
pub(crate) fn hub_search_json(query: &str, options: &HubSearchOptions) -> Result<String, String> {
    let text = query.trim();
    if text.chars().count() < MIN_QUERY_CHARS {
        return Err(format!(
            "search text must be at least {MIN_QUERY_CHARS} characters"
        ));
    }
    let sort = parse_sort(options.sort.as_deref())?;
    let format = parse_format(options.format.as_deref())?;
    let limit = options.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);

    // The machine and context only size the curated rows `HubClient` merges
    // in; live rows stay `unknown` fit until a check reads a header.
    let machine = local_machine(model_io::LoadGuard::default())?;
    let client = Client::new();
    let hub = HubClient::new(
        &client,
        Catalog::embedded()?,
        machine,
        4096,
        model_io::ExpertCacheSlots::Auto,
    );
    let refresh = if options.refresh {
        Refresh::Force
    } else {
        Refresh::AllowCache
    };
    let query = HubQuery::new(text)
        .with_limit(limit)
        .with_sort(sort)
        .with_format(format);
    Ok(match hub.search(&query, refresh) {
        Ok(page) => page_json(&page),
        Err(error) => failure_json(&error),
    }
    .to_string())
}

fn state_name(state: CheckState) -> &'static str {
    match state {
        CheckState::Runnable => "runnable",
        CheckState::Refused => "refused",
        CheckState::NeedsAuth => "needsAuth",
        CheckState::Unavailable => "unavailable",
    }
}

/// The real verdict for one repository. See `catalog::check_repo`.
///
/// `state` is the field to branch on. `needsAuth` and `unavailable` are NOT
/// verdicts on the checkpoint (a gated repository, or no answer), so a host
/// must not render them as "incompatible".
pub(crate) fn hub_check_json(
    repo: &str,
    sidecar_hint: Option<&str>,
    file: Option<&str>,
    context: Option<u32>,
    slots: model_io::ExpertCacheSlots,
    guard: model_io::LoadGuard,
) -> Result<String, String> {
    let repo = parse_repo(repo)?;
    let hint = sidecar_hint.map(parse_repo).transpose()?;
    let check = catalog::check_repo(&Client::new(), &repo, hint.as_ref(), file);

    let report = check.report.as_ref();
    let fit =
        report.and_then(|r| probe_fit(r, context.unwrap_or(4096), slots, guard, r.trained_context));
    Ok(json!({
        "repo": check.repo.repo,
        "revision": check.repo.revision,
        "state": state_name(check.state),
        "reason": check.reason,
        "kind": report.map(|r| r.kind.as_str()),
        // The representative GGUF the header was read from. The user's pick
        // may differ; `variants` lists the rest.
        "file": report.and_then(|r| r.file.clone()),
        "architecture": report.and_then(|r| r.architecture.clone()),
        "family": report.and_then(|r| r.family).map(|f| f.as_str()),
        "downloadBytes": report.and_then(|r| r.download_bytes),
        "trainedContext": report.and_then(|r| r.trained_context),
        "chatTemplate": report.and_then(|r| r.chat_template.clone()),
        "warnings": report.map(|r| r.warnings.clone()).unwrap_or_default(),
        "sidecarsMissing": report.map(|r| r.sidecars_missing.clone()).unwrap_or_default(),
        // Set only when the hint supplied the tokenizer. An install of this
        // result must pass the same repository as its sidecar.
        "sidecarRepo": check.sidecar_repo.as_ref().map(|r| r.repo.clone()),
        "variants": check.variants.iter().map(variant_json).collect::<Vec<_>>(),
        "shardedSkipped": check.sharded_skipped,
        "fit": fit,
    })
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use catalog::{EntryProvenance, FitSummary, HubEntry, LiveMeta, LiveTarget, RepoRef};
    use std::time::{Duration, UNIX_EPOCH};

    fn fit() -> FitSummary {
        FitSummary {
            verdict: "unknown".into(),
            is_estimate: false,
            counted_bytes: None,
            counted_source: "unknown".into(),
            mapped_bytes: None,
            notes: Vec::new(),
        }
    }

    fn live(repo: &str, meta: LiveMeta) -> HubEntry {
        HubEntry {
            name: repo.into(),
            repo_id: repo.into(),
            provenance: vec![EntryProvenance::LiveHub],
            bundled_targets: Vec::new(),
            live_target: Some(LiveTarget {
                repo: RepoRef::new(repo, "a".repeat(40)),
                evidence: "discovered".into(),
                fit: fit(),
                variants: None,
                meta,
            }),
            downloads: Some(10),
            likes: Some(2),
        }
    }

    #[test]
    fn options_reject_unknown_spellings_and_default_to_downloads_and_any() {
        assert_eq!(parse_sort(None).unwrap(), HubSort::Downloads);
        assert_eq!(parse_sort(Some("likes")).unwrap(), HubSort::Likes);
        assert!(parse_sort(Some("newest")).is_err());
        assert_eq!(parse_format(None).unwrap(), HubFormat::Any);
        assert_eq!(parse_format(Some("mlx")).unwrap(), HubFormat::Mlx);
        assert!(parse_format(Some("onnx")).is_err());
    }

    #[test]
    fn short_queries_are_refused_before_any_network_call() {
        for q in ["", " ", "a", " a "] {
            let error = hub_search_json(q, &HubSearchOptions::default()).unwrap_err();
            assert!(error.contains("at least 2"), "{q:?}: {error}");
        }
        // Option errors also precede the network.
        let bad = HubSearchOptions {
            sort: Some("newest".into()),
            ..Default::default()
        };
        assert!(hub_search_json("qwen", &bad).unwrap_err().contains("sort"));
    }

    #[test]
    fn a_page_keeps_live_rows_with_their_hints_and_drops_bundled_only_rows() {
        let gguf = live(
            "o/gguf",
            LiveMeta {
                gguf: true,
                gguf_architecture: Some("qwen35moe".into()),
                base_model: Some("Qwen/Base".into()),
                ..Default::default()
            },
        );
        let gated_mlx = live(
            "o/mlx",
            LiveMeta {
                mlx: true,
                gated: true,
                model_type: Some("brand_new".into()),
                ..Default::default()
            },
        );
        let mut bundled_only = live("o/curated", LiveMeta::default());
        bundled_only.live_target = None;
        let page = HubPage {
            entries: vec![gguf, gated_mlx, bundled_only],
            rejected: Vec::new(),
            fetched_at: UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            from_cache: true,
        };
        let value = page_json(&page);
        assert_eq!(value["status"], "ok");
        assert_eq!(value["fetchedAt"], 1_700_000_000u64);
        assert_eq!(value["fromCache"], true);
        let rows = value["results"].as_array().unwrap();
        assert_eq!(rows.len(), 2, "the bundled-only row is the host's own");

        // The Swift binding (HubSearchWireTests) decodes exactly these keys.
        // A rename here must be a rename there, so pin the set.
        let keys = |v: &serde_json::Value| {
            let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
            keys.sort();
            keys
        };
        assert_eq!(
            keys(&rows[0]),
            [
                "archHint",
                "archNeeds",
                "architecture",
                "baseModel",
                "bundledAliases",
                "downloads",
                "gated",
                "hasGguf",
                "hasMlx",
                "likes",
                "repoId",
                "revision"
            ]
        );
        assert_eq!(
            keys(&value),
            [
                "fetchedAt",
                "fromCache",
                "message",
                "rejected",
                "results",
                "retryAfterSecs",
                "status"
            ]
        );

        assert_eq!(rows[0]["repoId"], "o/gguf");
        assert_eq!(rows[0]["archHint"], "recognized");
        assert_eq!(rows[0]["hasGguf"], true);
        assert_eq!(rows[0]["baseModel"], "Qwen/Base");
        assert_eq!(rows[0]["revision"], "a".repeat(40));

        // An unknown safetensors model_type is "unknown", never "unrecognized".
        assert_eq!(rows[1]["archHint"], "unknown");
        assert_eq!(rows[1]["gated"], true);
        assert_eq!(rows[1]["architecture"], "brand_new");
    }

    #[test]
    fn failures_are_statuses_a_host_can_tell_apart() {
        assert_eq!(failure_json(&HubError::Offline)["status"], "offline");
        let limited = failure_json(&HubError::RateLimited {
            retry_after_secs: Some(30),
        });
        assert_eq!(limited["status"], "rateLimited");
        assert_eq!(limited["retryAfterSecs"], 30);
        assert_eq!(
            failure_json(&HubError::Network("GET x: HTTP 401".into()))["status"],
            "needsAuth"
        );
        assert_eq!(
            failure_json(&HubError::Network("connection reset".into()))["status"],
            "error"
        );
        for failed in [
            failure_json(&HubError::Offline),
            failure_json(&HubError::Cancelled),
        ] {
            assert_eq!(failed["results"], json!([]));
        }
    }
}
