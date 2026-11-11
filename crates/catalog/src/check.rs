//! A compatibility check for one repository that never throws on a verdict.
//!
//! [`crate::probe`] answers "what would this engine make of this repository"
//! with `Err` for several outcomes a search UI must SHOW rather than handle: a
//! repository with several GGUF files, one with neither format, a gated one.
//! This wraps it. Every outcome becomes a [`CheckState`] so a caller renders a
//! row instead of an error path, and it reuses the probe's gates untouched, so
//! the two cannot disagree about what runs.
//!
//! It is meant for ONE repository the user selected. It reads the file list,
//! one GGUF header or one `config.json`, and (for safetensors) a `HEAD` per
//! shard. Nothing here is sized for probing every row of a result page; the
//! search itself carries only advisory metadata ([`crate::LiveMeta`]).

use crate::hf::{Client, RepoFile, RepoRef};
use crate::probe::{probe, ProbeReport, Verdict};
use crate::recommend::{gguf_variants_from_files, GgufVariant, GgufVariants};

/// What the check concluded about a repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckState {
    /// Every probe gate passed for the representative file or checkpoint.
    Runnable,
    /// A gate failed. `reason` carries the probe's own words.
    Refused,
    /// The Hub answered 401 or 403: gated, or the token lacks access. This is
    /// NOT a verdict on the checkpoint, so it is kept apart from `Refused`.
    NeedsAuth,
    /// The check could not be completed (offline, rate limited, 404, a Hub
    /// error). Also not a verdict, and worth retrying.
    Unavailable,
}

/// The result of [`check_repo`].
#[derive(Debug, Clone)]
pub struct RepoCheck {
    pub repo: RepoRef,
    pub state: CheckState,
    /// The refusal or failure text. `None` when runnable.
    pub reason: Option<String>,
    /// The probe's full report, when one was produced. Its `file` is the
    /// representative GGUF the header was read from.
    pub report: Option<ProbeReport>,
    /// Every pickable `.gguf`, best quality first. Empty for safetensors.
    pub variants: Vec<GgufVariant>,
    /// `.gguf` files that are shard parts and so cannot be picked.
    pub sharded_skipped: usize,
    /// The repository the tokenizer sidecars came from, set only when the
    /// weights repository had none and the caller's hint supplied them. An
    /// install of this check's result must pass the same repository.
    pub sidecar_repo: Option<RepoRef>,
}

impl RepoCheck {
    fn bare(repo: &RepoRef, state: CheckState, reason: String) -> Self {
        Self {
            repo: repo.clone(),
            state,
            reason: Some(reason),
            report: None,
            variants: Vec::new(),
            sharded_skipped: 0,
            sidecar_repo: None,
        }
    }
}

/// What a search row's architecture string says about this engine, WITHOUT
/// reading the checkpoint.
///
/// **A recognized architecture is not a runnable checkpoint.** The registry
/// answers "does this port know this name"; quantization, a missing tokenizer
/// and the dense-versus-MoE split of one GGUF architecture string are all
/// decided later, by [`check_repo`]. Render this as a hint ("architecture
/// recognized"), never as "compatible". `minimax_m2` is the standing example:
/// it is in the supported HF table so other parsers can recognize it, and
/// catalog safetensors intake still refuses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchHint {
    /// The architecture is in this port's supported table.
    Recognized,
    /// Known and not yet ported. `needs` is the registry's own next step.
    Planned { needs: &'static str },
    /// A GGUF architecture string this port has never seen.
    Unrecognized,
    /// The search row carried nothing to judge by, or only a safetensors
    /// `model_type`, which cannot prove absence: the registry's HF table
    /// holds supported rows only, and a multimodal config can name its
    /// supported family in `text_config`, which a list response omits.
    Unknown,
}

/// Pure: the hint for one search row.
pub fn arch_hint(meta: &crate::LiveMeta) -> ArchHint {
    if let Some(architecture) = &meta.gguf_architecture {
        return match repack::gguf_arch_support(architecture) {
            Some(repack::ArchSupport::Supported(_)) => ArchHint::Recognized,
            Some(repack::ArchSupport::Planned(planned)) => ArchHint::Planned {
                needs: planned.needs,
            },
            None => ArchHint::Unrecognized,
        };
    }
    match &meta.model_type {
        Some(model_type) if repack::hf_family_for_model_type(model_type).is_some() => {
            ArchHint::Recognized
        }
        _ => ArchHint::Unknown,
    }
}

/// What a file listing says there is to check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Neither format is present.
    Nothing,
    /// GGUF files exist but every one is a shard part.
    OnlyShards { count: usize },
    /// GGUF: probe this representative file.
    Gguf {
        file: String,
        variants: GgufVariants,
    },
    /// Safetensors only: probe the config.
    Safetensors,
    /// The caller named a `.gguf` the repository does not list.
    NoSuchFile { name: String },
}

/// Pure: decide what to probe from a listing. GGUF wins when both formats are
/// present, because a repository publishing GGUF is publishing it as the
/// artifact (the same rule [`crate::probe`] applies).
///
/// With no `want_file` the representative is the best-quality file whose name
/// declares a type with kernels. A header read costs the same for any
/// quantization, and the family and tokenizer verdict is a property of the
/// model, so which file is read does not change them. Per-file executability
/// stays on the variants. A caller that names a file gets THAT file's header,
/// because the fit is per file: a smaller quantization can fit where the best
/// one does not.
pub(crate) fn plan(files: &[RepoFile], want_file: Option<&str>) -> Plan {
    let variants = gguf_variants_from_files(files);
    if let Some(name) = want_file.filter(|_| !variants.variants.is_empty()) {
        return if variants.variants.iter().any(|v| v.file == name) {
            Plan::Gguf {
                file: name.to_string(),
                variants,
            }
        } else {
            Plan::NoSuchFile {
                name: name.to_string(),
            }
        };
    }
    if let Some(chosen) = variants
        .variants
        .iter()
        .find(|v| v.executable)
        .or_else(|| variants.variants.first())
    {
        return Plan::Gguf {
            file: chosen.file.clone(),
            variants: variants.clone(),
        };
    }
    if variants.sharded_skipped > 0 {
        return Plan::OnlyShards {
            count: variants.sharded_skipped,
        };
    }
    if files.iter().any(|f| f.name.ends_with(".safetensors")) {
        return Plan::Safetensors;
    }
    Plan::Nothing
}

/// Pure: a probe or listing error string to a non-verdict state.
pub(crate) fn classify_failure(message: &str) -> CheckState {
    // `Client` formats these as `GET <url>: HTTP 401` (hf.rs).
    if message.contains("HTTP 401") || message.contains("HTTP 403") {
        CheckState::NeedsAuth
    } else {
        CheckState::Unavailable
    }
}

fn from_report(repo: &RepoRef, report: ProbeReport, listing: Option<GgufVariants>) -> RepoCheck {
    let (state, reason) = match &report.verdict {
        Verdict::Runnable => (CheckState::Runnable, None),
        Verdict::Refused(why) => (CheckState::Refused, Some(why.clone())),
    };
    let (variants, sharded_skipped) = listing
        .map(|found| (found.variants, found.sharded_skipped))
        .unwrap_or_default();
    RepoCheck {
        repo: repo.clone(),
        state,
        reason,
        report: Some(report),
        variants,
        sharded_skipped,
        sidecar_repo: None,
    }
}

fn from_probe_error(repo: &RepoRef, message: String) -> RepoCheck {
    RepoCheck::bare(repo, classify_failure(&message), message)
}

/// Checks one repository. `sidecar_hint` is the base-model repository a GGUF
/// conversion came from (a search row's `cardData.base_model`): when the GGUF
/// repository has no `tokenizer.json`, the check retries once against it.
/// `want_file` names the `.gguf` to read instead of the best one (see [`plan`]).
pub fn check_repo(
    client: &Client,
    repo: &RepoRef,
    sidecar_hint: Option<&RepoRef>,
    want_file: Option<&str>,
) -> RepoCheck {
    let files = match client.file_list_with_sizes(repo) {
        Ok(files) => files,
        Err(message) => return from_probe_error(repo, message),
    };

    match plan(&files, want_file) {
        Plan::Nothing => RepoCheck::bare(
            repo,
            CheckState::Refused,
            format!(
                "{repo} has neither a .gguf nor a .safetensors file, so there is nothing \
                 here this port could install."
            ),
        ),
        Plan::OnlyShards { count } => RepoCheck::bare(
            repo,
            CheckState::Refused,
            format!(
                "{repo} publishes only sharded GGUF files ({count} parts). This port installs \
                 a single-file GGUF, so a shard set cannot be installed."
            ),
        ),
        Plan::NoSuchFile { name } => RepoCheck::bare(
            repo,
            CheckState::Refused,
            format!("{repo} has no GGUF file named {name:?}."),
        ),
        Plan::Safetensors => match probe(client, repo, None, None) {
            Ok(report) => from_report(repo, report, None),
            Err(message) => from_probe_error(repo, message),
        },
        Plan::Gguf { file, variants } => {
            let first = match probe(client, repo, Some(&file), None) {
                Ok(report) => report,
                Err(message) => return from_probe_error(repo, message),
            };
            let lacks_tokenizer = !first.verdict.is_runnable()
                && first
                    .sidecars_missing
                    .iter()
                    .any(|name| name == "tokenizer.json");
            // A GGUF carries llama.cpp's tokenizer, not an HF `tokenizer.json`,
            // so most GGUF repositories fail only this gate. The conversion's
            // source repository usually has it.
            if let (true, Some(hint)) = (lacks_tokenizer, sidecar_hint) {
                if let Ok(retry) = probe(client, repo, Some(&file), Some(hint)) {
                    if retry.verdict.is_runnable() {
                        let mut check = from_report(repo, retry, Some(variants));
                        check.sidecar_repo = Some(hint.clone());
                        return check;
                    }
                }
            }
            from_report(repo, first, Some(variants))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str) -> RepoFile {
        RepoFile {
            name: name.to_string(),
            size: Some(1),
        }
    }

    fn files(names: &[&str]) -> Vec<RepoFile> {
        names.iter().map(|n| file(n)).collect()
    }

    #[test]
    fn several_gguf_files_pick_the_best_executable_one_instead_of_failing() {
        let plan = plan(
            &files(&["m-Q2_K.gguf", "m-Q8_0.gguf", "m-Q4_K_M.gguf", "config.json"]),
            None,
        );
        let Plan::Gguf { file, variants } = plan else {
            panic!("a multi-GGUF repository is a check, not an error");
        };
        assert_eq!(variants.variants.len(), 3);
        assert!(variants
            .variants
            .iter()
            .find(|v| v.file == file)
            .is_some_and(|v| v.executable));
        assert_eq!(file, variants.variants[0].file, "best quality first");
    }

    #[test]
    fn an_unrecognized_quant_name_is_still_checked_not_dropped() {
        let Plan::Gguf { file, variants } = plan(&files(&["weights.gguf"]), None) else {
            panic!("a lone GGUF is probed even when its name declares no type");
        };
        assert_eq!(file, "weights.gguf");
        assert!(!variants.variants[0].executable);
    }

    #[test]
    fn gguf_wins_over_safetensors_like_the_probe() {
        assert!(matches!(
            plan(&files(&["m-Q4_K_M.gguf", "model.safetensors"]), None),
            Plan::Gguf { .. }
        ));
        assert_eq!(
            plan(&files(&["model.safetensors"]), None),
            Plan::Safetensors
        );
    }

    #[test]
    fn only_shard_parts_and_empty_listings_are_named_outcomes() {
        assert_eq!(
            plan(
                &files(&[
                    "m-Q4_K_M-00001-of-00002.gguf",
                    "m-Q4_K_M-00002-of-00002.gguf"
                ]),
                None
            ),
            Plan::OnlyShards { count: 2 }
        );
        assert_eq!(
            plan(&files(&["README.md", "config.json"]), None),
            Plan::Nothing
        );
        assert_eq!(plan(&[], None), Plan::Nothing);
    }

    #[test]
    fn a_named_file_is_the_one_whose_header_is_read() {
        let listing = files(&["m-Q2_K.gguf", "m-Q8_0.gguf", "m-Q4_K_M.gguf"]);
        // The default is the best quantization; naming one overrides it, since
        // the fit is per file and a smaller quant can fit where Q8_0 does not.
        assert!(matches!(
            plan(&listing, None),
            Plan::Gguf { ref file, .. } if file == "m-Q8_0.gguf"
        ));
        let Plan::Gguf { file, variants } = plan(&listing, Some("m-Q2_K.gguf")) else {
            panic!("a listed file is checked");
        };
        assert_eq!(file, "m-Q2_K.gguf");
        assert_eq!(
            variants.variants.len(),
            3,
            "the picker still sees every file"
        );
        assert_eq!(
            plan(&listing, Some("nope.gguf")),
            Plan::NoSuchFile {
                name: "nope.gguf".into()
            }
        );
        // A file name means nothing for a safetensors repository.
        assert_eq!(
            plan(&files(&["model.safetensors"]), Some("m.gguf")),
            Plan::Safetensors
        );
    }

    fn meta(gguf: Option<&str>, model_type: Option<&str>) -> crate::LiveMeta {
        crate::LiveMeta {
            gguf_architecture: gguf.map(str::to_owned),
            model_type: model_type.map(str::to_owned),
            ..Default::default()
        }
    }

    #[test]
    fn arch_hint_reads_the_registry_and_never_claims_more() {
        assert_eq!(
            arch_hint(&meta(Some("qwen35moe"), None)),
            ArchHint::Recognized
        );
        assert_eq!(
            arch_hint(&meta(Some("definitely-not-an-arch"), None)),
            ArchHint::Unrecognized
        );
        let (name, planned) = repack::planned_gguf_architectures()
            .next()
            .expect("the registry lists at least one planned architecture");
        assert_eq!(
            arch_hint(&meta(Some(name), None)),
            ArchHint::Planned {
                needs: planned.needs
            }
        );
        assert_eq!(arch_hint(&meta(None, Some("gemma4"))), ArchHint::Recognized);
        // An unknown safetensors model_type is NOT "unrecognized": the HF table
        // holds supported rows only, so absence proves nothing.
        assert_eq!(arch_hint(&meta(None, Some("brand_new"))), ArchHint::Unknown);
        assert_eq!(arch_hint(&meta(None, None)), ArchHint::Unknown);
        // The GGUF string wins when a row carries both.
        assert_eq!(
            arch_hint(&meta(Some("definitely-not-an-arch"), Some("gemma4"))),
            ArchHint::Unrecognized
        );
    }

    #[test]
    fn auth_failures_are_not_verdicts_and_other_failures_are_retryable() {
        for status in ["HTTP 401", "HTTP 403"] {
            assert_eq!(
                classify_failure(&format!("GET https://huggingface.co/x: {status}. gated")),
                CheckState::NeedsAuth
            );
        }
        for message in [
            "GET https://huggingface.co/x: HTTP 404",
            "GET https://huggingface.co/x: HTTP 429",
            "GET https://huggingface.co/x: operation timed out",
        ] {
            assert_eq!(classify_failure(message), CheckState::Unavailable);
        }
    }
}
