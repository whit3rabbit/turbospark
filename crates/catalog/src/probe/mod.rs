//! The detection mechanism: decide what this engine would make of an
//! arbitrary Hugging Face repository, reading KB rather than GB.
//!
//! **This is the part that scales past the curated table.** The catalog says
//! what has been run; the probe says what COULD be run, off the same headers
//! the repack walk reads first anyway. Everything it does was already this
//! repo's practice, done by hand once per bring-up (`docs/NEW_MODEL.md`
//! Phase 0, ROADMAP's "no download" probes); this is that practice as a
//! command.
//!
//! Four gates, and the order is cheapest-first so a refusal costs the least
//! possible:
//!
//! 1. **File list** (one JSON GET). Decides GGUF or safetensors and finds
//!    which tokenizer sidecars EXIST, rather than guessing a list.
//! 2. **Architecture**. `repack::gguf_arch_support` for GGUF,
//!    `repack::config_json_family` for safetensors. A recognized-but-unported
//!    architecture gets the registry's own `needs` clause, which is a
//!    different and much more useful message than "unknown".
//! 3. **Quantization**. Per-block-type for GGUF against
//!    `model_io::EXECUTABLE_GGUF_TYPES`; the effective `(bits, group_size)`
//!    pair for MLX affine against `repack::is_supported_affine_shape`.
//! 4. **Expert granularity**. Reported for every MoE candidate and gating
//!    nothing, because it is a fit question rather than a correctness one --
//!    Mixtral is CORRECT here and wants 54.5 GiB of slot cache. This is the
//!    multiplication that was available off the header before Mixtral's 26 GB
//!    download and was not done (AGENTS.md Gotcha 36).
//!
//! **Two traps this file exists to avoid re-hitting.** F32/F16/BF16 are
//! transcoded at repack time and never reach a dispatch, so checking them
//! against `EXECUTABLE_GGUF_TYPES` marks every real candidate blocked -- that
//! is exactly what `scopes_the_dense_llama_candidates`' first run did
//! (`crates/repack/CLAUDE.md` Gotcha 5). And a ggml type this port cannot
//! SIZE is reported as unsized rather than as zero bytes, because an unknown
//! bucket rendered as zero sorts to the bottom of a share column, which is
//! the exact inverse of its real rank, and once made an imatrix file look 76%
//! Q8_0.

mod gguf;
mod safetensors;

use model_io::{ArchConfig, ModelFamily};

use crate::hf::{Client, RepoRef};

pub use gguf::evaluate_gguf;
pub use safetensors::evaluate_config;

use gguf::probe_gguf;
use safetensors::probe_safetensors;

/// Expert-cache slot counts the CLI offers, for the working-set table.
const SLOT_COUNTS: [u64; 3] = [8, 16, 32];

/// What the probe concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Every gate passed. `pull` may proceed.
    Runnable,
    /// A gate failed, with the reason already phrased for a user.
    Refused(String),
}

impl Verdict {
    pub fn is_runnable(&self) -> bool {
        matches!(self, Verdict::Runnable)
    }
}

impl ProbeReport {
    /// Record a refusal, keeping the FIRST one.
    ///
    /// The gates run cheapest-first, which is also most-fundamental-first, so
    /// the earliest refusal is the one a reader needs. Letting a later gate
    /// overwrite an earlier one is not hypothetical: probing
    /// `bartowski/Phi-3.5-mini-instruct-GGUF` refuses at the architecture
    /// (recognized, no decode flow, and the registry says what it would need)
    /// and then refuses again at the sidecars, because a GGUF repo carries no
    /// `tokenizer.json`. The second message is true and useless -- it sends
    /// the reader off to find a sidecar repo for a model that would not run
    /// with one.
    fn refuse(&mut self, why: String) {
        if self.verdict.is_runnable() {
            self.verdict = Verdict::Refused(why);
        }
    }
}

/// One ggml block type and how much of the file it accounts for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeShare {
    pub name: String,
    pub tensors: usize,
    /// `None` when this port cannot size the type, which is a louder
    /// statement than zero. See the module header.
    pub bytes: Option<u64>,
    /// Whether this type blocks the install. True for a type with kernels
    /// AND for one that is transcoded away at repack time.
    pub executable: bool,
    /// Whether it is executable because it is TRANSCODED rather than because
    /// it has kernels. Worth distinguishing in the output: printing "has
    /// kernels" beside F32 states something false about this port, and the
    /// false version is the one that invites somebody to go looking for the
    /// F32 GEMV that does not exist.
    pub transcoded: bool,
}

/// The full result of a probe.
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub repo: RepoRef,
    pub kind: crate::SourceKind,
    /// The weights file, for a GGUF probe.
    pub file: Option<String>,
    /// Total weight bytes on the wire.
    pub download_bytes: Option<u64>,
    /// The architecture string as the file spells it: `general.architecture`
    /// for GGUF, `model_type` for safetensors.
    pub architecture: Option<String>,
    pub family: Option<ModelFamily>,
    pub arch: Option<ArchConfig>,
    /// GGUF only, sorted by descending bytes with unsized types last.
    pub types: Vec<TypeShare>,
    /// MLX affine only: the effective default width and group size.
    pub affine: Option<(u32, u32)>,
    /// One routed expert's bytes, for an MoE. `None` for a dense model.
    pub expert_stride: Option<u64>,
    /// Sidecars this port can use, split by whether the repo has them.
    pub sidecars_present: Vec<String>,
    pub sidecars_missing: Vec<String>,
    /// Where a chat template was found, if anywhere.
    pub chat_template: Option<String>,
    pub verdict: Verdict,
    /// Advisory lines that do not gate: expert-cache arithmetic, quantization
    /// surprises, anything a reader should see before committing a download.
    pub warnings: Vec<String>,
}

impl ProbeReport {
    /// The pinned-slot working set implied by [`Self::expert_stride`], as
    /// `(slots, bytes)` pairs. Empty for a dense model.
    pub fn slot_cache_bytes(&self) -> Vec<(u64, u64)> {
        let (Some(stride), Some(arch)) = (self.expert_stride, self.arch.as_ref()) else {
            return Vec::new();
        };
        SLOT_COUNTS
            .iter()
            .map(|&slots| (slots, slots * arch.num_layers.max(0) as u64 * stride))
            .collect()
    }
}

/// The sidecar files this port can make use of, in the order it looks for
/// them. Only `tokenizer.json` is required; the rest change what the
/// tokenizer knows rather than whether it loads.
pub const KNOWN_SIDECARS: [&str; 6] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
];

/// Probe `repo`, optionally forcing which `.gguf` file to look at when the
/// repository offers several quantizations.
pub fn probe(
    client: &Client,
    repo: &RepoRef,
    want_file: Option<&str>,
    sidecar_repo: Option<&RepoRef>,
) -> Result<ProbeReport, String> {
    let files = client.file_list(repo)?;
    let ggufs: Vec<&String> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    let has_safetensors = files.iter().any(|f| f.ends_with(".safetensors"));

    // A repository can hold both; the explicit file wins, then GGUF, because
    // a repo publishing GGUF is publishing it as the artifact.
    let mut report = if let Some(name) = want_file {
        if !files.iter().any(|f| f == name) {
            return Err(format!(
                "{repo} has no file named {name:?}. It offers: {}",
                summarize(&ggufs)
            ));
        }
        probe_gguf(client, repo, name)?
    } else if !ggufs.is_empty() {
        if ggufs.len() > 1 {
            return Err(format!(
                "{repo} offers {} GGUF files; name one with --file. It offers: {}",
                ggufs.len(),
                summarize(&ggufs)
            ));
        }
        probe_gguf(client, repo, ggufs[0])?
    } else if has_safetensors {
        probe_safetensors(client, repo, &files)?
    } else {
        return Err(format!(
            "{repo} has neither a .gguf nor a .safetensors file, so there is nothing \
             here this port could install."
        ));
    };

    // Sidecars come from a different repository for every GGUF row, because
    // a GGUF carries llama.cpp's tokenizer representation rather than an HF
    // `tokenizer.json`. Probing the weights repo for them would report every
    // GGUF as tokenizer-less.
    let sidecar_files = match sidecar_repo {
        Some(other) => client.file_list(other)?,
        None => files,
    };
    check_sidecars(
        &mut report,
        &sidecar_files,
        client,
        sidecar_repo.unwrap_or(repo),
    );
    Ok(report)
}

fn summarize(names: &[&String]) -> String {
    if names.is_empty() {
        return "nothing".to_string();
    }
    names
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Fill in which sidecars the repository has, and where a chat template
/// would come from.
///
/// **A template hides in two places and the split does not follow family**
/// (AGENTS.md Gotcha 41): HF moved it out of `tokenizer_config.json`'s
/// `chat_template` key into a standalone `chat_template.jinja` partway
/// through, and llama.cpp's converter still writes the old one. Reporting
/// only the file made every GGUF-derived install look template-less, and a
/// template-less instruction-tuned model is fed its dialect's fallback
/// framing, which produces fluent output that is not an answer.
fn check_sidecars(report: &mut ProbeReport, files: &[String], client: &Client, repo: &RepoRef) {
    for name in KNOWN_SIDECARS {
        if files.iter().any(|f| f == name) {
            report.sidecars_present.push(name.to_string());
        } else {
            report.sidecars_missing.push(name.to_string());
        }
    }

    if !report
        .sidecars_present
        .iter()
        .any(|f| f == "tokenizer.json")
    {
        report.refuse(format!(
            "{repo} has no tokenizer.json. This port loads an HF tokenizer, so a \
             checkpoint without one needs --sidecar-repo pointing at the checkpoint \
             this artifact was converted from."
        ));
        return;
    }

    if report
        .sidecars_present
        .iter()
        .any(|f| f == "chat_template.jinja")
    {
        report.chat_template = Some("chat_template.jinja".to_string());
        return;
    }

    // The FETCH happens here, where a `Client` is in scope; the DECISION is
    // a pure function below, so the auth-failure branch (catalog Gotcha 12)
    // is testable with a mock `Result` and no HTTP round trip -- the same
    // split `evaluate_gguf`/`evaluate_config` already make for the gates
    // above them (this file's own module doc).
    let fetch = report
        .sidecars_present
        .iter()
        .any(|f| f == "tokenizer_config.json")
        .then(|| client.get_optional(&repo.file_url("tokenizer_config.json")));
    resolve_chat_template(report, fetch);
}

/// Decide the `chat_template` verdict from what fetching
/// `tokenizer_config.json` returned, or `None` when the sidecar list never
/// had that file at all.
///
/// **AN AUTH FAILURE IS NOT A CONFIRMED ABSENCE.** `Client::get_optional`
/// returns `Err` for anything other than 200/404, which on a gated
/// repository with no `HF_TOKEN` is a 401 -- silent evidence of nothing. It
/// gets its OWN warning rather than falling into the generic "no chat
/// template found" line, which would read as a fact about the checkpoint
/// rather than about this request. Measured directly: probing
/// `meta-llama/Meta-Llama-3-8B-Instruct` (gated) unauthenticated reports
/// `NONE FOUND`; with `HF_TOKEN` set, `tokenizer_config.json:chat_template`.
fn resolve_chat_template(report: &mut ProbeReport, fetch: Option<Result<Option<Vec<u8>>, String>>) {
    match fetch {
        Some(Ok(Some(bytes))) => {
            let has_key = serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|v| v.get("chat_template").cloned())
                .is_some();
            if has_key {
                report.chat_template = Some("tokenizer_config.json:chat_template".to_string());
                return;
            }
        }
        Some(Ok(None)) | None => {}
        Some(Err(e)) => {
            report.warnings.push(format!(
                "could not check tokenizer_config.json for a chat_template key: {e}. \
                 This may be a gated repository refusing an unauthenticated request \
                 (export HF_TOKEN) rather than a checkpoint that genuinely has none."
            ));
            return;
        }
    }
    report.warnings.push(
        "no chat template found in either place, so an instruction-tuned checkpoint will \
         fall back to its dialect's framing. That produces fluent output that is not an \
         answer (AGENTS.md Gotcha 41)."
            .to_string(),
    );
}

#[cfg(test)]
mod chat_template_resolution_tests {
    use super::*;

    /// No `evaluate_gguf`/`evaluate_config` call in this crate builds a
    /// `ProbeReport` for a case this narrow (only `warnings` and
    /// `chat_template` are read below), so this constructs one field by
    /// field, matching `evaluate_gguf`'s own construction in `gguf.rs`.
    fn blank_report() -> ProbeReport {
        ProbeReport {
            repo: RepoRef::new("owner/name", "main"),
            kind: crate::SourceKind::Gguf,
            file: None,
            download_bytes: None,
            architecture: None,
            family: None,
            arch: None,
            types: Vec::new(),
            affine: None,
            expert_stride: None,
            sidecars_present: Vec::new(),
            sidecars_missing: Vec::new(),
            chat_template: None,
            verdict: Verdict::Runnable,
            warnings: Vec::new(),
        }
    }

    /// The bug this whole function exists to fix: a 401 must NOT read as
    /// the generic "no chat template found" warning, which states something
    /// false about the checkpoint. It gets its own warning naming `HF_TOKEN`.
    #[test]
    fn an_http_error_gets_its_own_warning_naming_hf_token() {
        let mut report = blank_report();
        resolve_chat_template(
            &mut report,
            Some(Err("GET https://x: HTTP 401".to_string())),
        );
        assert_eq!(report.chat_template, None);
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(
            report.warnings[0].contains("HF_TOKEN"),
            "the auth-failure warning must point at the fix: {:?}",
            report.warnings[0]
        );
        assert!(
            !report.warnings[0].contains("no chat template found in either place"),
            "the generic absence warning must not also fire: {:?}",
            report.warnings[0]
        );
    }

    /// A genuine 404 (or no `tokenizer_config.json` in the sidecar list at
    /// all) is the one case that gets the generic warning -- this is what
    /// the auth-failure case above must NOT be confused with.
    #[test]
    fn a_confirmed_absence_gets_the_generic_warning() {
        for fetch in [Some(Ok(None)), None] {
            let mut report = blank_report();
            resolve_chat_template(&mut report, fetch);
            assert_eq!(report.chat_template, None);
            assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
            assert!(
                report.warnings[0].contains("no chat template found in either place"),
                "{:?}",
                report.warnings[0]
            );
        }
    }

    #[test]
    fn a_chat_template_key_present_resolves_and_warns_nothing() {
        let mut report = blank_report();
        let body = br#"{"chat_template": "{{ messages }}"}"#.to_vec();
        resolve_chat_template(&mut report, Some(Ok(Some(body))));
        assert_eq!(
            report.chat_template.as_deref(),
            Some("tokenizer_config.json:chat_template")
        );
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    }

    /// A `tokenizer_config.json` that parses but carries no `chat_template`
    /// key is a CONFIRMED absence (not an error), so it takes the generic
    /// warning same as a 404 -- the fetch succeeded and the key is not there.
    #[test]
    fn a_tokenizer_config_with_no_chat_template_key_is_a_confirmed_absence() {
        let mut report = blank_report();
        let body = br#"{"bos_token": "<s>"}"#.to_vec();
        resolve_chat_template(&mut report, Some(Ok(Some(body))));
        assert_eq!(report.chat_template, None);
        assert_eq!(report.warnings.len(), 1);
        assert!(report.warnings[0].contains("no chat template found in either place"));
    }
}
