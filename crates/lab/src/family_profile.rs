//! Per-family profiles: the settings known to pay, the sweeps worth running
//! first, and the findings behind both, as data.
//!
//! One JSON file per text family (named by `ModelFamily::as_str`) and per
//! audio kind (named by its `model_type`) under `families/`.
//! [README](../families/README.md) beside the files holds the format and the
//! field rules this module enforces. The set is embedded at compile time
//! ([`ProfileSet::embedded`]) and a scratch directory can overlay it
//! ([`ProfileSet::load_dir`]), the same mechanism as `pinned.rs`.
//!
//! A profile never decides a result. `recommended` names what the ledger and
//! the docs found, with the finding ids that back it; `sweeps` are the arm
//! lists `tslab sweep` runs under the fixed verdict rule; `findings` carry
//! their provenance, and a doc table is a `reading`, never a verdict.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use crate::audio_kind::AudioKind;
use crate::gates::GateClass;
use crate::workload::{Metric, Speculation, Workload};

/// The schema string every profile carries.
pub const SCHEMA: &str = "turbospark.lab.family/1";

/// What a finding is: a tslab verdict (ledger provenance) or a reading from
/// a doc table or an upstream report.
pub const VERDICTS: &[&str] = &["win", "loss", "noise", "inconclusive", "reading"];

/// Where a finding came from.
pub const PROVENANCE_KINDS: &[&str] = &["ledger", "doc", "upstream_reported"];

/// The coverage tags a `when` clause may name (`families.rs` builds them;
/// `workflow.rs` evaluates the same vocabulary). Prefixed tags take any
/// suffix: `family:<name>`, `flow:<name>`, `bench:<name>`, `audio:<task>`.
pub const TAGS: &[&str] = &[
    "text",
    "audio",
    "moe",
    "dense",
    "gdn",
    "mla",
    "compressed_attention",
    "full_attention",
    "sliding_attention",
    "sinks",
    "vision",
    "chunked_prefill",
    "steel",
    "ane",
    "probe",
    "standin",
    "runs",
    "drafter",
    "mtp",
    "dflash",
    "profile",
    "nomodel",
];
pub const TAG_PREFIXES: &[&str] = &["family:", "flow:", "bench:", "audio:"];

/// The embedded set: (file stem, file).
const EMBEDDED: &[(&str, &str)] = &[
    ("deepseek2", include_str!("../families/deepseek2.json")),
    (
        "deepseekV4Flash",
        include_str!("../families/deepseekV4Flash.json"),
    ),
    ("fish_drama3", include_str!("../families/fish_drama3.json")),
    ("flint", include_str!("../families/flint.json")),
    ("gemma4", include_str!("../families/gemma4.json")),
    ("gptOss", include_str!("../families/gptOss.json")),
    ("kokoro", include_str!("../families/kokoro.json")),
    ("lfm3", include_str!("../families/lfm3.json")),
    ("llama", include_str!("../families/llama.json")),
    ("megatts3", include_str!("../families/megatts3.json")),
    ("minimax_m2", include_str!("../families/minimax_m2.json")),
    (
        "minimax_music3",
        include_str!("../families/minimax_music3.json"),
    ),
    ("moonshine", include_str!("../families/moonshine.json")),
    ("museGlimmer", include_str!("../families/museGlimmer.json")),
    ("omnivoice", include_str!("../families/omnivoice.json")),
    ("qwen2", include_str!("../families/qwen2.json")),
    ("qwen3", include_str!("../families/qwen3.json")),
    ("qwen3_asr", include_str!("../families/qwen3_asr.json")),
    ("qwen3_vl", include_str!("../families/qwen3_vl.json")),
    ("qwen3moe", include_str!("../families/qwen3moe.json")),
    ("qwen35", include_str!("../families/qwen35.json")),
    ("qwen36", include_str!("../families/qwen36.json")),
    ("qwen4exp", include_str!("../families/qwen4exp.json")),
    ("spark2_5", include_str!("../families/spark2_5.json")),
    ("whisper", include_str!("../families/whisper.json")),
];

/// An install the family was measured on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstallRef {
    pub name: String,
    #[serde(default)]
    pub quant: String,
    #[serde(default)]
    pub machine: String,
    #[serde(default)]
    pub drafters: Vec<String>,
    #[serde(default)]
    pub note: String,
}

/// One recommended setting: a knob value, a speculation, open parameters or
/// a chunk size, with the findings that back it. Exactly one of `knob`,
/// `speculation`, `slots`, `residency`, `chunk` is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recommended {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knob: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculation: Option<Speculation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residency: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk: Option<String>,
    /// Coverage tags that must hold for the install.
    #[serde(default)]
    pub when: Vec<String>,
    pub why: String,
    /// Finding ids in this profile.
    #[serde(default)]
    pub findings: Vec<String>,
}

impl Recommended {
    /// `knob=value`, `~mtp:2`, `@32/streamed`, `chunk=off`: what an arm
    /// string would carry.
    pub fn as_arm_fragment(&self) -> String {
        if let (Some(k), Some(v)) = (&self.knob, &self.value) {
            return format!("{k}={v}");
        }
        if let Some(sp) = &self.speculation {
            let d = serde_json::to_value(sp.drafter)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            return format!("~{d}:{}{}", sp.block, if sp.adaptive { "a" } else { "" });
        }
        if let Some(s) = &self.slots {
            return format!("@{s}/{}", self.residency.as_deref().unwrap_or("auto"));
        }
        if let Some(r) = &self.residency {
            return format!("@auto/{r}");
        }
        if let Some(c) = &self.chunk {
            return format!("chunk={c}");
        }
        String::new()
    }
}

/// One sweep: a baseline arm and the candidates judged against it, each
/// through the paired A/B rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sweep {
    pub id: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub when: Vec<String>,
    pub metric: Metric,
    pub exactness: GateClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<Workload>,
    /// Arm strings (`NAME[@SLOTS/RES][~DRAFTER:BLOCK[a]][:KEY=V,...]`).
    pub baseline: String,
    pub candidates: Vec<String>,
    #[serde(default)]
    pub note: String,
}

/// Where a finding came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub kind: String,
    /// `ledger`: the experiment directory name under the lab dir.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment: Option<String>,
    /// `ledger`: the record id (`<unix_ms>-<pid>-<n>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record: Option<String>,
    /// `doc`: repo-relative path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// `doc`: the heading the table sits under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    /// `upstream_reported`: the engine and the file the number came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// One measured finding with its conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Finding {
    /// `<family>.<slug>-<YYYY-MM-DD>`, unique across every profile.
    pub id: String,
    pub date: String,
    pub install: String,
    pub machine: String,
    #[serde(default)]
    pub power: String,
    #[serde(default)]
    pub rev: String,
    pub workload: String,
    /// Arm name to the settings it ran under (free-form JSON).
    #[serde(default)]
    pub arms: BTreeMap<String, Value>,
    pub metric: String,
    /// Numbers keyed however the table was laid out (arm, or case then arm).
    pub values: Value,
    pub verdict: String,
    pub provenance: Provenance,
    pub conditions: String,
    #[serde(default)]
    pub note: String,
}

/// One family's profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyProfile {
    pub schema: String,
    pub family: String,
    /// `text` or `audio`.
    pub kind: String,
    pub title: String,
    /// The runtime flow (text) or the audio task (audio).
    #[serde(default)]
    pub flow: String,
    #[serde(default)]
    pub installs: Vec<InstallRef>,
    #[serde(default)]
    pub recommended: Vec<Recommended>,
    #[serde(default)]
    pub sweeps: Vec<Sweep>,
    #[serde(default)]
    pub findings: Vec<Finding>,
    #[serde(default)]
    pub open_questions: Vec<String>,
    #[serde(default)]
    pub traps: Vec<String>,
    #[serde(default)]
    pub docs: Vec<String>,
}

/// Parse and validate one profile file's contents.
pub fn parse(content: &str) -> Result<FamilyProfile, String> {
    let p: FamilyProfile =
        serde_json::from_str(content).map_err(|e| format!("invalid JSON: {e}"))?;
    p.validate()?;
    Ok(p)
}

fn is_date(s: &str) -> bool {
    s.len() == 10
        && s.as_bytes()[4] == b'-'
        && s.as_bytes()[7] == b'-'
        && s.chars()
            .enumerate()
            .all(|(i, c)| matches!(i, 4 | 7) || c.is_ascii_digit())
}

/// `<unix_ms>-<pid>-<n>`, as `ledger::new_id` writes it.
pub fn is_record_id(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// A `when` clause names only known tags (`!t`, `a|b` allowed).
pub fn check_when(when: &[String]) -> Result<(), String> {
    for clause in when {
        for alt in clause.split('|') {
            let t = alt.strip_prefix('!').unwrap_or(alt);
            let known = TAGS.contains(&t)
                || TAG_PREFIXES
                    .iter()
                    .any(|p| t.starts_with(p) && t.len() > p.len());
            if !known {
                return Err(format!("when clause {clause:?} names unknown tag {t:?}"));
            }
        }
    }
    Ok(())
}

fn check_ascii(value: &Value, path: &str) -> Result<(), String> {
    match value {
        Value::String(s) if !s.is_ascii() => Err(format!("{path}: string is not ASCII")),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .try_for_each(|(i, v)| check_ascii(v, &format!("{path}[{i}]"))),
        Value::Object(map) => map
            .iter()
            .try_for_each(|(k, v)| check_ascii(v, &format!("{path}.{k}"))),
        _ => Ok(()),
    }
}

fn has_number(v: &Value) -> bool {
    match v {
        Value::Number(_) => true,
        Value::Array(a) => a.iter().any(has_number),
        Value::Object(m) => m.values().any(has_number),
        _ => false,
    }
}

impl FamilyProfile {
    /// Enforce the README's field rules. Family names are checked against
    /// the audio kinds here; text family names are held to `ModelFamily`
    /// by `tests/family_profiles.rs`, which can see `model_io`.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: &str| -> String { format!("{}: {what}", self.family) };
        if self.schema != SCHEMA {
            return Err(bad(&format!("schema {:?} is not {SCHEMA:?}", self.schema)));
        }
        match self.kind.as_str() {
            "text" => {}
            "audio" => {
                if !AudioKind::ALL.iter().any(|k| k.as_str() == self.family) {
                    return Err(bad("audio family is not an AudioKind model_type"));
                }
            }
            other => return Err(bad(&format!("kind {other:?} is not text or audio"))),
        }
        if self.title.trim().is_empty() {
            return Err(bad("title is empty"));
        }
        let mut finding_ids = std::collections::BTreeSet::new();
        for f in &self.findings {
            let prefix = format!("{}.", self.family);
            if !f.id.starts_with(&prefix) {
                return Err(bad(&format!(
                    "finding id {:?} does not start with {prefix:?}",
                    f.id
                )));
            }
            let slug = &f.id[prefix.len()..];
            if slug.len() < 11 || !is_date(&slug[slug.len() - 10..]) {
                return Err(bad(&format!(
                    "finding id {:?} does not end in -YYYY-MM-DD",
                    f.id
                )));
            }
            if !slug[..slug.len() - 10]
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
            {
                return Err(bad(&format!(
                    "finding id {:?} slug is not [a-z0-9_-]",
                    f.id
                )));
            }
            if !finding_ids.insert(f.id.as_str()) {
                return Err(bad(&format!("finding id {:?} repeats", f.id)));
            }
            if !is_date(&f.date) {
                return Err(bad(&format!("finding {} date is not YYYY-MM-DD", f.id)));
            }
            for (name, v) in [
                ("install", &f.install),
                ("machine", &f.machine),
                ("workload", &f.workload),
                ("metric", &f.metric),
                ("conditions", &f.conditions),
            ] {
                if v.trim().is_empty() {
                    return Err(bad(&format!("finding {} has an empty {name}", f.id)));
                }
            }
            if !VERDICTS.contains(&f.verdict.as_str()) {
                return Err(bad(&format!(
                    "finding {} verdict {:?} is not one of {VERDICTS:?}",
                    f.id, f.verdict
                )));
            }
            if !has_number(&f.values) {
                return Err(bad(&format!("finding {} has no number in values", f.id)));
            }
            let p = &f.provenance;
            if !PROVENANCE_KINDS.contains(&p.kind.as_str()) {
                return Err(bad(&format!(
                    "finding {} provenance kind {:?} is not one of {PROVENANCE_KINDS:?}",
                    f.id, p.kind
                )));
            }
            match p.kind.as_str() {
                "ledger" => {
                    let exp = p.experiment.as_deref().unwrap_or("");
                    if exp.is_empty() || crate::paths::validate_name(exp).is_err() {
                        return Err(bad(&format!(
                            "finding {} ledger provenance needs an experiment name",
                            f.id
                        )));
                    }
                    if !p.record.as_deref().is_some_and(is_record_id) {
                        return Err(bad(&format!(
                            "finding {} ledger provenance needs a record id <unix_ms>-<pid>-<n>",
                            f.id
                        )));
                    }
                }
                "doc" => {
                    let path = p.path.as_deref().unwrap_or("");
                    if !(path.starts_with("docs/") || path.starts_with("crates/"))
                        || path.contains("..")
                    {
                        return Err(bad(&format!(
                            "finding {} doc provenance path {path:?} is not repo-relative under docs/ or crates/",
                            f.id
                        )));
                    }
                    if p.heading.as_deref().unwrap_or("").trim().is_empty() {
                        return Err(bad(&format!(
                            "finding {} doc provenance needs a heading",
                            f.id
                        )));
                    }
                    if f.verdict != "reading" {
                        return Err(bad(&format!(
                            "finding {} is a doc reading; its verdict must be \"reading\"",
                            f.id
                        )));
                    }
                }
                _ => {
                    if p.source.as_deref().unwrap_or("").trim().is_empty() {
                        return Err(bad(&format!(
                            "finding {} upstream provenance needs a source",
                            f.id
                        )));
                    }
                    if f.verdict != "reading" {
                        return Err(bad(&format!(
                            "finding {} is an upstream report; its verdict must be \"reading\"",
                            f.id
                        )));
                    }
                }
            }
        }
        for r in &self.recommended {
            let set = [
                r.knob.is_some(),
                r.speculation.is_some(),
                r.slots.is_some() || r.residency.is_some(),
                r.chunk.is_some(),
            ]
            .iter()
            .filter(|b| **b)
            .count();
            if set != 1 {
                return Err(bad(
                    "a recommended entry sets exactly one of knob, speculation, slots/residency, chunk",
                ));
            }
            if let Some(k) = &r.knob {
                let Some(knob) = crate::knobs::by_name(k) else {
                    return Err(bad(&format!("recommended knob {k:?} is not a KNOBS row")));
                };
                let Some(v) = &r.value else {
                    return Err(bad(&format!("recommended knob {k:?} has no value")));
                };
                if !knob.values.contains(&v.as_str()) {
                    return Err(bad(&format!(
                        "recommended knob {k:?} value {v:?} is not one of {:?}",
                        knob.values
                    )));
                }
            } else if r.value.is_some() {
                return Err(bad("a recommended entry has a value but no knob"));
            }
            if let Some(s) = &r.slots {
                crate::args::parse_slots(s)?;
            }
            if let Some(res) = &r.residency {
                crate::args::parse_residency(res)?;
            }
            if let Some(c) = &r.chunk {
                if c != "off" && c.parse::<usize>().is_err() {
                    return Err(bad(&format!("recommended chunk {c:?} is not N or off")));
                }
            }
            if r.why.trim().is_empty() {
                return Err(bad("a recommended entry has an empty why"));
            }
            check_when(&r.when).map_err(|e| bad(&e))?;
            for id in &r.findings {
                if !finding_ids.contains(id.as_str()) {
                    return Err(bad(&format!(
                        "recommended entry names finding {id:?}, which this profile lacks"
                    )));
                }
            }
        }
        let mut sweep_ids = std::collections::BTreeSet::new();
        for s in &self.sweeps {
            if s.id.is_empty()
                || !s
                    .id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                return Err(bad(&format!("sweep id {:?} is not [a-z0-9_]", s.id)));
            }
            if !sweep_ids.insert(s.id.as_str()) {
                return Err(bad(&format!("sweep id {:?} repeats", s.id)));
            }
            check_when(&s.when).map_err(|e| bad(&e))?;
            if s.candidates.is_empty() {
                return Err(bad(&format!("sweep {} has no candidates", s.id)));
            }
            let mut names = std::collections::BTreeSet::new();
            for arm in std::iter::once(&s.baseline).chain(s.candidates.iter()) {
                let parsed = crate::args::parse_arm(
                    arm,
                    crate::workload::OpenParams::default(),
                    &BTreeMap::new(),
                )
                .map_err(|e| bad(&format!("sweep {}: {e}", s.id)))?;
                if !names.insert(parsed.name) {
                    return Err(bad(&format!("sweep {} repeats an arm name", s.id)));
                }
            }
            if let Some(w) = &s.workload {
                if w.prompts.is_empty() && w.audio.is_none() {
                    return Err(bad(&format!("sweep {} workload has no prompts", s.id)));
                }
            }
            if self.kind == "audio"
                && !matches!(
                    s.metric,
                    Metric::AudioRunS | Metric::Rtf | Metric::PeakBytes
                )
            {
                return Err(bad(&format!(
                    "sweep {} on an audio family uses audio_run_s, rtf or peak_bytes",
                    s.id
                )));
            }
        }
        for d in &self.docs {
            if !d.starts_with("docs/") && !d.starts_with("crates/") {
                return Err(bad(&format!("docs entry {d:?} is not repo-relative")));
            }
        }
        let as_json = serde_json::to_value(self).map_err(|e| bad(&e.to_string()))?;
        check_ascii(&as_json, &self.family)
    }

    /// The `recommended` entries and `sweeps` whose `when` holds for `tags`.
    pub fn applicable(
        &self,
        tags: &std::collections::BTreeSet<String>,
    ) -> (Vec<&Recommended>, Vec<&Sweep>) {
        let holds = |when: &[String]| {
            when.iter().all(|clause| {
                clause.split('|').any(|alt| match alt.strip_prefix('!') {
                    Some(t) => !tags.contains(t),
                    None => tags.contains(alt),
                })
            })
        };
        (
            self.recommended.iter().filter(|r| holds(&r.when)).collect(),
            self.sweeps.iter().filter(|s| holds(&s.when)).collect(),
        )
    }

    pub fn sweep(&self, id: &str) -> Option<&Sweep> {
        self.sweeps.iter().find(|s| s.id == id)
    }
}

/// A validated set of profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileSet {
    profiles: Vec<FamilyProfile>,
}

impl ProfileSet {
    /// The set embedded at compile time.
    pub fn embedded() -> Result<Self, String> {
        let mut profiles = Vec::with_capacity(EMBEDDED.len());
        for (stem, content) in EMBEDDED {
            let p = parse(content).map_err(|e| format!("embedded {stem}: {e}"))?;
            if p.family != *stem {
                return Err(format!("embedded {stem}: family is {:?}", p.family));
            }
            profiles.push(p);
        }
        let set = Self { profiles };
        set.validate()?;
        Ok(set)
    }

    /// Every `*.json` directly inside `dir`, validated.
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        files.sort_by_key(|f| f.file_name());
        let mut profiles = Vec::new();
        for file in files {
            let path = file.path();
            if !path.extension().map(|e| e == "json").unwrap_or(false) {
                continue;
            }
            let content =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let p = parse(&content).map_err(|e| format!("{}: {e}", path.display()))?;
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if p.family != stem {
                return Err(format!(
                    "{}: family {:?} does not match the file name",
                    path.display(),
                    p.family
                ));
            }
            profiles.push(p);
        }
        let set = Self { profiles };
        set.validate()?;
        Ok(set)
    }

    /// Profiles from `overlay` replace same-family profiles in `self`.
    pub fn with_overlay(self, overlay: ProfileSet) -> Result<Self, String> {
        let mut profiles = self.profiles;
        for over in overlay.profiles {
            match profiles.iter_mut().find(|p| p.family == over.family) {
                Some(slot) => *slot = over,
                None => profiles.push(over),
            }
        }
        let set = Self { profiles };
        set.validate()?;
        Ok(set)
    }

    /// Every profile valid, families unique, finding ids unique across all.
    pub fn validate(&self) -> Result<(), String> {
        let mut families = std::collections::BTreeSet::new();
        let mut ids = std::collections::BTreeSet::new();
        for p in &self.profiles {
            p.validate()?;
            if !families.insert(p.family.as_str()) {
                return Err(format!("duplicate family profile {:?}", p.family));
            }
            for f in &p.findings {
                if !ids.insert(f.id.as_str()) {
                    return Err(format!("duplicate finding id {:?}", f.id));
                }
            }
        }
        Ok(())
    }

    /// The embedded set, with `--dir DIR` layered over it when given.
    pub fn resolve(dir: Option<&str>) -> Result<Self, String> {
        let base = Self::embedded()?;
        match dir {
            Some(d) => base.with_overlay(Self::load_dir(Path::new(d))?),
            None => Ok(base),
        }
    }

    pub fn profiles(&self) -> &[FamilyProfile] {
        &self.profiles
    }

    pub fn by_family(&self, family: &str) -> Option<&FamilyProfile> {
        self.profiles.iter().find(|p| p.family == family)
    }
}

/// The profile directory beside this source file, for tests and tools.
pub fn source_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("families")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_set_validates_and_has_every_audio_kind() {
        let set = ProfileSet::embedded().expect("embedded profiles parse");
        assert_eq!(set.profiles().len(), EMBEDDED.len());
        for k in AudioKind::ALL {
            let p = set
                .by_family(k.as_str())
                .unwrap_or_else(|| panic!("no profile for audio kind {}", k.as_str()));
            assert_eq!(p.kind, "audio");
        }
    }

    #[test]
    fn every_profile_file_is_embedded_and_unchanged() {
        let on_disk = ProfileSet::load_dir(&source_dir()).expect("families dir loads");
        let embedded = ProfileSet::embedded().expect("embedded profiles parse");
        assert_eq!(
            on_disk.profiles().len(),
            embedded.profiles().len(),
            "a profile was added or removed without updating EMBEDDED"
        );
        for p in embedded.profiles() {
            let disk = on_disk
                .by_family(&p.family)
                .unwrap_or_else(|| panic!("{} missing on disk", p.family));
            assert_eq!(disk, p, "{} drifted from the embedded copy", p.family);
        }
    }

    #[test]
    fn an_overlay_replaces_the_same_family() {
        let mut tweaked = ProfileSet::embedded().unwrap();
        tweaked.profiles[0].title = "tweaked".into();
        let fam = tweaked.profiles[0].family.clone();
        let merged = ProfileSet::embedded()
            .unwrap()
            .with_overlay(tweaked)
            .unwrap();
        assert_eq!(merged.by_family(&fam).unwrap().title, "tweaked");
        assert_eq!(merged.profiles().len(), EMBEDDED.len());
    }

    #[test]
    fn applicable_filters_by_tags() {
        let set = ProfileSet::embedded().unwrap();
        let p = set.by_family("qwen35").unwrap();
        let no_drafter: std::collections::BTreeSet<String> = ["text", "dense", "runs"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (_, sweeps) = p.applicable(&no_drafter);
        assert!(sweeps.iter().all(|s| !s.when.iter().any(|w| w == "mtp")));
        let with_mtp: std::collections::BTreeSet<String> = ["text", "dense", "runs", "mtp"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (_, sweeps) = p.applicable(&with_mtp);
        assert!(sweeps.iter().any(|s| s.id == "spec_block"));
    }

    #[test]
    fn parse_rejects_broken_profiles() {
        let base = ProfileSet::embedded().unwrap();
        let original = serde_json::to_string_pretty(base.by_family("qwen35").unwrap()).unwrap();
        let mutate = |f: &dyn Fn(&mut Value)| -> String {
            let mut v: Value = serde_json::from_str(&original).unwrap();
            f(&mut v);
            parse(&v.to_string()).unwrap_err()
        };
        assert!(mutate(&|v| v["schema"] = "other/1".into()).contains("schema"));
        assert!(mutate(&|v| v["kind"] = "video".into()).contains("kind"));
        assert!(mutate(&|v| v["findings"][0]["verdict"] = "great".into()).contains("verdict"));
        assert!(mutate(&|v| v["findings"][0]["conditions"] = "".into()).contains("conditions"));
        assert!(mutate(&|v| v["findings"][0]["id"] = "qwen35.x".into()).contains("YYYY-MM-DD"));
        assert!(
            mutate(&|v| v["findings"][0]["values"] = serde_json::json!({"a": "fast"}))
                .contains("number")
        );
        // A doc table cannot claim a verdict.
        assert!(mutate(&|v| {
            v["findings"][0]["provenance"] =
                serde_json::json!({"kind": "doc", "path": "docs/X.md", "heading": "h"});
            v["findings"][0]["verdict"] = "win".into();
        })
        .contains("reading"));
        // A ledger verdict needs a record id.
        assert!(mutate(&|v| {
            v["findings"][0]["provenance"] =
                serde_json::json!({"kind": "ledger", "experiment": "e"});
        })
        .contains("record id"));
        assert!(mutate(&|v| v["recommended"][1]["knob"] = "no_such_knob".into()).contains("KNOBS"));
        assert!(
            mutate(&|v| v["recommended"][0]["when"] = serde_json::json!(["warp"]))
                .contains("unknown tag")
        );
        assert!(
            mutate(&|v| v["sweeps"][0]["candidates"] = serde_json::json!(["x~eagle:2"]))
                .contains("drafter")
        );
        assert!(mutate(&|v| v["title"] = "t\u{fffd}".into()).contains("ASCII"));
        let mut v: Value = serde_json::from_str(&original).unwrap();
        v["findingz"] = serde_json::json!([]);
        assert!(parse(&v.to_string()).is_err());
    }

    #[test]
    fn record_ids_and_when_clauses() {
        assert!(is_record_id("1792959758519-40525-11"));
        assert!(!is_record_id("km-lossless"));
        assert!(check_when(&[
            "mtp|dflash".into(),
            "!audio".into(),
            "bench:int4_matmul".into()
        ])
        .is_ok());
        assert!(check_when(&["bench:".into()]).is_err());
    }

    #[test]
    fn arm_fragments() {
        let r = Recommended {
            knob: Some("int4_mma8".into()),
            value: Some("1".into()),
            speculation: None,
            slots: None,
            residency: None,
            chunk: None,
            when: vec![],
            why: "x".into(),
            findings: vec![],
        };
        assert_eq!(r.as_arm_fragment(), "int4_mma8=1");
        let r = Recommended {
            knob: None,
            value: None,
            speculation: Some(Speculation {
                drafter: crate::workload::Drafter::Mtp,
                block: 2,
                adaptive: false,
            }),
            slots: None,
            residency: None,
            chunk: None,
            when: vec![],
            why: "x".into(),
            findings: vec![],
        };
        assert_eq!(r.as_arm_fragment(), "~mtp:2");
    }
}
