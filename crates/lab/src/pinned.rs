//! Pinned kernel references: verified settings from the engines this
//! repository compares against (`nace-ai/kapitsa`, `jundot/omlx`,
//! `ddalcu/mlx-serve`).
//!
//! One JSON file per kernel entry under `pinned/<repo>/`, each carrying the
//! repo and revision it was read at, the models it serves, and the
//! hardcoded settings that shape it, traceable to a `path:line` in the
//! pinned source. [README](../../pinned/README.md) beside the files holds
//! the format contract and the field rules this module enforces.
//!
//! The verified baseline is embedded into the binary at compile time
//! ([`PinnedSet::embedded`]), so it travels with `tslab`. A sweep layers
//! tweaked copies over it through [`PinnedSet::load_dir`] and
//! [`PinnedSet::with_overlay`]: copy an entry, change the fields the sweep
//! varies, load the directory, and the colliding id replaces the embedded
//! baseline for that process.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// The schema string every pinned file carries.
pub const SCHEMA: &str = "turbospark.lab.pinned/1";

/// The upstream engines a pinned entry may come from.
pub const REPOS: &[&str] = &["kapitsa", "omlx", "mlx-serve"];

/// What class of kernel an entry covers.
pub const KERNEL_CLASSES: &[&str] = &[
    "gemv",
    "verify_matmul",
    "prefill_matmul",
    "attention_decode",
    "attention_prefill",
    "gdn",
    "moe",
    "sampling",
    "drafter",
    "scheduler",
    "fusion",
    "probe",
];

/// Where a setting's value came from.
pub const SETTING_KINDS: &[&str] = &["hardcoded", "derived", "measured"];

/// The only provenance a `measured` row may carry: these are the other
/// engine's own numbers under its own conditions, never ours.
pub const MEASURED_PROVENANCE: &str = "upstream_reported";

/// The verified baseline, embedded at compile time: (repo, file stem, file).
const EMBEDDED: &[(&str, &str, &str)] = &[
    (
        "kapitsa",
        "attn",
        include_str!("../pinned/kapitsa/attn.json"),
    ),
    ("kapitsa", "gdn", include_str!("../pinned/kapitsa/gdn.json")),
    ("kapitsa", "moe", include_str!("../pinned/kapitsa/moe.json")),
    ("kapitsa", "qmv", include_str!("../pinned/kapitsa/qmv.json")),
    (
        "kapitsa",
        "spec",
        include_str!("../pinned/kapitsa/spec.json"),
    ),
    (
        "kapitsa",
        "verify",
        include_str!("../pinned/kapitsa/verify.json"),
    ),
    (
        "kapitsa",
        "verify8",
        include_str!("../pinned/kapitsa/verify8.json"),
    ),
    (
        "mlx-serve",
        "attn_kv",
        include_str!("../pinned/mlx-serve/attn_kv.json"),
    ),
    (
        "mlx-serve",
        "gdn_decode",
        include_str!("../pinned/mlx-serve/gdn_decode.json"),
    ),
    (
        "mlx-serve",
        "keyed_sample",
        include_str!("../pinned/mlx-serve/keyed_sample.json"),
    ),
    (
        "mlx-serve",
        "prefill_fusions",
        include_str!("../pinned/mlx-serve/prefill_fusions.json"),
    ),
    (
        "mlx-serve",
        "row_attn",
        include_str!("../pinned/mlx-serve/row_attn.json"),
    ),
    (
        "mlx-serve",
        "row_exact",
        include_str!("../pinned/mlx-serve/row_exact.json"),
    ),
    (
        "mlx-serve",
        "spec_round",
        include_str!("../pinned/mlx-serve/spec_round.json"),
    ),
    (
        "mlx-serve",
        "verify_lanes",
        include_str!("../pinned/mlx-serve/verify_lanes.json"),
    ),
    (
        "omlx",
        "gather_qmm_nax",
        include_str!("../pinned/omlx/gather_qmm_nax.json"),
    ),
    (
        "omlx",
        "gdn_pipelined",
        include_str!("../pinned/omlx/gdn_pipelined.json"),
    ),
    (
        "omlx",
        "nax_attention",
        include_str!("../pinned/omlx/nax_attention.json"),
    ),
];

/// Where the entry came from: the repo, the exact revision, the license.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRef {
    pub url: String,
    pub rev: String,
    pub rev_date: String,
    pub license: String,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub symbols: Vec<String>,
}

/// One model the kernel serves or was measured on. `role` is "primary" or
/// "secondary", optionally followed by a short qualifier naming what that
/// model measured for this entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelUse {
    pub name: String,
    #[serde(default)]
    pub quant: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub machine: String,
    pub role: String,
}

/// One setting: a name, its pinned value, when it applies, and the source
/// line it was read at. `value` is any JSON value; `unit` is a string or
/// null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Setting {
    pub name: String,
    pub value: serde_json::Value,
    #[serde(default)]
    pub unit: Option<String>,
    pub condition: String,
    pub kind: String,
    pub source: String,
    #[serde(default)]
    pub note: String,
}

/// A dispatch predicate: when this kernel runs instead of an alternative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gate {
    pub condition: String,
    pub source: String,
    #[serde(default)]
    pub note: String,
}

/// A number the upstream engine itself reports, with its conditions. The
/// `value` is a number, a string when the upstream record states the result
/// as one compound comparison (before and after, a percentage spread), or a
/// curve of numbers over rows or widths.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    pub metric: String,
    pub value: serde_json::Value,
    pub model: String,
    pub machine: String,
    pub conditions: String,
    pub provenance: String,
    pub source: String,
    #[serde(default)]
    pub note: String,
}

/// Where a setting maps onto this lab's own measurement surfaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabMapping {
    /// The setting(s) the mapping applies to (comma separated when several).
    pub setting: String,
    pub maps_to: String,
    #[serde(default)]
    pub env: Option<String>,
    #[serde(default)]
    pub note: String,
}

/// One pinned kernel entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedRef {
    pub schema: String,
    pub id: String,
    pub kernel: String,
    pub kernel_class: String,
    pub repo: String,
    #[serde(default)]
    pub summary: String,
    pub source: SourceRef,
    pub models: Vec<ModelUse>,
    pub settings: Vec<Setting>,
    #[serde(default)]
    pub gates: Vec<Gate>,
    #[serde(default)]
    pub measured: Vec<Measured>,
    #[serde(default)]
    pub lab_mapping: Vec<LabMapping>,
    #[serde(default)]
    pub docs: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

/// Parse and validate one pinned file's contents.
pub fn parse(content: &str) -> Result<PinnedRef, String> {
    let entry: PinnedRef =
        serde_json::from_str(content).map_err(|e| format!("invalid JSON: {e}"))?;
    entry.validate()?;
    Ok(entry)
}

/// The `short` part of an id (`<repo>.<short>`).
pub fn id_short(id: &str) -> Option<&str> {
    id.split_once('.').map(|(_, short)| short)
}

fn is_snake_case(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// `path:line` with a positive line number and no whitespace in the path.
fn is_source_ref(s: &str) -> bool {
    match s.rsplit_once(':') {
        Some((path, line)) => {
            !path.is_empty()
                && !path.contains(char::is_whitespace)
                && line.parse::<usize>().map(|n| n > 0).unwrap_or(false)
        }
        None => false,
    }
}

/// A measured value: a number, a compound comparison string, or a curve of
/// numbers (the row says what each element is).
fn is_measured_value(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Number(_) | serde_json::Value::String(_) => true,
        serde_json::Value::Array(items) => items.iter().all(|i| i.is_number()),
        _ => false,
    }
}

fn check_ascii(value: &serde_json::Value, path: &str) -> Result<(), String> {
    match value {
        serde_json::Value::String(s) => {
            if s.is_ascii() {
                Ok(())
            } else {
                Err(format!("{path}: string is not ASCII"))
            }
        }
        serde_json::Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                check_ascii(item, &format!("{path}[{i}]"))?;
            }
            Ok(())
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                check_ascii(v, &format!("{path}.{k}"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

impl PinnedRef {
    /// Enforce the README's field rules. Structural only: line numbers are
    /// trusted to point where the writer read them, and the registry test
    /// holds the embedded set to the files on disk.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: &str| -> String { format!("{}: {what}", self.id) };
        if self.schema != SCHEMA {
            return Err(bad(&format!("schema {:?} is not {SCHEMA:?}", self.schema)));
        }
        if !REPOS.contains(&self.repo.as_str()) {
            return Err(bad(&format!(
                "repo {:?} is not one of {REPOS:?}",
                self.repo
            )));
        }
        let Some(short) = id_short(&self.id) else {
            return Err(bad("id has no <repo>.<short> form"));
        };
        if short.is_empty() || !is_snake_case(short) {
            return Err(bad(&format!("id short name {short:?} is not [a-z0-9_]")));
        }
        if self.repo != self.id.split('.').next().unwrap_or("") {
            return Err(bad(&format!(
                "id does not start with its repo {:?}",
                self.repo
            )));
        }
        if !KERNEL_CLASSES.contains(&self.kernel_class.as_str()) {
            return Err(bad(&format!(
                "kernel_class {:?} is not one of {KERNEL_CLASSES:?}",
                self.kernel_class
            )));
        }
        if !self.source.url.starts_with("https://") {
            return Err(bad("source.url is not an https URL"));
        }
        let rev_ok = self.source.rev.len() >= 7
            && self.source.rev.len() <= 40
            && self.source.rev.chars().all(|c| c.is_ascii_hexdigit());
        if !rev_ok {
            return Err(bad("source.rev is not 7 to 40 hex characters"));
        }
        let date = &self.source.rev_date;
        let date_ok = date.len() == 10
            && date.as_bytes()[4] == b'-'
            && date.as_bytes()[7] == b'-'
            && date
                .chars()
                .enumerate()
                .all(|(i, c)| matches!(i, 4 | 7) || c.is_ascii_digit());
        if !date_ok {
            return Err(bad("source.rev_date is not YYYY-MM-DD"));
        }
        if self.source.files.is_empty() {
            return Err(bad("source.files is empty"));
        }
        for f in &self.source.files {
            if f.starts_with('/') || f.contains("..") {
                return Err(bad(&format!(
                    "source.files entry {f:?} is not repo-relative"
                )));
            }
        }
        if self.models.is_empty() {
            return Err(bad("models is empty"));
        }
        for m in &self.models {
            if m.name.trim().is_empty() || m.role.trim().is_empty() {
                return Err(bad("a model row has an empty name or role"));
            }
        }
        if self.settings.is_empty() {
            return Err(bad("settings is empty"));
        }
        let mut names = std::collections::BTreeSet::new();
        for s in &self.settings {
            if !is_snake_case(&s.name) {
                return Err(bad(&format!("setting name {:?} is not snake_case", s.name)));
            }
            if !names.insert(s.name.as_str()) {
                return Err(bad(&format!("setting name {:?} repeats", s.name)));
            }
            if !SETTING_KINDS.contains(&s.kind.as_str()) {
                return Err(bad(&format!(
                    "setting {:?} kind {:?} is not one of {SETTING_KINDS:?}",
                    s.name, s.kind
                )));
            }
            if s.condition.trim().is_empty() {
                return Err(bad(&format!("setting {:?} has an empty condition", s.name)));
            }
            if !is_source_ref(&s.source) {
                return Err(bad(&format!(
                    "setting {:?} source {:?} is not path:line",
                    s.name, s.source
                )));
            }
        }
        for g in &self.gates {
            if g.condition.trim().is_empty() {
                return Err(bad("a gate has an empty condition"));
            }
            if !is_source_ref(&g.source) {
                return Err(bad(&format!("gate source {:?} is not path:line", g.source)));
            }
        }
        for m in &self.measured {
            if m.metric.trim().is_empty() || m.conditions.trim().is_empty() {
                return Err(bad("a measured row has an empty metric or conditions"));
            }
            if m.provenance != MEASURED_PROVENANCE {
                return Err(bad(&format!(
                    "measured provenance {:?} is not {MEASURED_PROVENANCE:?}",
                    m.provenance
                )));
            }
            if m.source.trim().is_empty() {
                return Err(bad("a measured row has an empty source"));
            }
            if !is_measured_value(&m.value) {
                return Err(bad(&format!(
                    "measured {} value is not a number, a string, or a number array",
                    m.metric
                )));
            }
        }
        for l in &self.lab_mapping {
            if l.setting.trim().is_empty() || l.maps_to.trim().is_empty() {
                return Err(bad("a lab_mapping row has an empty setting or maps_to"));
            }
            if let Some(env) = &l.env {
                if !env.starts_with("TURBOSPARK_") {
                    return Err(bad(&format!(
                        "lab_mapping env {env:?} is not a TURBOSPARK_ variable"
                    )));
                }
            }
        }
        for d in &self.docs {
            if !d.starts_with("docs/") {
                return Err(bad(&format!("docs entry {d:?} is not under docs/")));
            }
        }
        let as_json = serde_json::to_value(self).map_err(|e| bad(&e.to_string()))?;
        check_ascii(&as_json, &self.id)
    }

    /// The first model whose role marks it primary, else the first model.
    pub fn primary_model(&self) -> &ModelUse {
        self.models
            .iter()
            .find(|m| m.role.starts_with("primary"))
            .unwrap_or(&self.models[0])
    }
}

/// A validated set of pinned entries.
#[derive(Debug, Clone, PartialEq)]
pub struct PinnedSet {
    entries: Vec<PinnedRef>,
}

impl PinnedSet {
    /// The verified baseline embedded at compile time.
    pub fn embedded() -> Result<Self, String> {
        let mut entries = Vec::with_capacity(EMBEDDED.len());
        for (repo, stem, content) in EMBEDDED {
            let entry = parse(content).map_err(|e| format!("embedded {repo}.{stem}: {e}"))?;
            if entry.id != format!("{repo}.{stem}") {
                return Err(format!("embedded {repo}.{stem}: id is {:?}", entry.id));
            }
            entries.push(entry);
        }
        Ok(Self { entries })
    }

    /// Every `*.json` under `dir` (its subdirectories, one level deep),
    /// validated. When a file sits under a subdirectory named for a known
    /// repo, its id must belong to that repo.
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let mut entries = Vec::new();
        read_json_dir(dir, None, &mut entries)?;
        let mut subdirs: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        subdirs.sort_by_key(|f| f.file_name());
        for sub in subdirs {
            let path = sub.path();
            if !path.is_dir() {
                continue;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let repo_dir = REPOS.contains(&name).then_some(name);
            read_json_dir(&path, repo_dir, &mut entries)?;
        }
        Ok(Self { entries })
    }

    /// Entries from `overlay` replace same-id entries in `self`, so a sweep
    /// directory can carry a tweaked baseline under the original id.
    pub fn with_overlay(self, overlay: PinnedSet) -> Result<Self, String> {
        let mut entries = self.entries;
        for over in overlay.entries {
            match entries.iter_mut().find(|e| e.id == over.id) {
                Some(slot) => *slot = over,
                None => entries.push(over),
            }
        }
        let set = Self { entries };
        set.validate()?;
        Ok(set)
    }

    /// All ids unique, every entry valid.
    pub fn validate(&self) -> Result<(), String> {
        for entry in &self.entries {
            entry.validate()?;
        }
        let mut ids = std::collections::BTreeSet::new();
        for entry in &self.entries {
            if !ids.insert(entry.id.as_str()) {
                return Err(format!("duplicate pinned id {:?}", entry.id));
            }
        }
        Ok(())
    }

    pub fn entries(&self) -> &[PinnedRef] {
        &self.entries
    }

    pub fn by_id(&self, id: &str) -> Option<&PinnedRef> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn by_repo(&self, repo: &str) -> Vec<&PinnedRef> {
        self.entries.iter().filter(|e| e.repo == repo).collect()
    }

    pub fn by_class(&self, class: &str) -> Vec<&PinnedRef> {
        self.entries
            .iter()
            .filter(|e| e.kernel_class == class)
            .collect()
    }
}

/// The pinned directory beside this source file, for tests and tools.
pub fn source_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("pinned")
}

/// Read every `*.json` directly inside `dir`, sorted, into `out`.
fn read_json_dir(
    dir: &Path,
    repo_dir: Option<&str>,
    out: &mut Vec<PinnedRef>,
) -> Result<(), String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", dir.display()))?;
    files.sort_by_key(|f| f.file_name());
    for file in files {
        let path = file.path();
        if !path.extension().map(|e| e == "json").unwrap_or(false) {
            continue;
        }
        let content =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let entry = parse(&content).map_err(|e| format!("{}: {e}", path.display()))?;
        if let Some(repo) = repo_dir {
            if !entry.id.starts_with(&format!("{repo}.")) {
                return Err(format!(
                    "{}: id {:?} does not belong to repo {repo:?}",
                    path.display(),
                    entry.id
                ));
            }
        }
        out.push(entry);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_set_validates() {
        let set = PinnedSet::embedded().expect("embedded set parses");
        set.validate().expect("embedded set validates");
        assert_eq!(set.entries().len(), EMBEDDED.len());
        for repo in REPOS {
            assert!(!set.by_repo(repo).is_empty(), "no entries for repo {repo}");
        }
    }

    #[test]
    fn every_pinned_file_is_embedded_and_unchanged() {
        let dir = source_dir();
        let on_disk = PinnedSet::load_dir(&dir).expect("pinned dir loads");
        let embedded = PinnedSet::embedded().expect("embedded set parses");
        assert_eq!(
            on_disk.entries().len(),
            embedded.entries().len(),
            "a pinned file was added or removed without updating EMBEDDED"
        );
        for entry in embedded.entries() {
            let short = id_short(&entry.id).expect("embedded ids are repo.short");
            let path = dir.join(&entry.repo).join(format!("{short}.json"));
            let disk = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(
                parse(&disk).expect("disk file parses"),
                *entry,
                "{} drifted from the embedded copy",
                path.display()
            );
        }
    }

    #[test]
    fn an_overlay_entry_replaces_the_same_id() {
        let mut tweaked = PinnedSet::embedded().expect("embedded set parses");
        tweaked.entries[0].settings[0].value = serde_json::json!(999);
        let merged = PinnedSet::embedded()
            .expect("embedded set parses")
            .with_overlay(tweaked)
            .expect("overlay merges");
        assert_eq!(merged.entries().len(), EMBEDDED.len());
    }

    #[test]
    fn parse_rejects_broken_entries() {
        let base = PinnedSet::embedded().expect("embedded set parses");
        let original = serde_json::to_string_pretty(&base.entries()[0]).unwrap();
        let mutate = |f: &dyn Fn(&mut serde_json::Value)| -> String {
            let mut v: serde_json::Value = serde_json::from_str(&original).unwrap();
            f(&mut v);
            parse(&v.to_string()).unwrap_err()
        };
        assert!(mutate(&|v| v["schema"] = "other/1".into()).contains("schema"));
        assert!(mutate(&|v| v["repo"] = "unknown".into()).contains("repo"));
        assert!(mutate(&|v| v["kernel_class"] = "warp".into()).contains("kernel_class"));
        assert!(mutate(&|v| v["source"]["rev"] = "xyz".into()).contains("rev"));
        assert!(mutate(&|v| v["source"]["rev_date"] = "2026-10".into()).contains("rev_date"));
        assert!(mutate(&|v| {
            v["settings"][0]["source"] = "src/x.rs".into();
        })
        .contains("path:line"));
        assert!(mutate(&|v| {
            v["settings"][0]["kind"] = "guessed".into();
        })
        .contains("kind"));
        assert!(mutate(&|v| {
            v["measured"][0]["provenance"] = "ours".into();
        })
        .contains("provenance"));
        assert!(mutate(&|v| {
            v["settings"][0]["condition"] = "".into();
        })
        .contains("condition"));
        assert!(mutate(&|v| {
            v["models"] = serde_json::json!([]);
        })
        .contains("models"));
        assert!(mutate(&|v| {
            v["kernel"] = "k\u{fffd}l".into();
        })
        .contains("ASCII"));
        // A setting whose id's repo disagrees with the repo field.
        assert!(mutate(&|v| v["id"] = "omlx.attn".into()).contains("repo"));
        // Unknown fields are refused, so a typo cannot pass silently.
        let mut v: serde_json::Value = serde_json::from_str(&original).unwrap();
        v["settingz"] = serde_json::json!([]);
        assert!(parse(&v.to_string()).is_err());
    }

    #[test]
    fn source_refs_and_snake_case_rules() {
        assert!(is_source_ref("src/transformer.zig:398"));
        assert!(is_source_ref("docs/kflash.md:81"));
        assert!(!is_source_ref("src/transformer.zig"));
        assert!(!is_source_ref("src/x.zig:0"));
        assert!(!is_source_ref("two words.zig:3"));
        assert!(is_snake_case("splitk_columns_m2_6"));
        assert!(!is_snake_case("SplitK"));
        assert!(!is_snake_case("9lives"));
        assert_eq!(id_short("kapitsa.verify8"), Some("verify8"));
        assert_eq!(id_short("noverpo"), None);
    }
}
