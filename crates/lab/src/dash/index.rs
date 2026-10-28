//! The dashboard's view of the lab dir: every experiment's ledger read once
//! and kept, refreshed when a ledger's size or mtime moves, and the rows the
//! pages and the API list (verdicts, sweeps, profile medians) derived from
//! those records.
//!
//! Reading only. A ledger is append-only, so `(mtime, len)` is a complete
//! change signal, and a line a running measurement has half written is
//! skipped and counted rather than failing the page.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

/// What the experiment table shows for one experiment.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentSummary {
    pub name: String,
    pub ledger_path: String,
    pub mtime_ms: u64,
    pub bytes: u64,
    pub first_ts: Option<u64>,
    pub last_ts: Option<u64>,
    pub counts: BTreeMap<String, usize>,
    /// Model install basenames seen in the ledger.
    pub models: BTreeSet<String>,
    pub families: BTreeSet<String>,
    pub chips: BTreeSet<String>,
    /// Short git revisions from the run records.
    pub revs: BTreeSet<String>,
    pub runs_clean: usize,
    pub runs_dirty: usize,
    pub has_notes: bool,
    pub has_report: bool,
    /// Ledger lines that did not parse (a partial tail, or damage).
    pub skipped_lines: usize,
    pub broken: Option<String>,
}

/// One experiment: its summary, its records, its notes.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub summary: ExperimentSummary,
    pub records: Vec<Value>,
    pub notes: Option<String>,
}

/// One paired A/B, from a `verdict`, `experiment` or `confirm` record.
#[derive(Debug, Clone, Serialize)]
pub struct VerdictRow {
    pub experiment: String,
    pub record_id: String,
    pub ts_ms: u64,
    pub kind: String,
    pub sweep_id: Option<String>,
    pub baseline: String,
    pub candidate: String,
    pub metric: String,
    pub verdict: String,
    pub decision: Option<String>,
    pub median: Option<f64>,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    pub pairs: usize,
    pub null_floor: Option<f64>,
    pub model: Option<String>,
    pub family: Option<String>,
    pub chip: Option<String>,
    pub rev: Option<String>,
    pub eligible: bool,
    pub memory_passed: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SweepCandidate {
    pub arm: String,
    pub verdict: String,
    pub median: Option<f64>,
    pub clean_pairs: usize,
    pub dirty_pairs: usize,
    pub refused: Option<String>,
}

/// One `sweep` record.
#[derive(Debug, Clone, Serialize)]
pub struct SweepRow {
    pub experiment: String,
    pub record_id: String,
    pub ts_ms: u64,
    pub sweep_id: String,
    pub family: Option<String>,
    pub model: Option<String>,
    pub baseline: String,
    pub metric: String,
    pub floor: Option<f64>,
    pub best_arm: Option<String>,
    pub best_median: Option<f64>,
    pub candidates: Vec<SweepCandidate>,
    pub stopped: Option<String>,
}

/// One arm of a `profile_summary` record.
#[derive(Debug, Clone, Serialize)]
pub struct ProfileRow {
    pub experiment: String,
    pub record_id: String,
    pub ts_ms: u64,
    pub arm: String,
    pub model: Option<String>,
    pub family: Option<String>,
    pub prefill_tok_s: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub peak_gib: Option<f64>,
    pub clean_runs: Option<u64>,
}

/// Every experiment under the lab dir, with the rows derived from them.
#[derive(Debug, Clone)]
pub struct LabIndex {
    pub lab: PathBuf,
    pub experiments: BTreeMap<String, Experiment>,
    pub verdicts: Vec<VerdictRow>,
    pub sweeps: Vec<SweepRow>,
    pub profiles: Vec<ProfileRow>,
    /// Ledger stamps the last refresh saw: (mtime_ms, len).
    stamps: BTreeMap<String, (u64, u64)>,
}

/// The last path component.
pub fn basename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

/// `YYYY-MM-DD HH:MM` in the host's local zone, the zone the measurement
/// was taken in; `tz_label` names it for the page. Falls back to UTC when
/// the C library cannot resolve a zone. The API keeps raw `ts_ms`.
pub fn fmt_ts(ms: u64) -> String {
    match local_tm(ms) {
        Some(tm) => format!(
            "{:04}-{:02}-{:02} {:02}:{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min
        ),
        None => fmt_ts_utc(ms),
    }
}

/// The host zone's name (`CDT`, `UTC`), else its offset (`UTC-05:00`).
pub fn tz_label() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    match local_tm(now) {
        // SAFETY: `tm_zone` points at the C library's static zone name,
        // which outlives this call; it is read once and copied.
        Some(tm) if !tm.tm_zone.is_null() => unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }
            .to_string_lossy()
            .into_owned(),
        Some(tm) => format!(
            "UTC{:+03}:{:02}",
            tm.tm_gmtoff / 3600,
            (tm.tm_gmtoff.abs() % 3600) / 60
        ),
        None => "UTC".into(),
    }
}

/// The broken-down local time of a unix millisecond stamp. The lab carries
/// no time crate; `localtime_r` is the one call that knows the zone.
fn local_tm(ms: u64) -> Option<libc::tm> {
    let secs = (ms / 1000) as libc::time_t;
    // SAFETY: a zeroed `tm` is a valid out-parameter, `localtime_r` writes
    // only into it, and both locals outlive the call.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let p = unsafe { libc::localtime_r(&secs, &mut tm) };
    (!p.is_null()).then_some(tm)
}

/// `YYYY-MM-DD HH:MM` UTC from unix milliseconds (a day-count to civil-date
/// walk is a few lines, and it is the fallback when no zone resolves).
pub fn fmt_ts_utc(ms: u64) -> String {
    let secs = ms / 1000;
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Howard Hinnant's days_from_civil, inverted.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        rem / 3600,
        (rem % 3600) / 60
    )
}

fn stamp(path: &Path) -> Option<(u64, u64)> {
    let md = std::fs::metadata(path).ok()?;
    let mtime = md
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    Some((mtime, md.len()))
}

/// Every line that parses; the count of those that did not.
fn read_ledger(path: &Path) -> Result<(Vec<Value>, usize), String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    let mut skipped = 0;
    for line in BufReader::new(f).lines() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(v) if v.is_object() => out.push(v),
            _ => skipped += 1,
        }
    }
    Ok((out, skipped))
}

fn str_of(v: &Value) -> Option<String> {
    v.as_str().map(str::to_string)
}

fn short_rev(v: &Value) -> Option<String> {
    v.as_str().map(|r| r.chars().take(10).collect())
}

/// A run ledger line's `record` pointer is file content, not a URL: read it
/// only when it resolves inside the lab dir, so a crafted ledger cannot turn
/// the index into an arbitrary-file reader.
fn read_record_file(lab_root: &Path, path: &str) -> Option<String> {
    let inside = std::fs::canonicalize(path)
        .map(|p| p.starts_with(lab_root))
        .unwrap_or(false);
    if !inside {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

fn load_experiment(lab: &Path, name: &str) -> Experiment {
    let dir = lab.join(name);
    let lab_root = std::fs::canonicalize(lab).unwrap_or_else(|_| lab.to_path_buf());
    let ledger_path = dir.join("ledger.jsonl");
    let (mtime_ms, bytes) = stamp(&ledger_path).unwrap_or((0, 0));
    let mut summary = ExperimentSummary {
        name: name.to_string(),
        ledger_path: ledger_path.display().to_string(),
        mtime_ms,
        bytes,
        first_ts: None,
        last_ts: None,
        counts: BTreeMap::new(),
        models: BTreeSet::new(),
        families: BTreeSet::new(),
        chips: BTreeSet::new(),
        revs: BTreeSet::new(),
        runs_clean: 0,
        runs_dirty: 0,
        has_notes: dir.join("notes.md").is_file(),
        has_report: dir.join("report.html").is_file(),
        skipped_lines: 0,
        broken: None,
    };
    let notes = std::fs::read_to_string(dir.join("notes.md")).ok();
    let records = match read_ledger(&ledger_path) {
        Ok((recs, skipped)) => {
            summary.skipped_lines = skipped;
            recs
        }
        Err(e) => {
            summary.broken = Some(e);
            Vec::new()
        }
    };
    // Chip and revision live in the per-run record file, not the ledger
    // line; one file per distinct run_id prefix is enough, so read the
    // first few.
    let mut provenance_reads = 0;
    for rec in &records {
        let kind = rec["kind"].as_str().unwrap_or("");
        *summary.counts.entry(kind.to_string()).or_default() += 1;
        if let Some(ts) = rec["ts_ms"].as_u64() {
            summary.first_ts = Some(summary.first_ts.map_or(ts, |t| t.min(ts)));
            summary.last_ts = Some(summary.last_ts.map_or(ts, |t| t.max(ts)));
        }
        if let Some(f) = rec["family"].as_str() {
            summary.families.insert(f.to_string());
        }
        if let Some(m) = rec["model"].as_str() {
            summary.models.insert(basename(m));
        }
        if kind == "run" {
            if rec["eligible"].as_bool().unwrap_or(false) {
                summary.runs_clean += 1;
            } else {
                summary.runs_dirty += 1;
            }
            if provenance_reads < 3 {
                if let Some(text) = rec["record"]
                    .as_str()
                    .and_then(|p| read_record_file(&lab_root, p))
                {
                    if let Ok(r) = serde_json::from_str::<Value>(&text) {
                        provenance_reads += 1;
                        if let Some(m) = r["model_dir"].as_str() {
                            summary.models.insert(basename(m));
                        }
                        if let Some(f) = r["family"].as_str() {
                            summary.families.insert(f.to_string());
                        }
                        if let Some(c) = r["provenance"]["chip"].as_str() {
                            summary.chips.insert(c.to_string());
                        }
                        if let Some(rev) = short_rev(&r["provenance"]["git_rev"]) {
                            summary.revs.insert(rev);
                        }
                    }
                }
            }
        }
    }
    Experiment {
        summary,
        records,
        notes,
    }
}

fn rows_of(exp: &Experiment) -> (Vec<VerdictRow>, Vec<SweepRow>, Vec<ProfileRow>) {
    let s = &exp.summary;
    let one =
        |set: &BTreeSet<String>| (set.len() == 1).then(|| set.iter().next().cloned().unwrap());
    let family_default = one(&s.families);
    let chip = one(&s.chips);
    let rev = one(&s.revs);
    let mut verdicts = Vec::new();
    let mut sweeps = Vec::new();
    let mut profiles = Vec::new();
    for rec in &exp.records {
        let kind = rec["kind"].as_str().unwrap_or("");
        let id = rec["id"].as_str().unwrap_or("").to_string();
        let ts = rec["ts_ms"].as_u64().unwrap_or(0);
        let model = rec["model"]
            .as_str()
            .or_else(|| rec["spec"]["model"].as_str())
            .map(basename);
        let family = str_of(&rec["family"]).or_else(|| family_default.clone());
        match kind {
            "verdict" | "experiment" | "confirm" => {
                let v = &rec["verdict"];
                let (baseline, candidate) = if kind == "verdict" {
                    (
                        str_of(&rec["baseline"]["name"]).unwrap_or_default(),
                        str_of(&rec["candidate"]["name"]).unwrap_or_default(),
                    )
                } else {
                    (
                        str_of(&rec["spec"]["baseline"]["name"]).unwrap_or_default(),
                        str_of(&rec["spec"]["candidate"]["name"]).unwrap_or_default(),
                    )
                };
                let metric = str_of(&rec["metric"])
                    .or_else(|| str_of(&rec["spec"]["primary_metric"]))
                    .unwrap_or_default();
                verdicts.push(VerdictRow {
                    experiment: s.name.clone(),
                    record_id: id,
                    ts_ms: ts,
                    kind: kind.to_string(),
                    sweep_id: str_of(&rec["sweep_id"]),
                    baseline,
                    candidate,
                    metric,
                    verdict: str_of(&v["verdict"]).unwrap_or_else(|| "?".into()),
                    decision: str_of(&rec["decision"]),
                    median: v["median"].as_f64().filter(|m| m.is_finite()),
                    ci_low: v["ci_low"].as_f64().filter(|m| m.is_finite()),
                    ci_high: v["ci_high"].as_f64().filter(|m| m.is_finite()),
                    pairs: v["pairs"].as_u64().unwrap_or(0) as usize,
                    null_floor: rec["null_floor"].as_f64(),
                    model,
                    family,
                    chip: chip.clone(),
                    rev: rev.clone(),
                    eligible: rec["eligible"].as_bool().unwrap_or(true),
                    memory_passed: rec["memory"]["passed"].as_bool(),
                });
            }
            "sweep" => {
                let candidates = rec["candidates"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|c| SweepCandidate {
                        arm: str_of(&c["arm"]).unwrap_or_default(),
                        verdict: str_of(&c["verdict"]["verdict"]).unwrap_or_else(|| "?".into()),
                        median: c["verdict"]["median"].as_f64().filter(|m| m.is_finite()),
                        clean_pairs: c["clean_pairs"].as_u64().unwrap_or(0) as usize,
                        dirty_pairs: c["dirty_pairs"].as_u64().unwrap_or(0) as usize,
                        refused: str_of(&c["refused"]),
                    })
                    .collect();
                sweeps.push(SweepRow {
                    experiment: s.name.clone(),
                    record_id: id,
                    ts_ms: ts,
                    sweep_id: str_of(&rec["sweep_id"]).unwrap_or_default(),
                    family,
                    model,
                    baseline: str_of(&rec["baseline"]["name"]).unwrap_or_default(),
                    metric: str_of(&rec["metric"]).unwrap_or_default(),
                    floor: rec["floor"].as_f64(),
                    best_arm: str_of(&rec["best"]["arm"]),
                    best_median: rec["best"]["median"].as_f64(),
                    candidates,
                    stopped: str_of(&rec["stopped"]),
                });
            }
            "profile_summary" => {
                if let Some(summary) = rec["summary"].as_object() {
                    for (arm, v) in summary {
                        profiles.push(ProfileRow {
                            experiment: s.name.clone(),
                            record_id: id.clone(),
                            ts_ms: ts,
                            arm: arm.clone(),
                            model: model.clone(),
                            family: family.clone(),
                            prefill_tok_s: v["prefill_tok_s"].as_f64(),
                            decode_tok_s: v["decode_tok_s"].as_f64(),
                            peak_gib: v["peak_gib"].as_f64(),
                            clean_runs: v["clean_runs"].as_u64(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    (verdicts, sweeps, profiles)
}

impl LabIndex {
    pub fn build(lab: &Path) -> Self {
        let mut ix = Self {
            lab: lab.to_path_buf(),
            experiments: BTreeMap::new(),
            verdicts: Vec::new(),
            sweeps: Vec::new(),
            profiles: Vec::new(),
            stamps: BTreeMap::new(),
        };
        ix.refresh();
        ix
    }

    /// Experiment directories with a ledger and a valid name.
    fn names(&self) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(&self.lab)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join("ledger.jsonl").is_file())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| crate::paths::validate_name(n).is_ok())
            .collect();
        out.sort();
        out
    }

    /// Re-read every ledger whose stamp moved, drop the ones that vanished,
    /// and rebuild the rows when anything changed. Returns whether it did.
    pub fn refresh(&mut self) -> bool {
        let names = self.names();
        let mut changed = false;
        let present: BTreeSet<&String> = names.iter().collect();
        let gone: Vec<String> = self
            .experiments
            .keys()
            .filter(|k| !present.contains(k))
            .cloned()
            .collect();
        for g in gone {
            self.experiments.remove(&g);
            self.stamps.remove(&g);
            changed = true;
        }
        for name in &names {
            let st = stamp(&self.lab.join(name).join("ledger.jsonl")).unwrap_or((0, 0));
            if self.stamps.get(name) == Some(&st) {
                continue;
            }
            let exp = load_experiment(&self.lab, name);
            self.experiments.insert(name.clone(), exp);
            self.stamps.insert(name.clone(), st);
            changed = true;
        }
        if changed {
            self.verdicts.clear();
            self.sweeps.clear();
            self.profiles.clear();
            for exp in self.experiments.values() {
                let (v, s, p) = rows_of(exp);
                self.verdicts.extend(v);
                self.sweeps.extend(s);
                self.profiles.extend(p);
            }
            self.verdicts.sort_by_key(|a| std::cmp::Reverse(a.ts_ms));
            self.sweeps.sort_by_key(|a| std::cmp::Reverse(a.ts_ms));
            self.profiles.sort_by_key(|a| std::cmp::Reverse(a.ts_ms));
        }
        changed
    }

    pub fn record(&self, experiment: &str, id: &str) -> Option<&Value> {
        self.experiments
            .get(experiment)?
            .records
            .iter()
            .find(|r| r["id"].as_str() == Some(id))
    }

    /// Summaries, newest activity first.
    pub fn summaries(&self) -> Vec<&ExperimentSummary> {
        let mut v: Vec<&ExperimentSummary> =
            self.experiments.values().map(|e| &e.summary).collect();
        v.sort_by(|a, b| b.last_ts.cmp(&a.last_ts).then_with(|| a.name.cmp(&b.name)));
        v
    }

    /// The series of one experiment, model paths reduced to basenames.
    pub fn series_of(&self, experiment: &str) -> Vec<crate::series::Series> {
        let Some(exp) = self.experiments.get(experiment) else {
            return Vec::new();
        };
        exp.records
            .iter()
            .flat_map(|r| crate::series::from_record(r, experiment, crate::series::Paths::Basename))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_from_unix_ms() {
        assert_eq!(fmt_ts_utc(0), "1970-01-01 00:00");
        assert_eq!(fmt_ts_utc(1_793_063_027_807), "2026-10-27 01:03");
        assert_eq!(fmt_ts_utc(951_782_400_000), "2000-02-29 00:00");
    }

    #[test]
    fn local_stamps_resolve_to_the_host_zone() {
        // Whatever the host zone, the local rendering is a civil stamp that
        // sits within a day of UTC, and the label is non-empty.
        let local = fmt_ts(1_793_063_027_807);
        assert_eq!(local.len(), 16, "{local}");
        assert!(local.starts_with("2026-10-2"), "{local}");
        assert!(!tz_label().is_empty());
    }

    #[test]
    fn basenames() {
        assert_eq!(basename("/a/b/c.gturbo"), "c.gturbo");
        assert_eq!(basename("plain"), "plain");
    }
}
