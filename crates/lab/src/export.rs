//! `tslab export` and `tslab report`: ledgers out as series, or as a page.
//!
//! Reading only. Neither command takes the machine lock or touches a GPU, so
//! both are safe to run beside a measurement. Output files are write-once
//! unless `--force`: a report that silently replaced the one a review links
//! to is a changed result nobody saw change.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::args::Args;
use crate::ledger::Ledger;
use crate::paths::{experiment_dir, lab_dir};
use crate::schema::{CmdOutput, SCHEMA};
use crate::series::{from_record, to_csv, Paths, Series};

/// Experiments named by `--experiment A,B` or every one under the lab
/// directory with `--all`, in name order.
pub fn experiments(lab: &Path, args: &mut Args) -> Result<Vec<String>, String> {
    let all = args.flag("--all");
    let named: Vec<String> = args
        .take_all("--experiment")?
        .iter()
        .flat_map(|e| e.split(',').map(str::to_string).collect::<Vec<_>>())
        .collect();
    match (all, named.is_empty()) {
        (true, true) => {
            let mut out: Vec<String> = std::fs::read_dir(lab)
                .map_err(|e| format!("{}: {e}", lab.display()))?
                .flatten()
                .filter(|e| e.path().join("ledger.jsonl").is_file())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| crate::paths::validate_name(n).is_ok())
                .collect();
            out.sort();
            Ok(out)
        }
        (false, false) => Ok(named),
        (true, false) => Err("give --experiment or --all, not both".into()),
        (false, true) => Err("--experiment NAME[,NAME] or --all is required".into()),
    }
}

/// Every series of the named experiments, filtered to `kinds` (series kinds
/// or source ledger kinds) when non-empty.
pub fn collect(
    lab: &Path,
    names: &[String],
    kinds: &[String],
    paths: Paths,
) -> Result<Vec<Series>, String> {
    let mut out = Vec::new();
    for name in names {
        let ledger = Ledger::open(&experiment_dir(lab, name)?)?;
        for rec in ledger.read_all()? {
            for s in from_record(&rec, name, paths) {
                if kinds.is_empty() || kinds.iter().any(|k| *k == s.kind || *k == s.source_kind) {
                    out.push(s);
                }
            }
        }
    }
    Ok(out)
}

fn write_out(path: &Path, bytes: &[u8], force: bool) -> Result<(), String> {
    if force {
        let tmp = path.with_extension("tslab-tmp");
        std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    } else {
        crate::atomic::write_new(path, bytes)
    }
}

fn common(args: &mut Args) -> Result<(Vec<String>, Vec<String>, Paths, bool), String> {
    let lab = lab_dir();
    let names = experiments(&lab, args)?;
    let kinds: Vec<String> = args
        .take_all("--kind")?
        .iter()
        .flat_map(|k| k.split(',').map(str::to_string).collect::<Vec<_>>())
        .collect();
    let paths = if args.flag("--redact-paths") {
        Paths::Basename
    } else {
        Paths::Keep
    };
    let force = args.flag("--force");
    Ok((names, kinds, paths, force))
}

/// `tslab export`.
pub fn export_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let (names, kinds, paths, force) = common(&mut args)?;
    let format = args.take("--format")?.unwrap_or_else(|| "jsonl".into());
    let out_path = args.take("--out")?.map(PathBuf::from);
    args.finish()?;
    let series = collect(&lab_dir(), &names, &kinds, paths)?;
    let text = match format.as_str() {
        "jsonl" => series
            .iter()
            .map(|s| serde_json::to_string(s).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|l| l + "\n")
            .collect::<String>(),
        "json" => {
            let doc = json!({"schema": SCHEMA, "experiments": names, "series": series});
            serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n"
        }
        "csv" => to_csv(&series),
        other => return Err(format!("--format {other:?}: jsonl, json or csv")),
    };
    let summary = json!({"experiments": names, "series_count": series.len(),
                         "points": series.iter().map(|s| s.points.len()).sum::<usize>(),
                         "format": format});
    match out_path {
        Some(p) => {
            write_out(&p, text.as_bytes(), force)?;
            eprintln!("tslab export: {} series to {}", series.len(), p.display());
            let mut r = summary;
            r["path"] = json!(p.display().to_string());
            Ok(CmdOutput::ok(r))
        }
        None if crate::out::json_mode() => {
            // The envelope is stdout's one line, so the series ride in it.
            let mut r = summary;
            r["series"] = serde_json::to_value(&series).map_err(|e| e.to_string())?;
            Ok(CmdOutput::ok(r))
        }
        None => {
            print!("{text}");
            Ok(CmdOutput::ok(summary))
        }
    }
}

/// `tslab report --html OUT`.
pub fn report_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let (names, kinds, paths, force) = common(&mut args)?;
    let out = PathBuf::from(args.required("--html")?);
    let title = args.take("--title")?.unwrap_or_else(|| match names.len() {
        1..=3 => format!("tslab report: {}", names.join(", ")),
        n => format!("tslab report: {n} experiments"),
    });
    args.finish()?;
    let series = collect(&lab_dir(), &names, &kinds, paths)?;
    let generated = format!("unix ms {}", crate::ledger::unix_ms());
    let html = crate::viz::render(&title, &generated, &series);
    write_out(&out, html.as_bytes(), force)?;
    crate::say!("tslab report: {} series to {}", series.len(), out.display());
    Ok(CmdOutput::ok(json!({"path": out.display().to_string(),
                            "experiments": names, "series_count": series.len(),
                            "unquotable": series.iter().filter(|s| !s.quotable).count(),
                            "dirty_points": series.iter().flat_map(|s| &s.points).filter(|p| p.dirty).count()})))
}

/// Count of series per kind, for a caller that wants to know what an export
/// would hold before writing it.
pub fn kinds_present(series: &[Series]) -> Value {
    let mut m = std::collections::BTreeMap::<&str, usize>::new();
    for s in series {
        *m.entry(s.kind.as_str()).or_default() += 1;
    }
    json!(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_reads_a_ledger_and_filters_by_kind() {
        let lab = std::env::temp_dir().join(format!("tslab-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&lab);
        let ledger = Ledger::open(&lab.join("e1")).unwrap();
        ledger
            .append(
                "verdict",
                json!({"baseline": {"name": "a"}, "candidate": {"name": "b"},
                       "verdict": {"improvements": [0.1, 0.2, 0.1], "median": 0.1}}),
            )
            .unwrap();
        ledger.append("run", json!({"arm": "a"})).unwrap();
        let all = collect(&lab, &["e1".into()], &[], Paths::Keep).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].source_schema, 1);
        let none = collect(&lab, &["e1".into()], &["kernel_ms".into()], Paths::Keep).unwrap();
        assert!(none.is_empty());
        let mut args = Args::new(vec!["--all".into()]);
        assert_eq!(
            experiments(&lab, &mut args).unwrap(),
            vec!["e1".to_string()]
        );
        assert_eq!(kinds_present(&all)["verdict"], 1);
        std::fs::remove_dir_all(&lab).unwrap();
    }
}
