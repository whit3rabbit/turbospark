//! `meeting_eval`: score speaker-attributed transcripts against the frozen
//! Frozen E AMI panel.
//!
//! ```text
//! meeting_eval score --seglst HYP.json [--references REFS.json] [--clips matched6|all|ID,ID]
//! ```
//! `--references` defaults to the committed `testdata/frozen_e/references.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use turbospark_bench::meeting::{
    frozen_e_dir, read_frozen_counts, read_references, read_seglst, score_panel, MATCHED_SIX,
};

const USAGE: &str = "usage: meeting_eval score --seglst HYP.json [--references REFS.json] [--clips matched6|all|ID,ID]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("score") => match score(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("meeting_eval: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Reads the frozen counts, returning an empty map plus the reason when the
/// file is missing or malformed instead of swallowing the error.
fn frozen_counts_or_warning(path: &Path) -> (BTreeMap<String, (usize, usize)>, Option<String>) {
    match read_frozen_counts(path) {
        Ok(map) => (map, None),
        Err(e) => (
            BTreeMap::new(),
            Some(format!("cannot read frozen counts: {e}")),
        ),
    }
}

fn score(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut seglst = None;
    let mut references = frozen_e_dir().join("references.json");
    let mut clips_arg = "all".to_owned();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--seglst" => seglst = Some(PathBuf::from(value)),
            "--references" => references = PathBuf::from(value),
            "--clips" => clips_arg = value.clone(),
            other => return Err(format!("unknown flag {other}\n{USAGE}").into()),
        }
    }
    let seglst = seglst.ok_or_else(|| format!("--seglst is required\n{USAGE}"))?;
    let clips = read_references(&references)?;
    let ids: Vec<&str> = match clips_arg.as_str() {
        "all" => vec![],
        "matched6" => MATCHED_SIX.to_vec(),
        list => list.split(',').filter(|s| !s.is_empty()).collect(),
    };
    let (frozen, frozen_warning) =
        frozen_counts_or_warning(&frozen_e_dir().join("frozen_counts.json"));
    if let Some(warning) = &frozen_warning {
        // Scoring is still valid without the comparison column, but a blank
        // column must not look like "no frozen errors recorded".
        eprintln!("meeting_eval: warning: {warning}; frozen_errors column will show `-`");
    }
    let panel = score_panel(&clips, read_seglst(&seglst)?, &ids)?;
    println!("clip errors ref_tokens cpwer ref_spk hyp_spk frozen_errors");
    for row in &panel.clips {
        let s = &row.score;
        let f = frozen
            .get(&row.id)
            .map_or_else(|| "-".to_owned(), |(e, _)| e.to_string());
        println!(
            "{} {} {} {:.4} {} {} {}",
            row.id,
            s.errors,
            s.ref_tokens,
            s.errors as f64 / s.ref_tokens.max(1) as f64,
            s.ref_speakers,
            s.hyp_speakers,
            f
        );
    }
    let frozen_total: Option<usize> = panel
        .clips
        .iter()
        .map(|r| frozen.get(&r.id).map(|(e, _)| *e))
        .sum();
    println!(
        "TOTAL {} {} {:.4} frozen_errors={}",
        panel.errors,
        panel.ref_tokens,
        panel.cpwer(),
        frozen_total.map_or_else(|| "-".to_owned(), |e| e.to_string())
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `.unwrap_or_default()` hid a missing or malformed file entirely.
    #[test]
    fn missing_or_malformed_frozen_counts_produce_a_warning() {
        let dir = std::env::temp_dir().join(format!("tsp_frozen_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (map, warning) = frozen_counts_or_warning(&dir.join("absent.json"));
        assert!(map.is_empty());
        assert!(warning.is_some(), "missing file must warn");

        let bad = dir.join("bad.json");
        std::fs::write(&bad, "{not json").unwrap();
        let (map, warning) = frozen_counts_or_warning(&bad);
        assert!(map.is_empty());
        assert!(warning.is_some(), "malformed file must warn");

        let good = dir.join("good.json");
        std::fs::write(
            &good,
            r#"{"clips":{"a":{"cpwer_errors":3,"reference_tokens":10}}}"#,
        )
        .unwrap();
        let (map, warning) = frozen_counts_or_warning(&good);
        assert_eq!(map.get("a"), Some(&(3, 10)));
        assert!(warning.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
