//! `meeting_eval`: score speaker-attributed transcripts against the frozen
//! Frozen E AMI panel.
//!
//! ```text
//! meeting_eval score --seglst HYP.json [--references REFS.json] [--clips matched6|all|ID,ID]
//! ```
//! `--references` defaults to the committed `testdata/frozen_e/references.json`.

use std::path::PathBuf;
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
    let frozen = read_frozen_counts(&frozen_e_dir().join("frozen_counts.json")).unwrap_or_default();
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
