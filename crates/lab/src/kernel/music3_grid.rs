//! The Music 3 kernel corpus as a bench: `crates/gpu/examples/
//! music3_linear_bench.rs`, included by path exactly as
//! `crates/gpu/tests/music3_oracle.rs` includes it, so there is one corpus.
//!
//! Hash modes (`hashes`, `dit_hashes`, `conv_hashes`) print an FNV-1a of
//! every configuration's output bits; this bench records each as a point's
//! `out_fnv` and counts the lines that differ from the checked-in goldens
//! (`crates/gpu/tests/golden/`). The affine grid's golden holds on the chip
//! that produced it only, as the oracle says, and the record names the chip.
//! Timing modes (`timing`, `conv_timing`) record each configuration's mean
//! milliseconds.
//!
//! An exception to "shapes from the install": this corpus is fixed, the
//! point of a hash corpus. It runs for a Music 3 install, or for any install
//! with `--case`, since it needs no weights.

use std::collections::BTreeMap;

use gpu::MetalContext;
use serde_json::json;

use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{ArmCurve, Axis, BenchResult, Point};

#[path = "../../../gpu/examples/music3_linear_bench.rs"]
#[allow(dead_code)]
mod corpus;

const GOLDEN_HASHES: &str = include_str!("../../../gpu/tests/golden/music3_hashes.txt");
const GOLDEN_DIT: &str = include_str!("../../../gpu/tests/golden/music3_dit_hashes.txt");
const GOLDEN_CONV: &str = include_str!("../../../gpu/tests/golden/music3_conv_hashes.txt");
/// The chip the affine grid's golden was generated on (the oracle's rule).
const AFFINE_GOLDEN_DEVICE: &str = "Apple M4 Max";

const MODES: [&str; 5] = [
    "hashes",
    "dit_hashes",
    "conv_hashes",
    "timing",
    "conv_timing",
];

pub struct Music3Grid;

impl KernelBench for Music3Grid {
    fn name(&self) -> &'static str {
        "music3_grid"
    }

    fn describe(&self) -> &'static str {
        "Music 3 linear and conv corpus: output-bit hashes against the goldens, and per-config timing"
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        if subject.family != "minimax_music3" {
            return Err(
                "the Music 3 corpus runs for a Music 3 install (or pick a mode with --case)".into(),
            );
        }
        Ok(MODES
            .iter()
            .map(|m| Case {
                label: m.to_string(),
                params: json!({"mode": m}),
            })
            .collect())
    }

    fn run(
        &self,
        _context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        _opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let mode = case.params["mode"]
            .as_str()
            .unwrap_or(&case.label)
            .to_string();
        if !MODES.contains(&mode.as_str()) {
            return Err(format!("music3_grid mode {mode:?}: one of {MODES:?}"));
        }
        let mut lines = Vec::new();
        // The corpus panics on a Metal failure; keep that a bench error.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            corpus::run(&mode, &mut |l| lines.push(l))
        }))
        .map_err(|_| format!("music3 corpus mode {mode} panicked"))?;
        Ok(result(&mode, &lines, &device_name()))
    }
}

fn device_name() -> String {
    gpu::Music3Device::new()
        .map(|d| d.device_name())
        .unwrap_or_else(|_| "unknown".into())
}

/// One corpus line: its configuration label, mean ms if timed, GB/s if
/// given, and the output hash (the last 16-hex-digit token).
fn parse_line(line: &str) -> (String, Option<f64>, Option<f64>, Option<String>) {
    let mut label = Vec::new();
    let (mut ms, mut gbs, mut hash) = (None, None, None);
    for tok in line.split_whitespace() {
        let tok_hash = tok.strip_prefix("hash=").unwrap_or(tok);
        if tok_hash.len() == 16 && tok_hash.chars().all(|c| c.is_ascii_hexdigit()) {
            hash = Some(tok_hash.to_string());
        } else if let Some(v) = tok.strip_prefix("mean_ms=") {
            ms = v.parse().ok();
        } else if let Some(v) = tok.strip_prefix("weight_GBps=") {
            gbs = v.parse().ok();
        } else {
            label.push(tok);
        }
    }
    (label.join(" "), ms, gbs, hash)
}

/// The bench record for one mode's output lines.
pub fn result(mode: &str, lines: &[String], device: &str) -> BenchResult {
    let golden = match mode {
        "hashes" => Some(GOLDEN_HASHES),
        "dit_hashes" => Some(GOLDEN_DIT),
        "conv_hashes" => Some(GOLDEN_CONV),
        _ => None,
    };
    let timing = golden.is_none();
    let mut points = Vec::new();
    let mut configs = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let (label, ms, gbs, hash) = parse_line(l);
        let mut p = Point::new(i as f64, if timing { ms.unwrap_or(f64::NAN) } else { 0.0 });
        p.gb_s = gbs;
        p.out_fnv = hash;
        points.push(p);
        configs.push(label);
    }
    let mut subject = BTreeMap::new();
    subject.insert("mode".into(), json!(mode));
    subject.insert("device".into(), json!(device));
    subject.insert("configs".into(), json!(configs));
    let mut notes = Vec::new();
    if let Some(g) = golden {
        // Compare the hash field per line: the conv goldens carry their own
        // (zero) timing, which is not part of the identity.
        let want: Vec<(String, Option<String>)> = g
            .lines()
            .map(|l| {
                let (label, _, _, hash) = parse_line(l);
                (label, hash)
            })
            .collect();
        let got: Vec<(String, Option<String>)> = lines
            .iter()
            .map(|l| {
                let (label, _, _, hash) = parse_line(l);
                (label, hash)
            })
            .collect();
        let differing: Vec<&String> = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a != b)
            .map(|(a, _)| &a.0)
            .collect();
        let applies = mode != "hashes" || device == AFFINE_GOLDEN_DEVICE;
        subject.insert("golden_lines".into(), json!(want.len()));
        subject.insert(
            "golden_mismatches".into(),
            json!(differing.len() + want.len().abs_diff(got.len())),
        );
        subject.insert("golden_applies".into(), json!(applies));
        notes.push(format!(
            "{} of {} lines differ from the golden{}",
            differing.len() + want.len().abs_diff(got.len()),
            want.len(),
            if applies {
                String::new()
            } else {
                format!(" (not binding: it is for {AFFINE_GOLDEN_DEVICE}, this is {device})")
            }
        ));
        for d in differing.iter().take(5) {
            notes.push(format!("differs: {d}"));
        }
    }
    BenchResult {
        bench: "music3_grid".into(),
        case: mode.to_string(),
        subject,
        x: Axis::new("config", "index", "linear"),
        layout: "music3 corpus (fixed)".into(),
        arms: vec![ArmCurve {
            label: "music3".into(),
            points,
            fit: None,
            unsupported: None,
        }],
        break_even: Vec::new(),
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_parse_and_the_golden_matches_itself() {
        let (label, ms, gbs, hash) = parse_line(
            "rows=1 in=2048 out=6144 mean_ms=0.123 weight_GBps=51.2 hash=00112233aabbccdd",
        );
        assert_eq!(label, "rows=1 in=2048 out=6144");
        assert_eq!((ms, gbs), (Some(0.123), Some(51.2)));
        assert_eq!(hash.as_deref(), Some("00112233aabbccdd"));
        let golden: Vec<String> = GOLDEN_CONV.lines().map(str::to_string).collect();
        let r = result("conv_hashes", &golden, "Apple M4 Max");
        assert_eq!(r.subject["golden_mismatches"], 0);
        let mut changed = golden.clone();
        let last = changed[0].len() - 1;
        let flip = if changed[0].ends_with('0') { "1" } else { "0" };
        changed[0].replace_range(last.., flip);
        let r = result("conv_hashes", &changed, "Apple M4 Max");
        assert_eq!(r.subject["golden_mismatches"], 1, "{:?}", r.notes);
        // The affine grid is not binding off its chip.
        let grid: Vec<String> = GOLDEN_HASHES.lines().map(str::to_string).collect();
        assert_eq!(
            result("hashes", &grid, "Apple M1").subject["golden_applies"],
            json!(false)
        );
    }
}
