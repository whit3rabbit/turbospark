//! Fish Audio Drama 3 teacher-forced parity against mlx-audio, on the real
//! 4-bit checkpoint.
//!
//! The reference side is `crates/audio/tools/gen_fish_drama3_fixtures.py
//! teacher-forced`, which replays recorded mlx-audio code sequences and dumps
//! per-row backbone hidden states, semantic logits, the nine codebook head
//! logits and each layer's router selection. This example feeds the same rows
//! through the Metal trunk and the towers, writes the same dump layout, and
//! compares dumps. Three subcommands:
//!
//! ```sh
//! # Rust side. Sets TURBOSPARK_ROUTER_TRACE and TURBOSPARK_ROUTER_HIST itself.
//! cargo run --release -p turbospark-runtime --example fish_drama3_parity -- \
//!     run <install-dir> <mlx-dump-dir>/ref <out-dir> [expert-slots=16]
//!
//! # Any two dumps with the same layout (rust against mlx, or mlx variants).
//! cargo run --release -p turbospark-runtime --example fish_drama3_parity -- \
//!     compare <reference-dump> <other-dump> [--gate]
//!
//! # Router INT8 against BF16 on identical inputs, inside the reference dump.
//! cargo run --release -p turbospark-runtime --example fish_drama3_parity -- \
//!     router-isolation <mlx-dump-dir>/ref
//! ```
//!
//! What a dump holds, per case: `hidden` (the post-final-norm state of every
//! row), `semantic` (the semantic head's logits for the last prompt row and
//! every generated row), `heads` (the nine codebook heads for each generated
//! row, teacher-forced with the recorded next row) and `used` (each row's
//! router top-8 per layer). The Rust run writes the head arrays four ways: the
//! portable f32 tower (`semantic`, `heads`) and the Metal tower
//! (`*_metal`), each fed the Rust trunk's own hidden states and, with the
//! `_refh` suffix, the reference's. The `_refh` arrays separate the tower and
//! the semantic head from the trunk.
//!
//! The tower and the trunk are driven as `Drama3::generate` drives them: the
//! system block, then the rest of the prompt, as two prefills, then one step per
//! generated row. Nothing here is a timing measurement.
//!
//! `--gate` applies [`FROZEN`], the thresholds fixed from the first
//! measurement (`docs/FISH_DRAMA3_PHASE0.md`, Teacher-forced parity).

#![cfg(target_os = "macos")]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use audio::tts::fish_drama3::generate::{Backbone, BackboneRow, Drama3, TRUNK_DIR};
use audio::tts::fish_drama3::tower::TowerModel;
use turbospark_runtime::fish_drama3::{open_drama, TrunkBackbone};
use turbospark_runtime::{ExpertCacheSlots, ExpertResidency, HiddenTrunk};

type Boxed = Box<dyn std::error::Error>;

/// Trunk rows the example reserves. The longest replayed case is a few hundred.
const CONTEXT: usize = 2048;

// -- thresholds ---------------------------------------------------------------

/// What `--gate` holds a Rust dump to, compared with the reference dump.
#[derive(Clone, Copy, Debug)]
struct Thresholds {
    /// Mean per-row cosine of the backbone hidden state.
    hidden_cos_mean: f64,
    /// The 1st percentile of that cosine: the worst rows, not the average.
    hidden_cos_p1: f64,
    /// Semantic-head top-1 agreement of the product path (Metal tower, own
    /// hidden), as a fraction.
    semantic_top1: f64,
    /// Largest mean KL(reference || port) of the semantic logits, nats.
    semantic_kl_max: f64,
    /// Codebook-head argmax agreement of the product path.
    heads_top1: f64,
    /// Mean router top-8 overlap over every layer and row, as a fraction.
    router_overlap: f64,
}

/// `None` until the first measurement is frozen (see the doc).
const FROZEN: Option<Thresholds> = None;

/// The arrays the gate reads: the product path.
const GATE_SEMANTIC: &str = "semantic_metal";
const GATE_HEADS: &str = "heads_metal";

/// The reasons `summary` fails `t`, empty when it passes. A missing array is a
/// failure, not a skip: a gate that reads nothing must not pass.
fn check(summary: &Summary, t: &Thresholds) -> Vec<String> {
    let mut fails = Vec::new();
    match &summary.hidden {
        None => fails.push("no hidden array was compared".to_string()),
        Some(h) => {
            if h.cos_mean < t.hidden_cos_mean {
                fails.push(format!(
                    "hidden cosine mean {:.5} is under {:.5}",
                    h.cos_mean, t.hidden_cos_mean
                ));
            }
            if h.cos_p1 < t.hidden_cos_p1 {
                fails.push(format!(
                    "hidden cosine p1 {:.5} is under {:.5}",
                    h.cos_p1, t.hidden_cos_p1
                ));
            }
        }
    }
    match summary.logits(GATE_SEMANTIC) {
        None => fails.push(format!("no {GATE_SEMANTIC} array was compared")),
        Some(s) => {
            if s.top1 < t.semantic_top1 {
                fails.push(format!(
                    "semantic top-1 {:.4} is under {:.4}",
                    s.top1, t.semantic_top1
                ));
            }
            if s.kl > t.semantic_kl_max {
                fails.push(format!(
                    "semantic KL {:.4} is over {:.4}",
                    s.kl, t.semantic_kl_max
                ));
            }
        }
    }
    match summary.logits(GATE_HEADS) {
        None => fails.push(format!("no {GATE_HEADS} array was compared")),
        Some(s) => {
            if s.top1 < t.heads_top1 {
                fails.push(format!(
                    "codebook-head top-1 {:.4} is under {:.4}",
                    s.top1, t.heads_top1
                ));
            }
        }
    }
    match &summary.router {
        None => fails.push("no router selections were compared".to_string()),
        Some(r) => {
            if r.overlap < t.router_overlap {
                fails.push(format!(
                    "router top-8 overlap {:.4} is under {:.4}",
                    r.overlap, t.router_overlap
                ));
            }
        }
    }
    fails
}

// -- metrics ------------------------------------------------------------------

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// ||a - b|| / ||a||.
fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        diff += (x - y) * (x - y);
        norm += x * x;
    }
    if norm == 0.0 {
        return f64::INFINITY;
    }
    (diff / norm).sqrt()
}

/// The first index of the largest value, which is what a stable top-1 picks.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// The gap between the largest and second largest value.
fn top_gap(v: &[f32]) -> f64 {
    let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if x > first {
            second = first;
            first = x;
        } else if x > second {
            second = x;
        }
    }
    f64::from(first - second)
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max = f64::from(max);
    let sum: f64 = v.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    let log_sum = sum.ln();
    v.iter().map(|&x| f64::from(x) - max - log_sum).collect()
}

/// KL(P || Q) in nats for logits `p` (the reference) and `q`.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter()
        .zip(&lq)
        .map(|(&a, &b)| {
            let pa = a.exp();
            if pa == 0.0 {
                0.0
            } else {
                pa * (a - b)
            }
        })
        .sum()
}

/// How many ids the two selections share. Ids within a selection are distinct.
fn overlap(a: &[i16], b: &[i16]) -> usize {
    a.iter().filter(|x| b.contains(x)).count()
}

fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

// -- summary ------------------------------------------------------------------

#[derive(Clone, Debug)]
struct HiddenStat {
    rows: usize,
    cos_mean: f64,
    cos_p1: f64,
    cos_min: f64,
    rel_l2_mean: f64,
    prompt_cos_mean: f64,
    generated_cos_mean: f64,
}

#[derive(Clone, Debug)]
struct LogitStat {
    name: String,
    rows: usize,
    top1: f64,
    kl: f64,
    /// Agreement by the reference's top-1 gap: (label, rows, agreement).
    by_gap: Vec<(&'static str, usize, f64)>,
    /// Per-head agreement, for the codebook heads.
    per_head: Vec<f64>,
}

#[derive(Clone, Debug)]
struct RouterStat {
    rows: usize,
    layers: usize,
    top_k: usize,
    /// Mean share of the reference's selection the other run also picked.
    overlap: f64,
    /// Share of (row, layer) pairs with an identical set.
    exact: f64,
    per_layer: Vec<f64>,
    /// By the reference's top-k margin: (label, pairs, overlap, exact).
    by_margin: Vec<(&'static str, usize, f64, f64)>,
}

#[derive(Clone, Debug, Default)]
struct Summary {
    hidden: Option<HiddenStat>,
    logits: Vec<LogitStat>,
    router: Option<RouterStat>,
}

impl Summary {
    fn logits(&self, name: &str) -> Option<&LogitStat> {
        self.logits.iter().find(|l| l.name == name)
    }
}

const GAP_BUCKETS: [(&str, f64); 5] = [
    ("gap<0.25", 0.25),
    ("gap<0.5", 0.5),
    ("gap<1", 1.0),
    ("gap<2", 2.0),
    ("gap>=2", f64::INFINITY),
];

const MARGIN_BUCKETS: [(&str, f64); 5] = [
    ("tie (0)", 0.0),
    ("<1e-4", 1e-4),
    ("<1e-3", 1e-3),
    ("<1e-2", 1e-2),
    (">=1e-2", f64::INFINITY),
];

/// The bucket a value falls in: the first whose bound exceeds it, except the
/// tie bucket, which is exactly the bound.
fn bucket_of(buckets: &[(&str, f64)], x: f64, exact_first: bool) -> usize {
    if exact_first && x == buckets[0].1 {
        return 0;
    }
    let start = usize::from(exact_first);
    buckets[start..]
        .iter()
        .position(|&(_, bound)| x < bound)
        .map_or(buckets.len() - 1, |i| i + start)
}

// -- dump files ---------------------------------------------------------------

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: not a whole number of f32", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i16(path: &Path) -> Result<Vec<i16>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 2 != 0 {
        return Err(format!("{}: not a whole number of i16", path.display()));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn write_f32(path: &Path, v: &[f32]) -> Result<(), Boxed> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn write_i16(path: &Path, v: &[i16]) -> Result<(), Boxed> {
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// One replayed case: the frame rows and where the prompt ends.
struct Case {
    name: String,
    dir: PathBuf,
    rows: Vec<Vec<i32>>,
    prompt_rows: usize,
    system_rows: usize,
    meta: serde_json::Value,
}

fn meta_usize(meta: &serde_json::Value, key: &str) -> Result<usize, String> {
    meta.get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("meta.json: {key} missing"))
}

fn load_cases(root: &Path) -> Result<Vec<Case>, String> {
    let mut names: Vec<_> = std::fs::read_dir(root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("meta.json").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let dir = root.join(&name);
            let text = std::fs::read_to_string(dir.join("meta.json")).map_err(|e| e.to_string())?;
            let meta: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            let rows = match meta.get("rows") {
                Some(r) => serde_json::from_value::<Vec<Vec<i32>>>(r.clone())
                    .map_err(|e| format!("{name}: rows: {e}"))?,
                None => Vec::new(),
            };
            Ok(Case {
                prompt_rows: meta_usize(&meta, "prompt_rows")?,
                system_rows: meta_usize(&meta, "system_rows")?,
                name,
                dir,
                rows,
                meta,
            })
        })
        .collect()
}

fn reject_non_finite(what: &str, v: &[f32]) -> Result<(), String> {
    // A NaN reads as agreement on a rank or cosine instrument, so it is an
    // error here rather than a number.
    if v.iter().any(|x| !x.is_finite()) {
        return Err(format!("{what} holds a non-finite value"));
    }
    Ok(())
}

// -- compare ------------------------------------------------------------------

/// The array's base name: `semantic_metal_refh` is a `semantic` array.
fn base_name(name: &str) -> &str {
    name.split('_').next().unwrap_or(name)
}

fn compare_dirs(a: &Path, b: &Path) -> Result<Summary, String> {
    let same = |p: &Path| std::fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    if same(a)? == same(b)? {
        return Err("both dumps are the same directory: that compares a run with itself".into());
    }
    let cases_a = load_cases(a)?;
    let cases_b = load_cases(b)?;
    if cases_a.is_empty() {
        return Err(format!("{}: no cases", a.display()));
    }
    let mut summary = Summary::default();

    let mut cos: Vec<f64> = Vec::new();
    let mut rel: Vec<f64> = Vec::new();
    let (mut cos_prompt, mut cos_generated) = (Vec::new(), Vec::new());

    // name -> (rows, agree, kl sum, by-gap counts, per-head agree/count)
    struct Acc {
        rows: usize,
        agree: usize,
        kl: f64,
        gap_rows: [usize; 5],
        gap_agree: [usize; 5],
        head_rows: Vec<usize>,
        head_agree: Vec<usize>,
    }
    let mut logit_acc: Vec<(String, Acc)> = Vec::new();

    let mut router_pairs = 0usize;
    let mut router_overlap = 0usize;
    let mut router_exact = 0usize;
    let mut layer_overlap: Vec<usize> = Vec::new();
    let mut layer_rows = 0usize;
    let mut top_k = 0usize;
    let mut m_pairs = [0usize; 5];
    let mut m_overlap = [0usize; 5];
    let mut m_exact = [0usize; 5];
    let mut router_rows = 0usize;

    for ca in &cases_a {
        let cb = cases_b
            .iter()
            .find(|c| c.name == ca.name)
            .ok_or_else(|| format!("{}: not in {}", ca.name, b.display()))?;
        let dim = meta_usize(&ca.meta, "dim")?;
        let present = |dir: &Path, n: &str| dir.join(format!("{n}.bin")).is_file();

        // hidden
        if present(&cb.dir, "hidden") {
            let ha = read_f32(&ca.dir.join("hidden.bin"))?;
            let hb = read_f32(&cb.dir.join("hidden.bin"))?;
            reject_non_finite(&format!("{} hidden (reference)", ca.name), &ha)?;
            reject_non_finite(&format!("{} hidden", cb.name), &hb)?;
            if ha.len() != hb.len() || ha.len() % dim != 0 {
                return Err(format!(
                    "{}: hidden has {} values against {}",
                    ca.name,
                    ha.len(),
                    hb.len()
                ));
            }
            for (r, (x, y)) in ha.chunks_exact(dim).zip(hb.chunks_exact(dim)).enumerate() {
                let c = cosine(x, y);
                cos.push(c);
                rel.push(relative_l2(x, y));
                if r < ca.prompt_rows {
                    cos_prompt.push(c);
                } else {
                    cos_generated.push(c);
                }
            }
        }

        // semantic and heads, every variant the other dump carries
        let names: Vec<String> = cb
            .meta
            .get("arrays")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        for name in names {
            let base = base_name(&name);
            if base != "semantic" && base != "heads" {
                continue;
            }
            let la = read_f32(&ca.dir.join(format!("{base}.bin")))?;
            let lb = read_f32(&cb.dir.join(format!("{name}.bin")))?;
            reject_non_finite(&format!("{} {base} (reference)", ca.name), &la)?;
            reject_non_finite(&format!("{} {name}", cb.name), &lb)?;
            if la.len() != lb.len() {
                return Err(format!(
                    "{}: {name} has {} values against the reference's {}",
                    ca.name,
                    lb.len(),
                    la.len()
                ));
            }
            let (width, per_row_heads) = if base == "semantic" {
                (meta_usize(&ca.meta, "semantic_width")?, 1)
            } else {
                (
                    meta_usize(&ca.meta, "head_vocab")?,
                    meta_usize(&ca.meta, "heads")?,
                )
            };
            let idx = match logit_acc.iter().position(|(n, _)| *n == name) {
                Some(i) => i,
                None => {
                    logit_acc.push((
                        name.clone(),
                        Acc {
                            rows: 0,
                            agree: 0,
                            kl: 0.0,
                            gap_rows: [0; 5],
                            gap_agree: [0; 5],
                            head_rows: vec![0; per_row_heads],
                            head_agree: vec![0; per_row_heads],
                        },
                    ));
                    logit_acc.len() - 1
                }
            };
            let acc = &mut logit_acc[idx].1;
            for (i, (x, y)) in la
                .chunks_exact(width)
                .zip(lb.chunks_exact(width))
                .enumerate()
            {
                let agree = argmax(x) == argmax(y);
                acc.rows += 1;
                acc.agree += usize::from(agree);
                acc.kl += kl(x, y);
                let g = bucket_of(&GAP_BUCKETS, top_gap(x), false);
                acc.gap_rows[g] += 1;
                acc.gap_agree[g] += usize::from(agree);
                let head = i % per_row_heads;
                acc.head_rows[head] += 1;
                acc.head_agree[head] += usize::from(agree);
            }
        }

        // router
        if present(&cb.dir, "used") {
            let layers = meta_usize(&ca.meta, "layers")?;
            let k = meta_usize(&ca.meta, "top_k")?;
            let ua = read_i16(&ca.dir.join("used.bin"))?;
            let ub = read_i16(&cb.dir.join("used.bin"))?;
            if ua.len() != ub.len() || ua.len() % (layers * k) != 0 {
                return Err(format!(
                    "{}: used has {} values against the reference's {}",
                    ca.name,
                    ub.len(),
                    ua.len()
                ));
            }
            let margins = if present(&ca.dir, "margin") {
                read_f32(&ca.dir.join("margin.bin"))?
            } else {
                Vec::new()
            };
            let rows = ua.len() / (layers * k);
            top_k = k;
            if layer_overlap.is_empty() {
                layer_overlap = vec![0; layers];
            }
            for r in 0..rows {
                layer_rows += 1;
                for l in 0..layers {
                    let at = (r * layers + l) * k;
                    let o = overlap(&ua[at..at + k], &ub[at..at + k]);
                    router_pairs += 1;
                    router_overlap += o;
                    router_exact += usize::from(o == k);
                    layer_overlap[l] += o;
                    if let Some(&m) = margins.get(r * layers + l) {
                        let bkt = bucket_of(&MARGIN_BUCKETS, f64::from(m), true);
                        m_pairs[bkt] += 1;
                        m_overlap[bkt] += o;
                        m_exact[bkt] += usize::from(o == k);
                    }
                }
            }
            router_rows += rows;
        }
    }

    if !cos.is_empty() {
        let mut sorted = cos.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        summary.hidden = Some(HiddenStat {
            rows: cos.len(),
            cos_mean: mean(&cos),
            cos_p1: percentile_sorted(&sorted, 0.01),
            cos_min: sorted[0],
            rel_l2_mean: mean(&rel),
            prompt_cos_mean: mean(&cos_prompt),
            generated_cos_mean: mean(&cos_generated),
        });
    }
    for (name, acc) in logit_acc {
        let rows = acc.rows.max(1) as f64;
        summary.logits.push(LogitStat {
            name,
            rows: acc.rows,
            top1: acc.agree as f64 / rows,
            kl: acc.kl / rows,
            by_gap: GAP_BUCKETS
                .iter()
                .enumerate()
                .map(|(i, &(label, _))| {
                    (
                        label,
                        acc.gap_rows[i],
                        acc.gap_agree[i] as f64 / acc.gap_rows[i].max(1) as f64,
                    )
                })
                .collect(),
            per_head: if acc.head_rows.len() > 1 {
                acc.head_rows
                    .iter()
                    .zip(&acc.head_agree)
                    .map(|(&n, &a)| a as f64 / n.max(1) as f64)
                    .collect()
            } else {
                Vec::new()
            },
        });
    }
    if router_pairs > 0 {
        summary.router = Some(RouterStat {
            rows: router_rows,
            layers: layer_overlap.len(),
            top_k,
            overlap: router_overlap as f64 / (router_pairs * top_k) as f64,
            exact: router_exact as f64 / router_pairs as f64,
            per_layer: layer_overlap
                .iter()
                .map(|&o| o as f64 / (layer_rows * top_k) as f64)
                .collect(),
            by_margin: MARGIN_BUCKETS
                .iter()
                .enumerate()
                .map(|(i, &(label, _))| {
                    (
                        label,
                        m_pairs[i],
                        m_overlap[i] as f64 / (m_pairs[i].max(1) * top_k) as f64,
                        m_exact[i] as f64 / m_pairs[i].max(1) as f64,
                    )
                })
                .collect(),
        });
    }
    Ok(summary)
}

fn render(summary: &Summary) -> String {
    let mut out = String::new();
    if let Some(h) = &summary.hidden {
        let _ = writeln!(
            out,
            "hidden: {} rows | cosine mean {:.5} p1 {:.5} min {:.5} | prompt rows {:.5} \
             generated rows {:.5} | relative L2 mean {:.4}",
            h.rows,
            h.cos_mean,
            h.cos_p1,
            h.cos_min,
            h.prompt_cos_mean,
            h.generated_cos_mean,
            h.rel_l2_mean
        );
    }
    for l in &summary.logits {
        let _ = writeln!(
            out,
            "{}: {} rows | top-1 agreement {:.4} | KL(ref||port) {:.4} nats",
            l.name, l.rows, l.top1, l.kl
        );
        let gaps: Vec<String> = l
            .by_gap
            .iter()
            .map(|(label, n, a)| format!("{label} {a:.3} (n={n})"))
            .collect();
        let _ = writeln!(out, "    by reference top-1 gap: {}", gaps.join(" | "));
        if !l.per_head.is_empty() {
            let heads: Vec<String> = l.per_head.iter().map(|a| format!("{a:.3}")).collect();
            let _ = writeln!(
                out,
                "    per head 1..{}: {}",
                l.per_head.len(),
                heads.join(" ")
            );
        }
    }
    if let Some(r) = &summary.router {
        let _ = writeln!(
            out,
            "router: {} rows x {} layers | top-{} overlap {:.4} | identical sets {:.4}",
            r.rows, r.layers, r.top_k, r.overlap, r.exact
        );
        for chunk in r.per_layer.chunks(10).enumerate() {
            let line: Vec<String> = chunk.1.iter().map(|o| format!("{o:.3}")).collect();
            let _ = writeln!(out, "    layers {:>2}..: {}", chunk.0 * 10, line.join(" "));
        }
        let buckets: Vec<String> = r
            .by_margin
            .iter()
            .map(|(label, n, o, e)| format!("{label} overlap {o:.3} exact {e:.3} (n={n})"))
            .collect();
        let _ = writeln!(out, "    by reference margin: {}", buckets.join(" | "));
    }
    out
}

// -- router isolation ---------------------------------------------------------

struct Selections {
    pairs: usize,
    overlap: usize,
    exact: usize,
}

impl Selections {
    fn new() -> Self {
        Selections {
            pairs: 0,
            overlap: 0,
            exact: 0,
        }
    }
    fn add(&mut self, a: &[i16], b: &[i16]) {
        let o = overlap(a, b);
        self.pairs += 1;
        self.overlap += o;
        self.exact += usize::from(o == a.len());
    }
    fn render(&self, k: usize) -> String {
        format!(
            "overlap {:.5} | identical sets {:.5} | {} (row, layer) pairs",
            self.overlap as f64 / (self.pairs.max(1) * k) as f64,
            self.exact as f64 / self.pairs.max(1) as f64,
            self.pairs
        )
    }
}

/// Router INT8 against BF16 on the inputs each layer saw in one reference run.
fn router_isolation(dump: &Path) -> Result<String, String> {
    let cases = load_cases(dump)?;
    let mut out = String::new();
    let mut rounding = Selections::new();
    let mut quant = Selections::new();
    let mut total = Selections::new();
    // INT8 against BF16-in-f32 by the f32 logit gap at the k-th place.
    const GAPS: [(&str, f64); 6] = [
        ("<0.01", 0.01),
        ("<0.03", 0.03),
        ("<0.1", 0.1),
        ("<0.3", 0.3),
        ("<1", 1.0),
        (">=1", f64::INFINITY),
    ];
    let mut g_pairs = [0usize; 6];
    let mut g_exact = [0usize; 6];
    let mut g_overlap = [0usize; 6];
    let mut k = 0;
    for case in &cases {
        let layers = meta_usize(&case.meta, "layers")?;
        k = meta_usize(&case.meta, "top_k")?;
        let used = read_i16(&case.dir.join("used.bin"))?;
        let f32s = read_i16(&case.dir.join("ids_f32.bin"))
            .map_err(|e| format!("{e} (the reference dump needs the isolate arrays)"))?;
        let int8 = read_i16(&case.dir.join("ids_int8.bin"))?;
        let gaps = read_f32(&case.dir.join("margin_f32.bin"))?;
        if used.len() != f32s.len() || used.len() != int8.len() || used.len() % (layers * k) != 0 {
            return Err(format!(
                "{}: the three selections differ in size",
                case.name
            ));
        }
        let rows = used.len() / (layers * k);
        if gaps.len() != rows * layers {
            return Err(format!("{}: margin_f32 has the wrong size", case.name));
        }
        for p in 0..rows * layers {
            let s = p * k..(p + 1) * k;
            rounding.add(&used[s.clone()], &f32s[s.clone()]);
            quant.add(&f32s[s.clone()], &int8[s.clone()]);
            total.add(&used[s.clone()], &int8[s.clone()]);
            let b = GAPS
                .iter()
                .position(|&(_, bound)| f64::from(gaps[p]) < bound)
                .unwrap_or(GAPS.len() - 1);
            let o = overlap(&f32s[s.clone()], &int8[s]);
            g_pairs[b] += 1;
            g_overlap[b] += o;
            g_exact[b] += usize::from(o == k);
        }
    }
    let _ = writeln!(
        out,
        "BF16 gate as run (bf16 logits) against the same weights in f32 (rounding and ties):\n    {}",
        rounding.render(k)
    );
    let _ = writeln!(
        out,
        "BF16 weights in f32 against INT8 weights in f32 (the intake's quantizer alone):\n    {}",
        quant.render(k)
    );
    let _ = writeln!(
        out,
        "BF16 gate as run against INT8 (what the intake changes in total):\n    {}",
        total.render(k)
    );
    let line: Vec<String> = GAPS
        .iter()
        .enumerate()
        .map(|(i, &(label, _))| {
            format!(
                "{label} overlap {:.3} exact {:.3} (n={})",
                g_overlap[i] as f64 / (g_pairs[i].max(1) * k) as f64,
                g_exact[i] as f64 / g_pairs[i].max(1) as f64,
                g_pairs[i]
            )
        })
        .collect();
    let _ = writeln!(
        out,
        "INT8 against f32 by the f32 logit gap between the 8th and 9th expert:\n    {}",
        line.join("\n    ")
    );
    out.push_str(&check_quantizer(dump)?);
    Ok(out)
}

/// The reference's Python re-implementation of the intake's router quantizer
/// against `compute::quantize_int8_affine` itself, on layer 0's weights.
fn check_quantizer(dump: &Path) -> Result<String, String> {
    let sim = dump.parent().ok_or("no parent")?.join("gate_sim");
    let weights = read_f32(&sim.join("gate0.f32"))?;
    let q = std::fs::read(sim.join("gate0_q.u8")).map_err(|e| e.to_string())?;
    let scale = read_f32(&sim.join("gate0_scale.f32"))?;
    let bias = read_f32(&sim.join("gate0_bias.f32"))?;
    let cols = 2048;
    let rows = weights.len() / cols;
    let groups = cols / 64;
    let (mut bad_q, mut bad_s, mut bad_b) = (0usize, 0usize, 0usize);
    for r in 0..rows {
        let row = compute::quant::quantize_int8_affine(&weights[r * cols..(r + 1) * cols]);
        for (&mine, &theirs) in row.packed.iter().zip(&q[r * cols..(r + 1) * cols]) {
            bad_q += usize::from(mine != theirs);
        }
        for g in 0..groups {
            bad_s += usize::from(compute::bf16_to_f32(row.scales[g]) != scale[r * groups + g]);
            bad_b += usize::from(compute::bf16_to_f32(row.biases[g]) != bias[r * groups + g]);
        }
    }
    if bad_q + bad_s + bad_b != 0 {
        return Err(format!(
            "the Python quantizer differs from compute::quantize_int8_affine: {bad_q} codes, \
             {bad_s} scales, {bad_b} biases of layer 0 ({rows} rows)"
        ));
    }
    Ok(format!(
        "quantizer check: layer 0's {rows} router rows quantize identically in Python and in \
         compute::quantize_int8_affine\n"
    ))
}

// -- the Rust run -------------------------------------------------------------

fn row_for(
    tower: &dyn TowerModel,
    backbone: &mut TrunkBackbone,
    frame: &[i32],
) -> Result<BackboneRow, Boxed> {
    let embeddings = tower.embeddings();
    if !embeddings.is_semantic(frame[0]) {
        return Ok(BackboneRow::Token(frame[0]));
    }
    let mut text = vec![0.0f32; backbone.hidden_size()];
    backbone.embedding_row(frame[0], &mut text)?;
    Ok(BackboneRow::Embedding(
        embeddings.embed_backbone(frame, &text),
    ))
}

/// A tower over one case, as the generation loop drives it. Returns the
/// semantic logits of the last prompt row and every generated row, and the
/// head logits of the rows whose next row is a recorded frame.
fn run_tower(
    tower: &mut dyn TowerModel,
    case: &Case,
    hidden: &[f32],
    dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), Boxed> {
    let (rows, system, prompt) = (&case.rows, case.system_rows, case.prompt_rows);
    tower.truncate(0)?;
    for (lo, hi) in [(0, system), (system, prompt)] {
        let labelled = hi.min(prompt - 1);
        if labelled > lo {
            let frames: Vec<&[i32]> = rows[lo..labelled].iter().map(Vec::as_slice).collect();
            let labels: Vec<&[i32]> = rows[lo + 1..=labelled].iter().map(Vec::as_slice).collect();
            tower.prefill(&hidden[lo * dim..labelled * dim], &frames, &labels)?;
        }
    }
    let (mut semantic, mut heads) = (Vec::new(), Vec::new());
    for t in prompt - 1..rows.len() {
        let h = &hidden[t * dim..(t + 1) * dim];
        semantic.extend(tower.semantic_logits(h)?);
        if t + 1 < rows.len() {
            let next = &rows[t + 1];
            let mut sample = |i: usize, head: &[f32]| {
                heads.extend_from_slice(head);
                next[2 + i]
            };
            tower.residual_codes(h, next[0], &rows[t], &mut sample)?;
        }
    }
    Ok((semantic, heads))
}

fn run(install: &Path, reference: &Path, out: &Path, slots: usize) -> Result<(), Boxed> {
    std::fs::create_dir_all(out)?;
    let trace_path = out.join("router_trace.json");
    // Read when the trunk opens, so set before it does.
    std::env::set_var("TURBOSPARK_ROUTER_TRACE", "1");
    std::env::set_var("TURBOSPARK_ROUTER_HIST", &trace_path);
    let _ = std::fs::remove_file(&trace_path);

    let cases = load_cases(reference)?;
    if cases.is_empty() {
        return Err("the reference dump has no cases".into());
    }
    for case in &cases {
        if case.rows.len() > CONTEXT {
            return Err(format!("{}: {} rows exceed {CONTEXT}", case.name, case.rows.len()).into());
        }
    }
    let mut cpu = Drama3::open(install)?;
    let mut metal = open_drama(install, CONTEXT)?;
    let trunk_dir = install.join(TRUNK_DIR);
    let arch = turbospark_repack::peek_manifest_arch(&trunk_dir)?;
    let trunk = HiddenTrunk::open(
        &trunk_dir,
        arch,
        CONTEXT,
        ExpertCacheSlots::Fixed(slots),
        ExpertResidency::Streamed,
    )?;
    let mut backbone = TrunkBackbone::new(trunk);
    let dim = backbone.hidden_size();

    let mut fed = 0usize;
    let mut offsets = Vec::new();
    for case in &cases {
        let reference_hidden = read_f32(&case.dir.join("hidden.bin"))?;
        let total = case.rows.len();
        if reference_hidden.len() != total * dim {
            return Err(format!("{}: reference hidden has the wrong size", case.name).into());
        }
        backbone.reset();
        let inputs = case
            .rows
            .iter()
            .map(|f| row_for(cpu.tower.as_ref(), &mut backbone, f))
            .collect::<Result<Vec<_>, _>>()?;
        let mut hidden = vec![0.0f32; total * dim];
        // The system block, then the rest of the prompt, as `generate` does.
        for (lo, hi) in [(0, case.system_rows), (case.system_rows, case.prompt_rows)] {
            if hi > lo {
                backbone.prefill(&inputs[lo..hi], &mut hidden[lo * dim..hi * dim])?;
            }
        }
        for t in case.prompt_rows..total {
            backbone.step(&inputs[t], &mut hidden[t * dim..(t + 1) * dim])?;
        }
        offsets.push((fed, total));
        fed += total;

        let dir = out.join(&case.name);
        std::fs::create_dir_all(&dir)?;
        write_f32(&dir.join("hidden.bin"), &hidden)?;
        let mut arrays = vec!["hidden".to_string()];
        // (suffix, Metal tower, the reference's hidden states, label)
        let combos = [
            ("", false, false, "own hidden, f32 tower"),
            ("_refh", false, true, "reference hidden, f32 tower"),
            ("_metal", true, false, "own hidden, Metal tower"),
            ("_metal_refh", true, true, "reference hidden, Metal tower"),
        ];
        for (suffix, on_metal, use_reference, label) in combos {
            let tower: &mut dyn TowerModel = if on_metal {
                metal.tower.as_mut()
            } else {
                cpu.tower.as_mut()
            };
            let h: &[f32] = if use_reference {
                &reference_hidden
            } else {
                &hidden
            };
            let (semantic, heads) = run_tower(tower, case, h, dim)?;
            write_f32(&dir.join(format!("semantic{suffix}.bin")), &semantic)?;
            write_f32(&dir.join(format!("heads{suffix}.bin")), &heads)?;
            arrays.push(format!("semantic{suffix}"));
            arrays.push(format!("heads{suffix}"));
            println!(
                "{}: {label}: {} head rows",
                case.name,
                heads.len() / (9 * 1024)
            );
        }
        let meta = serde_json::json!({
            "variant": "rust",
            "case": case.name,
            "prompt_rows": case.prompt_rows,
            "system_rows": case.system_rows,
            "dim": dim,
            "semantic_width": case.meta["semantic_width"],
            "heads": case.meta["heads"],
            "head_vocab": case.meta["head_vocab"],
            "layers": case.meta["layers"],
            "top_k": case.meta["top_k"],
            "arrays": arrays,
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_vec(&meta)?)?;
    }

    // The trace is written when the runner drops.
    drop(backbone);
    let text = std::fs::read_to_string(&trace_path)
        .map_err(|e| format!("{}: {e}", trace_path.display()))?;
    let trace: serde_json::Value = serde_json::from_str(&text)?;
    let k = trace["top_k"].as_u64().ok_or("trace: top_k")? as usize;
    let layers = trace["trace"].as_array().ok_or("trace: no trace array")?;
    for (l, ids) in layers.iter().enumerate() {
        let n = ids.as_array().map_or(0, Vec::len);
        if n != fed * k {
            return Err(format!(
                "layer {l} recorded {} passes for {fed} rows fed: the trace does not line up",
                n / k.max(1)
            )
            .into());
        }
    }
    for (case, &(offset, total)) in cases.iter().zip(&offsets) {
        let mut used = vec![0i16; total * layers.len() * k];
        for (l, ids) in layers.iter().enumerate() {
            let ids = ids.as_array().ok_or("trace layer")?;
            for r in 0..total {
                for j in 0..k {
                    used[(r * layers.len() + l) * k + j] =
                        ids[(offset + r) * k + j].as_i64().ok_or("trace id")? as i16;
                }
            }
        }
        let dir = out.join(&case.name);
        write_i16(&dir.join("used.bin"), &used)?;
        let meta_path = dir.join("meta.json");
        let mut meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta_path)?)?;
        meta["arrays"]
            .as_array_mut()
            .ok_or("meta arrays")?
            .push("used".into());
        std::fs::write(meta_path, serde_json::to_vec(&meta)?)?;
    }
    println!(
        "rust dump: {} cases, {fed} rows, under {}",
        cases.len(),
        out.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result: Result<bool, Boxed> = (|| match args.first().map(String::as_str) {
        Some("run") if args.len() >= 4 => {
            let slots = args.get(4).map_or(Ok(16), |s| s.parse())?;
            run(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                slots,
            )?;
            Ok(true)
        }
        Some("compare") if args.len() >= 3 => {
            let summary = compare_dirs(Path::new(&args[1]), Path::new(&args[2]))?;
            print!("{}", render(&summary));
            if args.iter().any(|a| a == "--gate") {
                let Some(t) = FROZEN else {
                    println!("gate: the thresholds are not frozen yet, nothing was enforced");
                    return Ok(true);
                };
                let fails = check(&summary, &t);
                for f in &fails {
                    println!("gate FAIL: {f}");
                }
                if fails.is_empty() {
                    println!("gate: pass");
                }
                return Ok(fails.is_empty());
            }
            Ok(true)
        }
        Some("router-isolation") if args.len() >= 2 => {
            print!("{}", router_isolation(Path::new(&args[1]))?);
            Ok(true)
        }
        _ => Err(
            "usage: run <install> <reference-dump> <out> [slots] | compare <ref> <other> \
                  [--gate] | router-isolation <reference-dump>"
                .into(),
        ),
    })();
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_is_one_for_parallel_zero_for_orthogonal() {
        assert!((cosine(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-12);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-12);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn relative_l2_scales_with_the_difference() {
        assert_eq!(relative_l2(&[3.0, 4.0], &[3.0, 4.0]), 0.0);
        assert!((relative_l2(&[3.0, 4.0], &[0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn argmax_takes_the_first_of_equal_values() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn top_gap_is_first_minus_second() {
        assert!((top_gap(&[1.0, 4.0, 3.0]) - 1.0).abs() < 1e-9);
        assert!((top_gap(&[4.0, 4.0, 1.0])).abs() < 1e-9);
    }

    #[test]
    fn kl_is_zero_for_equal_logits_and_known_for_a_pair() {
        assert!(kl(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).abs() < 1e-12);
        // P = (0.5, 0.5), Q = (0.25, 0.75): 0.5 ln 2 + 0.5 ln (2/3).
        let p = [0.0f32, 0.0];
        let q = [(0.25f32 / 0.75).ln(), 0.0];
        let want = 0.5 * (0.5f64 / 0.25).ln() + 0.5 * (0.5f64 / 0.75).ln();
        assert!((kl(&p, &q) - want).abs() < 1e-6, "{} vs {want}", kl(&p, &q));
        // Not symmetric.
        assert!((kl(&p, &q) - kl(&q, &p)).abs() > 1e-3);
    }

    #[test]
    fn overlap_counts_shared_ids() {
        assert_eq!(overlap(&[1, 2, 3], &[3, 2, 9]), 2);
        assert_eq!(overlap(&[1, 2, 3], &[4, 5, 6]), 0);
        assert_eq!(overlap(&[1, 2, 3], &[3, 1, 2]), 3);
    }

    #[test]
    fn buckets_put_a_tie_apart_and_the_rest_by_bound() {
        assert_eq!(bucket_of(&MARGIN_BUCKETS, 0.0, true), 0);
        assert_eq!(bucket_of(&MARGIN_BUCKETS, 5e-5, true), 1);
        assert_eq!(bucket_of(&MARGIN_BUCKETS, 1e-4, true), 2);
        assert_eq!(bucket_of(&MARGIN_BUCKETS, 5.0, true), 4);
        assert_eq!(bucket_of(&GAP_BUCKETS, 0.0, false), 0);
        assert_eq!(bucket_of(&GAP_BUCKETS, 1.5, false), 3);
        assert_eq!(bucket_of(&GAP_BUCKETS, 9.0, false), 4);
    }

    // -- synthetic dumps -------------------------------------------------------

    const D: usize = 4;
    const SEM: usize = 3;
    const LAYERS: usize = 2;
    const K: usize = 2;
    const HV: usize = 3;

    fn dump(root: &Path, hidden: &[f32], semantic: &[f32], heads: &[f32], used: &[i16]) {
        let dir = root.join("c0");
        std::fs::create_dir_all(&dir).unwrap();
        write_f32(&dir.join("hidden.bin"), hidden).unwrap();
        write_f32(&dir.join("semantic.bin"), semantic).unwrap();
        write_f32(&dir.join("heads.bin"), heads).unwrap();
        write_i16(&dir.join("used.bin"), used).unwrap();
        let meta = serde_json::json!({
            "case": "c0", "prompt_rows": 2, "system_rows": 1, "dim": D,
            "semantic_width": SEM, "heads": 2, "head_vocab": HV, "layers": LAYERS,
            "top_k": K, "arrays": ["hidden", "semantic", "heads", "used"],
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
    }

    fn base() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<i16>) {
        // 4 rows (2 prompt, 2 generated), 3 semantic rows, 2 head rows of 2 heads.
        let hidden: Vec<f32> = (0..4 * D).map(|i| 1.0 + (i % 5) as f32).collect();
        let semantic = vec![3.0, 1.0, 0.0, 0.0, 4.0, 1.0, 1.0, 0.0, 5.0];
        let heads = vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0];
        let used: Vec<i16> = (0..4 * LAYERS * K).map(|i| (i % 7) as i16).collect();
        (hidden, semantic, heads, used)
    }

    fn scratch(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("fd3-parity-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn identical_dumps_agree_perfectly() {
        let (a, b) = (scratch("same-a"), scratch("same-b"));
        let (h, s, hd, u) = base();
        dump(&a, &h, &s, &hd, &u);
        dump(&b, &h, &s, &hd, &u);
        let sum = compare_dirs(&a, &b).unwrap();
        let hs = sum.hidden.clone().unwrap();
        assert!((hs.cos_mean - 1.0).abs() < 1e-12 && hs.rows == 4);
        let sem = sum.logits("semantic").unwrap();
        assert_eq!((sem.rows, sem.top1), (3, 1.0));
        assert!(sem.kl.abs() < 1e-12);
        let heads = sum.logits("heads").unwrap();
        assert_eq!((heads.rows, heads.top1), (4, 1.0));
        assert_eq!(heads.per_head, vec![1.0, 1.0]);
        let r = sum.router.clone().unwrap();
        assert_eq!((r.overlap, r.exact), (1.0, 1.0));
    }

    #[test]
    fn a_perturbed_dump_is_measured_not_passed() {
        let (a, b) = (scratch("pert-a"), scratch("pert-b"));
        let (h, s, hd, u) = base();
        dump(&a, &h, &s, &hd, &u);
        // Row 1's hidden flips sign, the second semantic row's argmax moves,
        // head 0 of the first head row moves, and one expert id changes in
        // every selection of row 0 layer 0.
        let mut h2 = h.clone();
        for x in &mut h2[D..2 * D] {
            *x = -*x;
        }
        let mut s2 = s.clone();
        s2[3..6].copy_from_slice(&[6.0, 0.0, 0.0]);
        let mut hd2 = hd.clone();
        hd2[0..3].copy_from_slice(&[0.0, 0.0, 5.0]);
        let mut u2 = u.clone();
        u2[0] += 1;
        dump(&b, &h2, &s2, &hd2, &u2);
        let sum = compare_dirs(&a, &b).unwrap();
        let hs = sum.hidden.clone().unwrap();
        assert!((hs.cos_min + 1.0).abs() < 1e-9);
        assert!((hs.cos_mean - 0.5).abs() < 1e-9, "{}", hs.cos_mean);
        let sem = sum.logits("semantic").unwrap();
        assert!((sem.top1 - 2.0 / 3.0).abs() < 1e-12);
        assert!(sem.kl > 0.1);
        let heads = sum.logits("heads").unwrap();
        assert!((heads.top1 - 0.75).abs() < 1e-12);
        assert_eq!(heads.per_head, vec![0.5, 1.0]);
        let r = sum.router.clone().unwrap();
        assert!(r.overlap < 1.0 && r.exact < 1.0);
        // 8 (row, layer) pairs... 4 rows x 2 layers: exactly one changed id.
        assert!((r.exact - 7.0 / 8.0).abs() < 1e-12, "{}", r.exact);
    }

    #[test]
    fn a_non_finite_value_is_an_error_not_agreement() {
        let (a, b) = (scratch("nan-a"), scratch("nan-b"));
        let (h, s, hd, u) = base();
        dump(&a, &h, &s, &hd, &u);
        let mut h2 = h.clone();
        h2[0] = f32::NAN;
        dump(&b, &h2, &s, &hd, &u);
        assert!(compare_dirs(&a, &b).unwrap_err().contains("non-finite"));
    }

    #[test]
    fn a_size_mismatch_and_a_self_comparison_are_errors() {
        let (a, b) = (scratch("size-a"), scratch("size-b"));
        let (h, s, hd, u) = base();
        dump(&a, &h, &s, &hd, &u);
        dump(&b, &h[..3 * D], &s, &hd, &u);
        assert!(compare_dirs(&a, &b).is_err());
        assert!(compare_dirs(&a, &a).unwrap_err().contains("same directory"));
    }

    fn summary(h: (f64, f64), sem: (f64, f64), heads: f64, router: f64) -> Summary {
        let logit = |name: &str, top1: f64, kl: f64| LogitStat {
            name: name.into(),
            rows: 10,
            top1,
            kl,
            by_gap: Vec::new(),
            per_head: Vec::new(),
        };
        Summary {
            hidden: Some(HiddenStat {
                rows: 10,
                cos_mean: h.0,
                cos_p1: h.1,
                cos_min: h.1,
                rel_l2_mean: 0.0,
                prompt_cos_mean: h.0,
                generated_cos_mean: h.0,
            }),
            logits: vec![
                logit(GATE_SEMANTIC, sem.0, sem.1),
                logit(GATE_HEADS, heads, 0.0),
            ],
            router: Some(RouterStat {
                rows: 10,
                layers: 40,
                top_k: 8,
                overlap: router,
                exact: router,
                per_layer: Vec::new(),
                by_margin: Vec::new(),
            }),
        }
    }

    const T: Thresholds = Thresholds {
        hidden_cos_mean: 0.99,
        hidden_cos_p1: 0.95,
        semantic_top1: 0.8,
        semantic_kl_max: 0.1,
        heads_top1: 0.7,
        router_overlap: 0.9,
    };

    #[test]
    fn the_gate_passes_numbers_at_the_thresholds_and_names_each_miss() {
        assert!(check(&summary((0.99, 0.95), (0.8, 0.1), 0.7, 0.9), &T).is_empty());
        let one = |s: Summary| check(&s, &T);
        assert_eq!(one(summary((0.98, 0.95), (0.8, 0.1), 0.7, 0.9)).len(), 1);
        assert_eq!(one(summary((0.99, 0.94), (0.8, 0.1), 0.7, 0.9)).len(), 1);
        assert_eq!(one(summary((0.99, 0.95), (0.79, 0.1), 0.7, 0.9)).len(), 1);
        assert_eq!(one(summary((0.99, 0.95), (0.8, 0.11), 0.7, 0.9)).len(), 1);
        assert_eq!(one(summary((0.99, 0.95), (0.8, 0.1), 0.69, 0.9)).len(), 1);
        assert_eq!(one(summary((0.99, 0.95), (0.8, 0.1), 0.7, 0.89)).len(), 1);
    }

    #[test]
    fn a_gate_that_read_nothing_does_not_pass() {
        assert_eq!(check(&Summary::default(), &T).len(), 4);
    }
}
