//! Ledger records as plottable series: the shape `tslab export` writes, the
//! HTML report draws, and a dashboard reads.
//!
//! A ledger kind's payload is shaped for the code that wrote it. A series is
//! shaped for a chart: an x axis, a y axis, points, and enough context
//! (model, family, chip, rev, eligibility) to label it and to refuse to
//! quote it. Both version-0 records (before `schema.rs`) and version-1
//! records adapt here, so old experiments plot next to new ones.
//!
//! Portable: it reads JSON and builds JSON, nothing else.

use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::bench::BenchResult;
use crate::schema::SCHEMA;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Axis {
    pub name: String,
    pub unit: String,
    /// `log2`, `linear` or `category`.
    pub scale: String,
}

fn axis(name: &str, unit: &str, scale: &str) -> Axis {
    Axis {
        name: name.into(),
        unit: unit.into(),
        scale: scale.into(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SPoint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub y: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lo: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub dirty: bool,
}

impl SPoint {
    fn xy(x: f64, y: f64) -> Self {
        Self {
            x: Some(x),
            label: None,
            y,
            lo: None,
            hi: None,
            n: None,
            dirty: false,
        }
    }

    fn cat(label: &str, y: f64) -> Self {
        Self {
            x: None,
            label: Some(label.into()),
            y,
            lo: None,
            hi: None,
            n: None,
            dirty: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Series {
    pub schema: &'static str,
    pub id: String,
    pub kind: String,
    pub experiment: String,
    pub source_kind: String,
    pub source_id: String,
    pub source_schema: u8,
    pub model: Option<String>,
    pub family: Option<String>,
    pub chip: Option<String>,
    pub rev: Option<String>,
    pub eligible: bool,
    pub quotable: bool,
    /// Series sharing a group belong on one chart.
    pub group: String,
    pub label: String,
    pub subject: Map<String, Value>,
    pub x: Axis,
    pub y: Axis,
    pub points: Vec<SPoint>,
    /// Reference values a chart draws as lines (an effect floor, a bar).
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub refs: Map<String, Value>,
}

/// How to treat paths in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Paths {
    Keep,
    /// Model directories become their last component, so an export carries
    /// no home directory.
    Basename,
}

fn model_name(rec: &Value, paths: Paths) -> Option<String> {
    let m = rec["model"]
        .as_str()
        .or_else(|| rec["record"]["model_dir"].as_str())?;
    Some(match paths {
        Paths::Keep => m.to_string(),
        Paths::Basename => std::path::Path::new(m)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| m.to_string()),
    })
}

/// Builder state for the series of one record.
struct Ctx<'a> {
    rec: &'a Value,
    experiment: &'a str,
    paths: Paths,
}

impl Ctx<'_> {
    fn series(&self, kind: &str, group: &str, label: &str, x: Axis, y: Axis) -> Series {
        let prov = if self.rec["provenance"].is_object() {
            &self.rec["provenance"]
        } else {
            &self.rec["record"]["provenance"]
        };
        let id = self.rec["id"].as_str().unwrap_or("?");
        Series {
            schema: SCHEMA,
            id: format!("{id}:{kind}:{group}:{label}"),
            kind: kind.into(),
            experiment: self.experiment.into(),
            source_kind: self.rec["kind"].as_str().unwrap_or("?").into(),
            source_id: id.into(),
            source_schema: if self.rec["schema"].is_string() { 1 } else { 0 },
            model: model_name(self.rec, self.paths),
            family: self.rec["family"].as_str().map(str::to_string),
            chip: prov["chip"].as_str().map(str::to_string),
            rev: prov["git_rev"]
                .as_str()
                .map(|r| r.chars().take(10).collect()),
            eligible: self.rec["eligible"].as_bool().unwrap_or(true),
            quotable: self.rec["quotable"].as_bool().unwrap_or(true),
            group: group.into(),
            label: label.into(),
            subject: Map::new(),
            x,
            y,
            points: Vec::new(),
            refs: Map::new(),
        }
    }
}

/// A version-0 `kernel_curve` result as a [`BenchResult`]. The two v0
/// shapes are the old int4 bench (`arms` keyed by label, points keyed `m`)
/// and the old attention bench (`points[].kernels[]`).
pub fn bench_v0(r: &Value) -> Option<BenchResult> {
    use crate::bench::{ArmCurve, Axis as BAxis, BenchResult as B, Point};
    use std::collections::BTreeMap;
    let mut subject = BTreeMap::new();
    match r["bench"].as_str()? {
        "int4_matmul" => {
            for k in ["rows", "cols", "weight_bytes", "copies", "one_row_gemv_ms"] {
                subject.insert(k.to_string(), r[k].clone());
            }
            let arms = r["arms"]
                .as_object()?
                .iter()
                .map(|(label, a)| {
                    if let Some(why) = a["unsupported"].as_str() {
                        return ArmCurve::unsupported(label, why);
                    }
                    ArmCurve {
                        label: label.clone(),
                        points: a["points"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|p| {
                                let mut pt = Point::new(p["m"].as_f64()?, p["ms"].as_f64()?);
                                pt.c_of_m = p["c_of_m"].as_f64();
                                pt.tflops = p["tflops"].as_f64();
                                pt.gb_s = p["weight_gb_s"].as_f64();
                                Some(pt)
                            })
                            .collect(),
                        fit: serde_json::from_value(a["fit"].clone()).ok(),
                        unsupported: None,
                    }
                })
                .collect();
            Some(B {
                bench: "int4_matmul".into(),
                case: format!("{}x{}", r["rows"], r["cols"]),
                subject,
                x: BAxis::new("m", "rows", "log2"),
                layout: crate::bench::INT4_AFFINE_LAYOUT.into(),
                arms,
                break_even: Vec::new(),
                notes: vec!["adapted from a version-0 record".into()],
            })
        }
        "attn_decode" => {
            for k in ["q_heads", "kv_heads", "head_dim", "layers", "gqa_supported"] {
                subject.insert(k.to_string(), r[k].clone());
            }
            let mut arms: Vec<ArmCurve> = Vec::new();
            for p in r["points"].as_array()? {
                let ctx = p["context"].as_f64()?;
                for k in p["kernels"].as_array()? {
                    let label = k["kernel"].as_str()?.to_string();
                    let mut pt = Point::new(ctx, k["ms"].as_f64()?);
                    pt.gb_s = k["kv_gb_s"].as_f64();
                    match arms.iter_mut().find(|a| a.label == label) {
                        Some(a) => a.points.push(pt),
                        None => arms.push(ArmCurve {
                            label,
                            points: vec![pt],
                            fit: None,
                            unsupported: None,
                        }),
                    }
                }
            }
            Some(B {
                bench: "attn_decode".into(),
                case: "full_attn".into(),
                subject,
                x: BAxis::new("context", "positions", "log2"),
                layout: "fp16_kv_linear".into(),
                arms,
                break_even: Vec::new(),
                notes: vec!["adapted from a version-0 record".into()],
            })
        }
        _ => None,
    }
}

/// The bench result in a `kernel_curve` record, either version.
pub fn bench_of(rec: &Value) -> Option<BenchResult> {
    serde_json::from_value(rec["result"].clone())
        .ok()
        .or_else(|| bench_v0(&rec["result"]))
}

fn kernel_series(c: &Ctx, out: &mut Vec<Series>) {
    let Some(r) = bench_of(c.rec) else { return };
    let group = format!("{} {}", r.bench, r.case);
    let x = axis(&r.x.name, &r.x.unit, &r.x.scale);
    let has_c = r
        .arms
        .iter()
        .any(|a| a.points.iter().any(|p| p.c_of_m.is_some()));
    for a in r.arms.iter().filter(|a| !a.points.is_empty()) {
        let mut s = c.series(
            "kernel_ms",
            &group,
            &a.label,
            x.clone(),
            axis("ms", "ms", "log2"),
        );
        s.subject = r
            .subject
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        s.subject.insert("bench".into(), json!(r.bench));
        s.subject.insert("case".into(), json!(r.case));
        s.subject.insert("layout".into(), json!(r.layout));
        s.points = a.points.iter().map(|p| SPoint::xy(p.x, p.ms)).collect();
        out.push(s.clone());
        if has_c {
            let mut cs = s;
            cs.kind = "kernel_c_of_m".into();
            cs.id = cs.id.replacen(":kernel_ms:", ":kernel_c_of_m:", 1);
            cs.y = axis("c(M)", "ratio", "linear");
            cs.points = a
                .points
                .iter()
                .filter_map(|p| Some(SPoint::xy(p.x, p.c_of_m?)))
                .collect();
            cs.refs.insert("one_row".into(), json!(1.0));
            out.push(cs);
        }
    }
}

fn profile_series(c: &Ctx, out: &mut Vec<Series>) {
    let Some(summary) = c.rec["summary"].as_object() else {
        return;
    };
    for (metric, unit) in [
        ("prefill_tok_s", "tok/s"),
        ("decode_tok_s", "tok/s"),
        ("audio_run_s", "s"),
        ("rtf", "ratio"),
    ] {
        let mut s = c.series(
            "profile_totals",
            metric,
            metric,
            axis("arm", "", "category"),
            axis(metric, unit, "linear"),
        );
        for (arm, v) in summary {
            if let Some(y) = v[metric].as_f64() {
                let mut p = SPoint::cat(arm, y);
                p.n = v["clean_runs"].as_u64();
                s.points.push(p);
            }
        }
        if !s.points.is_empty() {
            out.push(s);
        }
    }
    for phase in ["prefill_phases", "decode_phases"] {
        for (arm, v) in summary {
            let Some(buckets) = v[phase].as_object() else {
                continue;
            };
            let mut s = c.series(
                "profile_buckets",
                phase,
                arm,
                axis("bucket", "", "category"),
                axis("ms per token", "ms", "linear"),
            );
            s.points = buckets
                .iter()
                .filter_map(|(b, ms)| Some(SPoint::cat(b, ms.as_f64()?)))
                .collect();
            if !s.points.is_empty() {
                out.push(s);
            }
        }
    }
}

/// verdict, experiment, confirm and null_calibration all carry a verdict
/// report (null_calibration only its improvements).
fn verdict_series(c: &Ctx, out: &mut Vec<Series>) {
    let rec = c.rec;
    let v = &rec["verdict"];
    let improvements = v["improvements"]
        .as_array()
        .or_else(|| rec["improvements"].as_array());
    let Some(improvements) = improvements else {
        return;
    };
    let label = match rec["kind"].as_str() {
        Some("null_calibration") => format!("{} null", rec["arm"]["name"].as_str().unwrap_or("?")),
        Some("experiment" | "confirm") => rec["knob_env"]
            .as_str()
            .unwrap_or_else(|| rec["kind"].as_str().unwrap_or("?"))
            .to_string(),
        _ => format!(
            "{} vs {}",
            rec["baseline"]["name"].as_str().unwrap_or("?"),
            rec["candidate"]["name"].as_str().unwrap_or("?")
        ),
    };
    let metric = rec["metric"].as_str().unwrap_or("primary metric");
    let mut s = c.series(
        "verdict",
        "verdicts",
        &label,
        axis("pair", "index", "linear"),
        axis(&format!("improvement in {metric}"), "fraction", "linear"),
    );
    s.points = improvements
        .iter()
        .enumerate()
        .filter_map(|(i, y)| Some(SPoint::xy(i as f64, y.as_f64()?)))
        .collect();
    for k in ["median", "ci_low", "ci_high", "min_effect"] {
        if let Some(x) = v[k].as_f64() {
            s.refs.insert(k.into(), json!(x));
        }
    }
    if let Some(f) = rec["floor"].as_f64().or_else(|| rec["null_floor"].as_f64()) {
        s.refs.insert("null_floor".into(), json!(f));
    }
    for k in ["verdict", "decision"] {
        let val = if k == "verdict" {
            &v["verdict"]
        } else {
            &rec["decision"]
        };
        if let Some(t) = val.as_str() {
            s.subject.insert(k.into(), json!(t));
        }
    }
    out.push(s);
}

/// A sweep: one bar per candidate, the median improvement over the shared
/// baseline with its CI, the effect floor as a reference line.
fn sweep_series(c: &Ctx, out: &mut Vec<Series>) {
    let rec = c.rec;
    let Some(cands) = rec["candidates"].as_array() else {
        return;
    };
    let metric = rec["metric"].as_str().unwrap_or("primary metric");
    let label = format!(
        "{} vs {}",
        rec["sweep_id"].as_str().unwrap_or("sweep"),
        rec["baseline"]["name"].as_str().unwrap_or("?")
    );
    let mut s = c.series(
        "sweep",
        "sweeps",
        &label,
        axis("candidate", "", "category"),
        axis(&format!("improvement in {metric}"), "fraction", "linear"),
    );
    for cand in cands {
        let Some(name) = cand["arm"].as_str() else {
            continue;
        };
        let v = &cand["verdict"];
        let Some(median) = v["median"].as_f64().filter(|m| m.is_finite()) else {
            // An inconclusive candidate has no median: it stays in the table
            // as a point at zero, marked dirty, rather than vanishing.
            let mut p = SPoint::cat(name, 0.0);
            p.dirty = true;
            s.points.push(p);
            continue;
        };
        let mut p = SPoint::cat(name, median);
        p.lo = v["ci_low"].as_f64();
        p.hi = v["ci_high"].as_f64();
        p.n = v["pairs"].as_u64();
        p.dirty = v["verdict"].as_str() == Some("inconclusive");
        s.points.push(p);
    }
    if let Some(f) = rec["floor"].as_f64() {
        s.refs.insert("min_effect".into(), json!(f));
    }
    if let Some(b) = rec["best"]["arm"].as_str() {
        s.subject.insert("best".into(), json!(b));
    }
    if !s.points.is_empty() {
        out.push(s);
    }
}

fn dispatch_probe_series(c: &Ctx, out: &mut Vec<Series>) {
    let mut s = c.series(
        "dispatch_probe",
        "dispatch probe",
        "decode ms/token",
        axis("chained dispatches added per token", "count", "linear"),
        axis("decode ms per token", "ms", "linear"),
    );
    s.points = c.rec["points"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| {
            Some(SPoint::xy(
                p["added_dispatches"].as_f64()?,
                p["ms_per_token"].as_f64()?,
            ))
        })
        .collect();
    if let Some(us) = c.rec["us_per_chained_dispatch"].as_f64() {
        s.refs.insert("us_per_chained_dispatch".into(), json!(us));
    }
    if !s.points.is_empty() {
        out.push(s);
    }
}

/// One point per calibration: predicted against measured improvement. A
/// point on the diagonal is a projection the model agreed with; the floor is
/// the band either side of it that counts as agreement.
fn calibration_series(c: &Ctx, out: &mut Vec<Series>) {
    let cal = &c.rec["calibration"];
    let (Some(x), Some(y)) = (cal["predicted"].as_f64(), cal["measured"].as_f64()) else {
        return;
    };
    let label = format!(
        "{} {}->{} x{}",
        c.rec["bench"].as_str().unwrap_or("?"),
        c.rec["default_arm"].as_str().unwrap_or("?"),
        c.rec["mode_arm"].as_str().unwrap_or("?"),
        c.rec["x"]
    );
    let mut s = c.series(
        "calibration",
        "projection calibration",
        &label,
        axis("predicted improvement", "fraction", "linear"),
        axis("measured improvement", "fraction", "linear"),
    );
    s.subject.insert("miss".into(), cal["miss"].clone());
    if let Some(f) = cal["floor"].as_f64() {
        s.refs.insert("floor".into(), json!(f));
    }
    s.points.push(SPoint::xy(x, y));
    out.push(s);
}

fn round_cost_series(c: &Ctx, out: &mut Vec<Series>) {
    let group = format!(
        "round cost {} kv {}",
        c.rec["drafter"].as_str().unwrap_or("?"),
        c.rec["kv_bucket"]
    );
    let mut s = c.series(
        "round_cost",
        &group,
        c.rec["flags"].as_str().unwrap_or("default"),
        axis("verify width", "rows", "linear"),
        axis("ms per round", "ms", "linear"),
    );
    if let (Some(x), Some(y)) = (c.rec["width"].as_f64(), c.rec["ms_per_round"].as_f64()) {
        let mut p = SPoint::xy(x, y);
        p.n = c.rec["n"].as_u64();
        s.points.push(p);
        out.push(s);
    }
}

fn standin_series(c: &Ctx, out: &mut Vec<Series>) {
    let mut s = c.series(
        "standin",
        "stand-in",
        "in-situ block ms",
        axis("block", "", "category"),
        axis("ms per token", "ms", "linear"),
    );
    s.quotable = false;
    s.points = c.rec["blocks"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(b, v)| Some(SPoint::cat(b, v["ms_per_token"].as_f64()?)))
        .collect();
    if !s.points.is_empty() {
        out.push(s);
    }
}

fn trace_series(c: &Ctx, out: &mut Vec<Series>) {
    let mut s = c.series(
        "trace_shares",
        "trace",
        "shader share",
        axis("kernel", "", "category"),
        axis("share of shader samples", "fraction", "linear"),
    );
    s.quotable = false;
    s.points = c.rec["shaders"]
        .as_array()
        .into_iter()
        .flatten()
        .take(24)
        .filter_map(|k| Some(SPoint::cat(k["name"].as_str()?, k["share"].as_f64()?)))
        .collect();
    if !s.points.is_empty() {
        out.push(s);
    }
}

/// Audio stage medians per arm, from a profile summary.
fn audio_series(c: &Ctx, out: &mut Vec<Series>) {
    let Some(summary) = c.rec["summary"].as_object() else {
        return;
    };
    for (arm, v) in summary {
        let Some(stages) = v["audio_stages"].as_object() else {
            continue;
        };
        let mut s = c.series(
            "audio_stages",
            "audio stages",
            arm,
            axis("stage", "", "category"),
            axis("ms", "ms", "linear"),
        );
        s.points = stages
            .iter()
            .filter_map(|(k, ms)| Some(SPoint::cat(k, ms.as_f64()?)))
            .collect();
        if !s.points.is_empty() {
            out.push(s);
        }
    }
}

fn s_eligible(rec: &Value) -> bool {
    rec["eligible"].as_bool().unwrap_or(true)
}

/// Every series one ledger record yields (often none: a run record's
/// numbers reach charts through its summary record).
pub fn from_record(rec: &Value, experiment: &str, paths: Paths) -> Vec<Series> {
    let c = Ctx {
        rec,
        experiment,
        paths,
    };
    let mut out = Vec::new();
    match rec["kind"].as_str().unwrap_or("") {
        "kernel_curve" => kernel_series(&c, &mut out),
        "profile_summary" => {
            profile_series(&c, &mut out);
            audio_series(&c, &mut out);
        }
        "verdict" | "experiment" | "confirm" | "null_calibration" => verdict_series(&c, &mut out),
        "sweep" => sweep_series(&c, &mut out),
        "dispatch_probe" => dispatch_probe_series(&c, &mut out),
        "round_cost" => round_cost_series(&c, &mut out),
        "calibration" => calibration_series(&c, &mut out),
        "standin" => standin_series(&c, &mut out),
        "trace" => trace_series(&c, &mut out),
        _ => {}
    }
    // A record the host guard rejected stays visible, marked.
    if !s_eligible(rec) {
        for s in &mut out {
            for p in &mut s.points {
                p.dirty = true;
            }
        }
    }
    out
}

/// CSV, one row per point: the long format every plotting tool reads.
pub fn to_csv(series: &[Series]) -> String {
    let esc = |s: &str| {
        if s.contains([',', '"', '\n']) {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    };
    let mut out = String::from(
        "series_id,kind,experiment,group,label,model,family,chip,rev,eligible,quotable,x_name,x_unit,y_name,y_unit,x,x_label,y,lo,hi,n,dirty\n",
    );
    let opt = |v: Option<f64>| v.map_or(String::new(), |x| x.to_string());
    for s in series {
        for p in &s.points {
            let row = [
                esc(&s.id),
                esc(&s.kind),
                esc(&s.experiment),
                esc(&s.group),
                esc(&s.label),
                esc(s.model.as_deref().unwrap_or("")),
                esc(s.family.as_deref().unwrap_or("")),
                esc(s.chip.as_deref().unwrap_or("")),
                esc(s.rev.as_deref().unwrap_or("")),
                s.eligible.to_string(),
                s.quotable.to_string(),
                esc(&s.x.name),
                esc(&s.x.unit),
                esc(&s.y.name),
                esc(&s.y.unit),
                opt(p.x),
                esc(p.label.as_deref().unwrap_or("")),
                p.y.to_string(),
                opt(p.lo),
                opt(p.hi),
                p.n.map_or(String::new(), |n| n.to_string()),
                p.dirty.to_string(),
            ];
            out.push_str(&row.join(","));
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v0_int4() -> Value {
        json!({"id": "1-1-0", "kind": "kernel_curve", "model": "/Users/x/models/q.gturbo",
               "family": "qwen35", "eligible": true,
               "provenance": {"chip": "Apple M4 Max", "git_rev": "c4d6ee5e11df764591f9"},
               "result": {"bench": "int4_matmul", "rows": 48, "cols": 5120,
                          "weight_bytes": 1, "copies": 2, "one_row_gemv_ms": 0.1,
                          "arms": {"gemv_per_row": {"points": [
                                       {"m": 1, "ms": 0.1, "c_of_m": 1.0},
                                       {"m": 8, "ms": 0.8, "c_of_m": 1.0}],
                                     "fit": {"a": 0.0, "b": 0.1, "max_rel_residual": 0.0}},
                                   "steel_m32": {"unsupported": "no"}},
                          "break_even": []}})
    }

    #[test]
    fn a_v0_int4_record_becomes_ms_and_c_series() {
        let s = from_record(&v0_int4(), "e", Paths::Basename);
        assert_eq!(s.len(), 2, "{s:#?}");
        assert_eq!(s[0].kind, "kernel_ms");
        assert_eq!(s[0].source_schema, 0);
        assert_eq!(s[0].model.as_deref(), Some("q.gturbo"));
        assert_eq!(s[0].rev.as_deref(), Some("c4d6ee5e11"));
        assert_eq!(s[0].points.len(), 2);
        assert_eq!(s[1].kind, "kernel_c_of_m");
        assert_eq!(s[1].points[1].y, 1.0);
        assert!(s.iter().all(|x| x.group == "int4_matmul 48x5120"));
    }

    #[test]
    fn a_calibration_is_one_predicted_against_measured_point() {
        let rec = json!({"id": "4-1-0", "kind": "calibration", "schema": SCHEMA,
                         "bench": "gdn_delta", "default_arm": "prefill_lazy",
                         "mode_arm": "prefill_lazy", "x": 8,
                         "calibration": {"predicted": 0.035, "measured": 0.0407,
                                         "floor": 0.02, "miss": false}});
        let s = from_record(&rec, "e", Paths::Keep);
        assert_eq!(s.len(), 1, "{s:#?}");
        assert_eq!(s[0].kind, "calibration");
        assert_eq!(s[0].group, "projection calibration");
        assert_eq!(s[0].label, "gdn_delta prefill_lazy->prefill_lazy x8");
        assert_eq!((s[0].points[0].x, s[0].points[0].y), (Some(0.035), 0.0407));
        assert_eq!(s[0].refs["floor"], 0.02);
        assert_eq!(s[0].subject["miss"], false);
        // A record without the numbers yields nothing rather than a zero point.
        let bare = json!({"id": "5", "kind": "calibration", "calibration": {}});
        assert!(from_record(&bare, "e", Paths::Keep).is_empty());
    }

    #[test]
    fn a_v1_record_round_trips_through_its_bench_result() {
        let r = bench_v0(&v0_int4()["result"]).unwrap();
        let rec = json!({"id": "2-1-0", "kind": "kernel_curve", "schema": SCHEMA,
                         "eligible": false, "result": r});
        let s = from_record(&rec, "e", Paths::Keep);
        assert_eq!(s[0].source_schema, 1);
        assert!(
            s[0].points.iter().all(|p| p.dirty),
            "dirty runs stay marked"
        );
        assert!(!s[0].eligible);
    }

    #[test]
    fn v0_attention_points_regroup_by_kernel() {
        let rec = json!({"id": "3", "kind": "kernel_curve", "result": {
            "bench": "attn_decode", "q_heads": 24, "kv_heads": 4, "head_dim": 256,
            "layers": 16, "gqa_supported": true,
            "points": [{"context": 512, "kernels": [{"kernel": "barrier", "ms": 1.0},
                                                    {"kernel": "gqa", "ms": 0.5}]},
                       {"context": 1024, "kernels": [{"kernel": "barrier", "ms": 2.0},
                                                     {"kernel": "gqa", "ms": 0.6}]}]}});
        let s = from_record(&rec, "e", Paths::Keep);
        assert_eq!(s.len(), 2);
        assert_eq!(s[1].label, "gqa");
        assert_eq!(
            s[1].points.iter().map(|p| p.y).collect::<Vec<_>>(),
            vec![0.5, 0.6]
        );
    }

    #[test]
    fn verdict_and_profile_records() {
        let v = json!({"id": "4", "kind": "verdict", "metric": "prefill_tok_s",
                       "baseline": {"name": "off"}, "candidate": {"name": "on"},
                       "null_floor": 0.01,
                       "verdict": {"improvements": [0.05, 0.06, 0.04], "median": 0.05,
                                   "ci_low": 0.04, "ci_high": 0.06, "min_effect": 0.02,
                                   "verdict": "win"}});
        let s = from_record(&v, "e", Paths::Keep);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].label, "off vs on");
        assert_eq!(s[0].refs["null_floor"], 0.01);
        assert_eq!(s[0].subject["verdict"], "win");
        let p = json!({"id": "5", "kind": "profile_summary", "summary": {
            "a": {"clean_runs": 2, "prefill_tok_s": 80.0, "decode_tok_s": 40.0,
                  "prefill_phases": {"gpu wait (layer cb1)": 2.5}},
            "b": {"refused": "no"}}});
        let s = from_record(&p, "e", Paths::Keep);
        let kinds: Vec<_> = s.iter().map(|x| x.kind.as_str()).collect();
        assert_eq!(
            kinds,
            ["profile_totals", "profile_totals", "profile_buckets"]
        );
        assert_eq!(s[0].points[0].n, Some(2));
    }

    #[test]
    fn csv_quotes_and_counts_rows() {
        let s = from_record(&v0_int4(), "e,x", Paths::Keep);
        let csv = to_csv(&s);
        assert_eq!(csv.lines().count(), 1 + 4);
        assert!(csv.lines().nth(1).unwrap().contains("\"e,x\""));
    }
}
