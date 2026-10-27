//! `tslab report --html`: one self-contained page of SVG charts from series.
//!
//! No scripts, fonts or stylesheets from anywhere: the file opens offline, in
//! a browser pane, or attached to a review, and renders the same. Colors are
//! the dataviz skill's validated reference palette, assigned to series in a
//! fixed order per chart and defined once as CSS tokens with a dark mode.
//! Hover is the SVG `<title>` on each mark; every chart also carries its
//! numbers as a table, so nothing depends on color or hover alone.
//!
//! Chart form follows the series kind: curves (kernel ms, c(M), probe
//! slope, round cost) are lines, buckets and stages are horizontal bars,
//! verdicts are a forest of pair improvements against the effect floor.
//! Dirty points are hollow; unquotable series say so in the chart title.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::series::Series;

const W: f64 = 680.0;
const LINE_H: f64 = 300.0;
const PAD_L: f64 = 64.0;
const PAD_R: f64 = 16.0;
const PAD_T: f64 = 12.0;
const PAD_B: f64 = 40.0;
/// Categorical slots in the reference order. A chart with more series than
/// slots folds the rest into its table instead of cycling hues.
const SLOTS: usize = 8;

pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn fmt_num(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 {
        "0".into()
    } else if (v.fract() == 0.0 && a < 1e7) || a >= 1000.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else if a >= 0.1 {
        format!("{v:.3}")
    } else {
        format!("{v:.2e}")
    }
}

/// A numeric scale onto `[lo_px, hi_px]`.
#[derive(Clone, Copy)]
struct Scale {
    log: bool,
    d0: f64,
    d1: f64,
    r0: f64,
    r1: f64,
}

impl Scale {
    fn new(values: &[f64], log: bool, r0: f64, r1: f64, include_zero: bool) -> Self {
        let log = log && values.iter().all(|v| *v > 0.0);
        let f = |v: f64| if log { v.log2() } else { v };
        let mut d0 = values.iter().copied().map(f).fold(f64::INFINITY, f64::min);
        let mut d1 = values
            .iter()
            .copied()
            .map(f)
            .fold(f64::NEG_INFINITY, f64::max);
        if !d0.is_finite() {
            (d0, d1) = (0.0, 1.0);
        }
        if include_zero && !log {
            d0 = d0.min(0.0);
            d1 = d1.max(0.0);
        }
        if d1 - d0 < 1e-12 {
            d0 -= 0.5;
            d1 += 0.5;
        } else if !log {
            let pad = 0.05 * (d1 - d0);
            if !(include_zero && d0 == 0.0) {
                d0 -= pad;
            }
            d1 += pad;
        }
        Self {
            log,
            d0,
            d1,
            r0,
            r1,
        }
    }

    fn at(&self, v: f64) -> f64 {
        let v = if self.log { v.log2() } else { v };
        self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)
    }

    fn ticks(&self) -> Vec<f64> {
        if self.log {
            let (a, b) = (self.d0.floor() as i32, self.d1.ceil() as i32);
            let step = ((b - a) as f64 / 6.0).ceil().max(1.0) as i32;
            (a..=b)
                .step_by(step as usize)
                .map(|e| 2f64.powi(e))
                .filter(|t| {
                    let l = t.log2();
                    l >= self.d0 - 1e-9 && l <= self.d1 + 1e-9
                })
                .collect()
        } else {
            let span = self.d1 - self.d0;
            let raw = span / 5.0;
            let mag = 10f64.powf(raw.log10().floor());
            let step = [1.0, 2.0, 2.5, 5.0, 10.0]
                .iter()
                .map(|m| m * mag)
                .find(|s| span / s <= 6.0)
                .unwrap_or(mag * 10.0);
            let mut t = (self.d0 / step).ceil() * step;
            let mut out = Vec::new();
            while t <= self.d1 + 1e-12 {
                out.push(if t.abs() < step * 1e-9 { 0.0 } else { t });
                t += step;
            }
            out
        }
    }
}

pub(crate) const CSS: &str = r#"
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b;
  --text-secondary: #52514e; --text-muted: #898781; --grid: #e1e0d9;
  --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --good: #006300; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec;
    --text-secondary: #c3c2b7; --text-muted: #898781; --grid: #2c2c2a;
    --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
    --good: #0ca30c; --critical: #d03b3b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec;
  --text-secondary: #c3c2b7; --text-muted: #898781; --grid: #2c2c2a;
  --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  --good: #0ca30c; --critical: #d03b3b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 8px; border-top: 1px solid var(--border); padding-top: 16px; }
h3 { font-size: 14px; margin: 0 0 2px; }
.meta, .sub { color: var(--text-secondary); font-size: 12px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 12px 8px; margin: 12px 0; }
svg { width: 100%; height: auto; display: block; }
svg text { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 4px 0; font-size: 12px;
  color: var(--text-secondary); }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px;
  vertical-align: -1px; }
.badge { display: inline-block; font-size: 11px; padding: 0 6px; border-radius: 4px;
  border: 1px solid var(--border); color: var(--text-secondary); margin-left: 6px; }
details { margin-top: 6px; font-size: 12px; color: var(--text-secondary); }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 2px 8px 2px 0; border-bottom: 1px solid var(--border); }
.scroll { overflow-x: auto; }
"#;

pub(crate) fn color(i: usize) -> String {
    format!("var(--s{})", i % SLOTS + 1)
}

pub(crate) fn legend(series: &[&Series]) -> String {
    if series.len() < 2 {
        return String::new();
    }
    let mut out = String::from("<div class=\"legend\">");
    for (i, s) in series.iter().enumerate().take(SLOTS) {
        let _ = write!(
            out,
            "<span><span class=\"key\" style=\"background:{}\"></span>{}</span>",
            color(i),
            esc(&s.label)
        );
    }
    if series.len() > SLOTS {
        let _ = write!(
            out,
            "<span>+{} more in the table</span>",
            series.len() - SLOTS
        );
    }
    out.push_str("</div>");
    out
}

pub(crate) fn table(series: &[&Series]) -> String {
    let mut out = String::from(
        "<details><summary>Table</summary><div class=\"scroll\"><table><tr><th>series</th><th>x</th><th>y</th><th>dirty</th></tr>",
    );
    for s in series {
        for p in &s.points {
            let x = p
                .label
                .clone()
                .or_else(|| p.x.map(fmt_num))
                .unwrap_or_default();
            let _ = write!(
                out,
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                esc(&s.label),
                esc(&x),
                fmt_num(p.y),
                if p.dirty { "yes" } else { "" }
            );
        }
    }
    out.push_str("</table></div></details>");
    out
}

pub(crate) fn line_chart(series: &[&Series]) -> String {
    let shown: Vec<&Series> = series.iter().copied().take(SLOTS).collect();
    let xs: Vec<f64> = shown
        .iter()
        .flat_map(|s| s.points.iter().filter_map(|p| p.x))
        .collect();
    let ys: Vec<f64> = shown
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.y))
        .filter(|y| y.is_finite())
        .collect();
    let s0 = shown[0];
    let sx = Scale::new(&xs, s0.x.scale == "log2", PAD_L, W - PAD_R, false);
    let sy = Scale::new(&ys, s0.y.scale == "log2", LINE_H - PAD_B, PAD_T, true);
    let mut svg = format!(
        "<svg viewBox=\"0 0 {W} {LINE_H}\" role=\"img\" aria-label=\"{} by {}\">",
        esc(&s0.y.name),
        esc(&s0.x.name)
    );
    for t in sy.ticks() {
        let y = sy.at(t);
        let _ = write!(
            svg,
            "<line class=\"grid\" x1=\"{PAD_L}\" x2=\"{}\" y1=\"{y:.1}\" y2=\"{y:.1}\"/><text x=\"{}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>",
            W - PAD_R,
            PAD_L - 6.0,
            y + 3.5,
            fmt_num(t)
        );
    }
    for t in sx.ticks() {
        let x = sx.at(t);
        let _ = write!(
            svg,
            "<text x=\"{x:.1}\" y=\"{}\" text-anchor=\"middle\">{}</text>",
            LINE_H - PAD_B + 16.0,
            fmt_num(t)
        );
    }
    let _ = write!(
        svg,
        "<line class=\"axis\" x1=\"{PAD_L}\" x2=\"{}\" y1=\"{b}\" y2=\"{b}\"/>",
        W - PAD_R,
        b = LINE_H - PAD_B
    );
    let _ = write!(
        svg,
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\">{} ({})</text>",
        (PAD_L + W - PAD_R) / 2.0,
        LINE_H - 6.0,
        esc(&s0.x.name),
        esc(&s0.x.unit)
    );
    // A reference line at y = 1 for c(M): the one-row unit.
    if let Some(r) = s0.refs.get("one_row").and_then(|v| v.as_f64()) {
        if (sy.at(r) - PAD_T) > -1.0 && sy.at(r) <= LINE_H - PAD_B + 1.0 {
            let y = sy.at(r);
            let _ = write!(
                svg,
                "<line x1=\"{PAD_L}\" x2=\"{}\" y1=\"{y:.1}\" y2=\"{y:.1}\" stroke=\"var(--text-muted)\" stroke-dasharray=\"4 3\"/>",
                W - PAD_R
            );
        }
    }
    for (i, s) in shown.iter().enumerate() {
        let c = color(i);
        let mut pts: Vec<(f64, f64, bool, f64, f64)> = s
            .points
            .iter()
            .filter_map(|p| Some((sx.at(p.x?), sy.at(p.y), p.dirty, p.x?, p.y)))
            .filter(|p| p.1.is_finite())
            .collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        if pts.len() > 1 {
            let d: Vec<String> = pts
                .iter()
                .enumerate()
                .map(|(j, p)| format!("{}{:.1},{:.1}", if j == 0 { "M" } else { "L" }, p.0, p.1))
                .collect();
            let _ = write!(
                svg,
                "<path d=\"{}\" fill=\"none\" stroke=\"{c}\" stroke-width=\"2\" stroke-linejoin=\"round\"/>",
                d.join("")
            );
        }
        for p in &pts {
            let fill = if p.2 { "var(--surface-1)" } else { c.as_str() };
            let _ = write!(
                svg,
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"4\" fill=\"{fill}\" stroke=\"{c}\" stroke-width=\"2\"><title>{}: {} = {}, {} = {}{}</title></circle>",
                p.0,
                p.1,
                esc(&s.label),
                esc(&s.x.name),
                fmt_num(p.3),
                esc(&s.y.name),
                fmt_num(p.4),
                if p.2 { " (dirty run)" } else { "" }
            );
        }
    }
    svg.push_str("</svg>");
    svg
}

/// Horizontal grouped bars: one row per category, one bar per series.
pub(crate) fn bar_chart(series: &[&Series]) -> String {
    let shown: Vec<&Series> = series.iter().copied().take(SLOTS).collect();
    let mut cats: Vec<String> = Vec::new();
    for s in &shown {
        for p in &s.points {
            if let Some(l) = &p.label {
                if !cats.contains(l) {
                    cats.push(l.clone());
                }
            }
        }
    }
    let bar = 12.0;
    let gap = 2.0;
    let row = shown.len() as f64 * (bar + gap) + 10.0;
    let label_w = 190.0;
    let h = PAD_T + cats.len() as f64 * row + 28.0;
    let ys: Vec<f64> = shown
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.y))
        .collect();
    let sx = Scale::new(&ys, false, label_w, W - PAD_R - 48.0, true);
    let mut svg = format!(
        "<svg viewBox=\"0 0 {W} {h:.0}\" role=\"img\" aria-label=\"{}\">",
        esc(&shown[0].y.name)
    );
    let base = sx.at(0.0);
    for t in sx.ticks() {
        let x = sx.at(t);
        let _ = write!(
            svg,
            "<line class=\"grid\" x1=\"{x:.1}\" x2=\"{x:.1}\" y1=\"{PAD_T}\" y2=\"{:.1}\"/><text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
            h - 24.0,
            h - 10.0,
            fmt_num(t)
        );
    }
    for (ci, cat) in cats.iter().enumerate() {
        let top = PAD_T + ci as f64 * row;
        let _ = write!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" style=\"fill:var(--text-secondary)\">{}</text>",
            label_w - 8.0,
            top + row / 2.0,
            esc(cat)
        );
        for (si, s) in shown.iter().enumerate() {
            let Some(p) = s.points.iter().find(|p| p.label.as_deref() == Some(cat)) else {
                continue;
            };
            let y = top + 4.0 + si as f64 * (bar + gap);
            let x1 = sx.at(p.y);
            let (x, w) = if x1 >= base {
                (base, x1 - base)
            } else {
                (x1, base - x1)
            };
            let c = color(si);
            let fill = if p.dirty {
                "var(--surface-1)"
            } else {
                c.as_str()
            };
            let _ = write!(
                svg,
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{bar}\" rx=\"2\" fill=\"{fill}\" stroke=\"{c}\" stroke-width=\"{}\"><title>{} {}: {}{}</title></rect>",
                w.max(1.0),
                if p.dirty { 1.5 } else { 0.0 },
                esc(&s.label),
                esc(cat),
                fmt_num(p.y),
                if p.dirty { " (dirty run)" } else { "" }
            );
            // Direct value label: the table has every number, the bar end
            // gets one so the chart reads without hover.
            let _ = write!(
                svg,
                "<text x=\"{:.1}\" y=\"{:.1}\">{}</text>",
                x + w.max(1.0) + 4.0,
                y + bar - 2.0,
                fmt_num(p.y)
            );
        }
    }
    let _ = write!(
        svg,
        "<line class=\"axis\" x1=\"{base:.1}\" x2=\"{base:.1}\" y1=\"{PAD_T}\" y2=\"{:.1}\"/>",
        h - 24.0
    );
    svg.push_str("</svg>");
    svg
}

/// One row per verdict: each pair's improvement as a dot, the bootstrap CI
/// as a bar, the median as a tick, and the effect floor as dashed lines.
pub(crate) fn forest(series: &[&Series]) -> String {
    let row = 30.0;
    let label_w = 190.0;
    let h = PAD_T + series.len() as f64 * row + 30.0;
    let mut vals: Vec<f64> = vec![0.0];
    for s in series {
        vals.extend(s.points.iter().map(|p| p.y));
        for k in ["ci_low", "ci_high", "min_effect"] {
            if let Some(v) = s.refs.get(k).and_then(|v| v.as_f64()) {
                vals.push(v);
                if k == "min_effect" {
                    vals.push(-v);
                }
            }
        }
    }
    let sx = Scale::new(&vals, false, label_w, W - PAD_R, true);
    let mut svg = format!("<svg viewBox=\"0 0 {W} {h:.0}\" role=\"img\" aria-label=\"verdicts\">");
    for t in sx.ticks() {
        let x = sx.at(t);
        let _ = write!(
            svg,
            "<line class=\"grid\" x1=\"{x:.1}\" x2=\"{x:.1}\" y1=\"{PAD_T}\" y2=\"{:.1}\"/><text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{:+.1}%</text>",
            h - 26.0,
            h - 10.0,
            t * 100.0
        );
    }
    let zero = sx.at(0.0);
    let _ = write!(
        svg,
        "<line class=\"axis\" x1=\"{zero:.1}\" x2=\"{zero:.1}\" y1=\"{PAD_T}\" y2=\"{:.1}\"/>",
        h - 26.0
    );
    for (i, s) in series.iter().enumerate() {
        let y = PAD_T + i as f64 * row + row / 2.0;
        let verdict = s
            .subject
            .get("verdict")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let decision = s.subject.get("decision").and_then(|v| v.as_str());
        let tag = decision.map_or(verdict.to_string(), |d| format!("{verdict}, {d}"));
        let _ = write!(
            svg,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" style=\"fill:var(--text-secondary)\">{}</text><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\">{}</text>",
            label_w - 8.0,
            y - 1.0,
            esc(&s.label),
            label_w - 8.0,
            y + 11.0,
            esc(&tag)
        );
        if let Some(f) = s.refs.get("min_effect").and_then(|v| v.as_f64()) {
            for v in [f, -f] {
                let x = sx.at(v);
                let _ = write!(
                    svg,
                    "<line x1=\"{x:.1}\" x2=\"{x:.1}\" y1=\"{:.1}\" y2=\"{:.1}\" stroke=\"var(--text-muted)\" stroke-dasharray=\"3 3\"><title>effect floor {:+.2}%</title></line>",
                    y - row / 2.0 + 3.0,
                    y + row / 2.0 - 3.0,
                    v * 100.0
                );
            }
        }
        if let (Some(lo), Some(hi)) = (
            s.refs.get("ci_low").and_then(|v| v.as_f64()),
            s.refs.get("ci_high").and_then(|v| v.as_f64()),
        ) {
            let _ = write!(
                svg,
                "<line x1=\"{:.1}\" x2=\"{:.1}\" y1=\"{y:.1}\" y2=\"{y:.1}\" stroke=\"var(--s1)\" stroke-width=\"2\" stroke-linecap=\"round\"><title>95% CI {:+.2}% to {:+.2}%</title></line>",
                sx.at(lo),
                sx.at(hi),
                lo * 100.0,
                hi * 100.0
            );
        }
        for p in &s.points {
            let fill = if p.dirty {
                "var(--surface-1)"
            } else {
                "var(--s1)"
            };
            let _ = write!(
                svg,
                "<circle cx=\"{:.1}\" cy=\"{y:.1}\" r=\"4\" fill=\"{fill}\" stroke=\"var(--surface-1)\" stroke-width=\"2\"><title>pair {}: {:+.2}%</title></circle>",
                sx.at(p.y),
                p.x.unwrap_or(0.0),
                p.y * 100.0
            );
        }
        if let Some(m) = s.refs.get("median").and_then(|v| v.as_f64()) {
            let x = sx.at(m);
            let _ = write!(
                svg,
                "<line x1=\"{x:.1}\" x2=\"{x:.1}\" y1=\"{:.1}\" y2=\"{:.1}\" stroke=\"var(--text-primary)\" stroke-width=\"2\"><title>median {:+.2}%</title></line>",
                y - 8.0,
                y + 8.0,
                m * 100.0
            );
        }
    }
    svg.push_str("</svg>");
    svg
}

pub(crate) fn chart_kind(kind: &str) -> &'static str {
    match kind {
        "verdict" => "forest",
        "profile_buckets" | "profile_totals" | "audio_stages" | "trace_shares" | "standin"
        | "sweep" => "bars",
        _ => "line",
    }
}

/// Series grouped per (kind, group) in first-seen order: the members of one
/// chart. Across experiments the same key lands series from several
/// experiments on one chart, which is what the dashboard's compare view
/// wants; `render` groups within one experiment.
pub(crate) fn group_series<'a>(list: &[&'a Series]) -> Vec<((String, String), Vec<&'a Series>)> {
    let mut groups: Vec<((String, String), Vec<&'a Series>)> = Vec::new();
    for s in list {
        let key = (s.kind.clone(), s.group.clone());
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(s),
            None => groups.push((key, vec![s])),
        }
    }
    groups
}

/// One chart card: title, provenance line, legend, the chart, the table.
pub(crate) fn card(kind: &str, group: &str, members: &[&Series]) -> String {
    let s0 = members[0];
    let quotable = members.iter().all(|s| s.quotable);
    let mut sub = Vec::new();
    for v in [&s0.model, &s0.family, &s0.chip, &s0.rev]
        .into_iter()
        .flatten()
    {
        sub.push(esc(v));
    }
    let mut body = String::new();
    let _ = write!(
        body,
        "<div class=\"card\"><h3>{} <span class=\"sub\">{}</span>{}</h3><div class=\"sub\">{}</div>",
        esc(group),
        esc(kind),
        if quotable {
            ""
        } else {
            "<span class=\"badge\">unquotable</span>"
        },
        sub.join(" &middot; ")
    );
    body.push_str(&legend(members));
    body.push_str(&match chart_kind(kind) {
        "forest" => forest(members),
        "bars" => bar_chart(members),
        _ => line_chart(members),
    });
    body.push_str(&table(members));
    body.push_str("</div>");
    body
}

/// Every chart card of one list of series, grouped by (kind, group).
pub(crate) fn cards(list: &[&Series]) -> String {
    let mut body = String::new();
    for ((kind, group), members) in group_series(list) {
        body.push_str(&card(&kind, &group, &members));
    }
    body
}

/// The whole page. `title` names it; series are grouped per experiment,
/// then per (kind, group), in first-seen order.
pub fn render(title: &str, generated: &str, series: &[Series]) -> String {
    let mut by_exp: BTreeMap<&str, Vec<&Series>> = BTreeMap::new();
    for s in series {
        by_exp.entry(s.experiment.as_str()).or_default().push(s);
    }
    let mut body = String::new();
    let _ = write!(
        body,
        "<h1>{}</h1><div class=\"meta\">{} series from {} experiment(s), schema {}, generated {}. Hollow marks are runs the host guard rejected.</div>",
        esc(title),
        series.len(),
        by_exp.len(),
        crate::schema::SCHEMA,
        esc(generated)
    );
    if series.is_empty() {
        body.push_str("<p>No plottable records.</p>");
    }
    for (exp, list) in &by_exp {
        let _ = write!(body, "<h2>{}</h2>", esc(exp));
        body.push_str(&cards(list));
    }
    page(title, &body, "")
}

/// The document shell every page shares: the CSS, an optional extra style
/// block, and the body inside `<main>`. Nothing is fetched from anywhere.
pub(crate) fn page(title: &str, body: &str, extra_css: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{}</title><style>{CSS}{extra_css}</style></head><body><main>{body}</main></body></html>\n",
        esc(title)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::{from_record, Paths};
    use serde_json::json;

    fn sample() -> Vec<Series> {
        let k = json!({"id": "1", "kind": "kernel_curve", "result": {
            "bench": "attn_decode", "q_heads": 24, "kv_heads": 4, "head_dim": 256,
            "layers": 16, "gqa_supported": true,
            "points": [{"context": 512, "kernels": [{"kernel": "barrier", "ms": 1.0},
                                                    {"kernel": "gqa", "ms": 0.5}]},
                       {"context": 2048, "kernels": [{"kernel": "barrier", "ms": 3.0},
                                                     {"kernel": "gqa", "ms": 0.9}]}]}});
        let v = json!({"id": "2", "kind": "verdict", "eligible": false,
                       "baseline": {"name": "a<b"}, "candidate": {"name": "c"},
                       "verdict": {"improvements": [0.05, -0.01], "median": 0.02,
                                   "ci_low": -0.01, "ci_high": 0.05, "min_effect": 0.02,
                                   "verdict": "noise"}});
        let p = json!({"id": "3", "kind": "profile_summary", "summary": {
            "x": {"prefill_tok_s": 80.0}, "y": {"prefill_tok_s": 90.0}}});
        let mut out = from_record(&k, "exp-1", Paths::Keep);
        out.extend(from_record(&v, "exp-1", Paths::Keep));
        out.extend(from_record(&p, "exp-2", Paths::Keep));
        out
    }

    #[test]
    fn renders_every_chart_form_self_contained() {
        let html = render("tslab <report>", "now", &sample());
        assert!(html.starts_with("<!doctype html>"));
        assert_eq!(html.matches("<svg").count(), 3, "one chart per group");
        assert!(html.contains("<path"), "a line chart");
        assert!(html.contains("<rect"), "a bar chart");
        assert!(html.contains("effect floor"), "a forest");
        // Labels are escaped, and nothing is fetched from anywhere.
        assert!(html.contains("a&lt;b vs c"));
        assert!(html.contains("tslab &lt;report&gt;"));
        assert!(
            !html.contains("http://") && !html.contains("https://") && !html.contains("<script")
        );
        // The dirty verdict's dots are hollow.
        assert!(html.contains("fill=\"var(--surface-1)\""));
        assert!(html.contains("prefers-color-scheme: dark"));
    }

    #[test]
    fn log_ticks_are_powers_of_two_inside_the_domain() {
        let s = Scale::new(&[512.0, 32768.0], true, 0.0, 100.0, false);
        let t = s.ticks();
        assert!(t.contains(&512.0) && t.contains(&32768.0), "{t:?}");
        assert!(t.iter().all(|v| v.log2().fract() == 0.0));
        let lin = Scale::new(&[0.0, 9.0], false, 0.0, 100.0, true);
        assert!(lin.ticks().first() == Some(&0.0));
    }

    #[test]
    fn empty_input_still_renders() {
        let html = render("t", "now", &[]);
        assert!(html.contains("No plottable records"));
    }
}
