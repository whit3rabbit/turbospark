//! The dashboard's pages: server-rendered HTML from the index, the viz
//! charts and the family profiles, in the TurboSpark theme (Spark Blue,
//! dark by default, light under a light system preference). Nothing is
//! fetched from anywhere. Filters and sorting are query parameters on plain
//! forms, and the one inline script (`DASH_JS`) only adds hover tooltips,
//! legend toggles, line draw-in and the compare tray's count, so a page
//! reads the same saved to disk as it does live.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde_json::Value;

use super::index::{fmt_ts, tz_label, LabIndex, ProfileRow, SweepRow, VerdictRow};
use crate::series::Series;
use crate::viz::{self, esc};

/// The dashboard's theme over the report CSS: the TurboSpark tokens
/// (docs/design-system), the top bar, tiles, pills, bars and motion.
/// Every status is a glyph and a word, never color alone, and every
/// animation stops under Reduce Motion.
pub const DASH_CSS: &str = r#"
:root { color-scheme: dark;
  --page: #090d16; --surface-1: #161a23; --bar: #171a23; --elevated: #1e222b;
  --text-primary: #f2f5fb; --text-secondary: rgba(242,245,251,0.75); --text-muted: rgba(242,245,251,0.62);
  --grid: rgba(242,245,251,0.08); --axis: rgba(242,245,251,0.36);
  --border: rgba(242,245,251,0.256); --border-strong: rgba(242,245,251,0.36); --hairline: rgba(83,86,92,0.74);
  --hover: rgba(242,245,251,0.07); --chip: rgba(242,245,251,0.04); --track: rgba(242,245,251,0.1);
  --accent: #3b9eff; --accent-ink: #3b9eff; --on-accent: #141417;
  --accent-tint: rgba(59,158,255,0.14); --accent-edge: rgba(59,158,255,0.3); --selection: rgba(59,158,255,0.25);
  --glow: 0 2px 16px rgba(59,158,255,0.45);
  --good: #3ddba8; --good-tint: rgba(61,219,168,0.14); --warn: #ffb340; --warn-tint: rgba(255,179,64,0.14);
  --critical: #ff6961; --critical-tint: rgba(255,105,97,0.14);
  --s1: #3b9eff;
  --sans: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Geist", system-ui, sans-serif;
  --mono: ui-monospace, "SF Mono", "Geist Mono", Menlo, monospace; }
@media (prefers-color-scheme: light) { :root { color-scheme: light;
  --page: #ffffff; --surface-1: #f9f9fa; --bar: #f7f7f7; --elevated: #f3f4f4;
  --text-primary: #141a24; --text-secondary: rgba(20,26,36,0.75); --text-muted: rgba(20,26,36,0.62);
  --grid: rgba(20,26,36,0.08); --axis: rgba(20,26,36,0.36);
  --border: rgba(20,26,36,0.224); --border-strong: rgba(20,26,36,0.48); --hairline: rgba(209,209,209,0.66);
  --hover: rgba(20,26,36,0.07); --chip: rgba(20,26,36,0.04); --track: rgba(20,26,36,0.1);
  --accent: #1d6fe0; --accent-ink: #1a64cc; --on-accent: #ffffff;
  --accent-tint: rgba(29,111,224,0.14); --accent-edge: rgba(29,111,224,0.3); --selection: rgba(29,111,224,0.12);
  --glow: 0 2px 16px rgba(29,111,224,0.45);
  --good: #00704f; --good-tint: rgba(0,112,79,0.14); --warn: #8a5a00; --warn-tint: rgba(138,90,0,0.14);
  --critical: #c00013; --critical-tint: rgba(192,0,19,0.14);
  --s1: #1d6fe0; } }
body { font-family: var(--sans); font-size: 14px; line-height: 20px; }
main { max-width: none; margin: 0; padding: 0; }
.wrap { max-width: 1360px; margin: 0 auto; padding: 24px 18px 72px; }
a { color: var(--accent-ink); text-decoration: none; }
a:hover { text-decoration: underline; }
a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
h1 { font-size: 21px; line-height: 28px; font-weight: 600; margin: 0 0 3px; }
h2 { font-size: 18px; line-height: 24px; font-weight: 600; margin: 32px 0 8px; border-top: 1px solid var(--border); padding-top: 18px; }
h3 { font-size: 14px; line-height: 20px; font-weight: 500; margin: 0 0 2px; }
.meta, .sub { color: var(--text-secondary); font-size: 12px; line-height: 16px; }
.m, code, pre, td.m, .kv, .legend, svg text { font-family: var(--mono); }
.m, td.m { font-size: 12px; }
header.top { position: sticky; top: 0; z-index: 5; min-height: 44px; background: var(--bar);
  border-bottom: 1px solid var(--hairline); display: flex; align-items: center; flex-wrap: wrap; gap: 0 18px; padding: 0 18px; }
header.top .brand { display: flex; align-items: center; gap: 8px; font-weight: 600; color: var(--text-primary); }
header.top .brand span { font-weight: 400; font-size: 12px; color: var(--text-secondary); }
header.top nav { display: flex; align-items: stretch; gap: 4px; min-height: 44px; }
header.top nav a { display: flex; align-items: center; padding: 0 10px; font-weight: 500; color: var(--text-secondary); border-bottom: 2px solid transparent; }
header.top nav a:hover { color: var(--text-primary); text-decoration: none; }
header.top nav a.on { color: var(--text-primary); border-bottom-color: var(--accent); }
header.top .status { margin-left: auto; display: flex; align-items: center; gap: 12px; font-family: var(--mono); font-size: 11px; line-height: 16px; color: var(--text-secondary); padding: 6px 0; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 8px; border-radius: 999px; background: var(--good-tint); color: var(--good); }
.live { width: 7px; height: 7px; border-radius: 999px; background: var(--good); display: inline-block; animation: breathe 0.85s ease-in-out infinite alternate; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(176px, 1fr)); gap: 12px; margin: 14px 0 18px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px; min-width: 0;
  display: flex; flex-direction: column; gap: 8px; transition: transform 0.13s ease-out; }
.tile:hover { transform: scale(1.012); }
.lbl { font-size: 11px; line-height: 14px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-secondary); }
.val { font-size: 27px; line-height: 32px; font-weight: 600; font-variant-numeric: tabular-nums; }
.stack { height: 8px; border-radius: 999px; background: var(--track); overflow: hidden; display: flex; gap: 1px; }
.stack i, .mini i, .d i b { display: block; height: 100%; transform-origin: left center; animation: grow 0.44s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
.mini { height: 6px; width: 100px; border-radius: 999px; background: var(--track); overflow: hidden; display: flex; margin: 3px 0; }
.good { color: var(--good); } .warn { color: var(--warn); } .bad { color: var(--critical); }
.d { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.d i { display: inline-block; width: 60px; height: 6px; border-radius: 999px; background: var(--track); overflow: hidden; vertical-align: middle; }
.d i b { background: var(--accent); } .d.neg i b { background: var(--critical); }
form.filter { display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: end; margin: 8px 0 12px; font-size: 12px; color: var(--text-secondary); }
form.filter label { display: flex; flex-direction: column; gap: 3px; }
form.filter input, form.filter select { font: inherit; font-size: 12px; min-height: 28px; padding: 3px 6px; border: 1px solid var(--border);
  border-radius: 7px; background: var(--surface-1); color: var(--text-primary); min-width: 110px; }
form.filter button, .btn, button.primary { font: inherit; font-size: 14px; font-weight: 500; min-height: 32px; padding: 5px 12px; border: 1px solid var(--border-strong);
  border-radius: 7px; background: transparent; color: var(--text-primary); cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
form.filter button:hover, .btn:hover { background: var(--hover); text-decoration: none; }
button.primary { background: var(--accent); color: var(--on-accent); border-color: transparent; box-shadow: var(--glow); }
button.primary:disabled { opacity: 0.45; box-shadow: none; cursor: default; }
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.panel .scroll { padding: 0 18px; }
.panel table { min-width: 1080px; }
.panel td a { white-space: nowrap; }
.nw, td.m:first-child { white-space: nowrap; }
.reasons { max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin: 12px 0; }
table { font-size: 13px; }
th { font-size: 11px; line-height: 14px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; color: var(--text-secondary); padding: 10px 10px 8px 0; white-space: nowrap; }
td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--grid); vertical-align: middle; }
tr:hover td { background: var(--hover); }
th a { color: inherit; text-decoration: none; }
th a:hover { text-decoration: underline; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.dirty td { color: var(--text-muted); }
.v { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; line-height: 16px; font-weight: 500; padding: 1px 7px;
  border-radius: 999px; border: 1px solid var(--border); background: var(--chip); color: var(--text-secondary); white-space: nowrap; }
.v svg { width: 10px; height: 10px; }
.v.win { color: var(--good); background: var(--good-tint); border-color: rgba(61,219,168,0.3); }
.v.loss { color: var(--critical); background: var(--critical-tint); border-color: rgba(255,105,97,0.3); }
.v.inconclusive { background: transparent; border-style: dashed; }
.v.reading { color: var(--text-muted); }
.badge { border-radius: 8px; }
pre { background: var(--page); border: 1px solid var(--border); border-radius: 7px; padding: 10px; overflow-x: auto; font-size: 12px; line-height: 18px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
.kv { font-size: 12px; line-height: 18px; color: var(--text-secondary); display: flex; flex-wrap: wrap; gap: 4px 18px; }
.kv b { color: var(--text-primary); font-weight: 600; }
ul.tight { margin: 4px 0; padding-left: 18px; }
ul.tight li { margin: 2px 0; }
.note { white-space: pre-wrap; font-size: 12px; }
.cb { width: 22px; }
input[type=checkbox] { width: 16px; height: 16px; margin: 0; accent-color: var(--accent); cursor: pointer; }
tr.sel td { background: var(--selection); }
.tray { position: sticky; bottom: 0; z-index: 4; background: var(--bar); border-top: 1px solid var(--hairline);
  margin: 12px -18px 0; padding: 10px 18px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.tray .names { flex: 1 1 320px; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--mono); font-size: 12px; color: var(--text-secondary); }
.legend { gap: 6px; }
.legend .lg { font: inherit; font-family: var(--mono); font-size: 12px; min-height: 26px; padding: 2px 8px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--chip); color: var(--text-secondary); cursor: pointer; display: inline-flex; align-items: center; }
.legend .lg:hover { background: var(--hover); }
.legend .lg[aria-pressed="false"] { opacity: 0.5; text-decoration: line-through; }
svg .ser { transition: opacity 0.13s ease-out; }
svg .ser.off { opacity: 0.08; }
svg .mark { transition: r 0.13s ease-out; }
svg .mark:hover { r: 6; }
svg rect.bar { transform-box: fill-box; transform-origin: left center; animation: grow 0.44s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
.tip { position: fixed; z-index: 10; pointer-events: none; padding: 4px 8px; border-radius: 7px; background: var(--elevated);
  border: 1px solid var(--border); font-family: var(--mono); font-size: 11px; line-height: 16px; color: var(--text-primary);
  white-space: nowrap; box-shadow: 0 2px 12px rgba(0,0,0,0.28); }
.keys { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
.keys a { display: inline-flex; align-items: center; gap: 6px; min-height: 28px; padding: 3px 10px; border-radius: 8px; border: 1px solid var(--accent-edge);
  background: var(--accent-tint); color: var(--text-primary); font-family: var(--mono); font-size: 12px; }
@keyframes grow { from { transform: scaleX(0); } }
@keyframes breathe { from { opacity: 0.45; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
@media (max-width: 720px) { .wrap { padding: 16px 16px 72px; } .tray { margin: 12px -16px 0; } header.top { padding: 0 16px; } }
"#;

/// The dashboard's one script. It loads nothing and the page works without
/// it: SVG `<title>` hovers become a floating tip, a legend key hides its
/// series, a line draws in once, and the compare tray counts the ticks.
pub const DASH_JS: &str = r#"(function(){
var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
var tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; document.body.appendChild(tip);
document.querySelectorAll('svg title').forEach(function(t){ var el = t.parentNode; el.setAttribute('data-tip', t.textContent); el.removeChild(t); });
function at(e){ var x = e.clientX + 12, y = e.clientY - 30, w = tip.offsetWidth; if (x + w > innerWidth - 8) { x = e.clientX - w - 12; } tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
document.addEventListener('mouseover', function(e){ var el = e.target.closest && e.target.closest('[data-tip]'); if (!el) { return; } tip.textContent = el.getAttribute('data-tip'); tip.hidden = false; at(e); });
document.addEventListener('mousemove', function(e){ if (!tip.hidden) { at(e); } });
document.addEventListener('mouseout', function(e){ if (e.target.closest && e.target.closest('[data-tip]')) { tip.hidden = true; } });
document.addEventListener('click', function(e){ var b = e.target.closest && e.target.closest('.lg'); if (!b) { return; }
  var card = b.closest('.card'); if (!card) { return; } var on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', on ? 'true' : 'false');
  card.querySelectorAll('.ser[data-si="' + b.getAttribute('data-si') + '"]').forEach(function(g){ g.classList.toggle('off', !on); }); });
if (!reduce) { document.querySelectorAll('svg path.ln').forEach(function(p){ var n = p.getTotalLength(); p.style.strokeDasharray = n; p.style.strokeDashoffset = n; p.getBoundingClientRect(); p.style.transition = 'stroke-dashoffset 0.9s cubic-bezier(0.2, 0.8, 0.2, 1)'; p.style.strokeDashoffset = '0'; }); }
var tray = document.querySelector('.tray'); if (tray) { var boxes = document.querySelectorAll('input[name="exp"]'); var n = tray.querySelector('.count'), names = tray.querySelector('.names'), btn = tray.querySelector('button');
  var upd = function(){ var on = []; boxes.forEach(function(b){ var tr = b.closest('tr'); if (tr) { tr.classList.toggle('sel', b.checked); } if (b.checked) { on.push(b.value); } });
    n.textContent = on.length + ' selected'; names.textContent = on.join(', ') || 'Tick experiments to put their series on shared axes.'; btn.disabled = on.length === 0; };
  boxes.forEach(function(b){ b.addEventListener('change', upd); }); upd(); }
})();"#;

/// The status glyph for a verdict word: checkmark, octagon, minus, question.
fn glyph(verdict: &str) -> &'static str {
    match verdict {
        "win" => "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"m5 13 4 4L19 7\"/></svg>",
        "loss" => "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M7.8 2h8.4L22 7.8v8.4L16.2 22H7.8L2 16.2V7.8z\"/><path d=\"m9 9 6 6m0-6-6 6\"/></svg>",
        "noise" => "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"3\" stroke-linecap=\"round\" aria-hidden=\"true\"><path d=\"M5 12h14\"/></svg>",
        "inconclusive" => "<svg viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M9 9a3 3 0 1 1 4 2.8c-.7.4-1 1-1 1.7V15\"/><path d=\"M12 19h.01\"/></svg>",
        _ => "",
    }
}

/// `+12.34%` with a bar whose length is the magnitude, capped at 100%.
fn delta(v: Option<f64>) -> String {
    let Some(x) = v else {
        return String::new();
    };
    let w = (x.abs() * 100.0).min(100.0);
    format!(
        "<span class=\"d{}\"><i><b style=\"width:{w:.0}%\"></b></i><span class=\"m\">{:+.2}%</span></span>",
        if x < 0.0 { " neg" } else { "" },
        x * 100.0
    )
}

/// Query parameters a list page understands.
#[derive(Debug, Default, Clone)]
pub struct Params {
    pub family: Option<String>,
    pub model: Option<String>,
    pub chip: Option<String>,
    pub kind: Option<String>,
    pub metric: Option<String>,
    pub q: Option<String>,
    pub since: Option<String>,
    pub sort: Option<String>,
    pub dir: Option<String>,
    pub exp: Option<String>,
}

impl Params {
    pub fn from_query(q: &std::collections::HashMap<String, String>) -> Self {
        let get = |k: &str| {
            q.get(k)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        Self {
            family: get("family"),
            model: get("model"),
            chip: get("chip"),
            kind: get("kind"),
            metric: get("metric"),
            q: get("q"),
            since: get("since"),
            sort: get("sort"),
            dir: get("dir"),
            exp: get("exp"),
        }
    }

    /// The experiments a `exp=a,b,c` parameter names, validated.
    pub fn experiments(&self) -> Vec<String> {
        self.exp
            .as_deref()
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty() && crate::paths::validate_name(s).is_ok())
            .map(str::to_string)
            .collect()
    }

    fn desc(&self) -> bool {
        self.dir.as_deref() != Some("asc")
    }
}

fn contains_ci(hay: &str, needle: &str) -> bool {
    hay.to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn opt(v: &Option<String>) -> &str {
    v.as_deref().unwrap_or("")
}

fn pct(v: Option<f64>) -> String {
    v.map_or("".into(), |x| format!("{:+.2}%", x * 100.0))
}

fn num(v: Option<f64>, digits: usize) -> String {
    v.map_or("".into(), |x| format!("{x:.digits$}"))
}

fn verdict_badge(v: &str) -> String {
    format!("<span class=\"v {}\">{}{}</span>", esc(v), glyph(v), esc(v))
}

/// The top bar: what this is, where you are, and that it is read-only.
fn nav(active: &str) -> String {
    let items = [
        ("/", "Experiments"),
        ("/verdicts", "Verdicts"),
        ("/sweeps", "Sweeps"),
        ("/families", "Families"),
        ("/api/experiments", "API"),
    ];
    let mut out = String::from(
        "<header class=\"top\"><a class=\"brand\" href=\"/\"><svg width=\"18\" height=\"18\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"var(--accent)\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" aria-hidden=\"true\"><path d=\"M12 2c0 4-4 6-4 10a4 4 0 0 0 8 0c0-4-4-6-4-10z\"/><path d=\"M12 22v-4\"/></svg>tslab <span>lab</span></a><nav aria-label=\"Dashboard\">",
    );
    for (href, label) in items {
        let _ = write!(
            out,
            "<a href=\"{href}\"{}>{label}</a>",
            if href == active {
                " class=\"on\" aria-current=\"page\""
            } else {
                ""
            }
        );
    }
    let _ = write!(
        out,
        "</nav><div class=\"status\"><span class=\"pill\"><span class=\"live\"></span>read-only, lock not taken</span><span>times {}</span></div></header>",
        esc(&tz_label())
    );
    out
}

pub fn shell(title: &str, active: &str, body: &str) -> String {
    viz::page(
        title,
        &format!("{}<div class=\"wrap\">{body}</div>", nav(active)),
        DASH_CSS,
        DASH_JS,
    )
}

pub fn not_found(what: &str) -> String {
    shell(
        "not found",
        "",
        &format!("<h1>Not found</h1><p>{}</p>", esc(what)),
    )
}

/// A sortable column header: clicking toggles the direction.
fn th(label: &str, col: &str, p: &Params, base: &str, numeric: bool) -> String {
    let dir = if p.sort.as_deref() == Some(col) && p.desc() {
        "asc"
    } else {
        "desc"
    };
    let mut qs = query_string(p, &[("sort", Some(col)), ("dir", Some(dir))]);
    if !qs.is_empty() {
        qs = format!("?{qs}");
    }
    format!(
        "<th{}><a href=\"{base}{qs}\">{}{}</a></th>",
        if numeric { " class=\"num\"" } else { "" },
        esc(label),
        if p.sort.as_deref() == Some(col) {
            if p.desc() {
                " &#9662;"
            } else {
                " &#9652;"
            }
        } else {
            ""
        }
    )
}

/// The page's query string with some keys replaced.
fn query_string(p: &Params, overrides: &[(&str, Option<&str>)]) -> String {
    let mut pairs: Vec<(&str, String)> = Vec::new();
    for (k, v) in [
        ("family", &p.family),
        ("model", &p.model),
        ("chip", &p.chip),
        ("kind", &p.kind),
        ("metric", &p.metric),
        ("q", &p.q),
        ("since", &p.since),
        ("sort", &p.sort),
        ("dir", &p.dir),
        ("exp", &p.exp),
    ] {
        let over = overrides.iter().find(|(ok, _)| *ok == k);
        let val = match over {
            Some((_, Some(v))) => Some(v.to_string()),
            Some((_, None)) => None,
            None => v.clone(),
        };
        if let Some(v) = val {
            pairs.push((k, v));
        }
    }
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={}", urlenc(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn urlenc(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b',' => {
                out.push(b as char)
            }
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

fn filter_form(p: &Params, action: &str, ix: &LabIndex, with_metric: bool) -> String {
    let mut families: BTreeSet<&str> = BTreeSet::new();
    let mut models: BTreeSet<&str> = BTreeSet::new();
    let mut chips: BTreeSet<&str> = BTreeSet::new();
    for e in ix.experiments.values() {
        families.extend(e.summary.families.iter().map(String::as_str));
        models.extend(e.summary.models.iter().map(String::as_str));
        chips.extend(e.summary.chips.iter().map(String::as_str));
    }
    let select = |name: &str, current: &Option<String>, opts: &BTreeSet<&str>| {
        let mut s = format!("<label>{name}<select name=\"{name}\"><option value=\"\">any</option>");
        for o in opts {
            let _ = write!(
                s,
                "<option value=\"{}\"{}>{}</option>",
                esc(o),
                if current.as_deref() == Some(o) {
                    " selected"
                } else {
                    ""
                },
                esc(o)
            );
        }
        s.push_str("</select></label>");
        s
    };
    let mut f = format!("<form class=\"filter\" method=\"get\" action=\"{action}\">");
    f.push_str(&select("family", &p.family, &families));
    f.push_str(&select("model", &p.model, &models));
    f.push_str(&select("chip", &p.chip, &chips));
    if with_metric {
        let metrics: BTreeSet<&str> = ix.verdicts.iter().map(|v| v.metric.as_str()).collect();
        f.push_str(&select("metric", &p.metric, &metrics));
    }
    let _ = write!(
        f,
        "<label>since (YYYY-MM-DD)<input name=\"since\" value=\"{}\" size=\"10\"></label><label>search<input name=\"q\" value=\"{}\"></label>",
        esc(opt(&p.since)),
        esc(opt(&p.q))
    );
    if let Some(s) = &p.sort {
        let _ = write!(
            f,
            "<input type=\"hidden\" name=\"sort\" value=\"{}\">",
            esc(s)
        );
    }
    if let Some(d) = &p.dir {
        let _ = write!(
            f,
            "<input type=\"hidden\" name=\"dir\" value=\"{}\">",
            esc(d)
        );
    }
    f.push_str("<button type=\"submit\">Filter</button>");
    let _ = write!(f, "<a class=\"btn\" href=\"{action}\">Clear</a>");
    f.push_str("</form>");
    f
}

fn since_ok(ts: u64, since: &Option<String>) -> bool {
    match since {
        Some(s) => fmt_ts(ts).as_str() >= s.as_str(),
        None => true,
    }
}

/// `/`: the experiment table with filters, sorting and a compare form.
pub fn experiments_page(ix: &LabIndex, p: &Params) -> String {
    let mut rows: Vec<&super::index::ExperimentSummary> = ix
        .summaries()
        .into_iter()
        .filter(|s| {
            p.family.as_ref().is_none_or(|f| s.families.contains(f))
                && p.model.as_ref().is_none_or(|m| s.models.contains(m))
                && p.chip.as_ref().is_none_or(|c| s.chips.contains(c))
                && p.kind.as_ref().is_none_or(|k| s.counts.contains_key(k))
                && s.last_ts.is_none_or(|t| since_ok(t, &p.since))
                && p.q.as_ref().is_none_or(|q| {
                    contains_ci(&s.name, q)
                        || s.models.iter().any(|m| contains_ci(m, q))
                        || s.families.iter().any(|m| contains_ci(m, q))
                })
        })
        .collect();
    let count =
        |s: &super::index::ExperimentSummary, k: &str| s.counts.get(k).copied().unwrap_or(0);
    match p.sort.as_deref() {
        Some("name") => rows.sort_by(|a, b| a.name.cmp(&b.name)),
        Some("runs") => rows.sort_by_key(|s| s.runs_clean + s.runs_dirty),
        Some("verdicts") => rows.sort_by_key(|s| count(s, "verdict") + count(s, "experiment")),
        Some("curves") => rows.sort_by_key(|s| count(s, "kernel_curve")),
        _ => rows.sort_by_key(|s| s.last_ts),
    }
    if p.desc() {
        rows.reverse();
    }
    let mut body = String::new();
    let _ = write!(
        body,
        "<h1>Lab</h1><div class=\"meta\">Every ledger in <span class=\"m\">{}</span>. Nothing here is written to.</div>",
        esc(&ix.lab.display().to_string()),
    );
    body.push_str(&tiles(ix));
    body.push_str(&filter_form(p, "/", ix, false));
    let _ = write!(
        body,
        "<form method=\"get\" action=\"/compare\"><div class=\"panel\"><div class=\"scroll\"><table><caption style=\"caption-side:top;text-align:left;padding:10px 0;font-weight:500\">{} of {} experiments</caption><tr><th class=\"cb\"></th>",
        rows.len(),
        ix.experiments.len()
    );
    body.push_str(&th("experiment", "name", p, "/", false));
    body.push_str("<th>model</th>");
    body.push_str(&th("last activity", "last", p, "/", false));
    body.push_str(&th("runs clean / dirty", "runs", p, "/", false));
    body.push_str(&th("verdicts", "verdicts", p, "/", true));
    body.push_str("<th class=\"num\">sweeps</th>");
    body.push_str(&th("curves", "curves", p, "/", true));
    body.push_str("<th>best verdict</th><th>notes</th></tr>");
    for s in &rows {
        let best = ix
            .verdicts
            .iter()
            .filter(|v| v.experiment == s.name && v.verdict == "win")
            .max_by(|a, b| {
                a.median
                    .partial_cmp(&b.median)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let notes = ix
            .experiments
            .get(&s.name)
            .and_then(|e| e.notes.as_ref())
            .map(|n| {
                let first = n.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
                let cut: String = first.chars().take(90).collect();
                if first.chars().count() > 90 {
                    format!("{cut}...")
                } else {
                    cut
                }
            })
            .unwrap_or_default();
        let total = (s.runs_clean + s.runs_dirty).max(1) as f64;
        let list =
            |set: &BTreeSet<String>| esc(&set.iter().cloned().collect::<Vec<_>>().join(", "));
        let _ = write!(
            body,
            "<tr{}><td class=\"cb\"><input type=\"checkbox\" name=\"exp\" value=\"{n}\" aria-label=\"Select {n} for compare\"></td>\
             <td><a href=\"/experiment/{n}\">{n}</a>{}<div class=\"sub nw\"><span class=\"badge\" style=\"margin:0 6px 0 0\">{}</span><span class=\"m\">{} &middot; {}</span></div></td>\
             <td class=\"m\" title=\"{}\">{}</td><td class=\"m\">{}</td>\
             <td><div class=\"mini\" role=\"img\" aria-label=\"{} clean runs, {} dirty\"><i style=\"width:{:.0}%;background:var(--good)\"></i><i style=\"width:{:.0}%;background:var(--warn);opacity:0.6\"></i></div><span class=\"m sub\"><span class=\"good\">{}</span> / {}</span></td>\
             <td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"nw\">{}</td><td class=\"sub\">{}</td></tr>",
            if s.runs_clean == 0 && s.runs_dirty > 0 { " class=\"dirty\"" } else { "" },
            if s.broken.is_some() { " <span class=\"badge\">broken ledger</span>" } else { "" },
            list(&s.families),
            list(&s.chips),
            list(&s.revs),
            list(&s.models),
            list(&s.models),
            s.last_ts.map(fmt_ts).unwrap_or_default(),
            s.runs_clean,
            s.runs_dirty,
            100.0 * s.runs_clean as f64 / total,
            100.0 * s.runs_dirty as f64 / total,
            s.runs_clean,
            s.runs_dirty,
            count(s, "verdict") + count(s, "experiment") + count(s, "confirm"),
            count(s, "sweep"),
            count(s, "kernel_curve"),
            best.map(|v| format!("{} <span class=\"m\">{} vs {}</span> {}", verdict_badge(&v.verdict), esc(&v.baseline), esc(&v.candidate), delta(v.median)))
                .unwrap_or_else(|| "<span class=\"sub\">no paired A/B yet</span>".into()),
            esc(&notes),
            n = esc(&s.name)
        );
    }
    body.push_str("</table></div></div><div class=\"tray\"><span class=\"count\">0 selected</span><span class=\"names\">Tick experiments to put their series on shared axes.</span><span class=\"sub\">Shared axes show a difference, the verdict board shows significance.</span><button class=\"primary\" type=\"submit\">Compare Selected</button></div></form>");
    shell("tslab lab", "/", &body)
}

/// The totals over every ledger: experiments, runs with their clean share,
/// verdicts by outcome, sweeps, kernel curves.
fn tiles(ix: &LabIndex) -> String {
    let totals = kind_totals(ix);
    let n = |k: &str| totals.get(k).copied().unwrap_or(0);
    let (clean, dirty) = ix.experiments.values().fold((0, 0), |(c, d), e| {
        (c + e.summary.runs_clean, d + e.summary.runs_dirty)
    });
    let runs = (clean + dirty).max(1) as f64;
    let mut by: BTreeMap<&str, usize> = BTreeMap::new();
    for v in &ix.verdicts {
        *by.entry(v.verdict.as_str()).or_default() += 1;
    }
    let v = |k: &str| by.get(k).copied().unwrap_or(0);
    let nv = ix.verdicts.len().max(1) as f64;
    let families: BTreeSet<&str> = ix
        .experiments
        .values()
        .flat_map(|e| e.summary.families.iter().map(String::as_str))
        .collect();
    let models: BTreeSet<&str> = ix
        .experiments
        .values()
        .flat_map(|e| e.summary.models.iter().map(String::as_str))
        .collect();
    let best = ix
        .sweeps
        .iter()
        .filter_map(|s| s.best_median.map(|m| (m, s)))
        .max_by(|a, b| a.0.total_cmp(&b.0));
    let mut t = String::from("<section class=\"tiles\" aria-label=\"Lab totals\">");
    let _ = write!(
        t,
        "<div class=\"tile\"><div class=\"lbl\">Experiments</div><div class=\"val\">{}</div><div class=\"sub\">{} families, {} installs</div></div>",
        ix.experiments.len(),
        families.len(),
        models.len()
    );
    let _ = write!(
        t,
        "<div class=\"tile\"><div class=\"lbl\">Runs</div><div class=\"val\">{}</div><div class=\"stack\" role=\"img\" aria-label=\"{clean} clean runs, {dirty} dirty\"><i style=\"width:{:.1}%;background:var(--good)\"></i></div><div class=\"sub\"><span class=\"good\">{clean} clean</span> &middot; {dirty} dirty</div></div>",
        clean + dirty,
        100.0 * clean as f64 / runs
    );
    let _ = write!(
        t,
        "<div class=\"tile\"><div class=\"lbl\">Verdicts</div><div class=\"val\">{}</div><div class=\"stack\" role=\"img\" aria-label=\"{} wins, {} losses, {} noise, {} inconclusive\"><i style=\"width:{:.1}%;background:var(--good)\"></i><i style=\"width:{:.1}%;background:var(--critical)\"></i><i style=\"width:{:.1}%;background:var(--text-muted)\"></i><i style=\"width:{:.1}%;background:var(--track)\"></i></div><div class=\"sub\"><span class=\"good\">{} win</span> &middot; <span class=\"bad\">{} loss</span> &middot; {} noise &middot; {} inconclusive</div></div>",
        ix.verdicts.len(),
        v("win"),
        v("loss"),
        v("noise"),
        v("inconclusive"),
        100.0 * v("win") as f64 / nv,
        100.0 * v("loss") as f64 / nv,
        100.0 * v("noise") as f64 / nv,
        100.0 * v("inconclusive") as f64 / nv,
        v("win"),
        v("loss"),
        v("noise"),
        v("inconclusive")
    );
    let _ = write!(
        t,
        "<div class=\"tile\"><div class=\"lbl\">Sweeps</div><div class=\"val\">{}</div><div class=\"sub\">{}</div></div>",
        ix.sweeps.len(),
        best.map(|(m, s)| format!(
            "best WIN {:+.2}%, <span class=\"m\">{}</span> on <a href=\"/experiment/{e}\">{e}</a>",
            m * 100.0,
            esc(s.best_arm.as_deref().unwrap_or("")),
            e = esc(&s.experiment)
        ))
        .unwrap_or_else(|| "no candidate has won yet".into())
    );
    let _ = write!(
        t,
        "<div class=\"tile\"><div class=\"lbl\">Kernel curves</div><div class=\"val\">{}</div><div class=\"sub\">{} round-cost rows, {} projections</div></div>",
        n("kernel_curve"),
        n("round_cost"),
        n("projection")
    );
    t.push_str("</section>");
    t
}

fn provenance_card(s: &super::index::ExperimentSummary, e: &super::index::Experiment) -> String {
    let list = |set: &BTreeSet<String>| esc(&set.iter().cloned().collect::<Vec<_>>().join(", "));
    let mut c = String::from("<div class=\"card kv\">");
    let _ = write!(
        c,
        "<div><b>models</b> {}</div><div><b>family</b> {}</div><div><b>chip</b> {}</div><div><b>rev</b> {}</div><div><b>span</b> {} to {}</div><div><b>runs</b> {} clean, {} dirty</div><div><b>records</b> {}</div>",
        list(&s.models),
        list(&s.families),
        list(&s.chips),
        list(&s.revs),
        s.first_ts.map(fmt_ts).unwrap_or_default(),
        s.last_ts.map(fmt_ts).unwrap_or_default(),
        s.runs_clean,
        s.runs_dirty,
        s.counts
            .iter()
            .map(|(k, n)| format!("{} {}", n, esc(k)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if s.skipped_lines > 0 {
        let _ = write!(c, "<div><b>skipped lines</b> {}</div>", s.skipped_lines);
    }
    if let Some(b) = &s.broken {
        let _ = write!(c, "<div><b>broken</b> {}</div>", esc(b));
    }
    let _ = write!(
        c,
        "<div><a href=\"/api/ledger/{n}\">raw ledger</a> &middot; <a href=\"/api/experiment/{n}\">JSON</a>{}</div>",
        if s.has_report { " &middot; report.html on disk" } else { "" },
        n = esc(&s.name)
    );
    c.push_str("</div>");
    if let Some(n) = &e.notes {
        let _ = write!(c, "<h2>Notes</h2><div class=\"card note\">{}</div>", esc(n));
    }
    c
}

/// The verdict board: one row per paired A/B.
pub fn verdict_table(
    rows: &[&VerdictRow],
    p: &Params,
    base: &str,
    with_experiment: bool,
) -> String {
    let mut t = String::from("<div class=\"scroll\"><table><tr>");
    t.push_str(&th("when", "ts", p, base, false));
    if with_experiment {
        t.push_str("<th>experiment</th>");
    }
    t.push_str("<th>baseline vs candidate</th><th>metric</th><th>verdict</th>");
    t.push_str(&th("median", "median", p, base, true));
    t.push_str("<th>95% CI</th><th class=\"num\">pairs</th><th class=\"num\">floor</th><th>model</th><th>family</th><th>chip</th><th>rev</th><th>memory</th><th></th></tr>");
    for v in rows {
        let _ = write!(
            t,
            "<tr{}><td class=\"m\">{}</td>{}<td class=\"m\">{} vs {}{}</td><td class=\"m\">{}</td><td>{}{}</td><td class=\"num\">{}</td><td class=\"m nw\">{} to {}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"m\">{}</td><td>{}</td><td>{}</td><td class=\"m\">{}</td><td>{}</td><td><a href=\"/record/{}/{}\">record</a></td></tr>",
            if !v.eligible { " class=\"dirty\"" } else { "" },
            fmt_ts(v.ts_ms),
            if with_experiment {
                format!("<td class=\"m\"><a href=\"/experiment/{e}\">{e}</a></td>", e = esc(&v.experiment))
            } else {
                String::new()
            },
            esc(&v.baseline),
            esc(&v.candidate),
            v.sweep_id.as_ref().map(|s| format!(" <span class=\"badge\">sweep {}</span>", esc(s))).unwrap_or_default(),
            esc(&v.metric),
            verdict_badge(&v.verdict),
            v.decision.as_ref().map(|d| format!(" {}", esc(d))).unwrap_or_default(),
            delta(v.median),
            pct(v.ci_low),
            pct(v.ci_high),
            v.pairs,
            pct(v.null_floor),
            esc(opt(&v.model)),
            opt(&v.family),
            esc(opt(&v.chip)),
            esc(opt(&v.rev)),
            match v.memory_passed {
                Some(true) => "<span class=\"good\">ok</span>",
                Some(false) => "<span class=\"warn\">over allowance</span>",
                None => "",
            },
            esc(&v.experiment),
            esc(&v.record_id)
        );
    }
    t.push_str("</table></div>");
    t
}

fn sweep_table(rows: &[&SweepRow], with_experiment: bool) -> String {
    let mut t = String::from("<div class=\"scroll\"><table><tr><th>when</th>");
    if with_experiment {
        t.push_str("<th>experiment</th>");
    }
    t.push_str("<th>sweep</th><th>baseline</th><th>metric</th><th>candidates</th><th>best</th><th>model</th><th>family</th><th></th></tr>");
    for s in rows {
        let cands = s
            .candidates
            .iter()
            .map(|c| {
                format!(
                    "<div style=\"display:flex;align-items:center;gap:8px;margin:2px 0\"><span class=\"m\" style=\"min-width:48px\">{}</span>{} {}{}</div>",
                    esc(&c.arm),
                    verdict_badge(&c.verdict),
                    delta(c.median),
                    c.refused
                        .as_ref()
                        .map(|r| format!(" <span class=\"sub\">refused: {}</span>", esc(r)))
                        .unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("");
        let _ = write!(
            t,
            "<tr><td class=\"m\">{}</td>{}<td class=\"m\">{}</td><td class=\"m\">{}</td><td class=\"m\">{}</td><td>{}</td><td>{}</td><td class=\"m\">{}</td><td>{}</td><td><a href=\"/record/{}/{}\">record</a></td></tr>",
            fmt_ts(s.ts_ms),
            if with_experiment {
                format!("<td class=\"m\"><a href=\"/experiment/{e}\">{e}</a></td>", e = esc(&s.experiment))
            } else {
                String::new()
            },
            esc(&s.sweep_id),
            esc(&s.baseline),
            esc(&s.metric),
            cands,
            s.best_arm
                .as_ref()
                .map(|a| format!("<b class=\"m\">{}</b> {}", esc(a), pct(s.best_median)))
                .unwrap_or_else(|| "none".into()),
            esc(opt(&s.model)),
            esc(opt(&s.family)),
            esc(&s.experiment),
            esc(&s.record_id)
        );
    }
    t.push_str("</table></div>");
    t
}

fn profile_table(rows: &[&ProfileRow], with_experiment: bool) -> String {
    let mut t = String::from("<div class=\"scroll\"><table><tr><th>when</th>");
    if with_experiment {
        t.push_str("<th>experiment</th>");
    }
    t.push_str("<th>arm</th><th class=\"num\">prefill tok/s</th><th class=\"num\">decode tok/s</th><th class=\"num\">peak GiB</th><th class=\"num\">clean runs</th><th>model</th><th>family</th></tr>");
    for r in rows {
        let _ = write!(
            t,
            "<tr><td>{}</td>{}<td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td><td>{}</td></tr>",
            fmt_ts(r.ts_ms),
            if with_experiment {
                format!("<td><a href=\"/experiment/{e}\">{e}</a></td>", e = esc(&r.experiment))
            } else {
                String::new()
            },
            esc(&r.arm),
            num(r.prefill_tok_s, 1),
            num(r.decode_tok_s, 2),
            num(r.peak_gib, 2),
            r.clean_runs.map(|n| n.to_string()).unwrap_or_default(),
            esc(opt(&r.model)),
            esc(opt(&r.family))
        );
    }
    t.push_str("</table></div>");
    t
}

fn run_table(records: &[Value]) -> String {
    let mut runs: Vec<&Value> = records
        .iter()
        .filter(|r| r["kind"].as_str() == Some("run"))
        .collect();
    let total = runs.len();
    runs.truncate(500);
    let mut t = String::from("<div class=\"scroll\"><table><tr><th>when</th><th>arm</th><th class=\"num\">round</th><th>status</th><th>eligible</th><th class=\"num\">prefill tok/s</th><th class=\"num\">decode tok/s</th><th class=\"num\">peak MiB</th><th>reasons</th><th></th></tr>");
    for r in &runs {
        let reasons = r["reasons"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        let eligible = r["eligible"].as_bool().unwrap_or(false);
        let _ = write!(
            t,
            "<tr{}><td class=\"m nw\">{}</td><td>{}{}</td><td class=\"num\">{}</td><td>{}</td><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"sub reasons\" title=\"{}\">{}</td><td><a href=\"/record/{}/{}\">record</a></td></tr>",
            if eligible { "" } else { " class=\"dirty\"" },
            r["ts_ms"].as_u64().map(fmt_ts).unwrap_or_default(),
            esc(r["arm"].as_str().unwrap_or("?")),
            if r["warmup"].as_bool() == Some(true) { " <span class=\"badge\">warmup</span>" } else { "" },
            r["round"].as_u64().unwrap_or(0),
            esc(r["status"].as_str().unwrap_or("")),
            if eligible { "yes" } else { "no" },
            num(r["totals"]["prefill_tok_s"].as_f64(), 1),
            num(r["totals"]["decode_tok_s"].as_f64(), 2),
            num(r["peak_footprint_bytes"].as_f64().map(|b| b / 1048576.0), 0),
            esc(&reasons),
            esc(&reasons),
            esc(r["experiment"].as_str().unwrap_or("")),
            esc(r["id"].as_str().unwrap_or(""))
        );
    }
    t.push_str("</table></div>");
    if total > 500 {
        let _ = write!(
            t,
            "<p class=\"sub\">{} of {total} runs shown; the raw ledger has the rest.</p>",
            500
        );
    }
    t
}

/// `/experiment/{name}`.
pub fn experiment_page(ix: &LabIndex, name: &str) -> Option<String> {
    let e = ix.experiments.get(name)?;
    let s = &e.summary;
    let mut body = format!("<h1>{}</h1>", esc(name));
    body.push_str(&provenance_card(s, e));
    let verdicts: Vec<&VerdictRow> = ix
        .verdicts
        .iter()
        .filter(|v| v.experiment == name)
        .collect();
    if !verdicts.is_empty() {
        body.push_str("<h2>Verdicts</h2>");
        body.push_str(&verdict_table(
            &verdicts,
            &Params::default(),
            &format!("/experiment/{name}"),
            false,
        ));
    }
    let sweeps: Vec<&SweepRow> = ix.sweeps.iter().filter(|v| v.experiment == name).collect();
    if !sweeps.is_empty() {
        body.push_str("<h2>Sweeps</h2>");
        body.push_str(&sweep_table(&sweeps, false));
    }
    let profiles: Vec<&ProfileRow> = ix
        .profiles
        .iter()
        .filter(|v| v.experiment == name)
        .collect();
    if !profiles.is_empty() {
        body.push_str("<h2>Profile medians</h2>");
        body.push_str(&profile_table(&profiles, false));
    }
    let series = ix.series_of(name);
    if !series.is_empty() {
        body.push_str("<h2>Charts</h2><div class=\"meta\">Hollow marks are runs the host guard rejected.</div>");
        let refs: Vec<&Series> = series.iter().collect();
        body.push_str(&viz::cards(&refs));
    }
    // Run records carry no experiment field of their own; the link needs it.
    let mut records = e.records.clone();
    for r in &mut records {
        if let Value::Object(m) = r {
            m.insert("experiment".into(), Value::String(name.to_string()));
        }
    }
    if s.counts.contains_key("run") {
        body.push_str("<h2>Runs</h2>");
        body.push_str(&run_table(&records));
    }
    Some(shell(&format!("{name} - tslab"), "/", &body))
}

/// `/compare?exp=a,b,c`: the selected experiments' series regrouped across
/// experiments, labels prefixed with the experiment, and their verdict board.
pub fn compare_page(ix: &LabIndex, p: &Params) -> String {
    let names: Vec<String> = p
        .experiments()
        .into_iter()
        .filter(|n| ix.experiments.contains_key(n))
        .collect();
    let mut body = String::from("<h1>Compare</h1>");
    if names.is_empty() {
        body.push_str("<p>Pick experiments on the <a href=\"/\">experiment table</a>, or name them: <code>/compare?exp=a,b</code>.</p>");
        return shell("compare - tslab", "/", &body);
    }
    let _ = write!(
        body,
        "<div class=\"meta\">A shared axis shows a difference. Only the verdict board shows significance: bars from two experiments were never paired with each other.</div><div class=\"keys\">{}</div>",
        names
            .iter()
            .map(|n| format!("<a href=\"/experiment/{n}\">{n}</a>", n = esc(n)))
            .collect::<Vec<_>>()
            .join("")
    );
    let mut all: Vec<Series> = Vec::new();
    for n in &names {
        for mut s in ix.series_of(n) {
            if p.kind
                .as_ref()
                .is_some_and(|k| *k != s.kind && *k != s.source_kind)
            {
                continue;
            }
            s.label = format!("{n}: {}", s.label);
            all.push(s);
        }
    }
    let verdicts: Vec<&VerdictRow> = ix
        .verdicts
        .iter()
        .filter(|v| names.contains(&v.experiment))
        .collect();
    if !verdicts.is_empty() {
        body.push_str("<h2>Verdict board</h2>");
        body.push_str(&verdict_table(&verdicts, p, "/compare", true));
    }
    let sweeps: Vec<&SweepRow> = ix
        .sweeps
        .iter()
        .filter(|v| names.contains(&v.experiment))
        .collect();
    if !sweeps.is_empty() {
        body.push_str("<h2>Sweeps</h2>");
        body.push_str(&sweep_table(&sweeps, true));
    }
    let profiles: Vec<&ProfileRow> = ix
        .profiles
        .iter()
        .filter(|v| names.contains(&v.experiment))
        .collect();
    if !profiles.is_empty() {
        body.push_str("<h2>Profile medians</h2>");
        body.push_str(&profile_table(&profiles, true));
    }
    body.push_str("<h2>Charts</h2><div class=\"meta\">Series with the same kind and group share one chart; labels carry the experiment. Hollow marks are runs the host guard rejected.</div>");
    if all.is_empty() {
        body.push_str("<p>No plottable records.</p>");
    } else {
        let kinds: BTreeSet<&str> = all.iter().map(|s| s.kind.as_str()).collect();
        let _ = write!(
            body,
            "<div class=\"sub\">kinds: {}</div>",
            kinds
                .iter()
                .map(|k| format!(
                    "<a href=\"/compare?{}\">{k}</a>",
                    query_string(p, &[("kind", Some(k))])
                ))
                .collect::<Vec<_>>()
                .join(" &middot; ")
        );
        let refs: Vec<&Series> = all.iter().collect();
        body.push_str(&viz::cards(&refs));
    }
    shell("compare - tslab", "/", &body)
}

/// `/verdicts`: every paired A/B in the lab, filtered and sorted.
pub fn verdicts_page(ix: &LabIndex, p: &Params) -> String {
    let mut rows: Vec<&VerdictRow> = ix
        .verdicts
        .iter()
        .filter(|v| {
            p.family
                .as_ref()
                .is_none_or(|f| v.family.as_deref() == Some(f))
                && p.model
                    .as_ref()
                    .is_none_or(|m| v.model.as_deref() == Some(m))
                && p.chip.as_ref().is_none_or(|c| v.chip.as_deref() == Some(c))
                && p.metric.as_ref().is_none_or(|m| v.metric == *m)
                && since_ok(v.ts_ms, &p.since)
                && p.q.as_ref().is_none_or(|q| {
                    contains_ci(&v.experiment, q)
                        || contains_ci(&v.baseline, q)
                        || contains_ci(&v.candidate, q)
                        || contains_ci(&v.verdict, q)
                })
        })
        .collect();
    match p.sort.as_deref() {
        Some("median") => rows.sort_by(|a, b| {
            a.median
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&b.median.unwrap_or(f64::NEG_INFINITY))
        }),
        _ => rows.sort_by_key(|v| v.ts_ms),
    }
    if p.desc() {
        rows.reverse();
    }
    let mut body = format!(
        "<h1>Verdicts</h1><div class=\"meta\">{} of {} paired A/Bs. A WIN or LOSS needs 3 clean pairs over the effect floor; NOISE is a result, INCONCLUSIVE is not.</div>",
        rows.len(),
        ix.verdicts.len()
    );
    body.push_str(&filter_form(p, "/verdicts", ix, true));
    body.push_str(&verdict_table(&rows, p, "/verdicts", true));
    shell("verdicts - tslab", "/verdicts", &body)
}

/// `/sweeps`.
pub fn sweeps_page(ix: &LabIndex, p: &Params) -> String {
    let rows: Vec<&SweepRow> = ix
        .sweeps
        .iter()
        .filter(|s| {
            p.family
                .as_ref()
                .is_none_or(|f| s.family.as_deref() == Some(f))
                && p.model
                    .as_ref()
                    .is_none_or(|m| s.model.as_deref() == Some(m))
                && since_ok(s.ts_ms, &p.since)
                && p.q
                    .as_ref()
                    .is_none_or(|q| contains_ci(&s.experiment, q) || contains_ci(&s.sweep_id, q))
        })
        .collect();
    let mut body = format!(
        "<h1>Sweeps</h1><div class=\"meta\">{} of {} sweeps. Best is the largest median among WINs; none when no candidate won.</div>",
        rows.len(),
        ix.sweeps.len()
    );
    body.push_str(&filter_form(p, "/sweeps", ix, false));
    body.push_str(&sweep_table(&rows, true));
    shell("sweeps - tslab", "/sweeps", &body)
}

/// `/families`: one row per profile, with the lab's live counts.
pub fn families_page(ix: &LabIndex) -> String {
    let mut body = String::from("<h1>Families</h1><div class=\"meta\">The profiles in crates/lab/families with what the lab on this host has measured for each.</div><div class=\"scroll\"><table><tr><th>family</th><th>kind</th><th>title</th><th class=\"num\">recommended</th><th class=\"num\">sweeps</th><th class=\"num\">findings</th><th class=\"num\">live verdicts</th><th>last live verdict</th></tr>");
    match crate::family_profile::ProfileSet::embedded() {
        Ok(set) => {
            for pr in set.profiles() {
                let live: Vec<&VerdictRow> = ix
                    .verdicts
                    .iter()
                    .filter(|v| v.family.as_deref() == Some(pr.family.as_str()))
                    .collect();
                let last = live.first();
                let _ = write!(
                    body,
                    "<tr><td><a href=\"/family/{f}\">{f}</a></td><td>{}</td><td>{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td>{}</td></tr>",
                    esc(&pr.kind),
                    esc(&pr.title),
                    pr.recommended.len(),
                    pr.sweeps.len(),
                    pr.findings.len(),
                    live.len(),
                    last.map(|v| format!("{} {} vs {} {} ({})", fmt_ts(v.ts_ms), esc(&v.baseline), esc(&v.candidate), verdict_badge(&v.verdict), esc(&v.experiment))).unwrap_or_default(),
                    f = esc(&pr.family)
                );
            }
        }
        Err(e) => {
            let _ = write!(
                body,
                "<tr><td colspan=\"8\">profiles unreadable: {}</td></tr>",
                esc(&e)
            );
        }
    }
    body.push_str("</table></div>");
    shell("families - tslab", "/families", &body)
}

/// `/family/{family}`: the profile over everything the lab measured for it.
pub fn family_page(ix: &LabIndex, family: &str) -> Option<String> {
    let set = crate::family_profile::ProfileSet::embedded().ok()?;
    let pr = set.by_family(family)?;
    let mut body = format!(
        "<h1>{}</h1><div class=\"meta\">{} &middot; flow {} &middot; {} install(s) listed</div>",
        esc(&pr.family),
        esc(&pr.title),
        esc(&pr.flow),
        pr.installs.len()
    );
    body.push_str(
        "<div class=\"grid2\"><div class=\"card\"><h3>Recommended</h3><ul class=\"tight\">",
    );
    for r in &pr.recommended {
        let _ = write!(
            body,
            "<li><b>{}</b>{} <span class=\"sub\">{}</span>{}</li>",
            esc(&r.as_arm_fragment()),
            if r.when.is_empty() {
                String::new()
            } else {
                format!(
                    " <span class=\"badge\">when {}</span>",
                    esc(&r.when.join(" "))
                )
            },
            esc(&r.why),
            if r.findings.is_empty() {
                String::new()
            } else {
                format!(
                    " <span class=\"sub\">[{}]</span>",
                    r.findings
                        .iter()
                        .map(|f| format!("<a href=\"#{f}\">{f}</a>", f = esc(f)))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        );
    }
    body.push_str("</ul></div><div class=\"card\"><h3>Sweeps</h3><ul class=\"tight\">");
    for s in &pr.sweeps {
        let _ = write!(
            body,
            "<li><b>{}</b> {} vs {} <span class=\"sub\">[{:?}, {:?}] {}</span></li>",
            esc(&s.id),
            esc(&s.baseline),
            esc(&s.candidates.join(", ")),
            s.metric,
            s.exactness,
            esc(&s.description)
        );
    }
    body.push_str("</ul></div><div class=\"card\"><h3>Open questions</h3><ul class=\"tight\">");
    for q in &pr.open_questions {
        let _ = write!(body, "<li>{}</li>", esc(q));
    }
    body.push_str("</ul></div><div class=\"card\"><h3>Traps</h3><ul class=\"tight\">");
    for t in &pr.traps {
        let _ = write!(body, "<li>{}</li>", esc(t));
    }
    body.push_str("</ul></div></div>");

    let live: Vec<&VerdictRow> = ix
        .verdicts
        .iter()
        .filter(|v| v.family.as_deref() == Some(family))
        .collect();
    let _ = write!(body, "<h2>Live verdicts on this host ({})</h2>", live.len());
    if live.is_empty() {
        body.push_str("<p class=\"sub\">none in the lab dir</p>");
    } else {
        body.push_str(&verdict_table(
            &live,
            &Params::default(),
            &format!("/family/{family}"),
            true,
        ));
    }
    let sweeps: Vec<&SweepRow> = ix
        .sweeps
        .iter()
        .filter(|v| v.family.as_deref() == Some(family))
        .collect();
    if !sweeps.is_empty() {
        body.push_str("<h2>Live sweeps</h2>");
        body.push_str(&sweep_table(&sweeps, true));
    }
    let profiles: Vec<&ProfileRow> = ix
        .profiles
        .iter()
        .filter(|v| v.family.as_deref() == Some(family))
        .collect();
    if !profiles.is_empty() {
        body.push_str("<h2>Profile medians</h2>");
        body.push_str(&profile_table(&profiles, true));
    }

    let _ = write!(body, "<h2>Recorded findings ({})</h2><div class=\"meta\">A doc finding is a reading, not a tslab verdict. A ledger finding links to its record when that experiment is in this lab dir.</div><div class=\"scroll\"><table><tr><th>id</th><th>verdict</th><th>metric</th><th>install</th><th>values</th><th>provenance</th><th>conditions</th></tr>", pr.findings.len());
    for f in &pr.findings {
        let prov = match f.provenance.kind.as_str() {
            "ledger" => {
                let exp = f.provenance.experiment.as_deref().unwrap_or("");
                let rec = f.provenance.record.as_deref().unwrap_or("");
                if ix.record(exp, rec).is_some() {
                    format!(
                        "<a href=\"/record/{}/{}\">ledger {}:{}</a>",
                        esc(exp),
                        esc(rec),
                        esc(exp),
                        esc(rec)
                    )
                } else {
                    format!(
                        "ledger {}:{} <span class=\"badge\">not on this host</span>",
                        esc(exp),
                        esc(rec)
                    )
                }
            }
            "doc" => format!(
                "{} # {}",
                esc(f.provenance.path.as_deref().unwrap_or("")),
                esc(f.provenance.heading.as_deref().unwrap_or(""))
            ),
            _ => esc(f.provenance.source.as_deref().unwrap_or("")),
        };
        let _ = write!(
            body,
            "<tr id=\"{id}\"><td>{id}<br><span class=\"sub\">{}</span></td><td>{}</td><td>{}</td><td>{}</td><td><details><summary>values</summary><pre>{}</pre></details></td><td>{}</td><td class=\"sub\">{}{}</td></tr>",
            esc(&f.date),
            verdict_badge(&f.verdict),
            esc(&f.metric),
            esc(&f.install),
            esc(&serde_json::to_string_pretty(&f.values).unwrap_or_default()),
            prov,
            esc(&f.conditions),
            if f.note.is_empty() { String::new() } else { format!("<br><i>{}</i>", esc(&f.note)) },
            id = esc(&f.id)
        );
    }
    body.push_str("</table></div>");
    Some(shell(&format!("{family} - tslab"), "/families", &body))
}

/// `/record/{exp}/{id}`: one ledger record, with its run's stderr tail.
pub fn record_page(ix: &LabIndex, experiment: &str, id: &str) -> Option<String> {
    let rec = ix.record(experiment, id)?;
    let mut body = format!(
        "<h1>{} <span class=\"sub\">{}</span></h1><div class=\"meta\"><a href=\"/experiment/{e}\">{e}</a> &middot; {} &middot; <a href=\"/api/record/{e}/{}\">JSON</a></div>",
        esc(id),
        esc(rec["kind"].as_str().unwrap_or("?")),
        rec["ts_ms"].as_u64().map(fmt_ts).unwrap_or_default(),
        esc(id),
        e = esc(experiment)
    );
    let _ = write!(
        body,
        "<pre>{}</pre>",
        esc(&serde_json::to_string_pretty(rec).unwrap_or_default())
    );
    if rec["kind"].as_str() == Some("run") {
        if let Some(run_id) = rec["run_id"].as_str() {
            let stderr = ix
                .lab
                .join(experiment)
                .join("runs")
                .join(format!("{run_id}.stderr"));
            if let Ok(text) = std::fs::read_to_string(&stderr) {
                let tail: Vec<&str> = text
                    .lines()
                    .rev()
                    .take(60)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                let _ = write!(
                    body,
                    "<h2>stderr (last {} lines)</h2><pre>{}</pre>",
                    tail.len(),
                    esc(&tail.join("\n"))
                );
            }
        }
    }
    Some(shell(&format!("{id} - tslab"), "/", &body))
}

/// Counts by kind over every experiment, for the API's index payload.
pub fn kind_totals(ix: &LabIndex) -> BTreeMap<String, usize> {
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    for e in ix.experiments.values() {
        for (k, n) in &e.summary.counts {
            *out.entry(k.clone()).or_default() += n;
        }
    }
    out
}
