//! `tslab serve`: a read-only dashboard over the lab dir.
//!
//! Every experiment's ledger on one table, the charts `tslab report` draws
//! per experiment, a compare view that puts several experiments' series on
//! shared axes, the verdict board, the sweeps, and each family's profile
//! over what the lab has measured for it. Server-rendered pages in the
//! TurboSpark theme with nothing fetched from anywhere, one inline script
//! for hover, legend toggles and the compare tray, plus a JSON API.
//!
//! Reading only: no POST route exists, nothing under the lab dir is written,
//! and the machine lock is never taken, so the dashboard can stay open
//! beside a measurement. The index is re-read when a ledger's size or mtime
//! moves (`index.rs`).

pub mod api;
pub mod index;
pub mod pages;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxPath, Query, State};
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::get;
use axum::Router;

use index::LabIndex;
use pages::Params;

/// The server's state: the lab dir and its cached index.
pub struct App {
    pub lab: PathBuf,
    index: Mutex<LabIndex>,
}

pub type Shared = Arc<App>;

impl App {
    pub fn new(lab: &Path) -> Shared {
        Arc::new(Self {
            lab: lab.to_path_buf(),
            index: Mutex::new(LabIndex::build(lab)),
        })
    }

    /// Refresh the index if a ledger moved, then read it.
    pub fn with_index<T>(&self, f: impl FnOnce(&LabIndex) -> T) -> T {
        let mut ix = self.index.lock().unwrap_or_else(|e| e.into_inner());
        ix.refresh();
        f(&ix)
    }
}

type Page = (StatusCode, Html<String>);

fn ok(html: String) -> Page {
    (StatusCode::OK, Html(html))
}

fn not_found(what: &str) -> Page {
    (StatusCode::NOT_FOUND, Html(pages::not_found(what)))
}

async fn home(State(app): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Page {
    let p = Params::from_query(&q);
    ok(app.with_index(|ix| pages::experiments_page(ix, &p)))
}

async fn experiment(State(app): State<Shared>, AxPath(name): AxPath<String>) -> Page {
    if crate::paths::validate_name(&name).is_err() {
        return not_found("that is not an experiment name");
    }
    match app.with_index(|ix| pages::experiment_page(ix, &name)) {
        Some(html) => ok(html),
        None => not_found(&format!("no experiment {name} in the lab dir")),
    }
}

async fn compare_multi(
    State(app): State<Shared>,
    axum::extract::RawQuery(raw): axum::extract::RawQuery,
) -> Page {
    // `exp=a&exp=b` from the checkbox form: fold every `exp` into one list.
    let raw = raw.unwrap_or_default();
    let mut map: HashMap<String, String> = HashMap::new();
    let mut exps: Vec<String> = Vec::new();
    for pair in raw.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let v = urldecode(v);
        if k == "exp" {
            exps.extend(
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            );
        } else {
            map.insert(k.to_string(), v);
        }
    }
    if !exps.is_empty() {
        map.insert("exp".into(), exps.join(","));
    }
    let p = Params::from_query(&map);
    ok(app.with_index(|ix| pages::compare_page(ix, &p)))
}

fn urldecode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(b) => {
                    out.push(b);
                    i += 2;
                }
                Err(_) => out.push(b'%'),
            },
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

async fn verdicts(State(app): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Page {
    let p = Params::from_query(&q);
    ok(app.with_index(|ix| pages::verdicts_page(ix, &p)))
}

async fn sweeps(State(app): State<Shared>, Query(q): Query<HashMap<String, String>>) -> Page {
    let p = Params::from_query(&q);
    ok(app.with_index(|ix| pages::sweeps_page(ix, &p)))
}

async fn families(State(app): State<Shared>) -> Page {
    ok(app.with_index(pages::families_page))
}

async fn family(State(app): State<Shared>, AxPath(family): AxPath<String>) -> Page {
    if family.is_empty()
        || !family
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return not_found("that is not a family name");
    }
    match app.with_index(|ix| pages::family_page(ix, &family)) {
        Some(html) => ok(html),
        None => not_found(&format!("no family profile {family}")),
    }
}

async fn record(State(app): State<Shared>, AxPath((name, id)): AxPath<(String, String)>) -> Page {
    if crate::paths::validate_name(&name).is_err() || !crate::family_profile::is_record_id(&id) {
        return not_found("that is not an experiment and record id");
    }
    match app.with_index(|ix| pages::record_page(ix, &name, &id)) {
        Some(html) => ok(html),
        None => not_found(&format!("no record {id} in {name}")),
    }
}

async fn fallback() -> Page {
    not_found("no such page")
}

/// Every route, over a shared state.
pub fn router(app: Shared) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/experiment/:name", get(experiment))
        .route("/compare", get(compare_multi))
        .route("/verdicts", get(verdicts))
        .route("/sweeps", get(sweeps))
        .route("/families", get(families))
        .route("/family/:family", get(family))
        .route("/record/:name/:id", get(record))
        .route("/api/experiments", get(api::experiments))
        .route("/api/experiment/:name", get(api::experiment))
        .route("/api/ledger/:name", get(api::ledger))
        .route("/api/series", get(api::series))
        .route("/api/verdicts", get(api::verdicts))
        .route("/api/sweeps", get(api::sweeps))
        .route("/api/family/:family", get(api::family))
        .route("/api/record/:name/:id", get(api::record))
        .fallback(fallback)
        .with_state(app)
}

/// `tslab serve [--port 7878] [--bind 127.0.0.1] [--lab DIR] [--open]`.
/// Blocks until the process is stopped. Never takes the machine lock.
pub fn serve_cmd(mut args: crate::args::Args) -> Result<crate::schema::CmdOutput, String> {
    let port: u16 = args.parsed("--port", 7878)?;
    let bind = args.take("--bind")?.unwrap_or_else(|| "127.0.0.1".into());
    let lab = args
        .take("--lab")?
        .map(PathBuf::from)
        .unwrap_or_else(crate::paths::lab_dir);
    let open = args.flag("--open");
    args.finish()?;
    if !lab.is_dir() {
        return Err(format!("{}: not a directory", lab.display()));
    }
    let addr: SocketAddr = format!("{bind}:{port}")
        .parse()
        .map_err(|e| format!("--bind {bind} --port {port}: {e}"))?;
    let app = App::new(&lab);
    let n = app.with_index(|ix| ix.experiments.len());
    let url = format!("http://{addr}/");
    crate::say!(
        "tslab serve: {n} experiment(s) in {} at {url}",
        lab.display()
    );
    if !addr.ip().is_loopback() {
        crate::say!(
            "  bound to {}: the dashboard has no authentication",
            addr.ip()
        );
    }
    crate::say!("  read-only; the machine lock is not taken; Ctrl-C stops it");
    if open {
        let _ = std::process::Command::new("open").arg(&url).spawn();
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| format!("bind {addr}: {e}"))?;
        axum::serve(listener, router(app))
            .await
            .map_err(|e| e.to_string())
    })?;
    Ok(crate::schema::CmdOutput::ok(
        serde_json::json!({"url": url}),
    ))
}
