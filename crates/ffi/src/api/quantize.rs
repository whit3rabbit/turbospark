//! GGUF quantizer C ABI entry points: quantize a float GGUF under a recipe,
//! read a recipe from a published GGUF, list the producible types, and check
//! whether a run's recorded inputs have changed on Hugging Face.
//!
//! The work lives in `turbospark_repack::gguf_quantize`; this file is the
//! ABI shell around it (JSON in and out, the install-style progress callback,
//! one cancel switch, every entry inside the panic guard).

use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use repack::gguf_quantize as q;
use serde::Deserialize;

use crate::abi::{self, guard_result};
use crate::models::{TS_INSTALL_BYTES, TS_INSTALL_STAGE};
use crate::strings;
use crate::TsInstallCallback;

/// The run in flight, if any. One at a time: the tie-order switch is
/// process-wide, and two runs would also fight over the cores.
static ACTIVE: Mutex<Option<Arc<AtomicBool>>> = Mutex::new(None);

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Options {
    source: String,
    destination: String,
    /// A recipe object (see docs/QUANTIZE.md), or a bare type name such as
    /// "q4_k". Exactly one of `recipe` and `recipeFromGguf` is required.
    recipe: Option<serde_json::Value>,
    /// Follow the per-tensor types of this existing GGUF.
    recipe_from_gguf: Option<String>,
    imatrix: Option<String>,
    threads: Option<usize>,
    /// "stable" (default) or "apple".
    iq1_tie_order: Option<String>,
    /// Write `<destination>.provenance.json`. Defaults to true.
    provenance: Option<bool>,
    /// Hugging Face locations (`owner/repo:file`) of the inputs, recorded in
    /// the provenance so a later check can ask the Hub about them.
    hf: Option<HfLocations>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct HfLocations {
    source: Option<String>,
    reference: Option<String>,
    imatrix: Option<String>,
}

fn invalid(e: impl std::fmt::Display) -> (c_int, String) {
    (abi::TS_ERR_INVALID_ARGUMENT, e.to_string())
}

fn recipe_of(o: &Options) -> Result<q::Recipe, (c_int, String)> {
    match (&o.recipe, &o.recipe_from_gguf) {
        (Some(_), Some(_)) => Err(invalid("give recipe or recipeFromGguf, not both")),
        (None, None) => Err(invalid("recipe or recipeFromGguf is required")),
        (None, Some(p)) => q::recipe_from_gguf(Path::new(p)).map_err(invalid),
        (Some(serde_json::Value::String(t)), None) => {
            q::Recipe::from_json(&serde_json::json!({ "default": t }).to_string()).map_err(invalid)
        }
        (Some(v), None) => q::Recipe::from_json(&v.to_string()).map_err(invalid),
    }
}

fn report_json(r: &q::QuantizeReport, provenance: Option<&Path>) -> String {
    serde_json::json!({
        "tensors": r.tensors,
        "bytesIn": r.bytes_in,
        "bytesOut": r.bytes_out,
        "byType": r.by_type,
        "keptQuantized": r.kept_quantized,
        "imatrixUsed": r.imatrix_used,
        "imatrixMissing": r.imatrix_missing,
        "notInExecutableList": r.not_in_executable_list,
        "provenancePath": provenance.map(|p| p.display().to_string()),
    })
    .to_string()
}

/// Quantizes `source` to `destination`.
///
/// `options_json` is `{"source", "destination", "recipe" | "recipeFromGguf",
/// "imatrix"?, "threads"?, "iq1TieOrder"?, "provenance"?, "hf"?}`; see
/// `Options`. Progress arrives through `cb` as `TS_INSTALL_STAGE` (the tensor
/// being written) and `TS_INSTALL_BYTES` (output bytes done and total), on the
/// calling thread. The result is the report JSON. An existing destination is
/// refused. [`ts_quantize_cancel`] ends a run with `TS_ERR_CANCELLED` and
/// leaves no file.
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_gguf(
    options_json: *const c_char,
    cb: TsInstallCallback,
    userdata: *mut c_void,
    result_json: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if result_json.is_null() {
            return Err(invalid("resultJson must not be null"));
        }
        let text = strings::required(options_json, "optionsJson").map_err(invalid)?;
        let o: Options = serde_json::from_str(text)
            .map_err(|e| (abi::TS_ERR_JSON, format!("optionsJson: {e}")))?;
        let recipe = recipe_of(&o)?;
        let (src, dst) = (PathBuf::from(&o.source), PathBuf::from(&o.destination));
        if dst.exists() {
            return Err(invalid(format!(
                "{} exists; refusing to overwrite",
                dst.display()
            )));
        }
        let imatrix = match &o.imatrix {
            Some(p) => Some(q::Imatrix::load(Path::new(p)).map_err(invalid)?),
            None => None,
        };
        let locs = o.hf.as_ref();
        let loc = |s: Option<&String>| -> Result<Option<q::HfFile>, (c_int, String)> {
            s.map(|s| q::HfFile::parse(s)).transpose().map_err(invalid)
        };
        let (hf_src, hf_ref, hf_im) = (
            loc(locs.and_then(|l| l.source.as_ref()))?,
            loc(locs.and_then(|l| l.reference.as_ref()))?,
            loc(locs.and_then(|l| l.imatrix.as_ref()))?,
        );
        match o.iq1_tie_order.as_deref() {
            None | Some("stable") => q::set_tie_order(q::TieOrder::Stable),
            Some("apple") => q::set_tie_order(q::TieOrder::Apple),
            Some(other) => {
                return Err(invalid(format!(
                    "iq1TieOrder must be stable or apple, got {other:?}"
                )))
            }
        }

        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut slot = ACTIVE.lock().unwrap_or_else(|p| p.into_inner());
            if slot.is_some() {
                return Err((abi::TS_ERR_BUSY, "a quantization is already running".into()));
            }
            *slot = Some(Arc::clone(&cancel));
        }
        struct Release;
        impl Drop for Release {
            fn drop(&mut self) {
                *ACTIVE.lock().unwrap_or_else(|p| p.into_inner()) = None;
            }
        }
        let _release = Release;

        let emit = |kind: c_int, text: &str, done: u64, total: u64| {
            if let Some(f) = cb {
                f(
                    userdata,
                    kind,
                    text.as_ptr() as *const c_char,
                    text.len(),
                    done,
                    total,
                );
            }
        };
        let mut last_tensor = String::new();
        let progress = |p: &q::Progress| {
            // Called on this thread only (the writer's contract), so the
            // closure may keep state and the callback needs no locking.
            if p.tensor != last_tensor {
                last_tensor = p.tensor.clone();
                emit(TS_INSTALL_STAGE, &p.tensor, 0, 0);
            }
            emit(TS_INSTALL_BYTES, "", p.bytes_done, p.bytes_total);
        };
        let progress = std::cell::RefCell::new(progress);
        let progress_ref = |p: &q::Progress| (progress.borrow_mut())(p);
        let control = q::Control {
            progress: Some(&progress_ref),
            cancel: Some(&cancel),
        };
        let threads = o
            .threads
            .filter(|t| *t > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        let report = q::quantize_gguf_ctl(&src, &dst, &recipe, imatrix.as_ref(), threads, &control)
            .map_err(|e| match e {
                q::QuantizeError::Cancelled => (abi::TS_ERR_CANCELLED, e.to_string()),
                q::QuantizeError::Refused { .. } | q::QuantizeError::Header(_) => invalid(e),
                q::QuantizeError::Io(_) => (abi::TS_ERR_GENERATE, e.to_string()),
            })?;

        let mut prov_path = None;
        if o.provenance.unwrap_or(true) {
            let mut inputs: Vec<(&str, &Path, Option<q::HfFile>)> =
                vec![("source", src.as_path(), hf_src)];
            if let Some(r) = &o.recipe_from_gguf {
                inputs.push(("reference", Path::new(r), hf_ref));
            }
            if let Some(i) = &o.imatrix {
                inputs.push(("imatrix", Path::new(i), hf_im));
            }
            let p = q::Provenance::record(&inputs, &dst).map_err(|e| (abi::TS_ERR_GENERATE, e))?;
            let path = PathBuf::from(format!("{}.provenance.json", dst.display()));
            q::write_provenance(&path, &p).map_err(|e| (abi::TS_ERR_GENERATE, e))?;
            prov_path = Some(path);
        }
        strings::emit(&report_json(&report, prov_path.as_deref()), result_json).map_err(invalid)
    })
}

/// Asks the quantization in flight to stop at its next batch. Returns 1 when
/// one was running and has been signalled, 0 when none was (not an error).
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_cancel() -> c_int {
    abi::guard_value(0, || {
        let slot = ACTIVE.lock().unwrap_or_else(|p| p.into_inner());
        match slot.as_ref() {
            Some(flag) => {
                flag.store(true, Ordering::Relaxed);
                1
            }
            None => 0,
        }
    })
}

/// Reads the per-tensor types of the GGUF at `path` and writes the recipe JSON
/// that would reproduce them (the format `ts_quantize_gguf` takes as `recipe`).
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_recipe_from_gguf_json(
    path: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err(invalid("out must not be null"));
        }
        let path = strings::required(path, "path").map_err(invalid)?;
        let recipe = q::recipe_from_gguf(Path::new(path)).map_err(invalid)?;
        strings::emit(&recipe.to_json(), out).map_err(invalid)
    })
}

/// The types the quantizer can produce, as a JSON array of
/// `{"name", "ggmlType", "bitsPerWeight", "blockElements", "requiresImatrix",
/// "executable"}`. `requiresImatrix` is ggml's hard rule; llama.cpp's policy
/// for the other very low-bit types is in docs/QUANTIZE.md. `executable` says
/// whether turbospark's own install gate lists the type.
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_types_json(out: *mut *mut c_char) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err(invalid("out must not be null"));
        }
        let rows: Vec<serde_json::Value> = q::Target::ALL
            .iter()
            .map(|t| {
                let (elems, bytes) = t.block();
                serde_json::json!({
                    "name": t.name(),
                    "ggmlType": t.ggml_id(),
                    "bitsPerWeight": bytes as f64 * 8.0 / elems as f64,
                    "blockElements": elems,
                    "requiresImatrix": t.requires_imatrix(),
                    "executable": model_io::EXECUTABLE_GGUF_TYPES.contains(&t.name()),
                })
            })
            .collect();
        strings::emit(&serde_json::Value::Array(rows).to_string(), out).map_err(invalid)
    })
}

/// Asks Hugging Face whether the inputs recorded in a provenance file (written
/// next to a quantized GGUF) still have the recorded hashes. **Blocks on the
/// network** (up to about 30 seconds per repo): call it off the main thread.
/// Writes a JSON array of `{"role", "fileName", "status", "detail", "commit",
/// "needsAttention"}` where status is `upToDate`, `changed`, `sizeOnly`,
/// `missing`, `untracked` or `unreachable`. An unreachable Hub is reported per
/// input, never as "up to date".
#[no_mangle]
pub unsafe extern "C" fn ts_quantize_check_updates_json(
    provenance_path: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard_result(|| {
        if out.is_null() {
            return Err(invalid("out must not be null"));
        }
        let path = strings::required(provenance_path, "provenancePath").map_err(invalid)?;
        let p = q::read_provenance(Path::new(path)).map_err(invalid)?;
        let rows: Vec<serde_json::Value> = q::check_freshness(&p, &q::HfHub)
            .into_iter()
            .map(|f| {
                let (status, detail) = match &f.status {
                    q::Status::UpToDate => ("upToDate", None),
                    q::Status::Changed { recorded, current } => (
                        "changed",
                        Some(format!("recorded {recorded}, now {current}")),
                    ),
                    q::Status::SizeOnly { same } => (
                        "sizeOnly",
                        Some(if *same { "same size" } else { "size differs" }.to_string()),
                    ),
                    q::Status::Missing => ("missing", None),
                    q::Status::Untracked => ("untracked", None),
                    q::Status::Unreachable(e) => ("unreachable", Some(e.clone())),
                };
                serde_json::json!({
                    "role": f.role,
                    "fileName": f.file_name,
                    "status": status,
                    "detail": detail,
                    "commit": f.commit,
                    "needsAttention": f.needs_attention(),
                })
            })
            .collect();
        strings::emit(&serde_json::Value::Array(rows).to_string(), out).map_err(invalid)
    })
}
