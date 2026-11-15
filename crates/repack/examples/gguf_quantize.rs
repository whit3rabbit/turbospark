//! Rewrite a float GGUF under a JSON recipe. See `docs/QUANTIZE.md`.
//!
//! ```sh
//! cargo run -p turbospark-repack --release --example gguf_quantize -- \
//!     --recipe recipe.json [--imatrix imatrix.gguf|imatrix.dat] in-f16.gguf out.gguf
//! ```
//!
//! `--recipe-from <reference.gguf>` takes the per-tensor types of an existing
//! GGUF (for example a published Unsloth "UD" file) instead of a recipe file.
//! `--dump-recipe <out.json>` writes the resolved recipe and exits.
//!
//! Provenance: each run writes `<out>.provenance.json` (SHA-256 of the source,
//! reference and imatrix, and of the output) unless `--no-provenance`. Name an
//! input's Hugging Face location with `--hf <role>=<owner/repo>:<file>` (role
//! is `source`, `reference` or `imatrix`; repeatable) so it can be checked later:
//!
//! ```sh
//! cargo run -p turbospark-repack --release --example gguf_quantize -- \
//!     --check-updates out.gguf.provenance.json
//! ```
//!
//! asks the Hub whether those files still have the recorded hashes. Exit codes:
//! 0 all current, 3 something changed, is missing or could not be checked.
//!
//! `--recipe` also accepts a bare type (`q8_0`, `q4_0`, ...) as shorthand for
//! `{"default": "<type>"}`.

use std::path::PathBuf;
use std::process::ExitCode;

use turbospark_repack::gguf_quantize::{
    check_freshness, quantize_gguf_opts, read_provenance, recipe_from_gguf, set_tie_order,
    write_provenance, HfFile, HfHub, Imatrix, Provenance, Recipe, Status, TieOrder,
};

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut recipe, mut imatrix, mut from, mut dump, mut threads, mut paths) =
        (None, None, None, None, None::<usize>, Vec::new());
    let (mut hf, mut provenance, mut check) = (Vec::<(String, HfFile)>::new(), true, None);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--recipe" => recipe = Some(args.next().ok_or("--recipe needs a value")?),
            "--recipe-from" => from = Some(args.next().ok_or("--recipe-from needs a value")?),
            "--dump-recipe" => dump = Some(args.next().ok_or("--dump-recipe needs a value")?),
            "--threads" => {
                let v = args.next().ok_or("--threads needs a value")?;
                threads = Some(v.parse().map_err(|_| format!("bad --threads {v:?}"))?);
            }
            "--iq1-tie-order" => {
                let v = args.next().ok_or("--iq1-tie-order needs stable|apple")?;
                set_tie_order(match v.as_str() {
                    "stable" => TieOrder::Stable,
                    "apple" => TieOrder::Apple,
                    _ => {
                        return Err(format!(
                            "--iq1-tie-order must be stable or apple, got {v:?}"
                        ))
                    }
                });
            }
            "--hf" => {
                let v = args.next().ok_or("--hf needs <role>=<owner/repo>:<file>")?;
                let (role, loc) = v
                    .split_once('=')
                    .ok_or(format!("--hf expects role=repo:file, got {v:?}"))?;
                if !["source", "reference", "imatrix"].contains(&role) {
                    return Err(format!(
                        "--hf role must be source, reference or imatrix, got {role:?}"
                    ));
                }
                hf.push((role.to_string(), HfFile::parse(loc)?));
            }
            "--no-provenance" => provenance = false,
            "--check-updates" => check = Some(args.next().ok_or("--check-updates needs a file")?),
            "--imatrix" => imatrix = Some(args.next().ok_or("--imatrix needs a value")?),
            _ => paths.push(PathBuf::from(a)),
        }
    }
    if let Some(path) = &check {
        return check_updates(std::path::Path::new(path));
    }
    let resolve = || -> Result<Recipe, String> {
        if let Some(r) = &from {
            return recipe_from_gguf(std::path::Path::new(r)).map_err(|e| e.to_string());
        }
        let spec = recipe
            .clone()
            .ok_or("--recipe or --recipe-from is required")?;
        let json = match std::fs::read_to_string(&spec) {
            Ok(text) => text,
            Err(_) => format!("{{\"default\":\"{spec}\"}}"),
        };
        Recipe::from_json(&json).map_err(|e| e.to_string())
    };
    if let Some(out) = &dump {
        let r = resolve()?;
        std::fs::write(out, r.to_json()).map_err(|e| e.to_string())?;
        println!("{} rules written to {out}", r.rules.len());
        return Ok(());
    }
    let [src, dst] = paths.as_slice() else {
        return Err("usage: gguf_quantize (--recipe <file.json|type> | --recipe-from <ref.gguf>) [--imatrix <file>] [--hf role=repo:file] [--no-provenance] [--threads N] [--iq1-tie-order stable|apple] <in.gguf> <out.gguf>".into());
    };
    let recipe = resolve()?;
    if dst.exists() {
        return Err(format!("{} exists; refusing to overwrite", dst.display()));
    }
    let imatrix_path = imatrix.clone();
    let im = match imatrix {
        Some(p) => Some(Imatrix::load(std::path::Path::new(&p)).map_err(|e| e.to_string())?),
        None => None,
    };
    let r = quantize_gguf_opts(
        src,
        dst,
        &recipe,
        im.as_ref(),
        threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{} tensors, {} -> {} bytes",
        r.tensors, r.bytes_in, r.bytes_out
    );
    for (ty, n) in &r.by_type {
        println!("  {ty:8} {n}");
    }
    if !r.not_in_executable_list.is_empty() {
        println!(
            "note: {} not in turbospark's executable GGUF type list (valid GGUF, but an install would be refused)",
            r.not_in_executable_list.join(", ")
        );
    }
    if im.is_some() {
        println!("imatrix applied to {} tensors", r.imatrix_used);
        if !r.imatrix_missing.is_empty() {
            println!(
                "{} imatrix-aware tensors had no entry and were quantized without weights (first: {})",
                r.imatrix_missing.len(),
                r.imatrix_missing[0]
            );
        }
    }
    if !r.kept_quantized.is_empty() {
        println!(
            "left {} already-quantized tensors unchanged (first: {})",
            r.kept_quantized.len(),
            r.kept_quantized[0]
        );
    }
    if provenance {
        let loc = |role: &str| hf.iter().find(|(r, _)| r == role).map(|(_, f)| f.clone());
        let mut inputs: Vec<(&str, &std::path::Path, Option<HfFile>)> =
            vec![("source", src.as_path(), loc("source"))];
        if let Some(r) = &from {
            inputs.push(("reference", std::path::Path::new(r), loc("reference")));
        }
        if let Some(i) = &imatrix_path {
            inputs.push(("imatrix", std::path::Path::new(i), loc("imatrix")));
        }
        let p = Provenance::record(&inputs, dst)?;
        let out = PathBuf::from(format!("{}.provenance.json", dst.display()));
        write_provenance(&out, &p)?;
        println!("provenance written to {}", out.display());
    }
    Ok(())
}

/// Ask the Hub whether the inputs recorded in a provenance file are current.
fn check_updates(path: &std::path::Path) -> Result<(), String> {
    let p = read_provenance(path)?;
    println!(
        "made by {} (ggml parity commit {})",
        p.tool, p.ggml_parity_commit
    );
    let mut attention = false;
    for f in check_freshness(&p, &HfHub) {
        let line = match &f.status {
            Status::UpToDate => "up to date".to_string(),
            Status::Changed { recorded, current } => format!(
                "CHANGED upstream (recorded {}..., now {}...)",
                &recorded[..12.min(recorded.len())],
                &current[..12.min(current.len())]
            ),
            Status::SizeOnly { same: true } => "same size (the Hub gave no hash to compare)".into(),
            Status::SizeOnly { same: false } => "size differs (the Hub gave no hash)".into(),
            Status::Missing => "no longer listed in the repo".into(),
            Status::Untracked => "no Hugging Face location recorded, not checked".into(),
            Status::Unreachable(e) => format!("could not be checked: {e}"),
        };
        println!("  {:10} {:40} {line}", f.role, f.file_name);
        attention |= f.needs_attention();
    }
    if attention {
        Err("one or more inputs changed or could not be checked".into())
    } else {
        println!("all tracked inputs are current");
        Ok(())
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            // 3: a freshness check found something to look at.
            ExitCode::from(if e.starts_with("one or more inputs") {
                3
            } else {
                1
            })
        }
    }
}
