//! The `extract` verb: capture two contrastive corpora against the model's
//! own activations and reduce them into one control vector.
//!
//! This is `docs/OBLITERATION.md`'s Reproducing recipe (capture per prompt,
//! difference of means, write the GGUF) driven from one command. What it
//! adds over the manual recipe is the trait pack: the contrastive corpora,
//! their system conditioning, and the advice line travel with a name, so a
//! user picks a trait instead of curating two directories of capture files.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use compute::steering_extract::{derived_alpha_cap, difference_of_means, layer_stats};
use foundation::SteeringMode;

use super::packs;
use super::walk;

pub struct ExtractOptions {
    pub model: String,
    pub trait_id: Option<String>,
    pub out: Option<PathBuf>,
    pub limit: Option<usize>,
    pub positive_file: Option<PathBuf>,
    pub negative_file: Option<PathBuf>,
    pub alpha_budget: f64,
}

/// One prompt per line; empty lines are skipped. The same corpus format the
/// trait packs embed, so a custom corpus needs no new plumbing.
fn read_prompt_file(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("--positive-file/--negative-file {}: {e}", path.display()))?;
    let prompts: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    if prompts.len() < 4 {
        return Err(format!(
            "{} holds {} prompts; at least 4 per side are required",
            path.display(),
            prompts.len()
        ));
    }
    Ok(prompts)
}

/// Truncate one side to `limit`, keeping the head. Paired designs (slop,
/// sycophancy) keep their pairing because both sides truncate the same way.
fn apply_limit(prompts: Vec<String>, limit: Option<usize>) -> Vec<String> {
    match limit {
        Some(n) if n > 0 => prompts.into_iter().take(n).collect(),
        _ => prompts,
    }
}

/// Prefix of the per-run capture directory under the system temp dir.
const CAPTURE_DIR_PREFIX: &str = "turbospark-steer-extract-";

/// A run's capture directory, removed on every exit path that unwinds.
/// SIGTERM (the app's cancel) skips Drop, which is what
/// [`sweep_stale_capture_dirs`] is for.
struct CaptureDir(PathBuf);

impl Drop for CaptureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Remove capture directories a killed run left behind. Age-based rather
/// than pid-based: no extraction runs for a day, and a liveness probe would
/// need `unsafe` outside the documented boundaries.
fn sweep_stale_capture_dirs(temp_root: &Path, max_age: std::time::Duration) {
    let Ok(entries) = std::fs::read_dir(temp_root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(CAPTURE_DIR_PREFIX) {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > max_age);
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Write `bytes` to `out` through a sibling `.partial` file and a rename,
/// so a cancelled or failed run never leaves a truncated vector at a path
/// an existing preset already points at.
fn write_atomically(out: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut partial = out.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    std::fs::write(&partial, bytes).map_err(|e| format!("writing {}: {e}", partial.display()))?;
    std::fs::rename(&partial, out).map_err(|e| {
        let _ = std::fs::remove_file(&partial);
        format!("moving {} into place: {e}", out.display())
    })
}

pub fn run(opts: ExtractOptions) -> Result<(), String> {
    let pack = match &opts.trait_id {
        Some(id) => Some(packs::by_id(id)?),
        None => None,
    };
    if pack.is_none() && (opts.positive_file.is_none() || opts.negative_file.is_none()) {
        return Err(
            "extract needs --trait ID, or both --positive-file and --negative-file \
             for a custom corpus"
                .to_string(),
        );
    }
    let model = walk::resolve_model(&opts.model)?;

    let (positive, negative) = match (&opts.positive_file, &opts.negative_file) {
        (Some(pos), Some(neg)) => (read_prompt_file(pos)?, read_prompt_file(neg)?),
        _ => {
            let pack = pack.as_ref().expect("pack when no files are given");
            (pack.positive.prompts.clone(), pack.negative.prompts.clone())
        }
    };
    let positive = apply_limit(positive, opts.limit);
    let negative = apply_limit(negative, opts.limit);

    let (system_pos, system_neg) = match pack.as_ref() {
        Some(pack) if opts.positive_file.is_none() && opts.negative_file.is_none() => (
            pack.positive.system.as_deref(),
            pack.negative.system.as_deref(),
        ),
        _ => (None, None),
    };

    println!(
        "extracting '{}' direction from {} ({} x {} prompts, window {})",
        opts.trait_id.as_deref().unwrap_or("custom"),
        model.dir.display(),
        positive.len(),
        negative.len(),
        model.max_context
    );

    // One temp dir for both sides' capture files, removed by the guard on
    // every return. The capture writes the header beside an .f32 sidecar by
    // replacing the extension, so the name must carry one.
    let temp_root = std::env::temp_dir();
    sweep_stale_capture_dirs(&temp_root, std::time::Duration::from_secs(24 * 60 * 60));
    let temp = temp_root.join(format!(
        "{CAPTURE_DIR_PREFIX}{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp).map_err(|e| format!("temp dir {}: {e}", temp.display()))?;
    let temp = CaptureDir(temp);
    let pos_capture = temp.0.join("positive.json");
    let neg_capture = temp.0.join("negative.json");

    let pos_set = walk::capture_side(&model, system_pos, &positive, &pos_capture, "positive")?;
    let neg_set = walk::capture_side(&model, system_neg, &negative, &neg_capture, "negative")?;
    // Sanity line: the capture keys on the last prompt token, so positions
    // vary with prompt length. A constant position across every prompt of
    // one side would mean the corpus itself is one repeated prompt, which
    // extracts a direction for that prompt rather than for the trait.
    for (label, path, count) in [
        ("positive", &pos_capture, positive.len()),
        ("negative", &neg_capture, negative.len()),
    ] {
        let positions = super::capture_io::read_positions(path);
        if positions.len() == count && !positions.is_empty() {
            let min = positions.iter().min().expect("non-empty");
            let max = positions.iter().max().expect("non-empty");
            println!("  [{label}] {count} snapshots at last-prompt positions {min}..{max}");
        }
    }

    let directions = difference_of_means(&pos_set, &neg_set)?;
    let stats = layer_stats(&pos_set, &neg_set, &directions)?;
    let all_zero = stats.iter().all(|s| s.norm <= 0.0);
    if all_zero {
        return Err(
            "every layer's direction is zero; the two prompt sets produced identical \
             residuals, which means the contrast did not separate anything on this \
             checkpoint"
                .to_string(),
        );
    }

    print_diagnostics(&stats, opts.alpha_budget);

    // The control-vector format cannot name block 0 (llama.cpp rejects
    // `direction.0`), so block 0 is announced and dropped, exactly as the
    // script does.
    let mut layers_map: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    for (l, d) in directions.iter().enumerate().skip(1) {
        layers_map.insert(l, d.clone());
    }
    let mode = pack
        .as_ref()
        .and_then(|p| SteeringMode::parse(&p.recommended_mode));
    let arch = model.arch.family.as_str().to_string();
    let bytes = repack::control_vector::write_control_vector(&layers_map, &arch, mode)
        .map_err(|e| e.to_string())?;
    let out = opts.out.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "{}-direction.gguf",
            opts.trait_id.as_deref().unwrap_or("custom")
        ))
    });
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    write_atomically(&out, &bytes)?;
    drop(temp);

    let blocks = layers_map.len();
    println!(
        "\nwrote {} ({blocks} directions of {}, blocks 1..{}; block 0 is dropped, \
         the control-vector format cannot name it)",
        out.display(),
        model.arch.hidden_size,
        model.arch.num_layers
    );
    if let Some(pack) = pack.as_ref() {
        println!(
            "\ntry it:\n  turbospark-steer check --model {} --trait {} --vector {}",
            opts.model,
            pack.id,
            out.display()
        );
        println!("\nadvice ({}): {}", pack.id, pack.advice);
    } else {
        println!(
            "\nno built-in advice for a custom corpus; start at ablate 0.3 over the \
             full band and see docs/OBLITERATION.md before going higher"
        );
    }
    Ok(())
}

/// The four columns, each dividing the direction norm by something
/// different because each answers a different question. Mirrors
/// `scripts/extract_direction.py`'s table minus the SVD method.
fn print_diagnostics(stats: &[compute::steering_extract::LayerStats], alpha_budget: f64) {
    let max_sep = stats.iter().map(|s| s.separation).fold(0.0f64, f64::max);
    println!("\nper-layer direction (mean):");
    println!(
        "  {:>5}  {:>10}  {:>7}  {:>7}  {:>8}   (sep = effect size; \
         share = ||d||/||x||; removed = |c_hat|/||x||)",
        "layer", "norm", "sep", "share", "removed"
    );
    for (l, s) in stats.iter().enumerate() {
        let bar = "#".repeat((40.0 * s.separation / max_sep.max(1e-9)) as usize);
        println!(
            "  {l:5}  {:10.4}  {:7.3}  {:6.1}  {:7.1}  {bar}",
            s.norm,
            s.separation,
            s.share * 100.0,
            s.removed * 100.0
        );
    }
    let best = stats
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.separation.total_cmp(&b.1.separation))
        .expect("non-empty stats");
    let hottest = stats
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.share.total_cmp(&b.1.share))
        .expect("non-empty stats");
    println!(
        "\nby effect size: strongest at layer {} ({:.3}); the raw norm would name \
         layer {}, which is mostly where the residual stream is biggest",
        best.0,
        best.1.separation,
        stats
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm.total_cmp(&b.1.norm))
            .expect("non-empty")
            .0
    );
    println!(
        "by stream share: layer {} carries {:.1}%, so an ablation at alpha 1 \
         removes that much of the residual there",
        hottest.0,
        hottest.1.share * 100.0
    );
    let cap = derived_alpha_cap(stats, alpha_budget);
    if cap.is_infinite() {
        println!("  no ceiling implied at this budget (--alpha-budget {alpha_budget})");
    } else {
        println!(
            "  derived ceiling (diagnostic only): alpha <= {cap:.2} keeps every \
             layer's removal under {:.0}% of the stream",
            alpha_budget * 100.0
        );
    }
    println!(
        "  DO NOT STEER AT THE DERIVED CEILING: measured against \
         crates/bench/tests/steering_sweep.rs it under-called the usable band by \
         1.1x and 8.9x with the relationship INVERTED. Both columns rank layers; \
         neither predicts a strength."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "turbospark-steer-unit-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    /// The rename replaces an existing vector whole and leaves no
    /// `.partial` behind.
    #[test]
    fn atomic_write_replaces_and_leaves_no_partial() {
        let dir = scratch("atomic");
        let out = dir.join("model-refusals.gguf");
        std::fs::write(&out, b"old vector bytes, longer than the new").expect("seed");
        write_atomically(&out, b"new").expect("write");
        assert_eq!(std::fs::read(&out).expect("read"), b"new");
        assert!(!dir.join("model-refusals.gguf.partial").exists());
        std::fs::remove_dir_all(&dir).expect("cleanup");
    }

    /// The guard removes its directory when dropped on an error path.
    #[test]
    fn capture_dir_guard_removes_on_drop() {
        let dir = scratch("guard");
        std::fs::write(dir.join("positive.f32"), b"x").expect("seed");
        drop(CaptureDir(dir.clone()));
        assert!(!dir.exists());
    }

    /// The sweep removes only prefixed directories older than the cutoff.
    #[test]
    fn sweep_removes_only_old_prefixed_dirs() {
        let root = scratch("sweep");
        let ours = root.join(format!("{CAPTURE_DIR_PREFIX}1-2"));
        let other = root.join("unrelated-dir");
        std::fs::create_dir_all(&ours).expect("mkdir");
        std::fs::create_dir_all(&other).expect("mkdir");
        // Fresh directories survive a one-day cutoff...
        sweep_stale_capture_dirs(&root, std::time::Duration::from_secs(86_400));
        assert!(ours.exists());
        // ...and a zero cutoff removes the prefixed one only.
        std::thread::sleep(std::time::Duration::from_millis(20));
        sweep_stale_capture_dirs(&root, std::time::Duration::ZERO);
        assert!(!ours.exists());
        assert!(other.exists());
        std::fs::remove_dir_all(&root).expect("cleanup");
    }
}
