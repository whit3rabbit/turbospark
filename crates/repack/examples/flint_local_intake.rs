//! Builds a Flint install from a LOCAL checkpoint directory (an `hf download`
//! of `mlx-community/flint-mixed-3-4bit` or `RecursiveSI/flint-int4`) through
//! the same walk `turbospark-model pull` runs, with a file instead of HTTP
//! behind the range reads. For bring-up and verification on a machine that
//! already holds the bytes; the catalog pull stays the install path users take.
//!
//!   cargo run --release -p turbospark-repack --example flint_local_intake -- \
//!       <checkpoint dir> <new install dir> [model id]
//!
//! Refuses an existing destination. Copies the four tokenizer and template
//! sidecars the catalog row lists. Writes no registry row: point
//! `turbospark run` at the directory.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use turbospark_repack::{DownloadError, RangeSource};

/// Positional reads from one shard file.
struct FileRangeSource(File);

impl RangeSource for FileRangeSource {
    fn read_range(&self, start: u64, end_exclusive: u64) -> Result<Vec<u8>, DownloadError> {
        let len = end_exclusive.saturating_sub(start);
        let mut buf = vec![0u8; len as usize];
        let mut done = 0usize;
        while done < buf.len() {
            match self.0.read_at(&mut buf[done..], start + done as u64) {
                Ok(0) => {
                    return Err(DownloadError::ShortRead {
                        expected: len,
                        actual: done as u64,
                    })
                }
                Ok(n) => done += n,
                Err(e) => return Err(DownloadError::Request(e.to_string())),
            }
        }
        Ok(buf)
    }
}

const SIDECARS: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
];

fn main() {
    if let Err(e) = run() {
        eprintln!("flint_local_intake: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: flint_local_intake <checkpoint dir> <new install dir> [model id]";
    let src = PathBuf::from(args.next().ok_or(usage)?);
    let dst = PathBuf::from(args.next().ok_or(usage)?);
    let model_id = args
        .next()
        .unwrap_or_else(|| src.file_name().unwrap().to_string_lossy().into_owned());
    if dst.exists() {
        return Err(format!("{} exists; pick a new directory", dst.display()).into());
    }

    let config = std::fs::read_to_string(src.join("config.json"))?;
    let arch = turbospark_repack::parse_flint_config(&config)?;
    let quant = turbospark_repack::parse_flint_quantization(&config)?;

    let sources: Vec<FileRangeSource> = shard_names(&src)?
        .iter()
        .map(|name| File::open(src.join(name)).map(FileRangeSource))
        .collect::<Result<_, _>>()?;
    let mut headers = sources
        .iter()
        .map(|s| turbospark_repack::fetch_safetensors_header(s))
        .collect::<Result<Vec<_>, _>>()?;
    for header in &mut headers {
        turbospark_repack::canonicalize_flint_header(header)?;
    }
    let shards = turbospark_repack::Gemma4Shards::new(
        headers
            .iter()
            .zip(&sources)
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    )?;

    std::fs::create_dir_all(&dst)?;
    for name in SIDECARS {
        std::fs::copy(src.join(name), dst.join(name))?;
    }
    turbospark_repack::write_flint_install_streamed(
        &dst,
        &arch,
        &model_id,
        &shards,
        &quant,
        |stage| eprintln!("[intake] {stage}"),
    )?;
    // What the catalog pull records after the walk (`stream.rs`), so
    // `--max-context auto` reads the same trained context from either.
    if let Some(n) = turbospark_repack::trained_context_meta::from_config_json(&config) {
        turbospark_repack::trained_context_meta::record(&dst, n)?;
    }
    eprintln!("installed {model_id} to {}", dst.display());
    Ok(())
}

/// The shard files the index names, in order.
fn shard_names(src: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        src.join("model.safetensors.index.json"),
    )?)?;
    let mut names: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or("index has no weight_map")?
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}
