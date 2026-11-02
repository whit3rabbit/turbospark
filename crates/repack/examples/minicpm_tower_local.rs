//! Builds a TOWER-ONLY directory for MiniCPM-V 4.7 from a LOCAL checkpoint
//! (an `hf download` of shards 1 and 5 of
//! `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4`) through the same reader and
//! writer the combined install uses (`read_vision_entries_quantized`,
//! `write_vision_sidecar`), with a file instead of HTTP behind the range reads.
//!
//! For tower parity tests on a machine that holds the shards but not yet the
//! 20 GB trunk. **This directory is a test fixture, not an install**: the
//! catalog refuses a MiniCPM-V sidecar because its pairing record cannot tell
//! its trunk from a Qwen3.6 trunk of the same width, and the runtime refuses
//! to attach one (`encode_image_minicpm`).
//!
//!   cargo run --release -p turbospark-repack --example minicpm_tower_local -- \
//!       <checkpoint dir> <new tower dir>

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use turbospark_repack::{DownloadError, Gemma4Shards, RangeSource};

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

fn main() {
    if let Err(e) = run() {
        eprintln!("minicpm_tower_local: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: minicpm_tower_local <checkpoint dir> <new tower dir>";
    let src = PathBuf::from(args.next().ok_or(usage)?);
    let dst = PathBuf::from(args.next().ok_or(usage)?);
    if dst.exists() {
        return Err(format!("{} exists; pick a new directory", dst.display()).into());
    }
    let config = std::fs::read_to_string(src.join("config.json"))?;
    let vision = turbospark_repack::parse_vision_config(&config)?;
    if !vision.is_active() {
        return Err("config.json declares no vision tower".into());
    }

    // The shards the index names for the three tower prefixes.
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        src.join("model.safetensors.index.json"),
    )?)?;
    let mut names: Vec<String> = index["weight_map"]
        .as_object()
        .ok_or("index has no weight_map")?
        .iter()
        .filter(|(k, _)| {
            turbospark_repack::MINICPM_SOURCE_PREFIXES
                .iter()
                .any(|p| k.starts_with(p))
        })
        .map(|(_, v)| v.as_str().unwrap_or_default().to_string())
        .collect();
    names.sort();
    names.dedup();

    let sources: Vec<FileRangeSource> = names
        .iter()
        .map(|name| File::open(src.join(name)).map(FileRangeSource))
        .collect::<Result<_, _>>()?;
    let mut headers = sources
        .iter()
        .map(|s| turbospark_repack::fetch_safetensors_header(s))
        .collect::<Result<Vec<_>, _>>()?;
    for header in &mut headers {
        header.tensors.retain(|name, _| {
            turbospark_repack::MINICPM_SOURCE_PREFIXES
                .iter()
                .any(|p| name.starts_with(p))
        });
    }
    let shards = Gemma4Shards::new(
        headers
            .iter()
            .zip(sources.iter())
            .map(|(h, s)| (h, s as &dyn RangeSource))
            .collect(),
    )?;
    let bases: Vec<&str> = headers
        .iter()
        .flat_map(|h| h.tensors.keys())
        .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
        .map(String::as_str)
        .collect();
    println!("{} vision tensors in {} shard(s)", bases.len(), names.len());
    let read = turbospark_repack::read_vision_entries_quantized(&shards, &bases, &vision, None)?;
    println!(
        "{} resident entries, {} blocks, stride {}",
        read.entries.len(),
        read.blocks.experts.len(),
        read.block_stride
    );
    let family = model_io::ModelFamily::QwenGdnMoe;
    let record = model_io::SidecarRecord {
        kind: model_io::SIDECAR_KIND.to_string(),
        pairs_with: model_io::PairsWith {
            family: family.as_str().to_string(),
            hidden_size: vision.out_hidden_size,
        },
        source: model_io::SidecarSource {
            repo: "mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4".to_string(),
            revision: "810419dd09762d63aad6da5b6a418c4af8946e60".to_string(),
            prefix: turbospark_repack::VISION_PREFIX.to_string(),
            file: names.join(", "),
        },
        tower_blocks: vision.depth,
        block_stride: read.block_stride,
    };
    turbospark_repack::write_vision_sidecar(
        &dst,
        family,
        &vision,
        "mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4",
        &read,
        record,
    )?;
    std::fs::copy(
        src.join("preprocessor_config.json"),
        dst.join("preprocessor_config.json"),
    )?;
    println!("wrote {}", dst.display());
    Ok(())
}
