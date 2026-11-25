//! Installs a LOCAL GGUF through the same walk `turbospark-model pull` runs,
//! with a file instead of HTTP behind the range reads. For bring-up and
//! verification of a GGUF you already hold, such as one made by the
//! `gguf_quantize` example. The catalog pull stays the install path users take.
//!
//!   cargo run --release -p turbospark-repack --example gguf_local_install -- \
//!       <model.gguf> <new install dir> <sidecar dir> [model id]
//!
//! A GGUF carries llama.cpp's tokenizer, not an HF one. `<sidecar dir>` may hold
//! the model's HF `tokenizer.json` and `tokenizer_config.json` (and optionally
//! `generation_config.json`), which win when present. Pass `-` or a directory
//! without `tokenizer.json` to rebuild them from the GGUF's own metadata
//! (byte-level BPE with a reviewed pre-tokenizer only; anything else is refused
//! by name). Refuses an existing destination and writes no registry row: point
//! `turbospark run` at the directory.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use turbospark_repack::{DownloadError, RangeSource};

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
        eprintln!("gguf_local_install: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: gguf_local_install <model.gguf> <new install dir> <sidecar dir> [model id]";
    let gguf = PathBuf::from(args.next().ok_or(usage)?);
    let dst = PathBuf::from(args.next().ok_or(usage)?);
    let sidecars = PathBuf::from(args.next().ok_or(usage)?);
    let model_id = args
        .next()
        .unwrap_or_else(|| gguf.file_stem().unwrap().to_string_lossy().into_owned());
    if dst.exists() {
        return Err(format!("{} exists; pick a new directory", dst.display()).into());
    }
    let source = FileRangeSource(File::open(&gguf)?);
    let header = turbospark_repack::fetch_gguf_header(&source)?;
    std::fs::create_dir_all(&dst)?;
    if !sidecars.join("tokenizer.json").exists() {
        let files = turbospark_repack::tokenizer_files_from_gguf(&header)?;
        std::fs::write(
            dst.join("tokenizer.json"),
            serde_json::to_vec(&files.tokenizer_json)?,
        )?;
        std::fs::write(
            dst.join("tokenizer_config.json"),
            serde_json::to_vec_pretty(&files.tokenizer_config_json)?,
        )?;
        eprintln!("[install] tokenizer rebuilt from GGUF metadata");
    }
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ] {
        let from = sidecars.join(name);
        if from.exists() {
            std::fs::copy(from, dst.join(name))?;
        }
    }
    turbospark_repack::write_gguf_install_streamed(&dst, &header, &source, &model_id, |stage| {
        eprintln!("[install] {stage}")
    })?;
    eprintln!("installed {model_id} to {}", dst.display());
    Ok(())
}
