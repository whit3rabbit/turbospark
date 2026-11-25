//! Writes the `tokenizer.json` and `tokenizer_config.json` that
//! `tokenizer_files_from_gguf` rebuilds from a GGUF's own metadata, so they can
//! be loaded and compared with the checkpoint's HF files (see
//! `crates/tokenizer/tests/gguf_tokenizer_encode_equality.rs`).
//!
//!   cargo run -p turbospark-repack --example gguf_tokenizer_dump -- <url or path> <out dir>
//!
//! Reads only the header (a ranged fetch for a URL).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use turbospark_repack::{DownloadError, HttpRangeSource, RangeSource};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: gguf_tokenizer_dump <url or path> <out dir>";
    let from = args.next().ok_or(usage)?;
    let out = PathBuf::from(args.next().ok_or(usage)?);
    let header = if from.starts_with("http") {
        turbospark_repack::fetch_gguf_header(&HttpRangeSource::new(from))?
    } else {
        turbospark_repack::fetch_gguf_header(&FileRangeSource(File::open(from)?))?
    };
    let files = turbospark_repack::tokenizer_files_from_gguf(&header)?;
    std::fs::create_dir_all(&out)?;
    std::fs::write(
        out.join("tokenizer.json"),
        serde_json::to_vec(&files.tokenizer_json)?,
    )?;
    std::fs::write(
        out.join("tokenizer_config.json"),
        serde_json::to_vec_pretty(&files.tokenizer_config_json)?,
    )?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
