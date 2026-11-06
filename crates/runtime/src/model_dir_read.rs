//! Capped reads for small model-dir files (configs, tokenizers).
//!
//! These files come from a downloaded checkpoint, so an uncapped `fs::read`
//! lets a crafted multi-GB `config.json` exhaust memory before any field is
//! checked. Same pattern as `whisper/admission.rs` and `coreml/parakeet.rs`.

use std::io::Read;
use std::path::Path;

/// Real installs' configs are under 20 KB.
pub(crate) const CONFIG_LIMIT: u64 = 1 << 20;
/// Real installs' `tokenizer.json` files reach 32 MB (Gemma 4); the
/// headroom keeps a larger real vocabulary from being refused.
pub(crate) const TOKENIZER_LIMIT: u64 = 64 << 20;

/// Reads `path` whole, refusing past `limit` bytes. Open errors keep their
/// `io::ErrorKind` so callers that treat `NotFound` as optional still can.
/// `take` caps a file that grows or lies about its size.
pub(crate) fn read_bounded(path: &Path, limit: u64) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("exceeds the {limit} byte limit"),
        ));
    }
    Ok(bytes)
}

/// [`read_bounded`] as UTF-8 text, for callers that parsed `read_to_string`.
pub(crate) fn read_bounded_string(path: &Path, limit: u64) -> std::io::Result<String> {
    String::from_utf8(read_bounded(path, limit)?)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
