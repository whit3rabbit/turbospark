//! Size-capped metadata reads.

use std::io::Read;
use std::path::Path;

/// Reads at most `max_bytes` from `path`, or `Ok(None)` when more is there.
///
/// Callers stat first for an early, exact-size error, but the stat alone does
/// not bound the read: a file that grows after the stat, a FIFO, or a symlink
/// to `/dev/zero` reports a small length and `fs::read` then takes whatever
/// arrives. Reading one byte past the cap is what enforces it.
pub(crate) fn read_capped(path: &Path, max_bytes: u64) -> std::io::Result<Option<Vec<u8>>> {
    let file = std::fs::File::open(path)?;
    let mut data = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut data)?;
    if data.len() as u64 > max_bytes {
        return Ok(None);
    }
    Ok(Some(data))
}

#[cfg(test)]
mod tests {
    use super::read_capped;

    /// A file exactly at the cap is legal; one byte more is not.
    #[test]
    fn cap_is_inclusive() {
        let dir =
            std::env::temp_dir().join(format!("turbospark-read-capped-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta.json");
        std::fs::write(&path, [b'x'; 16]).unwrap();
        assert_eq!(read_capped(&path, 16).unwrap().map(|d| d.len()), Some(16));
        assert_eq!(read_capped(&path, 15).unwrap(), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
