//! The forced-decode logit trace: f16 rows with a header and a row-count
//! footer, so a truncated file is refused instead of read short.
//!
//! Layout: `TSLT1 <vocab>\n`, then `rows * vocab` little-endian f16 values,
//! then `END <rows>\n`. Rows are written one at a time, so a long trace never
//! sits in the worker's footprint.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &str = "TSLT1";
/// Longest footer: `END ` plus a u64 and the newline.
const FOOTER_MAX: u64 = 25;

pub struct TraceWriter {
    out: BufWriter<File>,
    path: PathBuf,
    vocab: usize,
    rows: u64,
}

impl TraceWriter {
    /// A new file; an existing one is refused.
    pub fn create(path: &Path, vocab: usize) -> Result<Self, String> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(err)?;
        let mut out = BufWriter::new(file);
        writeln!(out, "{MAGIC} {vocab}").map_err(err)?;
        Ok(Self {
            out,
            path: path.to_path_buf(),
            vocab,
            rows: 0,
        })
    }

    /// One row of f16 bit patterns.
    pub fn row(&mut self, bits: &[u8]) -> Result<(), String> {
        if bits.len() != 2 * self.vocab {
            return Err(format!(
                "trace row is {} bytes, vocab {} needs {}",
                bits.len(),
                self.vocab,
                2 * self.vocab
            ));
        }
        self.out
            .write_all(bits)
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.rows += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64, String> {
        let err = |e: std::io::Error| format!("{}: {e}", self.path.display());
        writeln!(self.out, "END {}", self.rows).map_err(err)?;
        self.out.flush().map_err(err)?;
        Ok(self.rows)
    }
}

pub struct TraceReader {
    input: BufReader<File>,
    path: PathBuf,
    pub vocab: usize,
    pub rows: u64,
    read: u64,
}

impl TraceReader {
    /// Open and check the header, the footer and the size against each other.
    pub fn open(path: &Path) -> Result<Self, String> {
        let bad = |what: &str| format!("{}: {what}", path.display());
        let file = File::open(path).map_err(|e| bad(&e.to_string()))?;
        let len = file.metadata().map_err(|e| bad(&e.to_string()))?.len();
        let mut input = BufReader::new(file);
        let mut header = String::new();
        input
            .read_line(&mut header)
            .map_err(|e| bad(&e.to_string()))?;
        let vocab: usize = header
            .trim_end()
            .strip_prefix(MAGIC)
            .and_then(|v| v.trim().parse().ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| bad("not a logit trace"))?;
        let tail_at = len.saturating_sub(FOOTER_MAX).max(header.len() as u64);
        input
            .seek(SeekFrom::Start(tail_at))
            .map_err(|e| bad(&e.to_string()))?;
        let mut tail = Vec::new();
        input
            .read_to_end(&mut tail)
            .map_err(|e| bad(&e.to_string()))?;
        let at = tail
            .windows(4)
            .rposition(|w| w == b"END ")
            .ok_or_else(|| bad("no row-count footer: the trace is truncated"))?;
        let rows: u64 = std::str::from_utf8(&tail[at + 4..])
            .ok()
            .and_then(|s| s.strip_suffix('\n'))
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| bad("unreadable footer: the trace is truncated"))?;
        let body = rows
            .checked_mul(2 * vocab as u64)
            .and_then(|b| b.checked_add(header.len() as u64 + (tail.len() - at) as u64));
        if body != Some(len) {
            return Err(bad(&format!(
                "size {len} does not hold the footer's {rows} rows of {vocab}"
            )));
        }
        input
            .seek(SeekFrom::Start(header.len() as u64))
            .map_err(|e| bad(&e.to_string()))?;
        Ok(Self {
            input,
            path: path.to_path_buf(),
            vocab,
            rows,
            read: 0,
        })
    }

    /// The next row as f64 logits, `None` after the last.
    pub fn next_row(&mut self) -> Result<Option<Vec<f64>>, String> {
        if self.read == self.rows {
            return Ok(None);
        }
        let mut bytes = vec![0u8; 2 * self.vocab];
        self.input
            .read_exact(&mut bytes)
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.read += 1;
        Ok(Some(
            bytes
                .chunks_exact(2)
                .map(|b| f64::from(f16_to_f32(u16::from_le_bytes([b[0], b[1]]))))
                .collect(),
        ))
    }
}

/// IEEE binary16 to f32. Here so the portable half of the crate can read a
/// trace without the macOS-only `half` dependency.
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from(h >> 10) & 0x1f;
    let frac = u32::from(h) & 0x3ff;
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: value is frac * 2^-24.
            let v = frac as f32 * (1.0 / 16_777_216.0);
            return if sign == 0 { v } else { -v };
        }
        (31, 0) => sign | 0x7f80_0000,
        (31, _) => sign | 0x7fc0_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("tslab-trace-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn f16_values_convert() {
        for (bits, v) in [
            (0x0000, 0.0f32),
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x3555, 0.333_251_95),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x8400, -6.103_515_6e-5),
            (0x7c00, f32::INFINITY),
        ] {
            assert_eq!(f16_to_f32(bits), v, "{bits:#x}");
        }
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn rows_round_trip_and_truncation_is_refused() {
        let p = temp("rt");
        let mut w = TraceWriter::create(&p, 2).unwrap();
        // 1.0, -2.0 then 0.0, 1.0
        w.row(&[0x00, 0x3c, 0x00, 0xc0]).unwrap();
        w.row(&[0x00, 0x00, 0x00, 0x3c]).unwrap();
        assert!(w.row(&[0, 0]).is_err());
        assert_eq!(w.finish().unwrap(), 2);
        let mut r = TraceReader::open(&p).unwrap();
        assert_eq!((r.vocab, r.rows), (2, 2));
        assert_eq!(r.next_row().unwrap().unwrap(), vec![1.0, -2.0]);
        assert_eq!(r.next_row().unwrap().unwrap(), vec![0.0, 1.0]);
        assert!(r.next_row().unwrap().is_none());
        // Cut the footer, then drop a row but keep the footer.
        let full = std::fs::read(&p).unwrap();
        std::fs::write(&p, &full[..full.len() - 3]).unwrap();
        assert!(TraceReader::open(&p).is_err());
        let mut short = full.clone();
        short.drain(8..12);
        std::fs::write(&p, &short).unwrap();
        assert!(TraceReader::open(&p).is_err());
        // An existing file is never overwritten.
        assert!(TraceWriter::create(&p, 2).is_err());
        std::fs::remove_file(&p).unwrap();
    }
}
