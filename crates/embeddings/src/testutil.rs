//! Shared helpers for the family parity tests.
//!
//! Fixtures live under `testdata/<family>/` and are written by
//! `tools/gen_<family>_fixtures.py`. Real checkpoints are never read from
//! testdata: the `#[ignore]`d real-model tests resolve a directory from an
//! environment variable, falling back to `~/models/embeddings/<name>`, and
//! skip with a message when it is absent.

#![allow(dead_code)] // each family feature uses a subset

use std::path::PathBuf;

/// `crates/embeddings/testdata/<family>/<name>`.
pub fn testdata(family: &str, name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(family)
        .join(name)
}

pub fn read_json(family: &str, name: &str) -> serde_json::Value {
    let path = testdata(family, name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A little-endian `.npy` array as f32. Accepts `<f4`, `<f8`, `<i4`, and
/// `<i8` (ids and masks are stored as integers); C order only.
pub fn read_npy(family: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
    let path = testdata(family, name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "{}", path.display());
    let (header_len, offset) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[offset..offset + header_len]).unwrap();
    assert!(
        header.contains("'fortran_order': False"),
        "{}: Fortran order",
        path.display()
    );
    let start = header.find('(').unwrap() + 1;
    let end = header[start..].find(')').unwrap() + start;
    let shape: Vec<usize> = header[start..end]
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect();
    let body = &bytes[offset + header_len..];
    let data: Vec<f32> = if header.contains("'<f4'") {
        body.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    } else if header.contains("'<f8'") {
        body.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()
    } else if header.contains("'<i4'") {
        body.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()
    } else if header.contains("'<i8'") {
        body.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()
    } else {
        panic!("{}: unsupported dtype in {header}", path.display());
    };
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "{}",
        path.display()
    );
    (shape, data)
}

/// Integer view of [`read_npy`] for ids and masks.
pub fn read_npy_u32(family: &str, name: &str) -> (Vec<usize>, Vec<u32>) {
    let (shape, data) = read_npy(family, name);
    (shape, data.into_iter().map(|v| v as u32).collect())
}

/// Largest absolute elementwise difference. Panics on a length mismatch.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Real checkpoint directory for an `#[ignore]`d test: `$env_var`, else
/// `~/models/embeddings/<default_name>`. `None` (with a skip message) when
/// neither exists, so an accidental `--ignored` run without the install
/// is a no-op rather than a failure.
pub fn model_dir(env_var: &str, default_name: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env_var).map(PathBuf::from).or_else(|| {
        std::env::var_os("HOME").map(|h| {
            PathBuf::from(h)
                .join("models")
                .join("embeddings")
                .join(default_name)
        })
    })?;
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("skip: {} not found (set {env_var})", dir.display());
        None
    }
}
