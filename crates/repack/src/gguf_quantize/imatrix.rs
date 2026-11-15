//! Reader for llama.cpp importance-matrix files, in both on-disk formats.
//!
//! Semantics follow llama.cpp's `common/imatrix-loader.cpp` and the
//! normalization in `tools/quantize/quantize.cpp::load_imatrix`:
//!
//! - **Legacy `.dat`**: `i32 n_entries`, then per entry `i32 name_len`, name,
//!   `i32 ncall`, `i32 nval`, `nval` f32 sums. Value = sum / ncall when
//!   `ncall > 0`, else the raw sum. An optional trailer (`i32 chunk_count`,
//!   `i32 len`, dataset name) is read and kept.
//! - **GGUF**: tensors `<name>.in_sum2` and `<name>.counts` (both F32; an
//!   optional `<name>.in_sum` is ignored). `counts` has one entry per expert
//!   and `in_sum2` is `counts.len()` slices of `ne0` values, so value =
//!   sum / count per slice, and a slice with count 0 becomes all ones.
//!
//! The whole file is read into memory. Large MoE imatrices run to gigabytes,
//! so lazy per-tensor reads are a later improvement, not done here.

use std::collections::HashMap;
use std::path::Path;

use crate::gguf_header::{parse_header, GgufValue, DEFAULT_MAX_HEADER_BYTES};

#[derive(Debug)]
pub struct ImatrixError(pub String);

impl std::fmt::Display for ImatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "imatrix: {}", self.0)
    }
}

impl std::error::Error for ImatrixError {}

fn err<T>(m: impl Into<String>) -> Result<T, ImatrixError> {
    Err(ImatrixError(m.into()))
}

#[derive(Debug, Default)]
pub struct Imatrix {
    entries: HashMap<String, Vec<f32>>,
    pub datasets: Vec<String>,
    pub chunk_count: u32,
    pub legacy: bool,
}

impl Imatrix {
    pub fn load(path: &Path) -> Result<Imatrix, ImatrixError> {
        let bytes =
            std::fs::read(path).map_err(|e| ImatrixError(format!("{}: {e}", path.display())))?;
        Imatrix::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Imatrix, ImatrixError> {
        let im = if bytes.starts_with(b"GGUF") {
            Self::from_gguf(bytes)?
        } else {
            Self::from_legacy(bytes)?
        };
        // llama.cpp refuses non-finite weights rather than quantizing with them.
        if let Some((name, _)) = im
            .entries
            .iter()
            .find(|(_, v)| v.iter().any(|x| !x.is_finite()))
        {
            return err(format!("entry {name} contains a non-finite value"));
        }
        Ok(im)
    }

    /// Per-column weights for tensor `name`: `ne0` values, or `ne0 * n_expert`
    /// for an expert tensor (slice `e` belongs to expert `e`).
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.entries.get(name).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn from_legacy(b: &[u8]) -> Result<Imatrix, ImatrixError> {
        struct C<'a>(&'a [u8], usize);
        impl<'a> C<'a> {
            fn take(&mut self, n: usize) -> Result<&'a [u8], ImatrixError> {
                let end = self.1.checked_add(n).filter(|e| *e <= self.0.len());
                match end {
                    Some(e) => {
                        let s = &self.0[self.1..e];
                        self.1 = e;
                        Ok(s)
                    }
                    None => err("truncated legacy file"),
                }
            }
            fn i32(&mut self) -> Result<i32, ImatrixError> {
                Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
            }
        }
        let mut c = C(b, 0);
        let n = c.i32()?;
        if n < 1 {
            return err("no entries");
        }
        let mut im = Imatrix {
            legacy: true,
            ..Default::default()
        };
        for _ in 0..n {
            let len = c.i32()?;
            let len =
                usize::try_from(len).map_err(|_| ImatrixError("negative name length".into()))?;
            let name = String::from_utf8_lossy(c.take(len)?).into_owned();
            let ncall = c.i32()?;
            let nval = c.i32()?;
            if nval < 1 {
                return err(format!("entry {name}: no values"));
            }
            let raw = c.take(nval as usize * 4)?;
            let div = if ncall > 0 { Some(ncall as f32) } else { None };
            let vals = raw
                .chunks_exact(4)
                .map(|x| {
                    let s = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                    div.map_or(s, |d| s / d)
                })
                .collect();
            im.entries.insert(name, vals);
        }
        // Optional trailer. A short or absent one is not an error.
        if let Ok(calls) = c.i32() {
            im.chunk_count = calls.max(0) as u32;
            if let Ok(len) = c.i32() {
                if len > 0 {
                    if let Ok(s) = c.take(len as usize) {
                        im.datasets.push(String::from_utf8_lossy(s).into_owned());
                    }
                }
            }
        }
        Ok(im)
    }

    fn from_gguf(b: &[u8]) -> Result<Imatrix, ImatrixError> {
        let h =
            parse_header(b, DEFAULT_MAX_HEADER_BYTES).map_err(|e| ImatrixError(e.to_string()))?;
        if h.tensors.is_empty() {
            return err("no tensors");
        }
        let mut im = Imatrix::default();
        if let Some(GgufValue::Array(a)) = h.metadata.get("imatrix.datasets") {
            im.datasets = a
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
        im.chunk_count = h.metadata_u64("imatrix.chunk_count").unwrap_or(0) as u32;

        let f32s = |name: &str| -> Result<Vec<f32>, ImatrixError> {
            let info = &h.tensors[name];
            if info.ggml_type != 0 {
                return err(format!("{name} must be F32"));
            }
            let (s, e) = match h.absolute_range(name) {
                Some(Ok(r)) => r,
                _ => return err(format!("{name}: bad range")),
            };
            let raw = b
                .get(s as usize..e as usize)
                .ok_or_else(|| ImatrixError(format!("{name}: data past end of file")))?;
            Ok(raw
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                .collect())
        };

        let mut bases: Vec<&str> = h
            .tensors
            .keys()
            .filter_map(|k| k.strip_suffix(".in_sum2"))
            .collect();
        bases.sort_unstable();
        for base in bases {
            let counts_name = format!("{base}.counts");
            if !h.tensors.contains_key(&counts_name) {
                return err(format!("{base}: in_sum2 without counts"));
            }
            let sums = f32s(&format!("{base}.in_sum2"))?;
            let counts: Vec<f32> = f32s(&counts_name)?.into_iter().map(f32::round).collect();
            if counts.is_empty() || sums.len() % counts.len() != 0 {
                return err(format!(
                    "{base}: {} sums do not split into {} experts",
                    sums.len(),
                    counts.len()
                ));
            }
            let ne0 = sums.len() / counts.len();
            let mut out = vec![0f32; sums.len()];
            for (j, &cnt) in counts.iter().enumerate() {
                for i in 0..ne0 {
                    out[j * ne0 + i] = if cnt > 0.0 {
                        sums[j * ne0 + i] / cnt
                    } else {
                        1.0
                    };
                }
            }
            im.entries.insert(base.to_string(), out);
        }
        if im.entries.is_empty() {
            return err("no <name>.in_sum2 tensors found");
        }
        Ok(im)
    }

    /// Build from raw entries (tests and tools).
    pub fn from_entries(entries: HashMap<String, Vec<f32>>) -> Imatrix {
        Imatrix {
            entries,
            ..Default::default()
        }
    }
}
