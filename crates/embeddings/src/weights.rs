//! Checkpoint access shared by every family.
//!
//! [`Weights`] opens the safetensors shards of a directory and answers
//! name lookups across them. Loads are shape-checked so a config that
//! disagrees with the checkpoint fails at load, not as noise at inference.
//! MLX affine-quantized layers are dequantized at load through
//! `turbospark_audio::quant`; a quantized embedding table stays packed and
//! is dequantized per gathered row (the Gemma vocabulary is 262144 rows,
//! about 800 MB as f32).

use std::path::{Path, PathBuf};

use turbospark_audio::nn::{open_shards, Linear};
use turbospark_audio::quant::{self, QuantScheme};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::error::{tensor_error, EmbeddingError};

/// The safetensors shards of one checkpoint directory.
pub struct Weights {
    dir: PathBuf,
    shards: Vec<SafetensorsFile>,
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights")
            .field("dir", &self.dir)
            .field("shards", &self.shards.len())
            .finish()
    }
}

impl Weights {
    /// Opens `model.safetensors` or the shards listed by
    /// `model.safetensors.index.json` under `dir`.
    pub fn open(dir: &Path) -> Result<Self, EmbeddingError> {
        Ok(Self {
            dir: dir.to_path_buf(),
            shards: open_shards(dir)?,
        })
    }

    /// Opens one explicit safetensors file (a sentence-transformers
    /// `2_Dense/model.safetensors`, for example).
    pub fn open_file(path: &Path) -> Result<Self, EmbeddingError> {
        Ok(Self {
            dir: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            shards: vec![SafetensorsFile::open(path)?],
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The shard holding `name`.
    pub fn file(&self, name: &str) -> Result<&SafetensorsFile, EmbeddingError> {
        self.shards
            .iter()
            .find(|s| s.contains_tensor(name))
            .ok_or_else(|| tensor_error(name, "tensor is missing"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.shards.iter().any(|s| s.contains_tensor(name))
    }

    /// Every tensor name across shards.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.shards.iter().flat_map(|s| s.tensor_names())
    }

    /// The stored shape of `name`.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>, EmbeddingError> {
        Ok(self
            .file(name)?
            .descriptor(name)
            .ok_or_else(|| tensor_error(name, "descriptor missing"))?
            .shape
            .clone())
    }

    /// Loads `name` as f32 and requires its stored shape to be `shape`.
    pub fn load(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, EmbeddingError> {
        let stored = self.shape(name)?;
        if stored != shape {
            return Err(tensor_error(
                name,
                format!("expected shape {shape:?}, got {stored:?}"),
            ));
        }
        Ok(self.file(name)?.load_as_f32(name)?)
    }

    /// Loads `name` as f32 with no shape check, returning the shape too.
    pub fn load_any(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), EmbeddingError> {
        let shape = self.shape(name)?;
        Ok((self.file(name)?.load_as_f32(name)?, shape))
    }

    /// True when `base` is an MLX-quantized layer (carries `.scales`).
    pub fn is_quantized(&self, base: &str) -> bool {
        self.contains(&format!("{base}.scales"))
    }

    /// Loads a linear layer `{base}.weight [output, input]` (plus
    /// `{base}.bias [output]` when `bias`), dequantizing under `scheme`
    /// when the checkpoint quantized it. A quantized layer with no scheme
    /// is refused.
    pub fn linear(
        &self,
        base: &str,
        input: usize,
        output: usize,
        bias: bool,
        scheme: Option<QuantScheme>,
    ) -> Result<Linear, EmbeddingError> {
        let weight_name = format!("{base}.weight");
        let (weight, loaded_bias) = if self.is_quantized(base) {
            let scheme = scheme.ok_or_else(|| {
                tensor_error(
                    &weight_name,
                    "quantized layer but config has no quantization",
                )
            })?;
            let (w, b) = quant::load_quantized(self.file(&weight_name)?, base, scheme)?;
            if w.len() != input * output {
                return Err(tensor_error(
                    &weight_name,
                    format!(
                        "dequantized to {} values, expected {output}x{input}",
                        w.len()
                    ),
                ));
            }
            (w, b)
        } else {
            let w = self.load(&weight_name, &[output, input])?;
            let bias_name = format!("{base}.bias");
            let b = if self.contains(&bias_name) {
                Some(self.load(&bias_name, &[output])?)
            } else {
                None
            };
            (w, b)
        };
        let bias_name = format!("{base}.bias");
        let loaded_bias = match (bias, loaded_bias) {
            (true, Some(b)) if b.len() == output => Some(b),
            (true, Some(b)) => {
                return Err(tensor_error(
                    &bias_name,
                    format!("expected {output} values, got {}", b.len()),
                ))
            }
            (true, None) => return Err(tensor_error(&bias_name, "tensor is missing")),
            (false, Some(_)) => {
                return Err(tensor_error(&bias_name, "present but config says no bias"))
            }
            (false, None) => None,
        };
        Ok(Linear::new(weight, loaded_bias, input, output))
    }

    /// An embedding table `{base}.weight [rows, cols]`, kept packed when
    /// the checkpoint quantized it.
    pub fn embedding(
        &self,
        base: &str,
        rows: usize,
        cols: usize,
        scheme: Option<QuantScheme>,
    ) -> Result<EmbeddingTable, EmbeddingError> {
        let weight_name = format!("{base}.weight");
        if !self.is_quantized(base) {
            return Ok(EmbeddingTable::Dense {
                weight: self.load(&weight_name, &[rows, cols])?,
                cols,
            });
        }
        let scheme = scheme.ok_or_else(|| {
            tensor_error(
                &weight_name,
                "quantized table but config has no quantization",
            )
        })?;
        if !matches!(scheme.bits, 2 | 3 | 4 | 5 | 6 | 8)
            || scheme.group_size == 0
            || cols % scheme.group_size != 0
        {
            return Err(tensor_error(
                &weight_name,
                format!("unsupported scheme {scheme:?} for width {cols}"),
            ));
        }
        let groups = cols / scheme.group_size;
        let words_per_row = (cols * scheme.bits as usize).div_ceil(32);
        let stored = self.shape(&weight_name)?;
        if stored != [rows, words_per_row] {
            return Err(tensor_error(
                &weight_name,
                format!("expected packed shape [{rows}, {words_per_row}], got {stored:?}"),
            ));
        }
        let scales = self.load(&format!("{base}.scales"), &[rows, groups])?;
        let biases = self.load(&format!("{base}.biases"), &[rows, groups])?;
        let packed = self.file(&weight_name)?.raw_bytes(&weight_name)?.to_vec();
        Ok(EmbeddingTable::Quantized {
            packed,
            scales,
            biases,
            scheme,
            rows,
            cols,
            words_per_row,
        })
    }
}

/// A token-embedding table, dense or MLX-packed.
#[derive(Debug, Clone)]
pub enum EmbeddingTable {
    Dense {
        weight: Vec<f32>,
        cols: usize,
    },
    Quantized {
        packed: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        scheme: QuantScheme,
        rows: usize,
        cols: usize,
        words_per_row: usize,
    },
}

impl EmbeddingTable {
    pub fn cols(&self) -> usize {
        match self {
            EmbeddingTable::Dense { cols, .. } | EmbeddingTable::Quantized { cols, .. } => *cols,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            EmbeddingTable::Dense { weight, cols } => weight.len() / cols,
            EmbeddingTable::Quantized { rows, .. } => *rows,
        }
    }

    /// Gathers `ids` into `[ids.len(), cols]`. An out-of-range id is an
    /// error, not a panic, because ids come from a tokenizer the caller
    /// may have swapped.
    pub fn gather(&self, ids: &[u32]) -> Result<Vec<f32>, EmbeddingError> {
        let cols = self.cols();
        let rows = self.rows();
        let mut out = vec![0.0f32; ids.len() * cols];
        for (slot, &id) in out.chunks_exact_mut(cols).zip(ids) {
            let r = id as usize;
            if r >= rows {
                return Err(EmbeddingError::Input {
                    why: format!("token id {id} out of range for {rows} embedding rows"),
                });
            }
            match self {
                EmbeddingTable::Dense { weight, .. } => {
                    slot.copy_from_slice(&weight[r * cols..(r + 1) * cols]);
                }
                EmbeddingTable::Quantized {
                    packed,
                    scales,
                    biases,
                    scheme,
                    words_per_row,
                    ..
                } => {
                    let groups = cols / scheme.group_size;
                    let bytes = words_per_row * 4;
                    quant::dequantize_rows(
                        slot,
                        &packed[r * bytes..(r + 1) * bytes],
                        *words_per_row,
                        *scheme,
                        groups,
                        &scales[r * groups..(r + 1) * groups],
                        &biases[r * groups..(r + 1) * groups],
                    );
                }
            }
        }
        Ok(out)
    }
}

/// The `quantization` block of an MLX `config.json`, when present.
pub fn quant_scheme(config: &serde_json::Value) -> Result<Option<QuantScheme>, EmbeddingError> {
    let Some(q) = config.get("quantization") else {
        return Ok(None);
    };
    let field = |k: &str| {
        q.get(k).and_then(serde_json::Value::as_u64).ok_or_else(|| {
            crate::error::bad_config(&format!("quantization.{k}"), "missing or not an integer")
        })
    };
    if let Some(mode) = q.get("mode").and_then(serde_json::Value::as_str) {
        if mode != "affine" {
            return Err(EmbeddingError::Unsupported {
                why: format!("quantization mode {mode}; only affine is implemented"),
            });
        }
    }
    Ok(Some(QuantScheme {
        bits: field("bits")? as u32,
        group_size: field("group_size")? as usize,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_gather_and_range_check() {
        let t = EmbeddingTable::Dense {
            weight: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            cols: 2,
        };
        assert_eq!(t.gather(&[2, 0]).unwrap(), vec![4.0, 5.0, 0.0, 1.0]);
        assert!(t.gather(&[3]).is_err());
    }

    #[test]
    fn quantized_gather_matches_full_dequantization() {
        // Two rows of 64 values at 4 bits: q = (row + i) % 16, scale 0.5,
        // bias -1 per group. Packed little-endian, 8 values per u32.
        let scheme = QuantScheme {
            bits: 4,
            group_size: 64,
        };
        let cols = 64;
        let words = cols * 4 / 32;
        let mut packed = Vec::new();
        let mut expect = Vec::new();
        for row in 0..2u32 {
            for w in 0..words {
                let mut word = 0u32;
                for k in 0..8 {
                    let qv = (row + (w * 8 + k) as u32) % 16;
                    word |= qv << (4 * k);
                    expect.push(qv as f32 * 0.5 - 1.0);
                }
                packed.extend_from_slice(&word.to_le_bytes());
            }
        }
        let t = EmbeddingTable::Quantized {
            packed,
            scales: vec![0.5, 0.5],
            biases: vec![-1.0, -1.0],
            scheme,
            rows: 2,
            cols,
            words_per_row: words,
        };
        assert_eq!(t.gather(&[1]).unwrap(), expect[cols..].to_vec());
        assert_eq!(t.gather(&[0, 1]).unwrap(), expect);
    }

    #[test]
    fn quant_scheme_reads_mlx_block() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"quantization": {"group_size": 64, "bits": 8}}"#).unwrap();
        assert_eq!(
            quant_scheme(&v).unwrap(),
            Some(QuantScheme {
                bits: 8,
                group_size: 64
            })
        );
        assert_eq!(quant_scheme(&serde_json::json!({})).unwrap(), None);
        let bad: serde_json::Value = serde_json::from_str(
            r#"{"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}"#,
        )
        .unwrap();
        assert!(quant_scheme(&bad).is_err());
    }
}
