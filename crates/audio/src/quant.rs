//! MLX groupwise affine dequantization over safetensors checkpoints.
//!
//! mlx-community conversions quantize linear weights to unsigned integers
//! packed into U32 words as a continuous little-endian bit stream per row.
//! Values at 3/5/6 bits can span word boundaries. This matches Apple's MLX
//! `mlx/backend/metal/kernels/quantized.h` at v0.31.1; floor-packing values
//! into each word changes row sizes and silently decodes the wrong weights.
//! Per-group F16 scales and biases dequantize as `w = q * scale + bias`.
//! Kernel fixtures cover packing arithmetic. They do not qualify a model.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{Result, SpeechError};

/// A quantization scheme: bits per value and values per affine group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantScheme {
    pub bits: u32,
    pub group_size: usize,
}

/// True when `base` is a quantized linear (carries `.scales`).
pub fn is_quantized(file: &SafetensorsFile, base: &str) -> bool {
    file.contains_tensor(&format!("{base}.scales"))
}

/// Loads one linear layer as `(weight [out, in], bias)` under `scheme`.
/// The weight is dequantized when the checkpoint carries
/// `.scales`/`.biases` for `base`, otherwise plain F32/F16/BF16 loading
/// applies (a plain layer inside a quantized checkpoint is accepted). The
/// bias is `None` when the checkpoint has none. The scheme must match the
/// checkpoint's config quantization block; a mismatched scheme fails the
/// row-width check rather than decoding noise.
pub fn load_quantized(
    file: &SafetensorsFile,
    base: &str,
    scheme: QuantScheme,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let bias_name = format!("{base}.bias");
    let bias = if file.contains_tensor(&bias_name) {
        Some(file.load_as_f32(&bias_name)?)
    } else {
        None
    };
    if !is_quantized(file, base) {
        let weight = file.load_as_f32(&format!("{base}.weight"))?;
        return Ok((weight, bias));
    }
    if !matches!(scheme.bits, 2 | 3 | 4 | 5 | 6 | 8) || scheme.group_size == 0 {
        return Err(SpeechError::Unsupported {
            why: "affine quantization requires 2/3/4/5/6/8 bits and a positive group size"
                .to_string(),
        });
    }
    let weight_name = format!("{base}.weight");
    let scales = file.load_as_f32(&format!("{base}.scales"))?;
    let biases = file.load_as_f32(&format!("{base}.biases"))?;
    let desc = file
        .descriptor(&weight_name)
        .ok_or_else(|| SpeechError::Tensor {
            name: weight_name.clone(),
            why: "descriptor missing".to_string(),
        })?;
    if desc.shape.len() != 2 || desc.shape.contains(&0) || desc.dtype != "U32" {
        return Err(SpeechError::Tensor {
            name: weight_name.clone(),
            why: format!("expected 2-D quantized weight, got shape {:?}", desc.shape),
        });
    }
    let out_dim = desc.shape[0];
    let words_per_row = desc.shape[1];
    let groups = scales.len() / out_dim;
    if groups == 0 || scales.len() % out_dim != 0 {
        return Err(SpeechError::Tensor {
            name: format!("{base}.scales"),
            why: format!(
                "scales length {} does not tile output rows {}",
                scales.len(),
                out_dim
            ),
        });
    }
    if biases.len() != scales.len() {
        return Err(SpeechError::Tensor {
            name: format!("{base}.biases"),
            why: format!(
                "biases length {} != scales length {}",
                biases.len(),
                scales.len()
            ),
        });
    }
    let in_dim = groups
        .checked_mul(scheme.group_size)
        .ok_or_else(|| SpeechError::Tensor {
            name: weight_name.clone(),
            why: "input width overflows".to_string(),
        })?;
    let row_bits = in_dim
        .checked_mul(scheme.bits as usize)
        .ok_or_else(|| SpeechError::Tensor {
            name: weight_name.clone(),
            why: "packed row width overflows".to_string(),
        })?;
    let expect_words = row_bits.div_ceil(32);
    if words_per_row != expect_words {
        return Err(SpeechError::Tensor {
            name: weight_name.clone(),
            why: format!(
                "row width {words_per_row} U32 words != {} for {}-bit groups of {} (in dim {in_dim})",
                expect_words,
                scheme.bits,
                scheme.group_size
            ),
        });
    }
    let raw = file.raw_bytes(&weight_name)?;
    if raw.len() != out_dim * words_per_row * 4 {
        return Err(SpeechError::Tensor {
            name: weight_name.clone(),
            why: format!(
                "byte length {} != {} U32 words",
                raw.len(),
                out_dim * words_per_row
            ),
        });
    }
    let output_len = out_dim
        .checked_mul(in_dim)
        .filter(|&count| count <= isize::MAX as usize / 4)
        .ok_or_else(|| SpeechError::Tensor {
            name: weight_name.clone(),
            why: "dequantized shape overflows".to_string(),
        })?;
    if bias.as_ref().is_some_and(|values| values.len() != out_dim) {
        return Err(SpeechError::Tensor {
            name: bias_name,
            why: "bias must have one value per output row".to_string(),
        });
    }
    let mut out = vec![0.0f32; output_len];
    let mask: u32 = if scheme.bits >= 32 {
        u32::MAX
    } else {
        (1u32 << scheme.bits) - 1
    };
    let value_at = |row: usize, v: usize, words: &[u8]| -> f32 {
        let bit = v * scheme.bits as usize;
        let offset = row * words_per_row * 4 + bit / 8;
        let shift = bit % 8;
        // At most six bits can remain above this byte. A second byte is
        // needed only when the value crosses the byte boundary.
        let packed = words[offset] as u32
            | if shift + scheme.bits as usize > 8 {
                (words[offset + 1] as u32) << 8
            } else {
                0
            };
        let q = ((packed >> shift) & mask) as f32;
        let g = v / scheme.group_size;
        q * scales[row * groups + g] + biases[row * groups + g]
    };
    for r in 0..out_dim {
        for v in 0..in_dim {
            out[r * in_dim + v] = value_at(r, v, raw);
        }
    }
    Ok((out, bias))
}

/// Reads a `config.json` style JSON object from `path`.
pub fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text = std::fs::read_to_string(path).map_err(|e| SpeechError::BadConfig {
        field: path.display().to_string(),
        why: e.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
        field: path.display().to_string(),
        why: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a minimal safetensors file: F32 or U32 tensors keyed by
    /// name. The format is an 8-byte little-endian header length, a JSON
    /// header of `name -> {dtype, shape, data_offsets}`, then the data
    /// blob.
    pub(super) fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    pub(super) fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    pub(super) fn u32_bytes(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Packs `values` (one row, length `in_dim`) into U32 words for the
    /// scheme, mirroring the MLX little-endian value order.
    fn pack_row(values: &[u32], bits: u32) -> Vec<u32> {
        let mut out = vec![0u32; (values.len() * bits as usize).div_ceil(32)];
        for (i, &v) in values.iter().enumerate() {
            let bit = i * bits as usize;
            out[bit / 32] |= v << (bit % 32);
            if bit % 32 + bits as usize > 32 {
                out[bit / 32 + 1] |= v >> (32 - bit % 32);
            }
        }
        out
    }

    fn quant_fixture(path: &Path, bits: u32, group_size: usize, in_dim: usize, out_dim: usize) {
        // Quantized values q[r][v] = (r*in + v) % (1 << bits); scales 0.1,
        // biases -0.05 so dequant = q * 0.1 - 0.05.
        assert_eq!(in_dim % group_size, 0);
        let words_per_row = (in_dim * bits as usize).div_ceil(32);
        let groups = in_dim / group_size;
        let mut words = Vec::new();
        let max = (1u32 << bits) - 1;
        for r in 0..out_dim {
            let row: Vec<u32> = (0..in_dim)
                .map(|v| ((r * in_dim + v) as u32) % (max + 1))
                .collect();
            words.extend(pack_row(&row, bits));
        }
        assert_eq!(words.len(), out_dim * words_per_row);
        let scales = vec![0.1f32; out_dim * groups];
        let biases = vec![-0.05f32; out_dim * groups];
        let bias = vec![7.0f32; out_dim];
        write_safetensors(
            path,
            &[
                (
                    "proj.weight",
                    "U32",
                    vec![out_dim, words_per_row],
                    u32_bytes(&words),
                ),
                (
                    "proj.scales",
                    "F32",
                    vec![out_dim, groups],
                    f32_bytes(&scales),
                ),
                (
                    "proj.biases",
                    "F32",
                    vec![out_dim, groups],
                    f32_bytes(&biases),
                ),
                ("proj.bias", "F32", vec![out_dim], f32_bytes(&bias)),
            ],
        );
    }

    #[test]
    fn dequant_8bit_group64_matches_whisper_verified_scheme() {
        let dir = std::env::temp_dir().join(format!("speech-quant-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q8.safetensors");
        let (out_dim, in_dim) = (2usize, 128usize);
        quant_fixture(&path, 8, 64, in_dim, out_dim);
        let file = SafetensorsFile::open(&path).unwrap();
        let (w, b) = load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 8,
                group_size: 64,
            },
        )
        .unwrap();
        assert_eq!(b.unwrap(), vec![7.0; out_dim]);
        let max = 255u32;
        for r in 0..out_dim {
            for v in 0..in_dim {
                let q = ((r * in_dim + v) as u32 % (max + 1)) as f32;
                assert!(
                    (w[r * in_dim + v] - (q * 0.1 - 0.05)).abs() < 1e-5,
                    "r{r} v{v} got {}",
                    w[r * in_dim + v]
                );
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dequant_4bit_low_nibble_first() {
        let dir = std::env::temp_dir().join(format!("speech-quant4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q4.safetensors");
        let (out_dim, in_dim) = (1usize, 64usize);
        quant_fixture(&path, 4, 64, in_dim, out_dim);
        let file = SafetensorsFile::open(&path).unwrap();
        let (w, _) = load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 4,
                group_size: 64,
            },
        )
        .unwrap();
        for (v, &wv) in w.iter().enumerate().take(in_dim) {
            let q = (v % 16) as f32;
            assert!((wv - (q * 0.1 - 0.05)).abs() < 1e-5, "v{v} got {wv}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dequant_6bit_crosses_word_boundaries() {
        // Value 5 straddles bits 30..35 of the first two U32 words.
        let dir = std::env::temp_dir().join(format!("speech-quant6-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q6.safetensors");
        let (out_dim, in_dim, group_size) = (1usize, 40usize, 20usize);
        quant_fixture(&path, 6, group_size, in_dim, out_dim);
        let file = SafetensorsFile::open(&path).unwrap();
        let (w, _) = load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 6,
                group_size,
            },
        )
        .unwrap();
        let max = 63u32;
        for (v, &wv) in w.iter().enumerate().take(in_dim) {
            let q = ((v as u32) % (max + 1)) as f32;
            assert!((wv - (q * 0.1 - 0.05)).abs() < 1e-5, "v{v} got {wv}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn plain_layer_passes_through() {
        let dir = std::env::temp_dir().join(format!("speech-plain-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plain.safetensors");
        write_safetensors(
            &path,
            &[
                (
                    "proj.weight",
                    "F32",
                    vec![2, 2],
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                ),
                ("proj.bias", "F32", vec![2], f32_bytes(&[0.5, -0.5])),
            ],
        );
        let file = SafetensorsFile::open(&path).unwrap();
        let (w, b) = load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 4,
                group_size: 64,
            },
        )
        .unwrap();
        assert_eq!(w, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b, Some(vec![0.5, -0.5]));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn wrong_scheme_refuses() {
        let dir = std::env::temp_dir().join(format!("speech-bits-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q8.safetensors");
        quant_fixture(&path, 8, 64, 128, 2);
        let file = SafetensorsFile::open(&path).unwrap();
        let err = load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 4,
                group_size: 64,
            },
        );
        assert!(err.is_err(), "4-bit read of an 8-bit tensor must refuse");
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod packing_regression {
    use super::tests::{f32_bytes, u32_bytes, write_safetensors};
    use super::*;

    #[test]
    fn six_bit_fixture_matches_mlx_contiguous_bitstream() {
        let path = std::env::temp_dir().join(format!(
            "speech-contiguous6-{}.safetensors",
            std::process::id()
        ));
        // 0,1,2,3,4,5,6,7 as six-bit values, repeated twice. Word 0
        // ends after the low two bits of value 5; word 1 continues it.
        write_safetensors(
            &path,
            &[
                (
                    "proj.weight",
                    "U32",
                    vec![1, 3],
                    u32_bytes(&[0x440c2040, 0x20401c61, 0x1c61440c]),
                ),
                ("proj.scales", "F32", vec![1, 1], f32_bytes(&[1.0])),
                ("proj.biases", "F32", vec![1, 1], f32_bytes(&[0.0])),
            ],
        );
        let file = SafetensorsFile::open(&path).unwrap();
        let (weight, _) = load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 6,
                group_size: 16,
            },
        )
        .unwrap();
        assert_eq!(weight, (0..16).map(|n| (n % 8) as f32).collect::<Vec<_>>());
        assert!(load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 0,
                group_size: 16
            }
        )
        .is_err());
        assert!(load_quantized(
            &file,
            "proj",
            QuantScheme {
                bits: 6,
                group_size: 0
            }
        )
        .is_err());
        std::fs::remove_file(path).unwrap();
    }
}
