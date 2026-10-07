//! Shared layer types for the model families.
//!
//! Every speech family used to carry its own private `Linear` and
//! `LayerNorm` struct around the same `ops` call. These are those structs:
//! owned f32 weights plus the one kernel call, with nothing added, so a
//! family that adopts them computes exactly what its copy did. Loading
//! stays with the family when it is not the plain f32 path (quantized
//! checkpoints, sharded files, name remapping), via [`Linear::new`] and
//! [`LayerNorm::new`].

use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

/// Loads a tensor as f32 and requires its safetensors shape to be `shape`.
///
/// The messages are the ones the STT families already reported, so tests
/// and callers that match on them keep working.
pub fn load_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let descriptor = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_owned(),
        why: "tensor is missing".into(),
    })?;
    if descriptor.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_owned(),
            why: format!("expected shape {shape:?}, got {:?}", descriptor.shape),
        });
    }
    Ok(file.load_as_f32(name)?)
}

/// `y = x @ weight.T + bias` with `weight` stored `[output, input]` (the
/// HF layout [`ops::linear`] takes).
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub input: usize,
    pub output: usize,
}

impl Linear {
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, input: usize, output: usize) -> Self {
        debug_assert_eq!(weight.len(), input * output);
        debug_assert!(bias.as_ref().is_none_or(|b| b.len() == output));
        Self {
            weight,
            bias,
            input,
            output,
        }
    }

    /// Plain f32 checkpoint: `{name}.weight [output, input]` and, when
    /// `has_bias`, `{name}.bias [output]`.
    pub fn load(
        file: &SafetensorsFile,
        name: &str,
        input: usize,
        output: usize,
        has_bias: bool,
    ) -> Result<Self> {
        let weight = load_tensor(file, &format!("{name}.weight"), &[output, input])?;
        let bias = if has_bias {
            Some(load_tensor(file, &format!("{name}.bias"), &[output])?)
        } else {
            None
        };
        Ok(Self::new(weight, bias, input, output))
    }

    /// `x [rows, input] -> [rows, output]`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        ops::linear(
            x,
            &self.weight,
            self.bias.as_deref(),
            rows,
            self.input,
            self.output,
        )
    }
}

/// Row-wise LayerNorm with a learned scale and optional shift.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub epsilon: f32,
}

impl LayerNorm {
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, epsilon: f32) -> Self {
        debug_assert!(bias.as_ref().is_none_or(|b| b.len() == weight.len()));
        Self {
            weight,
            bias,
            epsilon,
        }
    }

    /// Plain f32 checkpoint: `{name}.weight` and `{name}.bias`, both
    /// `[width]`.
    pub fn load(file: &SafetensorsFile, name: &str, width: usize, epsilon: f32) -> Result<Self> {
        Ok(Self::new(
            load_tensor(file, &format!("{name}.weight"), &[width])?,
            Some(load_tensor(file, &format!("{name}.bias"), &[width])?),
            epsilon,
        ))
    }

    pub fn width(&self) -> usize {
        self.weight.len()
    }

    /// Normalizes each of `rows` rows of `x` in place.
    pub fn apply(&self, x: &mut [f32], rows: usize) {
        ops::layernorm(
            x,
            rows,
            self.weight.len(),
            &self.weight,
            self.bias.as_deref(),
            self.epsilon,
        );
    }
}

/// Row-wise RMSNorm with a learned scale.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub epsilon: f32,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, epsilon: f32) -> Self {
        Self { weight, epsilon }
    }

    pub fn width(&self) -> usize {
        self.weight.len()
    }

    /// Normalizes each of `rows` rows of `x` in place.
    pub fn apply(&self, x: &mut [f32], rows: usize) {
        ops::rmsnorm(x, rows, self.weight.len(), &self.weight, self.epsilon);
    }
}

/// Opens the checkpoint shards: the `model.safetensors.index.json`
/// `weight_map` order when present, otherwise the single model file.
pub fn open_shards(dir: &Path) -> Result<Vec<SafetensorsFile>> {
    let index_path = dir.join("model.safetensors.index.json");
    let mut names: Vec<String> = Vec::new();
    if index_path.is_file() {
        let value: Value = serde_json::from_slice(
            &std::fs::read(&index_path)
                .map_err(|error| bad_config("model.safetensors.index.json", error))?,
        )
        .map_err(|error| bad_config("model.safetensors.index.json", error))?;
        let map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| bad_config("model.safetensors.index.json", "weight_map missing"))?;
        for shard in map.values().filter_map(Value::as_str) {
            if !names.iter().any(|name| name == shard) {
                names.push(shard.to_owned());
            }
        }
        names.sort();
    } else {
        names.push("model.safetensors".to_owned());
    }
    if names.is_empty() {
        return Err(SpeechError::Tensor {
            name: "model.safetensors".to_owned(),
            why: "no shards listed in model.safetensors.index.json".to_owned(),
        });
    }
    names
        .iter()
        .map(|name| {
            SafetensorsFile::open(&dir.join(name)).map_err(|error| SpeechError::BadConfig {
                field: name.clone(),
                why: error.to_string(),
            })
        })
        .collect()
}

/// Index of the first maximum of `values`.
///
/// Starts from negative infinity with a strict `>`, so ties keep the first
/// index and NaN entries are skipped. The NeMo transducer families use a
/// different scan (seeded from element 0) and keep their own.
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0
}

/// A `BadConfig` error naming `field`.
pub fn bad_config(field: &str, why: impl std::fmt::Display) -> SpeechError {
    SpeechError::BadConfig {
        field: field.to_owned(),
        why: why.to_string(),
    }
}

/// A `Tensor` error naming `name`.
pub fn tensor_error(name: &str, why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_owned(),
        why: why.into(),
    }
}

/// Symmetric Hann window (`hanning(size, periodic=False)`), the NeMo
/// frontends' analysis window. The crate-level `dsp::hann_window` is the
/// periodic variant and must not be substituted.
pub fn symmetric_hann(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0; size];
    }
    (0..size)
        .map(|i| {
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())) as f32
        })
        .collect()
}

/// Required-key readers for the NeMo-derived `config.json` files.
pub mod json {
    use serde_json::Value;

    use crate::{Result, SpeechError};

    fn bad(field: &str, why: &str) -> SpeechError {
        SpeechError::BadConfig {
            field: field.to_owned(),
            why: why.to_owned(),
        }
    }

    pub fn required<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
        v.get(key)
            .ok_or_else(|| bad(key, "missing from config.json"))
    }

    pub fn usize_field(v: &Value, key: &str) -> Result<usize> {
        required(v, key)?
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| bad(key, "must be a positive integer fitting usize"))
    }

    pub fn bool_field(v: &Value, key: &str) -> Result<bool> {
        required(v, key)?
            .as_bool()
            .ok_or_else(|| bad(key, "must be a boolean"))
    }

    pub fn text_field(v: &Value, key: &str) -> Result<String> {
        required(v, key)?
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| bad(key, "must be a string"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Verbatim copy of the per-family `argmax` this module replaced.
    fn argmax_reference(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 {
                    (index, value)
                } else {
                    best
                }
            })
            .0
    }

    /// Verbatim copy of the sensevoice loop form, which must agree with the
    /// fold form on every input.
    fn argmax_loop_reference(row: &[f32]) -> usize {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in row.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        best_index
    }

    #[test]
    fn argmax_matches_the_retired_copies_on_edge_inputs() {
        let cases: Vec<Vec<f32>> = vec![
            vec![],
            vec![1.0],
            vec![1.0, 3.0, 3.0, 2.0],
            vec![f32::NAN, 1.0, 0.5],
            vec![1.0, f32::NAN, 2.0],
            vec![f32::NEG_INFINITY; 4],
            vec![f32::NAN; 3],
            vec![-0.0, 0.0, -0.0],
            vec![f32::INFINITY, 1.0, f32::INFINITY],
        ];
        for case in &cases {
            assert_eq!(argmax(case), argmax_reference(case), "{case:?}");
            assert_eq!(argmax(case), argmax_loop_reference(case), "{case:?}");
        }
        let mut state = 0x1234_5678u32;
        for len in 1..64 {
            let values: Vec<f32> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    // Coarse grid so ties actually occur.
                    ((state >> 24) % 7) as f32 - 3.0
                })
                .collect();
            assert_eq!(argmax(&values), argmax_reference(&values));
        }
    }

    #[test]
    fn symmetric_hann_matches_the_retired_copies_bitwise() {
        // Verbatim nemo_mel form (match arms) and the parakeet/canary form.
        fn match_form(size: usize) -> Vec<f32> {
            match size {
                0 => Vec::new(),
                1 => vec![1.0],
                _ => (0..size)
                    .map(|i| {
                        (0.5 * (1.0
                            - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos()))
                            as f32
                    })
                    .collect(),
            }
        }
        for size in 0..600 {
            let got: Vec<u32> = symmetric_hann(size).iter().map(|v| v.to_bits()).collect();
            let want: Vec<u32> = match_form(size).iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "size {size}");
        }
        assert_eq!(symmetric_hann(400).len(), 400);
    }

    #[test]
    fn error_helpers_render_the_retired_messages() {
        assert_eq!(
            bad_config("field", "why").to_string(),
            SpeechError::BadConfig {
                field: "field".into(),
                why: "why".into()
            }
            .to_string()
        );
        assert_eq!(
            bad_config("field", String::from("owned")).to_string(),
            bad_config("field", "owned").to_string()
        );
        assert_eq!(
            tensor_error("t", "bad").to_string(),
            SpeechError::Tensor {
                name: "t".into(),
                why: "bad".into()
            }
            .to_string()
        );
    }

    #[test]
    fn open_shards_follows_the_index_or_the_single_file() {
        let dir = std::env::temp_dir().join(format!("turbospark_nn_shards_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Neither an index nor model.safetensors: the single-file default
        // fails naming that file.
        let missing = open_shards(&dir).err().unwrap();
        assert!(
            missing.to_string().contains("model.safetensors"),
            "{missing}"
        );

        let a = temp_safetensors("shard_a", &[("a", vec![1], vec![1.0])]);
        let b = temp_safetensors("shard_b", &[("b", vec![1], vec![2.0])]);
        std::fs::copy(&a, dir.join("b.safetensors")).unwrap();
        std::fs::copy(&b, dir.join("a.safetensors")).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {"x": "b.safetensors", "y": "a.safetensors", "z": "b.safetensors"}}"#,
        )
        .unwrap();
        let shards = open_shards(&dir).unwrap();
        // Deduplicated and sorted by file name, not by weight_map order.
        assert_eq!(shards.len(), 2);
        assert!(shards[0].contains_tensor("b"));
        assert!(shards[1].contains_tensor("a"));

        std::fs::write(dir.join("model.safetensors.index.json"), r#"{}"#).unwrap();
        let bad = open_shards(&dir).err().unwrap();
        assert!(bad.to_string().contains("weight_map missing"), "{bad}");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .unwrap();
        let empty = open_shards(&dir).err().unwrap();
        assert!(empty.to_string().contains("no shards listed"), "{empty}");

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(a);
        let _ = std::fs::remove_file(b);
    }

    #[test]
    fn json_fields_report_the_retired_messages() {
        use serde_json::json;
        let v = json!({"n": 3, "zero": 0, "flag": true, "s": "x"});
        assert_eq!(json::usize_field(&v, "n").unwrap(), 3);
        assert!(json::bool_field(&v, "flag").unwrap());
        assert_eq!(json::text_field(&v, "s").unwrap(), "x");
        for (result, why) in [
            (
                json::usize_field(&v, "zero").map(|_| ()),
                "must be a positive integer fitting usize",
            ),
            (json::bool_field(&v, "s").map(|_| ()), "must be a boolean"),
            (json::text_field(&v, "n").map(|_| ()), "must be a string"),
            (
                json::usize_field(&v, "absent").map(|_| ()),
                "missing from config.json",
            ),
        ] {
            let message = result.unwrap_err().to_string();
            assert!(message.contains(why), "{message}");
        }
    }

    fn temp_safetensors(tag: &str, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let header = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!(
            "turbospark_nn_{tag}_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, out).unwrap();
        path
    }

    fn open(path: &Path) -> SafetensorsFile {
        SafetensorsFile::open(path).unwrap()
    }

    #[test]
    fn linear_forward_is_the_ops_kernel() {
        let weight = vec![1.0, 0.0, 0.5, 2.0];
        let bias = Some(vec![0.5, -0.5]);
        let layer = Linear::new(weight.clone(), bias.clone(), 2, 2);
        let x = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(
            layer.forward(&x, 2),
            ops::linear(&x, &weight, bias.as_deref(), 2, 2, 2)
        );
        let bare = Linear::new(weight.clone(), None, 2, 2);
        assert_eq!(bare.forward(&x, 2), ops::linear(&x, &weight, None, 2, 2, 2));
    }

    #[test]
    fn layer_and_rms_norm_are_the_ops_kernels() {
        let w = vec![1.0f32, 2.0, 0.5];
        let b = vec![0.1f32, -0.2, 0.3];
        let mut got = vec![1.0f32, 2.0, 4.0, -1.0, 0.0, 3.0];
        let mut want = got.clone();
        LayerNorm::new(w.clone(), Some(b.clone()), 1e-5).apply(&mut got, 2);
        ops::layernorm(&mut want, 2, 3, &w, Some(&b), 1e-5);
        assert_eq!(got, want);

        let mut got = vec![1.0f32, 2.0, 4.0, -1.0, 0.0, 3.0];
        let mut want = got.clone();
        RmsNorm::new(w.clone(), 1e-6).apply(&mut got, 2);
        ops::rmsnorm(&mut want, 2, 3, &w, 1e-6);
        assert_eq!(got, want);
    }

    #[test]
    fn load_reads_weights_and_reports_missing_and_misshaped_tensors() {
        let path = temp_safetensors(
            "load",
            &[
                ("fc.weight", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ("fc.bias", vec![2], vec![0.5, -0.5]),
                ("ln.weight", vec![3], vec![1.0, 1.0, 1.0]),
                ("ln.bias", vec![3], vec![0.0, 0.0, 0.0]),
            ],
        );
        let file = open(&path);
        let fc = Linear::load(&file, "fc", 3, 2, true).unwrap();
        assert_eq!(fc.weight, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(fc.bias, Some(vec![0.5, -0.5]));
        assert!(Linear::load(&file, "fc", 3, 2, false)
            .unwrap()
            .bias
            .is_none());
        let ln = LayerNorm::load(&file, "ln", 3, 1e-5).unwrap();
        assert_eq!(ln.width(), 3);

        let missing = Linear::load(&file, "nope", 3, 2, true).unwrap_err();
        assert!(
            missing.to_string().contains("tensor is missing"),
            "{missing}"
        );
        let wrong = Linear::load(&file, "fc", 2, 3, true).unwrap_err();
        assert!(
            wrong.to_string().contains("expected shape [3, 2]"),
            "{wrong}"
        );
        let _ = std::fs::remove_file(path);
    }
}
