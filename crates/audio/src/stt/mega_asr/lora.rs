//! Mega-ASR LoRA adapter loading and delta materialization.
//!
//! Ported from `mlx_audio/stt/models/mega_asr/convert_lora.py` and
//! `mlx_audio/stt/models/mega_asr/lora.py` at the pinned mlx-audio 0.5.7
//! commit (`e1b19b9054bf163f5d812221a54fcc346f1890e9`). Two adapter formats
//! exist upstream and both are supported:
//!
//! - Factors format (`extras/lora.safetensors`, produced by the family's
//!   `convert_router`/`convert_lora` flow): tensors named `<module>.lora_A`
//!   and `<module>.lora_B`; the reference sets the scaling to exactly 1.0
//!   because the conversion already folded alpha/rank into the factors.
//! - Adapter format (mlx-audio LoRA training output): a directory with
//!   `adapter_config.json` and `adapter_model.safetensors`, tensors named
//!   `base_model.model.thinker.<module>.lora_A.weight` (and `.lora_B`);
//!   scaling is `alpha / rank` with the reference's `rank_pattern` and
//!   `alpha_pattern` lookups.
//!
//! The delta for a module is `scaling * (B @ A)`, a dense `[output, input]`
//! matrix the reference adds to the base linear weight (`apply_deltas`). The
//! pinned always-on-robust checkpoint carries the deltas already merged into
//! the quantized weights, so `MegaAsr::load` holds a loaded adapter only when
//! a dynamic-profile distribution ships `lora_weights`.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::{Result, SpeechError};

const THINKER_PREFIX: &str = "base_model.model.thinker.";
const LORA_A_WEIGHT_SUFFIX: &str = ".lora_A.weight";
const LORA_B_WEIGHT_SUFFIX: &str = ".lora_B.weight";
const LORA_A_FACTOR_SUFFIX: &str = ".lora_A";
const LORA_B_FACTOR_SUFFIX: &str = ".lora_B";

/// One LoRA module: `delta = scaling * (B @ A)` with shape
/// `[output, input] = B[output, rank] @ A[rank, input]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraModule {
    pub rank: usize,
    pub output: usize,
    pub input: usize,
    pub scaling: f32,
    /// `[rank, input]` row-major.
    pub a: Vec<f32>,
    /// `[output, rank]` row-major.
    pub b: Vec<f32>,
}

impl LoraModule {
    /// The reference `materialize_delta`: `scaling * (B @ A)` in f32.
    pub fn materialize_delta(&self) -> Vec<f32> {
        crate::ops::matmul(&self.b, &self.a, self.output, self.rank, self.input)
            .into_iter()
            .map(|value| self.scaling * value)
            .collect()
    }
}

/// The complete adapter table keyed by module path in Qwen3-ASR model
/// coordinates (for example `model.layers.5.self_attn.q_proj` or
/// `audio_tower.layers.0.fc1`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoraAdapter {
    modules: BTreeMap<String, LoraModule>,
}

fn bad_tensor(name: impl Into<String>, why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: name.into(),
        why: why.into(),
    }
}

/// Factor tensors collected by suffix: `(rows, cols, values)` per module.
type FactorTable = BTreeMap<String, (usize, usize, Vec<f32>)>;

fn collect_factors(
    file: &SafetensorsFile,
    a_suffix: &str,
    b_suffix: &str,
) -> Result<(FactorTable, FactorTable)> {
    let mut a_tensors: FactorTable = BTreeMap::new();
    let mut b_tensors: FactorTable = BTreeMap::new();
    for name in file.tensor_names() {
        let (module, side) = if let Some(base) = name.strip_suffix(a_suffix) {
            (base, 0usize)
        } else if let Some(base) = name.strip_suffix(b_suffix) {
            (base, 1usize)
        } else {
            continue;
        };
        let descriptor = file
            .descriptor(name)
            .ok_or_else(|| bad_tensor(name, "descriptor vanished while reading LoRA factors"))?;
        if descriptor.shape.len() != 2 {
            return Err(bad_tensor(
                name,
                format!("expected a rank-2 factor, got {:?}", descriptor.shape),
            ));
        }
        let rows = descriptor.shape[0];
        let columns = descriptor.shape[1];
        if rows == 0 || columns == 0 {
            return Err(bad_tensor(
                name,
                format!("factor carries an empty dimension {:?}", descriptor.shape),
            ));
        }
        let values = file.load_as_f32(name)?;
        let table = if side == 0 {
            &mut a_tensors
        } else {
            &mut b_tensors
        };
        table.insert(module.to_owned(), (rows, columns, values));
    }
    Ok((a_tensors, b_tensors))
}

impl LoraAdapter {
    /// The reference `load_lora_factors`: factors-format tensors with the
    /// scaling pinned to 1.0.
    pub fn load_factors(path: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        let (a_tensors, b_tensors) =
            collect_factors(&file, LORA_A_FACTOR_SUFFIX, LORA_B_FACTOR_SUFFIX)?;
        let mut modules = BTreeMap::new();
        for (module, (rank, input, a)) in a_tensors {
            let (output, b_rank, b) = b_tensors
                .get(&module)
                .ok_or_else(|| {
                    bad_tensor(
                        format!("{module}{LORA_B_FACTOR_SUFFIX}"),
                        "LoRA factors reference a B tensor that does not exist",
                    )
                })?
                .clone();
            if b_rank != rank {
                return Err(bad_tensor(
                    format!("{module}{LORA_B_FACTOR_SUFFIX}"),
                    format!("B rank {b_rank} disagrees with A rank {rank}"),
                ));
            }
            modules.insert(
                module,
                LoraModule {
                    rank,
                    output,
                    input,
                    scaling: 1.0,
                    a,
                    b,
                },
            );
        }
        if modules.is_empty() {
            return Err(SpeechError::Unsupported {
                why: format!("no <module>.lora_A factors found in {}", path.display()),
            });
        }
        Ok(Self { modules })
    }

    /// The reference `load_lora_adapter`: a LoRA training output directory
    /// with `adapter_config.json` and `adapter_model.safetensors`, including
    /// the `base_model.model.thinker.` prefix and rank/alpha patterns.
    pub fn load_adapter(directory: &Path) -> Result<Self> {
        let config_path = directory.join("adapter_config.json");
        let config: Value =
            serde_json::from_slice(&std::fs::read(&config_path).map_err(|error| {
                SpeechError::BadConfig {
                    field: "adapter_config.json".into(),
                    why: format!("cannot read: {error}"),
                }
            })?)
            .map_err(|error| SpeechError::BadConfig {
                field: "adapter_config.json".into(),
                why: error.to_string(),
            })?;
        let global_rank = config
            .get("r")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| SpeechError::BadConfig {
                field: "adapter_config.json.r".into(),
                why: "must be a positive integer".into(),
            })?;
        let global_alpha = config
            .get("lora_alpha")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .unwrap_or(global_rank);
        let rank_pattern = pattern_table(config.get("rank_pattern"))?;
        let alpha_pattern = pattern_table(config.get("alpha_pattern"))?;

        let model_path = directory.join("adapter_model.safetensors");
        let file = SafetensorsFile::open(&model_path)?;
        let (a_tensors, b_tensors) =
            collect_factors(&file, LORA_A_WEIGHT_SUFFIX, LORA_B_WEIGHT_SUFFIX)?;

        let mut modules = BTreeMap::new();
        for (raw_module, (tensor_rank, input, a)) in a_tensors {
            let module = match raw_module.strip_prefix(THINKER_PREFIX) {
                Some(stripped) => stripped.to_owned(),
                None => raw_module.clone(),
            };
            // Scaling follows the config exactly like the reference; a rank
            // disagreement between the config, A, and B is corruption the
            // reference would silently scale wrong, so it is refused here.
            let rank = pattern_lookup(&module, &rank_pattern, global_rank);
            let alpha = pattern_lookup(&module, &alpha_pattern, global_alpha);
            if tensor_rank != rank {
                return Err(bad_tensor(
                    format!("{raw_module}{LORA_A_WEIGHT_SUFFIX}"),
                    format!("factor rank {tensor_rank} disagrees with the resolved rank {rank}"),
                ));
            }
            let (output, b_rank, b) = b_tensors
                .get(&raw_module)
                .ok_or_else(|| {
                    bad_tensor(
                        format!("{raw_module}{LORA_B_WEIGHT_SUFFIX}"),
                        "adapter references a B tensor that does not exist",
                    )
                })?
                .clone();
            if b_rank != rank {
                return Err(bad_tensor(
                    format!("{raw_module}{LORA_B_WEIGHT_SUFFIX}"),
                    format!("B rank {b_rank} disagrees with the resolved rank {rank}"),
                ));
            }
            modules.insert(
                module,
                LoraModule {
                    rank,
                    output,
                    input,
                    scaling: alpha as f32 / rank as f32,
                    a,
                    b,
                },
            );
        }
        if modules.is_empty() {
            return Err(SpeechError::Unsupported {
                why: format!("no lora_A weights found in {}", model_path.display()),
            });
        }
        Ok(Self { modules })
    }

    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    pub fn module(&self, name: &str) -> Option<&LoraModule> {
        self.modules.get(name)
    }

    pub fn module_names(&self) -> impl Iterator<Item = &String> {
        self.modules.keys()
    }

    /// Adds the module's delta into a dense `[output, input]` f32 weight, the
    /// runtime counterpart of the reference `_accumulate` with sign +1.
    /// The pinned always-on-robust profile never reaches this at load time
    /// because its deltas ship pre-merged and quantized.
    pub fn merge_into(&self, name: &str, weight: &mut [f32]) -> Result<()> {
        let module = self
            .modules
            .get(name)
            .ok_or_else(|| bad_tensor(name, "no LoRA factors exist for this module"))?;
        let delta = module.materialize_delta();
        if weight.len() != delta.len() {
            return Err(bad_tensor(
                name,
                format!(
                    "delta has {} values but the target weight has {}",
                    delta.len(),
                    weight.len()
                ),
            ));
        }
        for (value, add) in weight.iter_mut().zip(delta) {
            *value += add;
        }
        Ok(())
    }
}

fn pattern_table(value: Option<&Value>) -> Result<BTreeMap<String, usize>> {
    let mut table = BTreeMap::new();
    if let Some(map) = value.and_then(Value::as_object) {
        for (key, entry) in map {
            let rank = entry
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| SpeechError::BadConfig {
                    field: "adapter_config.json pattern".into(),
                    why: format!("pattern {key} must map to a positive integer"),
                })?;
            table.insert(key.clone(), rank);
        }
    }
    Ok(table)
}

/// The reference `_pattern_lookup`: the module or its `thinker.`-prefixed
/// form matches exactly, then a suffix match, then the default.
fn pattern_lookup(module: &str, pattern: &BTreeMap<String, usize>, default: usize) -> usize {
    for candidate in [module, &format!("thinker.{module}")] {
        if let Some(value) = pattern.get(candidate) {
            return *value;
        }
    }
    for (key, value) in pattern {
        if module == key || module.ends_with(&format!(".{key}")) {
            return *value;
        }
    }
    default
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Minimal safetensors writer for loader tests; F32 tensors only.

    use std::collections::BTreeMap;

    pub fn write_safetensors(
        path: &std::path::Path,
        tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    ) {
        let mut header = serde_json::Map::new();
        let mut blob = 0usize;
        for (name, (shape, values)) in tensors {
            let bytes = values.len() * 4;
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [blob, blob + bytes],
                }),
            );
            blob += bytes;
        }
        let header_json = serde_json::Value::Object(header).to_string();
        let mut file = (header_json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header_json.as_bytes());
        for (_, values) in tensors.values() {
            for value in values {
                file.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(path, file).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn factor_name(module: &str, side: &str) -> String {
        format!("{module}{side}")
    }

    #[test]
    fn factors_loader_loads_modules_with_unit_scaling() {
        let dir = std::env::temp_dir().join(format!("mega-lora-factors-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lora.safetensors");
        let a: Vec<f32> = (0..2 * 5).map(|i| (i as f32) * 0.25 - 1.0).collect();
        let b: Vec<f32> = (0..3 * 2).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let mut tensors = BTreeMap::new();
        tensors.insert(
            factor_name("model.layers.0.self_attn.q_proj", LORA_A_FACTOR_SUFFIX),
            (vec![2, 5], a.clone()),
        );
        tensors.insert(
            factor_name("model.layers.0.self_attn.q_proj", LORA_B_FACTOR_SUFFIX),
            (vec![3, 2], b.clone()),
        );
        test_support::write_safetensors(&path, &tensors);

        let adapter = LoraAdapter::load_factors(&path).unwrap();
        assert_eq!(adapter.module_count(), 1);
        let module = adapter.module("model.layers.0.self_attn.q_proj").unwrap();
        assert_eq!(module.rank, 2);
        assert_eq!(module.input, 5);
        assert_eq!(module.output, 3);
        assert_eq!(module.scaling, 1.0);
        assert_eq!(module.a, a);
        assert_eq!(module.b, b);

        // delta = 1.0 * (B @ A), spot check one entry.
        let delta = module.materialize_delta();
        let expected = b[0] * a[0] + b[1] * a[5];
        assert!((delta[0] - expected).abs() < 1e-6);

        // merge_into adds the delta into a matching weight.
        let mut weight = vec![0.0f32; 3 * 5];
        adapter
            .merge_into("model.layers.0.self_attn.q_proj", &mut weight)
            .unwrap();
        assert!((weight[0] - expected).abs() < 1e-6);
        // A wrong-length weight is refused.
        assert!(adapter
            .merge_into("model.layers.0.self_attn.q_proj", &mut [0.0; 3])
            .is_err());
        // An unknown module is refused.
        assert!(adapter
            .merge_into("model.layers.1.mlp.down_proj", &mut weight)
            .is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn adapter_loader_resolves_thinker_prefix_and_patterns() {
        let dir = std::env::temp_dir().join(format!("mega-lora-adapter-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a0 = vec![0.5f32; 4 * 6];
        let b0 = vec![1.0f32; 6 * 4];
        let a1 = vec![0.25f32; 2 * 6];
        let b1 = vec![2.0f32; 6 * 2];
        let mut tensors = BTreeMap::new();
        tensors.insert(
            format!("{THINKER_PREFIX}model.layers.0.self_attn.q_proj{LORA_A_WEIGHT_SUFFIX}"),
            (vec![4, 6], a0),
        );
        tensors.insert(
            format!("{THINKER_PREFIX}model.layers.0.self_attn.q_proj{LORA_B_WEIGHT_SUFFIX}"),
            (vec![6, 4], b0),
        );
        tensors.insert(
            format!("{THINKER_PREFIX}model.layers.1.self_attn.q_proj{LORA_A_WEIGHT_SUFFIX}"),
            (vec![2, 6], a1),
        );
        tensors.insert(
            format!("{THINKER_PREFIX}model.layers.1.self_attn.q_proj{LORA_B_WEIGHT_SUFFIX}"),
            (vec![6, 2], b1),
        );
        test_support::write_safetensors(&dir.join("adapter_model.safetensors"), &tensors);
        std::fs::write(
            dir.join("adapter_config.json"),
            r#"{"r": 4, "lora_alpha": 8,
                "rank_pattern": {"model.layers.1.self_attn.q_proj": 2},
                "alpha_pattern": {"thinker.model.layers.1.self_attn.q_proj": 6}}"#,
        )
        .unwrap();

        let adapter = LoraAdapter::load_adapter(&dir).unwrap();
        assert_eq!(adapter.module_count(), 2);
        let plain = adapter.module("model.layers.0.self_attn.q_proj").unwrap();
        // scaling = lora_alpha / r = 8 / 4
        assert!((plain.scaling - 2.0).abs() < 1e-6);
        let patterned = adapter.module("model.layers.1.self_attn.q_proj").unwrap();
        // rank and alpha resolve through the patterns: 6 / 2
        assert!((patterned.scaling - 3.0).abs() < 1e-6);
        assert_eq!(patterned.rank, 2);

        // An adapter config without r is refused.
        std::fs::write(dir.join("adapter_config.json"), r#"{"lora_alpha": 8}"#).unwrap();
        assert!(LoraAdapter::load_adapter(&dir).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_b_factor_is_refused_with_the_module_named() {
        let dir = std::env::temp_dir().join(format!("mega-lora-broken-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lora.safetensors");
        let mut tensors = BTreeMap::new();
        tensors.insert(
            factor_name("model.layers.0.self_attn.q_proj", LORA_A_FACTOR_SUFFIX),
            (vec![2, 5], vec![0.0f32; 10]),
        );
        test_support::write_safetensors(&path, &tensors);
        let error = LoraAdapter::load_factors(&path).unwrap_err().to_string();
        assert!(error.contains("q_proj"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
