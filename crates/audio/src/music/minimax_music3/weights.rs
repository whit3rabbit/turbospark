//! Weight access shared by the converted-tree and official-tree loaders.
//!
//! Two backends feed one interface: safetensors files (converted MLX
//! trees, which may carry MLX groupwise-quantized linears) and an
//! in-memory map (the official modular tree after key sanitizing and
//! layout remap, which is never quantized). Both are strict: every
//! tensor in the source must be consumed by exactly one model field,
//! otherwise `finish` fails, mirroring `load_model(strict=True)`.

use std::collections::HashMap;
use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::quant::{self, QuantScheme};
use crate::Result;
use crate::SpeechError;

pub(crate) struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub(crate) enum WeightStore {
    File {
        files: Vec<SafetensorsFile>,
        scheme: QuantScheme,
        consumed: std::collections::HashSet<String>,
    },
    Map {
        tensors: HashMap<String, Tensor>,
    },
}

fn missing(name: &str) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why: "missing".to_string(),
    }
}

impl WeightStore {
    pub(crate) fn from_files(files: Vec<SafetensorsFile>, scheme: QuantScheme) -> WeightStore {
        WeightStore::File {
            files,
            scheme,
            consumed: std::collections::HashSet::new(),
        }
    }

    pub(crate) fn from_map(tensors: HashMap<String, Tensor>) -> WeightStore {
        WeightStore::Map { tensors }
    }

    pub(crate) fn has(&self, name: &str) -> bool {
        match self {
            WeightStore::File { files, .. } => files.iter().any(|f| f.contains_tensor(name)),
            WeightStore::Map { tensors } => tensors.contains_key(name),
        }
    }

    /// Take a plain (non-linear) tensor by exact name.
    pub(crate) fn tensor(&mut self, name: &str) -> Result<Tensor> {
        match self {
            WeightStore::File {
                files, consumed, ..
            } => {
                let file = files.iter().find(|f| f.contains_tensor(name));
                let file = file.ok_or_else(|| missing(name))?;
                let shape = file
                    .descriptor(name)
                    .map(|d| d.shape.clone())
                    .ok_or_else(|| missing(name))?;
                let data = file.load_as_f32(name)?;
                consumed.insert(name.to_string());
                Ok(Tensor { data, shape })
            }
            WeightStore::Map { tensors } => tensors.remove(name).ok_or_else(|| missing(name)),
        }
    }

    /// Load one linear as `(weight [out, in], bias)`; quantized
    /// linears dequantize through the shared affine kernel.
    pub(crate) fn linear(&mut self, base: &str) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        match self {
            WeightStore::File {
                files,
                scheme,
                consumed,
            } => {
                let weight_name = format!("{base}.weight");
                let file = files
                    .iter()
                    .find(|f| f.contains_tensor(&weight_name))
                    .ok_or_else(|| missing(&weight_name))?;
                let loaded = quant::load_quantized(file, base, *scheme)?;
                consumed.insert(weight_name);
                if file.contains_tensor(&format!("{base}.bias")) {
                    consumed.insert(format!("{base}.bias"));
                }
                if file.contains_tensor(&format!("{base}.scales")) {
                    consumed.insert(format!("{base}.scales"));
                    consumed.insert(format!("{base}.biases"));
                }
                Ok(loaded)
            }
            WeightStore::Map { tensors } => {
                if tensors.contains_key(&format!("{base}.scales")) {
                    return Err(SpeechError::Unsupported {
                        why: "quantized weights are not expected in an official tree".to_string(),
                    });
                }
                let weight = tensors
                    .remove(&format!("{base}.weight"))
                    .ok_or_else(|| missing(&format!("{base}.weight")))?;
                let bias = tensors
                    .remove(&format!("{base}.bias"))
                    .map(|tensor| tensor.data);
                Ok((weight.data, bias))
            }
        }
    }

    /// Fail when any source tensor was not consumed.
    pub(crate) fn finish(self) -> Result<()> {
        match self {
            WeightStore::File {
                files, consumed, ..
            } => {
                let extras: Vec<String> = files
                    .iter()
                    .flat_map(|file| file.tensor_names().map(str::to_string))
                    .filter(|name| !consumed.contains(name))
                    .collect();
                if extras.is_empty() {
                    Ok(())
                } else {
                    Err(SpeechError::Tensor {
                        name: extras[0].clone(),
                        why: format!("unconsumed tensors: {}", extras.join(", ")),
                    })
                }
            }
            WeightStore::Map { tensors } => {
                if tensors.is_empty() {
                    Ok(())
                } else {
                    let mut names: Vec<&String> = tensors.keys().collect();
                    names.sort();
                    Err(SpeechError::Tensor {
                        name: names[0].clone(),
                        why: format!(
                            "unconsumed tensors: {}",
                            names
                                .iter()
                                .map(|n| n.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    })
                }
            }
        }
    }
}

/// Open the safetensors shards of a converted tree: the index file's
/// `weight_map` order when present, otherwise the single model file.
pub(crate) fn open_converted_shards(dir: &Path) -> Result<Vec<SafetensorsFile>> {
    let mut names: Vec<String> = Vec::new();
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let value = crate::quant::read_json(&index)?;
        let empty = serde_json::Map::new();
        let map = value
            .get("weight_map")
            .and_then(|w| w.as_object())
            .unwrap_or(&empty);
        let mut shards: Vec<String> = Vec::new();
        for (tensor, shard) in map {
            let shard = shard.as_str().ok_or_else(|| SpeechError::BadConfig {
                field: format!("weight_map[{tensor}]"),
                why: "shard name is not a string".to_string(),
            })?;
            if !shards.iter().any(|s| s == shard) {
                shards.push(shard.to_string());
            }
        }
        shards.sort();
        names = shards;
    } else {
        names.push("model.safetensors".to_string());
    }
    if names.is_empty() {
        return Err(SpeechError::Tensor {
            name: "model.safetensors".to_string(),
            why: "no shards listed for the converted tree".to_string(),
        });
    }
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        let path = dir.join(&name);
        if !path.is_file() {
            return Err(SpeechError::Tensor {
                name,
                why: "shard file missing from the converted tree".to_string(),
            });
        }
        files.push(SafetensorsFile::open(&path)?);
    }
    Ok(files)
}
