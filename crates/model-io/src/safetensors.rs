//! Safetensors file reader for loading weights directly from local files.
//!
//! Provides memory-mapped access to tensors in `.safetensors` files without
//! copying data.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;

use crate::error::ModelError;

/// Tensor descriptor parsed from the safetensors JSON header.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TensorDescriptor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// The element count a `shape` implies, or `None` if the product overflows.
/// A hostile or corrupt header can declare any dimensions; this is what lets
/// a loader catch a shape that disagrees with the byte range it decodes
/// rather than silently returning a vector of the wrong length.
fn expected_element_count(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// A memory-mapped `.safetensors` file.
pub struct SafetensorsFile {
    mmap: Mmap,
    data_start: usize,
    tensors: BTreeMap<String, TensorDescriptor>,
}

impl SafetensorsFile {
    /// Open and memory-map a `.safetensors` file.
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let file = File::open(path).map_err(|e| ModelError::IoFailed {
            call: format!("open {}", path.display()),
            detail: e.to_string(),
        })?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| ModelError::IoFailed {
                call: format!("mmap {}", path.display()),
                detail: e.to_string(),
            })?
        };

        if mmap.len() < 8 {
            return Err(ModelError::IndexCorrupt {
                detail: "safetensors file too short for 8-byte header length".to_string(),
            });
        }

        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        let header_len = usize::try_from(header_len).map_err(|_| ModelError::IndexCorrupt {
            detail: format!("safetensors header length {header_len} overflows usize"),
        })?;
        let data_start =
            8usize
                .checked_add(header_len)
                .ok_or_else(|| ModelError::IndexCorrupt {
                    detail: format!("safetensors header length {header_len} overflows"),
                })?;
        if mmap.len() < data_start {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "safetensors file len {} shorter than declared header len {}",
                    mmap.len(),
                    data_start
                ),
            });
        }

        let header_bytes = &mmap[8..data_start];
        let raw_map: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header_bytes)
            .map_err(|e| ModelError::IndexCorrupt {
                detail: format!("failed to parse safetensors JSON header: {e}"),
            })?;

        let mut tensors = BTreeMap::new();
        for (k, v) in raw_map {
            if k == "__metadata__" {
                continue;
            }
            let desc: TensorDescriptor =
                serde_json::from_value(v).map_err(|e| ModelError::IndexCorrupt {
                    detail: format!("tensor descriptor {k}: {e}"),
                })?;
            tensors.insert(k, desc);
        }

        Ok(Self {
            mmap,
            data_start,
            tensors,
        })
    }

    /// Check if a tensor is present.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// List all tensor names in the file.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Get the descriptor for a tensor.
    pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    /// Get raw bytes for a tensor.
    pub fn raw_bytes(&self, name: &str) -> Result<&[u8], ModelError> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError::TensorNotFound {
                name: name.to_string(),
            })?;

        if desc.data_offsets.0 > desc.data_offsets.1 {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "tensor {name} has a backwards data_offsets range ({}..{})",
                    desc.data_offsets.0, desc.data_offsets.1
                ),
            });
        }
        let start = self
            .data_start
            .checked_add(desc.data_offsets.0)
            .ok_or_else(|| ModelError::IndexCorrupt {
                detail: format!("tensor {name} start offset overflows"),
            })?;
        let end = self
            .data_start
            .checked_add(desc.data_offsets.1)
            .ok_or_else(|| ModelError::IndexCorrupt {
                detail: format!("tensor {name} end offset overflows"),
            })?;
        if end > self.mmap.len() {
            return Err(ModelError::IndexCorrupt {
                detail: format!(
                    "tensor {name} offset ({start}..{end}) extends past file size {}",
                    self.mmap.len()
                ),
            });
        }

        Ok(&self.mmap[start..end])
    }

    /// Load tensor as FP32 vector, converting from BF16, F16, or F32.
    pub fn load_as_f32(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError::TensorNotFound {
                name: name.to_string(),
            })?;
        let bytes = self.raw_bytes(name)?;
        let expected_elements =
            expected_element_count(&desc.shape).ok_or_else(|| ModelError::IndexCorrupt {
                detail: format!(
                    "tensor {name}: shape {:?} overflows an element count",
                    desc.shape
                ),
            })?;
        let check_count = |actual: usize| -> Result<(), ModelError> {
            if actual != expected_elements {
                return Err(ModelError::IndexCorrupt {
                    detail: format!(
                        "tensor {name}: shape {:?} implies {expected_elements} elements, byte \
                         range holds {actual}",
                        desc.shape
                    ),
                });
            }
            Ok(())
        };

        match desc.dtype.to_uppercase().as_str() {
            "F32" => {
                if bytes.len() % 4 != 0 {
                    return Err(ModelError::IndexCorrupt {
                        detail: format!("F32 tensor {name} has unaligned length {}", bytes.len()),
                    });
                }
                check_count(bytes.len() / 4)?;
                let mut out = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                Ok(out)
            }
            "BF16" => {
                if bytes.len() % 2 != 0 {
                    return Err(ModelError::IndexCorrupt {
                        detail: format!("BF16 tensor {name} has unaligned length {}", bytes.len()),
                    });
                }
                check_count(bytes.len() / 2)?;
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let u = u16::from_le_bytes(chunk.try_into().unwrap());
                    out.push(f32::from_bits((u as u32) << 16));
                }
                Ok(out)
            }
            "F16" => {
                if bytes.len() % 2 != 0 {
                    return Err(ModelError::IndexCorrupt {
                        detail: format!("F16 tensor {name} has unaligned length {}", bytes.len()),
                    });
                }
                check_count(bytes.len() / 2)?;
                let mut out = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let u = u16::from_le_bytes(chunk.try_into().unwrap());
                    out.push(foundation::LogitValue::from_bits(u).to_f32());
                }
                Ok(out)
            }
            other => Err(ModelError::IndexCorrupt {
                detail: format!("unsupported dtype {other} for tensor {name}"),
            }),
        }
    }

    /// Load tensor as U16 vector (for BF16/F16 raw bits).
    pub fn load_as_u16(&self, name: &str) -> Result<Vec<u16>, ModelError> {
        let desc = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError::TensorNotFound {
                name: name.to_string(),
            })?;
        let bytes = self.raw_bytes(name)?;
        if bytes.len() % 2 != 0 {
            return Err(ModelError::IndexCorrupt {
                detail: format!("U16 tensor {name} has unaligned length {}", bytes.len()),
            });
        }
        if let Some(expected_elements) = expected_element_count(&desc.shape) {
            if bytes.len() / 2 != expected_elements {
                return Err(ModelError::IndexCorrupt {
                    detail: format!(
                        "tensor {name}: shape {:?} implies {expected_elements} elements, byte \
                         range holds {}",
                        desc.shape,
                        bytes.len() / 2
                    ),
                });
            }
        }
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            out.push(u16::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(out)
    }
}
