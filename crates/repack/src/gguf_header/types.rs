//! Data types and error definitions for parsing GGUF v3 file headers.

use std::collections::BTreeMap;

/// GGUF magic, little-endian "GGUF".
pub const MAGIC: u32 = 0x4655_4747;

/// Metadata cap, matching the safetensors reader's reasoning: reject a
/// corrupt or hostile count before allocating for it. Sized for a real
/// header, which is dominated by `tokenizer.ggml.tokens` (262144 strings on
/// Gemma 4) and runs to a few MB.
pub const DEFAULT_MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Alignment used when `general.alignment` is absent, per the spec.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// The only version this parser accepts. v1 and v2 differ in the width of
/// the count fields, so they are rejected by name rather than misparsed.
pub const SUPPORTED_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufHeaderError {
    /// The buffer ends before the header does. `needed` is the absolute
    /// offset the parse tried to reach, so a ranged fetcher can request
    /// exactly that much and retry.
    TooShort {
        needed: u64,
    },
    BadMagic {
        found: u32,
    },
    UnsupportedVersion {
        version: u32,
    },
    HeaderTooLarge {
        needed: u64,
        max_bytes: u64,
    },
    /// A metadata value carried a type id outside the 0..=12 enum.
    BadValueType {
        id: u32,
    },
    InvalidUtf8 {
        context: String,
    },
    DuplicateTensor {
        name: String,
    },
    DuplicateKey {
        name: String,
    },
    /// A tensor's element type is a real ggml type this port has no byte
    /// size for. Named rather than numbered: adding one is a single row in
    /// [`crate::gguf_header::ggml::ggml_type_name`]/[`crate::gguf_header::ggml::ggml_type_block`].
    UnsupportedType {
        name: String,
        id: u32,
    },
    UnknownType {
        id: u32,
    },
    /// `general.alignment` must be a non-zero power of two.
    BadAlignment {
        alignment: u64,
    },
    /// A tensor's element count is not a whole number of quantization
    /// blocks, so its byte size is not representable.
    RaggedTensor {
        name: String,
        elements: u64,
        block: u64,
    },
    /// More dimensions than ggml supports, or none at all.
    BadDimensions {
        name: String,
        n_dims: u32,
    },
    /// A tensor's element count, byte size, or absolute offset overflowed
    /// `u64` arithmetic. Header-supplied dims and offsets are otherwise
    /// unchecked, so a hostile or corrupt file can name a product or a sum
    /// that does not fit rather than one this port can address.
    Overflow {
        name: String,
    },
    /// A tensor's offset is not a multiple of the header's alignment, which
    /// the GGUF spec requires.
    MisalignedTensor {
        name: String,
        offset: u64,
        alignment: u64,
    },
}

impl std::fmt::Display for GgufHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufHeaderError::TooShort { needed } => {
                write!(f, "buffer ends before header offset {needed}")
            }
            GgufHeaderError::BadMagic { found } => {
                write!(f, "not a GGUF file: magic {found:#010x}, expected {MAGIC:#010x}")
            }
            GgufHeaderError::UnsupportedVersion { version } => write!(
                f,
                "GGUF version {version} is not supported (this parser reads v{SUPPORTED_VERSION})"
            ),
            GgufHeaderError::HeaderTooLarge { needed, max_bytes } => {
                write!(f, "header needs {needed} bytes, exceeding the {max_bytes}-byte cap")
            }
            GgufHeaderError::BadValueType { id } => {
                write!(f, "metadata value type {id} is outside the GGUF enum")
            }
            GgufHeaderError::InvalidUtf8 { context } => {
                write!(f, "{context} is not valid UTF-8")
            }
            GgufHeaderError::DuplicateTensor { name } => {
                write!(f, "duplicate tensor name {name}")
            }
            GgufHeaderError::DuplicateKey { name } => {
                write!(f, "duplicate metadata key {name}")
            }
            GgufHeaderError::UnsupportedType { name, id } => write!(
                f,
                "ggml type {name} (id {id}) has no byte size in this port; add it to ggml_type_block"
            ),
            GgufHeaderError::UnknownType { id } => {
                write!(f, "ggml type id {id} is not a known type")
            }
            GgufHeaderError::BadAlignment { alignment } => write!(
                f,
                "general.alignment {alignment} is not a non-zero power of two"
            ),
            GgufHeaderError::RaggedTensor {
                name,
                elements,
                block,
            } => write!(
                f,
                "tensor {name}: {elements} elements is not a whole number of {block}-element blocks"
            ),
            GgufHeaderError::BadDimensions { name, n_dims } => {
                write!(f, "tensor {name}: {n_dims} dimensions is out of range")
            }
            GgufHeaderError::Overflow { name } => {
                write!(f, "tensor {name}: size or offset arithmetic overflowed u64")
            }
            GgufHeaderError::MisalignedTensor {
                name,
                offset,
                alignment,
            } => write!(
                f,
                "tensor {name}: offset {offset} is not a multiple of the {alignment}-byte alignment"
            ),
        }
    }
}

impl std::error::Error for GgufHeaderError {}

/// One metadata value. The variants mirror the spec's type enum exactly;
/// the accessors below are what callers actually use.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Any unsigned-representable integer widened to `u64`. Signed variants
    /// yield `None` when negative rather than wrapping: a negative
    /// `block_count` should be an error, not a huge positive one.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(u64::from(*v)),
            GgufValue::U16(v) => Some(u64::from(*v)),
            GgufValue::U32(v) => Some(u64::from(*v)),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) => u64::try_from(*v).ok(),
            GgufValue::I16(v) => u64::try_from(*v).ok(),
            GgufValue::I32(v) => u64::try_from(*v).ok(),
            GgufValue::I64(v) => u64::try_from(*v).ok(),
            GgufValue::Bool(_)
            | GgufValue::F32(_)
            | GgufValue::F64(_)
            | GgufValue::String(_)
            | GgufValue::Array(_) => None,
        }
    }

    /// Any float widened to `f64`. Integers are deliberately NOT accepted:
    /// a caller reading `rope.freq_base` wants to know if the file stored an
    /// int where a float belongs, because `arch_validation` compares these
    /// with `!=` on `f64` (AGENTS.md Gotcha 24).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(f64::from(*v)),
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns string slice if value is String variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns bool value if value is Bool variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns array slice if value is Array variant.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// Metadata information for one tensor in a GGUF header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    /// Type id of ggml tensor element type.
    pub ggml_type: u32,
    /// Dimensions AS STORED, which is ggml's fastest-varying-first order:
    /// a logical `[out_features, in_features]` matrix is stored here as
    /// `[in_features, out_features]`. The parser keeps the file's order and
    /// leaves the reversal to callers that need logical shape.
    pub dims: Vec<u64>,
    /// Offset relative to the data region start, not to the file start.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Calculates total element count across dimensions.
    pub fn element_count(&self) -> u64 {
        self.dims.iter().copied().product()
    }

    /// Packed byte size, from the element count and the type's block shape.
    ///
    /// Every multiplication is checked: `dims` and `offset` come off an
    /// untrusted header, and an unchecked product wraps in release rather
    /// than reporting a size this port cannot address.
    pub fn byte_size(&self, name: &str) -> Result<u64, GgufHeaderError> {
        let (block_elems, block_bytes) =
            super::ggml::ggml_type_block(self.ggml_type).ok_or_else(|| {
                match super::ggml::ggml_type_name(self.ggml_type) {
                    Some(n) => GgufHeaderError::UnsupportedType {
                        name: n.to_string(),
                        id: self.ggml_type,
                    },
                    None => GgufHeaderError::UnknownType { id: self.ggml_type },
                }
            })?;
        let elements = self
            .dims
            .iter()
            .copied()
            .try_fold(1u64, |acc, d| acc.checked_mul(d))
            .ok_or_else(|| GgufHeaderError::Overflow {
                name: name.to_string(),
            })?;
        if elements % block_elems != 0 {
            return Err(GgufHeaderError::RaggedTensor {
                name: name.to_string(),
                elements,
                block: block_elems,
            });
        }
        (elements / block_elems)
            .checked_mul(block_bytes)
            .ok_or_else(|| GgufHeaderError::Overflow {
                name: name.to_string(),
            })
    }
}

/// Parsed GGUF header metadata and tensor table structure.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufHeader {
    /// GGUF file format version number.
    pub version: u32,
    /// Key-value metadata table.
    pub metadata: BTreeMap<String, GgufValue>,
    /// Tensor metadata table mapping tensor name to layout information.
    pub tensors: BTreeMap<String, GgufTensorInfo>,
    /// Resolved `general.alignment`, or [`DEFAULT_ALIGNMENT`].
    pub alignment: u64,
    /// Absolute file offset where the tensor data region begins: the end of
    /// the tensor table rounded up to `alignment`.
    pub data_region_start: u64,
}

impl GgufHeader {
    /// The absolute file byte range `[start, end)` for `name`'s tensor data,
    /// or `None` if `name` is not present.
    pub fn absolute_range(&self, name: &str) -> Option<Result<(u64, u64), GgufHeaderError>> {
        let info = self.tensors.get(name)?;
        Some(info.byte_size(name).and_then(|size| {
            let overflow = || GgufHeaderError::Overflow {
                name: name.to_string(),
            };
            let start = self
                .data_region_start
                .checked_add(info.offset)
                .ok_or_else(overflow)?;
            let end = start.checked_add(size).ok_or_else(overflow)?;
            Ok((start, end))
        }))
    }

    /// Looks up string metadata value for `key`.
    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    /// Looks up u64 metadata value for `key`.
    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(GgufValue::as_u64)
    }

    /// Looks up f64 metadata value for `key`.
    pub fn metadata_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(GgufValue::as_f64)
    }

    /// `general.architecture`, which every other architecture key is
    /// prefixed with (`gemma4.block_count`, and so on).
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_str("general.architecture")
    }
}
