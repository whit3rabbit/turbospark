//! Parses the GGUF v3 file header: a fixed 24-byte preamble, a metadata
//! key/value section, and a tensor table describing each tensor's element
//! type, dimensions, and offset within the data region that follows. See
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md> for the spec.
//!
//! Sibling of [`crate::safetensors_header`] and used the same way: the
//! header alone (a few MB even for a 27 GB checkpoint, most of it the
//! tokenizer's token list) tells the repacker which byte ranges of the
//! remote file it needs, so the checkpoint is never materialized locally.
//!
//! Two differences from safetensors that shape this module:
//!
//! 1. **There is no length prefix.** A safetensors reader learns the header
//!    size from the first 8 bytes; a GGUF reader only learns it by walking
//!    the whole variable-length metadata section. So [`GgufHeaderError::TooShort`]
//!    carries the offset the walk wanted, and the ranged fetcher re-requests
//!    exactly that rather than doubling blindly.
//! 2. **The data region is aligned, not adjacent.** The tensor table ends
//!    wherever it ends and the data region starts at the next multiple of
//!    `general.alignment` (default 32). Every tensor offset is relative to
//!    that rounded start, so getting the rounding wrong shifts the entire
//!    file by up to 31 bytes and reads plausible-looking garbage.

use std::collections::BTreeMap;

/// GGUF magic, little-endian "GGUF".
const MAGIC: u32 = 0x4655_4747;

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
    /// [`ggml_type_name`]/[`ggml_type_block`].
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

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// Human-readable name for a ggml type id, for error messages. Ids and gaps
/// are verbatim from `ggml/include/ggml.h`'s `enum ggml_type` (4 and 5 were
/// removed types and stay unused).
pub fn ggml_type_name(id: u32) -> Option<&'static str> {
    Some(match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        _ => return None,
    })
}

/// `(elements per block, bytes per block)` for a ggml type.
///
/// DELIBERATELY PARTIAL. Only types whose block size was read off the ggml
/// spec or source are listed; everything else returns `None` and parses to
/// [`GgufHeaderError::UnsupportedType`], which names the type. A guessed
/// constant here would not fail loudly, it would silently misalign every
/// tensor after the first one of that type. Adding a type is one row plus
/// the source citation.
///
/// A row here buys PARSING, not execution: whether an install of that type
/// runs is decided separately by [`model_io::EXECUTABLE_GGUF_TYPES`] and by
/// `RealForwardRunner::open` (AGENTS.md Gotcha 29). Q2_K/Q3_K/Q5_K and the
/// three IQ types are listed so a mixed sub-4-bit file can be HEADER-PROBED
/// for scoping (ROADMAP Phase S); none of the six has a kernel. The IQ rows
/// in particular size a tensor without being able to READ one: IQ3_XXS and
/// IQ4_XS decode through codebooks, which is a different job from every
/// affine and K-quant unpacker in this port.
///
/// The K-quant and IQ rows were read out of ggml itself rather than off a
/// spec page, which is one command against brew's `llama.cpp` (b10310 here):
///
/// ```text
/// // cc -I/opt/homebrew/include x.c -L/opt/homebrew/lib -lggml -lggml-base
/// for (int i = 0; i < GGML_TYPE_COUNT; i++)
///     printf("%s %lld %lld\n", ggml_type_name(i),
///            (long long) ggml_blck_size(i), (long long) ggml_type_size(i));
/// ```
pub fn ggml_type_block(id: u32) -> Option<(u64, u64)> {
    Some(match id {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        18 => (256, 98),  // IQ3_XXS
        20 => (32, 18),   // IQ4_NL
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        30 => (1, 2),     // BF16
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
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
    pub fn element_count(&self) -> u64 {
        self.dims.iter().copied().product()
    }

    /// Packed byte size, from the element count and the type's block shape.
    pub fn byte_size(&self, name: &str) -> Result<u64, GgufHeaderError> {
        let (block_elems, block_bytes) = ggml_type_block(self.ggml_type).ok_or_else(|| {
            match ggml_type_name(self.ggml_type) {
                Some(n) => GgufHeaderError::UnsupportedType {
                    name: n.to_string(),
                    id: self.ggml_type,
                },
                None => GgufHeaderError::UnknownType { id: self.ggml_type },
            }
        })?;
        let elements = self.element_count();
        if elements % block_elems != 0 {
            return Err(GgufHeaderError::RaggedTensor {
                name: name.to_string(),
                elements,
                block: block_elems,
            });
        }
        Ok(elements / block_elems * block_bytes)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufHeader {
    pub version: u32,
    pub metadata: BTreeMap<String, GgufValue>,
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
        Some(info.byte_size(name).map(|size| {
            let start = self.data_region_start + info.offset;
            (start, start + size)
        }))
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(GgufValue::as_u64)
    }

    pub fn metadata_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(GgufValue::as_f64)
    }

    /// `general.architecture`, which every other architecture key is
    /// prefixed with (`gemma4.block_count`, and so on).
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_str("general.architecture")
    }
}

/// Bounds-checked forward reader. Every read that runs off the end reports
/// the absolute offset it wanted, which is what makes the ranged fetch a
/// single extra request instead of a doubling loop.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    max_bytes: u64,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufHeaderError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(GgufHeaderError::HeaderTooLarge {
                needed: u64::MAX,
                max_bytes: self.max_bytes,
            })?;
        if end as u64 > self.max_bytes {
            return Err(GgufHeaderError::HeaderTooLarge {
                needed: end as u64,
                max_bytes: self.max_bytes,
            });
        }
        if end > self.data.len() {
            return Err(GgufHeaderError::TooShort { needed: end as u64 });
        }
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, GgufHeaderError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, GgufHeaderError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, GgufHeaderError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufHeaderError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self, context: &str) -> Result<String, GgufHeaderError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| GgufHeaderError::HeaderTooLarge {
            needed: len,
            max_bytes: self.max_bytes,
        })?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufHeaderError::InvalidUtf8 {
            context: context.to_string(),
        })
    }

    fn value(&mut self, type_id: u32, context: &str) -> Result<GgufValue, GgufHeaderError> {
        Ok(match type_id {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.u8()? as i8),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.u16()? as i16),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_bits(self.u32()?)),
            // Spec says a single byte, 0 or 1. Anything non-zero is true;
            // rejecting 2 would fail on files llama.cpp itself reads.
            7 => GgufValue::Bool(self.u8()? != 0),
            8 => GgufValue::String(self.string(context)?),
            9 => {
                let elem_type = self.u32()?;
                let len = self.u64()?;
                // An array header is 12 bytes and its shortest element is 1
                // byte, so a length past the cap cannot be legitimate. This
                // check exists to avoid reserving for a corrupt count.
                if len > self.max_bytes {
                    return Err(GgufHeaderError::HeaderTooLarge {
                        needed: len,
                        max_bytes: self.max_bytes,
                    });
                }
                let mut items = Vec::new();
                for _ in 0..len {
                    items.push(self.value(elem_type, context)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_bits(self.u64()?)),
            other => return Err(GgufHeaderError::BadValueType { id: other }),
        })
    }
}

/// Parse a GGUF header from the leading bytes of a file (or however much of
/// it the caller has fetched so far). A [`GgufHeaderError::TooShort`] carries
/// the offset needed to get further, so a ranged caller can fetch exactly
/// that and call again.
pub fn parse_header(leading_bytes: &[u8], max_bytes: u64) -> Result<GgufHeader, GgufHeaderError> {
    let mut c = Cursor {
        data: leading_bytes,
        pos: 0,
        max_bytes,
    };

    let magic = c.u32()?;
    if magic != MAGIC {
        return Err(GgufHeaderError::BadMagic { found: magic });
    }
    let version = c.u32()?;
    if version != SUPPORTED_VERSION {
        return Err(GgufHeaderError::UnsupportedVersion { version });
    }
    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;

    // Cheapest possible guard against a corrupt count: the smallest legal
    // tensor entry is 8 (name length) + 4 (n_dims) + 8 (one dim) + 4 (type)
    // + 8 (offset) = 32 bytes, and the smallest KV pair is 8 + 4 + 1 = 13.
    // Anything claiming more than the cap can hold is rejected before a
    // single allocation.
    let floor = tensor_count
        .saturating_mul(32)
        .saturating_add(kv_count.saturating_mul(13));
    if floor > max_bytes {
        return Err(GgufHeaderError::HeaderTooLarge {
            needed: floor,
            max_bytes,
        });
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = c.string("metadata key")?;
        let type_id = c.u32()?;
        let value = c.value(type_id, &key)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GgufHeaderError::DuplicateKey { name: key });
        }
    }

    let mut tensors = BTreeMap::new();
    for _ in 0..tensor_count {
        let name = c.string("tensor name")?;
        let n_dims = c.u32()?;
        // ggml's hard limit is 4. Zero dims would make element_count() an
        // empty product (1), which is not a tensor.
        if n_dims == 0 || n_dims > 4 {
            return Err(GgufHeaderError::BadDimensions { name, n_dims });
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(c.u64()?);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        if tensors
            .insert(
                name.clone(),
                GgufTensorInfo {
                    ggml_type,
                    dims,
                    offset,
                },
            )
            .is_some()
        {
            return Err(GgufHeaderError::DuplicateTensor { name });
        }
    }

    let alignment = match metadata.get("general.alignment") {
        Some(v) => v
            .as_u64()
            .ok_or(GgufHeaderError::BadAlignment { alignment: 0 })?,
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufHeaderError::BadAlignment { alignment });
    }

    let table_end = c.pos as u64;
    let data_region_start = table_end.div_ceil(alignment) * alignment;

    Ok(GgufHeader {
        version,
        metadata,
        tensors,
        alignment,
        data_region_start,
    })
}
