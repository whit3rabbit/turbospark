//! Synthetic GGUF v3 file builder.

pub mod tensors;

use crate::gguf_header::GgufValue;

/// A built GGUF file plus the absolute `[start, end)` byte range of each
/// tensor's data, in push order. The ranges come back alongside the bytes so
/// a test can assert byte identity without re-deriving the very layout it is
/// trying to verify.
pub type GgufFileAndRanges = (Vec<u8>, Vec<(String, (u64, u64))>);

/// Default alignment the spec assumes when `general.alignment` is absent.
/// The builder writes tensor data at multiples of this and does NOT emit
/// the key, so the round trip also covers the absent-key default path.
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub(crate) struct PendingTensor {
    pub(crate) name: String,
    pub(crate) ggml_type: u32,
    pub(crate) dims: Vec<u64>,
    pub(crate) data: Vec<u8>,
}

/// Accumulates metadata and tensors, then serializes one GGUF v3 file.
#[derive(Debug, Clone)]
pub struct GgufBuilder {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<PendingTensor>,
    alignment: u64,
    /// When set, `general.alignment` is written into the metadata. Left
    /// unset, the file relies on the spec's default and the reader's
    /// fallback.
    emit_alignment_key: bool,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    /// Creates a new `GgufBuilder` instance with default alignment.
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
            emit_alignment_key: false,
        }
    }

    /// Override the data-region alignment AND write `general.alignment`.
    /// Panics on a non-power-of-two, matching what the reader rejects:
    /// a fixture that cannot be read back is a broken fixture, not a test
    /// case, and the rejection path has its own hand-built bytes.
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        assert!(
            alignment > 0 && alignment.is_power_of_two(),
            "alignment {alignment} is not a non-zero power of two"
        );
        self.alignment = alignment;
        self.emit_alignment_key = true;
        self
    }

    /// Appends a key-value metadata entry to the GGUF header.
    pub fn metadata(mut self, key: &str, value: GgufValue) -> Self {
        self.metadata.push((key.to_string(), value));
        self
    }

    /// Appends a string metadata entry to the GGUF header.
    pub fn metadata_str(self, key: &str, value: &str) -> Self {
        self.metadata(key, GgufValue::String(value.to_string()))
    }

    /// Appends a `u32` metadata entry to the GGUF header.
    pub fn metadata_u32(self, key: &str, value: u32) -> Self {
        self.metadata(key, GgufValue::U32(value))
    }

    /// Appends an `f32` metadata entry to the GGUF header.
    pub fn metadata_f32(self, key: &str, value: f32) -> Self {
        self.metadata(key, GgufValue::F32(value))
    }

    /// Serialize. Returns the file bytes and, alongside them, the absolute
    /// `[start, end)` range of each tensor's data in push order, so a test
    /// can assert byte identity without re-deriving the layout it is trying
    /// to verify.
    pub fn build(&self) -> GgufFileAndRanges {
        let mut out = Vec::new();
        out.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        let kv_count = self.metadata.len() as u64 + u64::from(self.emit_alignment_key);
        out.extend_from_slice(&kv_count.to_le_bytes());

        for (key, value) in &self.metadata {
            write_string(&mut out, key);
            write_value(&mut out, value);
        }
        if self.emit_alignment_key {
            write_string(&mut out, "general.alignment");
            write_value(&mut out, &GgufValue::U32(self.alignment as u32));
        }

        // Data-region offsets have to be known before the table is written,
        // and the table's length depends on the names, so lay the region out
        // first. Offsets are relative to the data region, so this pass does
        // not depend on where the table ends.
        let mut relative = 0u64;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            offsets.push(relative);
            relative = (relative + t.data.len() as u64).div_ceil(self.alignment) * self.alignment;
        }

        for (t, offset) in self.tensors.iter().zip(offsets.iter()) {
            write_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
        }

        let data_region_start = (out.len() as u64).div_ceil(self.alignment) * self.alignment;
        out.resize(data_region_start as usize, 0);

        let mut ranges = Vec::with_capacity(self.tensors.len());
        for (t, offset) in self.tensors.iter().zip(offsets.iter()) {
            let start = data_region_start + offset;
            out.resize(start as usize, 0);
            out.extend_from_slice(&t.data);
            ranges.push((t.name.clone(), (start, start + t.data.len() as u64)));
        }

        (out, ranges)
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Writes a value's type id followed by its payload, mirroring the reader
/// in `gguf_header`'s `Cursor::value` exactly.
fn write_value(out: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::U8(v) => {
            out.extend_from_slice(&0u32.to_le_bytes());
            out.push(*v);
        }
        GgufValue::I8(v) => {
            out.extend_from_slice(&1u32.to_le_bytes());
            out.push(*v as u8);
        }
        GgufValue::U16(v) => {
            out.extend_from_slice(&2u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I16(v) => {
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::U32(v) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I32(v) => {
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F32(v) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        GgufValue::Bool(v) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*v));
        }
        GgufValue::String(v) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_string(out, v);
        }
        GgufValue::Array(items) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            // An array writes its ELEMENT type once, then the length, then
            // bare payloads: the elements do not repeat the type id. An
            // empty array has no element to take the type from, so it is
            // written as an empty array of U8 rather than being rejected.
            let elem_type = items.first().map_or(0, value_type_id);
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                assert_eq!(
                    value_type_id(item),
                    elem_type,
                    "GGUF arrays are homogeneous; mixed element types cannot be encoded"
                );
                write_payload(out, item);
            }
        }
        GgufValue::U64(v) => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::I64(v) => {
            out.extend_from_slice(&11u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::F64(v) => {
            out.extend_from_slice(&12u32.to_le_bytes());
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}

fn value_type_id(value: &GgufValue) -> u32 {
    match value {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

/// The payload alone, with no leading type id: what array elements use.
fn write_payload(out: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::U8(v) => out.push(*v),
        GgufValue::I8(v) => out.push(*v as u8),
        GgufValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        GgufValue::Bool(v) => out.push(u8::from(*v)),
        GgufValue::String(v) => write_string(out, v),
        GgufValue::Array(items) => {
            let elem_type = items.first().map_or(0, value_type_id);
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_payload(out, item);
            }
        }
        GgufValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
    }
}
