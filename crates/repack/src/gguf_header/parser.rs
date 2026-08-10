//! Binary parser and cursor logic for GGUF headers.

use std::collections::BTreeMap;

use super::types::{
    GgufHeader, GgufHeaderError, GgufTensorInfo, GgufValue, DEFAULT_ALIGNMENT, MAGIC,
    SUPPORTED_VERSION,
};

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
