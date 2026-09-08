//! Parses the `safetensors` file header: an 8-byte little-endian header
//! length, followed by that many bytes of UTF-8 JSON describing each
//! tensor's dtype, shape, and byte range within the data region that
//! immediately follows the header. See
//! <https://github.com/huggingface/safetensors> for the format spec.
//!
//! This is the format the repacker's ranged-download planning reads: the
//! header alone (typically a few KB, even for a multi-GB checkpoint) tells
//! the downloader exactly which byte ranges of the remote file it needs for
//! a given tensor, so the whole checkpoint is never materialized locally.

use std::collections::BTreeMap;

use serde::Deserialize;

/// Shape, data type, and byte range metadata for one tensor in a safetensors file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Data type string (e.g. "F32", "BF16", "U32").
    pub dtype: String,
    /// Tensor dimensions.
    pub shape: Vec<u64>,
    /// Byte offsets relative to the start of the data region (i.e.
    /// relative to file byte `8 + header_len`), not to the file start.
    pub data_offsets: (u64, u64),
}

/// Parsed safetensors file header containing tensor descriptors and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsHeader {
    /// Map of tensor names to their descriptors.
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Optional metadata key-value map.
    pub metadata: Option<BTreeMap<String, String>>,
    /// Byte length of the header JSON, as read from the file's leading
    /// 8-byte length prefix.
    pub header_len: u64,
}

impl SafetensorsHeader {
    /// The absolute file offset where the data region begins.
    pub fn data_region_start(&self) -> u64 {
        8 + self.header_len
    }

    /// The absolute file byte range `[start, end)` for `name`'s tensor
    /// data, or `None` if `name` is not present, or if the offset would
    /// overflow `u64` (a header field this parser accepts but cannot
    /// address is refused here rather than wrapping to a plausible-looking
    /// range).
    pub fn absolute_range(&self, name: &str) -> Option<(u64, u64)> {
        let info = self.tensors.get(name)?;
        let base = self.data_region_start();
        let start = base.checked_add(info.data_offsets.0)?;
        let end = base.checked_add(info.data_offsets.1)?;
        Some((start, end))
    }
}

/// Errors encountered while parsing a safetensors header.
#[derive(Debug, Clone, PartialEq)]
pub enum SafetensorsHeaderError {
    /// The buffer is too short to contain the 8-byte length prefix.
    TooShort,
    /// The declared header length exceeds the safety cap.
    HeaderTooLarge { header_len: u64, max_bytes: u64 },
    /// The header JSON is not valid UTF-8.
    InvalidUtf8,
    /// The header JSON failed to parse.
    InvalidJson(String),
}

impl std::fmt::Display for SafetensorsHeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetensorsHeaderError::TooShort => {
                write!(f, "file is shorter than the 8-byte header length prefix")
            }
            SafetensorsHeaderError::HeaderTooLarge {
                header_len,
                max_bytes,
            } => {
                write!(
                    f,
                    "header length {header_len} exceeds the {max_bytes}-byte cap"
                )
            }
            SafetensorsHeaderError::InvalidUtf8 => write!(f, "header JSON is not valid UTF-8"),
            SafetensorsHeaderError::InvalidJson(detail) => {
                write!(f, "header JSON is malformed: {detail}")
            }
        }
    }
}

impl std::error::Error for SafetensorsHeaderError {}

/// Metadata cap: no legitimate safetensors header (even for a
/// multi-hundred-tensor checkpoint) approaches this; it exists to reject a
/// corrupt or hostile length prefix before allocating a buffer for it.
pub const DEFAULT_MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Parse a safetensors header from the leading bytes of a file (or however
/// much of it the caller has fetched so far — the length prefix tells the
/// caller exactly how many more bytes it needs before calling this).
pub fn parse_header(
    leading_bytes: &[u8],
    max_bytes: u64,
) -> Result<SafetensorsHeader, SafetensorsHeaderError> {
    if leading_bytes.len() < 8 {
        return Err(SafetensorsHeaderError::TooShort);
    }
    let header_len = u64::from_le_bytes(leading_bytes[0..8].try_into().unwrap());
    if header_len > max_bytes {
        return Err(SafetensorsHeaderError::HeaderTooLarge {
            header_len,
            max_bytes,
        });
    }
    let end =
        8usize
            .checked_add(header_len as usize)
            .ok_or(SafetensorsHeaderError::HeaderTooLarge {
                header_len,
                max_bytes,
            })?;
    if leading_bytes.len() < end {
        return Err(SafetensorsHeaderError::TooShort);
    }
    let json_bytes = &leading_bytes[8..end];
    let text = std::str::from_utf8(json_bytes).map_err(|_| SafetensorsHeaderError::InvalidUtf8)?;

    #[derive(Deserialize)]
    struct TensorEntry {
        dtype: String,
        shape: Vec<u64>,
        data_offsets: (u64, u64),
    }

    // Every entry's value goes through `serde_json::Value` first rather than
    // an untagged `Entry` enum with a `Tensor` and a `Metadata` variant:
    // real converters write `"__metadata__": null` (confirmed against a real
    // published checkpoint, not assumed), and neither variant of an
    // object-shaped enum can deserialize from JSON `null` -- the untagged
    // form failed the whole file with "data did not match any variant",
    // which named neither the key nor why. `__metadata__` is handled by
    // NAME rather than by shape, because `null` and `{...}` are its only
    // two legitimate values and a tensor entry is never named that.
    let raw: BTreeMap<String, serde_json::Value> = serde_json::from_str(text)
        .map_err(|e| SafetensorsHeaderError::InvalidJson(e.to_string()))?;

    let mut tensors = BTreeMap::new();
    let mut metadata = None;
    for (key, value) in raw {
        if key == "__metadata__" {
            match value {
                serde_json::Value::Null => {}
                serde_json::Value::Object(_) => {
                    let m: BTreeMap<String, String> =
                        serde_json::from_value(value).map_err(|e| {
                            SafetensorsHeaderError::InvalidJson(format!("__metadata__: {e}"))
                        })?;
                    metadata = Some(m);
                }
                other => {
                    return Err(SafetensorsHeaderError::InvalidJson(format!(
                        "__metadata__ is neither null nor an object: {other}"
                    )))
                }
            }
            continue;
        }
        let entry: TensorEntry = serde_json::from_value(value)
            .map_err(|e| SafetensorsHeaderError::InvalidJson(format!("{key}: {e}")))?;
        // The format requires a monotone, non-overlapping range per tensor.
        // An inverted one reaches `absolute_range` as a plausible-looking
        // pair and then every unchecked `end - start` downstream of it
        // (ranged_download's chunking, the expert-blob planner) either
        // panics in debug or wraps to a huge allocation in release.
        if entry.data_offsets.0 > entry.data_offsets.1 {
            return Err(SafetensorsHeaderError::InvalidJson(format!(
                "{key}: data_offsets end {} before start {}",
                entry.data_offsets.1, entry.data_offsets.0
            )));
        }
        tensors.insert(
            key,
            TensorInfo {
                dtype: entry.dtype,
                shape: entry.shape,
                data_offsets: entry.data_offsets,
            },
        );
    }

    Ok(SafetensorsHeader {
        tensors,
        metadata,
        header_len,
    })
}

/// How many bytes of the file the caller must fetch before
/// [`parse_header`] can succeed, given only the first 8 bytes so far.
pub fn required_prefix_len(first_8_bytes: &[u8; 8]) -> u64 {
    8 + u64::from_le_bytes(*first_8_bytes)
}
