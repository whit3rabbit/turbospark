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

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte offsets relative to the start of the data region (i.e.
    /// relative to file byte `8 + header_len`), not to the file start.
    pub data_offsets: (u64, u64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsHeader {
    pub tensors: BTreeMap<String, TensorInfo>,
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
    /// data, or `None` if `name` is not present.
    pub fn absolute_range(&self, name: &str) -> Option<(u64, u64)> {
        let info = self.tensors.get(name)?;
        let base = self.data_region_start();
        Some((base + info.data_offsets.0, base + info.data_offsets.1))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SafetensorsHeaderError {
    TooShort,
    HeaderTooLarge { header_len: u64, max_bytes: u64 },
    InvalidUtf8,
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
    #[serde(untagged)]
    enum Entry {
        Tensor {
            dtype: String,
            shape: Vec<u64>,
            data_offsets: (u64, u64),
        },
        Metadata(BTreeMap<String, String>),
    }

    let raw: BTreeMap<String, Entry> = serde_json::from_str(text)
        .map_err(|e| SafetensorsHeaderError::InvalidJson(e.to_string()))?;

    let mut tensors = BTreeMap::new();
    let mut metadata = None;
    for (key, entry) in raw {
        match entry {
            Entry::Tensor {
                dtype,
                shape,
                data_offsets,
            } => {
                tensors.insert(
                    key,
                    TensorInfo {
                        dtype,
                        shape,
                        data_offsets,
                    },
                );
            }
            Entry::Metadata(m) if key == "__metadata__" => metadata = Some(m),
            Entry::Metadata(_) => {
                return Err(SafetensorsHeaderError::InvalidJson(format!(
                    "entry \"{key}\" is neither a tensor descriptor nor __metadata__"
                )))
            }
        }
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
