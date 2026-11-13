//! Reader for the `.npz` archive that carries the Underdog Woof checkpoint.
//!
//! The archive is a plain ZIP written by `numpy.savez` (method 0, stored),
//! so no deflate is needed. Each member is an `.npy` array. Three element
//! types occur in the checkpoint (`<f4`, `<f2`, `|i1`) plus one 0-d `<U<n>`
//! string, `__tree__`, that holds the JSON structure the arrays hang off.
//!
//! This is deliberately narrow: it refuses compressed members, ZIP64,
//! big-endian or Fortran-ordered arrays and any element type it has not been
//! checked against, with a message that names the member. Per-member CRC-32
//! is not verified (that would need a new dependency). Integrity rests on the
//! whole-file SHA-256 and the per-tensor hash gate in `tests.rs`.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::{Result, SpeechError};

const EOCD_SIG: u32 = 0x0605_4b50;
const CEN_SIG: u32 = 0x0201_4b50;
const LOC_SIG: u32 = 0x0403_4b50;

/// Element types the checkpoint uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    I8,
}

impl Dtype {
    fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::I8 => 1,
        }
    }
}

/// A parsed `.npy` member: shape plus a borrowed view of its payload bytes.
#[derive(Debug, Clone)]
pub struct Npy<'a> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl Npy<'_> {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as f32. f16 widens exactly; int8 values convert exactly.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            Dtype::I8 => self.data.iter().map(|&b| f32::from(b as i8)).collect(),
        }
    }

    /// The raw int8 elements, or an error for any other element type.
    pub fn to_i8(&self, name: &str) -> Result<Vec<i8>> {
        if self.dtype != Dtype::I8 {
            return Err(tensor_err(
                name,
                format!("expected int8, found {:?}", self.dtype),
            ));
        }
        Ok(self.data.iter().map(|&b| b as i8).collect())
    }
}

/// An opened archive: the file bytes plus an index of stored members.
pub struct Npz {
    bytes: Vec<u8>,
    // member name without the ".npy" suffix -> (data start, data length)
    members: HashMap<String, (usize, usize)>,
}

impl Npz {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| SpeechError::BadConfig {
            field: path.display().to_string(),
            why: format!("cannot read checkpoint: {e}"),
        })?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let members = index_members(&bytes)?;
        Ok(Self { bytes, members })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.members.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.members.contains_key(name)
    }

    /// The payload range of a member within the file, for tests that need to
    /// corrupt one byte of one tensor.
    pub fn member_range(&self, name: &str) -> Option<std::ops::Range<usize>> {
        self.members
            .get(name)
            .map(|&(start, len)| start..start + len)
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn array(&self, name: &str) -> Result<Npy<'_>> {
        let &(start, len) = self
            .members
            .get(name)
            .ok_or_else(|| tensor_err(name, "not in the archive".into()))?;
        parse_npy(name, &self.bytes[start..start + len])
    }

    /// The JSON structure stored in the 0-d string member `__tree__`.
    pub fn tree(&self) -> Result<Value> {
        let &(start, len) = self
            .members
            .get("__tree__")
            .ok_or_else(|| tensor_err("__tree__", "not in the archive".into()))?;
        let text = parse_npy_string("__tree__", &self.bytes[start..start + len])?;
        serde_json::from_str(&text).map_err(|e| SpeechError::BadConfig {
            field: "__tree__".into(),
            why: format!("not valid JSON: {e}"),
        })
    }
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn zip_err(why: impl Into<String>) -> SpeechError {
    SpeechError::BadConfig {
        field: "npz archive".into(),
        why: why.into(),
    }
}

fn tensor_err(name: &str, why: String) -> SpeechError {
    SpeechError::Tensor {
        name: name.to_string(),
        why,
    }
}

fn index_members(b: &[u8]) -> Result<HashMap<String, (usize, usize)>> {
    // The end-of-central-directory record is the last 22 bytes plus an
    // optional comment of up to 65535 bytes.
    let lowest = b.len().saturating_sub(22 + 65_535);
    let eocd = (lowest..=b.len().saturating_sub(22))
        .rev()
        .find(|&i| u32_at(b, i) == Some(EOCD_SIG))
        .ok_or_else(|| zip_err("no end-of-central-directory record, not a ZIP file"))?;
    let entries = u16_at(b, eocd + 10).ok_or_else(|| zip_err("truncated end record"))?;
    let cen_off = u32_at(b, eocd + 16).ok_or_else(|| zip_err("truncated end record"))?;
    if entries == 0xFFFF || cen_off == 0xFFFF_FFFF {
        return Err(zip_err("ZIP64 archives are not supported"));
    }
    let mut at = cen_off as usize;
    let mut out = HashMap::with_capacity(entries as usize);
    for _ in 0..entries {
        if u32_at(b, at) != Some(CEN_SIG) {
            return Err(zip_err(format!("bad central directory entry at {at}")));
        }
        let method = u16_at(b, at + 10).ok_or_else(|| zip_err("truncated entry"))?;
        let comp = u32_at(b, at + 20).ok_or_else(|| zip_err("truncated entry"))? as usize;
        let raw = u32_at(b, at + 24).ok_or_else(|| zip_err("truncated entry"))? as usize;
        let name_len = u16_at(b, at + 28).ok_or_else(|| zip_err("truncated entry"))? as usize;
        let extra_len = u16_at(b, at + 30).ok_or_else(|| zip_err("truncated entry"))? as usize;
        let comment_len = u16_at(b, at + 32).ok_or_else(|| zip_err("truncated entry"))? as usize;
        let local = u32_at(b, at + 42).ok_or_else(|| zip_err("truncated entry"))? as usize;
        let name = b
            .get(at + 46..at + 46 + name_len)
            .and_then(|n| std::str::from_utf8(n).ok())
            .ok_or_else(|| zip_err(format!("bad member name at {at}")))?
            .to_string();
        if method != 0 || comp != raw {
            return Err(zip_err(format!(
                "member {name} is compressed (method {method}); only stored members are supported"
            )));
        }
        if u32_at(b, local) != Some(LOC_SIG) {
            return Err(zip_err(format!("member {name} has no local header")));
        }
        let lname =
            u16_at(b, local + 26).ok_or_else(|| zip_err("truncated local header"))? as usize;
        let lextra =
            u16_at(b, local + 28).ok_or_else(|| zip_err("truncated local header"))? as usize;
        let start = local + 30 + lname + lextra;
        let end = start
            .checked_add(raw)
            .filter(|&e| e <= b.len())
            .ok_or_else(|| zip_err(format!("member {name} runs past the end of the file")))?;
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(key, (start, end - start));
        at += 46 + name_len + extra_len + comment_len;
    }
    Ok(out)
}

/// Split an `.npy` payload into its header text and the data that follows.
fn split_npy<'a>(name: &str, b: &'a [u8]) -> Result<(String, &'a [u8])> {
    if b.get(..6) != Some(b"\x93NUMPY") {
        return Err(tensor_err(name, "not an .npy member".into()));
    }
    let (hlen, off) = match b.get(6) {
        Some(1) => (u16_at(b, 8).map(usize::from), 10),
        Some(2) | Some(3) => (u32_at(b, 8).map(|v| v as usize), 12),
        v => return Err(tensor_err(name, format!("unsupported .npy version {v:?}"))),
    };
    let hlen = hlen.ok_or_else(|| tensor_err(name, "truncated .npy header".into()))?;
    let header = b
        .get(off..off + hlen)
        .ok_or_else(|| tensor_err(name, "truncated .npy header".into()))?;
    let header = String::from_utf8_lossy(header).into_owned();
    Ok((header, &b[off + hlen..]))
}

/// Value of `'key': '...'` in the header dict.
fn header_str<'h>(header: &'h str, key: &str) -> Option<&'h str> {
    let after = &header[header.find(&format!("'{key}':"))? + key.len() + 3..];
    let after = after.trim_start().strip_prefix('\'')?;
    Some(&after[..after.find('\'')?])
}

fn header_shape(header: &str) -> Option<Vec<usize>> {
    let after = &header[header.find("'shape':")? + 8..];
    let inner = &after[after.find('(')? + 1..after.find(')')?];
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().ok())
        .collect()
}

fn parse_npy<'a>(name: &str, b: &'a [u8]) -> Result<Npy<'a>> {
    let (header, data) = split_npy(name, b)?;
    if header.contains("'fortran_order': True") {
        return Err(tensor_err(
            name,
            "Fortran-ordered arrays are not supported".into(),
        ));
    }
    let descr = header_str(&header, "descr")
        .ok_or_else(|| tensor_err(name, "no dtype in header".into()))?;
    let dtype = match descr {
        "<f4" => Dtype::F32,
        "<f2" => Dtype::F16,
        "|i1" => Dtype::I8,
        other => {
            return Err(tensor_err(
                name,
                format!("unsupported dtype {other}; verified dtypes are <f4, <f2, |i1"),
            ))
        }
    };
    let shape =
        header_shape(&header).ok_or_else(|| tensor_err(name, "no shape in header".into()))?;
    let want = shape.iter().product::<usize>() * dtype.size();
    if data.len() != want {
        return Err(tensor_err(
            name,
            format!(
                "payload is {} bytes, shape {shape:?} needs {want}",
                data.len()
            ),
        ));
    }
    Ok(Npy { dtype, shape, data })
}

/// A 0-d `<U<n>` array (UCS-4 little-endian), which is how `__tree__` is stored.
fn parse_npy_string(name: &str, b: &[u8]) -> Result<String> {
    let (header, data) = split_npy(name, b)?;
    let descr = header_str(&header, "descr")
        .ok_or_else(|| tensor_err(name, "no dtype in header".into()))?;
    let chars: usize = descr
        .strip_prefix("<U")
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| tensor_err(name, format!("expected a <U string, found {descr}")))?;
    if header_shape(&header) != Some(vec![]) || data.len() != chars * 4 {
        return Err(tensor_err(name, "expected a 0-d string array".into()));
    }
    data.chunks_exact(4)
        .map(|c| char::from_u32(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect::<Option<String>>()
        .ok_or_else(|| tensor_err(name, "invalid UCS-4 code point".into()))
}

/// IEEE 754 binary16 to binary32. Exact for every input: normals, subnormals,
/// signed zero, infinities, and NaN (payload preserved). The stable toolchain
/// at this crate's MSRV has no `f16`, and numpy's `astype(float32)` is exact,
/// so a lossy shortcut here would break the tensor hashes.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let frac = u32::from(h & 0x3FF);
    let bits = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // Subnormal: value = f * 2^-24, with f in 1..=1023. With the
            // leading one at bit p = 31 - lz, the value is 1.x * 2^(p - 24),
            // so the f32 exponent field is 103 + p = 113 - shift, where
            // shift = 10 - p moves the leading one to the implicit-bit slot.
            let shift = f.leading_zeros() - 21;
            let frac32 = (f << shift) & 0x3FF;
            sign | ((113 - shift) << 23) | (frac32 << 13)
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, f) => sign | 0x7F80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}
