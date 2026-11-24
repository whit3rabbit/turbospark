//! A stored-only `.npz` reader (NumPy `np.savez`): a zip archive of `.npy`
//! members, no compression. Used by the LookAtMyQuant intake
//! (`docs/LAMQ.md`); no zip dependency.
//!
//! What is accepted, and why:
//! - Sizes, CRCs and offsets come from the central directory. `np.savez`
//!   writes members with `force_zip64=True`, so each LOCAL header carries
//!   `0xFFFFFFFF` sizes and a zip64 extra (id 1) with the real ones. Those are
//!   checked against the central directory, never trusted on their own.
//! - Archive-level zip64 (a zip64 EOCD, or `0xFFFF`/`0xFFFFFFFF` in the EOCD
//!   or a central entry) is refused: the releases are all below 4 GiB per file.
//! - Method 0 only. Deflated (`savez_compressed`), encrypted, multi-disk and
//!   data-descriptor members are refused by name.
//! - Members may not overlap each other or the central directory, names must
//!   be unique, and every member's CRC32 is checked on read.
//! - `.npy` v1/v2/v3 headers, little-endian or byte dtypes, C order only.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Central directory bytes accepted (the releases use under 4 KiB).
const MAX_CD_BYTES: u64 = 16 << 20;
/// Members accepted in one archive.
const MAX_MEMBERS: usize = 65_536;
/// `.npy` header bytes accepted.
const MAX_NPY_HEADER: usize = 64 << 10;
/// EOCD (22 bytes) plus the largest zip comment.
const EOCD_SEARCH: u64 = 22 + 0xFFFF;

#[derive(Debug)]
pub struct NpzError(pub String);

impl std::fmt::Display for NpzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NpzError {}

fn err<T>(msg: impl Into<String>) -> Result<T, NpzError> {
    Err(NpzError(msg.into()))
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// IEEE CRC-32 (zip's), table-driven.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *e = c;
        }
        t
    });
    !data.iter().fold(!0u32, |c, &b| {
        t[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8)
    })
}

#[derive(Clone, Debug)]
struct Member {
    header_offset: u64,
    size: u64,
    crc: u32,
}

/// One opened `.npz`. Members are read on demand.
#[derive(Debug)]
pub struct NpzArchive {
    path: PathBuf,
    file: File,
    members: BTreeMap<String, Member>,
}

impl NpzArchive {
    pub fn open(path: &Path) -> Result<Self, NpzError> {
        let ctx = |e: std::io::Error| NpzError(format!("{}: {e}", path.display()));
        let mut file = File::open(path).map_err(ctx)?;
        let len = file.metadata().map_err(ctx)?.len();
        let tail_len = len.min(EOCD_SEARCH);
        let mut tail = vec![0u8; tail_len as usize];
        file.seek(SeekFrom::Start(len - tail_len)).map_err(ctx)?;
        file.read_exact(&mut tail).map_err(ctx)?;
        let p = path.display();

        // The EOCD whose comment ends exactly at end of file.
        let eocd = (0..tail.len().saturating_sub(21))
            .rev()
            .find(|&i| {
                u32_at(&tail, i) == 0x0605_4B50
                    && i + 22 + usize::from(u16_at(&tail, i + 20)) == tail.len()
            })
            .ok_or_else(|| NpzError(format!("{p}: no zip end-of-central-directory record")))?;
        if eocd >= 20 && u32_at(&tail, eocd - 20) == 0x0706_4B50 {
            return err(format!("{p}: zip64 archive refused"));
        }
        let (disk, cd_disk) = (u16_at(&tail, eocd + 4), u16_at(&tail, eocd + 6));
        let (on_disk, total) = (u16_at(&tail, eocd + 8), u16_at(&tail, eocd + 10));
        let cd_size = u64::from(u32_at(&tail, eocd + 12));
        let cd_offset = u64::from(u32_at(&tail, eocd + 16));
        if disk != 0 || cd_disk != 0 || on_disk != total {
            return err(format!("{p}: multi-disk zip refused"));
        }
        if total == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_offset == 0xFFFF_FFFF {
            return err(format!("{p}: zip64 sizes in the end record refused"));
        }
        let eocd_abs = len - tail_len + eocd as u64;
        if cd_size > MAX_CD_BYTES || cd_offset.checked_add(cd_size) != Some(eocd_abs) {
            return err(format!(
                "{p}: central directory [{cd_offset}, +{cd_size}) does not end at the end record ({eocd_abs})"
            ));
        }
        let mut cd = vec![0u8; cd_size as usize];
        file.seek(SeekFrom::Start(cd_offset)).map_err(ctx)?;
        file.read_exact(&mut cd).map_err(ctx)?;

        let mut members = BTreeMap::new();
        let mut spans: Vec<(u64, u64, String)> = Vec::new();
        let mut i = 0usize;
        for _ in 0..total {
            if i + 46 > cd.len() || u32_at(&cd, i) != 0x0201_4B50 {
                return err(format!("{p}: central directory entry at {i} is malformed"));
            }
            let flags = u16_at(&cd, i + 8);
            let method = u16_at(&cd, i + 10);
            let crc = u32_at(&cd, i + 16);
            let csize = u32_at(&cd, i + 20);
            let usize_ = u32_at(&cd, i + 24);
            let (fl, el, cl) = (
                usize::from(u16_at(&cd, i + 28)),
                usize::from(u16_at(&cd, i + 30)),
                usize::from(u16_at(&cd, i + 32)),
            );
            let start_disk = u16_at(&cd, i + 34);
            let header_offset = u64::from(u32_at(&cd, i + 42));
            let end = i + 46 + fl + el + cl;
            if end > cd.len() {
                return err(format!(
                    "{p}: central directory entry at {i} runs past the directory"
                ));
            }
            let name = std::str::from_utf8(&cd[i + 46..i + 46 + fl])
                .map_err(|_| NpzError(format!("{p}: member name is not UTF-8")))?
                .to_string();
            if flags & 0x1 != 0 {
                return err(format!("{p}: {name}: encrypted member refused"));
            }
            if flags & 0x8 != 0 {
                return err(format!("{p}: {name}: data-descriptor member refused"));
            }
            if method != 0 {
                return err(format!(
                    "{p}: {name}: compression method {method} refused (stored only; was it np.savez_compressed?)"
                ));
            }
            if csize != usize_
                || csize == 0xFFFF_FFFF
                || header_offset == 0xFFFF_FFFF
                || start_disk != 0
            {
                return err(format!(
                    "{p}: {name}: zip64 or inconsistent sizes in the central directory"
                ));
            }
            let Some(stem) = name.strip_suffix(".npy") else {
                return err(format!("{p}: member {name} is not a .npy"));
            };
            if stem.is_empty() || stem.contains('/') || stem.contains('\\') {
                return err(format!("{p}: member name {name:?} refused"));
            }
            if members.len() >= MAX_MEMBERS {
                return err(format!("{p}: more than {MAX_MEMBERS} members"));
            }
            let m = Member {
                header_offset,
                size: u64::from(usize_),
                crc,
            };
            if members.insert(stem.to_string(), m).is_some() {
                return err(format!("{p}: duplicate member {name}"));
            }
            spans.push((header_offset, u64::from(usize_), stem.to_string()));
            i = end;
        }
        if i != cd.len() {
            return err(format!(
                "{p}: {} trailing central directory bytes",
                cd.len() - i
            ));
        }

        let mut archive = Self {
            path: path.to_path_buf(),
            file,
            members,
        };
        // Local headers: agree with the central directory, give the data start,
        // and no member overlaps another or the directory.
        let mut ranges = Vec::with_capacity(spans.len());
        for (off, size, stem) in spans {
            let data = archive.local_data_offset(&stem, off, size)?;
            ranges.push((off, data + size, stem));
        }
        ranges.sort();
        let mut prev_end = 0u64;
        for (start, end, stem) in &ranges {
            if *start < prev_end {
                return err(format!("{p}: member {stem} overlaps the one before it"));
            }
            prev_end = *end;
        }
        if prev_end > cd_offset {
            return err(format!("{p}: a member overlaps the central directory"));
        }
        Ok(archive)
    }

    /// Validates `stem`'s local header and returns where its data starts.
    fn local_data_offset(&mut self, stem: &str, off: u64, size: u64) -> Result<u64, NpzError> {
        let p = self.path.display().to_string();
        let mut h = [0u8; 30];
        self.file
            .seek(SeekFrom::Start(off))
            .and_then(|_| self.file.read_exact(&mut h))
            .map_err(|e| NpzError(format!("{p}: {stem}: local header: {e}")))?;
        if u32_at(&h, 0) != 0x0403_4B50 {
            return err(format!("{p}: {stem}: bad local header signature"));
        }
        let (flags, method) = (u16_at(&h, 6), u16_at(&h, 8));
        let (lcs, lus) = (u32_at(&h, 18), u32_at(&h, 22));
        let (fl, el) = (usize::from(u16_at(&h, 26)), usize::from(u16_at(&h, 28)));
        if flags & 0x9 != 0 || method != 0 {
            return err(format!(
                "{p}: {stem}: local header flags {flags:#x} / method {method} disagree"
            ));
        }
        let mut rest = vec![0u8; fl + el];
        self.file
            .read_exact(&mut rest)
            .map_err(|e| NpzError(format!("{p}: {stem}: local header: {e}")))?;
        if rest[..fl] != *format!("{stem}.npy").as_bytes() {
            return err(format!("{p}: {stem}: local header name disagrees"));
        }
        let sizes_ok = if lcs == 0xFFFF_FFFF && lus == 0xFFFF_FFFF {
            // np.savez's force_zip64: the zip64 extra holds the real sizes.
            let extra = &rest[fl..];
            let mut j = 0;
            let mut found = None;
            while j + 4 <= extra.len() {
                let (id, n) = (u16_at(extra, j), usize::from(u16_at(extra, j + 2)));
                if j + 4 + n > extra.len() {
                    break;
                }
                if id == 1 && n >= 16 {
                    found = Some((u64_at(extra, j + 4), u64_at(extra, j + 12)));
                }
                j += 4 + n;
            }
            found == Some((size, size))
        } else {
            u64::from(lcs) == size && u64::from(lus) == size
        };
        if !sizes_ok {
            return err(format!(
                "{p}: {stem}: local header sizes disagree with the central directory"
            ));
        }
        Ok(off + 30 + (fl + el) as u64)
    }

    /// Member names without `.npy`, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.members.keys().map(String::as_str)
    }

    pub fn contains(&self, stem: &str) -> bool {
        self.members.contains_key(stem)
    }

    /// Reads, CRC-checks and parses one member.
    pub fn read(&mut self, stem: &str) -> Result<NpyArray, NpzError> {
        let p = self.path.display().to_string();
        let m = self
            .members
            .get(stem)
            .cloned()
            .ok_or_else(|| NpzError(format!("{p}: no member {stem}")))?;
        let data_off = self.local_data_offset(stem, m.header_offset, m.size)?;
        let mut buf = vec![0u8; m.size as usize];
        self.file
            .seek(SeekFrom::Start(data_off))
            .and_then(|_| self.file.read_exact(&mut buf))
            .map_err(|e| NpzError(format!("{p}: {stem}: {e}")))?;
        if crc32(&buf) != m.crc {
            return err(format!("{p}: {stem}: CRC mismatch"));
        }
        NpyArray::parse(&buf).map_err(|e| NpzError(format!("{p}: {stem}: {}", e.0)))
    }
}

/// The `.npy` dtypes the LAMQ releases use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpyDtype {
    F16,
    F32,
    I8,
    U8,
    I16,
    I32,
    U32,
}

impl NpyDtype {
    fn from_descr(d: &str) -> Option<Self> {
        Some(match d {
            "<f2" => Self::F16,
            "<f4" => Self::F32,
            "|i1" => Self::I8,
            "|u1" => Self::U8,
            "<i2" => Self::I16,
            "<i4" => Self::I32,
            "<u4" => Self::U32,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::F16 | Self::I16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
        }
    }
}

/// One parsed `.npy`: little-endian bytes in C order.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    pub dtype: NpyDtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// The value of `'key': <value>` in a NumPy header dict, as raw text.
fn header_field<'a>(h: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("'{key}':");
    let at = h.find(&k)? + k.len();
    let v = h[at..].trim_start();
    let end = if v.starts_with('(') {
        v.find(')')? + 1
    } else if let Some(rest) = v.strip_prefix('\'') {
        rest.find('\'')? + 2
    } else {
        v.find([',', '}']).unwrap_or(v.len())
    };
    Some(v[..end].trim())
}

impl NpyArray {
    pub fn parse(b: &[u8]) -> Result<Self, NpzError> {
        if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
            return err("not a .npy");
        }
        let (hl, start) = match b[6] {
            1 => (usize::from(u16_at(b, 8)), 10),
            2 | 3 if b.len() >= 12 => (u32_at(b, 8) as usize, 12),
            v => return err(format!(".npy version {v} refused")),
        };
        if hl > MAX_NPY_HEADER || start + hl > b.len() {
            return err(".npy header length out of range");
        }
        let h = std::str::from_utf8(&b[start..start + hl])
            .map_err(|_| NpzError(".npy header not UTF-8".into()))?;
        let descr =
            header_field(h, "descr").ok_or_else(|| NpzError(".npy header has no descr".into()))?;
        let descr = descr.trim_matches('\'');
        let dtype = NpyDtype::from_descr(descr)
            .ok_or_else(|| NpzError(format!(".npy dtype {descr} refused")))?;
        if header_field(h, "fortran_order") != Some("False") {
            return err(".npy fortran_order is not False");
        }
        let shape_txt =
            header_field(h, "shape").ok_or_else(|| NpzError(".npy header has no shape".into()))?;
        let inner = shape_txt
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| NpzError(format!(".npy shape {shape_txt} malformed")))?;
        let shape = inner
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| NpzError(format!(".npy shape {shape_txt} malformed")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let count = shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| NpzError(".npy shape overflows".into()))?;
        let data = &b[start + hl..];
        if Some(data.len()) != count.checked_mul(dtype.size()) {
            return err(format!(
                ".npy {descr} {shape:?} wants {} data bytes, has {}",
                count.saturating_mul(dtype.size()),
                data.len()
            ));
        }
        Ok(Self {
            dtype,
            shape,
            data: data.to_vec(),
        })
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dtype.size()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Refuses any other dtype or shape.
    pub fn expect(&self, dtype: NpyDtype, shape: &[usize]) -> Result<&Self, NpzError> {
        if self.dtype != dtype || self.shape != shape {
            return err(format!(
                "got {:?} {:?}, want {dtype:?} {shape:?}",
                self.dtype, self.shape
            ));
        }
        Ok(self)
    }

    fn words<const N: usize>(
        &self,
        want: NpyDtype,
    ) -> Result<impl Iterator<Item = [u8; N]> + '_, NpzError> {
        if self.dtype != want {
            return err(format!("array is {:?}, read as {want:?}", self.dtype));
        }
        Ok(self.data.chunks_exact(N).map(|c| c.try_into().unwrap()))
    }

    pub fn to_u32(&self) -> Result<Vec<u32>, NpzError> {
        Ok(self
            .words::<4>(NpyDtype::U32)?
            .map(u32::from_le_bytes)
            .collect())
    }

    pub fn to_i32(&self) -> Result<Vec<i32>, NpzError> {
        Ok(self
            .words::<4>(NpyDtype::I32)?
            .map(i32::from_le_bytes)
            .collect())
    }

    pub fn to_f32(&self) -> Result<Vec<f32>, NpzError> {
        Ok(self
            .words::<4>(NpyDtype::F32)?
            .map(f32::from_le_bytes)
            .collect())
    }

    pub fn to_i16(&self) -> Result<Vec<i16>, NpzError> {
        Ok(self
            .words::<2>(NpyDtype::I16)?
            .map(i16::from_le_bytes)
            .collect())
    }

    /// Raw f16 bit patterns.
    pub fn to_f16_bits(&self) -> Result<Vec<u16>, NpzError> {
        Ok(self
            .words::<2>(NpyDtype::F16)?
            .map(u16::from_le_bytes)
            .collect())
    }

    pub fn to_i8(&self) -> Result<Vec<i8>, NpzError> {
        Ok(self.words::<1>(NpyDtype::I8)?.map(|b| b[0] as i8).collect())
    }

    pub fn bytes(&self, want: NpyDtype) -> Result<&[u8], NpzError> {
        if self.dtype != want {
            return err(format!("array is {:?}, read as {want:?}", self.dtype));
        }
        Ok(&self.data)
    }
}

#[cfg(test)]
#[path = "npz_tests.rs"]
mod tests;
