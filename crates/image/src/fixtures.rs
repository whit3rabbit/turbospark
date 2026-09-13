use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Loaded NPY array data with shape.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> NpyArray<T> {
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape.iter().product()
        }
    }
}

/// Parse NPY header, returning (descr, fortran_order, shape, data_offset).
fn parse_npy_header(bytes: &[u8]) -> Result<(String, bool, Vec<usize>, usize), String> {
    if bytes.len() < 10 {
        return Err("NPY bytes too short for header".to_string());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("invalid NPY magic number".to_string());
    }

    let major = bytes[6];
    let (_header_len, data_offset) = if major == 1 {
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (hlen, 10 + hlen)
    } else if major == 2 {
        if bytes.len() < 12 {
            return Err("NPY v2 bytes too short for header length".to_string());
        }
        let hlen = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (hlen, 12 + hlen)
    } else {
        return Err(format!("unsupported NPY version {major}"));
    };

    if bytes.len() < data_offset {
        return Err("NPY file truncated before end of header".to_string());
    }

    let header_start = if major == 1 { 10 } else { 12 };
    let header_str = std::str::from_utf8(&bytes[header_start..data_offset])
        .map_err(|e| format!("invalid UTF-8 in NPY header: {e}"))?;

    // Parse descr
    let descr_start = header_str
        .find("'descr'")
        .ok_or_else(|| "missing 'descr' in NPY header".to_string())?;
    let descr_substr = &header_str[descr_start..];
    let q1 = descr_substr
        .find('\'')
        .and_then(|i| descr_substr[i + 1..].find('\'').map(|j| i + 1 + j + 1))
        .ok_or_else(|| "cannot find descr opening quote".to_string())?;
    let descr_rest = &descr_substr[q1..];
    let q_open = descr_rest
        .find('\'')
        .ok_or_else(|| "cannot find descr value start".to_string())?;
    let q_close = descr_rest[q_open + 1..]
        .find('\'')
        .ok_or_else(|| "cannot find descr value end".to_string())?;
    let descr = descr_rest[q_open + 1..q_open + 1 + q_close].to_string();

    // Parse fortran_order
    let fortran_order = if header_str.contains("'fortran_order': True") {
        true
    } else if header_str.contains("'fortran_order': False") {
        false
    } else {
        return Err("cannot parse fortran_order in NPY header".to_string());
    };

    // Parse shape
    let shape_start = header_str
        .find("'shape'")
        .ok_or_else(|| "missing 'shape' in NPY header".to_string())?;
    let shape_substr = &header_str[shape_start..];
    let paren_open = shape_substr
        .find('(')
        .ok_or_else(|| "missing '(' in shape".to_string())?;
    let paren_close = shape_substr
        .find(')')
        .ok_or_else(|| "missing ')' in shape".to_string())?;
    let tuple_content = &shape_substr[paren_open + 1..paren_close];

    let mut shape = Vec::new();
    for part in tuple_content.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let dim: usize = trimmed
                .parse()
                .map_err(|e| format!("invalid shape dimension '{trimmed}': {e}"))?;
            shape.push(dim);
        }
    }

    Ok((descr, fortran_order, shape, data_offset))
}

/// Read NPY file containing float32 (<f4) data.
pub fn read_npy_f32(bytes: &[u8]) -> Result<NpyArray<f32>, String> {
    let (descr, fortran_order, shape, offset) = parse_npy_header(bytes)?;
    if descr != "<f4" && descr != "f4" {
        return Err(format!("expected <f4 dtype, got {descr}"));
    }
    if fortran_order {
        return Err("fortran_order=True is not supported".to_string());
    }

    let expected_count: usize = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };
    let data_bytes = &bytes[offset..];
    if data_bytes.len() != expected_count * 4 {
        return Err(format!(
            "payload length mismatch: expected {} bytes ({} floats), got {}",
            expected_count * 4,
            expected_count,
            data_bytes.len()
        ));
    }

    let mut data = Vec::with_capacity(expected_count);
    for chunk in data_bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    Ok(NpyArray { shape, data })
}

/// Read NPY file containing int64 (<i8) data.
pub fn read_npy_i64(bytes: &[u8]) -> Result<NpyArray<i64>, String> {
    let (descr, fortran_order, shape, offset) = parse_npy_header(bytes)?;
    if descr != "<i8" && descr != "i8" {
        return Err(format!("expected <i8 dtype, got {descr}"));
    }
    if fortran_order {
        return Err("fortran_order=True is not supported".to_string());
    }

    let expected_count: usize = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };
    let data_bytes = &bytes[offset..];
    if data_bytes.len() != expected_count * 8 {
        return Err(format!(
            "payload length mismatch: expected {} bytes ({} i64s), got {}",
            expected_count * 8,
            expected_count,
            data_bytes.len()
        ));
    }

    let mut data = Vec::with_capacity(expected_count);
    for chunk in data_bytes.chunks_exact(8) {
        data.push(i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }

    Ok(NpyArray { shape, data })
}

/// Read NPY file containing bool (|b1 or b1) data.
pub fn read_npy_bool(bytes: &[u8]) -> Result<NpyArray<bool>, String> {
    let (descr, fortran_order, shape, offset) = parse_npy_header(bytes)?;
    if descr != "|b1" && descr != "b1" {
        return Err(format!("expected |b1 or b1 dtype, got {descr}"));
    }
    if fortran_order {
        return Err("fortran_order=True is not supported".to_string());
    }

    let expected_count: usize = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };
    let data_bytes = &bytes[offset..];
    if data_bytes.len() != expected_count {
        return Err(format!(
            "payload length mismatch: expected {} bytes ({} bools), got {}",
            expected_count,
            expected_count,
            data_bytes.len()
        ));
    }

    let data: Vec<bool> = data_bytes.iter().map(|&b| b != 0).collect();
    Ok(NpyArray { shape, data })
}

/// Read NPY file containing complex64 (<c8 or c8) data as (real, imag) f32 pairs.
pub fn read_npy_complex64(bytes: &[u8]) -> Result<NpyArray<(f32, f32)>, String> {
    let (descr, fortran_order, shape, offset) = parse_npy_header(bytes)?;
    if descr != "<c8" && descr != "c8" {
        return Err(format!("expected <c8 dtype, got {descr}"));
    }
    if fortran_order {
        return Err("fortran_order=True is not supported".to_string());
    }

    let expected_count: usize = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };
    let data_bytes = &bytes[offset..];
    if data_bytes.len() != expected_count * 8 {
        return Err(format!(
            "payload length mismatch: expected {} bytes ({} complex pairs), got {}",
            expected_count * 8,
            expected_count,
            data_bytes.len()
        ));
    }

    let mut data = Vec::with_capacity(expected_count);
    for chunk in data_bytes.chunks_exact(8) {
        let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let i = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        data.push((r, i));
    }

    Ok(NpyArray { shape, data })
}

/// Read an uncompressed .npz ZIP archive into a map of filename -> payload bytes.
pub fn read_npz(bytes: &[u8]) -> Result<std::collections::BTreeMap<String, Vec<u8>>, String> {
    let mut entries = std::collections::BTreeMap::new();
    let mut offset = 0;
    while offset + 30 <= bytes.len() {
        if &bytes[offset..offset + 4] != b"PK\x03\x04" {
            break;
        }
        let method = u16::from_le_bytes([bytes[offset + 8], bytes[offset + 9]]);
        if method != 0 {
            return Err(format!(
                "unsupported compression method {method} in npz entry at {offset}"
            ));
        }
        let mut comp_size = u32::from_le_bytes([
            bytes[offset + 18],
            bytes[offset + 19],
            bytes[offset + 20],
            bytes[offset + 21],
        ]) as usize;
        let fn_len = u16::from_le_bytes([bytes[offset + 26], bytes[offset + 27]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[offset + 28], bytes[offset + 29]]) as usize;

        if offset + 30 + fn_len + extra_len > bytes.len() {
            return Err("npz local header overflows archive buffer".to_string());
        }

        let filename = std::str::from_utf8(&bytes[offset + 30..offset + 30 + fn_len])
            .map_err(|e| format!("invalid UTF-8 in zip filename: {e}"))?
            .to_string();

        let extra = &bytes[offset + 30 + fn_len..offset + 30 + fn_len + extra_len];
        if comp_size == 0xFFFF_FFFF {
            let mut ei = 0;
            let mut found = false;
            while ei + 4 <= extra.len() {
                let tag = u16::from_le_bytes([extra[ei], extra[ei + 1]]);
                let esz = u16::from_le_bytes([extra[ei + 2], extra[ei + 3]]) as usize;
                if tag == 1 && esz >= 16 && ei + 4 + esz <= extra.len() {
                    let csz = u64::from_le_bytes([
                        extra[ei + 12],
                        extra[ei + 13],
                        extra[ei + 14],
                        extra[ei + 15],
                        extra[ei + 16],
                        extra[ei + 17],
                        extra[ei + 18],
                        extra[ei + 19],
                    ]);
                    comp_size = usize::try_from(csz)
                        .map_err(|_| "zip64 comp_size overflows usize".to_string())?;
                    found = true;
                    break;
                }
                ei += 4 + esz;
            }
            if !found {
                return Err(format!(
                    "missing ZIP64 extra record for 0xFFFFFFFF size in entry {filename}"
                ));
            }
        }

        let data_start = offset + 30 + fn_len + extra_len;
        let data_end = data_start
            .checked_add(comp_size)
            .ok_or_else(|| "entry size overflows usize".to_string())?;
        if data_end > bytes.len() {
            return Err(format!(
                "entry {filename} extends past end of zip buffer: {data_end} > {}",
                bytes.len()
            ));
        }

        entries.insert(filename, bytes[data_start..data_end].to_vec());
        offset = data_end;
    }

    Ok(entries)
}

/// Read an uncompressed .npz ZIP archive from a file path.
pub fn read_npz_file(path: &Path) -> Result<std::collections::BTreeMap<String, Vec<u8>>, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    read_npz(&bytes)
}

/// Read NPY f32 from a file path.
pub fn read_npy_file_f32(path: &Path) -> Result<NpyArray<f32>, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    read_npy_f32(&bytes)
}

/// Read NPY i64 from a file path.
pub fn read_npy_file_i64(path: &Path) -> Result<NpyArray<i64>, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    read_npy_i64(&bytes)
}

/// Read NPY bool from a file path.
pub fn read_npy_file_bool(path: &Path) -> Result<NpyArray<bool>, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    read_npy_bool(&bytes)
}

/// Read NPY complex64 from a file path.
pub fn read_npy_file_complex64(path: &Path) -> Result<NpyArray<(f32, f32)>, String> {
    let mut file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    read_npy_complex64(&bytes)
}

/// Resolve the Z-Image fixture root honoring Z_IMAGE_TEST_ROOT.
pub fn fixture_root() -> PathBuf {
    if let Ok(var) = std::env::var("Z_IMAGE_TEST_ROOT") {
        PathBuf::from(var)
    } else {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

/// Path to run array file in target/ig0/runs/<case>/<filename>.
pub fn run_array_path(case: &str, filename: &str) -> PathBuf {
    fixture_root()
        .join("target")
        .join("ig0")
        .join("runs")
        .join(case)
        .join(filename)
}

/// Path to committed capture manifest in docs/verification/z-image-ig0-captures/<case>/<filename>.
pub fn capture_manifest_path(case: &str, filename: &str) -> PathBuf {
    fixture_root()
        .join("docs")
        .join("verification")
        .join("z-image-ig0-captures")
        .join(case)
        .join(filename)
}

/// Path to local model checkout under target/ig0/model/<subpath>.
pub fn model_subpath(subpath: &str) -> PathBuf {
    fixture_root()
        .join("target")
        .join("ig0")
        .join("model")
        .join(subpath)
}

/// Path to contracts JSON in docs/verification/z-image-ig0-contracts.json.
pub fn contracts_json_path() -> PathBuf {
    fixture_root()
        .join("docs")
        .join("verification")
        .join("z-image-ig0-contracts.json")
}
