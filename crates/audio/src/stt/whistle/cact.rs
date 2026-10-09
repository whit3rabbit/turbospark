//! Bounded reader for the Needle 3 container used by Whistle.
//!
//! Reference: cactus-compute/needle at
//! af2654dffa4987f7e2cbbacdf59c2f854fd18302, needle/model/export.py.
//! CQ indices represent Hadamard-rotated weights, not affine integers.

use std::io::Read;
use std::path::Path;

use crate::{Result, SpeechError};

const HEADER_BYTES: usize = 196;
const RECORD_BYTES: usize = 44;
const MAX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ELEMENTS: usize = 1 << 27;

fn invalid(why: impl Into<String>) -> SpeechError {
    SpeechError::Tensor {
        name: "whistle.cact".into(),
        why: why.into(),
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

pub(super) fn half(value: u16) -> f32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = (value >> 10) & 31;
    let fraction = u32::from(value & 1023);
    match exponent {
        0 => {
            let value = fraction as f32 * (1.0 / 16_777_216.0);
            if sign == 0 {
                value
            } else {
                -value
            }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        _ => f32::from_bits(sign | ((u32::from(exponent) + 112) << 23) | (fraction << 13)),
    }
}

/// Normalized Walsh-Hadamard transform in the exporter's Sylvester order.
pub(super) fn hadamard(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut width = 1;
    while width < values.len() {
        for block in values.chunks_exact_mut(2 * width) {
            for i in 0..width {
                let (a, b) = (block[i], block[i + width]);
                block[i] = a + b;
                block[i + width] = a - b;
            }
        }
        width *= 2;
    }
    let scale = (values.len() as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
}

#[derive(Debug)]
pub(super) struct Header {
    pub fields: [u32; 48],
    pub rope_theta: f32,
}

#[derive(Debug)]
struct Record {
    dtype: u8,
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
    group: usize,
    bits: u32,
}

#[derive(Debug)]
pub(super) struct Archive {
    bytes: Vec<u8>,
    pub header: Header,
    records: Vec<Record>,
    codebooks: [f32; 28],
}

impl Archive {
    pub fn read(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| invalid(format!("open {}: {e}", path.display())))?;
        // A bounded read also covers a file growing after metadata was read.
        let mut bytes = Vec::new();
        std::io::Read::take(file, MAX_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| invalid(format!("read {}: {e}", path.display())))?;
        Self::parse(bytes)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < HEADER_BYTES || bytes.len() as u64 > MAX_BYTES {
            return Err(invalid("truncated header or archive exceeds 64 MiB"));
        }
        let fields = std::array::from_fn(|i| u32_at(&bytes, i * 4));
        if fields[0] != 0x05e12a84 {
            return Err(invalid("expected Needle 3 container tag 0x05e12a84"));
        }
        let count = fields[1] as usize;
        if count == 0 || count > 4096 || fields[2] != 28 {
            return Err(invalid("invalid tensor count or CQ codebook length"));
        }
        let directory = HEADER_BYTES + 28 * 4;
        let end = directory + count * RECORD_BYTES;
        if end > bytes.len() {
            return Err(invalid("truncated tensor directory"));
        }
        let codebooks =
            std::array::from_fn(|i| f32::from_bits(u32_at(&bytes, HEADER_BYTES + i * 4)));
        if codebooks.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite CQ codebook"));
        }
        let rope_theta = f32::from_bits(u32_at(&bytes, 192));
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(invalid("invalid rotary frequency base"));
        }
        let mut records = Vec::with_capacity(count);
        let mut previous_end = end;
        for index in 0..count {
            let start = directory + index * RECORD_BYTES;
            let dtype = bytes[start];
            let rank = bytes[start + 1] as usize;
            if rank > 4 || u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap()) != 0
            {
                return Err(invalid(format!(
                    "tensor {index}: invalid rank or reserved field"
                )));
            }
            let shape: Vec<usize> = (0..rank)
                .map(|i| u32_at(&bytes, start + 4 + i * 4) as usize)
                .collect();
            let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
            if shape.contains(&0) || elements.is_none_or(|n| n > MAX_ELEMENTS) {
                return Err(invalid(format!(
                    "tensor {index}: invalid or oversized shape"
                )));
            }
            let offset = usize::try_from(u64_at(&bytes, start + 20))
                .map_err(|_| invalid("tensor offset overflows address space"))?;
            let length = usize::try_from(u64_at(&bytes, start + 28))
                .map_err(|_| invalid("tensor length overflows address space"))?;
            let last = offset
                .checked_add(length)
                .filter(|&last| last <= bytes.len())
                .ok_or_else(|| invalid(format!("tensor {index}: payload outside archive")))?;
            if offset < previous_end || offset % 64 != 0 || length == 0 {
                return Err(invalid(format!(
                    "tensor {index}: overlapping, empty, or unaligned payload"
                )));
            }
            previous_end = last;
            let group = u32_at(&bytes, start + 36) as usize;
            let bits = u32_at(&bytes, start + 40);
            let elements = elements.unwrap();
            let expected = match dtype {
                1 | 2 if group == 0 && bits == 0 => {
                    elements.checked_mul(if dtype == 1 { 2 } else { 4 })
                }
                3 if rank == 2 && group == 128 && matches!(bits, 2 | 4) => {
                    let padded = shape[1].checked_add(127).map(|v| v / 128 * 128);
                    padded.and_then(|n| shape[0].checked_mul(n / 8 * bits as usize + n / 128 * 2))
                }
                4 if group == 0 && bits == 0 => Some(length),
                _ => {
                    return Err(invalid(format!(
                        "tensor {index}: unverified dtype {dtype}, CQ{bits}, group {group}"
                    )))
                }
            };
            if expected != Some(length) {
                return Err(invalid(format!(
                    "tensor {index}: payload length disagrees with shape"
                )));
            }
            records.push(Record {
                dtype,
                shape,
                offset,
                bytes: length,
                group,
                bits,
            });
        }
        Ok(Self {
            bytes,
            header: Header { fields, rope_theta },
            records,
            codebooks,
        })
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn shape(&self, index: usize) -> Result<&[usize]> {
        self.records
            .get(index)
            .map(|r| r.shape.as_slice())
            .ok_or_else(|| invalid(format!("missing tensor {index}")))
    }

    pub fn raw(&self, index: usize) -> Result<&[u8]> {
        let r = self
            .records
            .get(index)
            .ok_or_else(|| invalid(format!("missing tensor {index}")))?;
        if r.dtype != 4 {
            return Err(invalid(format!(
                "tensor {index}: expected tokenizer attachment"
            )));
        }
        Ok(&self.bytes[r.offset..r.offset + r.bytes])
    }

    pub fn floats(&self, index: usize) -> Result<Vec<f32>> {
        let r = self
            .records
            .get(index)
            .ok_or_else(|| invalid(format!("missing tensor {index}")))?;
        let raw = &self.bytes[r.offset..r.offset + r.bytes];
        let values: Vec<f32> = match r.dtype {
            1 => raw
                .chunks_exact(2)
                .map(|b| half(u16::from_le_bytes(b.try_into().unwrap())))
                .collect(),
            2 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            _ => {
                return Err(invalid(format!(
                    "tensor {index}: expected floating-point tensor"
                )))
            }
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(format!("tensor {index}: non-finite weight")));
        }
        Ok(values)
    }

    pub fn quantized(&self, index: usize) -> Result<CqMatrix> {
        let r = self
            .records
            .get(index)
            .ok_or_else(|| invalid(format!("missing tensor {index}")))?;
        if r.dtype != 3 {
            return Err(invalid(format!("tensor {index}: expected CQ matrix")));
        }
        let (rows, cols) = (r.shape[0], r.shape[1]);
        let padded = cols.div_ceil(r.group) * r.group;
        let packed_len = rows * padded / 8 * r.bits as usize;
        let raw = &self.bytes[r.offset..r.offset + r.bytes];
        let norms: Vec<f32> = raw[packed_len..]
            .chunks_exact(2)
            .map(|b| half(u16::from_le_bytes(b.try_into().unwrap())))
            .collect();
        if norms.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(invalid(format!("tensor {index}: invalid CQ norm")));
        }
        let (start, count) = if r.bits == 2 { (0, 4) } else { (12, 16) };
        let mut codebook = [0.0; 16];
        codebook[..count].copy_from_slice(&self.codebooks[start..start + count]);
        Ok(CqMatrix {
            rows,
            cols,
            padded,
            bits: r.bits,
            packed: raw[..packed_len].to_vec(),
            norms,
            codebook,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct CqMatrix {
    pub rows: usize,
    pub cols: usize,
    padded: usize,
    bits: u32,
    packed: Vec<u8>,
    norms: Vec<f32>,
    codebook: [f32; 16],
}

impl CqMatrix {
    /// Apply the compressed matrix without expanding its embedding tables.
    /// A8 is applied after rotation, as in the released CPU runtime.
    pub fn project(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len() % self.cols, 0);
        let mut output = vec![0.0; input.len() / self.cols * self.rows];
        let mut rotated = vec![0.0; self.padded];
        let groups = self.padded / 128;
        let row_bytes = self.padded / 8 * self.bits as usize;
        let peak = self.codebook[..1 << self.bits]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let weight_scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
        let codebook: [i32; 16] = self.codebook.map(|v| (v / weight_scale).round() as i32);
        let mut activation = vec![0i32; self.padded];
        let mut scales = vec![0.0; groups];
        for (t, x) in input.chunks_exact(self.cols).enumerate() {
            rotated.fill(0.0);
            rotated[..self.cols].copy_from_slice(x);
            for (g, block) in rotated.chunks_exact_mut(128).enumerate() {
                hadamard(block);
                let peak = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
                scales[g] = scale;
                for (j, v) in block.iter().enumerate() {
                    activation[g * 128 + j] = (*v / scale).round().clamp(-127.0, 127.0) as i32;
                }
            }
            for row in 0..self.rows {
                let packed = &self.packed[row * row_bytes..(row + 1) * row_bytes];
                let mut sum = 0.0;
                for (g, &scale) in scales.iter().enumerate() {
                    let mut dot = 0i32;
                    for j in 0..128 {
                        let column = g * 128 + j;
                        let bit = column * self.bits as usize;
                        let index = (packed[bit / 8] >> (bit % 8)) & ((1 << self.bits) - 1);
                        dot += activation[column] * codebook[index as usize];
                    }
                    sum += dot as f32 * scale * weight_scale * self.norms[row * groups + g];
                }
                output[t * self.rows + row] = sum;
            }
        }
        output
    }

    /// Decode one row, chiefly for token and n-gram embedding gathers.
    pub fn row(&self, row: usize) -> Vec<f32> {
        assert!(row < self.rows);
        let mut values = vec![0.0; self.padded];
        let row_bytes = self.padded / 8 * self.bits as usize;
        let packed = &self.packed[row * row_bytes..(row + 1) * row_bytes];
        for (g, block) in values.chunks_exact_mut(128).enumerate() {
            let norm = self.norms[row * (self.padded / 128) + g];
            for (j, value) in block.iter_mut().enumerate() {
                let column = g * 128 + j;
                let bit = column * self.bits as usize;
                let index = (packed[bit / 8] >> (bit % 8)) & ((1 << self.bits) - 1);
                *value = self.codebook[index as usize] * norm;
            }
            hadamard(block);
        }
        values.truncate(self.cols);
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive(dtype: u8, shape: &[u32], payload: &[u8], bits: u32) -> Vec<u8> {
        let mut b = vec![0u8; 384 + payload.len()];
        for (offset, value) in [
            (0, 0x05e12a84u32),
            (4, 1),
            (8, 28),
            (192, 100000.0f32.to_bits()),
        ] {
            b[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        let start = 308;
        b[start] = dtype;
        b[start + 1] = shape.len() as u8;
        for (i, value) in shape.iter().enumerate() {
            b[start + 4 + i * 4..start + 8 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        b[start + 20..start + 28].copy_from_slice(&384u64.to_le_bytes());
        b[start + 28..start + 36].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        if dtype == 3 {
            b[start + 36..start + 40].copy_from_slice(&128u32.to_le_bytes());
            b[start + 40..start + 44].copy_from_slice(&bits.to_le_bytes());
        }
        b[384..].copy_from_slice(payload);
        b
    }

    #[test]
    fn reads_half_subnormals_and_f32_without_native_endian_assumptions() {
        assert_eq!(half(0x0001), 1.0 / 16_777_216.0);
        assert_eq!(half(0x8001), -1.0 / 16_777_216.0);
        let a = Archive::parse(archive(1, &[2], &[0, 0x3c, 0, 0xc0], 0)).unwrap();
        assert_eq!(a.floats(0).unwrap(), [1.0, -2.0]);
        let a = Archive::parse(archive(2, &[1], &3.5f32.to_le_bytes(), 0)).unwrap();
        assert_eq!(a.floats(0).unwrap(), [3.5]);
    }

    #[test]
    fn refuses_corrupt_directory_bounds_shapes_and_unverified_quantization() {
        let base = archive(1, &[1], &[0, 0x3c], 0);
        for offset in [0, 4, 8, 309, 312, 328, 336, 344] {
            let mut changed = base.clone();
            changed[offset..offset + 4].fill(0xff);
            assert!(Archive::parse(changed).is_err(), "mutation at {offset}");
        }
        assert!(Archive::parse(base[..base.len() - 1].to_vec()).is_err());
        assert!(Archive::parse(archive(3, &[1, 128], &[0; 50], 3)).is_err());
        let a = Archive::parse(archive(1, &[1], &[0, 0x7c], 0)).unwrap();
        assert!(a.floats(0).is_err());
    }

    #[test]
    fn cq_indices_are_lsb_first_and_inverse_rotated() {
        let mut payload = vec![0u8; 34];
        // All code 0 except code 1 at coordinate 1. A single rotated
        // coordinate must spread to alternating signs in the decoded row.
        payload[0] = 1 << 2;
        payload[32..].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut b = archive(3, &[1, 128], &payload, 2);
        b[200..204].copy_from_slice(&1.0f32.to_le_bytes());
        let matrix = Archive::parse(b).unwrap().quantized(0).unwrap();
        let row = matrix.row(0);
        for (i, value) in row.iter().enumerate() {
            let expected = if i % 2 == 0 {
                128.0f32.sqrt().recip()
            } else {
                -128.0f32.sqrt().recip()
            };
            assert!((value - expected).abs() < 1e-7);
        }
    }
}
