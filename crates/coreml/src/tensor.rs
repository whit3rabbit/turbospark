use half::f16;

use crate::error::{CoreMlError, Result};

/// Element types the runner marshals. These are the Core ML multi-array
/// types the supported models use; anything else is refused at the
/// boundary rather than converted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    I32,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::I32 => "int32",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    I32(Vec<i32>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F16(_) => DType::F16,
            Self::I32(_) => DType::I32,
        }
    }
}

/// A dense, row-major (C order) tensor. Strided Core ML buffers are
/// gathered into this layout on the way out and scattered from it on the way
/// in, so callers never see padding.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: TensorData,
}

pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: TensorData) -> Result<Self> {
        let n = element_count(&shape);
        if n != data.len() {
            return Err(CoreMlError::Shape {
                name: "<tensor>".into(),
                expected: shape,
                got: vec![data.len()],
            });
        }
        Ok(Self { shape, data })
    }

    pub fn f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        Self::new(shape, TensorData::F32(data))
    }

    pub fn i32(shape: Vec<usize>, data: Vec<i32>) -> Result<Self> {
        Self::new(shape, TensorData::I32(data))
    }

    /// Converts to f16 and refuses any element that is not finite afterwards
    /// (|x| > 65504 overflows to inf). Most of these models run in fp16, and
    /// an inf input propagates silently through the graph.
    pub fn f16_from_f32(name: &str, shape: Vec<usize>, data: &[f32]) -> Result<Self> {
        let mut out = Vec::with_capacity(data.len());
        for (index, &x) in data.iter().enumerate() {
            let h = f16::from_f32(x);
            if !h.is_finite() {
                return Err(CoreMlError::NonFinite {
                    name: name.into(),
                    index,
                });
            }
            out.push(h);
        }
        Self::new(shape, TensorData::F16(out))
    }

    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Self {
        let n = element_count(&shape);
        let data = match dtype {
            DType::F32 => TensorData::F32(vec![0.0; n]),
            DType::F16 => TensorData::F16(vec![f16::ZERO; n]),
            DType::I32 => TensorData::I32(vec![0; n]),
        };
        Self { shape, data }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn data(&self) -> &TensorData {
        &self.data
    }

    pub fn into_data(self) -> TensorData {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            TensorData::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f16(&self) -> Option<&[f16]> {
        match &self.data {
            TensorData::F16(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<&[i32]> {
        match &self.data {
            TensorData::I32(v) => Some(v),
            _ => None,
        }
    }

    /// Widens float data to f32. Integer data is refused rather than cast,
    /// so a model whose output dtype changed fails loudly.
    pub fn to_f32_vec(&self) -> Option<Vec<f32>> {
        match &self.data {
            TensorData::F32(v) => Some(v.clone()),
            TensorData::F16(v) => Some(v.iter().map(|h| h.to_f32()).collect()),
            TensorData::I32(_) => None,
        }
    }

    /// Raw bytes in native endianness, row-major. Copies rather than
    /// reinterpreting, so this module needs no unsafe.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        match &self.data {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            TensorData::F16(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            TensorData::I32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
        }
    }

    pub(crate) fn from_bytes(dtype: DType, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        let n = element_count(&shape);
        if bytes.len() != n * dtype.size() {
            return Err(CoreMlError::Prediction(format!(
                "buffer of {} bytes does not hold {n} {dtype} elements",
                bytes.len()
            )));
        }
        let data = match dtype {
            DType::F32 => TensorData::F32(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            DType::I32 => TensorData::I32(
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            DType::F16 => TensorData::F16(
                bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_ne_bytes([c[0], c[1]]))
                    .collect(),
            ),
        };
        Ok(Self { shape, data })
    }
}

/// Byte offset span a strided view touches: the index one past the largest
/// element offset, in elements. Zero for an empty shape. Used to refuse a
/// buffer that is smaller than its declared strides imply before any copy.
pub(crate) fn strided_extent(shape: &[usize], strides: &[isize]) -> Result<usize> {
    if shape.len() != strides.len() {
        return Err(CoreMlError::Prediction(format!(
            "rank mismatch: shape {shape:?} vs strides {strides:?}"
        )));
    }
    if shape.contains(&0) {
        return Ok(0);
    }
    let mut max = 0usize;
    for (&d, &s) in shape.iter().zip(strides) {
        if s < 0 {
            return Err(CoreMlError::Prediction(format!(
                "negative stride {s} is not supported"
            )));
        }
        max = max
            .checked_add((d - 1).checked_mul(s as usize).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
    }
    Ok(max + 1)
}

fn overflow() -> CoreMlError {
    CoreMlError::Prediction("strided extent overflows usize".into())
}

/// Calls `f(dense_index, strided_offset)` for every element in row-major
/// order. Both copy directions are built on this, so the index math is
/// tested once.
pub(crate) fn for_each_offset(shape: &[usize], strides: &[usize], mut f: impl FnMut(usize, usize)) {
    let n = element_count(shape);
    if n == 0 {
        return;
    }
    let rank = shape.len();
    let mut idx = vec![0usize; rank];
    let mut off = 0usize;
    for dense in 0..n {
        f(dense, off);
        // Increment the multi-index from the innermost axis.
        for ax in (0..rank).rev() {
            idx[ax] += 1;
            off += strides[ax];
            if idx[ax] < shape[ax] {
                break;
            }
            off -= strides[ax] * shape[ax];
            idx[ax] = 0;
        }
    }
}

/// Gathers a strided buffer of `elem` byte elements into dense row-major
/// bytes. `src` must already be bounds-checked with `strided_extent`.
pub(crate) fn gather_strided(
    src: &[u8],
    shape: &[usize],
    strides: &[usize],
    elem: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; element_count(shape) * elem];
    for_each_offset(shape, strides, |dense, off| {
        out[dense * elem..(dense + 1) * elem].copy_from_slice(&src[off * elem..(off + 1) * elem]);
    });
    out
}

/// Scatters dense row-major bytes into a strided destination.
pub(crate) fn scatter_strided(
    dst: &mut [u8],
    src: &[u8],
    shape: &[usize],
    strides: &[usize],
    elem: usize,
) {
    for_each_offset(shape, strides, |dense, off| {
        dst[off * elem..(off + 1) * elem].copy_from_slice(&src[dense * elem..(dense + 1) * elem]);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of(t: &Tensor) -> Vec<u8> {
        t.to_bytes()
    }

    #[test]
    fn offsets_follow_row_major_for_contiguous_strides() {
        let mut seen = Vec::new();
        for_each_offset(&[2, 3], &[3, 1], |d, o| seen.push((d, o)));
        assert_eq!(seen, (0..6).map(|i| (i, i)).collect::<Vec<_>>());
    }

    #[test]
    fn gather_skips_padding() {
        // A [2,3] view with a row stride of 4: element 3 of each row is
        // padding the Neural Engine may add, and must not appear.
        let src = Tensor::f32(vec![8], vec![0., 1., 2., 99., 10., 11., 12., 99.]).unwrap();
        let out = gather_strided(&bytes_of(&src), &[2, 3], &[4, 1], 4);
        let t = Tensor::from_bytes(DType::F32, vec![2, 3], &out).unwrap();
        assert_eq!(t.as_f32().unwrap(), &[0., 1., 2., 10., 11., 12.]);
        assert_eq!(strided_extent(&[2, 3], &[4, 1]).unwrap(), 7);
    }

    #[test]
    fn gather_handles_transposed_strides() {
        // Column-major [2,3]: element (r,c) sits at r + 2c.
        let src = Tensor::i32(vec![6], vec![0, 3, 1, 4, 2, 5]).unwrap();
        let out = gather_strided(&bytes_of(&src), &[2, 3], &[1, 2], 4);
        let t = Tensor::from_bytes(DType::I32, vec![2, 3], &out).unwrap();
        assert_eq!(t.as_i32().unwrap(), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn scatter_inverts_gather() {
        let dense = Tensor::f32(vec![2, 3], (0..6).map(|i| i as f32).collect()).unwrap();
        let mut dst = vec![0u8; 8 * 4];
        scatter_strided(&mut dst, &bytes_of(&dense), &[2, 3], &[4, 1], 4);
        let back = gather_strided(&dst, &[2, 3], &[4, 1], 4);
        assert_eq!(back, bytes_of(&dense));
    }

    #[test]
    fn extent_rejects_negative_and_handles_empty() {
        assert!(strided_extent(&[2], &[-1]).is_err());
        assert_eq!(strided_extent(&[0, 5], &[5, 1]).unwrap(), 0);
        assert!(strided_extent(&[2, 3], &[3]).is_err());
    }

    #[test]
    fn f16_conversion_refuses_overflow_and_nan() {
        assert!(Tensor::f16_from_f32("x", vec![2], &[1.0, 65504.0]).is_ok());
        let e = Tensor::f16_from_f32("x", vec![2], &[1.0, 70000.0]).unwrap_err();
        assert_eq!(
            e,
            CoreMlError::NonFinite {
                name: "x".into(),
                index: 1
            }
        );
        assert!(Tensor::f16_from_f32("x", vec![1], &[f32::NAN]).is_err());
    }

    #[test]
    fn new_checks_element_count() {
        assert!(Tensor::f32(vec![2, 2], vec![0.0; 3]).is_err());
        assert!(Tensor::f32(vec![2, 0], vec![]).is_ok());
    }
}
