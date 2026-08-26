//! In-memory safetensors assembly and synthetic quantization helpers for
//! test install generators.

use compute::{f32_to_bf16, quantize_int4_affine, quantize_int8_affine};

pub(crate) struct Tensor {
    pub(crate) name: String,
    pub(crate) dtype: &'static str,
    pub(crate) shape: Vec<u64>,
    pub(crate) bytes: Vec<u8>,
}

pub(crate) fn assemble_safetensors(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut cursor = 0u64;
    for t in tensors {
        let end = cursor + t.bytes.len() as u64;
        header.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [cursor, end],
            }),
        );
        cursor = end;
    }
    let header_json = serde_json::Value::Object(header).to_string().into_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
    }
    out
}

pub(crate) fn deterministic_row(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state.wrapping_add(i as u64);
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

pub(crate) fn u16_le(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A rank-2 INT4-quantized tensor triple (weight + scales + biases) in the
/// MLX safetensors shape: `U32 [rows, cols/8]` plus `BF16 [rows, cols/64]`
/// companions. The packed bytes are this port's own nibble layout (the two
/// layouts are LE-byte identical).
pub(crate) fn int4_triple(name: &str, rows: usize, cols: usize, seed: u64) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let q = quantize_int4_affine(&deterministic_row(
            seed.wrapping_add(r as u64 * 97 + 1),
            cols,
        ));
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols / 8) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// INT8 sibling of [`int4_triple`]: `U32 [rows, cols/4]` packed bytes.
pub(crate) fn int8_triple(name: &str, rows: usize, cols: usize, seed: u64) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for r in 0..rows {
        let q = quantize_int8_affine(&deterministic_row(
            seed.wrapping_add(r as u64 * 97 + 1),
            cols,
        ));
        packed.extend_from_slice(&q.packed);
        scales.extend_from_slice(&q.scales);
        biases.extend_from_slice(&q.biases);
    }
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![rows as u64, (cols / 4) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![rows as u64, (cols / 64) as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// Rank-3 expert-major INT4 triple: `U32 [experts, rows, cols/8]`.
pub(crate) fn expert_int4_triple(
    name: &str,
    experts: usize,
    rows: usize,
    cols: usize,
    seed: u64,
) -> Vec<Tensor> {
    let base = name.strip_suffix(".weight").unwrap();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    for e in 0..experts {
        for r in 0..rows {
            let row_seed = seed.wrapping_add(e as u64 * 10_000 + r as u64 * 97 + 1);
            let q = quantize_int4_affine(&deterministic_row(row_seed, cols));
            packed.extend_from_slice(&q.packed);
            scales.extend_from_slice(&q.scales);
            biases.extend_from_slice(&q.biases);
        }
    }
    vec![
        Tensor {
            name: name.to_string(),
            dtype: "U32",
            shape: vec![experts as u64, rows as u64, (cols / 8) as u64],
            bytes: packed,
        },
        Tensor {
            name: format!("{base}.scales"),
            dtype: "BF16",
            shape: vec![experts as u64, rows as u64, (cols / 64) as u64],
            bytes: u16_le(&scales),
        },
        Tensor {
            name: format!("{base}.biases"),
            dtype: "BF16",
            shape: vec![experts as u64, rows as u64, (cols / 64) as u64],
            bytes: u16_le(&biases),
        },
    ]
}

/// A BF16 vector near `center` with small deterministic jitter.
pub(crate) fn bf16_vector(name: &str, n: usize, center: f32, seed: u64) -> Tensor {
    let jitter = deterministic_row(seed, n);
    let bits: Vec<u16> = jitter
        .iter()
        .map(|&j| f32_to_bf16(center + j * 0.05))
        .collect();
    Tensor {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![n as u64],
        bytes: u16_le(&bits),
    }
}
