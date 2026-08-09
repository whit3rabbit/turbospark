//! Low-level byte packing, float conversions, resident matrix resolution,
//! and host tensor helpers shared across real forward execution passes.

use half::f16;
use model_io::ResidentIndex;

use crate::real_forward_types::RealForwardError;

pub(crate) fn le_bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

pub(crate) fn f32_to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

pub(crate) fn f16_to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

pub(crate) fn f16_slice_to_le_bytes(v: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    out
}

pub(crate) fn tensor_bytes<'a>(
    index: &'a ResidentIndex,
    data: &'a [u8],
    name: &str,
) -> Result<&'a [u8], RealForwardError> {
    let entry = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    let local = (entry.file_offset - index.header.index_size) as usize;
    Ok(&data[local..local + entry.size_bytes as usize])
}

/// Resolves `name` to an offset-bound matrix view into the one shared
/// resident `MTLBuffer` -- the zero-copy binding every GPU projection
/// dispatches against. Validates the entry's packed size against the
/// caller's expected shape (the offsets are trusted after that; the index
/// was already bounds-validated at load).
pub(crate) fn resident_matrix<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<gpu::Int4ResidentMatrix<'a>, RealForwardError> {
    let entry = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    if entry.size_bytes as usize != rows * cols / 2 {
        return Err(RealForwardError::Unsupported(format!(
            "tensor {name}: packed size {} does not match shape {rows}x{cols}",
            entry.size_bytes
        )));
    }
    let base = index.header.index_size;
    Ok(gpu::Int4ResidentMatrix {
        buffer: weights.buffer(),
        weights_offset: weights.gpu_offset(entry.file_offset - base),
        scales_offset: weights.gpu_offset(entry.scale_offset - base),
        biases_offset: weights.gpu_offset(entry.bias_offset - base),
        rows,
        cols,
    })
}

/// Host copies for the CPU FFN bridge only (`compute::run_ffn` has no GPU
/// counterpart yet -- see module docs). Converts the tensor's BF16
/// scale/bias bytes on each call; goes away with the GPU FFN/MoE kernels.
pub(crate) fn owned_rows(
    index: &ResidentIndex,
    data: &[u8],
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<compute::quant::Int4AffineRow>, RealForwardError> {
    let packed_all = tensor_bytes(index, data, name)?;
    let entry = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    let base = index.header.index_size;
    let scale_local = (entry.scale_offset - base) as usize;
    let bias_local = (entry.bias_offset - base) as usize;
    let scales = le_bytes_to_u16(&data[scale_local..scale_local + entry.scale_size as usize]);
    let biases = le_bytes_to_u16(&data[bias_local..bias_local + entry.bias_size as usize]);
    let row_bytes = cols / 2;
    let groups = cols / 64;
    Ok((0..rows)
        .map(|r| compute::quant::Int4AffineRow {
            packed: packed_all[r * row_bytes..(r + 1) * row_bytes].to_vec(),
            scales: scales[r * groups..(r + 1) * groups].to_vec(),
            biases: biases[r * groups..(r + 1) * groups].to_vec(),
        })
        .collect())
}

pub(crate) fn layer_name(prefix: &str, layer: usize) -> String {
    format!("layer{layer}.{prefix}")
}

pub(crate) fn layer_tensor(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

pub(crate) fn entry<'a>(
    index: &'a ResidentIndex,
    name: &str,
) -> Result<&'a model_io::ResidentIndexEntry, RealForwardError> {
    index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))
}

/// Decodes a resident BF16 tensor to `f32` host values.
pub(crate) fn read_bf16_host(
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
) -> Result<Vec<f32>, RealForwardError> {
    let e = entry(index, name)?;
    let base = index.header.index_size;
    let local = (e.file_offset - base) as usize;
    let bytes = &weights.data()[local..local + e.size_bytes as usize];
    Ok(bytes
        .chunks_exact(2)
        .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Resolves a raw (norm) tensor to a `(buffer, gpu offset)` view, checking
/// its byte size against the expected element count.
pub(crate) fn norm_view<'a>(
    weights: &'a gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    name: &str,
    expect_elems: usize,
) -> Result<(&'a gpu::MetalBuffer, u64), RealForwardError> {
    let e = entry(index, name)?;
    if e.size_bytes as usize != expect_elems * 2 {
        return Err(RealForwardError::Unsupported(format!(
            "tensor {name}: {} bytes does not match expected BF16 [{expect_elems}]",
            e.size_bytes
        )));
    }
    let base = index.header.index_size;
    Ok((weights.buffer(), weights.gpu_offset(e.file_offset - base)))
}

/// Softmax over all `logits`, then the top-`k` entries with their
/// probabilities renormalized to sum to `1` over just the survivors
/// (Gemma 4's `router_scaled` convention). Plain host arithmetic -- cheap
/// enough at any real `num_experts` count that no kernel is warranted.
pub(crate) fn topk_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let selected: Vec<usize> = order.into_iter().take(k).collect();

    let selected_sum: f32 = selected.iter().map(|&i| probs[i]).sum();
    let weights: Vec<f32> = selected
        .iter()
        .map(|&i| {
            if selected_sum > 0.0 {
                probs[i] / selected_sum
            } else {
                0.0
            }
        })
        .collect();
    (selected, weights)
}
