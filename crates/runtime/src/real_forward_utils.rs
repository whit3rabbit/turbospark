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

/// The GROUP SIZE of a sub-4-bit affine tensor, read off its own companion
/// planes rather than recalled from the checkpoint that motivated the type.
///
/// The resident index records no group size, and the obvious alternative was
/// a `BONSAI_GROUP_SIZE = 128` here. That is exactly the shape AGENTS.md
/// Gotchas 37 and 38 are both instances of -- a per-checkpoint number
/// standing in for a per-tensor property, correct for as long as one
/// checkpoint exercises it -- and it is unnecessary, because the entry states
/// the answer: one FP16 scale per group per row, so
/// `groups_per_row = scale_size / (2 * rows)`. `crates/gpu` declined the same
/// constant for the same reason and takes the value as an argument.
///
/// Every conjunct below is also a check on the install: the two companion
/// planes must agree in size, the packed run must be exactly `bits` per
/// element, and the group must divide the row and be a whole number of bytes
/// (the kernel's own precondition, which it ASSERTS -- so a malformed install
/// that reached the dispatch would abort the process rather than return an
/// error).
///
/// **`bits` is a PARAMETER and the packed-size conjunct is the only thing it
/// changes**, which is why the ternary entry widened this function rather than
/// copying it: everything else here is a statement about the companion planes,
/// and the companions are FP16 at one group per row at either width. The
/// packed check is what tells a 1-bit tensor from a 2-bit one, since the two
/// entry SHAPES are identical.
pub(crate) fn affine_group_size(
    e: &model_io::ResidentIndexEntry,
    name: &str,
    rows: usize,
    cols: usize,
    bits: usize,
) -> Result<usize, RealForwardError> {
    let bad = |detail: String| Err(RealForwardError::Unsupported(detail));
    if rows == 0 || cols == 0 {
        return bad(format!(
            "tensor {name}: {bits}-bit shape {rows}x{cols} has a zero dimension"
        ));
    }
    let elements_per_byte = 8 / bits;
    if (rows * cols) % elements_per_byte != 0
        || e.size_bytes as usize != rows * cols / elements_per_byte
    {
        return bad(format!(
            "tensor {name}: {bits}-bit packed size {} does not match {rows}x{cols} ({} bytes)",
            e.size_bytes,
            rows * cols / elements_per_byte
        ));
    }
    if e.scale_size != e.bias_size {
        return bad(format!(
            "tensor {name}: {bits}-bit scale plane is {} bytes against {} bias bytes; the two \
             carry one FP16 value per group each",
            e.scale_size, e.bias_size
        ));
    }
    let plane = e.scale_size as usize;
    if plane == 0 || plane % (2 * rows) != 0 {
        return bad(format!(
            "tensor {name}: {plane} companion bytes is not a whole number of FP16 values across \
             {rows} rows"
        ));
    }
    let groups = plane / (2 * rows);
    if cols % groups != 0 {
        return bad(format!(
            "tensor {name}: {groups} groups per row does not divide {cols} columns"
        ));
    }
    let group_size = cols / groups;
    if group_size % elements_per_byte != 0 {
        return bad(format!(
            "tensor {name}: derived group size {group_size} is not a whole number of bytes"
        ));
    }
    Ok(group_size)
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

#[cfg(test)]
mod tests {
    use super::affine_group_size;
    use model_io::ResidentIndexEntry;

    /// The real checkpoints' shape, in miniature: `bits` per element, one FP16
    /// scale and one FP16 bias per group.
    fn entry(rows: usize, cols: usize, group: usize, bits: usize) -> ResidentIndexEntry {
        let groups = rows * (cols / group);
        ResidentIndexEntry {
            name: "w".to_string(),
            dtype: if bits == 1 { 15 } else { 16 },
            file_offset: 4096,
            size_bytes: (rows * cols * bits / 8) as u64,
            shape: (rows as u32, cols as u32, 0, 0),
            scale_offset: 8192,
            scale_size: (groups * 2) as u64,
            bias_offset: 16384,
            bias_size: (groups * 2) as u64,
        }
    }

    /// THE POINT OF THE DERIVATION: the group size comes off the tensor, so
    /// two tensors in one install may disagree and neither has to match a
    /// constant somebody wrote down. 128 is the published checkpoints'; 64 is
    /// what a per-checkpoint constant would have forced on this row.
    #[test]
    fn the_group_size_is_read_off_the_companion_planes() {
        assert_eq!(
            affine_group_size(&entry(8, 256, 128, 1), "w", 8, 256, 1).unwrap(),
            128
        );
        assert_eq!(
            affine_group_size(&entry(8, 256, 64, 1), "w", 8, 256, 1).unwrap(),
            64
        );
        assert_eq!(
            affine_group_size(&entry(3, 384, 128, 1), "w", 3, 384, 1).unwrap(),
            128
        );
    }

    /// The same derivation at TWO bits, where only the packed run moves.
    ///
    /// The pair of assertions is the point: identical companion planes and a
    /// packed run of exactly twice the size yield the same group size, which
    /// is what says `bits` reaches the one conjunct it should and no other.
    #[test]
    fn the_derivation_takes_the_width_as_a_parameter() {
        assert_eq!(
            affine_group_size(&entry(8, 256, 128, 2), "w", 8, 256, 2).unwrap(),
            128
        );
        // ...and each width REFUSES the other's packed run, which is the only
        // thing that tells the two entry shapes apart.
        assert!(affine_group_size(&entry(8, 256, 128, 1), "w", 8, 256, 2).is_err());
        assert!(affine_group_size(&entry(8, 256, 128, 2), "w", 8, 256, 1).is_err());
    }

    /// Each conjunct is also a check on the install, and this is the one that
    /// matters most: the two planes are the same width, so a companion region
    /// that is half the size it should be passes every other length check.
    #[test]
    fn a_malformed_sub_four_bit_entry_is_refused_rather_than_dispatched() {
        let mut half_scales = entry(8, 256, 128, 1);
        half_scales.scale_size /= 2;
        assert!(affine_group_size(&half_scales, "w", 8, 256, 1).is_err());

        let mut no_companions = entry(8, 256, 128, 1);
        no_companions.scale_size = 0;
        no_companions.bias_size = 0;
        assert!(affine_group_size(&no_companions, "w", 8, 256, 1).is_err());

        // A shape the caller and the file disagree about.
        assert!(affine_group_size(&entry(8, 256, 128, 1), "w", 4, 256, 1).is_err());

        // A group that is not a whole number of bytes: the kernel ASSERTS
        // this, so reaching it would abort the process rather than error.
        let ragged = entry(1, 12, 4, 1);
        assert!(affine_group_size(&ragged, "w", 1, 12, 1).is_err());
    }
}
