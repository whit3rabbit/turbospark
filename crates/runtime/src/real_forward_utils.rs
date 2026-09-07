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
    let group_size = affine_group_size(entry, name, rows, cols, 4)?;
    let groups = cols / group_size;
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

/// How a `rows`-row write starting at `base` splits across a RING cache's
/// wrap: `[(row offset within the write, row count); 2]`, the second span
/// empty whenever the write does not straddle.
///
/// `KvCacheManager::k_slot` addresses `position % capacity` and validates
/// ONE row, while a batched projection hands it `rows` ADJACENT slots -- so
/// a straddling write runs past the layer's buffer with no assertion in the
/// way. Splitting costs one extra dispatch per projection on the one write
/// that straddles and is a no-op on every other one (`spans[1].1 == 0`, and
/// `spans[0]` is exactly the single call that would otherwise be made).
///
/// **TWO CALLERS, AND NEITHER CAN REFUSE INSTEAD.** The DFlash2 drafter's
/// cache is a real ring of `DFLASH_WINDOW + DFLASH_RING_SLACK`, so it wraps
/// every 2,176 positions and refusing would end an ordinary generation. The
/// chunked-prefill driver's batched attention projections write Gemma 4's
/// sliding-window rings, which wrap every `sliding_window +
/// MAX_PREFILL_CHUNK_TOKENS` positions. `produce_batched` on the `qwen3_5`
/// trunk is the one place that DOES refuse, because its full layers wrap
/// only at `max_context`.
///
/// It is written for the ring case and is correct for a LINEAR layer too:
/// `physical_slot` is `position % capacity` whatever the layer kind, so a
/// linear layer simply never reaches the second span.
pub(crate) fn ring_spans(capacity: usize, base: usize, rows: usize) -> [(usize, usize); 2] {
    assert!(
        capacity > 0 && rows <= capacity,
        "capacity must be positive and rows <= capacity: rows={rows}, capacity={capacity}"
    );
    let first = rows.min(capacity - base % capacity);
    [(0, first), (first, rows - first)]
}

#[cfg(test)]
#[path = "real_forward_utils_tests.rs"]
mod tests;
