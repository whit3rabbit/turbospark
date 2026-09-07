//! CPU reference for `qwen4_exp`'s PLE (per-layer n-gram embedding) gate
//! (`docs/QWEN4_PHASE0.md` section 4). PORT-LOCAL: the Swift engine has no
//! architecture with this mechanism.

/// NumPy/PyTorch/Metal's `sign(x)`: `-1` for negative, `0` for exactly zero
/// (including `-0.0`), `+1` for positive.
///
/// **Deliberately NOT `f32::signum`.** That function's documented behaviour
/// returns `1.0` at `+0.0` (Rust's own docs: "1.0 if the number is positive,
/// +0.0 ..."), which is a real divergence from the reference's `sign(0) ==
/// 0` -- silent everywhere the dot product this feeds is nonzero, and wrong
/// exactly where it is not. Metal's own `sign()` builtin returns `+0.0` /
/// `-0.0` at zero (matching this), so the kernel can use it directly; the
/// CPU side cannot reach for the analogous Rust builtin and needs this
/// instead.
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `gv[c*H+h] = sigmoid(gate[c]) * value[h]`, where
/// `gate[c] = sign(dot[c]) * sqrt(max(|dot[c]|, 1e-6))` and
/// `dot[c] = sum_h(key[c*H+h] * query[c*H+h]) / sqrt(H)`.
///
/// The contract for `ple_gate_fp16`. `key`/`query` are `[groups * H]`,
/// ALREADY grouped-centered-normed (`rms_norm_grouped_centered`, applied to
/// `key_proj(emb)` and the hidden residual respectively BEFORE this
/// function runs -- this function does not norm them again); `value` is
/// `[H]`, `value_proj(emb)`'s raw output with no norm at all.
///
/// **THE SIGNED SQRT IS DELIBERATE, NOT A TYPO**
/// (`docs/QWEN4_PHASE0.md` section 4's own words). A plain
/// `sqrt(dot / sqrt(H))` on a negative `dot` produces NaN instead of a
/// negative gate, which is Gotcha 59's failure mode -- NaN then reads as a
/// perfect score on every downstream rank instrument.
pub fn ple_gate(key: &[f32], query: &[f32], value: &[f32], groups: usize) -> Vec<f32> {
    let h = value.len();
    assert_eq!(key.len(), groups * h, "key must be groups * H");
    assert_eq!(query.len(), key.len(), "query must match key's length");
    let sqrt_h = (h as f32).sqrt();
    let mut gv = vec![0.0f32; groups * h];
    for c in 0..groups {
        let base = c * h;
        let dot: f32 = (0..h).map(|i| key[base + i] * query[base + i]).sum();
        let scaled = dot / sqrt_h;
        let gate = sign(scaled) * scaled.abs().max(1e-6).sqrt();
        let sig = 1.0 / (1.0 + (-gate).exp());
        for i in 0..h {
            gv[base + i] = sig * value[i];
        }
    }
    gv
}

/// One step of the DILATED depthwise causal conv `qwen4_exp`'s PLE module
/// applies to `norm_conv(gv).flatten()` (`docs/QWEN4_PHASE0.md` section 4):
/// `kernel_size=4`, `dilation=ngram_size=3`, so its history is `(4-1)*3 = 9`
/// rows, not `kernel_size - 1`. The contract for
/// `gdn_conv_mix_decode`/`gdn_conv_mix_prefill`/`gdn_conv_tail_update` at
/// `dilation > 1` -- see those kernels' own doc comment for why `dilation=1`
/// reproduces the plain (undilated) GDN causal conv exactly.
///
/// **A STANDALONE FUNCTION, DELIBERATELY NOT THREADED INTO
/// `GdnReference::step`.** Same reason [`gated_norm_sigmoid`](crate::gdn::gated_norm_sigmoid)
/// and [`ple_gate`] are standalone: that struct and its one call site are
/// what the already-verified `qwen3_5`/`qwen3_6` GDN chain runs through end
/// to end, at `dilation=1` implicitly, and `crates/runtime` Gotcha 11
/// records exactly the failure mode of touching shared flow code for what
/// looks like a no-op change.
///
/// `tail` holds `(k - 1) * dilation` PAST raw rows, oldest first, one row
/// per TIMESTEP (never one row per tap -- the taps read every
/// `dilation`-th row of it). `conv_w` is `[channels, k]`, tap-minor.
/// Returns `silu(conv)` and shifts `tail` in place (drops the oldest row,
/// appends `raw`), mirroring `gdn_conv_mix_decode`'s exact contract.
pub fn dilated_conv_step(
    tail: &mut Vec<Vec<f32>>,
    raw: &[f32],
    conv_w: &[f32],
    channels: usize,
    k: usize,
    dilation: usize,
) -> Vec<f32> {
    assert!(k >= 1, "k must be at least 1");
    assert!(dilation >= 1, "dilation must be at least 1");
    assert_eq!(raw.len(), channels, "raw must be one row of `channels`");
    assert_eq!(conv_w.len(), channels * k, "conv_w must be channels * k");
    let history = (k - 1) * dilation;
    assert_eq!(
        tail.len(),
        history,
        "tail must hold (k - 1) * dilation rows"
    );

    let mut out = vec![0.0f32; channels];
    for ch in 0..channels {
        let mut acc = raw[ch] * conv_w[ch * k + (k - 1)];
        for j in 0..k - 1 {
            acc += tail[j * dilation][ch] * conv_w[ch * k + j];
        }
        out[ch] = crate::gdn::silu(acc);
    }

    if history > 0 {
        tail.remove(0);
        tail.push(raw.to_vec());
    }
    out
}

/// Dequantizes one `qwen4_exp` PLE n-gram table record: INT4 affine at
/// `group_size`-wide groups (32 for the real checkpoint, NOT this port's
/// usual 64 -- `head_dim` (160) is not divisible by 64,
/// `docs/QWEN4_PHASE0.md` item 0's finding 5), one BF16 scale and one BF16
/// bias per group. The nibble-pair convention (low nibble first within a
/// byte) is the same as [`crate::quant::dequantize_int4_affine`]'s -- a
/// property of the checkpoint's MLX `affine` format, not of the group
/// size -- so this is a group-size-parameterized sibling of that function
/// rather than an unrelated one.
///
/// **NOT `dequantize_int4_affine` ITSELF**, which hardcodes `GROUP_SIZE`
/// (64) and reads three SEPARATE planar tensors. The n-gram table's own
/// writer (`NgramTableWriter`) interleaves one row's weight, scale and bias
/// planes into ONE contiguous record so a decode step's 16 lookups are 16
/// reads instead of 48; `packed`/`scales`/`biases` here are the caller's
/// three slices out of that one record (`model_io::NgramTableLayout`'s
/// `weight_bytes` / `scale_bytes` / `bias_bytes` regions), kept as plain
/// byte slices rather than a `model_io` type so this crate takes on no new
/// dependency for one function.
pub fn dequant_ngram_row(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    head_dim: usize,
    group_size: usize,
) -> Vec<f32> {
    assert!(
        group_size > 0 && head_dim % group_size == 0,
        "head_dim {head_dim} must be a whole number of {group_size}-value groups"
    );
    let groups = head_dim / group_size;
    assert_eq!(
        packed.len(),
        head_dim.div_ceil(2),
        "4-bit packed row must be head_dim/2 bytes"
    );
    assert_eq!(scales.len(), groups * 2, "one bf16 scale per group");
    assert_eq!(biases.len(), groups * 2, "one bf16 bias per group");

    let read_bf16 = |bytes: &[u8], g: usize| -> f32 {
        crate::bf16_to_f32(u16::from_le_bytes([bytes[g * 2], bytes[g * 2 + 1]]))
    };

    let mut out = vec![0f32; head_dim];
    for g in 0..groups {
        let scale = read_bf16(scales, g);
        let bias = read_bf16(biases, g);
        for k in 0..group_size {
            let elem = g * group_size + k;
            let byte_idx = elem / 2;
            let b = packed[byte_idx];
            let nibble = if elem & 1 == 0 { b & 0x0F } else { b >> 4 };
            out[elem] = nibble as f32 * scale + bias;
        }
    }
    out
}

#[cfg(test)]
mod dilated_conv_step_tests {
    use super::dilated_conv_step;

    #[test]
    #[should_panic(expected = "k must be at least 1")]
    fn refuses_a_zero_kernel_width() {
        let mut tail: Vec<Vec<f32>> = Vec::new();
        let _ = dilated_conv_step(&mut tail, &[1.0], &[], 1, 0, 1);
    }

    #[test]
    #[should_panic(expected = "dilation must be at least 1")]
    fn refuses_a_zero_dilation() {
        let mut tail: Vec<Vec<f32>> = Vec::new();
        let _ = dilated_conv_step(&mut tail, &[1.0], &[0.0; 4], 1, 4, 0);
    }
}

#[cfg(test)]
mod ngram_row_tests {
    use super::dequant_ngram_row;
    use crate::f32_to_bf16;

    fn bf16_bytes(v: f32) -> [u8; 2] {
        f32_to_bf16(v).to_le_bytes()
    }

    /// Two groups, each with its OWN scale and bias, so a group-boundary
    /// bug (reading group 0's scale for group 1's elements, or vice versa)
    /// reads a wrong number rather than a coincidentally-plausible one.
    #[test]
    fn dequants_two_groups_with_independent_scale_and_bias() {
        // Nibbles 1..8, low nibble first within each byte.
        let packed = [0x21u8, 0x43, 0x65, 0x87];
        let mut scales = Vec::new();
        scales.extend_from_slice(&bf16_bytes(2.0));
        scales.extend_from_slice(&bf16_bytes(1.0));
        let mut biases = Vec::new();
        biases.extend_from_slice(&bf16_bytes(0.5));
        biases.extend_from_slice(&bf16_bytes(-1.0));

        let out = dequant_ngram_row(&packed, &scales, &biases, 8, 4);
        let expected = [2.5f32, 4.5, 6.5, 8.5, 4.0, 5.0, 6.0, 7.0];
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-3, "elem {i}: got {got} want {want}");
        }
    }

    #[test]
    #[should_panic(expected = "whole number")]
    fn refuses_head_dim_not_divisible_by_group_size() {
        dequant_ngram_row(&[0u8; 3], &[0u8; 2], &[0u8; 2], 5, 4);
    }

    #[test]
    #[should_panic(expected = "packed row")]
    fn refuses_wrong_packed_length() {
        dequant_ngram_row(&[0u8; 1], &[0u8; 2], &[0u8; 2], 8, 4);
    }

    #[test]
    #[should_panic(expected = "scale per group")]
    fn refuses_wrong_scale_length() {
        dequant_ngram_row(&[0u8; 4], &[0u8; 2], &[0u8; 4], 8, 4);
    }
}
