//! CPU reference for the Inkling 2 Mini ("inkling2_loop") building blocks.
//!
//! PORT-LOCAL and UNVERIFIED AGAINST WEIGHTS: the formulas are transcribed from
//! the public reference implementations (Hugging Face `modeling_inkling2.py`
//! for the layer math, llama.cpp `inkling2.cpp` / `inkling2-loop.cpp` for the
//! loop, the controllers and the checkpoint conventions). No logit parity with
//! a real checkpoint has been run. See `docs/INKLING2_PHASE0.md`.
//!
//! Checkpoint conventions that are easy to get fluently wrong:
//!
//! * NVFP4 expert weights: value = `E2M1[nibble] * E4M3(block_scale) *
//!   amax[expert] / outer_scale_denominator`, 16-element blocks, the LOW
//!   nibble is the EVEN element.
//! * MXFP8 expert weights: value = `E4M3(byte) * 2^(scale_byte - 127)`,
//!   32-element blocks; scale byte 255 is NaN.
//! * `w13` rows are INTERLEAVED gate/up: row `2k` is gate `k`, row `2k+1` is
//!   up `k`.
//! * Short convolutions finish with `(conv + residual) / sqrt(K + 1)`, or
//!   `sqrt(2K + 1)` for the dynamic (full-attention K/V) variant.

pub mod forward;

/// E2M1 (FP4) codebook, index = 4-bit code.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decodes an E4M3FN byte (bias 7, no infinities, 0x7F / 0xFF are NaN).
pub fn e4m3_to_f32(code: u8) -> f32 {
    if code & 0x7F == 0x7F {
        return f32::NAN;
    }
    let exp = i32::from((code & 0x7F) >> 3);
    let man = f32::from(code & 7);
    let mag = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    if code & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

/// Dequantizes one NVFP4 row. `packed` holds `cols / 2` bytes, `block_scales`
/// holds `cols / 16` E4M3 bytes. `outer` is `amax[expert] / denominator`.
pub fn dequant_nvfp4_row(packed: &[u8], block_scales: &[u8], outer: f32, out: &mut [f32]) {
    assert_eq!(out.len(), packed.len() * 2);
    assert_eq!(out.len() % 16, 0);
    assert_eq!(block_scales.len(), out.len() / 16);
    for (i, &b) in packed.iter().enumerate() {
        let s = e4m3_to_f32(block_scales[(2 * i) / 16]) * outer;
        out[2 * i] = E2M1[usize::from(b & 15)] * s;
        out[2 * i + 1] = E2M1[usize::from(b >> 4)] * s;
    }
}

/// Dequantizes one MXFP8 row (`cols` E4M3 bytes, `cols / 32` E8M0 scales).
pub fn dequant_mxfp8_row(bytes: &[u8], scales: &[u8], out: &mut [f32]) {
    assert_eq!(out.len(), bytes.len());
    assert_eq!(out.len() % 32, 0);
    assert_eq!(scales.len(), out.len() / 32);
    for (i, &b) in bytes.iter().enumerate() {
        let e = scales[i / 32];
        let scale = if e == 255 {
            f32::NAN
        } else {
            2f32.powi(i32::from(e) - 127)
        };
        out[i] = e4m3_to_f32(b) * scale;
    }
}

/// Elements in one transcoded NVFP4 block (ggml `block_nvfp4`).
pub const NVFP4_BLOCK_ELEMS: usize = 64;
/// Bytes in one transcoded NVFP4 block: 4 E4M3 scales, then 32 code bytes.
pub const NVFP4_BLOCK_BYTES: usize = 36;
/// Elements in one transcoded MXFP8 block.
pub const MXFP8_BLOCK_ELEMS: usize = 32;
/// Bytes in one transcoded MXFP8 block: the E8M0 exponent, then 32 E4M3 bytes.
pub const MXFP8_BLOCK_BYTES: usize = 33;

/// Repacks one HF NVFP4 row into ggml `block_nvfp4` blocks, so the routed
/// expert kernels reuse the GGUF NVFP4 decoder (`crate::dequantize_nvfp4`).
///
/// Lossless: the codes and scale bytes are moved, never re-rounded. Only the
/// nibble order changes (HF packs element `2i` low / `2i+1` high in byte `i`;
/// ggml packs `s*16 + j` low / `s*16 + j + 8` high in `qs[s*8 + j]`). The HF
/// code 8 (-0) becomes ggml's +0, which is the same value in any dot product.
/// The per-expert outer scale is NOT folded in: it is stored beside the row,
/// because folding it into the E4M3 scales would re-round them.
///
/// Refuses a scale byte with the sign bit set or the E4M3 NaN code: ggml's
/// decoder ignores bit 7 and reads `0x7F` as zero, so either would silently
/// change the weight rather than fail.
pub fn nvfp4_row_to_blocks(packed: &[u8], block_scales: &[u8]) -> Result<Vec<u8>, String> {
    let n = packed.len() * 2;
    if n % NVFP4_BLOCK_ELEMS != 0 {
        return Err(format!(
            "NVFP4 row of {n} elements is not a multiple of {NVFP4_BLOCK_ELEMS}"
        ));
    }
    if block_scales.len() != n / 16 {
        return Err(format!(
            "NVFP4 row of {n} elements needs {} scales, got {}",
            n / 16,
            block_scales.len()
        ));
    }
    let nibble = |e: usize| (packed[e / 2] >> (4 * (e % 2))) & 15;
    let mut out = Vec::with_capacity(n / NVFP4_BLOCK_ELEMS * NVFP4_BLOCK_BYTES);
    for b in 0..n / NVFP4_BLOCK_ELEMS {
        for s in 0..4 {
            let i = b * 4 + s;
            let byte = block_scales[i];
            if byte & 0x80 != 0 || byte == 0x7F {
                return Err(format!("NVFP4 scale {i} is 0x{byte:02x} (negative or NaN)"));
            }
            out.push(byte);
        }
        for s in 0..4 {
            for j in 0..8 {
                let lo = b * NVFP4_BLOCK_ELEMS + s * 16 + j;
                out.push(nibble(lo) | (nibble(lo + 8) << 4));
            }
        }
    }
    Ok(out)
}

/// Repacks one MXFP8 row into 33-byte blocks (exponent byte first, then the
/// block's 32 E4M3 bytes), so a routed row is one contiguous run the way the
/// GGUF blobs are. Lossless. Refuses the E8M0 NaN exponent (255) and E4M3 NaN
/// codes: a NaN weight is a corrupt checkpoint, not something to stream.
pub fn mxfp8_row_to_blocks(bytes: &[u8], scales: &[u8]) -> Result<Vec<u8>, String> {
    let n = bytes.len();
    if n % MXFP8_BLOCK_ELEMS != 0 {
        return Err(format!(
            "MXFP8 row of {n} elements is not a multiple of {MXFP8_BLOCK_ELEMS}"
        ));
    }
    if scales.len() != n / MXFP8_BLOCK_ELEMS {
        return Err(format!(
            "MXFP8 row of {n} elements needs {} scales, got {}",
            n / MXFP8_BLOCK_ELEMS,
            scales.len()
        ));
    }
    let mut out = Vec::with_capacity(n / MXFP8_BLOCK_ELEMS * MXFP8_BLOCK_BYTES);
    for (b, (block, &e)) in bytes
        .chunks_exact(MXFP8_BLOCK_ELEMS)
        .zip(scales)
        .enumerate()
    {
        if e == 255 {
            return Err(format!("MXFP8 scale {b} is the E8M0 NaN code"));
        }
        if let Some(k) = block.iter().position(|&c| c & 0x7F == 0x7F) {
            return Err(format!(
                "MXFP8 element {} is an E4M3 NaN code",
                b * MXFP8_BLOCK_ELEMS + k
            ));
        }
        out.push(e);
        out.extend_from_slice(block);
    }
    Ok(out)
}

/// Decodes `n` elements of [`mxfp8_row_to_blocks`] output: the CPU reference
/// for the transcoded layout the GPU reads.
pub fn dequantize_mxfp8_blocks(blocks: &[u8], n: usize) -> Vec<f32> {
    assert_eq!(n % MXFP8_BLOCK_ELEMS, 0);
    assert_eq!(blocks.len(), n / MXFP8_BLOCK_ELEMS * MXFP8_BLOCK_BYTES);
    let mut out = Vec::with_capacity(n);
    for block in blocks.chunks_exact(MXFP8_BLOCK_BYTES) {
        let scale = 2f32.powi(i32::from(block[0]) - 127);
        out.extend(block[1..].iter().map(|&c| e4m3_to_f32(c) * scale));
    }
    out
}

/// Splits interleaved `w13` rows into `(gate_rows, up_rows)` row indices.
pub fn w13_gate_up_rows(rows: usize) -> (Vec<usize>, Vec<usize>) {
    assert_eq!(rows % 2, 0);
    (
        (0..rows / 2).map(|k| 2 * k).collect(),
        (0..rows / 2).map(|k| 2 * k + 1).collect(),
    )
}

/// RMS norm with no learned scale.
pub fn rms_norm_unweighted(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * r).collect()
}

/// RMS norm with a learned scale `w` (plain `w`, not `1 + w`).
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    rms_norm_unweighted(x, eps)
        .iter()
        .zip(w)
        .map(|(a, b)| a * b)
        .collect()
}

/// Per-vector L2 clip: scales down to `threshold` when the norm exceeds it,
/// after an optional per-channel `scale`. Shorter vectors are unchanged.
pub fn l2_clip(x: &[f32], scale: Option<&[f32]>, threshold: f32) -> Vec<f32> {
    let y: Vec<f32> = match scale {
        Some(s) => x.iter().zip(s).map(|(a, b)| a * b).collect(),
        None => x.to_vec(),
    };
    let norm = y.iter().map(|v| v * v).sum::<f32>().sqrt();
    let f = (threshold / (norm + 1e-6)).clamp(0.0, 1.0);
    y.iter().map(|v| v * f).collect()
}

/// `1 + alpha * ln(max(1, (pos + 1) / floor))`, the full-attention log scale
/// that multiplies Q and the relative states.
pub fn log_scale_tau(pos: usize, floor: f32, alpha: f32) -> f32 {
    1.0 + alpha * (((pos + 1) as f32) / floor).max(1.0).ln()
}

/// Relative-position bias for one query head: `rel = r @ proj` is a profile
/// over distance; the bias at distance `d` is `rel[d]` for `0 <= d < extent`,
/// else 0. `proj` is `[d_rel, extent]` row-major.
pub fn relative_bias(r: &[f32], proj: &[f32], extent: usize, distance: isize) -> f32 {
    if distance < 0 || distance as usize >= extent {
        return 0.0;
    }
    let d = distance as usize;
    r.iter()
        .enumerate()
        .map(|(i, ri)| ri * proj[i * extent + d])
        .sum()
}

/// The relative-bias table for one query: `table[h * extent + d] = r_h .
/// proj[:, d]`, with `r` `[heads, d_rel]` (already clipped, and times `tau` on
/// full layers) and `proj` `[d_rel, extent]` row-major.
pub fn rel_bias_table(
    r: &[f32],
    proj: &[f32],
    heads: usize,
    d_rel: usize,
    extent: usize,
) -> Vec<f32> {
    assert_eq!(r.len(), heads * d_rel);
    assert_eq!(proj.len(), d_rel * extent);
    let mut t = vec![0.0f32; heads * extent];
    for h in 0..heads {
        for d in 0..extent {
            t[h * extent + d] = (0..d_rel)
                .map(|i| r[h * d_rel + i] * proj[i * extent + d])
                .sum();
        }
    }
    t
}

/// One query's attention over cached positions `first..=pos`. `k` and `v` are
/// `[n_pos, kv_heads * head_dim]` from position 0; `table` is
/// [`rel_bias_table`]'s output (distance `d >= extent` gets no bias); `sinks`
/// holds the per-head sink logit, which joins the softmax denominator with a
/// zero value. Scale `1/sqrt(head_dim)`, query head `h` reads kv head
/// `h / (heads / kv_heads)`. Returns `[heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    table: &[f32],
    sinks: &[f32],
    first: usize,
    pos: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    extent: usize,
) -> Vec<f32> {
    let kvd = kv_heads * head_dim;
    let group = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut o = vec![0.0f32; heads * head_dim];
    for h in 0..heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let logits: Vec<f32> = (first..=pos)
            .map(|kp| {
                let kk = &k[kp * kvd + kvh * head_dim..kp * kvd + (kvh + 1) * head_dim];
                let d = pos - kp;
                let bias = if d < extent {
                    table[h * extent + d]
                } else {
                    0.0
                };
                qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f32>() * scale + bias
            })
            .collect();
        let t = sinks[h];
        let m = logits.iter().copied().fold(t, f32::max);
        let mut z = (t - m).exp();
        let oh = &mut o[h * head_dim..(h + 1) * head_dim];
        for (j, &l) in logits.iter().enumerate() {
            let w = (l - m).exp();
            z += w;
            let kp = first + j;
            let vv = &v[kp * kvd + kvh * head_dim..kp * kvd + (kvh + 1) * head_dim];
            for (x, y) in oh.iter_mut().zip(vv) {
                *x += w * y;
            }
        }
        for x in oh.iter_mut() {
            *x /= z;
        }
    }
    o
}

fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

/// Softcapped SwiGLU: `softcap(g) * sigmoid(g) * softcap(u)` (the sigmoid
/// sees the RAW gate; the caps only bound the linear factors).
pub fn situ(gate: f32, up: f32, gate_cap: f32, up_cap: f32) -> f32 {
    softcap(gate, gate_cap) * (1.0 / (1.0 + (-gate).exp())) * softcap(up, up_cap)
}

/// Sigmoid router: selection uses `sigmoid(logit) + bias`, weights are the
/// UNBIASED scores of the selected experts, normalized to sum 1, times
/// `route_scale`. Returns `(indices, weights)` in rank order (best first,
/// ties to the lower index) because slot order is reduction order.
pub fn route_sigmoid(
    logits: &[f32],
    bias: &[f32],
    k: usize,
    route_scale: f32,
) -> (Vec<usize>, Vec<f32>) {
    let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        (scores[b] + bias[b])
            .partial_cmp(&(scores[a] + bias[a]))
            .unwrap()
            .then(a.cmp(&b))
    });
    order.truncate(k);
    let sum: f32 = order.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
    let w = order
        .iter()
        .map(|&i| scores[i] / sum * route_scale)
        .collect();
    (order, w)
}

/// Block attention residual ("depth attention"). `states` are the earlier
/// block outputs plus the current partial state (last); each query attends
/// over them with the VALUES being the RMS-normed states, scale
/// `1/sqrt(hidden)`. Returns one mixed vector per query.
pub fn block_attn_res_mix(states: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let normed: Vec<Vec<f32>> = states
        .iter()
        .map(|s| rms_norm_unweighted(s, 1e-6))
        .collect();
    let h = normed[0].len();
    let scale = 1.0 / (h as f32).sqrt();
    queries
        .iter()
        .map(|q| {
            let mut sc: Vec<f32> = normed
                .iter()
                .map(|s| s.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() * scale)
                .collect();
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut z = 0.0;
            for v in &mut sc {
                *v = (*v - m).exp();
                z += *v;
            }
            let mut out = vec![0.0; h];
            for (w, s) in sc.iter().zip(&normed) {
                for (o, x) in out.iter_mut().zip(s) {
                    *o += w / z * x;
                }
            }
            out
        })
        .collect()
}

/// Causal short convolution with residual over `rows` consecutive rows.
/// `weight` is `[channels, taps]`, tap `taps-1` multiplies the CURRENT row
/// (standard left-padded cross-correlation). `tail` is `[taps-1, channels]`,
/// oldest first, advanced in place. With `dynamic` (`[rows, heads, taps]`,
/// tap index = backward lag, 0 = current), per-head taps are added on top and
/// the output scale becomes `1/sqrt(2*taps + 1)`.
pub fn short_conv(
    x: &[Vec<f32>],
    weight: &[f32],
    taps: usize,
    tail: &mut Vec<Vec<f32>>,
    dynamic: Option<(&[f32], usize)>,
) -> Vec<Vec<f32>> {
    let ch = x[0].len();
    assert_eq!(tail.len(), taps - 1);
    let mut hist: Vec<Vec<f32>> = tail.clone();
    hist.extend(x.iter().cloned());
    let mut out = Vec::new();
    for t in 0..x.len() {
        let cur = t + taps - 1;
        let mut row = vec![0.0f32; ch];
        for (c, r) in row.iter_mut().enumerate() {
            for j in 0..taps {
                *r += weight[c * taps + j] * hist[cur + j + 1 - taps][c];
            }
            if let Some((dw, heads)) = dynamic {
                let per = ch / heads;
                let head = c / per;
                for lag in 0..taps {
                    *r += dw[(t * heads + head) * taps + lag] * hist[cur - lag][c];
                }
            }
        }
        let scale = 1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * taps + 1) as f32).sqrt();
        out.push((0..ch).map(|c| (row[c] + x[t][c]) * scale).collect());
    }
    *tail = hist[hist.len() - (taps - 1)..].to_vec();
    out
}

/// Loop controller: `(1 - forget) * cur + gain * msg`, where
/// `gate = down @ [rms(cur) ; rms(msg)]`, `gain = tanh(gain_w @ gate)`,
/// `forget = tanh(forget_w @ gate)`. `down` is `[rank, 2h]`, the others
/// `[h, rank]`, all row-major.
pub fn loop_control(
    cur: &[f32],
    msg: &[f32],
    down: &[f32],
    gain_w: &[f32],
    forget_w: &[f32],
    rank: usize,
) -> Vec<f32> {
    let h = cur.len();
    let mut cat = rms_norm_unweighted(cur, 1e-6);
    cat.extend(rms_norm_unweighted(msg, 1e-6));
    let gate: Vec<f32> = (0..rank)
        .map(|r| (0..2 * h).map(|i| down[r * 2 * h + i] * cat[i]).sum())
        .collect();
    (0..h)
        .map(|i| {
            let g = (0..rank)
                .map(|r| gain_w[i * rank + r] * gate[r])
                .sum::<f32>()
                .tanh();
            let f = (0..rank)
                .map(|r| forget_w[i * rank + r] * gate[r])
                .sum::<f32>()
                .tanh();
            cur[i] - f * cur[i] + g * msg[i]
        })
        .collect()
}

/// Physical layer behind cache slot `slot` of the unrolled loop schedule:
/// slots `< n_phys` are the layers themselves, later slots repeat the core
/// `[core_start, core_end)` once per extra pass.
pub fn layer_phys(slot: usize, n_phys: usize, core_start: usize, core_end: usize) -> usize {
    if slot < n_phys {
        slot
    } else {
        core_start + (slot - n_phys) % (core_end - core_start)
    }
}

/// Total cache slots: `n_phys + (passes - 1) * core_len`.
pub fn cache_slots(n_phys: usize, core_start: usize, core_end: usize, passes: usize) -> usize {
    n_phys + (passes - 1) * (core_end - core_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exp 7, man 0
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal
        assert!(e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn nvfp4_low_nibble_is_even_element() {
        // byte 0x21: low nibble 1 (0.5) -> element 0, high nibble 2 (1.0) -> element 1
        let mut packed = vec![0u8; 8];
        packed[0] = 0x21;
        let scales = [0x38u8]; // 1.0
        let mut out = vec![9.0; 16];
        dequant_nvfp4_row(&packed, &scales, 2.0, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn nvfp4_scale_block_is_16_wide() {
        let packed = vec![0x77u8; 16]; // 6.0, 6.0 ...
        let scales = [0x38u8, 0x40]; // 1.0 then 2.0
        let mut out = vec![0.0; 32];
        dequant_nvfp4_row(&packed, &scales, 1.0, &mut out);
        assert_eq!(out[15], 6.0);
        assert_eq!(out[16], 12.0);
    }

    /// Every valid scale byte (0x00..=0x7E) and every code, in every one of a
    /// block's four 16-element sub-blocks, decodes to the same value through
    /// the transcoded ggml layout as through the HF reference.
    #[test]
    fn nvfp4_transcode_is_lossless_for_every_scale_and_code() {
        let n_scales = 0x7Fusize; // 0x00..=0x7E
        let blocks = n_scales.div_ceil(4) * 4; // whole blocks of four scales
        let n = blocks / 4 * NVFP4_BLOCK_ELEMS * 4;
        let scales: Vec<u8> = (0..n / 16).map(|i| (i % n_scales) as u8).collect();
        // Codes run through 0..16 at a stride coprime to 16 so every
        // sub-block holds every code in a different position.
        let packed: Vec<u8> = (0..n / 2)
            .map(|i| ((2 * i * 7) % 16) as u8 | ((((2 * i + 1) * 7 + 3) % 16) as u8) << 4)
            .collect();
        let mut want = vec![0.0; n];
        dequant_nvfp4_row(&packed, &scales, 1.0, &mut want);
        let got = crate::dequantize_nvfp4(&nvfp4_row_to_blocks(&packed, &scales).unwrap(), n);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g == w, "element {i}: transcoded {g} vs reference {w}");
        }
        assert!(want.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn nvfp4_transcode_refuses_negative_and_nan_scales() {
        let packed = vec![0x11u8; 32];
        for bad in [0x7Fu8, 0x80, 0xB8] {
            let mut scales = vec![0x38u8; 4];
            scales[2] = bad;
            assert!(
                nvfp4_row_to_blocks(&packed, &scales).is_err(),
                "0x{bad:02x}"
            );
        }
        assert!(nvfp4_row_to_blocks(&packed[..16], &[0x38; 2]).is_err());
        assert!(nvfp4_row_to_blocks(&packed, &[0x38; 3]).is_err());
    }

    /// Every non-NaN E4M3 code under a spread of exponents, bit-exact.
    #[test]
    fn mxfp8_transcode_is_lossless_for_every_code() {
        let codes: Vec<u8> = (0..=255u8).filter(|c| c & 0x7F != 0x7F).collect();
        let mut bytes = Vec::new();
        let mut scales = Vec::new();
        for (b, e) in [0u8, 1, 100, 127, 130, 200, 254].into_iter().enumerate() {
            for k in 0..MXFP8_BLOCK_ELEMS {
                bytes.push(codes[(b * MXFP8_BLOCK_ELEMS + k * 5) % codes.len()]);
            }
            scales.push(e);
        }
        let n = bytes.len();
        let mut want = vec![0.0; n];
        dequant_mxfp8_row(&bytes, &scales, &mut want);
        let got = dequantize_mxfp8_blocks(&mxfp8_row_to_blocks(&bytes, &scales).unwrap(), n);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.to_bits() == w.to_bits(), "element {i}: {g} vs {w}");
        }
    }

    #[test]
    fn mxfp8_transcode_refuses_nan() {
        let ok = vec![0x38u8; 32];
        assert!(mxfp8_row_to_blocks(&ok, &[255]).is_err());
        for bad in [0x7Fu8, 0xFF] {
            let mut bytes = ok.clone();
            bytes[9] = bad;
            assert!(mxfp8_row_to_blocks(&bytes, &[127]).is_err(), "0x{bad:02x}");
        }
        assert!(mxfp8_row_to_blocks(&ok[..16], &[127]).is_err());
    }

    #[test]
    fn mxfp8_scale_and_nan() {
        let bytes = vec![0x38u8; 64];
        let mut out = vec![0.0; 64];
        dequant_mxfp8_row(&bytes, &[128, 126], &mut out);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[31], 2.0);
        assert_eq!(out[32], 0.5);
        dequant_mxfp8_row(&bytes, &[255, 127], &mut out);
        assert!(out[0].is_nan() && out[40] == 1.0);
    }

    #[test]
    fn w13_is_interleaved() {
        let (g, u) = w13_gate_up_rows(6);
        assert_eq!(g, vec![0, 2, 4]);
        assert_eq!(u, vec![1, 3, 5]);
    }

    #[test]
    fn l2_clip_only_shrinks() {
        let y = l2_clip(&[3.0, 4.0], None, 10.0);
        assert_eq!(y, vec![3.0, 4.0]);
        let y = l2_clip(&[3.0, 4.0], None, 1.0);
        let n = (y[0] * y[0] + y[1] * y[1]).sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }

    #[test]
    fn tau_is_one_below_floor() {
        assert_eq!(log_scale_tau(10, 8192.0, 0.1), 1.0);
        assert!(log_scale_tau(100_000, 8192.0, 0.1) > 1.0);
    }

    #[test]
    fn relative_bias_zero_outside_extent() {
        let proj = vec![1.0, 2.0, 3.0, 4.0]; // d_rel 1, extent 4
        assert_eq!(relative_bias(&[2.0], &proj, 4, 2), 6.0);
        assert_eq!(relative_bias(&[2.0], &proj, 4, 4), 0.0);
        assert_eq!(relative_bias(&[2.0], &proj, 4, -1), 0.0);
    }

    #[test]
    fn situ_sigmoid_sees_raw_gate() {
        // gate far past the cap: softcap saturates at cap, sigmoid still ~1
        let v = situ(100.0, 100.0, 4.0, 25.0);
        assert!((v - 4.0 * 25.0).abs() < 0.5, "{v}");
        assert_eq!(situ(0.0, 5.0, 4.0, 25.0), 0.0);
    }

    #[test]
    fn router_bias_selects_but_does_not_weight() {
        // expert 0 has the lower score but a huge bias: selected, weighted by score
        let (idx, w) = route_sigmoid(&[0.0, 2.0, -5.0], &[10.0, 0.0, 0.0], 2, 8.0);
        assert_eq!(idx, vec![0, 1]);
        let s0 = 0.5f32;
        let s1 = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((w[0] - s0 / (s0 + s1) * 8.0).abs() < 1e-4);
        assert!((w.iter().sum::<f32>() - 8.0).abs() < 1e-3);
    }

    #[test]
    fn mix_single_state_returns_its_norm() {
        let s = vec![vec![3.0, 4.0]];
        let out = block_attn_res_mix(&s, &[vec![0.0, 0.0]]);
        let n = rms_norm_unweighted(&s[0], 1e-6);
        assert_eq!(out[0], n);
    }

    #[test]
    fn mix_weights_sum_to_one() {
        let states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let out = block_attn_res_mix(&states, &[vec![1.0, -1.0], vec![0.5, 0.5]]);
        assert_eq!(out.len(), 2);
        assert!(out.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn short_conv_current_tap_is_last_and_scaled() {
        // 1 channel, taps 2, weight [w_prev, w_cur]
        let mut tail = vec![vec![0.0]];
        let out = short_conv(&[vec![1.0], vec![2.0]], &[10.0, 100.0], 2, &mut tail, None);
        let s = 1.0 / 3f32.sqrt();
        assert!((out[0][0] - (100.0 + 1.0) * s).abs() < 1e-4);
        assert!((out[1][0] - (10.0 + 200.0 + 2.0) * s).abs() < 1e-4);
        assert_eq!(tail, vec![vec![2.0]]);
    }

    #[test]
    fn short_conv_chunked_equals_stepwise() {
        let x: Vec<Vec<f32>> = (0..5).map(|t| vec![t as f32 + 1.0, -(t as f32)]).collect();
        let w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut t1 = vec![vec![0.0; 2]; 3];
        let all = short_conv(&x, &w, 4, &mut t1, None);
        let mut t2 = vec![vec![0.0; 2]; 3];
        let mut step = Vec::new();
        for r in &x {
            step.extend(short_conv(std::slice::from_ref(r), &w, 4, &mut t2, None));
        }
        assert_eq!(all, step);
        assert_eq!(t1, t2);
    }

    #[test]
    fn dynamic_conv_lag0_is_current() {
        // 2 channels, 1 head, taps 2; dynamic taps [lag0=1, lag1=0], static 0
        let mut tail = vec![vec![5.0, 5.0]];
        let out = short_conv(
            &[vec![2.0, 3.0]],
            &[0.0; 4],
            2,
            &mut tail,
            Some((&[1.0, 0.0], 1)),
        );
        let s = 1.0 / 5f32.sqrt();
        assert!((out[0][0] - (2.0 + 2.0) * s).abs() < 1e-5);
    }

    #[test]
    fn control_zero_gate_is_identity() {
        let cur = vec![1.0, 2.0];
        let msg = vec![3.0, 4.0];
        let z = [0.0; 8];
        let out = loop_control(&cur, &msg, &[0.0; 4], &z[..2], &z[..2], 1);
        assert_eq!(out, cur);
    }

    #[test]
    fn loop_schedule_matches_mini() {
        // 24 layers, core [6, 18), 3 passes -> 48 slots, target ids 5/23/41/48 are in range
        assert_eq!(cache_slots(24, 6, 18, 3), 48);
        assert_eq!(layer_phys(5, 24, 6, 18), 5);
        assert_eq!(layer_phys(24, 24, 6, 18), 6);
        assert_eq!(layer_phys(35, 24, 6, 18), 17);
        assert_eq!(layer_phys(36, 24, 6, 18), 6);
        assert_eq!(layer_phys(47, 24, 6, 18), 17);
    }
}
