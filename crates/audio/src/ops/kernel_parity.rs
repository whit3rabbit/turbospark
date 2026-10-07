//! Bitwise parity of the optimized `ops` kernels against retained naive
//! references.
//!
//! The `*_reference` functions below are the original single-threaded
//! loops, kept verbatim. Every model's golden-tensor test was recorded
//! against that reduction order, so a faster kernel is only acceptable if
//! it produces the same bits, not merely close values. Inputs mix
//! magnitudes and exact zeros so a changed summation order or a changed
//! zero-skip would show up in the bits.

use super::*;

/// One conv case: channel counts, kernel geometry, grouping and length.
type EightDims = (usize, usize, usize, usize, usize, usize, usize, usize);

/// Deterministic xorshift stream; avoids a dev-dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Values in about [-8, 8] over several magnitudes, with ~1 in 6
    /// exact zeros.
    fn value(&mut self) -> f32 {
        let r = self.next();
        if r % 6 == 0 {
            return 0.0;
        }
        let mantissa = ((r >> 8) % 20001) as f32 / 10000.0 - 1.0;
        let scale = [0.001f32, 0.1, 1.0, 8.0][((r >> 40) % 4) as usize];
        mantissa * scale
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.value()).collect()
    }
}

fn assert_bits_eq(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {i}: got {g} want {w}"
        );
    }
}

fn matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                out[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    out
}

fn linear_reference(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[i * k + ki] * w[ni * k + ki];
            }
            out[i * n + ni] = acc + bias.map_or(0.0, |b| b[ni]);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn conv1d_reference(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Vec<f32> {
    let seq = x.len() / in_ch;
    let in_per_g = in_ch / groups;
    let out_per_g = out_ch / groups;
    let k_eff = (kernel - 1) * dilation + 1;
    let out_seq = (seq + 2 * padding - k_eff) / stride + 1;
    let mut out = vec![0.0f32; out_ch * out_seq];
    for oc in 0..out_ch {
        let g = oc / out_per_g;
        for os in 0..out_seq {
            let base_in = (os * stride) as isize - padding as isize;
            let mut acc = bias.map_or(0.0, |b| b[oc]);
            for ic_in_g in 0..in_per_g {
                let ic = g * in_per_g + ic_in_g;
                let w_base = ((oc * in_per_g) + ic_in_g) * kernel;
                let x_ch = &x[ic * seq..(ic + 1) * seq];
                for kk in 0..kernel {
                    let pos = base_in + (kk * dilation) as isize;
                    if pos < 0 || pos as usize >= seq {
                        continue;
                    }
                    acc += x_ch[pos as usize] * w[w_base + kk];
                }
            }
            out[oc * out_seq + os] = acc;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn conv_transpose1d_reference(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
) -> Vec<f32> {
    let seq = x.len() / in_ch;
    let out_seq = (seq - 1) * stride + kernel - 2 * padding + output_padding;
    let out_per_g = out_ch / groups;
    let in_per_g = in_ch / groups;
    let mut out = vec![0.0f32; out_ch * out_seq];
    for ic in 0..in_ch {
        let g = ic / in_per_g;
        let x_ch = &x[ic * seq..(ic + 1) * seq];
        for (is, &xv) in x_ch.iter().enumerate() {
            if xv == 0.0 {
                continue;
            }
            for oc_in_g in 0..out_per_g {
                let oc = g * out_per_g + oc_in_g;
                let w_base = ((ic * out_per_g) + oc_in_g) * kernel;
                for kk in 0..kernel {
                    let pos = is * stride + kk;
                    if pos < padding || pos - padding >= out_seq {
                        continue;
                    }
                    out[oc * out_seq + (pos - padding)] += xv * w[w_base + kk];
                }
            }
        }
    }
    if let Some(b) = bias {
        for oc in 0..out_ch {
            for v in &mut out[oc * out_seq..(oc + 1) * out_seq] {
                *v += b[oc];
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn sdpa_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    qh: usize,
    kv: usize,
    dq: usize,
    dv: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; qh * dv];
    let mut scores = vec![0.0f32; qh * kv.max(1)];
    for h in 0..qh {
        for j in 0..kv {
            let mut acc = 0.0f32;
            for d in 0..dq {
                acc += q[h * dq + d] * k[j * dq + d];
            }
            scores[h * kv + j] = acc * scale + mask.map_or(0.0, |m| m[h * kv + j]);
        }
        softmax_row(&mut scores[h * kv..(h + 1) * kv]);
        for j in 0..kv {
            let p = scores[h * kv + j];
            for d in 0..dv {
                out[h * dv + d] += p * v[j * dv + d];
            }
        }
    }
    out
}

#[test]
fn matmul_is_bitwise_identical_to_reference() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // The last two shapes clear the threading threshold.
    for &(m, k, n) in &[
        (1, 1, 1),
        (1, 9, 7),
        (3, 17, 5),
        (5, 31, 13),
        (8, 64, 70),
        (16, 256, 130),
        (2, 700, 600),
    ] {
        let a = rng.vec(m * k);
        let b = rng.vec(k * n);
        assert_bits_eq(
            &format!("matmul {m}x{k}x{n}"),
            &matmul(&a, &b, m, k, n),
            &matmul_reference(&a, &b, m, k, n),
        );
    }
}

#[test]
fn linear_is_bitwise_identical_to_reference() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    // Column counts of 1..=9 cover every remainder of a 4- and 8-wide
    // column block; m == 1 is the decode shape and the large n cases
    // clear the threading threshold.
    let mut shapes = vec![(1, 1, 1), (1, 1, 4), (1, 5, 3), (1, 64, 33), (3, 17, 9)];
    for n in 1..=9 {
        shapes.push((1, 13, n));
        shapes.push((2, 13, n));
    }
    shapes.extend([
        (4, 128, 65),
        (7, 40, 11),
        (1, 512, 2051),
        (1, 1030, 515),
        (6, 300, 512),
        (33, 64, 130),
    ]);
    for (m, k, n) in shapes {
        let x = rng.vec(m * k);
        let w = rng.vec(n * k);
        let bias = rng.vec(n);
        for with_bias in [false, true] {
            let b = with_bias.then_some(&bias[..]);
            assert_bits_eq(
                &format!("linear {m}x{k}x{n} bias={with_bias}"),
                &linear(&x, &w, b, m, k, n),
                &linear_reference(&x, &w, b, m, k, n),
            );
        }
    }
}

#[test]
fn linear_keeps_negative_zero_when_there_is_no_bias() {
    // `acc + 0.0` turns a -0.0 accumulator into +0.0; the reference does
    // that for the bias-free case, so the optimized kernel must too.
    let x = [-1.0f32];
    let w = [0.0f32];
    let out = linear(&x, &w, None, 1, 1, 1);
    assert_eq!(
        out[0].to_bits(),
        linear_reference(&x, &w, None, 1, 1, 1)[0].to_bits()
    );
}

#[test]
fn conv1d_is_bitwise_identical_to_reference() {
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    // (in_ch, out_ch, kernel, stride, padding, dilation, groups, seq)
    let cases: &[EightDims] = &[
        (1, 1, 1, 1, 0, 1, 1, 5),
        (2, 3, 3, 1, 1, 1, 1, 17),
        (4, 4, 7, 1, 3, 1, 1, 40),
        (4, 4, 3, 1, 3, 3, 1, 25),
        (3, 5, 4, 2, 2, 1, 1, 33),
        (4, 8, 5, 3, 2, 2, 2, 50),
        (6, 6, 3, 1, 1, 1, 6, 21),
        (6, 12, 7, 2, 3, 1, 3, 64),
        (8, 8, 15, 1, 7, 1, 1, 100),
        (2, 2, 5, 1, 8, 4, 1, 12),
        // Kernel reach exceeds the signal: most taps fall in the padding.
        (2, 3, 9, 1, 8, 2, 1, 4),
        (1, 2, 5, 2, 0, 1, 1, 9),
        // Large enough to clear the threading threshold.
        (32, 64, 7, 1, 3, 1, 1, 400),
        (64, 64, 3, 1, 1, 1, 1, 600),
    ];
    for &(in_ch, out_ch, kernel, stride, padding, dilation, groups, seq) in cases {
        let x = rng.vec(in_ch * seq);
        let w = rng.vec(out_ch * (in_ch / groups) * kernel);
        let bias = rng.vec(out_ch);
        for with_bias in [false, true] {
            let b = with_bias.then_some(&bias[..]);
            assert_bits_eq(
                &format!(
                    "conv1d in={in_ch} out={out_ch} k={kernel} s={stride} p={padding} \
                     d={dilation} g={groups} seq={seq} bias={with_bias}"
                ),
                &conv1d(
                    &x, &w, b, in_ch, out_ch, kernel, stride, padding, dilation, groups,
                ),
                &conv1d_reference(
                    &x, &w, b, in_ch, out_ch, kernel, stride, padding, dilation, groups,
                ),
            );
        }
    }
}

#[test]
fn conv_transpose1d_is_bitwise_identical_to_reference() {
    let mut rng = Rng(0xE703_7ED1_A0B4_28DB);
    // (in_ch, out_ch, kernel, stride, padding, output_padding, groups, seq)
    let cases: &[EightDims] = &[
        (1, 1, 1, 1, 0, 0, 1, 4),
        (2, 3, 3, 1, 1, 0, 1, 9),
        (4, 2, 8, 4, 2, 0, 1, 11),
        (4, 4, 16, 8, 4, 0, 1, 7),
        (3, 6, 4, 2, 1, 1, 1, 13),
        (4, 4, 5, 2, 2, 1, 2, 10),
        (6, 6, 7, 3, 0, 0, 3, 8),
        (2, 2, 4, 2, 0, 0, 1, 1),
        // Overlapping windows (kernel > stride) and padding past the edge.
        (2, 2, 10, 2, 4, 0, 1, 6),
        (1, 3, 3, 3, 0, 0, 1, 5),
        // Large enough to clear the threading threshold.
        (32, 16, 8, 4, 2, 0, 1, 200),
        (64, 32, 16, 8, 4, 0, 1, 100),
    ];
    for &(in_ch, out_ch, kernel, stride, padding, output_padding, groups, seq) in cases {
        let x = rng.vec(in_ch * seq);
        let w = rng.vec(in_ch * (out_ch / groups) * kernel);
        let bias = rng.vec(out_ch);
        for with_bias in [false, true] {
            let b = with_bias.then_some(&bias[..]);
            assert_bits_eq(
                &format!(
                    "conv_transpose1d in={in_ch} out={out_ch} k={kernel} s={stride} \
                     p={padding} op={output_padding} g={groups} seq={seq} bias={with_bias}"
                ),
                &conv_transpose1d(
                    &x,
                    &w,
                    b,
                    in_ch,
                    out_ch,
                    kernel,
                    stride,
                    padding,
                    output_padding,
                    groups,
                ),
                &conv_transpose1d_reference(
                    &x,
                    &w,
                    b,
                    in_ch,
                    out_ch,
                    kernel,
                    stride,
                    padding,
                    output_padding,
                    groups,
                ),
            );
        }
    }
}

#[test]
fn sdpa_is_bitwise_identical_to_reference() {
    let mut rng = Rng(0x8EBC_6AF0_9C88_C6E3);
    for &(qh, kv, dq, dv) in &[
        (1, 1, 1, 1),
        (1, 7, 5, 3),
        (1, 33, 64, 64),
        (3, 9, 13, 11),
        (4, 50, 32, 40),
        (8, 130, 64, 64),
        (16, 200, 128, 128),
    ] {
        let q = rng.vec(qh * dq);
        let k = rng.vec(kv * dq);
        let v = rng.vec(kv * dv);
        let mask: Vec<f32> = (0..qh * kv)
            .map(|i| if i % 5 == 3 { f32::NEG_INFINITY } else { 0.0 })
            .collect();
        let scale = 1.0 / (dq as f32).sqrt();
        for with_mask in [false, true] {
            let m = with_mask.then_some(&mask[..]);
            let want = sdpa_reference(&q, &k, &v, m, qh, kv, dq, dv, scale);
            assert_bits_eq(
                &format!("sdpa qh={qh} kv={kv} dq={dq} dv={dv} mask={with_mask}"),
                &sdpa(&q, &k, &v, m, qh, kv, dq, dv, scale),
                &want,
            );

            // The strided entry point reads K/V rows in place from a
            // wider time-major buffer; it must match the contiguous one.
            let (k_stride, v_stride) = (dq + 3, dv + 5);
            let (k_base, v_base) = (2, 4);
            let mut kb = vec![f32::NAN; k_base + kv * k_stride];
            let mut vb = vec![f32::NAN; v_base + kv * v_stride];
            for j in 0..kv {
                kb[k_base + j * k_stride..k_base + j * k_stride + dq]
                    .copy_from_slice(&k[j * dq..(j + 1) * dq]);
                vb[v_base + j * v_stride..v_base + j * v_stride + dv]
                    .copy_from_slice(&v[j * dv..(j + 1) * dv]);
            }
            assert_bits_eq(
                &format!("sdpa_strided qh={qh} kv={kv} dq={dq} dv={dv} mask={with_mask}"),
                &sdpa_strided(
                    &q, &kb, k_base, k_stride, &vb, v_base, v_stride, m, qh, kv, dq, dv, scale,
                ),
                &want,
            );
        }
    }
}

/// The original table builder, with `powf` evaluated per (position,
/// feature) pair. Kept structurally identical (cos and sin computed side
/// by side from one angle) because release builds may fuse the pair into
/// a single `sincos` call, which is not always bit-equal to two separate
/// calls; the optimized builder must be compared against this shape.
fn rope_tables_reference(seq: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for t in 0..seq {
        for d in 0..half {
            let freq = theta.powf(-2.0 * d as f32 / rotary_dim as f32);
            let ang = t as f32 * freq;
            cos[t * half + d] = ang.cos();
            sin[t * half + d] = ang.sin();
        }
    }
    (cos, sin)
}

#[test]
fn rope_tables_match_original_builder_bitwise() {
    for &(seq, rotary_dim, theta) in &[
        (1, 2, 10_000.0f32),
        (37, 64, 10_000.0),
        (5, 128, 1_000_000.0),
    ] {
        let half = rotary_dim / 2;
        let theta = std::hint::black_box(theta);
        let (want_cos, want_sin) = rope_tables_reference(seq, rotary_dim, theta);
        let (cos, sin) = rope_tables(seq, rotary_dim, theta);
        assert_bits_eq("rope cos", &cos, &want_cos);
        assert_bits_eq("rope sin", &sin, &want_sin);
        // A decode step asks for one position at a time; it must agree
        // with the same row of the full table.
        for t in [0, seq / 2, seq - 1] {
            let (c1, s1) = rope_tables_range(t, 1, rotary_dim, theta);
            assert_bits_eq("rope cos row", &c1, &want_cos[t * half..(t + 1) * half]);
            assert_bits_eq("rope sin row", &s1, &want_sin[t * half..(t + 1) * half]);
        }
    }
}

/// Verbatim copy of the head-major half-split loop the S3, Qwen3-ASR and
/// Granite decoders carried locally (`values [heads, rows, dim]`,
/// `[rows, half]` tables).
fn rope_neox_reference(
    values: &mut [f32],
    heads: usize,
    seq: usize,
    dim: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = rotary_dim / 2;
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            for d in 0..half {
                let a = values[base + d];
                let b = values[base + half + d];
                let table = t * half + d;
                values[base + d] = a * cos[table] - b * sin[table];
                values[base + half + d] = b * cos[table] + a * sin[table];
            }
        }
    }
}

/// The MOSS interleaved loop. GLM-ASR's copy writes its second output as
/// `a * sin + b * cos`; IEEE addition is commutative, so the same bits
/// come out, which `rope_interleaved_matches_both_local_forms` checks.
#[allow(clippy::too_many_arguments)]
fn rope_interleaved_reference(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    dim: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
    glm_operand_order: bool,
) {
    let half = rotary_dim / 2;
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            for d in 0..half {
                let a = x[base + 2 * d];
                let b = x[base + 2 * d + 1];
                let c = cos[t * half + d];
                let s = sin[t * half + d];
                x[base + 2 * d] = a * c - b * s;
                // Clippy sees the two arms as the same expression; the test exists
                // to prove the swapped operand order gives the same bits.
                #[allow(clippy::if_same_then_else)]
                let second = if glm_operand_order {
                    a * s + b * c
                } else {
                    b * c + a * s
                };
                x[base + 2 * d + 1] = second;
            }
        }
    }
}

#[test]
fn rope_neox_matches_local_decoder_loops() {
    let mut rng = Rng(0xC2B2_AE3D_27D4_EB4F);
    // (heads, seq, dim, rotary_dim): full-width rotation, partial rotation
    // with a pass-through tail, a single decode row, and an odd head count.
    for &(heads, seq, dim, rotary) in &[
        (1, 1, 2, 2),
        (2, 5, 8, 8),
        (3, 7, 16, 8),
        (4, 1, 64, 64),
        (16, 33, 128, 128),
        (2, 9, 20, 12),
    ] {
        let x = rng.vec(heads * seq * dim);
        let (cos, sin) = rope_tables(seq, rotary, 10_000.0);
        let mut want = x.clone();
        rope_neox_reference(&mut want, heads, seq, dim, rotary, &cos, &sin);
        let mut got = x.clone();
        rope_neox(&mut got, heads, seq, rotary, &cos, &sin);
        assert_bits_eq(
            &format!("rope_neox {heads}x{seq}x{dim}/{rotary}"),
            &got,
            &want,
        );
    }
}

#[test]
fn rope_interleaved_matches_both_local_forms() {
    let mut rng = Rng(0x1656_67B1_9E37_79F9);
    for &(heads, seq, dim, rotary) in &[
        (1, 1, 2, 2),
        (2, 5, 8, 8),
        (3, 7, 16, 8),
        (16, 33, 128, 128),
        (2, 9, 20, 12),
    ] {
        let x = rng.vec(heads * seq * dim);
        let (cos, sin) = rope_tables(seq, rotary, 10_000.0);
        let mut got = x.clone();
        rope_interleaved(&mut got, heads, seq, rotary, &cos, &sin);
        for glm in [false, true] {
            let mut want = x.clone();
            rope_interleaved_reference(&mut want, heads, seq, dim, rotary, &cos, &sin, glm);
            assert_bits_eq(
                &format!("rope_interleaved {heads}x{seq}x{dim}/{rotary} glm={glm}"),
                &got,
                &want,
            );
        }
    }
}

/// MOSS's fused-QKV head split, verbatim: `qkv [seq, 3 * dim]` into one
/// head-major plane per third.
fn split_qkv_reference(
    qkv: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = heads * head_dim;
    let mut qh = vec![0.0f32; heads * seq * head_dim];
    let mut kh = qh.clone();
    let mut vh = qh.clone();
    for h in 0..heads {
        for t in 0..seq {
            for d in 0..head_dim {
                qh[(h * seq + t) * head_dim + d] = qkv[t * 3 * dim + h * head_dim + d];
                kh[(h * seq + t) * head_dim + d] = qkv[t * 3 * dim + dim + h * head_dim + d];
                vh[(h * seq + t) * head_dim + d] = qkv[t * 3 * dim + 2 * dim + h * head_dim + d];
            }
        }
    }
    (qh, kh, vh)
}

#[test]
fn head_split_attention_merge_matches_the_per_head_loop() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for &(seq, heads, head_dim) in &[(1, 1, 2), (5, 2, 8), (9, 3, 16), (40, 4, 32), (130, 8, 64)] {
        let dim = heads * head_dim;
        let qkv = rng.vec(seq * 3 * dim);
        let (qr, kr, vr) = split_qkv_reference(&qkv, seq, heads, head_dim);
        let q = split_heads_strided(&qkv, 3 * dim, 0, seq, heads, head_dim);
        let k = split_heads_strided(&qkv, 3 * dim, dim, seq, heads, head_dim);
        let v = split_heads_strided(&qkv, 3 * dim, 2 * dim, seq, heads, head_dim);
        assert_bits_eq("split q", &q, &qr);
        assert_bits_eq("split k", &k, &kr);
        assert_bits_eq("split v", &v, &vr);

        // The dense (non-fused) split reads rows of `dim` floats.
        let dense = rng.vec(seq * dim);
        let plane = split_heads(&dense, seq, heads, head_dim);
        for h in 0..heads {
            for t in 0..seq {
                let want = &dense[t * dim + h * head_dim..t * dim + (h + 1) * head_dim];
                let got = &plane[(h * seq + t) * head_dim..(h * seq + t + 1) * head_dim];
                assert_bits_eq("split_heads row", got, want);
            }
        }
        assert_bits_eq(
            "merge(split) roundtrip",
            &merge_heads(&plane, seq, heads, head_dim),
            &dense,
        );

        // The MOSS attention core: additive mask, per-head sdpa into a
        // scratch Vec, copy into the output plane.
        let mask: Vec<f32> = (0..seq * seq)
            .map(|i| if (i / seq) < (i % seq) { f32::MIN } else { 0.0 })
            .collect();
        let scale = (head_dim as f32).powf(-0.5);
        for with_mask in [false, true] {
            let m = with_mask.then_some(&mask[..]);
            let mut want = vec![0.0f32; qr.len()];
            for h in 0..heads {
                let plane = h * seq * head_dim;
                let o = sdpa(
                    &qr[plane..plane + seq * head_dim],
                    &kr[plane..plane + seq * head_dim],
                    &vr[plane..plane + seq * head_dim],
                    m,
                    seq,
                    seq,
                    head_dim,
                    head_dim,
                    scale,
                );
                want[plane..plane + o.len()].copy_from_slice(&o);
            }
            assert_bits_eq(
                &format!("mha seq={seq} heads={heads} hd={head_dim} mask={with_mask}"),
                &mha(&q, &k, &v, m, heads, seq, seq, head_dim, scale),
                &want,
            );
        }
    }
}

#[test]
fn mha_handles_cross_attention_lengths() {
    let mut rng = Rng(0x94D0_49BB_1331_11EB);
    let (heads, seq_q, seq_kv, hd) = (3, 4, 11, 8);
    let q = rng.vec(heads * seq_q * hd);
    let k = rng.vec(heads * seq_kv * hd);
    let v = rng.vec(heads * seq_kv * hd);
    let mut want = vec![0.0f32; q.len()];
    for h in 0..heads {
        let o = sdpa(
            &q[h * seq_q * hd..(h + 1) * seq_q * hd],
            &k[h * seq_kv * hd..(h + 1) * seq_kv * hd],
            &v[h * seq_kv * hd..(h + 1) * seq_kv * hd],
            None,
            seq_q,
            seq_kv,
            hd,
            hd,
            0.25,
        );
        want[h * seq_q * hd..(h + 1) * seq_q * hd].copy_from_slice(&o);
    }
    assert_bits_eq(
        "mha cross",
        &mha(&q, &k, &v, None, heads, seq_q, seq_kv, hd, 0.25),
        &want,
    );
}

/// Micro-benchmark for the shared kernels. Ignored by default; run with
/// `cargo test -p turbospark-audio --release --lib -- --ignored --nocapture
/// kernel_bench`. Reports the median of several runs after discarding a
/// warmup, so compare only runs taken on the same quiet, AC-powered host.
#[test]
#[ignore = "timing harness, not a correctness test"]
fn kernel_bench() {
    use std::time::Instant;

    fn median_ms(mut f: impl FnMut()) -> f64 {
        f(); // warmup, discarded
        let mut runs: Vec<f64> = (0..9)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        runs[runs.len() / 2]
    }

    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut sink = 0.0f32; // keeps the optimizer from dropping results

    for &(m, k, n) in &[
        (1, 1024, 1024),
        (1, 1024, 4096),
        (1, 1024, 151936),
        (64, 1024, 1024),
    ] {
        let x = rng.vec(m * k);
        let w = rng.vec(n * k);
        let ms = median_ms(|| sink += linear(&x, &w, None, m, k, n)[0]);
        println!("linear m={m} k={k} n={n}: {ms:.3} ms");
    }
    for &(ic, oc, ks, seq, dil) in &[
        (64, 64, 7, 4000, 1),
        (128, 128, 3, 2000, 9),
        (1, 32, 7, 24000, 1),
    ] {
        let x = rng.vec(ic * seq);
        let w = rng.vec(oc * ic * ks);
        let pad = (ks - 1) * dil / 2;
        let ms = median_ms(|| sink += conv1d(&x, &w, None, ic, oc, ks, 1, pad, dil, 1)[0]);
        println!("conv1d in={ic} out={oc} k={ks} d={dil} seq={seq}: {ms:.3} ms");
    }
    for &(ic, oc, ks, stride, seq) in &[
        (256, 128, 16, 8, 200),
        (128, 64, 8, 4, 1600),
        (64, 1, 16, 8, 800),
    ] {
        let x = rng.vec(ic * seq);
        let w = rng.vec(ic * oc * ks);
        let pad = (ks - stride) / 2;
        let ms =
            median_ms(|| sink += conv_transpose1d(&x, &w, None, ic, oc, ks, stride, pad, 0, 1)[0]);
        println!("conv_transpose1d in={ic} out={oc} k={ks} s={stride} seq={seq}: {ms:.3} ms");
    }
    for &(qh, kv, dq) in &[(1, 1500, 64), (1, 4000, 128), (16, 1500, 64)] {
        let q = rng.vec(qh * dq);
        let k = rng.vec(kv * dq);
        let v = rng.vec(kv * dq);
        let ms = median_ms(|| sink += sdpa(&q, &k, &v, None, qh, kv, dq, dq, 0.125)[0]);
        println!("sdpa qh={qh} kv={kv} d={dq}: {ms:.3} ms");
    }
    println!("sink {sink}");
}

// ---------------------------------------------------------------------
// Shared helpers lifted out of the codec/VAD/TTS families. Each
// `*_reference` below is the family's original code, kept verbatim.
// ---------------------------------------------------------------------

/// mimi `layernorm_rows` (explicit loops, `0.0` start).
fn layernorm_rows_reference(
    x: &mut [f32],
    frames: usize,
    dim: usize,
    w: &[f32],
    b: &[f32],
    eps: f32,
) {
    for t in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..dim {
            mean += x[c * frames + t];
        }
        mean /= dim as f32;
        let mut var = 0.0f32;
        for c in 0..dim {
            let dd = x[c * frames + t] - mean;
            var += dd * dd;
        }
        var /= dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..dim {
            x[c * frames + t] = (x[c * frames + t] - mean) * inv * w[c] + b[c];
        }
    }
}

/// vocos `layernorm_affine_free`.
fn layernorm_affine_free_reference(x: &mut [f32], ch: usize, frames: usize, eps: f32) {
    for f in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..ch {
            mean += x[c * frames + f];
        }
        mean /= ch as f32;
        let mut var = 0.0f32;
        for c in 0..ch {
            let d = x[c * frames + f] - mean;
            var += d * d;
        }
        var /= ch as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..ch {
            x[c * frames + f] = (x[c * frames + f] - mean) * inv;
        }
    }
}

/// vocos `layernorm_channels` (weight always, bias optional).
fn layernorm_channels_reference(
    x: &mut [f32],
    ch: usize,
    frames: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
) {
    for f in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..ch {
            mean += x[c * frames + f];
        }
        mean /= ch as f32;
        let mut var = 0.0f32;
        for c in 0..ch {
            let d = x[c * frames + f] - mean;
            var += d * d;
        }
        var /= ch as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..ch {
            x[c * frames + f] = (x[c * frames + f] - mean) * inv * w[c] + b.map_or(0.0, |bb| bb[c]);
        }
    }
}

#[test]
fn layernorm_cm_matches_family_references_bitwise() {
    let mut rng = Rng(0x1357_9bdf_2468_ace1);
    for &(ch, frames) in &[(1, 1), (2, 3), (8, 5), (64, 17), (96, 4), (257, 9)] {
        for &eps in &[1e-5f32, 1e-6, 1e-4] {
            let x = rng.vec(ch * frames);
            let w = rng.vec(ch);
            let b = rng.vec(ch);

            let mut want = x.clone();
            layernorm_rows_reference(&mut want, frames, ch, &w, &b, eps);
            let mut got = x.clone();
            layernorm_cm(&mut got, ch, frames, Some(&w), Some(&b), eps);
            assert_bits_eq("mimi layernorm_rows", &got, &want);

            let mut want = x.clone();
            layernorm_affine_free_reference(&mut want, ch, frames, eps);
            let mut got = x.clone();
            layernorm_cm(&mut got, ch, frames, None, None, eps);
            assert_bits_eq("vocos affine free", &got, &want);

            let mut want = x.clone();
            layernorm_channels_reference(&mut want, ch, frames, &w, None, eps);
            let mut got = x.clone();
            layernorm_cm(&mut got, ch, frames, Some(&w), None, eps);
            assert_bits_eq("vocos channels, no bias", &got, &want);

            let mut want = x.clone();
            layernorm_channels_reference(&mut want, ch, frames, &w, Some(&b), eps);
            let mut got = x.clone();
            layernorm_cm(&mut got, ch, frames, Some(&w), Some(&b), eps);
            assert_bits_eq("vocos channels, bias", &got, &want);
        }
    }
}

/// Constant frames (every channel equal) make the variance exactly zero
/// and exercise the `-0.0`/`0.0` corner of the channel sum, including a
/// column of negative zeros.
#[test]
fn layernorm_cm_degenerate_columns_match_references_bitwise() {
    let ch = 6;
    let frames = 4;
    let w = [1.5f32, -2.0, 0.0, 0.25, -0.0, 3.0];
    let b = [0.5f32, -0.5, 0.0, 1.0, 2.0, -3.0];
    let mut x = vec![0.0f32; ch * frames];
    let col_values = [0.0f32, -0.0, 3.5, -2.25];
    for c in 0..ch {
        for (t, v) in col_values.iter().enumerate() {
            x[c * frames + t] = *v;
        }
    }
    let mut want = x.clone();
    layernorm_rows_reference(&mut want, frames, ch, &w, &b, 1e-5);
    let mut got = x.clone();
    layernorm_cm(&mut got, ch, frames, Some(&w), Some(&b), 1e-5);
    assert_bits_eq("loop reference", &got, &want);

    // The vocos forms: weight with and without bias, and affine-free.
    let mut want = x.clone();
    layernorm_channels_reference(&mut want, ch, frames, &w, None, 1e-6);
    let mut got = x.clone();
    layernorm_cm(&mut got, ch, frames, Some(&w), None, 1e-6);
    assert_bits_eq("vocos no bias", &got, &want);
    let mut want = x.clone();
    layernorm_affine_free_reference(&mut want, ch, frames, 1e-6);
    let mut got = x.clone();
    layernorm_cm(&mut got, ch, frames, None, None, 1e-6);
    assert_bits_eq("vocos affine free", &got, &want);
}

#[test]
fn sigmoid_and_elu_match_inline_originals_bitwise() {
    let mut rng = Rng(0x0dd_ba11_f00d);
    let mut xs = rng.vec(4096);
    xs.extend([
        0.0, -0.0, 1e-30, -1e-30, 20.0, -20.0, 88.0, -88.0, 104.0, -104.0,
    ]);
    for &v in &xs {
        let want = 1.0 / (1.0 + (-v).exp());
        assert_eq!(sigmoid(v).to_bits(), want.to_bits(), "sigmoid({v})");
    }
    // encodec / mimi `elu`, verbatim.
    fn elu_reference(x: &mut [f32]) {
        for v in x.iter_mut() {
            if *v <= 0.0 {
                *v = v.exp() - 1.0;
            }
        }
    }
    let mut want = xs.clone();
    elu_reference(&mut want);
    let mut got = xs.clone();
    elu(&mut got);
    assert_bits_eq("elu", &got, &want);
}

// The dense loops are the originals (MiMo bin-outer, Vocos band-outer, both
// from +0.0), kept index-for-index. The shared projector must reproduce them
// for the non-negative finite magnitudes both front ends feed it.
#[allow(clippy::needless_range_loop)]
#[test]
fn mel_projector_matches_both_dense_front_end_loops_bitwise() {
    use crate::mel::{mel_projector_cached, MelScale};
    let mut rng = Rng(0x0bad_cafe_d00d_f00d);
    for &(n_mels, nfft, sr, fmin, fmax) in &[
        (100usize, 1024usize, 24_000u32, 0.0f32, None),
        (128, 960, 24_000, 0.0, None),
        (80, 400, 16_000, 20.0, Some(7600.0f32)),
    ] {
        let projector = mel_projector_cached(n_mels, nfft, sr, fmin, fmax, MelScale::Htk).unwrap();
        let fb = projector.filterbank();
        assert_eq!(fb.num_mels, n_mels);
        let bins = fb.num_bins;
        // Magnitude-like input: non-negative with exact zeros.
        let frames = 5;
        let mags: Vec<f32> = rng.vec(bins * frames).iter().map(|v| v.abs()).collect();
        for f in 0..frames {
            let mut want_mimo = vec![0.0f32; n_mels];
            for b in 0..bins {
                let mag = mags[b * frames + f];
                for (m, slot) in want_mimo.iter_mut().enumerate() {
                    *slot += mag * fb.weights[m * bins + b];
                }
            }
            let mut want_vocos = vec![0.0f32; n_mels];
            for m in 0..n_mels {
                let weights = &fb.weights[m * bins..(m + 1) * bins];
                let mut acc = 0.0f32;
                for b in 0..bins {
                    acc += mags[b * frames + f] * weights[b];
                }
                want_vocos[m] = acc;
            }
            let spectrum: Vec<f32> = (0..bins).map(|b| mags[b * frames + f]).collect();
            let mut got = Vec::new();
            projector.project_into(&spectrum, &mut got).unwrap();
            assert_bits_eq("mimo dense loop", &got, &want_mimo);
            assert_bits_eq("vocos dense loop", &got, &want_vocos);
        }
    }
}

/// kokoro's per-channel inline snake loop, verbatim, against `ops::snake`.
#[test]
fn ops_snake_matches_kokoro_inline_snake_bitwise() {
    let mut rng = Rng(0x5eed_0f5a_4e00_0001);
    let (c, seq) = (7, 33);
    let x = rng.vec(c * seq);
    // Alphas must be nonzero in the checkpoint; keep them away from 0.
    let alpha: Vec<f32> = rng.vec(c).iter().map(|v| v + 8.5).collect();
    let mut want = x.clone();
    for ch in 0..c {
        let a1 = alpha[ch];
        for v in &mut want[ch * seq..(ch + 1) * seq] {
            *v += (a1 * *v).sin().powi(2) / a1;
        }
    }
    let mut got = x.clone();
    snake(&mut got, &alpha[..c], c, seq);
    assert_bits_eq("kokoro snake", &got, &want);
}

/// dacvae's local snake, verbatim, against `codec::wnconv::snake1d`.
#[test]
fn snake1d_matches_dacvae_local_snake_bitwise() {
    let mut rng = Rng(0x00da_c0de_1234_5678);
    let (c, seq) = (6, 41);
    let x = rng.vec(c * seq);
    // Includes a zero alpha: the 1e-9 guard must behave identically.
    let mut alpha: Vec<f32> = rng.vec(c);
    alpha[0] = 0.0;
    alpha[1] = -1e-9;
    let mut want = x.clone();
    for ch in 0..c {
        let a = alpha[ch];
        let recip = 1.0 / (a + 1e-9);
        for f in 0..seq {
            let v = &mut want[ch * seq + f];
            *v += recip * (a * *v).sin().powi(2);
        }
    }
    let mut got = x.clone();
    crate::codec::wnconv::snake1d(&mut got, &alpha, c, seq);
    assert_bits_eq("dacvae snake", &got, &want);
}
