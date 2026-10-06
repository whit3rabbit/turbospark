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
