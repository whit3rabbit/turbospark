#![cfg(target_os = "macos")]
use turbospark_gpu::{Music3Device, Music3Encoding};

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite() && (a - b).abs() < 2e-5, "{a} vs {b}");
    }
}
fn pack(codes: &[u32], rows: usize, cols: usize, bits: u32) -> Vec<u8> {
    let stride = (cols * bits as usize).div_ceil(32);
    let mut out = vec![0u32; rows * stride];
    for r in 0..rows {
        for c in 0..cols {
            let bit = c * bits as usize;
            let i = r * stride + bit / 32;
            let shift = bit % 32;
            out[i] |= codes[r * cols + c] << shift;
            if shift + bits as usize > 32 {
                out[i + 1] |= codes[r * cols + c] >> (32 - shift);
            }
        }
    }
    out.iter().flat_map(|v| v.to_le_bytes()).collect()
}
#[test]
fn packed_float_formats_match_known_values_without_expansion() {
    let device = Music3Device::new().unwrap();
    for (encoding, bits, group, scale) in [
        (Music3Encoding::MxFp4, 4, 32, 127u8),
        (Music3Encoding::MxFp8, 8, 32, 127),
        (Music3Encoding::NvFp4, 4, 16, 56),
    ] {
        let cols = 64;
        let rows = 2;
        let codes: Vec<u32> = (0..rows * cols)
            .map(|i| {
                if bits == 8 {
                    [
                        0, 48, 56, 60, 64, 68, 72, 76, 128, 176, 184, 188, 192, 196, 200, 204,
                    ][i % 16]
                } else {
                    (i % 16) as u32
                }
            })
            .collect();
        let table = [
            0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
        ];
        let expected: Vec<f32> = (0..rows * cols).map(|i| table[i % 16]).collect();
        let data = pack(&codes, rows, cols, bits);
        let scales: Vec<u8> = (0..rows * cols / group)
            .map(|i| {
                if encoding == Music3Encoding::NvFp4 {
                    [48, 56, 64][i % 3]
                } else {
                    scale - 1 + (i % 3) as u8
                }
            })
            .collect();
        let expected: Vec<f32> = expected
            .iter()
            .enumerate()
            .map(|(i, v)| v * [0.5, 1.0, 2.0][(i / group) % 3])
            .collect();
        let weight = device
            .load_weight(&[rows, cols], &data, encoding, &[], &[], &scales)
            .unwrap();
        let ids = vec![1, 0];
        let actual = weight.embedding(&ids, cols).unwrap();
        close(&actual, &[&expected[cols..], &expected[..cols]].concat());
        let input: Vec<f32> = (0..cols * 3)
            .map(|i| (i % 9) as f32 * 0.125 - 0.5)
            .collect();
        let mut reference = Vec::new();
        for input in input.chunks(cols) {
            for row in expected.chunks(cols) {
                reference.push(input.iter().zip(row).map(|(x, w)| x * w).sum::<f32>() + 0.25);
            }
        }
        close(
            &weight
                .linear(&input, Some(&[0.25, 0.25]), 3, cols, rows)
                .unwrap(),
            &reference,
        );
    }
    assert_eq!(device.resident_weight_bytes(), 0);
}
#[test]
fn affine_all_bit_widths_handle_cross_word_values_and_nonuniform_groups() {
    let device = Music3Device::new().unwrap();
    for bits in [2, 3, 4, 5, 6, 8] {
        let rows = 3;
        let cols = 80;
        let group = 16;
        let codes: Vec<u32> = (0..rows * cols)
            .map(|i| (i * 7) as u32 % (1 << bits))
            .collect();
        let sc: Vec<f32> = (0..rows * cols / group)
            .map(|i| 0.01 * (i + 1) as f32)
            .collect();
        let off: Vec<f32> = (0..sc.len()).map(|i| -0.02 * i as f32).collect();
        let data = pack(&codes, rows, cols, bits);
        let w = device
            .load_weight(
                &[rows, cols],
                &data,
                Music3Encoding::Affine {
                    bits,
                    group_size: group,
                },
                &sc,
                &off,
                &[],
            )
            .unwrap();
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| codes[i] as f32 * sc[i / group] + off[i / group])
            .collect();
        close(&w.embedding(&[0, 1, 2], cols).unwrap(), &values);
    }
}
#[test]
fn convolution_and_transpose_match_scalar_reference_at_edges() {
    let device = Music3Device::new().unwrap();
    let ic = 3;
    let oc = 2;
    let kernel = 5;
    let len = 7;
    let weight: Vec<f32> = (0..ic * oc * kernel)
        .map(|i| (i % 7) as f32 * 0.02 - 0.04)
        .collect();
    let x: Vec<f32> = (0..ic * len).map(|i| (i % 9) as f32 * 0.1).collect();
    for transpose in [false, true] {
        for dilation in [1, 2] {
            if transpose && dilation > 1 {
                continue;
            }
            let stride = 2;
            let pad = 2;
            let outlen = if transpose {
                (len - 1) * stride + kernel - 2 * pad
            } else {
                (len + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1
            };
            let mut expected = vec![0.; oc * outlen];
            for o in 0..oc {
                for t in 0..outlen {
                    expected[o * outlen + t] = 0.1;
                    for i in 0..ic {
                        for k in 0..kernel {
                            let p = if transpose {
                                t as isize + pad as isize - k as isize
                            } else {
                                (t * stride + k * dilation) as isize - pad as isize
                            };
                            if p < 0 || (transpose && p as usize % stride != 0) {
                                continue;
                            }
                            let p = if transpose {
                                p as usize / stride
                            } else {
                                p as usize
                            };
                            if p < len {
                                let wi = if transpose {
                                    (i * oc + o) * kernel + k
                                } else {
                                    (o * ic + i) * kernel + k
                                };
                                expected[o * outlen + t] += x[i * len + p] * weight[wi];
                            }
                        }
                    }
                }
            }
            let shape = if transpose {
                vec![ic, oc, kernel]
            } else {
                vec![oc, ic, kernel]
            };
            let w = device
                .load_weight(&shape, &bytes(&weight), Music3Encoding::F32, &[], &[], &[])
                .unwrap();
            if transpose {
                assert!(w
                    .convolution(&x, None, ic, oc, kernel, stride, pad, 2, true)
                    .is_err());
            }
            close(
                &w.convolution(
                    &x,
                    Some(&[0.1, 0.1]),
                    ic,
                    oc,
                    kernel,
                    stride,
                    pad,
                    dilation,
                    transpose,
                )
                .unwrap(),
                &expected,
            );
        }
    }
}
#[test]
fn invalid_packing_nan_and_shape_are_refused_before_dispatch() {
    let d = Music3Device::new().unwrap();
    assert!(d
        .load_weight(&[1, 32], &[0; 16], Music3Encoding::MxFp4, &[], &[], &[255])
        .is_err());
    assert!(d
        .load_weight(
            &[1, 32],
            &[127; 32],
            Music3Encoding::MxFp8,
            &[],
            &[],
            &[127]
        )
        .is_err());
    assert!(d
        .load_weight(&[1, 16], &[0; 8], Music3Encoding::NvFp4, &[], &[], &[127])
        .is_err());
    assert!(d
        .load_weight(&[1, 32], &[0; 15], Music3Encoding::MxFp4, &[], &[], &[127])
        .is_err());
    let w = d
        .load_weight(
            &[1, 2],
            &bytes(&[1., 2.]),
            Music3Encoding::F32,
            &[],
            &[],
            &[],
        )
        .unwrap();
    assert!(w.embedding(&[-1], 2).is_err());
    assert!(w.linear(&[1.], None, 1, 2, 1).is_err());
    assert!(d
        .attention(&[0.], &[0.], &[0.], 1, 1, 1, 1, 1, 257, false, false, 0)
        .is_err());
}

#[test]
fn attention_matches_independent_reference_for_gqa_cache_and_causal_windows() {
    let d = Music3Device::new().unwrap();
    let (batch, queries, keys, heads, kh, dim) = (2, 3, 7, 4, 2, 128);
    let q: Vec<f32> = (0..batch * queries * heads * dim)
        .map(|i| ((i * 7) % 31) as f32 * 0.015 - 0.2)
        .collect();
    let k: Vec<f32> = (0..batch * keys * kh * dim)
        .map(|i| ((i * 11) % 37) as f32 * 0.02 - 0.3)
        .collect();
    let v: Vec<f32> = (0..k.len())
        .map(|i| ((i * 17) % 23) as f32 * 0.03 - 0.2)
        .collect();
    for time_major in [false, true] {
        for causal in [false, true] {
            let offset = if causal { 4 } else { 0 };
            let mut expected = vec![0.; q.len()];
            for b in 0..batch {
                for t in 0..queries {
                    for h in 0..heads {
                        let qi = ((b * queries + t) * heads + h) * dim;
                        let mut scores = Vec::new();
                        let count = if causal {
                            keys.min(t + offset + 1)
                        } else {
                            keys
                        };
                        for s in 0..count {
                            let ki = if time_major {
                                (s * batch * kh + b * kh + h / (heads / kh)) * dim
                            } else {
                                ((b * keys + s) * kh + h / (heads / kh)) * dim
                            };
                            scores.push(
                                (0..dim)
                                    .map(|f| q[qi + f] as f64 * k[ki + f] as f64)
                                    .sum::<f64>()
                                    / (dim as f64).sqrt(),
                            );
                        }
                        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let probabilities: Vec<f64> =
                            scores.iter().map(|s| (s - max).exp()).collect();
                        let denom: f64 = probabilities.iter().sum();
                        for f in 0..dim {
                            let value = (0..count)
                                .map(|s| {
                                    let ki = if time_major {
                                        (s * batch * kh + b * kh + h / (heads / kh)) * dim
                                    } else {
                                        ((b * keys + s) * kh + h / (heads / kh)) * dim
                                    };
                                    probabilities[s] * v[ki + f] as f64 / denom
                                })
                                .sum::<f64>();
                            expected[qi + f] = value as f32;
                        }
                    }
                }
            }
            close(
                &d.attention(
                    &q, &k, &v, batch, queries, keys, heads, kh, dim, time_major, causal, offset,
                )
                .unwrap(),
                &expected,
            );
        }
    }
}

#[test]
fn dense_checkpoint_precisions_preserve_values_and_projection_bias() {
    use half::{bf16, f16};
    let d = Music3Device::new().unwrap();
    let values = [0.125, -2., 3.5, 0., 4., 0.75, -1.25, 2.];
    for mode in [
        Music3Encoding::F32,
        Music3Encoding::F16,
        Music3Encoding::Bf16,
    ] {
        let data = match mode {
            Music3Encoding::F32 => bytes(&values),
            Music3Encoding::F16 => values
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            _ => values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
        };
        let w = d.load_weight(&[2, 4], &data, mode, &[], &[], &[]).unwrap();
        close(
            &w.embedding(&[1, 0], 4).unwrap(),
            &[&values[4..], &values[..4]].concat(),
        );
        let x = [1., 2., 3., 4.];
        let expected: Vec<f32> = values
            .chunks(4)
            .map(|row| row.iter().zip(x).map(|(w, x)| w * x).sum::<f32>() + 0.25)
            .collect();
        close(
            &w.linear(&x, Some(&[0.25, 0.25]), 1, 4, 2).unwrap(),
            &expected,
        );
        assert!(w
            .convolution(&x, None, 1, 2, 0, 1, usize::MAX, 1, false)
            .is_err());
    }
}
