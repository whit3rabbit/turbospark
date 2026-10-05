//! Runs the `whisper_encoder.metal` kernels on real Metal hardware against
//! the CPU reference kernels (`compute::whisper` and the `compute::vision`
//! helpers they borrow), at whisper shapes and at small hand-check shapes.
#![cfg(target_os = "macos")]

use turbospark_gpu::{
    encode_whisper_add, encode_whisper_attn_step, encode_whisper_gelu_erf,
    encode_whisper_layer_norm, encode_whisper_matmul_bias, encode_whisper_softmax_rows,
    encode_whisper_transpose_pos, F32View, MetalBuffer, MetalContext,
};

/// Deterministic, INDEPENDENT pseudo-random values in `[-1, 1)` (splitmix64
/// on the flat index, the same fixture family as `whisper_conv_parity.rs`):
/// no period the kernel's tile strides can alias against, bounded magnitude
/// so f32 accumulation error stays predictable.
fn unit(i: usize, salt: u64) -> f32 {
    let mut z = (i as u64)
        .wrapping_add(salt)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn buf(context: &MetalContext, data: &[f32]) -> MetalBuffer {
    context.new_buffer_with_data(data)
}

fn out_buf(context: &MetalContext, elements: usize) -> MetalBuffer {
    context.new_output_buffer((elements * 4) as u64)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "same length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// One GEMM dispatch, committed and read back. The strides parameterize
/// the head-slice cases; `w_transposed` and `out_scale` exercise the
/// attention-value-mix and the score-scale paths.
#[allow(clippy::too_many_arguments)]
fn run_matmul(
    context: &mut MetalContext,
    a: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    a_stride: usize,
    w_stride: usize,
    out_stride: usize,
    w_transposed: bool,
    out_scale: f32,
) -> (Vec<f32>, MetalBuffer) {
    let a_buf = buf(context, a);
    let w_buf = buf(context, w);
    let bias_buf = bias
        .map(|b| buf(context, b))
        .unwrap_or_else(|| a_buf.clone());
    let out_len = (m - 1) * out_stride + n;
    let out = out_buf(context, out_len);
    let pass = context.begin_pass();
    encode_whisper_matmul_bias(
        context,
        &pass,
        F32View::new(&a_buf),
        F32View::new(&w_buf),
        bias.as_ref().map(|_| F32View::new(&bias_buf)),
        F32View::new(&out),
        m as u32,
        k as u32,
        n as u32,
        a_stride as u32,
        w_stride as u32,
        out_stride as u32,
        w_transposed,
        out_scale,
    )
    .expect("matmul dispatch");
    pass.commit_and_wait();
    let raw = turbospark_gpu::read_f32_buffer(&out, out_len);
    (raw, out)
}

#[test]
fn parity_matmul_dense_with_bias() {
    let mut context = MetalContext::new().expect("Metal device available");
    // Whisper FC1 shape at tiny.en: seq 64 rows (a tail window's worth is
    // still a GEMM), d 384 in, 1536 out.
    let (m, k, n) = (64usize, 384usize, 1536usize);
    let a: Vec<f32> = (0..m * k).map(|i| unit(i, 1)).collect();
    let w: Vec<f32> = (0..n * k).map(|i| unit(i, 2)).collect();
    let bias: Vec<f32> = (0..n).map(|i| unit(i, 3) * 0.1).collect();
    let (gpu, _) = run_matmul(
        &mut context,
        &a,
        &w,
        Some(&bias),
        m,
        k,
        n,
        k,
        k,
        n,
        false,
        1.0,
    );
    let cpu = turbospark_compute::vision::matmul_bias(&a, &w, Some(&bias), m, k, n);
    let err = max_diff(&cpu, &gpu);
    // f32 accumulation on both sides, 384-long dots: measured noise is
    // ~1e-4; structural bugs (transposed read, wrong stride) read 10+.
    assert!(err < 2e-3, "dense matmul max abs diff {err}");
}

#[test]
fn parity_matmul_head_slice_with_transpose_and_scale() {
    let mut context = MetalContext::new().expect("Metal device available");
    // A head slice of a [seq, d] stream: rows are d_model apart while the
    // reduction walks head_dim, exactly the encoder's scores/mix reads.
    let (seq, d, head_dim) = (48usize, 384usize, 64usize);
    let stream: Vec<f32> = (0..seq * d).map(|i| unit(i, 4)).collect();
    // Scores: Q_h x K_h^T, w = K head slice [seq, head_dim] rows d apart.
    let q_off = 2 * head_dim; // head 2
    let (scores_gpu, _) = run_matmul(
        &mut context,
        &stream[q_off..],
        &stream[q_off..],
        None,
        seq,
        head_dim,
        seq,
        d,
        d,
        seq,
        false,
        (head_dim as f32).powf(-0.5),
    );
    let mut scores_cpu = vec![0.0f32; seq * seq];
    let scale = (head_dim as f32).powf(-0.5);
    for i in 0..seq {
        for j in 0..seq {
            let mut dot = 0.0f32;
            for t in 0..head_dim {
                dot += stream[i * d + q_off + t] * stream[j * d + q_off + t];
            }
            scores_cpu[i * seq + j] = dot * scale;
        }
    }
    let err = max_diff(&scores_cpu, &scores_gpu);
    assert!(err < 2e-3, "scores head slice max abs diff {err}");

    // Mix: scores x V_h with the transposed-B read (V rows are keys).
    let v: Vec<f32> = (0..seq * d).map(|i| unit(i, 5)).collect();
    let (read_mix, _) = run_matmul(
        &mut context,
        &scores_cpu,
        &v[q_off..],
        None,
        seq,
        seq,
        head_dim,
        seq,
        d,
        d,
        true,
        1.0,
    );
    let mut mix_cpu = vec![0.0f32; seq * head_dim];
    for i in 0..seq {
        for t in 0..seq {
            let s = scores_cpu[i * seq + t];
            for o in 0..head_dim {
                mix_cpu[i * head_dim + o] += s * v[t * d + q_off + o];
            }
        }
    }
    // The GPU mix wrote the head's column slice of a [seq, d] stream
    // (out_stride = d_model; the dispatch's output view starts at column
    // 0 of a fresh buffer, so the slice is columns 0..head_dim); extract
    // the same slice from the raw readback.
    let mix_gpu = extract_head_columns(&read_mix, seq, d, head_dim, 0);
    let err = max_diff(&mix_cpu, &mix_gpu);
    // 1500-long f32 dots at seq 48: measured 1e-4; a wrong transposed read
    // reads orders of magnitude higher.
    assert!(err < 2e-3, "value mix max abs diff {err}");
}

#[test]
fn parity_softmax_rows() {
    let mut context = MetalContext::new().expect("Metal device available");
    let (rows, n) = (6usize, 1500usize);
    let x: Vec<f32> = (0..rows * n).map(|i| unit(i, 6) * 30.0).collect();
    let x_buf = buf(&context, &x);
    let y = out_buf(&context, rows * n);
    let pass = context.begin_pass();
    encode_whisper_softmax_rows(
        &mut context,
        &pass,
        F32View::new(&x_buf),
        F32View::new(&y),
        rows as u32,
        n as u32,
    )
    .expect("softmax dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&y, rows * n);
    // Independent reference: max-subtracted softmax per row.
    let mut want = vec![0.0f32; rows * n];
    for r in 0..rows {
        let row = &x[r * n..(r + 1) * n];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let total: f32 = exps.iter().sum();
        for (i, e) in exps.iter().enumerate() {
            want[r * n + i] = e / total;
        }
    }
    let err = max_diff(&want, &gpu);
    assert!(err < 1e-6, "softmax max abs diff {err}");
    // Row sums are 1.
    for r in 0..rows {
        let sum: f32 = gpu[r * n..(r + 1) * n].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {r} sums to {sum}");
    }
}

#[test]
fn parity_gelu_erf() {
    let mut context = MetalContext::new().expect("Metal device available");
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| unit(i, 7) * 6.0).collect();
    let x_buf = buf(&context, &x);
    let y = out_buf(&context, n);
    let pass = context.begin_pass();
    encode_whisper_gelu_erf(
        &mut context,
        &pass,
        F32View::new(&x_buf),
        F32View::new(&y),
        n as u32,
    )
    .expect("gelu dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&y, n);
    let cpu = turbospark_compute::vision::gelu_erf(&x);
    let err = max_diff(&cpu, &gpu);
    // The CPU evaluates erf in f64, the GPU in f32 (A&S both): the conv
    // parity measured 6.9e-5 on this pair; same bound here.
    assert!(err < 2e-4, "gelu max abs diff {err}");
}

#[test]
fn parity_add_transpose_pos_layer_norm() {
    let mut context = MetalContext::new().expect("Metal device available");
    let (seq, d) = (48usize, 384usize);

    // add
    let a: Vec<f32> = (0..seq * d).map(|i| unit(i, 8)).collect();
    let b: Vec<f32> = (0..seq * d).map(|i| unit(i, 9)).collect();
    let a_buf = buf(&context, &a);
    let b_buf = buf(&context, &b);
    let y = out_buf(&context, seq * d);
    let pass = context.begin_pass();
    encode_whisper_add(
        &mut context,
        &pass,
        F32View::new(&a_buf),
        F32View::new(&b_buf),
        F32View::new(&y),
        (seq * d) as u32,
    )
    .expect("add dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&y, seq * d);
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, v)| x + v).collect();
    assert_eq!(max_diff(&want, &gpu), 0.0, "add is exact");

    // transpose_pos: [d, seq] band-major + positional rows -> [seq, d]
    let band: Vec<f32> = (0..d * seq).map(|i| unit(i, 10)).collect();
    let pos: Vec<f32> = (0..seq * d).map(|i| unit(i, 11) * 0.01).collect();
    let band_buf = buf(&context, &band);
    let pos_buf = buf(&context, &pos);
    let y = out_buf(&context, seq * d);
    let pass = context.begin_pass();
    encode_whisper_transpose_pos(
        &mut context,
        &pass,
        F32View::new(&band_buf),
        F32View::new(&pos_buf),
        F32View::new(&y),
        seq as u32,
        d as u32,
    )
    .expect("transpose_pos dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&y, seq * d);
    let mut want = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d {
            want[t * d + i] = band[i * seq + t] + pos[t * d + i];
        }
    }
    assert_eq!(max_diff(&want, &gpu), 0.0, "transpose_pos is exact");

    // layer_norm per row, against compute::vision::layer_norm.
    let x: Vec<f32> = (0..seq * d).map(|i| unit(i, 12) * 5.0 + 2.0).collect();
    let weight: Vec<f32> = (0..d).map(|i| 0.5 + unit(i, 13).abs()).collect();
    let bias: Vec<f32> = (0..d).map(|i| unit(i, 14) * 0.1).collect();
    let eps = 1e-5f32;
    let x_buf = buf(&context, &x);
    let w_buf = buf(&context, &weight);
    let b_buf = buf(&context, &bias);
    let y = out_buf(&context, seq * d);
    let pass = context.begin_pass();
    encode_whisper_layer_norm(
        &mut context,
        &pass,
        F32View::new(&x_buf),
        F32View::new(&w_buf),
        F32View::new(&b_buf),
        F32View::new(&y),
        seq as u32,
        d as u32,
        eps,
    )
    .expect("layer_norm dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&y, seq * d);
    for t in 0..seq {
        let want =
            turbospark_compute::vision::layer_norm(&x[t * d..(t + 1) * d], &weight, &bias, eps);
        let err = max_diff(&want, &gpu[t * d..(t + 1) * d]);
        assert!(err < 1e-4, "layer_norm row {t} max abs diff {err}");
    }
}

#[test]
fn parity_attn_step_self_and_cross() {
    let mut context = MetalContext::new().expect("Metal device available");
    let d = 384usize;
    let heads = 6usize;
    let head_dim = d / heads;
    let scale = (head_dim as f32).powf(-0.5);
    let max_len = 64usize;

    // Cross-shaped first: caches pre-filled with `filled` rows, query is
    // one row, append off.
    let filled = 40usize;
    let k_cache: Vec<f32> = (0..max_len * d).map(|i| unit(i, 15)).collect();
    let v_cache: Vec<f32> = (0..max_len * d).map(|i| unit(i, 16)).collect();
    let q: Vec<f32> = (0..d).map(|i| unit(i, 17)).collect();
    let k_buf = buf(&context, &k_cache);
    let v_buf = buf(&context, &v_cache);
    let q_buf = buf(&context, &q);
    let out = out_buf(&context, d);
    let pass = context.begin_pass();
    encode_whisper_attn_step(
        &mut context,
        &pass,
        F32View::new(&q_buf),
        F32View::new(&q_buf),
        F32View::new(&q_buf),
        F32View::new(&k_buf),
        F32View::new(&v_buf),
        F32View::new(&out),
        0,
        filled as u32,
        d as u32,
        head_dim as u32,
        false,
        scale,
        false,
    )
    .expect("cross attn dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&out, d);
    // Per-head reference over the same caches (mirrors
    // compute::whisper::query_attention's math in f32).
    let mut want = vec![0.0f32; d];
    for h in 0..heads {
        let off = h * head_dim;
        let mut scores = vec![0.0f32; filled];
        let mut max = f32::NEG_INFINITY;
        for t in 0..filled {
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q[off + i] * k_cache[t * d + off + i];
            }
            scores[t] = dot * scale;
            max = max.max(scores[t]);
        }
        let total: f32 = scores.iter().map(|s| (s - max).exp()).sum();
        for t in 0..filled {
            let w = (scores[t] - max).exp() / total;
            for i in 0..head_dim {
                want[off + i] += w * v_cache[t * d + off + i];
            }
        }
    }
    let err = max_diff(&want, &gpu);
    assert!(err < 1e-3, "cross attn max abs diff {err}");

    // Self-shaped: append happens at pos, then attention over 0..=pos.
    let k_new: Vec<f32> = (0..d).map(|i| unit(i, 18)).collect();
    let v_new: Vec<f32> = (0..d).map(|i| unit(i, 19)).collect();
    let pos = filled;
    let k_buf = buf(&context, &k_cache);
    let v_buf = buf(&context, &v_cache);
    let kn_buf = buf(&context, &k_new);
    let vn_buf = buf(&context, &v_new);
    let q_buf = buf(&context, &q);
    let out = out_buf(&context, d);
    let pass = context.begin_pass();
    encode_whisper_attn_step(
        &mut context,
        &pass,
        F32View::new(&q_buf),
        F32View::new(&kn_buf),
        F32View::new(&vn_buf),
        F32View::new(&k_buf),
        F32View::new(&v_buf),
        F32View::new(&out),
        pos as u32,
        (pos + 1) as u32,
        d as u32,
        head_dim as u32,
        true,
        scale,
        false,
    )
    .expect("self attn dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&out, d);
    let mut k_full = k_cache.clone();
    let mut v_full = v_cache.clone();
    for i in 0..d {
        k_full[pos * d + i] = k_new[i];
        v_full[pos * d + i] = v_new[i];
    }
    let mut want = vec![0.0f32; d];
    let len = pos + 1;
    for h in 0..heads {
        let off = h * head_dim;
        let mut scores = vec![0.0f32; len];
        let mut max = f32::NEG_INFINITY;
        for t in 0..len {
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q[off + i] * k_full[t * d + off + i];
            }
            scores[t] = dot * scale;
            max = max.max(scores[t]);
        }
        let total: f32 = scores.iter().map(|s| (s - max).exp()).sum();
        for t in 0..len {
            let w = (scores[t] - max).exp() / total;
            for i in 0..head_dim {
                want[off + i] += w * v_full[t * d + off + i];
            }
        }
    }
    let err = max_diff(&want, &gpu);
    assert!(err < 1e-3, "self attn max abs diff {err}");
}

/// Pulls one head's column slice out of a `[seq, d_model]` strided readback.
fn extract_head_columns(
    raw: &[f32],
    seq: usize,
    d: usize,
    head_dim: usize,
    head_offset: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(seq * head_dim);
    for t in 0..seq {
        out.extend_from_slice(&raw[t * d + head_offset..t * d + head_offset + head_dim]);
    }
    out
}

#[test]
fn parity_transpose_and_transposed_attn() {
    let mut context = MetalContext::new().expect("Metal device available");
    let (seq, d, heads, head_dim) = (64usize, 384usize, 6usize, 64usize);
    let scale = (head_dim as f32).powf(-0.5);

    // transpose: [seq, d] -> [d, seq]
    let v: Vec<f32> = (0..seq * d).map(|i| unit(i, 21)).collect();
    let v_buf = buf(&context, &v);
    let vt = out_buf(&context, seq * d);
    let pass = context.begin_pass();
    turbospark_gpu::encode_whisper_transpose(
        &mut context,
        &pass,
        F32View::new(&v_buf),
        F32View::new(&vt),
        seq as u32,
        d as u32,
    )
    .expect("transpose dispatch");
    pass.commit_and_wait();
    let vt_gpu = turbospark_gpu::read_f32_buffer(&vt, seq * d);
    let mut vt_cpu = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d {
            vt_cpu[i * seq + t] = v[t * d + i];
        }
    }
    assert_eq!(max_diff(&vt_cpu, &vt_gpu), 0.0, "transpose is exact");

    // attention over the transposed cache matches the natural-layout math
    let q: Vec<f32> = (0..d).map(|i| unit(i, 22)).collect();
    let k: Vec<f32> = (0..seq * d).map(|i| unit(i, 23)).collect();
    let q_buf = buf(&context, &q);
    let k_buf = buf(&context, &k);
    let vt_buf = buf(&context, &vt_cpu);
    let out = out_buf(&context, d);
    let pass = context.begin_pass();
    encode_whisper_attn_step(
        &mut context,
        &pass,
        F32View::new(&q_buf),
        F32View::new(&q_buf),
        F32View::new(&q_buf),
        F32View::new(&k_buf),
        F32View::new(&vt_buf),
        F32View::new(&out),
        0,
        seq as u32,
        d as u32,
        head_dim as u32,
        false,
        scale,
        true,
    )
    .expect("transposed attn dispatch");
    pass.commit_and_wait();
    let gpu = turbospark_gpu::read_f32_buffer(&out, d);
    let mut want = vec![0.0f32; d];
    for h in 0..heads {
        let off = h * head_dim;
        let mut scores = vec![0.0f32; seq];
        let mut max = f32::NEG_INFINITY;
        for t in 0..seq {
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q[off + i] * k[t * d + off + i];
            }
            scores[t] = dot * scale;
            max = max.max(scores[t]);
        }
        let total: f32 = scores.iter().map(|s| (s - max).exp()).sum();
        for t in 0..seq {
            let w = (scores[t] - max).exp() / total;
            for i in 0..head_dim {
                want[off + i] += w * vt_cpu[(off + i) * seq + t];
            }
        }
    }
    let err = max_diff(&want, &gpu);
    assert!(err < 1e-3, "transposed attn max abs diff {err}");
}
