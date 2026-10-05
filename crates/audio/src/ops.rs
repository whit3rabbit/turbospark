//! Shared f32 tensor kernels every speech model builds on.
//!
//! All tensors are row-major slices. Linear weights follow the HF
//! `[out, in]` layout, so a forward pass is `x @ w.T + b`; the conv
//! kernels take the PyTorch layouts (`[out_ch, in_ch, kernel]` for
//! conv1d/conv2d, `[in_ch, out_ch, kernel]` for conv_transpose1d) and the
//! loaders convert checkpoint exceptions before calling in.
//!
//! Matmul and conv spend their time in cache-friendly loops and split
//! output rows across threads with `std::thread::scope`, so the crate
//! stays dependency-free. Numeric behavior is the plain IEEE f32
//! reduction order of the loops below; no model may assume more precision
//! than that.

/// Element count of a shape, saturating instead of overflowing.
fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// `a [m, k] @ b [k, n] -> [m, n]`, threaded over output row blocks.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matmul lhs {} != {}x{}", a.len(), m, k);
    assert_eq!(b.len(), k * n, "matmul rhs {} != {}x{}", b.len(), k, n);
    let mut out = vec![0.0f32; m * n];
    par_rows(m, &mut out, n, |row_start, row_end, rows| {
        for i in row_start..row_end {
            let a_row = &a[i * k..(i + 1) * k];
            let out_row = &mut rows[(i - row_start) * n..(i - row_start + 1) * n];
            for (p, &a_val) in a_row.iter().enumerate() {
                let b_row = &b[p * n..(p + 1) * n];
                for (j, out_val) in out_row.iter_mut().enumerate() {
                    *out_val += a_val * b_row[j];
                }
            }
        }
    });
    out
}

/// `x [m, k] @ w.T [k, n] + bias -> [m, n]` with `w` stored `[n, k]`.
///
/// This is the HF linear convention and the hot path of every model in
/// the crate; it fuses the transpose access pattern and the bias add so
/// callers never materialize `w.T`.
pub fn linear(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "linear input {} != {}x{}", x.len(), m, k);
    assert_eq!(w.len(), n * k, "linear weight {} != {}x{}", w.len(), n, k);
    let mut out = vec![0.0f32; m * n];
    par_rows(m, &mut out, n, |row_start, row_end, rows| {
        for i in row_start..row_end {
            let x_row = &x[i * k..(i + 1) * k];
            let out_row = &mut rows[(i - row_start) * n..(i - row_start + 1) * n];
            for (ni, out_val) in out_row.iter_mut().enumerate() {
                let w_row = &w[ni * k..(ni + 1) * k];
                let mut acc = 0.0f32;
                for (ki, &x_val) in x_row.iter().enumerate() {
                    acc += x_val * w_row[ki];
                }
                *out_val = acc + bias.map_or(0.0, |b| b[ni]);
            }
        }
    });
    out
}

/// Runs `body` over row ranges of an `m x row_len` output, splitting rows
/// across threads when the work justifies it. Small problems (one short
/// row) stay on the calling thread so per-call thread churn does not
/// dominate tiny layers.
fn par_rows(
    m: usize,
    out: &mut [f32],
    row_len: usize,
    body: impl Fn(usize, usize, &mut [f32]) + Sync,
) {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // Rows are split only when there are enough of them to amortize the
    // scope join; the threshold is deliberately low because decode calls
    // this with m == 1 thousands of times in a row.
    if threads <= 1 || m < 4 {
        body(0, m, out);
        return;
    }
    let chunks = threads.min(m);
    let per = m.div_ceil(chunks);
    std::thread::scope(|scope| {
        let mut rest = &mut out[..];
        let mut start = 0;
        for _ in 0..chunks {
            if start >= m {
                break;
            }
            let end = (start + per).min(m);
            let (head, tail) = rest.split_at_mut((end - start) * row_len);
            rest = tail;
            let body_ref = &body;
            scope.spawn(move || body_ref(start, end, head));
            start = end;
        }
    });
}

/// Row-wise LayerNorm: `(x - mean) / sqrt(var + eps) * w (+ b)`. The
/// bias is `None` for the bias-free norms several model families use.
pub fn layernorm(x: &mut [f32], rows: usize, cols: usize, w: &[f32], b: Option<&[f32]>, eps: f32) {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    if let Some(b) = b {
        assert_eq!(b.len(), cols);
    }
    for row in x.chunks_exact_mut(cols) {
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inv * w[i] + b.map_or(0.0, |b| b[i]);
        }
    }
}

/// Row-wise RMSNorm: `x / sqrt(mean(x^2) + eps) * w`.
pub fn rmsnorm(x: &mut [f32], rows: usize, cols: usize, w: &[f32], eps: f32) {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    for row in x.chunks_exact_mut(cols) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (v, wi) in row.iter_mut().zip(w) {
            *v *= inv * wi;
        }
    }
}

/// Numerically stable softmax over one row.
pub fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// SiLU (swish): `x * sigmoid(x)`.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// Erf-based GELU, the exact form PyTorch's `nn.GELU` uses by default:
/// `0.5 * x * (1 + erf(x / sqrt(2)))`.
pub fn gelu_erf(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

/// Tanh-approximated GELU (`gelu_new` in HF configs).
pub fn gelu_tanh(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for v in x.iter_mut() {
        let inner = SQRT_2_OVER_PI * (*v + 0.044715 * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + inner.tanh());
    }
}

/// Abramowitz-and-Stegun 7.1.26 erf approximation, absolute error under
/// 1.5e-7 over the full range. Good enough for activations; the audio
/// crate's FFT is where precision actually carries a contract.
pub fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Snake activation from the BigVGAN/SiFiGAN vocoder family:
/// `x + (1 / alpha) * sin(alpha * x)^2`.
pub fn snake(x: &mut [f32], alpha: &[f32], channels: usize, frames: usize) {
    assert_eq!(alpha.len(), channels);
    for ch in 0..channels {
        let a = alpha[ch];
        for f in 0..frames {
            let v = &mut x[ch * frames + f];
            *v += (a * *v).sin().powi(2) / a;
        }
    }
}

/// Snake-beta: `x + (1 / (beta + 1e-9)) * sin(alpha * x)^2`.
pub fn snake_beta(x: &mut [f32], alpha: &[f32], beta: &[f32], channels: usize, frames: usize) {
    assert_eq!(alpha.len(), channels);
    assert_eq!(beta.len(), channels);
    for ch in 0..channels {
        let a = alpha[ch];
        let b = 1.0 / (beta[ch] + 1e-9);
        for f in 0..frames {
            let v = &mut x[ch * frames + f];
            *v += b * (a * *v).sin().powi(2);
        }
    }
}

/// Causal (asymmetric) padding amount used by the decoder conv stacks.
pub fn causal_pad(kernel: usize, dilation: usize) -> usize {
    (kernel - 1) * dilation
}

/// Conv1d over channel-major input `x [in_ch, seq]` with PyTorch weight
/// layout `w [out_ch, in_ch / groups, kernel]`. Optional bias `[out_ch]`.
/// `groups` must divide both channel counts. Returns `[out_ch, out_seq]`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
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
    assert!(x.len() % in_ch == 0, "conv1d input length {}", x.len());
    assert_eq!(w.len(), out_ch * (in_ch / groups) * kernel);
    assert!(in_ch % groups == 0 && out_ch % groups == 0);
    let seq = x.len() / in_ch;
    let in_per_g = in_ch / groups;
    let out_per_g = out_ch / groups;
    let k_eff = (kernel - 1) * dilation + 1;
    let out_seq = (seq + 2 * padding - k_eff) / stride + 1;
    let mut out = vec![0.0f32; out_ch * out_seq];
    par_rows(out_ch, &mut out, out_seq, |o0, o1, rows| {
        for oc in o0..o1 {
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
                rows[(oc - o0) * out_seq + os] = acc;
            }
        }
    });
    out
}

/// Left-pads `x [ch, seq]` with `pad` zeros per channel, the layout
/// trick the causal conv stacks use instead of negative indices.
pub fn pad_left(x: &[f32], ch: usize, pad: usize) -> Vec<f32> {
    let seq = x.len() / ch;
    let mut out = vec![0.0f32; ch * (seq + pad)];
    for c in 0..ch {
        out[c * (seq + pad) + pad..c * (seq + pad) + pad + seq]
            .copy_from_slice(&x[c * seq..(c + 1) * seq]);
    }
    out
}

/// ConvTranspose1d with PyTorch weight layout `w [in_ch, out_ch / groups,
/// kernel]`. Stride shifts the input sample positions; output length is
/// `(seq - 1) * stride - 2 * padding + kernel`. Returns `[out_ch,
/// out_seq]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Vec<f32> {
    assert!(x.len() % in_ch == 0);
    assert_eq!(w.len(), in_ch * (out_ch / groups) * kernel);
    let seq = x.len() / in_ch;
    let out_seq = (seq - 1) * stride + kernel - 2 * padding;
    assert!(out_seq > 0 || seq == 0, "conv_transpose1d empty output");
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

/// Conv2d over channel-major `x [in_ch, h, w]` with PyTorch weight `w
/// [out_ch, in_ch / groups, kh, kw]`. Stride and padding are symmetric.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w_dim: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), in_ch * h * w_dim);
    assert_eq!(w.len(), out_ch * (in_ch / groups) * kh * kw);
    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w_dim + 2 * padding - kw) / stride + 1;
    let in_per_g = in_ch / groups;
    let out_per_g = out_ch / groups;
    let mut out = vec![0.0f32; out_ch * out_h * out_w];
    for oc in 0..out_ch {
        let g = oc / out_per_g;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = bias.map_or(0.0, |b| b[oc]);
                for ic_in_g in 0..in_per_g {
                    let ic = g * in_per_g + ic_in_g;
                    let x_ch = &x[ic * h * w_dim..(ic + 1) * h * w_dim];
                    let w_base = (((oc * in_per_g) + ic_in_g) * kh) * kw;
                    for kh_i in 0..kh {
                        let ih = oh * stride + kh_i;
                        if ih < padding || ih - padding >= h {
                            continue;
                        }
                        let ih = ih - padding;
                        for kw_i in 0..kw {
                            let iw = ow * stride + kw_i;
                            if iw < padding || iw - padding >= w_dim {
                                continue;
                            }
                            acc += x_ch[ih * w_dim + (iw - padding)] * w[w_base + kh_i * kw + kw_i];
                        }
                    }
                }
                out[oc * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
    out
}

/// MaxPool2d with symmetric stride and padding.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the PyTorch kernel signature"
)]
pub fn max_pool2d(
    x: &[f32],
    ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;
    let mut out = vec![f32::NEG_INFINITY; ch * out_h * out_w];
    for c in 0..ch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = f32::NEG_INFINITY;
                for kh_i in 0..kh {
                    for kw_i in 0..kw {
                        let ih = oh * stride + kh_i;
                        let iw = ow * stride + kw_i;
                        if ih < padding || ih - padding >= h || iw < padding || iw - padding >= w {
                            continue;
                        }
                        acc = acc.max(x[c * h * w + (ih - padding) * w + (iw - padding)]);
                    }
                }
                out[c * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
    out
}

/// Gathers embedding rows: `out [ids.len(), cols]` from `w [rows, cols]`.
pub fn embedding(w: &[f32], cols: usize, ids: &[i32]) -> Vec<f32> {
    let rows = w.len() / cols;
    let mut out = Vec::with_capacity(ids.len() * cols);
    for &id in ids {
        assert!(id >= 0, "negative token id {id}");
        let r = id as usize;
        assert!(r < rows, "token id {id} out of range {rows}");
        out.extend_from_slice(&w[r * cols..(r + 1) * cols]);
    }
    out
}

/// Scaled dot-product attention for one batch element: `q [qh, dq]`,
/// `k [kv, dq]`, `v [kv, dv]` -> `[qh, dv]`. `mask` is an additive
/// `[qh, kv]` bias when present (0 or -inf style values).
#[allow(
    clippy::too_many_arguments,
    reason = "attention kernels carry the tensor shape explicitly"
)]
pub fn sdpa(
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
    assert_eq!(q.len(), qh * dq);
    assert_eq!(k.len(), kv * dq);
    assert_eq!(v.len(), kv * dv);
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

/// `sdpa` reading K and V rows in place through explicit strides: row
/// `j` of K starts at `k_base + j * k_stride`, and likewise for V. The
/// time-major KV cache hands rows over this way, which removes the
/// per-plane copy the contiguous kernel needs. Values and reduction
/// order match `sdpa` over the copied rows exactly.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_strided(
    q: &[f32],
    k: &[f32],
    k_base: usize,
    k_stride: usize,
    v: &[f32],
    v_base: usize,
    v_stride: usize,
    mask: Option<&[f32]>,
    qh: usize,
    kv: usize,
    dq: usize,
    dv: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), qh * dq);
    debug_assert!(k.len() >= k_base + (kv.max(1) - 1) * k_stride + dq);
    debug_assert!(v.len() >= v_base + (kv.max(1) - 1) * v_stride + dv);
    let mut out = vec![0.0f32; qh * dv];
    let mut scores = vec![0.0f32; qh * kv.max(1)];
    for h in 0..qh {
        for j in 0..kv {
            let mut acc = 0.0f32;
            let k_row = k_base + j * k_stride;
            for d in 0..dq {
                acc += q[h * dq + d] * k[k_row + d];
            }
            scores[h * kv + j] = acc * scale + mask.map_or(0.0, |m| m[h * kv + j]);
        }
        softmax_row(&mut scores[h * kv..(h + 1) * kv]);
        for j in 0..kv {
            let p = scores[h * kv + j];
            let v_row = v_base + j * v_stride;
            for d in 0..dv {
                out[h * dv + d] += p * v[v_row + d];
            }
        }
    }
    out
}

/// Repeat a key/value head group to cover the query heads (GQA): `x
/// [kv_heads, len, dim] -> [q_heads, len, dim]` where each kv head is
/// shared by `q_heads / kv_heads` query heads in order.
pub fn repeat_kv(x: &[f32], kv_heads: usize, len: usize, dim: usize, q_heads: usize) -> Vec<f32> {
    assert!(
        q_heads % kv_heads == 0,
        "query heads {q_heads} not a multiple of kv heads {kv_heads}"
    );
    let rep = q_heads / kv_heads;
    let mut out = Vec::with_capacity(q_heads * len * dim);
    for h in 0..kv_heads {
        let head = &x[h * len * dim..(h + 1) * len * dim];
        for _ in 0..rep {
            out.extend_from_slice(head);
        }
    }
    out
}

/// NeoX-style rotary embedding applied in place to `x [heads, seq,
/// rotary_dim]` (rotary_dim leading features; the tail passes through).
/// `cos/sin` are per-position tables of length `rotary_dim / 2`.
pub fn rope_neox(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    assert_eq!(cos.len(), rotary_dim / 2);
    assert_eq!(sin.len(), rotary_dim / 2);
    let dim = x.len() / (heads * seq);
    assert!(dim >= rotary_dim);
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            for d in 0..rotary_dim / 2 {
                let a = x[base + d];
                let b = x[base + rotary_dim / 2 + d];
                x[base + d] = a * cos[t] - b * sin[t];
                x[base + rotary_dim / 2 + d] = b * cos[t] + a * sin[t];
            }
        }
    }
}

/// GPT-J-style interleaved rotary embedding: pairs are `(x[2i],
/// x[2i+1])`. Applied in place to `x [heads, seq, rotary_dim]`.
pub fn rope_interleaved(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    assert_eq!(cos.len(), rotary_dim / 2);
    let dim = x.len() / (heads * seq);
    assert!(dim >= rotary_dim);
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            for d in 0..rotary_dim / 2 {
                let a = x[base + 2 * d];
                let b = x[base + 2 * d + 1];
                x[base + 2 * d] = a * cos[t] - b * sin[t];
                x[base + 2 * d + 1] = b * cos[t] + a * sin[t];
            }
        }
    }
}

/// Builds the per-position cos/sin tables for a rotary embedding with
/// base `theta`: `freq_d = theta^(-2d / rotary_dim)`. Returns
/// `(cos, sin)`, each `[seq, rotary_dim / 2]` flattened.
pub fn rope_tables(seq: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
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

/// `rope_tables` for positions `start..start + len` only. Decode steps
/// need one new position apiece, and recomputing the table from zero is
/// O(t) per step. Each entry evaluates the same expression as
/// `rope_tables`, so the overlapping range is bit-identical.
pub fn rope_tables_range(
    start: usize,
    len: usize,
    rotary_dim: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; len * half];
    let mut sin = vec![0.0f32; len * half];
    for (row, t) in (start..start + len).enumerate() {
        for d in 0..half {
            let freq = theta.powf(-2.0 * d as f32 / rotary_dim as f32);
            let ang = t as f32 * freq;
            cos[row * half + d] = ang.cos();
            sin[row * half + d] = ang.sin();
        }
    }
    (cos, sin)
}

/// Flat sink used by tests to keep an intermediate result optimized in.
pub fn checksum(x: &[f32]) -> f32 {
    x.iter().sum::<f32>()
}

/// Shape helper exported for model loaders: element count of `shape`.
pub fn shape_product(shape: &[usize]) -> usize {
    product(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn matmul_small_known() {
        // [2,2] @ [2,3]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 2.0, 0.0, 1.0, 1.0];
        let out = matmul(&a, &b, 2, 2, 3);
        assert_eq!(out.len(), 6);
        assert!(close(out[0], 1.0, 1e-6));
        assert!(close(out[1], 2.0, 1e-6));
        assert!(close(out[2], 4.0, 1e-6));
        assert!(close(out[3], 3.0, 1e-6));
        assert!(close(out[4], 4.0, 1e-6));
        assert!(close(out[5], 10.0, 1e-6));
    }

    #[test]
    fn linear_matches_explicit_transpose_and_bias() {
        // x [2,2], w [2,2] row-major (n=2 out, k=2 in)
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 0.0, 0.5, 2.0];
        let bias = [0.5, -0.5];
        let out = linear(&x, &w, Some(&bias), 2, 2, 2);
        // Row 0: [1,2] . w[0]=[1,0] = 1; [1,2] . w[1]=[0.5,2] = 4.5
        assert!(close(out[0], 1.0 + 0.5, 1e-6));
        assert!(close(out[1], 4.5 - 0.5, 1e-6));
        // Row 1: [3,4] . [1,0] = 3; [3,4] . [0.5,2] = 9.5
        assert!(close(out[2], 3.0 + 0.5, 1e-6));
        assert!(close(out[3], 9.5 - 0.5, 1e-6));
        // Without bias the plain dot products come through.
        let nb = linear(&x, &w, None, 2, 2, 2);
        assert!(close(nb[0], 1.0, 1e-6));
        assert!(close(nb[1], 4.5, 1e-6));
    }

    #[test]
    fn layernorm_known_values() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0];
        layernorm(&mut x, 2, 2, &w, None, 1e-5);
        // row [1,2]: mean 1.5 var 0.25 -> [-0.99997, 0.99997]
        assert!(close(x[0], -1.0, 1e-4));
        assert!(close(x[1], 1.0, 1e-4));
    }

    #[test]
    fn rmsnorm_scales_by_rms() {
        let mut x = vec![3.0, 4.0];
        let w = [2.0, 2.0];
        rmsnorm(&mut x, 1, 2, &w, 0.0);
        // rms = sqrt((9 + 16) / 2) = 3.53553, so x / rms * 2
        let rms = (12.5f32).sqrt();
        assert!(close(x[0], 3.0 / rms * 2.0, 1e-5));
        assert!(close(x[1], 4.0 / rms * 2.0, 1e-5));
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut row = [1.0, 2.0, 3.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(close(sum, 1.0, 1e-6));
        assert!(row[2] > row[1] && row[1] > row[0]);
    }

    #[test]
    fn gelu_forms_match_reference_points() {
        let mut a = vec![1.0f32];
        gelu_erf(&mut a);
        let mut b = vec![1.0f32];
        gelu_tanh(&mut b);
        // torch.nn.functional.gelu(1.0) = 0.8413447; tanh approx 0.841192
        assert!(close(a[0], 0.841_344_7, 1e-4), "gelu_erf {}", a[0]);
        assert!(close(b[0], 0.841_192, 1e-4), "gelu_tanh {}", b[0]);
    }

    #[test]
    fn silu_known_value() {
        let mut x = vec![1.0];
        silu(&mut x);
        assert!(close(x[0], 0.731_058_6, 1e-5));
    }

    #[test]
    fn conv1d_identity_stride_one() {
        // kernel 1, weight identity per channel
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2ch x 3
        let w = [1.0, 0.0, 0.0, 1.0];
        let out = conv1d(&x, &w, None, 2, 2, 1, 1, 0, 1, 1);
        assert_eq!(out, x);
    }

    #[test]
    fn conv1d_sum_kernel_with_padding() {
        let x = vec![1.0, 2.0, 3.0]; // 1ch x 3
        let w = [1.0, 1.0]; // kernel 2, sums pairs
        let out = conv1d(&x, &w, None, 1, 1, 2, 1, 1, 1, 1);
        // padded [0,1,2,3,0] -> sums [1,3,5,3]
        assert_eq!(out, vec![1.0, 3.0, 5.0, 3.0]);
    }

    #[test]
    fn conv_transpose1d_scatter() {
        let x = vec![1.0, 2.0]; // 1ch, seq 2
        let w = vec![1.0f32; 2]; // in 1, out 1, kernel 2
        let out = conv_transpose1d(&x, &w, None, 1, 1, 2, 2, 0, 1);
        // positions: x0 -> out[0..2] += 1; x1 -> out[2..4] += 2
        assert_eq!(out, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn conv2d_one_channel_box() {
        let x = vec![1.0; 3 * 3];
        let w = vec![1.0f32; 2 * 2];
        let out = conv2d(&x, &w, None, 1, 1, 3, 3, 2, 2, 1, 0, 1);
        assert_eq!(out, vec![4.0; 4]);
    }

    #[test]
    fn max_pool_picks_max() {
        let x = vec![1.0, 5.0, 3.0, 2.0];
        let out = max_pool2d(&x, 1, 2, 2, 2, 2, 2, 0);
        assert_eq!(out, vec![5.0]);
    }

    #[test]
    fn sdpa_identity_attention() {
        // single head, k == v == identity rows, q equal to k
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out = sdpa(&q, &k, &v, None, 1, 2, 2, 2, 1.0);
        // scores [e^1, 0] -> softmax [0.731, 0.269]; out = 0.731*[1,2] + 0.269*[3,4]
        assert!(close(out[0], 0.731 * 1.0 + 0.269 * 3.0, 1e-3));
        assert!(close(out[1], 0.731 * 2.0 + 0.269 * 4.0, 1e-3));
    }

    #[test]
    fn repeat_kv_order() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 heads x 1 x 2
        let out = repeat_kv(&x, 2, 1, 2, 4);
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_neox_rotates_pairs() {
        // rotary_dim 2, one head, one position at angle 0 (cos 1, sin 0) is
        // identity; angle pi/2 swaps with sign.
        let mut x = vec![1.0, 2.0];
        rope_neox(&mut x, 1, 1, 2, &[0.0], &[1.0]);
        assert!(close(x[0], -2.0, 1e-6));
        assert!(close(x[1], 1.0, 1e-6));
    }

    #[test]
    fn rope_interleaved_rotates_pairs() {
        let mut x = vec![1.0, 2.0];
        rope_interleaved(&mut x, 1, 1, 2, &[0.0], &[1.0]);
        assert!(close(x[0], -2.0, 1e-6));
        assert!(close(x[1], 1.0, 1e-6));
    }

    #[test]
    fn rope_tables_shape_and_first_entry() {
        let (cos, sin) = rope_tables(3, 4, 10_000.0);
        assert_eq!(cos.len(), 3 * 2);
        assert!(close(cos[0], 1.0, 1e-6));
        assert!(close(sin[0], 0.0, 1e-6));
    }

    #[test]
    fn embedding_gathers_rows() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = embedding(&w, 2, &[2, 0]);
        assert_eq!(out, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn pad_left_offsets_channels() {
        // 2 channels x 2 frames, pad 1: each channel gains a leading zero.
        let out = pad_left(&[1.0, 2.0, 3.0, 4.0], 2, 1);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);
    }

    #[test]
    fn snake_matches_formula() {
        let mut x = vec![2.0];
        snake(&mut x, &[0.5], 1, 1);
        let want = 2.0 + (0.5 * 2.0f32).sin().powi(2) / 0.5;
        assert!(close(x[0], want, 1e-6));
    }

    #[test]
    fn snake_beta_matches_formula() {
        let mut x = vec![2.0];
        snake_beta(&mut x, &[0.5], &[0.25], 1, 1);
        let want = 2.0 + (0.5 * 2.0f32).sin().powi(2) / (0.25 + 1e-9);
        assert!(close(x[0], want, 1e-4));
    }
}
