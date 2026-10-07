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
    par_rows(m, &mut out, n, k * n, |row_start, row_end, rows| {
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
///
/// Each output is one sequential `k`-ascending sum starting at `0.0`
/// with the bias added last, exactly as the plain loop computes it. The
/// speedup comes only from computing several independent outputs at once
/// and from splitting outputs across threads; a single dot product is
/// never split, so the bits do not change.
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
    if m >= 4 {
        par_rows(m, &mut out, n, k * n, |row_start, row_end, rows| {
            for i in row_start..row_end {
                let x_row = &x[i * k..(i + 1) * k];
                let out_row = &mut rows[(i - row_start) * n..(i - row_start + 1) * n];
                linear_cols(x_row, w, bias, k, 0, out_row);
            }
        });
    } else {
        // Decode shapes (m == 1) have no rows to split, and the vocabulary
        // projection is the largest matvec in the crate, so split the
        // output columns instead.
        for i in 0..m {
            let x_row = &x[i * k..(i + 1) * k];
            let out_row = &mut out[i * n..(i + 1) * n];
            par_rows(n, out_row, 1, k, |c0, _c1, cols| {
                linear_cols(x_row, w, bias, k, c0, cols);
            });
        }
    }
    out
}

/// Output columns `col0..col0 + out.len()` of one `linear` row. Four
/// columns share each pass over `x_row` so four independent accumulators
/// overlap their add latency; every accumulator still walks `k`
/// ascending.
fn linear_cols(
    x_row: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    k: usize,
    col0: usize,
    out: &mut [f32],
) {
    let mut chunks = out.chunks_exact_mut(4);
    let mut ni = col0;
    for quad in &mut chunks {
        let w0 = &w[ni * k..(ni + 1) * k];
        let w1 = &w[(ni + 1) * k..(ni + 2) * k];
        let w2 = &w[(ni + 2) * k..(ni + 3) * k];
        let w3 = &w[(ni + 3) * k..(ni + 4) * k];
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for ki in 0..k {
            let xv = x_row[ki];
            a0 += xv * w0[ki];
            a1 += xv * w1[ki];
            a2 += xv * w2[ki];
            a3 += xv * w3[ki];
        }
        quad[0] = a0 + bias.map_or(0.0, |b| b[ni]);
        quad[1] = a1 + bias.map_or(0.0, |b| b[ni + 1]);
        quad[2] = a2 + bias.map_or(0.0, |b| b[ni + 2]);
        quad[3] = a3 + bias.map_or(0.0, |b| b[ni + 3]);
        ni += 4;
    }
    for out_val in chunks.into_remainder() {
        let w_row = &w[ni * k..(ni + 1) * k];
        let mut acc = 0.0f32;
        for (&x_val, &w_val) in x_row.iter().zip(w_row) {
            acc += x_val * w_val;
        }
        *out_val = acc + bias.map_or(0.0, |b| b[ni]);
        ni += 1;
    }
}

/// Smallest total multiply-add count worth waking threads for. A scoped
/// thread costs tens of microseconds to start and join, so smaller
/// problems stay on the caller; decode calls this with tiny `m` thousands
/// of times in a row.
const PAR_MIN_WORK: usize = 1 << 18;

/// Worker threads available to the kernels, read once. The OS query is
/// not free on every platform and the answer does not change mid-run.
pub(crate) fn thread_count() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// Runs `body(start, end, rows)` over contiguous row ranges of an
/// `m x row_len` output, splitting across threads when `m * work_per_row`
/// multiply-adds justify it. Rows are independent, so how they are
/// grouped never changes any value.
fn par_rows(
    m: usize,
    out: &mut [f32],
    row_len: usize,
    work_per_row: usize,
    body: impl Fn(usize, usize, &mut [f32]) + Sync,
) {
    let threads = thread_count();
    if threads <= 1 || m < 2 || m.saturating_mul(work_per_row) < PAR_MIN_WORK {
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

/// Runs `job(0..count)` on scoped worker threads and returns the results in
/// index order. For independent, single-threaded jobs (each result depends
/// only on its index), so the output equals `(0..count).map(job)` exactly;
/// jobs are handed out dynamically but placed by index.
pub(crate) fn par_map<T: Send>(count: usize, job: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = thread_count().min(count);
    if threads <= 1 {
        return (0..count).map(job).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        mine.push((index, job(index)));
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });
    let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
    for (index, value) in parts.into_iter().flatten() {
        slots[index] = Some(value);
    }
    slots
        .into_iter()
        .map(|slot| slot.expect("every job index ran once"))
        .collect()
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
    par_rows(
        out_ch,
        &mut out,
        out_seq,
        in_per_g * kernel * out_seq,
        |o0, o1, rows| {
            for oc in o0..o1 {
                let g = oc / out_per_g;
                let out_row = &mut rows[(oc - o0) * out_seq..(oc - o0 + 1) * out_seq];
                // Each output starts at its bias and then takes taps in
                // (input channel, kernel offset) ascending order, skipping
                // taps that fall in the padding: the same sequence the
                // per-output loop walks. Making the output index the inner
                // loop turns each tap into one branch-free pass over a
                // contiguous run.
                out_row.fill(bias.map_or(0.0, |b| b[oc]));
                for ic_in_g in 0..in_per_g {
                    let ic = g * in_per_g + ic_in_g;
                    let w_base = ((oc * in_per_g) + ic_in_g) * kernel;
                    let x_ch = &x[ic * seq..(ic + 1) * seq];
                    for kk in 0..kernel {
                        let w_val = w[w_base + kk];
                        // Input position of output `os` is `os * stride + off`.
                        let off = (kk * dilation) as isize - padding as isize;
                        // Smallest `os` with a non-negative position, and one
                        // past the largest `os` still inside the signal.
                        let lo = if off >= 0 {
                            0
                        } else {
                            ((-off) as usize).div_ceil(stride)
                        };
                        let hi = if off > seq as isize - 1 {
                            0
                        } else {
                            (((seq as isize - 1 - off) as usize) / stride + 1).min(out_seq)
                        };
                        if lo >= hi {
                            continue;
                        }
                        let pos0 = (lo as isize * stride as isize + off) as usize;
                        if stride == 1 {
                            let src = &x_ch[pos0..pos0 + (hi - lo)];
                            for (o, &xv) in out_row[lo..hi].iter_mut().zip(src) {
                                *o += xv * w_val;
                            }
                        } else {
                            for (n, o) in out_row[lo..hi].iter_mut().enumerate() {
                                *o += x_ch[pos0 + n * stride] * w_val;
                            }
                        }
                    }
                }
            }
        },
    );
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
/// `(seq - 1) * stride - 2 * padding + kernel + output_padding`.
/// Returns `[out_ch, out_seq]`.
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
    output_padding: usize,
    groups: usize,
) -> Vec<f32> {
    assert!(x.len() % in_ch == 0);
    assert_eq!(w.len(), in_ch * (out_ch / groups) * kernel);
    let seq = x.len() / in_ch;
    let out_seq = (seq - 1) * stride + kernel - 2 * padding + output_padding;
    assert!(out_seq > 0 || seq == 0, "conv_transpose1d empty output");
    let out_per_g = out_ch / groups;
    let in_per_g = in_ch / groups;
    let mut out = vec![0.0f32; out_ch * out_seq];
    // One output channel per row, so rows are independent and can be
    // threaded. Within a row the input channels, then input positions,
    // are visited in ascending order, which is the order each output
    // element accumulated in when the loops scattered channel by channel;
    // for a fixed element every (channel, position) pair contributes at
    // most one kernel tap, so the per-element sum is unchanged.
    par_rows(
        out_ch,
        &mut out,
        out_seq,
        in_per_g * seq * kernel,
        |o0, o1, rows| {
            for oc in o0..o1 {
                let g = oc / out_per_g;
                let oc_in_g = oc % out_per_g;
                let out_row = &mut rows[(oc - o0) * out_seq..(oc - o0 + 1) * out_seq];
                for ic in g * in_per_g..(g + 1) * in_per_g {
                    let w_base = ((ic * out_per_g) + oc_in_g) * kernel;
                    let w_taps = &w[w_base..w_base + kernel];
                    let x_ch = &x[ic * seq..(ic + 1) * seq];
                    for (is, &xv) in x_ch.iter().enumerate() {
                        if xv == 0.0 {
                            continue;
                        }
                        // Taps land at `is * stride + kk - padding`; keep the
                        // ones inside `0..out_seq`.
                        let base = is * stride;
                        if base >= padding + out_seq {
                            continue;
                        }
                        let lo = padding.saturating_sub(base);
                        let hi = kernel.min(padding + out_seq - base);
                        if lo >= hi {
                            continue;
                        }
                        let dst = base + lo - padding;
                        for (o, &wv) in out_row[dst..dst + (hi - lo)]
                            .iter_mut()
                            .zip(&w_taps[lo..hi])
                        {
                            *o += xv * wv;
                        }
                    }
                }
                if let Some(b) = bias {
                    for v in out_row.iter_mut() {
                        *v += b[oc];
                    }
                }
            }
        },
    );
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
    sdpa_strided(q, k, 0, dq, v, 0, dv, mask, qh, kv, dq, dv, scale)
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
    let mut out = vec![0.0f32; qh * dv];
    sdpa_strided_into(
        &mut out, q, k, k_base, k_stride, v, v_base, v_stride, mask, qh, kv, dq, dv, scale,
    );
    out
}

/// [`sdpa_strided`] writing into a caller-owned `out [qh, dv]`, which is
/// zeroed first. Lets multi-head callers fill one buffer instead of
/// allocating and copying a result per head.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_strided_into(
    out: &mut [f32],

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
) {
    assert_eq!(out.len(), qh * dv);
    out.fill(0.0);
    assert_eq!(q.len(), qh * dq);
    debug_assert!(k.len() >= k_base + (kv.max(1) - 1) * k_stride + dq);
    debug_assert!(v.len() >= v_base + (kv.max(1) - 1) * v_stride + dv);
    // Heads are independent, so they split across threads; each worker
    // reuses one scores row for all of its heads.
    par_rows(qh, out, dv, kv * (dq + dv), |h0, h1, rows| {
        let mut scores = vec![0.0f32; kv];
        for h in h0..h1 {
            let q_row = &q[h * dq..(h + 1) * dq];
            for (j, score) in scores.iter_mut().enumerate() {
                let k_row = &k[k_base + j * k_stride..k_base + j * k_stride + dq];
                let mut acc = 0.0f32;
                for (&qv, &kv_val) in q_row.iter().zip(k_row) {
                    acc += qv * kv_val;
                }
                *score = acc * scale + mask.map_or(0.0, |m| m[h * kv + j]);
            }
            softmax_row(&mut scores);
            let out_row = &mut rows[(h - h0) * dv..(h - h0 + 1) * dv];
            for (j, &p) in scores.iter().enumerate() {
                let v_row = &v[v_base + j * v_stride..v_base + j * v_stride + dv];
                for (o, &vv) in out_row.iter_mut().zip(v_row) {
                    *o += p * vv;
                }
            }
        }
    });
}

/// Splits a row-major `[seq, heads * head_dim]` activation into head-major
/// planes `[heads, seq, head_dim]`.
pub fn split_heads(x: &[f32], seq: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    split_heads_strided(x, heads * head_dim, 0, seq, heads, head_dim)
}

/// [`split_heads`] reading rows of `row_stride` floats and starting each
/// row at `col_offset`, for the q/k/v thirds of a fused `[seq, 3 * dim]`
/// projection (`col_offset` is `0`, `dim`, `2 * dim`).
pub fn split_heads_strided(
    x: &[f32],
    row_stride: usize,
    col_offset: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert!(seq == 0 || x.len() >= (seq - 1) * row_stride + col_offset + heads * head_dim);
    let mut out = vec![0.0f32; heads * seq * head_dim];
    for h in 0..heads {
        for t in 0..seq {
            let src = t * row_stride + col_offset + h * head_dim;
            let dst = (h * seq + t) * head_dim;
            out[dst..dst + head_dim].copy_from_slice(&x[src..src + head_dim]);
        }
    }
    out
}

/// Inverse of [`split_heads`]: head-major `[heads, seq, head_dim]` back to
/// row-major `[seq, heads * head_dim]`.
pub fn merge_heads(x: &[f32], seq: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), heads * seq * head_dim);
    let dim = heads * head_dim;
    let mut out = vec![0.0f32; seq * dim];
    for h in 0..heads {
        for t in 0..seq {
            let src = (h * seq + t) * head_dim;
            out[t * dim + h * head_dim..t * dim + (h + 1) * head_dim]
                .copy_from_slice(&x[src..src + head_dim]);
        }
    }
    out
}

/// Multi-head attention over head-major planes: `q [heads, seq_q,
/// head_dim]`, `k`/`v [heads, seq_kv, head_dim]` to `[heads, seq_q,
/// head_dim]`. `mask` is one additive `[seq_q, seq_kv]` bias shared by
/// every head. Each head runs [`sdpa`] unchanged and writes straight into
/// the result, so the values match a per-head `sdpa` loop bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn mha(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), heads * seq_q * head_dim);
    assert_eq!(k.len(), heads * seq_kv * head_dim);
    assert_eq!(v.len(), heads * seq_kv * head_dim);
    let mut out = vec![0.0f32; heads * seq_q * head_dim];
    for h in 0..heads {
        let qp = h * seq_q * head_dim;
        let kp = h * seq_kv * head_dim;
        sdpa_strided_into(
            &mut out[qp..qp + seq_q * head_dim],
            &q[qp..qp + seq_q * head_dim],
            k,
            kp,
            head_dim,
            v,
            kp,
            head_dim,
            mask,
            seq_q,
            seq_kv,
            head_dim,
            head_dim,
            scale,
        );
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

/// NeoX-style (half-split) rotary embedding applied in place to
/// `x [heads, seq, dim]`: the leading `rotary_dim` features of every row
/// rotate as pairs `(x[d], x[d + rotary_dim / 2])` and the tail passes
/// through. `cos`/`sin` are per-position tables `[seq, rotary_dim / 2]`
/// (see [`rope_tables`]); `dim` is inferred from the buffer length. A
/// single decode row is `heads == 1, seq == 1` with one table row.
pub fn rope_neox(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = rotary_dim / 2;
    assert!(cos.len() >= seq * half && sin.len() >= seq * half);
    if heads * seq == 0 {
        return;
    }
    let dim = x.len() / (heads * seq);
    assert!(dim >= rotary_dim);
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            let c = &cos[t * half..(t + 1) * half];
            let s = &sin[t * half..(t + 1) * half];
            let (lo, hi) = x[base..base + rotary_dim].split_at_mut(half);
            for d in 0..half {
                let a = lo[d];
                let b = hi[d];
                lo[d] = a * c[d] - b * s[d];
                hi[d] = b * c[d] + a * s[d];
            }
        }
    }
}

/// GPT-J-style interleaved rotary embedding: pairs are `(x[2i],
/// x[2i + 1])`. Applied in place to `x [heads, seq, dim]` with the same
/// `[seq, rotary_dim / 2]` tables and tail pass-through as [`rope_neox`].
pub fn rope_interleaved(
    x: &mut [f32],
    heads: usize,
    seq: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = rotary_dim / 2;
    assert!(cos.len() >= seq * half && sin.len() >= seq * half);
    if heads * seq == 0 {
        return;
    }
    let dim = x.len() / (heads * seq);
    assert!(dim >= rotary_dim);
    for h in 0..heads {
        for t in 0..seq {
            let base = (h * seq + t) * dim;
            let c = &cos[t * half..(t + 1) * half];
            let s = &sin[t * half..(t + 1) * half];
            for (d, pair) in x[base..base + rotary_dim].chunks_exact_mut(2).enumerate() {
                let a = pair[0];
                let b = pair[1];
                pair[0] = a * c[d] - b * s[d];
                pair[1] = b * c[d] + a * s[d];
            }
        }
    }
}

/// `theta^(-2d / rotary_dim)` for `d in 0..half`. It depends only on the
/// feature index, so the table builders evaluate it once per index rather
/// than once per position; the expression itself is unchanged.
fn rope_freqs(half: usize, rotary_dim: usize, theta: f32) -> Vec<f32> {
    (0..half)
        .map(|d| theta.powf(-2.0 * d as f32 / rotary_dim as f32))
        .collect()
}

/// Builds the per-position cos/sin tables for a rotary embedding with
/// base `theta`: `freq_d = theta^(-2d / rotary_dim)`. Returns
/// `(cos, sin)`, each `[seq, rotary_dim / 2]` flattened.
pub fn rope_tables(seq: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    let freqs = rope_freqs(half, rotary_dim, theta);
    for t in 0..seq {
        for (d, &freq) in freqs.iter().enumerate() {
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
    let freqs = rope_freqs(half, rotary_dim, theta);
    for (row, t) in (start..start + len).enumerate() {
        for (d, &freq) in freqs.iter().enumerate() {
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

/// Logistic sigmoid `1 / (1 + exp(-v))`. One shared copy of the
/// expression the VAD, TTS and speech-enhancement models each wrote
/// inline; the operation order is unchanged.
#[inline]
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// ELU with alpha 1 (`nn.ELU`), in place.
pub fn elu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v <= 0.0 {
            *v = v.exp() - 1.0;
        }
    }
}

/// LayerNorm over the channel axis of a channel-major `x [ch, frames]`,
/// one frame (column) at a time. `w`/`b` are optional so the same
/// kernel serves the affine and affine-free norms.
///
/// Each frame's channel sums run channel-ascending from `0.0`, exactly
/// the strided loops the codec families wrote by hand. With a weight and
/// no bias, `0.0` is still added after the scale (as the hand loops did,
/// and as [`layernorm`] does); with neither, the plain `(x - mean) * inv`
/// is stored.
pub fn layernorm_cm(
    x: &mut [f32],
    ch: usize,
    frames: usize,
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
) {
    assert_eq!(x.len(), ch * frames);
    if let Some(w) = w {
        assert_eq!(w.len(), ch);
    }
    if let Some(b) = b {
        assert_eq!(b.len(), ch);
    }
    for t in 0..frames {
        let mut mean = 0.0f32;
        for c in 0..ch {
            mean += x[c * frames + t];
        }
        mean /= ch as f32;
        let mut var = 0.0f32;
        for c in 0..ch {
            let d = x[c * frames + t] - mean;
            var += d * d;
        }
        var /= ch as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..ch {
            let i = c * frames + t;
            let n = (x[i] - mean) * inv;
            x[i] = match (w, b) {
                (Some(w), b) => n * w[c] + b.map_or(0.0, |b| b[c]),
                (None, Some(b)) => n + b[c],
                (None, None) => n,
            };
        }
    }
}

#[cfg(test)]
mod kernel_parity;

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
        let out = conv_transpose1d(&x, &w, None, 1, 1, 2, 2, 0, 0, 1);
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

    #[test]
    fn par_map_returns_results_in_index_order() {
        for count in [0usize, 1, 2, 7, 64] {
            let got = par_map(count, |index| index * index + 1);
            let want: Vec<usize> = (0..count).map(|index| index * index + 1).collect();
            assert_eq!(got, want);
        }
    }
}
