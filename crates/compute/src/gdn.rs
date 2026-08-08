//! FP32 reference for the gated-DeltaNet (GDN) linear-attention chain that
//! Qwen 3.6's mask-2 layers run: causal depthwise conv + SiLU, per-head
//! no-weight q/k RMS norm with the delta-rule scales folded in, the gated
//! delta recurrence, and the gated output norm. Ported from the private
//! `Reference` struct in Swift's `GDNKernelTests.swift`, which is the only
//! straight-line model of this math in either tree.
//!
//! **The FP16 rounding points are part of the contract, not an artifact.**
//! The GPU kernels store `conv_out`, the normed q/k slices, the raw conv
//! tail rows, and `y` as `half`; this reference rounds at exactly those
//! four places and nowhere else (the recurrence itself and the gated norm
//! stay FP32). Move one and the parity test's tolerance stops meaning
//! anything. The recurrent state `S` is FP32 in both.
//!
//! Recurrence, per value head `h`, following mlx-vlm's `gated_delta.py`:
//!
//! ```text
//! g    = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
//! beta = sigmoid(b[h])
//! S    = S * g
//! kv   = S @ k
//! S   += outer((v - kv) * beta, k)
//! y    = S @ q
//! ```

// `half::f16` by its public path: `turbospark-core` owns the FP16 element
// type for the whole workspace (see AGENTS.md Gotcha 3 -- never hand-roll
// binary16), and this crate already depends on core, so nothing new is
// pulled in to round at the kernels' storage points.
use foundation::LogitValue as F16;

/// The RMS epsilon both no-weight GDN norms use (`kGdnRmsEps` in
/// `gdn.metal`). Distinct from the model's own `rms_eps`.
pub const GDN_RMS_EPS: f32 = 1e-6;

/// Rounds through FP16, the way a `half` store in the kernels does.
fn h(x: f32) -> f32 {
    F16::from_f32(x).to_f32()
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `log1p(exp(x))` with the large-`x` shortcut `gdn.metal` uses; matches
/// `mlx.nn.softplus` to FP32 precision.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// GDN head geometry. Mirrors `model_io::LinearAttentionConfig` without
/// this crate depending on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDims {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
}

impl GdnDims {
    /// Conv channel count, and the row width of `mixed_qkv`/`conv_out`:
    /// `[q: Hk*Dk][k: Hk*Dk][v: Hv*Dv]`.
    pub fn qkv_dim(&self) -> usize {
        2 * self.num_k_heads * self.key_head_dim + self.num_v_heads * self.value_head_dim
    }

    /// `Hv * Dv`: the z-gate width, the `y` width, and out_proj's columns.
    pub fn value_dim(&self) -> usize {
        self.num_v_heads * self.value_head_dim
    }
}

/// One GDN layer's recurrent state plus its (fixed) weights, stepped one
/// token at a time.
#[derive(Debug, Clone)]
pub struct GdnReference {
    dims: GdnDims,
    /// `[C, K]`, tap-minor (`conv_w[ch * K + j]`).
    conv_w: Vec<f32>,
    /// `[Hv]`.
    a_log: Vec<f32>,
    /// `[Hv]`.
    dt_bias: Vec<f32>,
    /// `[Dv]`, the gated output norm's learned weight.
    norm_w: Vec<f32>,
    /// The last `K - 1` RAW (pre-conv, pre-activation) rows, oldest first.
    pub tail: Vec<Vec<f32>>,
    /// `[Hv, Dv, Dk]`, FP32.
    pub state: Vec<f32>,
}

impl GdnReference {
    pub fn new(
        dims: GdnDims,
        conv_w: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm_w: &[f32],
    ) -> Self {
        assert_eq!(conv_w.len(), dims.qkv_dim() * dims.conv_kernel_size);
        assert_eq!(a_log.len(), dims.num_v_heads);
        assert_eq!(dt_bias.len(), dims.num_v_heads);
        assert_eq!(norm_w.len(), dims.value_head_dim);
        assert!(dims.num_v_heads % dims.num_k_heads == 0);
        Self {
            tail: vec![vec![0.0; dims.qkv_dim()]; dims.conv_kernel_size - 1],
            state: vec![0.0; dims.num_v_heads * dims.value_head_dim * dims.key_head_dim],
            dims,
            conv_w: conv_w.to_vec(),
            a_log: a_log.to_vec(),
            dt_bias: dt_bias.to_vec(),
            norm_w: norm_w.to_vec(),
        }
    }

    /// One decode step. `qkv_raw` is the fused input projection's `[C]`
    /// output, `a`/`b` are the `[Hv]` gate projections, `z` is the `[Hv*Dv]`
    /// output gate. Returns the gated output (out_proj's input) and
    /// advances `tail` and `state`.
    pub fn step(&mut self, qkv_raw: &[f32], a: &[f32], b: &[f32], z: &[f32]) -> Vec<f32> {
        let d = self.dims;
        let (c, k) = (d.qkv_dim(), d.conv_kernel_size);
        let (hk, hv) = (d.num_k_heads, d.num_v_heads);
        let (dk, dv) = (d.key_head_dim, d.value_head_dim);
        assert_eq!(qkv_raw.len(), c);
        assert_eq!(a.len(), hv);
        assert_eq!(b.len(), hv);
        assert_eq!(z.len(), hv * dv);

        // Causal depthwise conv + SiLU, over [tail rows..., current row].
        let mut conv = vec![0.0f32; c];
        for (ch, slot) in conv.iter_mut().enumerate() {
            let mut acc = qkv_raw[ch] * self.conv_w[ch * k + (k - 1)];
            for j in 0..k - 1 {
                acc += self.tail[j][ch] * self.conv_w[ch * k + j];
            }
            *slot = h(silu(acc));
        }
        self.tail.remove(0);
        self.tail.push(qkv_raw.iter().map(|&x| h(x)).collect());

        // Per-head no-weight RMS norm over the q and k slices, with the
        // delta-rule scales folded in: q *= 1/Dk, k *= 1/sqrt(Dk). The v
        // slice passes through as conv wrote it.
        let mut normed = conv.clone();
        for head_index in 0..2 * hk {
            let is_q = head_index < hk;
            let head = if is_q { head_index } else { head_index - hk };
            let base = if is_q { 0 } else { hk * dk } + head * dk;
            let sumsq: f32 = conv[base..base + dk].iter().map(|x| x * x).sum();
            let inv_rms = 1.0 / (sumsq / dk as f32 + GDN_RMS_EPS).sqrt();
            let scale = if is_q {
                1.0 / dk as f32
            } else {
                1.0 / (dk as f32).sqrt()
            };
            for i in 0..dk {
                normed[base + i] = h(conv[base + i] * inv_rms * scale);
            }
        }

        // Gated delta rule, per value head, FP32 state.
        let mut y = vec![0.0f32; hv * dv];
        for head in 0..hv {
            let hk_index = head / (hv / hk);
            let q_base = hk_index * dk;
            let k_base = hk * dk + hk_index * dk;
            let v_base = 2 * hk * dk + head * dv;
            let g = (-self.a_log[head].exp() * softplus(a[head] + self.dt_bias[head])).exp();
            let beta = sigmoid(b[head]);
            for row in 0..dv {
                let s = (head * dv + row) * dk;
                let mut kv = 0.0f32;
                for i in 0..dk {
                    self.state[s + i] *= g;
                    kv += self.state[s + i] * normed[k_base + i];
                }
                let delta = (normed[v_base + row] - kv) * beta;
                let mut out = 0.0f32;
                for i in 0..dk {
                    self.state[s + i] += normed[k_base + i] * delta;
                    out += self.state[s + i] * normed[q_base + i];
                }
                y[head * dv + row] = h(out);
            }
        }

        // Gated output norm: rmsnorm(y; norm_w) * silu(z), per value head.
        let mut gated = vec![0.0f32; hv * dv];
        for head in 0..hv {
            let base = head * dv;
            let sumsq: f32 = y[base..base + dv].iter().map(|x| x * x).sum();
            let inv_rms = 1.0 / (sumsq / dv as f32 + GDN_RMS_EPS).sqrt();
            for i in 0..dv {
                gated[base + i] = y[base + i] * inv_rms * self.norm_w[i] * silu(z[base + i]);
            }
        }
        gated
    }
}
