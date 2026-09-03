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
