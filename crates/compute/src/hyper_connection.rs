//! CPU reference for `qwen4_exp`'s hyper-connection MIX
//! (`docs/QWEN4_PHASE0.md` section 3). PORT-LOCAL: the Swift engine has no
//! architecture with this mechanism.
//!
//! **NOT `HyperConnectionConfig`'s Sinkhorn-normalised mHC.** That is
//! DeepSeek-V4-Flash's mechanism, a different algorithm with its own
//! `mult`/`sinkhorn_iters`/`eps` fields and no reference or kernel anywhere
//! in this port. `qwen4_exp`'s mechanism is a low-rank silu/sigmoid mix, and
//! `lowrank` is the field that distinguishes the two configs (zero for
//! DeepSeek's, whose mix carries no bottleneck).

/// `mixed[h] = mean over c in [0, C) of w[c*H+h] * normed[c*H+h]`.
///
/// The contract for `hc_mix_fp16`. `w` and `normed` are both `[C * H]`,
/// viewed as `C` streams of `H` each: `w` is the low-rank gate
/// (`sigmoid(input_mix_weight_up(silu(input_mix_weight_down(normed) / C)))`,
/// an ordinary GEMV pair with no new kernel) and `normed` is `hc_norm`'s
/// output (`rms_norm_grouped_centered`). The MEAN over the stream axis --
/// not a sum -- is what collapses the `C * H`-wide hyper-connection residual
/// back down to the `H`-wide value the sublayer underneath actually reads.
pub fn hc_mix(w: &[f32], normed: &[f32], c: usize, h: usize) -> Vec<f32> {
    assert_eq!(w.len(), c * h, "w must be C * H");
    assert_eq!(normed.len(), c * h, "normed must be C * H");
    assert!(c > 0, "C must be nonzero");
    (0..h)
        .map(|i| {
            let sum: f32 = (0..c).map(|g| w[g * h + i] * normed[g * h + i]).sum();
            sum / c as f32
        })
        .collect()
}

/// `hidden[c*H+h] = raw[c*H+h] + out[h] * inject_w[c]`, for `c` in
/// `[0, C)` and `h` in `[0, H)`.
///
/// The contract for `hc_inject_add_fp16`: the SCATTER half of a
/// hyper-connection sublayer step. `out` is the sublayer's `H`-wide output
/// (attention/GDN or the FFN), `inject_w` is a `C`-wide per-stream gate
/// (`2 * sigmoid(block_inject_weight(normed) / C)`, an ordinary GEMV with no
/// kernel of its own), and `raw` is the `C * H`-wide hyper-connection
/// residual FROM BEFORE `hc_norm` ran -- the un-normalized value, not
/// `normed`. Broadcasting one `H`-wide row against `C` per-stream scalars
/// and accumulating is what re-widens the sublayer's single output back
/// across every stream.
pub fn hc_inject_add(raw: &[f32], out: &[f32], inject_w: &[f32], c: usize, h: usize) -> Vec<f32> {
    assert_eq!(raw.len(), c * h, "raw must be C * H");
    assert_eq!(out.len(), h, "out must be H");
    assert_eq!(inject_w.len(), c, "inject_w must be C");
    let mut hidden = raw.to_vec();
    for ci in 0..c {
        for hi in 0..h {
            hidden[ci * h + hi] += out[hi] * inject_w[ci];
        }
    }
    hidden
}
