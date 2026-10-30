//! CPU reference for LFM3's gated short convolution (`Lfm3ShortConv` in
//! `LiquidAI/LFM3-1.7B`'s `modeling_lfm3.py`), the ground truth for
//! `crates/gpu/src/shaders/short_conv.metal` (PORT-LOCAL, no Swift original,
//! so this file is that kernel's only contract).
//!
//! Per conv layer, for hidden width `H` and `K` taps, no bias:
//!
//! ```text
//! BCx     = x @ in_proj                 // [3H], split B | C | xx in that order
//! u       = B * xx
//! y[t,ch] = sum_{j<K} w[ch, j] * u[t-(K-1)+j, ch]   // causal depthwise, no activation
//! o       = C * y                       // then @ out_proj (not part of this op)
//! ```
//!
//! Positions before the first row read the tail, which starts at zero. The
//! tail holds the last `K - 1` rows of `u` (the PRE-conv product), oldest
//! first.
//!
//! The kernels store `u` (in the tail) and `o` as `half`; this reference
//! rounds at exactly those two points, and sums each output in the kernels'
//! order (current row's tap first as a plain product, then the `K - 1`
//! earlier taps oldest first), so a parity test can hold it tight.

use foundation::LogitValue as F16;

/// Rounds through FP16, the way a `half` store in the kernels does.
fn h(x: f32) -> f32 {
    F16::from_f32(x).to_f32()
}

/// Gated short conv over `rows` consecutive in_proj outputs, advancing
/// `tail` as the same number of single steps would.
///
/// * `in_proj`: `[rows, 3 * hidden]`, each row `B | C | xx`.
/// * `weight`: `[hidden, taps]` channel-major (`weight[ch * taps + j]`, tap
///   `taps - 1` multiplies the current row).
/// * `tail`: `[taps - 1, hidden]`, oldest row first; updated in place to the
///   last `taps - 1` rows of `[tail | u]`.
///
/// Returns `o = C * y` as `[rows, hidden]`, each value rounded through FP16.
pub fn short_conv_reference(
    in_proj: &[f32],
    weight: &[f32],
    tail: &mut [f32],
    rows: usize,
    hidden: usize,
    taps: usize,
) -> Vec<f32> {
    assert!(taps >= 1, "short conv needs at least one tap");
    assert_eq!(in_proj.len(), rows * 3 * hidden, "in_proj is [rows, 3H]");
    assert_eq!(weight.len(), hidden * taps, "weight is [H, K]");
    let history = taps - 1;
    assert_eq!(tail.len(), history * hidden, "tail is [K-1, H]");

    // Virtual sequence [tail rows | chunk rows] of u, rounded as stored.
    let mut u = Vec::with_capacity((history + rows) * hidden);
    u.extend_from_slice(tail);
    for t in 0..rows {
        let row = &in_proj[t * 3 * hidden..(t + 1) * 3 * hidden];
        for ch in 0..hidden {
            u.push(h(row[ch] * row[2 * hidden + ch]));
        }
    }

    let mut out = vec![0.0f32; rows * hidden];
    for t in 0..rows {
        let gate = &in_proj[t * 3 * hidden + hidden..t * 3 * hidden + 2 * hidden];
        // Row t of the chunk is virtual row t + history.
        let base = t + history;
        for ch in 0..hidden {
            let mut acc = u[base * hidden + ch] * weight[ch * taps + history];
            for j in 0..history {
                acc = u[(t + j) * hidden + ch].mul_add(weight[ch * taps + j], acc);
            }
            out[t * hidden + ch] = h(gate[ch] * acc);
        }
    }

    tail.copy_from_slice(&u[rows * hidden..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One channel, K = 3, hand-computed: u = B * xx = [1, 2, 3] over three
    /// steps from a zero tail, w = [0.5, 0.25, 1] (oldest first), C = 2.
    #[test]
    fn hand_computed_single_channel() {
        let rows = 3;
        // Rows are B | C | xx with B = t + 1, C = 2, xx = 1.
        let in_proj = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 3.0, 2.0, 1.0];
        let weight = [0.5, 0.25, 1.0];
        let mut tail = vec![0.0f32; 2];
        let out = short_conv_reference(&in_proj, &weight, &mut tail, rows, 1, 3);
        // y0 = 1; y1 = 2 + 0.25; y2 = 3 + 0.5 + 0.5.
        assert_eq!(out, vec![2.0, 4.5, 8.0]);
        assert_eq!(tail, vec![2.0, 3.0]);
    }

    /// Splitting a sequence into chunks must equal running it whole,
    /// including a chunk shorter than the tail.
    #[test]
    fn chunked_equals_whole() {
        let (hidden, taps, rows) = (5, 4, 9);
        let in_proj: Vec<f32> = (0..rows * 3 * hidden)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) / 8.0)
            .collect();
        let weight: Vec<f32> = (0..hidden * taps)
            .map(|i| ((i * 13 % 7) as f32 - 3.0) / 4.0)
            .collect();
        let mut whole_tail = vec![0.0f32; (taps - 1) * hidden];
        let whole = short_conv_reference(&in_proj, &weight, &mut whole_tail, rows, hidden, taps);

        let mut tail = vec![0.0f32; (taps - 1) * hidden];
        let mut chunked = Vec::new();
        let mut at = 0;
        for n in [1, 2, 6] {
            let slice = &in_proj[at * 3 * hidden..(at + n) * 3 * hidden];
            chunked.extend(short_conv_reference(
                slice, &weight, &mut tail, n, hidden, taps,
            ));
            at += n;
        }
        assert_eq!(chunked, whole);
        assert_eq!(tail, whole_tail);
    }
}
