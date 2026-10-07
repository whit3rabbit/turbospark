//! Shared conv containers, checkpoint weight-layout permutes, and the
//! weight-norm fold used by the codec families.
//!
//! Every codec used to carry a private Conv1d / ConvTranspose1d wrapper
//! with the same `ops::conv1d` call and the same hand-written MLX ->
//! PyTorch weight transpose. The helpers here are pure copies of those
//! loops, so each family keeps its own geometry, tensor names, and error
//! text and only the duplicated arithmetic-free plumbing moved.
//!
//! Layout contract: weights live in the PyTorch layout the kernels take
//! (`[out, in/groups, K]` for conv, `[in, out/groups, K]` for transposed
//! conv); MLX checkpoints store `[out, K, in]` for both.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::{ops, Result, SpeechError};

/// Conv1d over channel-major `[in_ch, seq]` activations. Weight norm,
/// when the family has it, is already folded into `weight`.
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    /// PyTorch layout `[out_ch, in_ch / groups, kernel]`.
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
}

impl Conv1d {
    /// `[in_ch, seq]` -> `[out_ch, out_seq]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}

/// ConvTranspose1d over channel-major `[in_ch, seq]` activations.
#[derive(Debug, Clone)]
pub struct ConvTranspose1d {
    pub in_ch: usize,
    pub out_ch: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub groups: usize,
    /// PyTorch layout `[in_ch, out_ch / groups, kernel]`.
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
}

impl ConvTranspose1d {
    /// `[in_ch, seq]` -> `[out_ch, out_seq]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        ops::conv_transpose1d(
            x,
            &self.weight,
            self.bias.as_deref(),
            self.in_ch,
            self.out_ch,
            self.kernel,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
    }
}

/// Whether a layer's bias tensor must exist in the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BiasMode {
    /// Load it when present (the reference layer may be bias-free).
    Optional,
    /// The reference layer always has a bias; absence is an error.
    Required,
}

impl BiasMode {
    /// `Required` when the reference layer always has a bias.
    pub(crate) fn required(with_bias: bool) -> Self {
        if with_bias {
            BiasMode::Required
        } else {
            BiasMode::Optional
        }
    }
}

/// Loads the `[out_ch]` bias at `name` according to `mode`.
pub(crate) fn load_bias(
    file: &SafetensorsFile,
    name: &str,
    out_ch: usize,
    mode: BiasMode,
) -> Result<Option<Vec<f32>>> {
    match mode {
        BiasMode::Optional | BiasMode::Required => {
            if file.contains_tensor(name) {
                Ok(Some(load_f32_shaped(file, name, &[out_ch])?))
            } else if mode == BiasMode::Required {
                Err(SpeechError::Tensor {
                    name: name.to_string(),
                    why: "required by the reference layer".to_string(),
                })
            } else {
                Ok(None)
            }
        }
    }
}

/// Copies `stored [a, K, b]` into `[a, b, K]`. This is the MLX conv
/// layout `[out, K, in/groups]` -> PyTorch `[out, in/groups, K]`, and
/// also MLX `[in, K, out]` -> PyTorch `[in, out, K]` for the transposed
/// convs whose checkpoints keep the input axis first.
pub(crate) fn permute_okc_to_ock(stored: &[f32], a: usize, kernel: usize, b: usize) -> Vec<f32> {
    let mut weight = vec![0.0f32; stored.len()];
    for o in 0..a {
        for k in 0..kernel {
            for i in 0..b {
                weight[o * b * kernel + i * kernel + k] = stored[o * kernel * b + k * b + i];
            }
        }
    }
    weight
}

/// Copies `stored [out, K, in]` into `[in, out, K]`: the MLX
/// out-first transposed-conv layout -> PyTorch `[in, out, K]`.
pub(crate) fn permute_okc_to_cok(
    stored: &[f32],
    out_ch: usize,
    kernel: usize,
    in_ch: usize,
) -> Vec<f32> {
    let mut weight = vec![0.0f32; stored.len()];
    for i in 0..in_ch {
        for k in 0..kernel {
            for o in 0..out_ch {
                weight[i * out_ch * kernel + o * kernel + k] =
                    stored[o * kernel * in_ch + k * in_ch + i];
            }
        }
    }
    weight
}

/// Loads `name` as an MLX conv weight `[out, K, in_g]` and returns it in
/// the PyTorch `[out, in_g, K]` layout.
pub(crate) fn load_mlx_conv_weight(
    file: &SafetensorsFile,
    name: &str,
    out_ch: usize,
    kernel: usize,
    in_g: usize,
) -> Result<Vec<f32>> {
    let stored = load_f32_shaped(file, name, &[out_ch, kernel, in_g])?;
    Ok(permute_okc_to_ock(&stored, out_ch, kernel, in_g))
}

/// Loads `name` as an MLX transposed-conv weight `[out, K, in]` and
/// returns it in the PyTorch `[in, out, K]` layout.
pub(crate) fn load_mlx_convt_weight(
    file: &SafetensorsFile,
    name: &str,
    out_ch: usize,
    kernel: usize,
    in_ch: usize,
) -> Result<Vec<f32>> {
    let stored = load_f32_shaped(file, name, &[out_ch, kernel, in_ch])?;
    Ok(permute_okc_to_cok(&stored, out_ch, kernel, in_ch))
}

/// How a weight-norm fold turns `(g, v, ||v||)` into the weight. The
/// copies differ in guard arithmetic, which is load-bearing for bit
/// parity with each family's reference, so the caller picks one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum WnFold {
    /// `g * v / ||v||` with no guard (the DAC-family reference).
    Div,
    /// `g * v / max(||v||, eps)` (MiniMax conversion, eps 1e-12).
    DivMax(f32),
    /// `v * (g / (||v|| + eps))` (Kokoro's per-call normalize, eps 1e-7).
    ScaleAdd(f32),
}

/// Folds weight norm in place. `v` is split into `g.len()` contiguous
/// groups of `per` elements; each group's squared sum is accumulated in
/// element order (the same f32 order every copy used) and the group is
/// rewritten per `mode`. Callers whose stored order is not group
/// contiguous permute first (the fold is elementwise after the norm, so
/// permute-then-fold and fold-then-permute give the same bits).
pub(crate) fn wn_fold_in_place(v: &mut [f32], g: &[f32], per: usize, mode: WnFold) {
    for (row, &gv) in g.iter().enumerate() {
        let group = &mut v[row * per..(row + 1) * per];
        let mut acc = 0.0f32;
        for &x in group.iter() {
            acc += x * x;
        }
        let norm = acc.sqrt();
        match mode {
            WnFold::Div => {
                for x in group.iter_mut() {
                    *x = gv * *x / norm;
                }
            }
            WnFold::DivMax(eps) => {
                let norm = norm.max(eps);
                for x in group.iter_mut() {
                    *x = gv * *x / norm;
                }
            }
            WnFold::ScaleAdd(eps) => {
                let scale = gv / (norm + eps);
                for x in group.iter_mut() {
                    *x *= scale;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values with sign changes, a few exact zeros, and one
    /// all-zero row so the 0/0 and guard paths are exercised.
    fn sample(len: usize, zero_row: Option<(usize, usize)>) -> Vec<f32> {
        let mut state = 0x9E37_79B9u32;
        let mut v: Vec<f32> = (0..len)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if i % 17 == 5 {
                    0.0
                } else {
                    ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 3.7
                }
            })
            .collect();
        if let Some((start, per)) = zero_row {
            for x in &mut v[start..start + per] {
                *x = 0.0;
            }
        }
        v
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    // Verbatim copy of the old `WnConv1d::load` fold (stored [out, K, in]).
    fn old_wnconv1d(v: &[f32], g: &[f32], out_ch: usize, kernel: usize, in_g: usize) -> Vec<f32> {
        let mut weight = vec![0.0f32; out_ch * in_g * kernel];
        for (oc, &gv) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            let base = oc * kernel * in_g;
            for kk in 0..kernel {
                for ii in 0..in_g {
                    acc += v[base + kk * in_g + ii] * v[base + kk * in_g + ii];
                }
            }
            let norm = acc.sqrt();
            for kk in 0..kernel {
                for ii in 0..in_g {
                    weight[oc * in_g * kernel + ii * kernel + kk] =
                        gv * v[base + kk * in_g + ii] / norm;
                }
            }
        }
        weight
    }

    // Verbatim copy of the old `WnConvTranspose1d::load_out_first` and
    // `DacvaeConvT::load` fold (stored [out, K, in], norm per in channel).
    fn old_out_first(v: &[f32], g: &[f32], out_ch: usize, kernel: usize, in_g: usize) -> Vec<f32> {
        let mut weight = vec![0.0f32; in_g * out_ch * kernel];
        for (ic, &gv) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            for oo in 0..out_ch {
                for kk in 0..kernel {
                    acc += v[oo * kernel * in_g + kk * in_g + ic]
                        * v[oo * kernel * in_g + kk * in_g + ic];
                }
            }
            let norm = acc.sqrt();
            for oo in 0..out_ch {
                for kk in 0..kernel {
                    weight[ic * out_ch * kernel + oo * kernel + kk] =
                        gv * v[oo * kernel * in_g + kk * in_g + ic] / norm;
                }
            }
        }
        weight
    }

    // Verbatim copy of Kokoro's old `weight_norm_oki` (kept as written so
    // the parity check compares against the exact old code).
    #[allow(clippy::needless_range_loop)]
    fn old_kokoro(v: &[f32], g: &[f32], out: usize, in_ch: usize, kernel: usize) -> Vec<f32> {
        let mut weight = Vec::with_capacity(v.len());
        let per = in_ch * kernel;
        for o in 0..out {
            let norm = v[o * per..(o + 1) * per]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let scale = g[o] / (norm + 1e-7);
            for i in o * per..(o + 1) * per {
                weight.push(v[i] * scale);
            }
        }
        weight
    }

    // Verbatim copy of the MiniMax `fuse_weight_norm_pairs` row fold.
    fn old_minimax(v: &[f32], g: &[f32], rows: usize) -> Vec<f32> {
        let per = v.len() / rows;
        let mut fused = Vec::with_capacity(v.len());
        for o in 0..rows {
            let norm = v[o * per..(o + 1) * per]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            let norm = norm.max(1e-12);
            for value in &v[o * per..(o + 1) * per] {
                fused.push(g[o] * value / norm);
            }
        }
        fused
    }

    // Verbatim copy of the old Fish torch-layout fold (norm over (in, K)).
    fn old_fish_conv(v: &[f32], g: &[f32], in_g: usize, kernel: usize) -> Vec<f32> {
        let mut weight = vec![0.0f32; v.len()];
        for (oc, &gain) in g.iter().enumerate() {
            let mut acc = 0.0f32;
            for i in 0..in_g {
                for k in 0..kernel {
                    let value = v[oc * in_g * kernel + i * kernel + k];
                    acc += value * value;
                }
            }
            let norm = acc.sqrt();
            for i in 0..in_g {
                for k in 0..kernel {
                    weight[oc * in_g * kernel + i * kernel + k] =
                        gain * v[oc * in_g * kernel + i * kernel + k] / norm;
                }
            }
        }
        weight
    }

    #[test]
    fn fold_div_then_permute_matches_old_wnconv1d() {
        for &(out_ch, kernel, in_g) in &[(5usize, 7usize, 3usize), (4, 1, 6), (3, 2, 1)] {
            let per = kernel * in_g;
            let v = sample(out_ch * per, Some((per, per)));
            let g = sample(out_ch, None);
            let want = old_wnconv1d(&v, &g, out_ch, kernel, in_g);
            let mut folded = v.clone();
            wn_fold_in_place(&mut folded, &g, per, WnFold::Div);
            let got = permute_okc_to_ock(&folded, out_ch, kernel, in_g);
            assert_eq!(bits(&got), bits(&want), "{out_ch}x{kernel}x{in_g}");
        }
    }

    #[test]
    fn permute_then_fold_matches_old_out_first_convt() {
        for &(out_ch, kernel, in_g) in &[(5usize, 4usize, 3usize), (2, 8, 6)] {
            let per = out_ch * kernel;
            let v = sample(out_ch * kernel * in_g, None);
            let g = sample(in_g, None);
            let want = old_out_first(&v, &g, out_ch, kernel, in_g);
            let mut got = permute_okc_to_cok(&v, out_ch, kernel, in_g);
            wn_fold_in_place(&mut got, &g, per, WnFold::Div);
            assert_eq!(bits(&got), bits(&want), "{out_ch}x{kernel}x{in_g}");
        }
    }

    #[test]
    fn fold_div_matches_old_fish_torch_layout() {
        let (out_ch, in_g, kernel) = (6usize, 5usize, 3usize);
        let v = sample(out_ch * in_g * kernel, None);
        let g = sample(out_ch, None);
        let want = old_fish_conv(&v, &g, in_g, kernel);
        let mut got = v.clone();
        wn_fold_in_place(&mut got, &g, in_g * kernel, WnFold::Div);
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn fold_scale_add_matches_old_kokoro() {
        let (out, in_ch, kernel) = (4usize, 6usize, 3usize);
        let v = sample(out * in_ch * kernel, Some((in_ch * kernel, in_ch * kernel)));
        let g = sample(out, None);
        let want = old_kokoro(&v, &g, out, in_ch, kernel);
        let mut got = v.clone();
        wn_fold_in_place(&mut got, &g, in_ch * kernel, WnFold::ScaleAdd(1e-7));
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn fold_div_max_matches_old_minimax() {
        let (rows, per) = (5usize, 9usize);
        let v = sample(rows * per, Some((per * 2, per)));
        let g = sample(rows, None);
        let want = old_minimax(&v, &g, rows);
        let mut got = v.clone();
        wn_fold_in_place(&mut got, &g, per, WnFold::DivMax(1e-12));
        assert_eq!(bits(&got), bits(&want));
    }

    // Verbatim copies of the hand-written MLX -> PyTorch loops.
    #[test]
    fn permutes_match_old_loops() {
        let (out_ch, kernel, in_ch) = (3usize, 5usize, 4usize);
        let stored = sample(out_ch * kernel * in_ch, None);

        let mut conv = vec![0.0f32; stored.len()];
        for oc in 0..out_ch {
            for kk in 0..kernel {
                for ic in 0..in_ch {
                    conv[oc * in_ch * kernel + ic * kernel + kk] =
                        stored[oc * kernel * in_ch + kk * in_ch + ic];
                }
            }
        }
        assert_eq!(
            bits(&permute_okc_to_ock(&stored, out_ch, kernel, in_ch)),
            bits(&conv)
        );

        let mut convt = vec![0.0f32; stored.len()];
        for ic in 0..in_ch {
            for kk in 0..kernel {
                for oc in 0..out_ch {
                    convt[ic * out_ch * kernel + oc * kernel + kk] =
                        stored[oc * kernel * in_ch + kk * in_ch + ic];
                }
            }
        }
        assert_eq!(
            bits(&permute_okc_to_cok(&stored, out_ch, kernel, in_ch)),
            bits(&convt)
        );
    }

    #[test]
    fn conv_forward_matches_kernel_call() {
        let conv = Conv1d {
            in_ch: 2,
            out_ch: 3,
            kernel: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            weight: sample(3 * 2 * 3, None),
            bias: Some(sample(3, None)),
        };
        let x = sample(2 * 8, None);
        let want = ops::conv1d(&x, &conv.weight, conv.bias.as_deref(), 2, 3, 3, 1, 1, 1, 1);
        assert_eq!(bits(&conv.forward(&x)), bits(&want));

        let convt = ConvTranspose1d {
            in_ch: 2,
            out_ch: 3,
            kernel: 4,
            stride: 2,
            padding: 1,
            output_padding: 0,
            groups: 1,
            weight: sample(2 * 3 * 4, None),
            bias: None,
        };
        let want = ops::conv_transpose1d(&x, &convt.weight, None, 2, 3, 4, 2, 1, 0, 1);
        assert_eq!(bits(&convt.forward(&x)), bits(&want));
    }
}
