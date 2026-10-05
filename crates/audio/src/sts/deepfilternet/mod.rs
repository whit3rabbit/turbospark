//! DeepFilterNet 1/2/3: speech enhancement (noise suppression) at 48 kHz.
//!
//! Reference: `mlx_audio/sts/models/deepfilternet/` (model.py, network.py,
//! weight_loader.py) at the cloned v0.5.7 revision, which itself pins the
//! DeepFilterNet architecture by Schroeter et al.
//!
//! Pipeline: 48 kHz audio -> pad [hop, fft] -> STFT (960/480, Vorbis
//! window, no center) -> ERB + deep-filter features with EMA norms ->
//! conv/GRU encoder -> ERB mask decoder + deep-filter coefficient decoder
//! -> deep filtering of the low bins -> inverse STFT -> delay
//! compensation. The v3 path (enc_concat = false) deep-filters the raw
//! spectrum and keeps the masked spectrum only above `nb_df`; the v2
//! layout deep-filters the masked spectrum instead.
//!
//! Checkpoints (mlx-community/DeepFilterNet-mlx) store conv weights in
//! the PyTorch `[out, in/g, kH, kW]` layout, but the block tensor
//! offsets differ per stage in the shipped file (erb_conv0's main conv
//! sits at `.1` with BN at `.2`; erb_conv1/2/3, df_conv1, df_convp and
//! the decoder blocks use `.0`/`.1`/`.2` or `.1`/`.2`/`.3`), so the
//! loader pins exact tensor names. GRU biases fold PyTorch-style:
//! `b = bias_ih + bias_hh[:2H]`, `bhn = bias_hh[2H:]`; the recurrence
//! always starts from a zero hidden state so `Wh` never contributes.

use std::path::Path;

use turbospark_audio::stft::{istft, stft, StftOptions};
use turbospark_model_io::safetensors::SafetensorsFile;

use crate::ops;
use crate::{Result, SpeechError};

// ---------------------------------------------------------------------------
// Tensor / layer primitives
// ---------------------------------------------------------------------------

struct F32Tensor {
    data: Vec<f32>,
    #[allow(dead_code, reason = "shape kept for loader diagnostics")]
    shape: Vec<usize>,
}

fn f32t(file: &SafetensorsFile, name: &str) -> Result<F32Tensor> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing".to_string(),
    })?;
    let data = file.load_as_f32(name)?;
    Ok(F32Tensor {
        data,
        shape: desc.shape.clone(),
    })
}

/// Grouped linear: weight `[groups, in/g, out/g]`, input
/// `[T, groups * in/g]` -> `[T, groups * out/g]` (batch 1).
struct GroupedLinear {
    w: Vec<f32>,
    groups: usize,
    ws: usize,
    hs: usize,
}

impl GroupedLinear {
    fn load(file: &SafetensorsFile, name: &str) -> Result<Self> {
        let t = f32t(file, name)?;
        if t.shape.len() != 3 {
            return Err(SpeechError::Tensor {
                name: name.to_string(),
                why: format!("expected 3-D grouped weight, got {:?}", t.shape),
            });
        }
        Ok(GroupedLinear {
            w: t.data,
            groups: t.shape[0],
            ws: t.shape[1],
            hs: t.shape[2],
        })
    }

    fn forward(&self, x: &[f32], t_len: usize) -> Vec<f32> {
        let in_dim = self.groups * self.ws;
        let out_dim = self.groups * self.hs;
        let mut out = vec![0.0f32; t_len * out_dim];
        for t in 0..t_len {
            for g in 0..self.groups {
                let x_base = t * in_dim + g * self.ws;
                let o_base = t * out_dim + g * self.hs;
                for i in 0..self.ws {
                    let xv = x[x_base + i];
                    let w_base = (g * self.ws + i) * self.hs;
                    for h in 0..self.hs {
                        out[o_base + h] += xv * self.w[w_base + h];
                    }
                }
            }
        }
        out
    }
}

fn relu_in_place(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.max(0.0);
    }
}

/// MLX nn.GRU semantics with a zero initial hidden state (the only form
/// DeepFilterNet uses): `b` carries the folded PyTorch
/// `bias_ih + bias_hh[:2H]`, `bhn` the remaining `bias_hh[2H:]`. Weight
/// rows are gate-major `[r; z; n]`.
struct MlxGru {
    wx: Vec<f32>,  // [3H, in]
    wh: Vec<f32>,  // [3H, H]
    b: Vec<f32>,   // [3H] = bias_ih + bias_hh[rz gates]
    bhn: Vec<f32>, // [H] = bias_hh[n gate]
    hidden: usize,
    in_dim: usize,
}

impl MlxGru {
    fn forward(&self, x: &[f32], t_len: usize) -> Vec<f32> {
        let h = self.hidden;
        let mut proj = ops::linear(x, &self.wx, Some(&self.b), t_len, self.in_dim, 3 * h);
        let mut hidden = vec![0.0f32; h];
        let mut out = vec![0.0f32; t_len * h];
        let mut h_proj = vec![0.0f32; t_len * 3 * h];
        for t in 0..t_len {
            let row = t * 3 * h;
            // hidden @ Wh: the hidden state is nonzero from frame 1 on,
            // so the recurrent projection affects every gate.
            for g in 0..3 * h {
                let w_row = &self.wh[g * h..(g + 1) * h];
                h_proj[row + g] = hidden
                    .iter()
                    .zip(w_row.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
            let (r_slice, z_slice, n_slice) = (
                proj[row..row + h].to_vec(),
                proj[row + h..row + 2 * h].to_vec(),
                proj[row + 2 * h..row + 3 * h].to_vec(),
            );
            let mut r_row = r_slice;
            let mut z_row = z_slice;
            let n_row = n_slice;
            for d in 0..h {
                r_row[d] = 1.0 / (1.0 + (-(r_row[d] + h_proj[row + d])).exp());
                z_row[d] = 1.0 / (1.0 + (-(z_row[d] + h_proj[row + h + d])).exp());
            }
            for d in 0..h {
                proj[row + 2 * h + d] =
                    (n_row[d] + r_row[d] * (h_proj[row + 2 * h + d] + self.bhn[d])).tanh();
                hidden[d] = (1.0 - z_row[d]) * proj[row + 2 * h + d] + z_row[d] * hidden[d];
            }
            out[t * h..(t + 1) * h].copy_from_slice(&hidden);
        }
        out
    }
}

/// Grouped-linear + ReLU + GRU stack (the reference SqueezedGRU).
struct SqueezedGru {
    linear_in: GroupedLinear,
    grus: Vec<MlxGru>,
    linear_out: Option<GroupedLinear>,
}

impl SqueezedGru {
    fn forward(&self, x: &[f32], t_len: usize) -> Vec<f32> {
        let mut cur = self.linear_in.forward(x, t_len);
        relu_in_place(&mut cur);
        for gru in &self.grus {
            cur = gru.forward(&cur, t_len);
        }
        if let Some(lo) = &self.linear_out {
            cur = lo.forward(&cur, t_len);
            relu_in_place(&mut cur);
        }
        cur
    }
}

/// Inference batch norm over channel-major `[C, T, F]`.
struct BatchNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    mean: Vec<f32>,
    rstd: Vec<f32>,
}

impl BatchNorm {
    fn load(file: &SafetensorsFile, name: &str) -> Result<Self> {
        let weight = file.load_as_f32(&format!("{name}.weight"))?;
        let bias = file.load_as_f32(&format!("{name}.bias"))?;
        let mean = file.load_as_f32(&format!("{name}.running_mean"))?;
        let var = file.load_as_f32(&format!("{name}.running_var"))?;
        let rstd = var.iter().map(|v| 1.0 / (v + 1e-5).sqrt()).collect();
        Ok(BatchNorm {
            weight,
            bias,
            mean,
            rstd,
        })
    }

    fn forward(&self, x: &mut [f32], channels: usize) {
        let n = x.len() / channels;
        for c in 0..channels {
            let scale = self.weight[c] * self.rstd[c];
            let shift = self.bias[c] - self.mean[c] * scale;
            for v in &mut x[c * n..(c + 1) * n] {
                *v = *v * scale + shift;
            }
        }
    }
}

/// Grouped conv2d over channel-major `[C, T, F]`. Time stride is 1;
/// frequency stride is `fstride`. The time offset combines the
/// reference's crop-then-pad as input index `t + kt - left + crop`;
/// frequency uses symmetric zero padding `kF / 2`. Weight is PyTorch
/// layout `[out, in/g, kT, kF]`.
struct Conv2d {
    w: Vec<f32>, // [out, in_pg, kt, kf]
    out_ch: usize,
    in_pg: usize,
    groups: usize,
    kt: usize,
    kf: usize,
    left_t: usize,
    crop_t: usize,
    pad_f: usize,
    fstride: usize,
}

impl Conv2d {
    fn forward(&self, x: &[f32], t_len: usize, f_len: usize) -> (Vec<f32>, usize, usize) {
        let out_f = (f_len + 2 * self.pad_f - self.kf) / self.fstride + 1;
        let out_t = t_len;
        let mut out = vec![0.0f32; self.out_ch * out_t * out_f];
        let out_pg = self.out_ch / self.groups;
        for o in 0..self.out_ch {
            let g = o / out_pg;
            for t in 0..out_t {
                for f in 0..out_f {
                    let mut acc = 0.0f32;
                    for cg in 0..self.in_pg {
                        let ic = g * self.in_pg + cg;
                        for kt in 0..self.kt {
                            let ti = t + kt + self.crop_t;
                            let ti = match ti.checked_sub(self.left_t) {
                                Some(v) => v,
                                None => continue,
                            };
                            if ti >= t_len {
                                continue;
                            }
                            for kf in 0..self.kf {
                                let fi = f * self.fstride + kf;
                                let fi = match fi.checked_sub(self.pad_f) {
                                    Some(v) => v,
                                    None => continue,
                                };
                                if fi >= f_len {
                                    continue;
                                }
                                acc += x[ic * t_len * f_len + ti * f_len + fi]
                                    * self.w[((o * self.in_pg + cg) * self.kt + kt) * self.kf + kf];
                            }
                        }
                    }
                    out[o * out_t * out_f + t * out_f + f] = acc;
                }
            }
        }
        (out, out_t, out_f)
    }
}

/// Depthwise transposed conv2d over `[C, T, F]` (the decoder only uses
/// depthwise 1x3, kT = 1): doubles F with stride 2.
struct ConvTransposeDw {
    w: Vec<f32>, // [C, 1, 1, kF]
    channels: usize,
    kf: usize,
    pad_f: usize,
    out_pad_f: usize,
    fstride: usize,
}

impl ConvTransposeDw {
    fn forward(&self, x: &[f32], t_len: usize, f_len: usize) -> (Vec<f32>, usize, usize) {
        let out_f = (f_len - 1) * self.fstride - 2 * self.pad_f + self.kf + self.out_pad_f;
        let mut out = vec![0.0f32; self.channels * t_len * out_f];
        for c in 0..self.channels {
            for t in 0..t_len {
                for fi in 0..f_len {
                    let xv = x[c * t_len * f_len + t * f_len + fi];
                    for k in 0..self.kf {
                        let fo = fi * self.fstride + k;
                        let fo = match fo.checked_sub(self.pad_f) {
                            Some(v) => v,
                            None => continue,
                        };
                        if fo >= out_f {
                            continue;
                        }
                        out[c * t_len * out_f + t * out_f + fo] += xv * self.w[c * self.kf + k];
                    }
                }
            }
        }
        (out, t_len, out_f)
    }
}

/// 1x1 pointwise conv over `[C, T, F]`.
struct PointwiseConv {
    w: Vec<f32>, // [out, in, 1, 1]
    out_ch: usize,
    in_ch: usize,
}

impl PointwiseConv {
    fn forward(&self, x: &[f32], t_len: usize, f_len: usize) -> Vec<f32> {
        let n = t_len * f_len;
        let mut out = vec![0.0f32; self.out_ch * n];
        for o in 0..self.out_ch {
            let w_row = &self.w[o * self.in_ch..(o + 1) * self.in_ch];
            for (i, out_v) in out[o * n..(o + 1) * n].iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (c, wv) in w_row.iter().enumerate() {
                    acc += x[c * n + i] * wv;
                }
                *out_v = acc;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Network blocks
// ---------------------------------------------------------------------------

/// Encoder/decoder conv stage: main conv + optional pointwise + BN +
/// relu. The pointwise BN rides on the pointwise output; a stage without
/// pointwise applies the BN to the main conv output.
struct ConvStage {
    main: Conv2d,
    pointwise: Option<(PointwiseConv, BatchNorm)>,
    bn: BatchNorm,
}

impl ConvStage {
    fn forward(&self, x: &[f32], t_len: usize, f_len: usize) -> (Vec<f32>, usize, usize) {
        let (mut x, t_out, f_mid) = self.main.forward(x, t_len, f_len);
        match &self.pointwise {
            Some((pw, bn)) => {
                let mut y = pw.forward(&x, t_out, f_mid);
                bn.forward(&mut y, pw.out_ch);
                relu_in_place(&mut y);
                (y, t_out, f_mid)
            }
            None => {
                self.bn.forward(&mut x, self.main.out_ch);
                relu_in_place(&mut x);
                (x, t_out, f_mid)
            }
        }
    }
}

struct Encoder {
    erb_conv0: ConvStage,
    erb_conv1: ConvStage,
    erb_conv2: ConvStage,
    erb_conv3: ConvStage,
    df_conv0: ConvStage,
    df_conv1: ConvStage,
    df_fc_emb: GroupedLinear,
    emb_gru: SqueezedGru,
    // lsnr head weights: loaded for completeness; the head output does
    // not affect the enhanced spectrum.
    #[allow(dead_code, reason = "lsnr head is side-output only")]
    lsnr_w: Vec<f32>,
    #[allow(dead_code, reason = "lsnr head is side-output only")]
    lsnr_b: Vec<f32>,
    #[allow(dead_code, reason = "lsnr head is side-output only")]
    lsnr_scale: f32,
    #[allow(dead_code, reason = "lsnr head is side-output only")]
    lsnr_offset: f32,
    #[allow(dead_code, reason = "v2 layout switch; v3 keeps it false")]
    enc_concat: bool,
}

struct ErbDecoder {
    emb_gru: SqueezedGru,
    conv3p: (Vec<f32>, BatchNorm),
    conv2p: (Vec<f32>, BatchNorm),
    conv1p: (Vec<f32>, BatchNorm),
    conv0p: (Vec<f32>, BatchNorm),
    convt3: (Conv2d, PointwiseConv, BatchNorm),
    convt2: (ConvTransposeDw, PointwiseConv, BatchNorm),
    convt1: (ConvTransposeDw, PointwiseConv, BatchNorm),
    conv0_out: (Conv2d, BatchNorm),
}

struct DfDecoder {
    convp_main: Conv2d,
    convp_pw: PointwiseConv,
    convp_bn: BatchNorm,
    df_gru: SqueezedGru,
    df_skip: GroupedLinear,
    df_out: GroupedLinear,
    #[allow(dead_code, reason = "mirrors the reference config")]
    df_order: usize,
    #[allow(dead_code, reason = "mirrors the reference config")]
    df_bins: usize,
}

struct DfNet {
    erb_fb: Vec<f32>,     // [481, 32]
    erb_inv_fb: Vec<f32>, // [32, 481]
    enc: Encoder,
    erb_dec: ErbDecoder,
    df_dec: DfDecoder,
    nb_df: usize,
    freq_bins: usize,
    #[allow(dead_code, reason = "kept for parity with the reference config")]
    df_order: usize,
    #[allow(dead_code, reason = "kept for parity with the reference config")]
    df_lookahead: usize,
    #[allow(dead_code, reason = "v2 layout switch; v3 checkpoints keep it false")]
    enc_concat: bool,
}

impl DfNet {
    /// Forward: `spec` interleaved [T, F, 2] re/im; `feat_erb` [T, E];
    /// `feat_df` [T, D, 2]. Returns the enhanced spectrum interleaved
    /// [T, F, 2].
    fn forward(&self, spec: &[f32], feat_erb: &[f32], feat_df: &[f32], t_len: usize) -> Vec<f32> {
        let f_bins = self.freq_bins;
        // feat_df [T, D, 2] -> [2, T, D]
        let mut feat_spec = vec![0.0f32; 2 * t_len * self.nb_df];
        for t in 0..t_len {
            for d in 0..self.nb_df {
                for c in 0..2 {
                    feat_spec[c * t_len * self.nb_df + t * self.nb_df + d] =
                        feat_df[(t * self.nb_df + d) * 2 + c];
                }
            }
        }
        // conv_lookahead = 2: shift left 2, pad right 2 on time.
        let feat_erb = lookahead_shift(feat_erb, t_len, 32, 2);
        let feat_spec = lookahead_shift(&feat_spec, t_len, self.nb_df, 2);

        // Encoder. ERB chain: [1,1,T,32] -> C=64, F 32 -> 16 -> 8 -> 8.
        // DF chain: [1,2,T,96] -> C=64, F stays 96 (df_conv1 feeds only
        // the embedding; the DF decoder consumes c0 at 96 bins).
        let (e0, _, f_e0) = self.enc.erb_conv0.forward(&feat_erb, t_len, 32);
        let (e1, _, f_e1) = self.enc.erb_conv1.forward(&e0, t_len, f_e0);
        let (e2, _, f_e2) = self.enc.erb_conv2.forward(&e1, t_len, f_e1);
        let (e3, _, f_e3) = self.enc.erb_conv3.forward(&e2, t_len, f_e2);
        let (c0_df, _, f_c0) = self.enc.df_conv0.forward(&feat_spec, t_len, self.nb_df);
        let (c1, _, f_c1) = self.enc.df_conv1.forward(&c0_df, t_len, f_c0);

        // cemb from df chain, emb from erb chain.
        let cemb_t = f_c1 * 64;
        // The reference flattens [B, T, F, C] -> [T, F*C] (f-major).
        let mut cemb_flat = vec![0.0f32; t_len * cemb_t];
        for t in 0..t_len {
            for c in 0..64 {
                for f in 0..f_c1 {
                    cemb_flat[t * cemb_t + f * 64 + c] = c1[c * t_len * f_c1 + t * f_c1 + f];
                }
            }
        }
        let mut cemb = self.enc.df_fc_emb.forward(&cemb_flat, t_len);
        // df_fc_emb = Sequential(GroupedLinearEinsum, ReLU).
        relu_in_place(&mut cemb);
        let emb_t = f_e3 * 64;
        let mut emb = vec![0.0f32; t_len * emb_t];
        for t in 0..t_len {
            for c in 0..64 {
                for f in 0..f_e3 {
                    emb[t * emb_t + f * 64 + c] = e3[c * t_len * f_e3 + t * f_e3 + f];
                }
            }
        }
        if self.enc_concat {
            // v2 layout: concatenate over the feature axis.
            let mut out = vec![0.0f32; t_len * (emb_t + cemb_t)];
            for t in 0..t_len {
                out[t * (emb_t + cemb_t)..t * (emb_t + cemb_t) + emb_t]
                    .copy_from_slice(&emb[t * emb_t..(t + 1) * emb_t]);
                out[t * (emb_t + cemb_t) + emb_t..(t + 1) * (emb_t + cemb_t)]
                    .copy_from_slice(&cemb[t * cemb_t..(t + 1) * cemb_t]);
            }
            emb = out;
        } else {
            for (e, c) in emb.iter_mut().zip(&cemb) {
                *e += c;
            }
        }

        let emb_gru_out = self.enc.emb_gru.forward(&emb, t_len);

        // ERB decoder.
        let emb_dec = self.erb_dec.emb_gru.forward(&emb_gru_out, t_len);
        // reshape [T, f_e3, 64] -> [64, T, f_e3]
        // emb_dec [T, 64, f_e3] (f-major flatten of [B, T, f8, C*?]) ->
        // back to channel-major [64, T, f_e3].
        let mut emb_map = vec![0.0f32; 64 * t_len * f_e3];
        for t in 0..t_len {
            for c in 0..64 {
                for f in 0..f_e3 {
                    emb_map[c * t_len * f_e3 + t * f_e3 + f] = emb_dec[t * 64 * f_e3 + f * 64 + c];
                }
            }
        }
        // pathway + merge + convt chain
        let (p3, _, _) = pathway(&self.erb_dec.conv3p, &e3, t_len, f_e3);
        let d3 = add(&p3, &emb_map);
        let (mut d3, _, f_d3) = convt_regular(&self.erb_dec.convt3, &d3, t_len, f_e3);
        relu_in_place(&mut d3);

        let (p2, _, _) = pathway(&self.erb_dec.conv2p, &e2, t_len, f_e2);
        let d2 = add(&p2, &d3);
        let (mut d2, _, f_d2) = {
            let (conv_t, pw, bn) = &self.erb_dec.convt2;
            let (y, t2, f2) = conv_t.forward(&d2, t_len, f_d3);
            let mut y2 = pw.forward(&y, t2, f2);
            bn.forward(&mut y2, pw.out_ch);
            (y2, t2, f2)
        };
        relu_in_place(&mut d2);

        let (p1, _, _) = pathway(&self.erb_dec.conv1p, &e1, t_len, f_e1);
        let d1_pre = add(&p1, &d2);
        let (mut d1, _, _) = {
            let (conv_t, pw, bn) = &self.erb_dec.convt1;
            let (y, t2, f2) = conv_t.forward(&d1_pre, t_len, f_d2);
            let mut y2 = pw.forward(&y, t2, f2);
            bn.forward(&mut y2, pw.out_ch);
            (y2, t2, f2)
        };
        relu_in_place(&mut d1);

        let (p0, _, _) = pathway(&self.erb_dec.conv0p, &e0, t_len, f_e0);
        let d0 = add(&p0, &d1);
        // output conv [1, 64, T, 32] -> [1, 1, T, 32]
        let (mut m, _, _) = self.erb_dec.conv0_out.0.forward(&d0, t_len, f_e0);
        self.erb_dec.conv0_out.1.forward(&mut m, 1);
        for v in m.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        // mask[T, 32] @ inv_fb[32, 481]
        let mut mask = vec![0.0f32; t_len * f_bins];
        for t in 0..t_len {
            for f in 0..f_bins {
                let mut acc = 0.0f32;
                for (erb, mv) in m[t * 32..t * 32 + 32].iter().enumerate() {
                    acc += mv * self.erb_inv_fb[erb * f_bins + f];
                }
                mask[t * f_bins + f] = acc;
            }
        }

        // DF coefficients consume c0_df (96 bins), not c1.
        let (c0, _, _) = self.df_dec.convp_main.forward(&c0_df, t_len, f_c0);
        let mut c0 = self.df_dec.convp_pw.forward(&c0, t_len, self.nb_df);
        self.df_dec.convp_bn.forward(&mut c0, 10);
        relu_in_place(&mut c0);
        // c0 layout [10, T, 96] -> [T, 96, 10]
        let mut c0_t = vec![0.0f32; t_len * self.nb_df * 10];
        for t in 0..t_len {
            for o in 0..10 {
                for f in 0..self.nb_df {
                    c0_t[(t * self.nb_df + f) * 10 + o] =
                        c0[o * t_len * self.nb_df + t * self.nb_df + f];
                }
            }
        }
        let c = self.df_dec.df_gru.forward(&emb_gru_out, t_len);
        let skip = self.df_dec.df_skip.forward(&emb_gru_out, t_len);
        let mut c_skip = c.clone();
        for (cv, sv) in c_skip.iter_mut().zip(&skip) {
            *cv += sv;
        }
        let mut coefs = self.df_dec.df_out.forward(&c_skip, t_len);
        for v in coefs.iter_mut() {
            *v = v.tanh();
        }
        // [T, 96, 10] + c0_t
        for (cv, bv) in coefs.iter_mut().zip(&c0_t) {
            *cv += bv;
        }

        // Deep filtering over the raw spectrum: coefs[T, 96, 10] viewed
        // as [T, 96, order, 2] with order-major inside the 10.
        let mut low = vec![0.0f32; t_len * self.nb_df * 2];
        let pad_l = self.df_order - 1 - self.df_lookahead;
        #[allow(
            clippy::needless_range_loop,
            reason = "time/bins/order are tensor axes, not collections"
        )]
        for t in 0..t_len {
            for f in 0..self.nb_df {
                let mut or_ = 0.0f32;
                let mut oi = 0.0f32;
                for k in 0..self.df_order {
                    let ts = t as isize + k as isize - pad_l as isize;
                    if ts < 0 || ts as usize >= t_len {
                        continue;
                    }
                    let ts = ts as usize;
                    let sr = spec[(ts * f_bins + f) * 2];
                    let si = spec[(ts * f_bins + f) * 2 + 1];
                    let ci = (t * self.nb_df + f) * self.df_order * 2 + k * 2;
                    let cr = coefs[ci];
                    let cim = coefs[ci + 1];
                    or_ += sr * cr - si * cim;
                    oi += sr * cim + si * cr;
                }
                low[(t * self.nb_df + f) * 2] = or_;
                low[(t * self.nb_df + f) * 2 + 1] = oi;
            }
        }
        // DF3: low bins from deep filtering, high bins from the masked
        // spectrum. Mask was computed against the raw spec.
        let mut out = vec![0.0f32; t_len * f_bins * 2];
        for t in 0..t_len {
            for f in 0..f_bins {
                for c in 0..2 {
                    let v = if f < self.nb_df {
                        low[(t * self.nb_df + f) * 2 + c]
                    } else {
                        let m = mask[t * f_bins + f];
                        m * spec[(t * f_bins + f) * 2 + c]
                    };
                    out[(t * f_bins + f) * 2 + c] = v;
                }
            }
        }
        out
    }
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn lookahead_shift(x: &[f32], t_len: usize, inner: usize, shift: usize) -> Vec<f32> {
    // [C, T, inner] -> shift left by `shift` on T, pad right.
    let channels = x.len() / (t_len * inner);
    let mut out = vec![0.0f32; x.len()];
    for c in 0..channels {
        for t in 0..t_len {
            let src = t + shift;
            if src < t_len {
                for i in 0..inner {
                    out[(c * t_len + t) * inner + i] = x[(c * t_len + src) * inner + i];
                }
            }
        }
    }
    out
}

fn pathway(
    dw_bn: &(Vec<f32>, BatchNorm),
    x: &[f32],
    t_len: usize,
    f_len: usize,
) -> (Vec<f32>, usize, usize) {
    let (dw, bn) = dw_bn;
    let mut y = x.to_vec();
    for (c, wv) in dw.iter().enumerate() {
        for v in &mut y[c * t_len * f_len..(c + 1) * t_len * f_len] {
            *v *= wv;
        }
    }
    bn.forward(&mut y, dw.len());
    relu_in_place(&mut y);
    (y, t_len, f_len)
}

fn convt_regular(
    conv_pw_bn: &(Conv2d, PointwiseConv, BatchNorm),
    x: &[f32],
    t_len: usize,
    f_len: usize,
) -> (Vec<f32>, usize, usize) {
    let (conv, pw, bn) = conv_pw_bn;
    let (y, t, f) = conv.forward(x, t_len, f_len);
    let mut y = pw.forward(&y, t, f);
    bn.forward(&mut y, pw.out_ch);
    (y, t, f)
}

// ---------------------------------------------------------------------------
// Model driver
// ---------------------------------------------------------------------------

/// The loaded DeepFilterNet enhancement model (v3 layout; v2 shares the
/// same code path through `enc_concat`).
pub struct DeepFilterNet {
    net: DfNet,
    fft_size: usize,
    hop_size: usize,
    nb_erb: usize,
    nb_df: usize,
    #[allow(dead_code, reason = "fixed 48 kHz by the architecture")]
    sample_rate: u32,
    window: Vec<f32>,
    wnorm: f32,
    norm_alpha: f32,
}

fn vorbis_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let n = i as f32;
            let inner = (0.5 * std::f32::consts::PI * (n + 0.5) / (size as f32 / 2.0)).sin();
            (0.5 * std::f32::consts::PI * inner * inner).sin()
        })
        .collect()
}

impl DeepFilterNet {
    /// Opens a model directory holding `config.json` and
    /// `model.safetensors` (the v3 conversion; v2 loads through the same
    /// code path via `enc_concat`).
    pub fn open(dir: &Path) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(
                |e| SpeechError::BadConfig {
                    field: "config.json".to_string(),
                    why: e.to_string(),
                },
            )?)
            .map_err(|e| SpeechError::BadConfig {
                field: "config.json".to_string(),
                why: e.to_string(),
            })?;
        let num = |key: &str, default: u64| {
            config.get(key).and_then(|v| v.as_u64()).unwrap_or(default) as usize
        };
        let fft_size = num("fft_size", 960);
        let hop_size = num("hop_size", 480);
        let nb_erb = num("nb_erb", 32);
        let nb_df = num("nb_df", 96);
        let df_order = num("df_order", 5);
        let df_lookahead = num("df_lookahead", 0);
        let sample_rate = num("sample_rate", 48_000) as u32;
        let enc_concat = config
            .get("enc_concat")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let file = SafetensorsFile::open(&dir.join("model.safetensors"))?;
        let erb_fb = file.load_as_f32("erb_fb")?;
        let erb_inv_fb = file.load_as_f32("mask.erb_inv_fb")?;

        // ---- encoder convs ----
        // erb_conv0: non-separable {1: conv 3x3, 2: BN}
        let erb_conv0 = ConvStage {
            main: load_conv2d(&file, "enc.erb_conv0.1.weight", 1, 64, 3, 3, 2, 0, 1, 1)?,
            pointwise: None,
            bn: BatchNorm::load(&file, "enc.erb_conv0.2")?,
        };
        // erb_conv1/2/3: depthwise {0: conv k[1,3], 1: pointwise, 2: BN}
        let erb_conv1 = ConvStage {
            main: load_conv2d(&file, "enc.erb_conv1.0.weight", 64, 64, 1, 3, 0, 0, 1, 2)?,
            pointwise: Some((
                load_pointwise(&file, "enc.erb_conv1.1.weight", 64, 64)?,
                BatchNorm::load(&file, "enc.erb_conv1.2")?,
            )),
            bn: BatchNorm::load(&file, "enc.erb_conv1.2")?,
        };
        let erb_conv2 = ConvStage {
            main: load_conv2d(&file, "enc.erb_conv2.0.weight", 64, 64, 1, 3, 0, 0, 1, 2)?,
            pointwise: Some((
                load_pointwise(&file, "enc.erb_conv2.1.weight", 64, 64)?,
                BatchNorm::load(&file, "enc.erb_conv2.2")?,
            )),
            bn: BatchNorm::load(&file, "enc.erb_conv2.2")?,
        };
        let erb_conv3 = ConvStage {
            main: load_conv2d(&file, "enc.erb_conv3.0.weight", 64, 64, 1, 3, 0, 0, 1, 1)?,
            pointwise: Some((
                load_pointwise(&file, "enc.erb_conv3.1.weight", 64, 64)?,
                BatchNorm::load(&file, "enc.erb_conv3.2")?,
            )),
            bn: BatchNorm::load(&file, "enc.erb_conv3.2")?,
        };
        // df_conv0: {1: conv 3x3 groups 2, 2: pointwise, 3: BN}
        let df_conv0 = ConvStage {
            main: load_conv2d(&file, "enc.df_conv0.1.weight", 2, 64, 3, 3, 2, 0, 1, 1)?,
            pointwise: Some((
                load_pointwise(&file, "enc.df_conv0.2.weight", 64, 64)?,
                BatchNorm::load(&file, "enc.df_conv0.3")?,
            )),
            bn: BatchNorm::load(&file, "enc.df_conv0.3")?,
        };
        // df_conv1: {0: depthwise k[1,3] stride 2, 1: pointwise, 2: BN}
        let df_conv1 = ConvStage {
            main: load_conv2d(&file, "enc.df_conv1.0.weight", 64, 64, 1, 3, 0, 0, 1, 2)?,
            pointwise: Some((
                load_pointwise(&file, "enc.df_conv1.1.weight", 64, 64)?,
                BatchNorm::load(&file, "enc.df_conv1.2")?,
            )),
            bn: BatchNorm::load(&file, "enc.df_conv1.2")?,
        };

        let df_fc_emb = GroupedLinear::load(&file, "enc.df_fc_emb.0.weight")?;
        let emb_gru = load_squeezed_gru(
            &file,
            "enc.emb_gru",
            1,
            Some("enc.emb_gru.linear_out.0.weight"),
        )?;
        let lsnr_w = file.load_as_f32("enc.lsnr_fc.0.weight")?;
        let lsnr_b = file.load_as_f32("enc.lsnr_fc.0.bias")?;

        let enc = Encoder {
            erb_conv0,
            erb_conv1,
            erb_conv2,
            erb_conv3,
            df_conv0,
            df_conv1,
            df_fc_emb,
            emb_gru,
            lsnr_w,
            lsnr_b,
            lsnr_scale: 50.0,
            lsnr_offset: -15.0,
            enc_concat,
        };

        // ---- ERB decoder ----
        let emb_gru_dec = load_squeezed_gru(
            &file,
            "erb_dec.emb_gru",
            2,
            Some("erb_dec.emb_gru.linear_out.0.weight"),
        )?;
        // Pathway convs are depthwise 1x1: one scale per channel.
        let conv3p = (
            file.load_as_f32("erb_dec.conv3p.0.weight")?,
            BatchNorm::load(&file, "erb_dec.conv3p.1")?,
        );
        let conv2p = (
            file.load_as_f32("erb_dec.conv2p.0.weight")?,
            BatchNorm::load(&file, "erb_dec.conv2p.1")?,
        );
        let conv1p = (
            file.load_as_f32("erb_dec.conv1p.0.weight")?,
            BatchNorm::load(&file, "erb_dec.conv1p.1")?,
        );
        let conv0p = (
            file.load_as_f32("erb_dec.conv0p.0.weight")?,
            BatchNorm::load(&file, "erb_dec.conv0p.1")?,
        );
        // convt3: regular conv block {0: depthwise k[1,3], 1: pw, 2: BN}
        let convt3 = (
            load_conv2d(&file, "erb_dec.convt3.0.weight", 64, 64, 1, 3, 0, 0, 1, 1)?,
            load_pointwise(&file, "erb_dec.convt3.1.weight", 64, 64)?,
            BatchNorm::load(&file, "erb_dec.convt3.2")?,
        );
        let convt2 = (
            load_conv_t_dw(&file, "erb_dec.convt2.0.weight", 64, 3, 1, 1, 2)?,
            load_pointwise(&file, "erb_dec.convt2.1.weight", 64, 64)?,
            BatchNorm::load(&file, "erb_dec.convt2.2")?,
        );
        let convt1 = (
            load_conv_t_dw(&file, "erb_dec.convt1.0.weight", 64, 3, 1, 1, 2)?,
            load_pointwise(&file, "erb_dec.convt1.1.weight", 64, 64)?,
            BatchNorm::load(&file, "erb_dec.convt1.2")?,
        );
        let conv0_out = (
            load_conv2d(&file, "erb_dec.conv0_out.0.weight", 1, 1, 1, 3, 0, 0, 1, 1)?,
            BatchNorm::load(&file, "erb_dec.conv0_out.1")?,
        );
        let erb_dec = ErbDecoder {
            emb_gru: emb_gru_dec,
            conv3p,
            conv2p,
            conv1p,
            conv0p,
            convt3,
            convt2,
            convt1,
            conv0_out,
        };

        // ---- DF decoder ----
        let convp_main = load_conv2d(&file, "df_dec.df_convp.1.weight", 2, 10, 5, 1, 4, 0, 1, 1)?;
        let convp_pw = load_pointwise(&file, "df_dec.df_convp.2.weight", 10, 10)?;
        let convp_bn = BatchNorm::load(&file, "df_dec.df_convp.3")?;
        let df_gru = load_squeezed_gru(&file, "df_dec.df_gru", 2, None)?;
        let df_skip = GroupedLinear::load(&file, "df_dec.df_skip.weight")?;
        let df_out = GroupedLinear::load(&file, "df_dec.df_out.0.weight")?;
        let df_dec = DfDecoder {
            convp_main,
            convp_pw,
            convp_bn,
            df_gru,
            df_skip,
            df_out,
            df_order,
            df_bins: nb_df,
        };

        let net = DfNet {
            erb_fb,
            erb_inv_fb,
            enc,
            erb_dec,
            df_dec,
            nb_df,
            freq_bins: fft_size / 2 + 1,
            df_order,
            df_lookahead,
            enc_concat,
        };

        // Match df.utils.get_norm_alpha() rounding: the first 3-decimal
        // rounding that drops below 1.0.
        let a_raw = (-(hop_size as f64) / sample_rate as f64).exp();
        let mut precision = 3;
        let mut alpha = 1.0f64;
        while alpha >= 1.0 {
            alpha = round_to(a_raw, precision);
            precision += 1;
        }

        Ok(DeepFilterNet {
            net,
            fft_size,
            hop_size,
            nb_erb,
            nb_df,
            sample_rate,
            window: vorbis_window(fft_size),
            wnorm: (2.0 * hop_size as f64 / (fft_size as f64 * fft_size as f64)) as f32,
            norm_alpha: alpha as f32,
        })
    }

    /// Enhances 48 kHz mono samples, matching the reference
    /// `enhance_array` (pad, STFT, features, network, ISTFT, delay
    /// compensation, clip).
    pub fn enhance(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let orig = samples.len();
        let mut padded = vec![0.0f32; orig + self.hop_size + self.fft_size];
        padded[self.hop_size..self.hop_size + orig].copy_from_slice(samples);

        let opts = StftOptions {
            fft_size: self.fft_size,
            hop: self.hop_size,
            window: self.window.clone(),
            center: false,
        };
        let spectra = stft(&padded, &opts).map_err(|e| SpeechError::Audio(e.to_string()))?;
        let t_len = spectra.len();
        let f_bins = self.fft_size / 2 + 1;
        // Interleaved [T, F, 2], scaled by wnorm.
        let mut spec = vec![0.0f32; t_len * f_bins * 2];
        for (t, frame) in spectra.iter().enumerate() {
            for (f, z) in frame.iter().enumerate() {
                spec[(t * f_bins + f) * 2] = z.re * self.wnorm;
                spec[(t * f_bins + f) * 2 + 1] = z.im * self.wnorm;
            }
        }

        // Features.
        let alpha = self.norm_alpha;
        // ERB
        let mut erb = vec![0.0f32; t_len * self.nb_erb];
        for t in 0..t_len {
            for e in 0..self.nb_erb {
                let mut acc = 0.0f32;
                for f in 0..f_bins {
                    let re = spec[(t * f_bins + f) * 2];
                    let im = spec[(t * f_bins + f) * 2 + 1];
                    acc += (re * re + im * im) * self.net.erb_fb[f * self.nb_erb + e];
                }
                erb[t * self.nb_erb + e] = acc;
            }
        }
        // erb_db = 10 log10(erb + 1e-10), then sequential mean norm.
        let mut erb_db = erb.clone();
        for v in erb_db.iter_mut() {
            *v = 10.0 * (*v + 1e-10).log10();
        }
        let mut state: Vec<f32> = (0..self.nb_erb)
            .map(|i| -60.0 + (i as f32 / (self.nb_erb - 1) as f32) * (-30.0))
            .collect();
        let mut feat_erb = vec![0.0f32; t_len * self.nb_erb];
        let one_minus = 1.0 - alpha;
        for t in 0..t_len {
            for e in 0..self.nb_erb {
                state[e] = erb_db[t * self.nb_erb + e] * one_minus + state[e] * alpha;
                feat_erb[t * self.nb_erb + e] = (erb_db[t * self.nb_erb + e] - state[e]) / 40.0;
            }
        }

        // DF unit-norm features over the first nb_df bins.
        let mut unit_state: Vec<f32> = (0..self.nb_df)
            .map(|i| 0.001 + (i as f32 / (self.nb_df - 1) as f32) * (-0.0009))
            .collect();
        let mut feat_df = vec![0.0f32; t_len * self.nb_df * 2];
        for t in 0..t_len {
            for d in 0..self.nb_df {
                let re = spec[(t * f_bins + d) * 2];
                let im = spec[(t * f_bins + d) * 2 + 1];
                unit_state[d] = (re * re + im * im).sqrt() * one_minus + unit_state[d] * alpha;
                let denom = unit_state[d].sqrt().max(1e-12);
                feat_df[(t * self.nb_df + d) * 2] = re / denom;
                feat_df[(t * self.nb_df + d) * 2 + 1] = im / denom;
            }
        }

        let spec_e = self.net.forward(&spec, &feat_erb, &feat_df, t_len);

        // Inverse STFT: /wnorm, vorbis window, w^2 normalization, then
        // delay compensation d = fft - hop.
        let mut enh = vec![0.0f32; spec_e.len()];
        for (o, v) in enh.iter_mut().zip(&spec_e) {
            *o = v / self.wnorm;
        }
        let iopts = StftOptions {
            fft_size: self.fft_size,
            hop: self.hop_size,
            window: self.window.clone(),
            center: false,
        };
        let mut spectra_inv = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let mut frame = Vec::with_capacity(f_bins);
            for f in 0..f_bins {
                // DC and Nyquist are purely real for a real signal.
                let re = enh[(t * f_bins + f) * 2];
                let im = enh[(t * f_bins + f) * 2 + 1];
                if f == 0 || f == f_bins - 1 {
                    frame.push(turbospark_audio::fft::ComplexF32::new(re, 0.0));
                } else {
                    frame.push(turbospark_audio::fft::ComplexF32::new(re, im));
                }
            }
            spectra_inv.push(frame);
        }
        let length = orig + self.hop_size + self.fft_size;
        let audio =
            istft(&spectra_inv, &iopts, length).map_err(|e| SpeechError::Audio(e.to_string()))?;

        let d = self.fft_size - self.hop_size;
        let start = d.min(audio.len());
        let end = (orig + d).min(audio.len());
        let mut y = audio[start..end].to_vec();
        if y.len() < orig {
            y.resize(orig, 0.0);
        }
        for v in y.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        Ok(y)
    }
}

fn round_to(v: f64, precision: u32) -> f64 {
    let m = 10f64.powi(precision as i32);
    (v * m).round() / m
}

#[allow(
    clippy::too_many_arguments,
    reason = "kernel geometry mirrors the checkpoint"
)]
fn load_conv2d(
    file: &SafetensorsFile,
    name: &str,
    groups: usize,
    out_ch: usize,
    kt: usize,
    kf: usize,
    left_t: usize,
    crop_t: usize,
    pad_f: usize,
    fstride: usize,
) -> Result<Conv2d> {
    let t = f32t(file, name)?;
    let in_pg = t.shape[1];
    if t.shape[0] != out_ch || t.shape[2] != kt || t.shape[3] != kf {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("unexpected conv shape {:?}", t.shape),
        });
    }
    if fstride == 0 {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: "fstride 0".to_string(),
        });
    }
    Ok(Conv2d {
        w: t.data,
        out_ch,
        in_pg,
        groups,
        kt,
        kf,
        left_t,
        crop_t,
        pad_f,
        fstride,
    })
}

fn load_pointwise(
    file: &SafetensorsFile,
    name: &str,
    out_ch: usize,
    in_ch: usize,
) -> Result<PointwiseConv> {
    let t = f32t(file, name)?;
    if t.data.len() != out_ch * in_ch {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("pointwise weight {} != {}x{}", t.data.len(), out_ch, in_ch),
        });
    }
    Ok(PointwiseConv {
        w: t.data,
        out_ch,
        in_ch,
    })
}

fn load_conv_t_dw(
    file: &SafetensorsFile,
    name: &str,
    channels: usize,
    kf: usize,
    pad_f: usize,
    out_pad_f: usize,
    fstride: usize,
) -> Result<ConvTransposeDw> {
    let t = f32t(file, name)?;
    if t.data.len() != channels * kf {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("transposed depthwise weight {} wrong size", t.data.len()),
        });
    }
    Ok(ConvTransposeDw {
        w: t.data,
        channels,
        kf,
        pad_f,
        out_pad_f,
        fstride,
    })
}

fn load_squeezed_gru(
    file: &SafetensorsFile,
    base: &str,
    layers: usize,
    linear_out: Option<&str>,
) -> Result<SqueezedGru> {
    let linear_in = GroupedLinear::load(file, &format!("{base}.linear_in.0.weight"))?;
    let mut grus = Vec::new();
    for l in 0..layers {
        let wx = file.load_as_f32(&format!("{base}.gru.weight_ih_l{l}"))?;
        let bias_ih = file.load_as_f32(&format!("{base}.gru.bias_ih_l{l}"))?;
        let bias_hh = file.load_as_f32(&format!("{base}.gru.bias_hh_l{l}"))?;
        let hidden = bias_ih.len() / 3;
        let in_dim = wx.len() / (3 * hidden);
        // PyTorch fold: b = bias_ih + [bias_hh[:2H] | 0], bhn = bias_hh[2H:].
        let mut b = bias_ih;
        for g in 0..2 * hidden {
            b[g] += bias_hh[g];
        }
        let bhn = bias_hh[2 * hidden..].to_vec();
        let wh = file.load_as_f32(&format!("{base}.gru.weight_hh_l{l}"))?;
        grus.push(MlxGru {
            wx,
            wh,
            b,
            bhn,
            hidden,
            in_dim,
        });
    }
    let linear_out = match linear_out {
        Some(name) => Some(GroupedLinear::load(file, name)?),
        None => None,
    };
    Ok(SqueezedGru {
        linear_in,
        grus,
        linear_out,
    })
}
