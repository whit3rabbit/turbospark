//! S2A flow-matching estimator (`s2a.py`): semantic codes + the T2S LM
//! latent -> the target mel via a 13-block DiT with U-Net skips, RoPE
//! tables loaded from the checkpoint, a gated WaveNet tail, and the
//! Euler CFG solver that re-zeros the prompt region every step.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::load_tensor;
use crate::{ops, Result};

#[derive(Debug, Clone)]
pub struct S2AConfig {
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub depth: usize,
    pub wavenet_layers: usize,
    pub freqs_positions: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub speaker_dim: usize,
    pub encoder_dim: usize,
    pub mels: usize,
    /// The T2S latent width concatenated with the code projection.
    pub latent_dim: usize,
}

impl Default for S2AConfig {
    fn default() -> Self {
        S2AConfig {
            hidden: 512,
            num_heads: 8,
            head_dim: 64,
            depth: 13,
            wavenet_layers: 8,
            freqs_positions: 4096,
            codebook_size: 8194,
            codebook_dim: 8,
            speaker_dim: 192,
            encoder_dim: 1024,
            mels: 80,
            latent_dim: 1280,
        }
    }
}

struct DiTBlock {
    attn_modulation: (Vec<f32>, Vec<f32>),
    attn_norm: Vec<f32>,
    ffn_modulation: (Vec<f32>, Vec<f32>),
    ffn_norm: Vec<f32>,
    wqkv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    skip_in: Option<(Vec<f32>, Vec<f32>)>,
}

pub struct S2A {
    pub config: S2AConfig,
    hidden: usize,
    freqs: Vec<f32>,
    freqs_rows: usize,
    freqs_half: usize,
    input_embedding: Vec<f32>,
    out_project: (Vec<f32>, Vec<f32>),
    encoder_proj: (Vec<f32>, Vec<f32>),
    content_in_proj: (Vec<f32>, Vec<f32>),
    length_convs: Vec<(Vec<f32>, Vec<f32>)>,
    length_norms: Vec<(Vec<f32>, Vec<f32>)>,
    length_final: (Vec<f32>, Vec<f32>),
    prompt_cond: Vec<f32>,
    t_embedders: [(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>); 2],
    input_proj: (Vec<f32>, Vec<f32>),
    mu_projection: (Vec<f32>, Vec<f32>),
    blocks: Vec<DiTBlock>,
    transformer_modulation: (Vec<f32>, Vec<f32>),
    transformer_norm: Vec<f32>,
    skip_linear: (Vec<f32>, Vec<f32>),
    conv1: (Vec<f32>, Vec<f32>),
    wavenet_cond: (Vec<f32>, Vec<f32>),
    wavenet_in: Vec<(Vec<f32>, Vec<f32>)>,
    wavenet_res_skip: Vec<(Vec<f32>, Vec<f32>)>,
    res_projection: (Vec<f32>, Vec<f32>),
    final_modulation: (Vec<f32>, Vec<f32>),
    final_linear: (Vec<f32>, Vec<f32>),
    conv2: (Vec<f32>, Vec<f32>),
}

fn lin2(
    file: &SafetensorsFile,
    name: &str,
    input: usize,
    output: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        load_tensor(file, &format!("{name}.weight"), &[output, input])?,
        load_tensor(file, &format!("{name}.bias"), &[output])?,
    ))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn mish(x: f32) -> f32 {
    x * (x.softplus()).tanh()
}

trait Softplus {
    fn softplus(self) -> Self;
}

impl Softplus for f32 {
    fn softplus(self) -> f32 {
        // tanh(softplus(x)) in the reference: log(1 + exp(x)).
        if self > 20.0 {
            self
        } else {
            (1.0 + self.exp()).ln()
        }
    }
}

impl S2A {
    pub fn open(path: &Path, config: S2AConfig) -> Result<Self> {
        let file = SafetensorsFile::open(path).map_err(crate::SpeechError::from)?;
        Self::load(&file, config)
    }

    pub fn load(file: &SafetensorsFile, config: S2AConfig) -> Result<Self> {
        let hid = config.hidden;
        let freqs_half = config.head_dim / 2;
        let freqs = load_tensor(
            file,
            "decoder.estimator.freqs_cis",
            &[config.freqs_positions, freqs_half, 2],
        )?;
        let mut length_convs = Vec::new();
        let mut length_norms = Vec::new();
        for (ci, gi) in [(0usize, 1usize), (3, 4), (6, 7), (9, 10)] {
            length_convs.push((
                load_tensor(
                    file,
                    &format!("length_regulator.model.{ci}.weight"),
                    &[hid, hid, 3],
                )?,
                load_tensor(file, &format!("length_regulator.model.{ci}.bias"), &[hid])?,
            ));
            length_norms.push((
                load_tensor(file, &format!("length_regulator.model.{gi}.weight"), &[hid])?,
                load_tensor(file, &format!("length_regulator.model.{gi}.bias"), &[hid])?,
            ));
        }
        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            let p = format!("decoder.estimator.transformer_blocks.{index}");
            let skip_in = if index > config.depth / 2 {
                Some(lin2(file, &format!("{p}.skip_in_linear"), 2 * hid, hid)?)
            } else {
                None
            };
            blocks.push(DiTBlock {
                attn_modulation: lin2(
                    file,
                    &format!("{p}.attention_norm.modulation"),
                    hid,
                    2 * hid,
                )?,
                attn_norm: load_tensor(file, &format!("{p}.attention_norm.norm.weight"), &[hid])?,
                ffn_modulation: lin2(file, &format!("{p}.ffn_norm.modulation"), hid, 2 * hid)?,
                ffn_norm: load_tensor(file, &format!("{p}.ffn_norm.norm.weight"), &[hid])?,
                wqkv: load_tensor(file, &format!("{p}.attention.wqkv.weight"), &[3 * hid, hid])?,
                wo: load_tensor(file, &format!("{p}.attention.wo.weight"), &[hid, hid])?,
                w1: load_tensor(
                    file,
                    &format!("{p}.feed_forward.w1.weight"),
                    &[4 * hid, hid],
                )?,
                w2: load_tensor(
                    file,
                    &format!("{p}.feed_forward.w2.weight"),
                    &[hid, 4 * hid],
                )?,
                w3: load_tensor(
                    file,
                    &format!("{p}.feed_forward.w3.weight"),
                    &[4 * hid, hid],
                )?,
                skip_in,
            });
        }
        let t_embed = |prefix: &str| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
            Ok((
                load_tensor(file, &format!("{prefix}.time_mlp.0.weight"), &[hid, 256])?,
                load_tensor(file, &format!("{prefix}.time_mlp.0.bias"), &[hid])?,
                load_tensor(file, &format!("{prefix}.time_mlp.2.weight"), &[hid, hid])?,
                load_tensor(file, &format!("{prefix}.time_mlp.2.bias"), &[hid])?,
            ))
        };
        Ok(S2A {
            config: config.clone(),
            hidden: hid,
            input_embedding: load_tensor(
                file,
                "input_embedding.embedding.weight",
                &[config.codebook_size, config.codebook_dim],
            )?,
            out_project: (
                load_tensor(
                    file,
                    "input_embedding.out_project.weight",
                    &[config.encoder_dim, config.codebook_dim, 1],
                )?,
                load_tensor(
                    file,
                    "input_embedding.out_project.bias",
                    &[config.encoder_dim],
                )?,
            ),
            encoder_proj: lin2(
                file,
                "encoder_proj",
                config.latent_dim + config.encoder_dim,
                config.encoder_dim,
            )?,
            content_in_proj: lin2(
                file,
                "length_regulator.content_in_proj",
                config.encoder_dim,
                hid,
            )?,
            length_convs,
            length_norms,
            length_final: (
                load_tensor(file, "length_regulator.model.12.weight", &[hid, hid, 1])?,
                load_tensor(file, "length_regulator.model.12.bias", &[hid])?,
            ),
            // prompt_cond saves broadcastable ([1, 1, hid] in the
            // fixture; the real checkpoint saves [hid]).
            prompt_cond: {
                let descriptor =
                    file.descriptor("prompt_cond")
                        .ok_or_else(|| crate::SpeechError::Tensor {
                            name: "prompt_cond".into(),
                            why: "tensor is missing".into(),
                        })?;
                let shape = descriptor.shape.clone();
                let mut data = load_tensor(file, "prompt_cond", &shape)?;
                data.truncate(hid);
                data
            },
            t_embedders: [
                t_embed("decoder.estimator.t_embedder")?,
                t_embed("decoder.estimator.t_embedder2")?,
            ],
            input_proj: lin2(
                file,
                "decoder.estimator.input_embed.proj",
                2 * config.mels + hid + config.speaker_dim,
                hid,
            )?,
            mu_projection: lin2(
                file,
                "decoder.estimator.input_embed.mu_projection",
                hid,
                hid,
            )?,
            blocks,
            transformer_modulation: lin2(
                file,
                "decoder.estimator.transformer_norm.modulation",
                hid,
                2 * hid,
            )?,
            transformer_norm: load_tensor(
                file,
                "decoder.estimator.transformer_norm.norm.weight",
                &[hid],
            )?,
            // The tail concat is [x_res (hidden), x (mels)].
            skip_linear: lin2(
                file,
                "decoder.estimator.skip_linear",
                hid + config.mels,
                hid,
            )?,
            conv1: (
                load_tensor(file, "decoder.estimator.conv1.weight", &[hid, hid, 1])?,
                load_tensor(file, "decoder.estimator.conv1.bias", &[hid])?,
            ),
            wavenet_cond: (
                load_tensor(
                    file,
                    "decoder.estimator.wavenet.cond_layer.conv.weight",
                    &[config.wavenet_layers * 2 * hid, hid, 1],
                )?,
                load_tensor(
                    file,
                    "decoder.estimator.wavenet.cond_layer.conv.bias",
                    &[config.wavenet_layers * 2 * hid],
                )?,
            ),
            wavenet_in: (0..config.wavenet_layers)
                .map(|i| {
                    Ok((
                        load_tensor(
                            file,
                            &format!("decoder.estimator.wavenet.in_layers.{i}.conv.weight"),
                            &[2 * hid, hid, 5],
                        )?,
                        load_tensor(
                            file,
                            &format!("decoder.estimator.wavenet.in_layers.{i}.conv.bias"),
                            &[2 * hid],
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            wavenet_res_skip: (0..config.wavenet_layers)
                .map(|i| {
                    let out = if i < config.wavenet_layers - 1 {
                        2 * hid
                    } else {
                        hid
                    };
                    Ok((
                        load_tensor(
                            file,
                            &format!("decoder.estimator.wavenet.res_skip_layers.{i}.conv.weight"),
                            &[out, hid, 1],
                        )?,
                        load_tensor(
                            file,
                            &format!("decoder.estimator.wavenet.res_skip_layers.{i}.conv.bias"),
                            &[out],
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            res_projection: lin2(file, "decoder.estimator.res_projection", hid, hid)?,
            final_modulation: lin2(
                file,
                "decoder.estimator.final_layer.adaLN_modulation.1",
                hid,
                2 * hid,
            )?,
            final_linear: lin2(file, "decoder.estimator.final_layer.linear", hid, hid)?,
            conv2: (
                load_tensor(
                    file,
                    "decoder.estimator.conv2.weight",
                    &[config.mels, hid, 1],
                )?,
                load_tensor(file, "decoder.estimator.conv2.bias", &[config.mels])?,
            ),
            freqs,
            freqs_rows: config.freqs_positions,
            freqs_half,
        })
    }

    fn t_embed(&self, t: f32, which: usize) -> Vec<f32> {
        let half = 128;
        let weights = &self.t_embedders[which];
        let mut emb = vec![0.0f32; 256];
        for j in 0..half {
            let freq = 1000.0f32 * (-10000.0f32.ln() * j as f32 / half as f32).exp();
            let angle = t * freq;
            emb[j] = angle.cos();
            emb[half + j] = angle.sin();
        }
        let hidden = self.hidden;
        let mut out = ops::linear(&emb, &weights.0, Some(&weights.1), 1, 256, hidden);
        for value in &mut out {
            *value *= sigmoid(*value);
        }
        ops::linear(&out, &weights.2, Some(&weights.3), 1, hidden, hidden)
    }

    fn angle(&self, position: usize, pair: usize) -> (f32, f32) {
        if position < self.freqs_rows {
            let base = (position * self.freqs_half + pair) * 2;
            (self.freqs[base], self.freqs[base + 1])
        } else {
            // Beyond the table the reference extends it with
            // cos(p * theta), sin(p * theta) where theta is recovered
            // from table row 1 (row 0 is the identity).
            let base = (self.freqs_half + pair) * 2;
            let theta = self.freqs[base + 1].atan2(self.freqs[base]);
            let angle = position as f32 * theta;
            (angle.cos(), angle.sin())
        }
    }

    /// Interleaved-pair rope over `[rows, heads, head_dim]` flat data.
    fn rope(&self, x: &mut [f32], rows: usize, heads: usize, head_dim: usize) {
        let half = head_dim / 2;
        for t in 0..rows {
            for h in 0..heads {
                let row = (t * heads + h) * head_dim;
                for p in 0..half {
                    let (cos, sin) = self.angle(t, p);
                    let even = x[row + 2 * p];
                    let odd = x[row + 2 * p + 1];
                    x[row + 2 * p] = even * cos - odd * sin;
                    x[row + 2 * p + 1] = odd * cos + even * sin;
                }
            }
        }
    }

    /// `build_mu`: codes `[T]` + the LM latent `[T, d]` -> `mu
    /// [T_ref + ceil(1.72 * T), hidden]` (prompt broadcast then the
    /// length-regulated target condition).
    pub fn build_mu(&self, codes: &[i32], latent: &[f32], t_ref: usize) -> Result<Vec<f32>> {
        let hid = self.hidden;
        let enc = self.config.encoder_dim;
        let d = self.config.latent_dim;
        let t = codes.len();
        // Code embeddings -> k1 conv to encoder_dim (out_project).
        let mut emb = Vec::with_capacity(t * self.config.codebook_dim);
        for &code in codes {
            let row = usize::try_from(code).map_err(|_| crate::SpeechError::Input {
                why: format!("negative semantic code {code}"),
            })?;
            if row >= self.config.codebook_size {
                return Err(crate::SpeechError::Input {
                    why: format!("semantic code {code} outside the embedding table"),
                });
            }
            emb.extend_from_slice(
                &self.input_embedding
                    [row * self.config.codebook_dim..(row + 1) * self.config.codebook_dim],
            );
        }
        // conv1d k1 over [T, codebook_dim]: weight [enc, 8] torch layout.
        let sem = {
            let mut out = vec![0.0f32; t * enc];
            for (oc, w_row) in self
                .out_project
                .0
                .chunks_exact(self.config.codebook_dim)
                .enumerate()
            {
                for tt in 0..t {
                    let mut acc = self.out_project.1[oc];
                    for (ic, w) in w_row.iter().enumerate() {
                        acc += w * emb[tt * self.config.codebook_dim + ic];
                    }
                    out[tt * enc + oc] = acc;
                }
            }
            out
        };
        // text_cond = linear([latent, sem]) -> [T, enc].
        let width = self.config.latent_dim + enc;
        let mut concat = vec![0.0f32; t * width];
        for tt in 0..t {
            concat[tt * width..tt * width + d].copy_from_slice(&latent[tt * d..(tt + 1) * d]);
            concat[tt * width + d..(tt + 1) * width]
                .copy_from_slice(&sem[tt * enc..(tt + 1) * enc]);
        }
        let text_cond = apply_linear(
            &concat,
            &self.encoder_proj.0,
            &self.encoder_proj.1,
            t,
            width,
            enc,
        );
        let mut x = apply_linear(
            &text_cond,
            &self.content_in_proj.0,
            &self.content_in_proj.1,
            t,
            enc,
            hid,
        );
        // Length regulator: nearest interpolation to int(1.72 * T).
        let out_len = (t as f32 * 1.72) as usize;
        let mut interpolated = vec![0.0f32; out_len * hid];
        for (i, slot) in interpolated.chunks_exact_mut(hid).enumerate() {
            let src = ((i as f32 * (t as f32 / out_len as f32)) as usize).min(t - 1);
            slot.copy_from_slice(&x[src * hid..(src + 1) * hid]);
        }
        x = interpolated;
        // Four [conv k3 pad1, GroupNorm(1, C), mish] rounds.
        for round in 0..4 {
            let (w, b) = &self.length_convs[round];
            x = conv_time(&x, w, Some(b), hid, hid, 3, 1);
            let (gw, gb) = &self.length_norms[round];
            group_norm1(&mut x, out_len, hid, gw, gb);
            for value in &mut x {
                *value = mish(*value);
            }
        }
        x = conv_time(
            &x,
            &self.length_final.0,
            Some(&self.length_final.1),
            hid,
            hid,
            1,
            1,
        );
        // mu = [prompt_cond broadcast to T_ref, cond_target].
        let mut mu = vec![0.0f32; (t_ref + out_len) * hid];
        for tt in 0..t_ref {
            mu[tt * hid..(tt + 1) * hid].copy_from_slice(&self.prompt_cond);
        }
        mu[t_ref * hid..].copy_from_slice(&x);
        Ok(mu)
    }

    /// The full conditional forward (`forward`): `x`, `cond` `[rows,
    /// mels]` channel-last; `mu` `[rows, hidden]`; `spks [speaker_dim]`;
    /// `t` scalar. Returns `[rows, mels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x_bct: &[f32],
        cond_bct: &[f32],
        mu: &[f32],
        t: f32,
        spks: &[f32],
        rows: usize,
    ) -> Vec<f32> {
        let hid = self.hidden;
        let mels = self.config.mels;
        let t1 = self.t_embed(t, 0);
        // input embed: [x, cond, mu_proj, spks] -> proj.
        let mu_proj = apply_linear(
            mu,
            &self.mu_projection.0,
            &self.mu_projection.1,
            rows,
            hid,
            hid,
        );
        let mut concat = vec![0.0f32; rows * (2 * mels + hid + spks.len())];
        let width = 2 * mels + hid + spks.len();
        for row in 0..rows {
            concat[row * width..row * width + mels]
                .copy_from_slice(&x_bct[row * mels..(row + 1) * mels]);
            concat[row * width + mels..row * width + 2 * mels]
                .copy_from_slice(&cond_bct[row * mels..(row + 1) * mels]);
            concat[row * width + 2 * mels..row * width + 2 * mels + hid]
                .copy_from_slice(&mu_proj[row * hid..(row + 1) * hid]);
            concat[row * width + 2 * mels + hid..(row + 1) * width].copy_from_slice(spks);
        }
        let mut h = apply_linear(
            &concat,
            &self.input_proj.0,
            &self.input_proj.1,
            rows,
            width,
            hid,
        );
        // U-Net: emit the first depth/2 blocks, receive in the tail.
        let mut stack: Vec<Vec<f32>> = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            let skip = if index > self.config.depth / 2 && !stack.is_empty() {
                stack.pop()
            } else {
                None
            };
            h = self.block_forward(block, &h, &t1, rows, skip.as_deref());
            if index < self.config.depth / 2 {
                stack.push(h.clone());
            }
        }
        // transformer_norm AdaLN + input skip.
        let modulated = apply_linear(
            &t1,
            &self.transformer_modulation.0,
            &self.transformer_modulation.1,
            1,
            hid,
            2 * hid,
        );
        let (w, b) = modulated.split_at(hid);
        let mut x_res = h.clone();
        {
            let mut normed = h.clone();
            self.rms_norm_rows(&mut normed, &self.transformer_norm, rows);
            for row in 0..rows {
                for col in 0..hid {
                    x_res[row * hid + col] = normed[row * hid + col] * w[col] + b[col];
                }
            }
        }
        // The tail concat is [x_res (hidden), x (mels)].
        let skip_width = hid + mels;
        let mut concat2 = vec![0.0f32; rows * skip_width];
        for row in 0..rows {
            concat2[row * skip_width..row * skip_width + hid]
                .copy_from_slice(&x_res[row * hid..(row + 1) * hid]);
            concat2[row * skip_width + hid..(row + 1) * skip_width]
                .copy_from_slice(&x_bct[row * mels..(row + 1) * mels]);
        }
        let x_res = apply_linear(
            &concat2,
            &self.skip_linear.0,
            &self.skip_linear.1,
            rows,
            skip_width,
            hid,
        );
        // conv1 -> wavenet -> res projection -> final layer -> conv2.
        let mut x_out = apply_linear(&x_res, &self.conv1.0, &self.conv1.1, rows, hid, hid);
        let t2 = self.t_embed(t, 1);
        x_out = self.wavenet(&x_out, &t2, rows);
        let res = apply_linear(
            &x_res,
            &self.res_projection.0,
            &self.res_projection.1,
            rows,
            hid,
            hid,
        );
        for (v, r) in x_out.iter_mut().zip(&res) {
            *v += r;
        }
        // final adaLN (LayerNorm without affine) + linear. The
        // reference modulates with `silu(c)`.
        let mut t1_gate = t1.to_vec();
        for value in &mut t1_gate {
            *value *= sigmoid(*value);
        }
        let fmod = apply_linear(
            &t1_gate,
            &self.final_modulation.0,
            &self.final_modulation.1,
            1,
            hid,
            2 * hid,
        );
        let (shift, scale) = fmod.split_at(hid);
        let mut final_in = x_out.clone();
        for row in 0..rows {
            let base = row * hid;
            let mut mean = 0.0f32;
            for value in &final_in[base..base + hid] {
                mean += value;
            }
            mean /= hid as f32;
            let mut var = 0.0f32;
            for value in &final_in[base..base + hid] {
                let d = value - mean;
                var += d * d;
            }
            let inv = 1.0 / (var / hid as f32 + 1e-6).sqrt();
            for col in 0..hid {
                final_in[base + col] =
                    (final_in[base + col] - mean) * inv * (1.0 + scale[col]) + shift[col];
            }
        }
        let mut out = apply_linear(
            &final_in,
            &self.final_linear.0,
            &self.final_linear.1,
            rows,
            hid,
            hid,
        );
        out = apply_linear(&out, &self.conv2.0, &self.conv2.1, rows, hid, mels);
        out
    }

    fn rms_norm_rows(&self, x: &mut [f32], w: &[f32], rows: usize) {
        let n = w.len();
        for row in 0..rows {
            let base = row * n;
            let mut mean_sq = 0.0f32;
            for value in &x[base..base + n] {
                mean_sq += value * value;
            }
            let inv = 1.0 / (mean_sq / n as f32 + 1e-5).sqrt();
            for (col, value) in x[base..base + n].iter_mut().enumerate() {
                *value *= inv * w[col];
            }
        }
    }

    fn attention(&self, block: &DiTBlock, h: &[f32], rows: usize) -> Vec<f32> {
        let hid = self.hidden;
        let heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let qkv = ops::linear(h, &block.wqkv, None, rows, hid, 3 * hid);
        // Row-major [rows, 3 * hid]: split each row into q | k | v.
        let mut q = vec![0.0f32; rows * hid];
        let mut k = vec![0.0f32; rows * hid];
        let mut v = vec![0.0f32; rows * hid];
        for row in 0..rows {
            let src = row * 3 * hid;
            q[row * hid..(row + 1) * hid].copy_from_slice(&qkv[src..src + hid]);
            k[row * hid..(row + 1) * hid].copy_from_slice(&qkv[src + hid..src + 2 * hid]);
            v[row * hid..(row + 1) * hid].copy_from_slice(&qkv[src + 2 * hid..src + 3 * hid]);
        }
        self.rope(&mut q, rows, heads, head_dim);
        self.rope(&mut k, rows, heads, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; rows * hid];
        for t in 0..rows {
            for head in 0..heads {
                let mut weighted = vec![0.0f32; head_dim];
                let mut scores = vec![0.0f32; rows];
                let mut max = f32::NEG_INFINITY;
                for r in 0..rows {
                    let mut dot = 0.0f32;
                    for e in 0..head_dim {
                        dot += q[(t * heads + head) * head_dim + e]
                            * k[(r * heads + head) * head_dim + e];
                    }
                    scores[r] = dot * scale;
                    if scores[r] > max {
                        max = scores[r];
                    }
                }
                let mut sum = 0.0f32;
                for score in &mut scores {
                    let weight = (*score - max).exp();
                    *score = weight;
                    sum += weight;
                }
                let inv = 1.0 / sum;
                for r in 0..rows {
                    let weight = scores[r] * inv;
                    for e in 0..head_dim {
                        weighted[e] += weight * v[(r * heads + head) * head_dim + e];
                    }
                }
                out[t * hid + head * head_dim..t * hid + (head + 1) * head_dim]
                    .copy_from_slice(&weighted);
            }
        }
        ops::linear(&out, &block.wo, None, rows, hid, hid)
    }

    fn adaln_rms(
        &self,
        h: &[f32],
        cond: &[f32],
        modulation: &(Vec<f32>, Vec<f32>),
        norm: &[f32],
        rows: usize,
    ) -> Vec<f32> {
        let hid = self.hidden;
        let modulated = apply_linear(cond, &modulation.0, &modulation.1, 1, hid, 2 * hid);
        let (w, b) = modulated.split_at(hid);
        let mut normed = h.to_vec();
        self.rms_norm_rows(&mut normed, norm, rows);
        for row in 0..rows {
            for col in 0..hid {
                normed[row * hid + col] = normed[row * hid + col] * w[col] + b[col];
            }
        }
        normed
    }

    fn block_forward(
        &self,
        block: &DiTBlock,
        h: &[f32],
        t1: &[f32],
        rows: usize,
        skip: Option<&[f32]>,
    ) -> Vec<f32> {
        let hid = self.hidden;
        let mut h = h.to_vec();
        if let (Some((w, b)), Some(skip)) = (&block.skip_in, skip) {
            let mut concat = vec![0.0f32; rows * 2 * hid];
            for row in 0..rows {
                concat[row * 2 * hid..row * 2 * hid + hid]
                    .copy_from_slice(&h[row * hid..(row + 1) * hid]);
                concat[row * 2 * hid + hid..(row + 1) * 2 * hid]
                    .copy_from_slice(&skip[row * hid..(row + 1) * hid]);
            }
            h = apply_linear(&concat, w, b, rows, 2 * hid, hid);
        }
        let conditioned = self.adaln_rms(&h, t1, &block.attn_modulation, &block.attn_norm, rows);
        let attended = self.attention(block, &conditioned, rows);
        for (value, add) in h.iter_mut().zip(&attended) {
            *value += add;
        }
        let conditioned_ff = self.adaln_rms(&h, t1, &block.ffn_modulation, &block.ffn_norm, rows);
        let mut gated = ops::linear(&conditioned_ff, &block.w1, None, rows, hid, hid * 4);
        let up = ops::linear(&conditioned_ff, &block.w3, None, rows, hid, hid * 4);
        for (g, u) in gated.iter_mut().zip(&up) {
            *g *= sigmoid(*g) * u;
        }
        let ff = ops::linear(&gated, &block.w2, None, rows, hid * 4, hid);
        for (value, add) in h.iter_mut().zip(&ff) {
            *value += add;
        }
        h
    }

    /// The gated WaveNet over `[rows, hidden]` with the timestep
    /// condition broadcast per layer (`_wavenet`).
    fn wavenet(&self, x: &[f32], g: &[f32], rows: usize) -> Vec<f32> {
        let hid = self.hidden;
        let n = self.config.wavenet_layers;
        // cond_layer k1 over the single-row condition: [1, hid] -> [n *
        // 2 * hid], sliced per layer.
        let gcond = apply_linear(
            g,
            &self.wavenet_cond.0,
            &self.wavenet_cond.1,
            1,
            hid,
            n * 2 * hid,
        );
        let mut x = x.to_vec();
        let mut out = vec![0.0f32; rows * hid];
        for i in 0..n {
            let gl = &gcond[i * 2 * hid..(i + 1) * 2 * hid];
            let mut xin = conv_time(
                &x,
                &self.wavenet_in[i].0,
                Some(&self.wavenet_in[i].1),
                hid,
                2 * hid,
                5,
                1,
            );
            for row in 0..rows {
                for col in 0..2 * hid {
                    xin[row * 2 * hid + col] += gl[col];
                }
            }
            // tanh(x[:hid]) * sigmoid(x[hid:]).
            let mut acts = vec![0.0f32; rows * hid];
            for row in 0..rows {
                for col in 0..hid {
                    acts[row * hid + col] =
                        xin[row * 2 * hid + col].tanh() * sigmoid(xin[row * 2 * hid + hid + col]);
                }
            }
            let out_ch = self.wavenet_res_skip[i].0.len() / hid;
            let rs = apply_linear(
                &acts,
                &self.wavenet_res_skip[i].0,
                &self.wavenet_res_skip[i].1,
                rows,
                hid,
                out_ch,
            );
            if i < n - 1 {
                for row in 0..rows {
                    for col in 0..hid {
                        x[row * hid + col] += rs[row * out_ch + col];
                        out[row * hid + col] += rs[row * out_ch + hid + col];
                    }
                }
            } else {
                for row in 0..rows {
                    for col in 0..hid {
                        out[row * hid + col] += rs[row * out_ch + col];
                    }
                }
            }
        }
        out
    }

    /// `solve_euler`: the CFG Euler ODE over `t_span`, with the prompt
    /// region of x re-zeroed every step.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_euler(
        &self,
        z: &[f32],
        prompt: &[f32],
        mu: &[f32],
        spks: &[f32],
        t_span: &[f32],
        cfg: f32,
        rows: usize,
        t_ref: usize,
    ) -> Vec<f32> {
        let mels = self.config.mels;
        // `z` is channel-major `[mels, rows]` like the reference
        // `(1, 80, T)` draw.
        let mut x = vec![0.0f32; rows * mels];
        for row in t_ref..rows {
            for (col, slot) in x[row * mels..(row + 1) * mels].iter_mut().enumerate() {
                *slot = z[col * rows + row];
            }
        }
        // cond: prompt mel padded with zeros to the total length.
        let mut prompt_x = vec![0.0f32; rows * mels];
        prompt_x[..t_ref * mels].copy_from_slice(&prompt[..t_ref * mels]);
        let zeros_mel = vec![0.0f32; rows * mels];
        let zeros_mu = vec![0.0f32; mu.len()];
        let zeros_spk = vec![0.0f32; spks.len()];
        let t_span = t_span.to_vec();
        let mut t = t_span[0];
        let mut dt = t_span[1] - t_span[0];
        for step in 1..t_span.len() {
            let d_cond = self.forward(&x, &prompt_x, mu, t, spks, rows);
            let d_uncond = self.forward(&x, &zeros_mel, &zeros_mu, t, &zeros_spk, rows);
            for ((x_value, cond), uncond) in x.iter_mut().zip(&d_cond).zip(&d_uncond) {
                *x_value += dt * ((1.0 + cfg) * cond - cfg * uncond);
            }
            t += dt;
            if step < t_span.len() - 1 {
                dt = t_span[step + 1] - t;
            }
            // Re-zero the prompt region of x every step.
            for row in 0..t_ref {
                for col in 0..mels {
                    x[row * mels + col] = 0.0;
                }
            }
        }
        x
    }
}

fn apply_linear(
    x: &[f32],
    w: &Vec<f32>,
    b: &Vec<f32>,
    rows: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    ops::linear(x, w, Some(b), rows, k, n)
}

/// Time-axis conv over `[frames, channels]` with kernel, stride 1,
/// zero padding. Weight torch layout `[out, in, k]`.
fn conv_time(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    _dilation: usize,
) -> Vec<f32> {
    let frames = x.len() / in_ch;
    let pad = (kernel - 1) / 2;
    let mut out = vec![0.0f32; frames * out_ch];
    for t in 0..frames {
        for oc in 0..out_ch {
            let mut acc = b.map_or(0.0, |b| b[oc]);
            for k in 0..kernel {
                let src = t as i64 + k as i64 - pad as i64;
                if src < 0 || src as usize >= frames {
                    continue;
                }
                for ic in 0..in_ch {
                    acc += x[src as usize * in_ch + ic] * w[(oc * in_ch + ic) * kernel + k];
                }
            }
            out[t * out_ch + oc] = acc;
        }
    }
    out
}

/// `GroupNorm(1, C)` over `[frames, channels]`: normalize over both
/// axes per sample, affine per channel.
fn group_norm1(x: &mut [f32], frames: usize, channels: usize, w: &[f32], b: &[f32]) {
    let n = frames * channels;
    let mut mean = 0.0f32;
    for value in x.iter() {
        mean += value;
    }
    mean /= n as f32;
    let mut var = 0.0f32;
    for value in x.iter() {
        let d = value - mean;
        var += d * d;
    }
    let inv = 1.0 / (var / n as f32 + 1e-5).sqrt();
    for (index, value) in x.iter_mut().enumerate() {
        let c = index % channels;
        *value = (*value - mean) * inv * w[c] + b[c];
    }
}
