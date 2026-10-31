//! T2S prefix encoders (`prefix.py`): the text projector (embedding ->
//! two-linears with SiLU -> learned positions) and the ECAPA-TDNN
//! speaker encoder over the w2v condition vector (reflect-padded 'same'
//! convs, Res2Net chains, squeeze-excite, attentive statistics
//! pooling). The prefix tensors live in `t2s_model.safetensors`.

use std::path::Path;

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::load_tensor;
use crate::{Result, SpeechError};

#[derive(Debug, Clone)]
pub struct PrefixConfig {
    pub text_vocab: usize,
    pub text_embed_dim: usize,
    pub hidden: usize,
    /// The w2v condition vector width (1024).
    pub cond_dim: usize,
    pub max_positions: usize,
}

impl Default for PrefixConfig {
    fn default() -> Self {
        PrefixConfig {
            text_vocab: 130,
            text_embed_dim: 4096,
            hidden: 1280,
            cond_dim: 1024,
            max_positions: 4096,
        }
    }
}

/// A 'same' 1D conv over `[frames, in_ch]` with reflect padding, torch
/// weight `[out, in, k]`.
#[allow(clippy::too_many_arguments)]
pub fn conv_same(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    frames: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
) -> Vec<f32> {
    let pad = dilation * (kernel - 1) / 2;
    let mut out = vec![0.0f32; frames * out_ch];
    for t in 0..frames {
        for oc in 0..out_ch {
            let mut acc = b.map_or(0.0, |b| b[oc]);
            for k in 0..kernel {
                // Reflect index: pad copies x[p - i] on the left and
                // x[T - 2 - i] on the right.
                let raw = t as i64 + (k as i64 * dilation as i64 - pad as i64);
                let src: usize = if raw < 0 {
                    (-raw) as usize
                } else if raw as usize >= frames {
                    // torch 'reflect' right edge: x[T-2-(raw-T)].
                    let over = raw as usize - frames;
                    if frames >= 2 {
                        (frames as i64 - 2 - over as i64).max(0) as usize
                    } else {
                        0usize
                    }
                } else {
                    raw as usize
                };
                for ic in 0..in_ch {
                    acc += x[src * in_ch + ic] * w[(oc * in_ch + ic) * kernel + k];
                }
            }
            out[t * out_ch + oc] = acc;
        }
    }
    out
}

fn relu(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = value.max(0.0);
    }
}

pub struct Prefix {
    pub config: PrefixConfig,
    text_embed: Vec<f32>,
    text_fc1: (Vec<f32>, Vec<f32>),
    text_fc2: (Vec<f32>, Vec<f32>),
    text_position: Vec<f32>,
    speaker: SpeakerEncoder,
}

/// The ECAPA-TDNN speaker branch (`speaker_encoder.*`).
struct SpeakerEncoder {
    blocks0: (Vec<f32>, Vec<f32>),
    blocks: Vec<SeRes2NetBlock>,
    mfa: (Vec<f32>, Vec<f32>),
    asp_tdnn: (Vec<f32>, Vec<f32>),
    asp_conv: (Vec<f32>, Vec<f32>),
    fc: (Vec<f32>, Vec<f32>),
    channels: usize,
    feat: usize,
}

struct SeRes2NetBlock {
    tdnn1: (Vec<f32>, Vec<f32>),
    res2net: Vec<(Vec<f32>, Vec<f32>)>,
    tdnn2: (Vec<f32>, Vec<f32>),
    se_conv1: (Vec<f32>, Vec<f32>),
    se_conv2: (Vec<f32>, Vec<f32>),
    channels: usize,
    scale: usize,
}

impl Prefix {
    pub fn open(path: &Path, config: PrefixConfig) -> Result<Self> {
        let file = SafetensorsFile::open(path).map_err(SpeechError::from)?;
        Self::load(&file, config)
    }

    pub fn load(file: &SafetensorsFile, config: PrefixConfig) -> Result<Self> {
        let channels = config.cond_dim;
        let speaker = SpeakerEncoder {
            blocks0: lin_pair(file, "speaker_encoder.blocks.0", channels, channels)?,
            blocks: (1..4)
                .map(|i| {
                    load_se_res2net(
                        file,
                        &format!("speaker_encoder.blocks.{i}"),
                        channels,
                        i + 1,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            mfa: lin_pair(file, "speaker_encoder.mfa", channels * 3, channels)?,
            asp_tdnn: lin_pair(file, "speaker_encoder.asp.tdnn", channels * 3, channels)?,
            // The reference reads `asp.conv.*` and `fc.*` without the
            // inner `.conv.` segment.
            asp_conv: lin_direct(file, "speaker_encoder.asp.conv", channels, 1)?,
            fc: lin_direct(file, "speaker_encoder.fc", channels * 2, config.hidden)?,
            channels,
            feat: channels,
        };
        Ok(Prefix {
            text_embed: load_tensor(
                file,
                "text_projector.embed.weight",
                &[config.text_vocab, config.text_embed_dim],
            )?,
            // The projector linears are plain [out, in] matrices.
            text_fc1: (
                load_tensor(
                    file,
                    "text_projector.text_projection_fc1.weight",
                    &[config.text_embed_dim, config.text_embed_dim],
                )?,
                load_tensor(
                    file,
                    "text_projector.text_projection_fc1.bias",
                    &[config.text_embed_dim],
                )?,
            ),
            text_fc2: (
                load_tensor(
                    file,
                    "text_projector.text_projection_fc2.weight",
                    &[config.hidden, config.text_embed_dim],
                )?,
                load_tensor(
                    file,
                    "text_projector.text_projection_fc2.bias",
                    &[config.hidden],
                )?,
            ),
            text_position: load_tensor(
                file,
                "text_position_embedding.embedding.weight",
                &[config.max_positions, config.hidden],
            )?,
            config,
            speaker,
        })
    }

    /// `text_emb`: ids -> `[T, hidden]` (`silu(fc1(embed))` -> fc2 ->
    /// learned positions).
    pub fn text_emb(&self, ids: &[i32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden;
        let width = self.config.text_embed_dim;
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for (position, &id) in ids.iter().enumerate() {
            let row = usize::try_from(id).map_err(|_| SpeechError::Input {
                why: format!("negative text token id {id}"),
            })?;
            if row >= self.config.text_vocab {
                return Err(SpeechError::Input {
                    why: format!("text id {id} outside vocab {}", self.config.text_vocab),
                });
            }
            let e = self.text_embed[row * width..(row + 1) * width].to_vec();
            let mut projected = crate::ops::linear(
                &e,
                &self.text_fc1.0,
                Some(&self.text_fc1.1),
                1,
                width,
                width,
            );
            // The reference wraps the FIRST projection in silu.
            for value in &mut projected {
                *value *= sigmoid(*value);
            }
            let mut projected = crate::ops::linear(
                &projected,
                &self.text_fc2.0,
                Some(&self.text_fc2.1),
                1,
                width,
                hidden,
            );
            for (col, value) in projected.iter_mut().enumerate() {
                *value += self.text_position[position * hidden + col];
            }
            out.extend_from_slice(&projected);
        }
        Ok(out)
    }

    /// `cond_emb`: the w2v condition vector `[frames, cond_dim]` ->
    /// `[1, hidden]`.
    pub fn cond_emb(&self, cond: &[f32], frames: usize) -> Vec<f32> {
        self.speaker.forward(cond, frames)
    }
}

fn lin_direct(
    file: &SafetensorsFile,
    name: &str,
    input: usize,
    output: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        load_tensor(file, &format!("{name}.weight"), &[output, input, 1])?,
        load_tensor(file, &format!("{name}.bias"), &[output])?,
    ))
}

fn lin_pair(
    file: &SafetensorsFile,
    name: &str,
    input: usize,
    output: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    Ok((
        load_tensor(file, &format!("{name}.conv.weight"), &[output, input, 1])?,
        load_tensor(file, &format!("{name}.conv.bias"), &[output])?,
    ))
}

fn load_se_res2net(
    file: &SafetensorsFile,
    prefix: &str,
    channels: usize,
    _dilation: usize,
) -> Result<SeRes2NetBlock> {
    let scale = 8;
    Ok(SeRes2NetBlock {
        tdnn1: lin_pair(file, &format!("{prefix}.tdnn1"), channels, channels)?,
        res2net: (0..scale - 1)
            .map(|i| {
                lin_pair(
                    file,
                    &format!("{prefix}.res2net_block.blocks.{i}"),
                    channels / scale,
                    channels / scale,
                )
            })
            .collect::<Result<Vec<_>>>()?,
        tdnn2: lin_pair(file, &format!("{prefix}.tdnn2"), channels, channels)?,
        // The SE convs read their weights without the inner .conv.
        // segment (reference `g(p + ".conv1.weight")`).
        se_conv1: lin_direct(
            file,
            &format!("{prefix}.se_block.conv1"),
            channels,
            channels,
        )?,
        se_conv2: lin_direct(
            file,
            &format!("{prefix}.se_block.conv2"),
            channels,
            channels,
        )?,
        channels,
        scale,
    })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl SpeakerEncoder {
    /// `[frames, cond]` -> `[1, hidden]`: block-0 TDNN, three
    /// SE-Res2Net blocks, MFA, attentive statistics pooling, fc.
    fn forward(&self, cond: &[f32], frames: usize) -> Vec<f32> {
        let pooled = self.attentive_pool(&self.mfa_stage(cond, frames), frames);
        crate::ops::linear(
            &pooled,
            &self.fc.0,
            Some(&self.fc.1),
            1,
            self.channels * 2,
            self.feat,
        )
    }

    fn mfa_stage(&self, cond: &[f32], frames: usize) -> Vec<f32> {
        let c = self.channels;
        let mut x = conv_same(
            cond,
            &self.blocks0.0,
            Some(&self.blocks0.1),
            frames,
            c,
            c,
            1,
            1,
        );
        relu(&mut x);
        let mut feats: Vec<Vec<f32>> = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            x = se_res2net_forward(block, &x, frames, index + 2);
            feats.push(x.clone());
        }
        // mx.concatenate(feats, axis=2): per-frame channel concat
        // [block1 | block2 | block3], not whole-tensor blocks.
        let mut concat = vec![0.0f32; frames * c * 3];
        for t in 0..frames {
            for (index, feat) in feats.iter().enumerate() {
                concat[t * c * 3 + index * c..t * c * 3 + (index + 1) * c]
                    .copy_from_slice(&feat[t * c..(t + 1) * c]);
            }
        }
        let mut x = conv_same(
            &concat,
            &self.mfa.0,
            Some(&self.mfa.1),
            frames,
            c * 3,
            c,
            1,
            1,
        );
        relu(&mut x);
        x
    }

    /// `_asp` + the fc input: attentive statistics pooling over time.
    fn attentive_pool(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let c = self.channels;
        // First-pass mean/std with uniform weights.
        let mut means = vec![0.0f32; c];
        for t in 0..frames {
            for d in 0..c {
                means[d] += x[t * c + d] / frames as f32;
            }
        }
        let mut stds = vec![0.0f32; c];
        for t in 0..frames {
            for d in 0..c {
                let diff = x[t * c + d] - means[d];
                stds[d] += diff * diff / frames as f32;
            }
        }
        for value in stds.iter_mut() {
            *value = value.max(1e-12).sqrt();
        }
        // Attention over the [x, mean, std] features.
        let mut att_in = vec![0.0f32; frames * c * 3];
        for t in 0..frames {
            for d in 0..c {
                att_in[t * c * 3 + d] = x[t * c + d];
                att_in[t * c * 3 + c + d] = means[d];
                att_in[t * c * 3 + 2 * c + d] = stds[d];
            }
        }
        let mut att = conv_same(
            &att_in,
            &self.asp_tdnn.0,
            Some(&self.asp_tdnn.1),
            frames,
            c * 3,
            c,
            1,
            1,
        );
        // `_tdnn` ends in relu; the reference then takes the tanh.
        relu(&mut att);
        for value in &mut att {
            *value = value.tanh();
        }
        let mut att = conv_same(
            &att,
            &self.asp_conv.0,
            Some(&self.asp_conv.1),
            frames,
            c,
            1,
            1,
            1,
        );
        // Softmax over time.
        let mut max = f32::NEG_INFINITY;
        for &value in &att[..frames] {
            max = max.max(value);
        }
        let mut sum = 0.0f32;
        for value in &mut att[..frames] {
            let w = (*value - max).exp();
            *value = w;
            sum += w;
        }
        for value in &mut att[..frames] {
            *value /= sum;
        }
        let mut mean = vec![0.0f32; c];
        let mut std = vec![0.0f32; c];
        for t in 0..frames {
            for d in 0..c {
                mean[d] += att[t] * x[t * c + d];
            }
        }
        for d in 0..c {
            let mut var = 0.0f32;
            for t in 0..frames {
                let diff = x[t * c + d] - mean[d];
                var += att[t] * diff * diff;
            }
            std[d] = var.max(1e-12).sqrt();
        }
        let mut stats = Vec::with_capacity(c * 2);
        stats.extend_from_slice(&mean);
        stats.extend_from_slice(&std);
        stats
    }
}

/// The Res2Net chunk chain over the post-tdnn1 state
/// (`_res2net`): chunk 0 passes through, chunk 1 runs the first conv
/// without the running add, chunks 2+ add the previous output before
/// their conv.
fn res2net_chain(block: &SeRes2NetBlock, h: &[f32], frames: usize, dilation: usize) -> Vec<f32> {
    let c = block.channels;
    let scale = block.scale;
    let sc = c / scale;
    let chunks: Vec<Vec<f32>> = (0..scale)
        .map(|i| {
            let mut out = vec![0.0f32; frames * sc];
            for t in 0..frames {
                out[t * sc..(t + 1) * sc].copy_from_slice(&h[t * c + i * sc..t * c + (i + 1) * sc]);
            }
            out
        })
        .collect();
    let mut outs: Vec<Vec<f32>> = Vec::with_capacity(scale);
    let mut prev: Option<Vec<f32>> = None;
    for i in 0..scale {
        let out = if i == 0 {
            chunks[0].clone()
        } else {
            let (w, b) = &block.res2net[i - 1];
            let mut input = chunks[i].clone();
            if i > 1 {
                if let Some(prev) = &prev {
                    for (v, p) in input.iter_mut().zip(prev) {
                        *v += p;
                    }
                }
            }
            let mut out = conv_same(&input, w, Some(b), frames, sc, sc, 1, dilation);
            relu(&mut out);
            out
        };
        prev = Some(out.clone());
        outs.push(out);
    }
    let mut out = vec![0.0f32; frames * c];
    for (i, o) in outs.iter().enumerate() {
        for t in 0..frames {
            out[t * c + i * sc..t * c + (i + 1) * sc].copy_from_slice(&o[t * sc..(t + 1) * sc]);
        }
    }
    out
}

fn se_res2net_forward(
    block: &SeRes2NetBlock,
    x: &[f32],
    frames: usize,
    dilation: usize,
) -> Vec<f32> {
    let c = block.channels;
    let mut h = conv_same(x, &block.tdnn1.0, Some(&block.tdnn1.1), frames, c, c, 1, 1);
    relu(&mut h);
    let h = res2net_chain(block, &h, frames, dilation);
    let mut h = conv_same(&h, &block.tdnn2.0, Some(&block.tdnn2.1), frames, c, c, 1, 1);
    // `_tdnn` ends in relu.
    relu(&mut h);
    // SE block over time-mean context.
    let mut context = vec![0.0f32; c];
    for t in 0..frames {
        for d in 0..c {
            context[d] += h[t * c + d] / frames as f32;
        }
    }
    let mut s = conv_same(
        &context,
        &block.se_conv1.0,
        Some(&block.se_conv1.1),
        1,
        c,
        c,
        1,
        1,
    );
    relu(&mut s);
    let s = conv_same(
        &s,
        &block.se_conv2.0,
        Some(&block.se_conv2.1),
        1,
        c,
        c,
        1,
        1,
    );
    for (d, value) in s.iter().enumerate() {
        let gate = sigmoid(*value);
        for t in 0..frames {
            h[t * c + d] *= gate;
        }
    }
    for (value, r) in h.iter_mut().zip(x) {
        *value += r;
    }
    h
}
