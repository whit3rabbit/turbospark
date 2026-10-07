//! DAC-style residual unit, encoder block, and decoder block shared by
//! the SNAC, Descript DAC, DACVAE, and Higgs codecs. Each family keeps
//! its own tensor names and loaders and builds these structs; the
//! forward math is the one the four copies agreed on.
//!
//! Activations are channel-major `[ch, seq]`. Weight norm is already
//! folded into the convs (see [`crate::codec::conv`]).

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::conv::{Conv1d, ConvTranspose1d};
use crate::codec::wnconv::{load_f32_shaped, snake1d};
use crate::Result;

/// Snake1d alpha stored `[1, 1, C]` (Descript, DACVAE, Higgs layout).
pub(crate) fn load_alpha(file: &SafetensorsFile, name: &str, channels: usize) -> Result<Vec<f32>> {
    load_f32_shaped(file, name, &[1, 1, channels])
}

/// Snake1d alpha stored `[1, C, 1]` (SNAC and Fish layout).
pub(crate) fn load_alpha_mid(
    file: &SafetensorsFile,
    name: &str,
    channels: usize,
) -> Result<Vec<f32>> {
    load_f32_shaped(file, name, &[1, channels, 1])
}

/// Adds `x` (cropped by `pad` frames at the start) to the conv output
/// `y [ch, out_seq]` in place and returns it. Frames of `x` past its end
/// contribute 0. The residual is the left operand of the add, as in the
/// reference, and the add keeps that operand order, so writing into `y`
/// instead of a fresh buffer does not change any output bit.
pub(crate) fn add_cropped_residual(
    x: &[f32],
    seq: usize,
    mut y: Vec<f32>,
    ch: usize,
    pad: usize,
) -> Vec<f32> {
    let out_seq = y.len() / ch;
    for c in 0..ch {
        for t in 0..out_seq {
            let residual = if t + pad < seq {
                x[c * seq + t + pad]
            } else {
                0.0
            };
            let sum = residual + y[c * out_seq + t];
            y[c * out_seq + t] = sum;
        }
    }
    y
}

/// ResidualUnit: Snake, (dilated) conv, Snake, 1x1 conv, then a
/// symmetric crop-add of the input (the reference crops the residual by
/// half the length difference; equal lengths are the norm).
#[derive(Debug, Clone)]
pub(crate) struct ResidualUnit {
    pub snake1: Vec<f32>,
    pub conv1: Conv1d,
    pub snake2: Vec<f32>,
    pub conv2: Conv1d,
    /// Channel count of the unit's input and output.
    pub ch: usize,
}

impl ResidualUnit {
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let ch = self.ch;
        let seq = x.len() / ch;
        let mut h = x.to_vec();
        snake1d(&mut h, &self.snake1, ch, seq);
        let mut h = self.conv1.forward(&h);
        let seq2 = h.len() / ch;
        snake1d(&mut h, &self.snake2, ch, seq2);
        let h = self.conv2.forward(&h);
        let out_seq = h.len() / ch;
        let pad = seq.saturating_sub(out_seq) / 2;
        add_cropped_residual(x, seq, h, ch, pad)
    }
}

/// Three residual units, a Snake, then the strided down conv.
#[derive(Debug, Clone)]
pub(crate) struct EncoderBlock {
    pub units: [ResidualUnit; 3],
    pub snake: Vec<f32>,
    pub down: Conv1d,
    /// Block input width (the units' channel count).
    pub ch: usize,
}

impl EncoderBlock {
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = self.units[0].forward(x);
        h = self.units[1].forward(&h);
        h = self.units[2].forward(&h);
        let seq = h.len() / self.ch;
        snake1d(&mut h, &self.snake, self.ch, seq);
        self.down.forward(&h)
    }
}

/// Snake, the strided up conv, then three residual units.
#[derive(Debug, Clone)]
pub(crate) struct DecoderBlock {
    pub snake: Vec<f32>,
    pub up: ConvTranspose1d,
    pub units: [ResidualUnit; 3],
}

impl DecoderBlock {
    pub(crate) fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.forward_with(x, |h, _| h)
    }

    /// `between` runs on the up-conv output before the residual units
    /// (SNAC's noise block, Higgs's odd-stride crop). It also receives the
    /// frame count of `x`.
    pub(crate) fn forward_with(
        &self,
        x: &[f32],
        between: impl FnOnce(Vec<f32>, usize) -> Vec<f32>,
    ) -> Vec<f32> {
        let seq = x.len() / self.up.in_ch;
        let mut h = x.to_vec();
        snake1d(&mut h, &self.snake, self.up.in_ch, seq);
        let h = between(self.up.forward(&h), seq);
        let h = self.units[0].forward(&h);
        let h = self.units[1].forward(&h);
        self.units[2].forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.3
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    // Verbatim copy of the old per-family crop-add (separate output buffer).
    fn old_crop_add(x: &[f32], seq: usize, y: &[f32], ch: usize, pad: usize) -> Vec<f32> {
        let out_seq = y.len() / ch;
        let mut out = vec![0.0f32; ch * out_seq];
        for c in 0..ch {
            for t in 0..out_seq {
                let residual = if t + pad < seq {
                    x[c * seq + t + pad]
                } else {
                    0.0
                };
                out[c * out_seq + t] = residual + y[c * out_seq + t];
            }
        }
        out
    }

    #[test]
    fn in_place_crop_add_matches_old_separate_buffer() {
        let ch = 3;
        // Equal length, shorter output (symmetric crop), and an output
        // longer than the input (residual past the end reads 0).
        for &(seq, out_seq) in &[(10usize, 10usize), (10, 6), (6, 9)] {
            let x = sample(ch * seq, 7);
            let y = sample(ch * out_seq, 11);
            let pad = seq.saturating_sub(out_seq) / 2;
            let want = old_crop_add(&x, seq, &y, ch, pad);
            let got = add_cropped_residual(&x, seq, y.clone(), ch, pad);
            assert_eq!(bits(&got), bits(&want), "seq {seq} out_seq {out_seq}");
            // Fish's causal path crops the tail: pad 0.
            let want = old_crop_add(&x, seq, &y, ch, 0);
            let got = add_cropped_residual(&x, seq, y.clone(), ch, 0);
            assert_eq!(
                bits(&got),
                bits(&want),
                "causal seq {seq} out_seq {out_seq}"
            );
        }
    }

    fn conv(in_ch: usize, out_ch: usize, k: usize, pad: usize, dil: usize, seed: u32) -> Conv1d {
        Conv1d {
            in_ch,
            out_ch,
            kernel: k,
            stride: 1,
            padding: pad,
            dilation: dil,
            groups: 1,
            weight: sample(out_ch * in_ch * k, seed),
            bias: Some(sample(out_ch, seed + 1)),
        }
    }

    // Verbatim copy of the old SNAC ResidualUnit::forward.
    fn old_unit_forward(u: &ResidualUnit, x: &[f32], ch: usize) -> Vec<f32> {
        let seq = x.len() / ch;
        let mut h = x.to_vec();
        snake1d(&mut h, &u.snake1, ch, seq);
        let h = u.conv1.forward(&h);
        let seq2 = h.len() / ch;
        let mut h = h;
        snake1d(&mut h, &u.snake2, ch, seq2);
        let h = u.conv2.forward(&h);
        let out_seq = h.len() / ch;
        let pad = seq.saturating_sub(out_seq) / 2;
        let mut out = vec![0.0f32; ch * out_seq];
        for c in 0..ch {
            for t in 0..out_seq {
                let residual = if t + pad < seq {
                    x[c * seq + t + pad]
                } else {
                    0.0
                };
                out[c * out_seq + t] = residual + h[c * out_seq + t];
            }
        }
        out
    }

    fn unit(ch: usize, dilation: usize, seed: u32) -> ResidualUnit {
        ResidualUnit {
            snake1: sample(ch, seed).iter().map(|a| a.abs() + 0.2).collect(),
            conv1: conv(ch, ch, 7, 3 * dilation, dilation, seed + 10),
            snake2: sample(ch, seed + 2).iter().map(|a| a.abs() + 0.2).collect(),
            conv2: conv(ch, ch, 1, 0, 1, seed + 20),
            ch,
        }
    }

    #[test]
    fn residual_unit_matches_old_forward() {
        let ch = 4;
        let x = sample(ch * 23, 99);
        let u = unit(ch, 3, 5);
        assert_eq!(bits(&u.forward(&x)), bits(&old_unit_forward(&u, &x, ch)));
    }

    #[test]
    fn blocks_match_old_composition() {
        let ch = 4;
        let x = sample(ch * 16, 3);
        let enc = EncoderBlock {
            units: [unit(ch, 1, 1), unit(ch, 3, 2), unit(ch, 9, 3)],
            snake: sample(ch, 4).iter().map(|a| a.abs() + 0.2).collect(),
            down: Conv1d {
                stride: 2,
                ..conv(ch, 8, 4, 1, 1, 40)
            },
            ch,
        };
        // Old SNAC/Descript EncoderBlock::forward.
        let mut want = old_unit_forward(&enc.units[0], &x, ch);
        want = old_unit_forward(&enc.units[1], &want, ch);
        want = old_unit_forward(&enc.units[2], &want, ch);
        let seq = want.len() / ch;
        snake1d(&mut want, &enc.snake, ch, seq);
        let want = enc.down.forward(&want);
        assert_eq!(bits(&enc.forward(&x)), bits(&want));

        let out_ch = 2;
        let dec = DecoderBlock {
            snake: sample(ch, 5).iter().map(|a| a.abs() + 0.2).collect(),
            up: ConvTranspose1d {
                in_ch: ch,
                out_ch,
                kernel: 4,
                stride: 2,
                padding: 1,
                output_padding: 0,
                groups: 1,
                weight: sample(ch * out_ch * 4, 50),
                bias: Some(sample(out_ch, 51)),
            },
            units: [unit(out_ch, 1, 6), unit(out_ch, 3, 7), unit(out_ch, 9, 8)],
        };
        // Old DecoderBlock::forward (Descript).
        let seq = x.len() / ch;
        let mut h = x.to_vec();
        snake1d(&mut h, &dec.snake, ch, seq);
        let mut h = dec.up.forward(&h);
        h = old_unit_forward(&dec.units[0], &h, out_ch);
        h = old_unit_forward(&dec.units[1], &h, out_ch);
        let want = old_unit_forward(&dec.units[2], &h, out_ch);
        assert_eq!(bits(&dec.forward(&x)), bits(&want));
    }
}
