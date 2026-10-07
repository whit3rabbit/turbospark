//! SANM (FSMN-augmented self-attention) encoder layer shared by SenseVoice
//! Small and Fun-ASR-Nano.
//!
//! Both families carry the same SANM block from FunASR: a fused q/k/v
//! projection, plain multi-head attention, and an FSMN memory (depthwise
//! convolution over the value plane) added back before the output
//! residual. They differ only in how the checkpoint stores the FSMN weight
//! (`[width, 1, kernel]` for SenseVoice, `[width, kernel, 1]` for
//! Fun-ASR-Nano, flat layout `channel * kernel + tap` either way) and in
//! which geometry the loader refuses, so both are parameters here.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::{load_tensor, LayerNorm, Linear};
use crate::ops;
use crate::Result;

/// How the FSMN depthwise weight is shaped in the checkpoint.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FsmnLayout {
    /// `[width, 1, kernel]`
    ChannelsOneKernel,
    /// `[width, kernel, 1]`
    ChannelsKernelOne,
}

/// The per-family SANM geometry and its loader checks.
#[derive(Clone, Copy)]
pub(crate) struct SanmConfig {
    pub(crate) output_size: usize,
    pub(crate) linear_units: usize,
    pub(crate) attention_heads: usize,
    pub(crate) fsmn_kernel: usize,
    pub(crate) sanm_shift: usize,
    pub(crate) layer_norm_eps: f32,
    pub(crate) fsmn_layout: FsmnLayout,
    /// Runs inside the attention load after the layer norms, as each
    /// family's own check always did: `(width, heads, kernel, left_padding)`.
    pub(crate) validate: fn(usize, usize, usize, usize) -> Result<()>,
}

struct SanmAttention {
    qkv: Linear,
    output: Linear,
    fsmn_weight: Vec<f32>,
    width: usize,
    heads: usize,
    kernel: usize,
    left_padding: usize,
}

impl SanmAttention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input: usize,
        config: &SanmConfig,
    ) -> Result<Self> {
        let width = config.output_size;
        let kernel = config.fsmn_kernel;
        let left_padding = (kernel - 1) / 2 + config.sanm_shift;
        (config.validate)(width, config.attention_heads, kernel, left_padding)?;
        let fsmn_shape = match config.fsmn_layout {
            FsmnLayout::ChannelsOneKernel => [width, 1, kernel],
            FsmnLayout::ChannelsKernelOne => [width, kernel, 1],
        };
        Ok(Self {
            qkv: Linear::load(
                file,
                &format!("{prefix}.linear_q_k_v"),
                input,
                3 * width,
                true,
            )?,
            output: Linear::load(file, &format!("{prefix}.linear_out"), width, width, true)?,
            fsmn_weight: load_tensor(file, &format!("{prefix}.fsmn_block.weight"), &fsmn_shape)?,
            width,
            heads: config.attention_heads,
            kernel,
            left_padding,
        })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let width = self.width;
        let head_width = width / self.heads;
        let qkv = self.qkv.forward(x, rows);
        let mut attended = vec![0.0f32; rows * width];
        let mut scores = vec![0.0f32; rows];
        let scale = (head_width as f32).sqrt().recip();

        for time in 0..rows {
            for head in 0..self.heads {
                for source in 0..rows {
                    let mut dot = 0.0f32;
                    for dim in 0..head_width {
                        let col = head * head_width + dim;
                        let q = qkv[time * 3 * width + col];
                        let k = qkv[source * 3 * width + width + col];
                        dot += q * k;
                    }
                    scores[source] = dot * scale;
                }
                ops::softmax_row(&mut scores);
                for dim in 0..head_width {
                    let col = head * head_width + dim;
                    let mut sum = 0.0f32;
                    for source in 0..rows {
                        let value = qkv[source * 3 * width + 2 * width + col];
                        sum += scores[source] * value;
                    }
                    attended[time * width + col] = sum;
                }
            }
        }
        let mut output = self.output.forward(&attended, rows);
        for time in 0..rows {
            for channel in 0..width {
                let mut memory = 0.0f32;
                for kernel_index in 0..self.kernel {
                    let source = time as isize + kernel_index as isize - self.left_padding as isize;
                    if source >= 0 && source < rows as isize {
                        let value = qkv[source as usize * 3 * width + 2 * width + channel];
                        memory += value * self.fsmn_weight[channel * self.kernel + kernel_index];
                    }
                }
                let value = qkv[time * 3 * width + 2 * width + channel];
                output[time * width + channel] += memory + value;
            }
        }
        output
    }
}

pub(crate) struct SanmEncoderLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: SanmAttention,
    ff1: Linear,
    ff2: Linear,
    input: usize,
    output: usize,
}

impl SanmEncoderLayer {
    pub(crate) fn load(
        file: &SafetensorsFile,
        prefix: &str,
        input: usize,
        config: &SanmConfig,
    ) -> Result<Self> {
        let output = config.output_size;
        let attention_prefix = format!("{prefix}.self_attn");
        Ok(Self {
            norm1: LayerNorm::load(
                file,
                &format!("{prefix}.norm1"),
                input,
                config.layer_norm_eps,
            )?,
            norm2: LayerNorm::load(
                file,
                &format!("{prefix}.norm2"),
                output,
                config.layer_norm_eps,
            )?,
            attention: SanmAttention::load(file, &attention_prefix, input, config)?,
            ff1: Linear::load(
                file,
                &format!("{prefix}.feed_forward.w_1"),
                output,
                config.linear_units,
                true,
            )?,
            ff2: Linear::load(
                file,
                &format!("{prefix}.feed_forward.w_2"),
                config.linear_units,
                output,
                true,
            )?,
            input,
            output,
        })
    }

    pub(crate) fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
        let mut normalized = input.to_vec();
        self.norm1.apply(&mut normalized, rows);
        let attended = self.attention.forward(&normalized, rows);
        let mut hidden = vec![0.0f32; rows * self.output];
        if self.input == self.output {
            for index in 0..hidden.len() {
                hidden[index] = input[index] + attended[index];
            }
        } else {
            hidden.copy_from_slice(&attended);
        }

        let residual = hidden.clone();
        self.norm2.apply(&mut hidden, rows);
        let mut feedforward = self.ff1.forward(&hidden, rows);
        for value in &mut feedforward {
            *value = value.max(0.0);
        }
        let mut feedforward = self.ff2.forward(&feedforward, rows);
        for (value, skip) in feedforward.iter_mut().zip(residual) {
            *value += skip;
        }
        feedforward
    }
}

/// Scales `x [rows, width]` by `sqrt(scale_width)` and adds the sinusoidal
/// position signal (positions start at 1).
pub(crate) fn add_sinusoidal_positions(
    x: &mut [f32],
    rows: usize,
    width: usize,
    scale_width: usize,
) {
    let scale = (scale_width as f32).sqrt();
    let half = width / 2;
    let log_increment = 10_000.0f32.ln() / (half - 1) as f32;
    for row in 0..rows {
        let position = (row + 1) as f32;
        for dim in 0..half {
            let inverse_timescale = (-log_increment * dim as f32).exp();
            let phase = position * inverse_timescale;
            x[row * width + dim] = x[row * width + dim] * scale + phase.sin();
            x[row * width + half + dim] = x[row * width + half + dim] * scale + phase.cos();
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(dead_code)]

    use super::*;
    use crate::{Result, SpeechError};

    // ---- retired SenseVoice implementation (verbatim, loaders dropped) ----
    mod sensevoice_ref {
        use crate::nn::{LayerNorm, Linear};
        use crate::ops;

        pub(super) struct SelfAttentionSanm {
            pub(super) qkv: Linear,
            pub(super) output: Linear,
            pub(super) fsmn_weight: Vec<f32>,
            pub(super) width: usize,
            pub(super) heads: usize,
            pub(super) kernel: usize,
            pub(super) left_padding: usize,
        }

        impl SelfAttentionSanm {
            pub(super) fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
                let width = self.width;
                let head_width = width / self.heads;
                let qkv = self.qkv.forward(x, rows);
                let mut attended = vec![0.0f32; rows * width];
                let mut scores = vec![0.0f32; rows];
                let scale = (head_width as f32).sqrt().recip();

                for time in 0..rows {
                    for head in 0..self.heads {
                        for source in 0..rows {
                            let mut dot = 0.0f32;
                            for dim in 0..head_width {
                                let col = head * head_width + dim;
                                let q = qkv[time * 3 * width + col];
                                let k = qkv[source * 3 * width + width + col];
                                dot += q * k;
                            }
                            scores[source] = dot * scale;
                        }
                        ops::softmax_row(&mut scores);
                        for dim in 0..head_width {
                            let col = head * head_width + dim;
                            let mut sum = 0.0f32;
                            for source in 0..rows {
                                let value = qkv[source * 3 * width + 2 * width + col];
                                sum += scores[source] * value;
                            }
                            attended[time * width + col] = sum;
                        }
                    }
                }
                let mut output = self.output.forward(&attended, rows);
                for time in 0..rows {
                    for channel in 0..width {
                        let mut memory = 0.0f32;
                        for kernel_index in 0..self.kernel {
                            let source =
                                time as isize + kernel_index as isize - self.left_padding as isize;
                            if source >= 0 && source < rows as isize {
                                let value = qkv[source as usize * 3 * width + 2 * width + channel];
                                memory +=
                                    value * self.fsmn_weight[channel * self.kernel + kernel_index];
                            }
                        }
                        let value = qkv[time * 3 * width + 2 * width + channel];
                        output[time * width + channel] += memory + value;
                    }
                }
                output
            }
        }

        pub(super) struct EncoderLayer {
            pub(super) norm1: LayerNorm,
            pub(super) norm2: LayerNorm,
            pub(super) attention: SelfAttentionSanm,
            pub(super) ff1: Linear,
            pub(super) ff2: Linear,
            pub(super) input: usize,
            pub(super) output: usize,
        }

        impl EncoderLayer {
            pub(super) fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
                let mut normalized = input.to_vec();
                self.norm1.apply(&mut normalized, rows);
                let attended = self.attention.forward(&normalized, rows);
                let mut hidden = vec![0.0f32; rows * self.output];
                if self.input == self.output {
                    for index in 0..hidden.len() {
                        hidden[index] = input[index] + attended[index];
                    }
                } else {
                    hidden.copy_from_slice(&attended);
                }

                let residual = hidden.clone();
                self.norm2.apply(&mut hidden, rows);
                let mut feedforward = self.ff1.forward(&hidden, rows);
                for value in &mut feedforward {
                    *value = value.max(0.0);
                }
                let mut feedforward = self.ff2.forward(&feedforward, rows);
                for (value, skip) in feedforward.iter_mut().zip(residual) {
                    *value += skip;
                }
                feedforward
            }
        }

        pub(super) fn add_sinusoidal_positions(
            x: &mut [f32],
            rows: usize,
            width: usize,
            scale_width: usize,
        ) {
            let scale = (scale_width as f32).sqrt();
            let half = width / 2;
            let log_increment = 10_000.0f32.ln() / (half - 1) as f32;
            for row in 0..rows {
                let position = (row + 1) as f32;
                for dim in 0..half {
                    let inverse_timescale = (-log_increment * dim as f32).exp();
                    let phase = position * inverse_timescale;
                    x[row * width + dim] = x[row * width + dim] * scale + phase.sin();
                    x[row * width + half + dim] = x[row * width + half + dim] * scale + phase.cos();
                }
            }
        }
    }

    // ---- retired Fun-ASR-Nano implementation (verbatim, loaders dropped) ----
    mod funasr_ref {
        use crate::nn::{LayerNorm, Linear};
        use crate::ops;

        pub(super) fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
            debug_assert_eq!(left.len(), right.len());
            left.iter().zip(right).map(|(&a, &b)| a + b).collect()
        }

        pub(super) struct SanmAttention {
            pub(super) qkv: Linear,
            pub(super) output: Linear,
            pub(super) fsmn_weight: Vec<f32>,
            pub(super) width: usize,
            pub(super) heads: usize,
            pub(super) kernel: usize,
            pub(super) left_padding: usize,
        }

        impl SanmAttention {
            pub(super) fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
                let width = self.width;
                let head_width = width / self.heads;
                let qkv = self.qkv.forward(x, rows);
                let mut attended = vec![0.0f32; rows * width];
                let mut scores = vec![0.0f32; rows];
                let scale = (head_width as f32).sqrt().recip();
                for time in 0..rows {
                    for head in 0..self.heads {
                        for source in 0..rows {
                            let mut dot = 0.0f32;
                            for dim in 0..head_width {
                                let col = head * head_width + dim;
                                dot += qkv[time * 3 * width + col]
                                    * qkv[source * 3 * width + width + col];
                            }
                            scores[source] = dot * scale;
                        }
                        ops::softmax_row(&mut scores);
                        for dim in 0..head_width {
                            let col = head * head_width + dim;
                            let mut sum = 0.0f32;
                            for source in 0..rows {
                                sum += scores[source] * qkv[source * 3 * width + 2 * width + col];
                            }
                            attended[time * width + col] = sum;
                        }
                    }
                }
                let mut output = self.output.forward(&attended, rows);
                for time in 0..rows {
                    for channel in 0..width {
                        let mut memory = 0.0f32;
                        for kernel_index in 0..self.kernel {
                            let source =
                                time as isize + kernel_index as isize - self.left_padding as isize;
                            if source >= 0 && source < rows as isize {
                                let value = qkv[source as usize * 3 * width + 2 * width + channel];
                                memory +=
                                    value * self.fsmn_weight[channel * self.kernel + kernel_index];
                            }
                        }
                        let value = qkv[time * 3 * width + 2 * width + channel];
                        output[time * width + channel] += memory + value;
                    }
                }
                output
            }
        }

        pub(super) struct EncoderLayer {
            pub(super) norm1: LayerNorm,
            pub(super) norm2: LayerNorm,
            pub(super) attention: SanmAttention,
            pub(super) ff1: Linear,
            pub(super) ff2: Linear,
            pub(super) input: usize,
            pub(super) output: usize,
        }

        impl EncoderLayer {
            pub(super) fn forward(&self, input: &[f32], rows: usize) -> Vec<f32> {
                let mut normalized = input.to_vec();
                self.norm1.apply(&mut normalized, rows);
                let attended = self.attention.forward(&normalized, rows);
                let mut hidden = if self.input == self.output {
                    add(input, &attended)
                } else {
                    attended
                };
                let residual = hidden.clone();
                self.norm2.apply(&mut hidden, rows);
                let mut feedforward = self.ff1.forward(&hidden, rows);
                for value in &mut feedforward {
                    *value = value.max(0.0);
                }
                add(&residual, &self.ff2.forward(&feedforward, rows))
            }
        }

        pub(super) fn add_sinusoidal_position(
            values: &mut [f32],
            rows: usize,
            width: usize,
            scale_width: usize,
        ) {
            let half = width / 2;
            let log_timescale_increment = 10_000.0f32.ln() / (half - 1) as f32;
            let scale = (scale_width as f32).sqrt();
            for row in 0..rows {
                for dim in 0..half {
                    let inverse_timescale = (-(dim as f32) * log_timescale_increment).exp();
                    let scaled_time = (row + 1) as f32 * inverse_timescale;
                    let sin_index = row * width + dim;
                    let cos_index = row * width + half + dim;
                    values[sin_index] = values[sin_index] * scale + scaled_time.sin();
                    values[cos_index] = values[cos_index] * scale + scaled_time.cos();
                }
            }
        }
    }

    struct Rng(u32);

    impl Rng {
        fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len)
                .map(|_| {
                    self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                })
                .collect()
        }

        fn linear(&mut self, input: usize, output: usize) -> Linear {
            Linear::new(
                self.vec(input * output, 0.5),
                Some(self.vec(output, 0.2)),
                input,
                output,
            )
        }

        fn norm(&mut self, width: usize) -> LayerNorm {
            LayerNorm::new(
                self.vec(width, 1.0).iter().map(|v| v + 1.0).collect(),
                Some(self.vec(width, 0.2)),
                1e-5,
            )
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// The same random layer in the shared type and in both retired ones.
    struct Fixture {
        shared: SanmEncoderLayer,
        old_sensevoice: sensevoice_ref::EncoderLayer,
        old_funasr: funasr_ref::EncoderLayer,
    }

    fn fixture(seed: u32, input: usize, output: usize, kernel: usize, shift: usize) -> Fixture {
        let heads = 4;
        let units = 20;
        let mut rng = Rng(seed);
        let norm1 = rng.norm(input);
        let norm2 = rng.norm(output);
        let qkv = rng.linear(input, 3 * output);
        let out = rng.linear(output, output);
        let fsmn = rng.vec(output * kernel, 0.7);
        let ff1 = rng.linear(output, units);
        let ff2 = rng.linear(units, output);
        let left_padding = (kernel - 1) / 2 + shift;
        Fixture {
            shared: SanmEncoderLayer {
                norm1: norm1.clone(),
                norm2: norm2.clone(),
                attention: SanmAttention {
                    qkv: qkv.clone(),
                    output: out.clone(),
                    fsmn_weight: fsmn.clone(),
                    width: output,
                    heads,
                    kernel,
                    left_padding,
                },
                ff1: ff1.clone(),
                ff2: ff2.clone(),
                input,
                output,
            },
            old_sensevoice: sensevoice_ref::EncoderLayer {
                norm1: norm1.clone(),
                norm2: norm2.clone(),
                attention: sensevoice_ref::SelfAttentionSanm {
                    qkv: qkv.clone(),
                    output: out.clone(),
                    fsmn_weight: fsmn.clone(),
                    width: output,
                    heads,
                    kernel,
                    left_padding,
                },
                ff1: ff1.clone(),
                ff2: ff2.clone(),
                input,
                output,
            },
            old_funasr: funasr_ref::EncoderLayer {
                norm1,
                norm2,
                attention: funasr_ref::SanmAttention {
                    qkv,
                    output: out,
                    fsmn_weight: fsmn,
                    width: output,
                    heads,
                    kernel,
                    left_padding,
                },
                ff1,
                ff2,
                input,
                output,
            },
        }
    }

    #[test]
    fn shared_layer_matches_both_retired_layers_bitwise() {
        // (input, output, kernel, shift): the first block widens, later
        // blocks keep the width; odd kernels and a shifted window both occur.
        for (seed, input, output, kernel, shift) in [
            (1, 12, 16, 5, 0),
            (2, 16, 16, 5, 0),
            (3, 16, 16, 4, 1),
            (4, 16, 16, 11, 0),
            (5, 8, 16, 3, 2),
        ] {
            let f = fixture(seed, input, output, kernel, shift);
            for rows in [1usize, 2, 9, 23] {
                let mut rng = Rng(seed * 100 + rows as u32);
                let x = rng.vec(rows * input, 2.0);
                let got = f.shared.forward(&x, rows);
                let sensevoice = f.old_sensevoice.forward(&x, rows);
                let funasr = f.old_funasr.forward(&x, rows);
                assert_eq!(bits(&got), bits(&sensevoice), "sensevoice rows {rows}");
                assert_eq!(bits(&got), bits(&funasr), "funasr rows {rows}");
            }
        }
    }

    #[test]
    fn sinusoidal_positions_match_both_retired_copies_bitwise() {
        for (rows, width, scale_width) in [(1usize, 8usize, 16usize), (7, 560, 512), (40, 512, 512)]
        {
            let mut rng = Rng(9);
            let base = rng.vec(rows * width, 3.0);
            let mut got = base.clone();
            add_sinusoidal_positions(&mut got, rows, width, scale_width);
            let mut sensevoice = base.clone();
            sensevoice_ref::add_sinusoidal_positions(&mut sensevoice, rows, width, scale_width);
            let mut funasr = base.clone();
            funasr_ref::add_sinusoidal_position(&mut funasr, rows, width, scale_width);
            assert_eq!(bits(&got), bits(&sensevoice));
            assert_eq!(bits(&got), bits(&funasr));
        }
    }

    fn reject(_: usize, _: usize, _: usize, _: usize) -> Result<()> {
        Err(SpeechError::BadConfig {
            field: "sanm_shift".into(),
            why: "padding exceeds the FSMN kernel".into(),
        })
    }

    fn safetensors_file(tag: &str, tensors: &[(&str, Vec<usize>)]) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape) in tensors {
            let count: usize = shape.iter().product();
            let start = data.len();
            data.resize(data.len() + count * 4, 0);
            header.insert(
                (*name).to_string(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let header = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!(
            "turbospark_sanm_{tag}_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, out).unwrap();
        path
    }

    fn accept(_: usize, _: usize, _: usize, _: usize) -> Result<()> {
        Ok(())
    }

    #[test]
    fn fsmn_layout_selects_the_checkpoint_shape() {
        let path = safetensors_file(
            "layout",
            &[
                ("a.linear_q_k_v.weight", vec![48, 16]),
                ("a.linear_q_k_v.bias", vec![48]),
                ("a.linear_out.weight", vec![16, 16]),
                ("a.linear_out.bias", vec![16]),
                ("a.fsmn_block.weight", vec![16, 1, 5]),
            ],
        );
        let file = SafetensorsFile::open(&path).unwrap();
        let mut config = SanmConfig {
            output_size: 16,
            linear_units: 20,
            attention_heads: 4,
            fsmn_kernel: 5,
            sanm_shift: 0,
            layer_norm_eps: 1e-5,
            fsmn_layout: FsmnLayout::ChannelsOneKernel,
            validate: accept,
        };
        let layer = SanmAttention::load(&file, "a", 16, &config).unwrap();
        assert_eq!(layer.fsmn_weight.len(), 16 * 5);
        assert_eq!(layer.left_padding, 2);
        config.fsmn_layout = FsmnLayout::ChannelsKernelOne;
        let error = SanmAttention::load(&file, "a", 16, &config).err().unwrap();
        assert!(
            error.to_string().contains("expected shape [16, 5, 1]"),
            "{error}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_family_check_runs_before_any_tensor_is_read() {
        // A missing file path would fail on the first tensor; the family's
        // own check must win, with its own text.
        let path = std::env::temp_dir().join(format!(
            "turbospark_sanm_{}.safetensors",
            std::process::id()
        ));
        let header = br#"{}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        std::fs::write(&path, bytes).unwrap();
        let file = SafetensorsFile::open(&path).unwrap();
        let config = SanmConfig {
            output_size: 16,
            linear_units: 20,
            attention_heads: 4,
            fsmn_kernel: 5,
            sanm_shift: 0,
            layer_norm_eps: 1e-5,
            fsmn_layout: FsmnLayout::ChannelsOneKernel,
            validate: reject,
        };
        let error = SanmAttention::load(&file, "a", 16, &config).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("padding exceeds the FSMN kernel"),
            "{error}"
        );
        let _ = std::fs::remove_file(path);
    }
}
