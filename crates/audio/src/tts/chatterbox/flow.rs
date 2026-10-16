//! The chatterbox S3Gen flow: speech tokens + speaker prompt -> mel.
//!
//! Reference: `chatterbox/s3gen/flow.py`
//! (`CausalMaskedDiffWithXvec`), `s3gen/flow_matching.py`
//! (`CausalConditionalCFM`), `s3gen/decoder.py`
//! (`ConditionalDecoder` and its causal blocks), and
//! `s3gen/matcha/` (`SinusoidalPosEmb`, `TimestepEmbedding`,
//! `ResnetBlock1D`, `Downsample1D`, `Upsample1D`,
//! `BasicTransformerBlock`), all at the pinned commit recorded in the
//! parent module.
//!
//! Structure: tokens embed (vocab clipped), the conformer encoder
//! ([`super::conformer`]) doubles the frame rate, a linear projects to
//! 80 mel dims, and a matcha U-Net estimator (causal resnet blocks,
//! self-attention transformer blocks, one down stage, mid stack, one up
//! stage) runs a 10-step Euler conditional flow match over the cosine
//! schedule `1 - cos(t*pi/2)` with classifier-free guidance rate 0.7
//! (the uncond half zeroes mu, the speaker embedding, and the
//! conditioning mel). The output slice drops the prompt frames.
//!
//! Batch-of-one with exact lengths: every padding/chunk mask in the
//! reference call is all-valid, so the mask multiplies reduce to
//! identity and the decoder attention runs unmasked (the reference
//! builds a zero additive bias for the non-streaming path, which is
//! exactly no mask). The reference construction hardwires one decoder
//! width (`channels = [256]` at the reference geometry), so there is
//! exactly one down block and one up block; only the mid stack repeats.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::ops;
use crate::{Result, SpeechError};

use super::conformer::{pad_left_cm, UpsampleConformerEncoder};
use super::s3gen::{mish, to_cm, to_rows, SaConv1d, SaLayerNorm, SaLinear};

/// Geometry of the flow model, one-to-one with the reference keyword
/// defaults (`S3Token2Mel.__init__` and the estimator it builds).
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Token embedding width and encoder width.
    pub input_size: usize,
    /// Mel bins (fixed by the vocoder at 80).
    pub output_size: usize,
    /// CAMPPlus speaker embedding width feeding `spk_embed_affine_layer`.
    pub spk_embed_dim: usize,
    /// Speech token vocabulary (6561 for the 25 Hz S3 FSQ tokenizer).
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub linear_units: usize,
    pub num_blocks: usize,
    pub num_up_blocks: usize,
    /// Mel frames per token: the encoder upsample factor.
    pub upsample_stride: usize,
    pub pre_lookahead_len: usize,
    /// Euler steps of the flow match.
    pub n_timesteps: usize,
    /// Classifier-free guidance rate of the Euler solver.
    pub inference_cfg_rate: f32,
    /// Estimator (matcha U-Net) width; the reference builds one
    /// down/up block each at this width.
    pub decoder_channels: usize,
    pub decoder_n_blocks: usize,
    pub decoder_num_mid_blocks: usize,
    pub decoder_num_heads: usize,
    pub decoder_head_dim: usize,
}

impl Default for FlowConfig {
    fn default() -> Self {
        FlowConfig {
            input_size: 512,
            output_size: 80,
            spk_embed_dim: 192,
            vocab_size: 6561,
            attention_heads: 8,
            linear_units: 2048,
            num_blocks: 6,
            num_up_blocks: 4,
            upsample_stride: 2,
            pre_lookahead_len: 3,
            n_timesteps: 10,
            inference_cfg_rate: 0.7,
            decoder_channels: 256,
            decoder_n_blocks: 4,
            decoder_num_mid_blocks: 12,
            decoder_num_heads: 8,
            decoder_head_dim: 64,
        }
    }
}

impl FlowConfig {
    /// Parses a JSON object over the reference defaults; missing or null
    /// fields keep the default.
    pub fn from_json_with_defaults(
        value: &serde_json::Value,
        defaults: FlowConfig,
    ) -> Result<Self> {
        let int = |name: &str, fallback: usize| -> Result<usize> {
            match value.get(name) {
                None | Some(serde_json::Value::Null) => Ok(fallback),
                Some(v) => v
                    .as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: name.to_string(),
                        why: "expected an integer".to_string(),
                    }),
            }
        };
        let float = |name: &str, fallback: f32| -> Result<f32> {
            match value.get(name) {
                None | Some(serde_json::Value::Null) => Ok(fallback),
                Some(v) => v
                    .as_f64()
                    .map(|n| n as f32)
                    .ok_or_else(|| SpeechError::BadConfig {
                        field: name.to_string(),
                        why: "expected a number".to_string(),
                    }),
            }
        };
        Ok(FlowConfig {
            input_size: int("input_size", defaults.input_size)?,
            output_size: int("output_size", defaults.output_size)?,
            spk_embed_dim: int("spk_embed_dim", defaults.spk_embed_dim)?,
            vocab_size: int("vocab_size", defaults.vocab_size)?,
            attention_heads: int("attention_heads", defaults.attention_heads)?,
            linear_units: int("linear_units", defaults.linear_units)?,
            num_blocks: int("num_blocks", defaults.num_blocks)?,
            num_up_blocks: int("num_up_blocks", defaults.num_up_blocks)?,
            upsample_stride: int("upsample_stride", defaults.upsample_stride)?,
            pre_lookahead_len: int("pre_lookahead_len", defaults.pre_lookahead_len)?,
            n_timesteps: int("n_timesteps", defaults.n_timesteps)?,
            inference_cfg_rate: float("inference_cfg_rate", defaults.inference_cfg_rate)?,
            decoder_channels: int("decoder_channels", defaults.decoder_channels)?,
            decoder_n_blocks: int("decoder_n_blocks", defaults.decoder_n_blocks)?,
            decoder_num_mid_blocks: int("decoder_num_mid_blocks", defaults.decoder_num_mid_blocks)?,
            decoder_num_heads: int("decoder_num_heads", defaults.decoder_num_heads)?,
            decoder_head_dim: int("decoder_head_dim", defaults.decoder_head_dim)?,
        })
    }
}

/// `SinusoidalPosEmb` for the timestep (`matcha/decoder.py`): half
/// exponential frequencies, `scale * t * freq`, `[sin, cos]` halves.
fn sinusoidal_pos_emb(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let step = 10_000f32.ln() / (half - 1) as f32;
    let mut out = vec![0.0f32; dim];
    for k in 0..half {
        let arg = 1000.0 * t * (-(k as f32) * step).exp();
        out[k] = arg.sin();
        out[half + k] = arg.cos();
    }
    out
}

/// `TimestepEmbedding`: linear -> SiLU -> linear.
#[derive(Debug, Clone)]
struct TimeMlp {
    linear_1: SaLinear,
    linear_2: SaLinear,
}

impl TimeMlp {
    fn load(file: &SafetensorsFile, prefix: &str, in_dim: usize, out_dim: usize) -> Result<Self> {
        Ok(TimeMlp {
            linear_1: SaLinear::load(file, &format!("{prefix}.linear_1"), in_dim, out_dim)?,
            linear_2: SaLinear::load(file, &format!("{prefix}.linear_2"), out_dim, out_dim)?,
        })
    }

    fn forward(&self, t: f32) -> Vec<f32> {
        let mut h = self
            .linear_1
            .forward(&sinusoidal_pos_emb(t, self.linear_1.in_dim), 1);
        ops::silu(&mut h);
        self.linear_2.forward(&h, 1)
    }
}

/// `CausalBlock1D`: causal conv k3 -> LayerNorm -> Mish. Channel-major
/// in and out; the LayerNorm runs over channels (the reference
/// transposes to `[B, T, C]`).
#[derive(Debug, Clone)]
struct CausalBlock1d {
    channels: usize,
    conv: SaConv1d,
    norm: SaLayerNorm,
}

impl CausalBlock1d {
    fn load(file: &SafetensorsFile, prefix: &str, in_ch: usize, out_ch: usize) -> Result<Self> {
        Ok(CausalBlock1d {
            channels: out_ch,
            conv: SaConv1d::load(
                file,
                &format!("{prefix}.conv.conv"),
                in_ch,
                out_ch,
                3,
                1,
                0,
                1,
                1,
            )?,
            norm: SaLayerNorm::load(file, &format!("{prefix}.norm"), out_ch)?,
        })
    }

    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let in_ch = x.len() / frames.max(1);
        let padded = pad_left_cm(x, in_ch, frames, 2);
        let h = self.conv.forward(&padded);
        let mut rows = to_rows(&h, self.channels, frames);
        self.norm.forward(&mut rows, frames, 1e-5);
        mish(&mut rows);
        to_cm(&rows, frames, self.channels)
    }
}

/// `CausalResnetBlock1D`: block1 -> time MLP (Mish first) -> block2 +
/// residual conv.
#[derive(Debug, Clone)]
struct CausalResnetBlock {
    block1: CausalBlock1d,
    block2: CausalBlock1d,
    mlp_linear: SaLinear,
    res_conv: SaConv1d,
    in_ch: usize,
    out_ch: usize,
}

impl CausalResnetBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        time_emb_dim: usize,
    ) -> Result<Self> {
        Ok(CausalResnetBlock {
            block1: CausalBlock1d::load(file, &format!("{prefix}.block1"), in_ch, out_ch)?,
            block2: CausalBlock1d::load(file, &format!("{prefix}.block2"), out_ch, out_ch)?,
            mlp_linear: SaLinear::load(
                file,
                &format!("{prefix}.mlp_linear"),
                time_emb_dim,
                out_ch,
            )?,
            res_conv: SaConv1d::load(
                file,
                &format!("{prefix}.res_conv"),
                in_ch,
                out_ch,
                1,
                1,
                0,
                1,
                1,
            )?,
            in_ch,
            out_ch,
        })
    }

    fn forward(&self, x: &[f32], frames: usize, t_emb: &[f32]) -> Vec<f32> {
        let mut h = self.block1.forward(x, frames);
        // h += mlp_linear(mish(time_emb)) broadcast over frames.
        let mut cond = t_emb.to_vec();
        mish(&mut cond);
        let projected = self.mlp_linear.forward(&cond, 1);
        for frame in 0..frames {
            for (c, v) in projected.iter().enumerate() {
                h[frame * self.out_ch + c] += v;
            }
        }
        let h = self.block2.forward(&h, frames);
        let res = self.res_conv.forward(x);
        let mut out = vec![0.0f32; frames * self.out_ch];
        for (o, (a, b)) in out.iter_mut().zip(h.iter().zip(&res)) {
            *o = a + b;
        }
        out
    }
}

/// `DiffusersAttention`: q/k/v projections to `heads * head_dim`
/// (no bias), out projection back with bias, full self-attention
/// (the reference runs it with a zero additive bias, which is no mask).
#[derive(Debug, Clone)]
struct DiffusersAttention {
    heads: usize,
    head_dim: usize,
    query_proj: SaLinear,
    key_proj: SaLinear,
    value_proj: SaLinear,
    out_proj: SaLinear,
}

impl DiffusersAttention {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let inner = heads * head_dim;
        Ok(DiffusersAttention {
            heads,
            head_dim,
            query_proj: SaLinear::load_no_bias(file, &format!("{prefix}.query_proj"), dim, inner)?,
            key_proj: SaLinear::load_no_bias(file, &format!("{prefix}.key_proj"), dim, inner)?,
            value_proj: SaLinear::load_no_bias(file, &format!("{prefix}.value_proj"), dim, inner)?,
            out_proj: SaLinear::load(file, &format!("{prefix}.out_proj"), inner, dim)?,
        })
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let dh = self.head_dim;
        let q = self.query_proj.forward(x, t);
        let k = self.key_proj.forward(x, t);
        let v = self.value_proj.forward(x, t);
        let scale = (dh as f32).powf(-0.5);
        let mut out = vec![0.0f32; t * self.heads * dh];
        let mut scores = vec![0.0f32; t];
        for h in 0..self.heads {
            let off = h * dh;
            for i in 0..t {
                for (j, s) in scores.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for d in 0..dh {
                        acc += q[i * self.heads * dh + off + d] * k[j * self.heads * dh + off + d];
                    }
                    *s = acc * scale;
                }
                ops::softmax_row(&mut scores);
                for (j, &s) in scores.iter().enumerate() {
                    for d in 0..dh {
                        out[i * self.heads * dh + off + d] += s * v[j * self.heads * dh + off + d];
                    }
                }
            }
        }
        self.out_proj.forward(&out, t)
    }
}

/// `BasicTransformerBlock`: self-attention and an exact-erf-GELU feed
/// forward, each on its own pre-norm.
#[derive(Debug, Clone)]
struct TransformerBlock {
    norm1: SaLayerNorm,
    attn: DiffusersAttention,
    norm3: SaLayerNorm,
    ff_0: SaLinear,
    ff_1: SaLinear,
}

impl TransformerBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Ok(TransformerBlock {
            norm1: SaLayerNorm::load(file, &format!("{prefix}.norm1"), dim)?,
            attn: DiffusersAttention::load(file, &format!("{prefix}.attn"), dim, heads, head_dim)?,
            norm3: SaLayerNorm::load(file, &format!("{prefix}.norm3"), dim)?,
            ff_0: SaLinear::load(file, &format!("{prefix}.ff.layers.0"), dim, dim * 4)?,
            ff_1: SaLinear::load(file, &format!("{prefix}.ff.layers.1"), dim * 4, dim)?,
        })
    }

    fn forward(&self, x: &mut [f32], t: usize) {
        let mut normed = x.to_vec();
        self.norm1.forward(&mut normed, t, 1e-5);
        let attended = self.attn.forward(&normed, t);
        for (v, a) in x.iter_mut().zip(&attended) {
            *v += a;
        }
        let mut normed = x.to_vec();
        self.norm3.forward(&mut normed, t, 1e-5);
        let mut hidden = self.ff_0.forward(&normed, t);
        ops::gelu_erf(&mut hidden);
        let projected = self.ff_1.forward(&hidden, t);
        for (v, p) in x.iter_mut().zip(&projected) {
            *v += p;
        }
    }
}

/// One down/mid/up stage: a resnet followed by transformer blocks. The
/// down stage ends in a causal conv (the single-width construction
/// makes it the list's last, which replaces the stride-2 downsample
/// with `CausalConv1d`), the up stage ends the same way; mid stages
/// carry no tail.
#[derive(Debug, Clone)]
struct StageBlock {
    resnet: CausalResnetBlock,
    transformers: Vec<TransformerBlock>,
    /// The trailing `CausalConv1d` stage (`downsample.conv` on the down
    /// path, `upsample.conv` on the up path); mid stages have none.
    tail_conv: Option<SaConv1d>,
}

impl StageBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        time_emb_dim: usize,
        n_blocks: usize,
        heads: usize,
        head_dim: usize,
        tail_name: Option<&str>,
    ) -> Result<Self> {
        let transformers = (0..n_blocks)
            .map(|j| {
                TransformerBlock::load(
                    file,
                    &format!("{prefix}.transformer_{j}"),
                    out_ch,
                    heads,
                    head_dim,
                )
            })
            .collect::<Result<_>>()?;
        let tail_conv = match tail_name {
            None => None,
            Some(name) => Some(SaConv1d::load(
                file,
                &format!("{prefix}.{name}.conv"),
                out_ch,
                out_ch,
                3,
                1,
                0,
                1,
                1,
            )?),
        };
        Ok(StageBlock {
            resnet: CausalResnetBlock::load(
                file,
                &format!("{prefix}.resnet"),
                in_ch,
                out_ch,
                time_emb_dim,
            )?,
            transformers,
            tail_conv,
        })
    }

    fn run_transformers(&self, h: Vec<f32>, frames: usize, channels: usize) -> Vec<f32> {
        let mut rows = to_rows(&h, channels, frames);
        for transformer in &self.transformers {
            transformer.forward(&mut rows, frames);
        }
        to_cm(&rows, frames, channels)
    }
}

/// The matcha U-Net estimator (`decoder.py::ConditionalDecoder`, causal
/// variant at the single-width construction). Channel-major `[C, T]`
/// between stages.
#[derive(Debug, Clone)]
struct ConditionalDecoder {
    time_mlp: TimeMlp,
    down_block: StageBlock,
    mid_blocks: Vec<StageBlock>,
    up_block: StageBlock,
    final_block: CausalBlock1d,
    final_proj: SaConv1d,
    in_channels: usize,
    out_channels: usize,
    channels: usize,
    spk_width: usize,
}

impl ConditionalDecoder {
    fn load(file: &SafetensorsFile, prefix: &str, config: &FlowConfig) -> Result<Self> {
        let channels = config.decoder_channels;
        let time_embed_dim = channels * 4;
        // The decoder input is [x, mu, projected speaker, cond]; the
        // speaker embedding is projected to the mel width first, so the
        // width is always 4 * mel.
        let in_channels = config.output_size * 4;
        let spk_width = config.output_size;
        let heads = config.decoder_num_heads;
        let head_dim = config.decoder_head_dim;
        let n_blocks = config.decoder_n_blocks;
        let down_block = StageBlock::load(
            file,
            &format!("{prefix}.down_blocks_0"),
            in_channels,
            channels,
            time_embed_dim,
            n_blocks,
            heads,
            head_dim,
            Some("downsample"),
        )?;
        let mid_blocks = (0..config.decoder_num_mid_blocks)
            .map(|i| {
                StageBlock::load(
                    file,
                    &format!("{prefix}.mid_blocks_{i}"),
                    channels,
                    channels,
                    time_embed_dim,
                    n_blocks,
                    heads,
                    head_dim,
                    None,
                )
            })
            .collect::<Result<_>>()?;
        Ok(ConditionalDecoder {
            time_mlp: TimeMlp::load(
                file,
                &format!("{prefix}.time_mlp"),
                in_channels,
                time_embed_dim,
            )?,
            down_block,
            mid_blocks,
            up_block: StageBlock::load(
                file,
                &format!("{prefix}.up_blocks_0"),
                channels * 2,
                channels,
                time_embed_dim,
                n_blocks,
                heads,
                head_dim,
                Some("upsample"),
            )?,
            final_block: CausalBlock1d::load(
                file,
                &format!("{prefix}.final_block"),
                channels,
                channels,
            )?,
            final_proj: SaConv1d::load(
                file,
                &format!("{prefix}.final_proj"),
                channels,
                config.output_size,
                1,
                1,
                0,
                1,
                1,
            )?,
            in_channels,
            out_channels: config.output_size,
            channels,
            spk_width,
        })
    }

    /// `x`, `mu`, `cond` are channel-major `[mel, T]`; `spks` is
    /// `[spk_dim]` broadcast over time; `t_emb` is the
    /// `[time_embed_dim]` embedding of the current step. Channel-major
    /// `[mel, T]` out.
    fn forward(
        &self,
        x_in: &[f32],
        mu: &[f32],
        spks: &[f32],
        cond: &[f32],
        t_emb: &[f32],
        frames: usize,
    ) -> Vec<f32> {
        let mel = self.out_channels;
        // Concatenate [x, mu, spks, cond] channel-major.
        let mut x = vec![0.0f32; self.in_channels * frames];
        x[..mel * frames].copy_from_slice(x_in);
        x[mel * frames..2 * mel * frames].copy_from_slice(mu);
        {
            let start = 2 * mel * frames;
            let spk_slice = &mut x[start..start + self.spk_width * frames];
            for (i, slot) in spk_slice.iter_mut().enumerate() {
                *slot = spks[i / frames];
            }
        }
        x[2 * mel * frames + self.spk_width * frames..].copy_from_slice(cond);

        // Down: resnet + transformers + causal conv tail. The skip is
        // the pre-tail state.
        let ch = self.channels;
        let down = &self.down_block;
        let mut h = down.resnet.forward(&x, frames, t_emb);
        h = down.run_transformers(h, frames, ch);
        let skip = h;
        let mut x = down
            .tail_conv
            .as_ref()
            .expect("down stage carries a tail conv")
            .forward(&skip);

        // Mid stack: resnet + transformers, no tail.
        for mid in &self.mid_blocks {
            let mut h = mid.resnet.forward(&x, frames, t_emb);
            h = mid.run_transformers(h, frames, ch);
            x = h;
        }

        // Up: truncate x to the skip length, concat channels
        // ([x, skip]), resnet + transformers + causal conv tail.
        let up = &self.up_block;
        let up_in = up.resnet.in_ch;
        let half = up_in / 2;
        let mut concat = vec![0.0f32; up_in * frames];
        for c in 0..half {
            concat[c * frames..(c + 1) * frames].copy_from_slice(&x[c * frames..(c + 1) * frames]);
        }
        concat[half * frames..].copy_from_slice(&skip);
        let mut h = up.resnet.forward(&concat, frames, t_emb);
        h = up.run_transformers(h, frames, ch);
        let x = up
            .tail_conv
            .as_ref()
            .expect("up stage carries a tail conv")
            .forward(&h);

        // Final block + projection.
        let h = self.final_block.forward(&x, frames);
        self.final_proj.forward(&h)
    }
}

/// `CausalConditionalCFM`: 10-step Euler over the cosine schedule with
/// classifier-free guidance. The noise start point is the first `T`
/// frames of the reference's seed-0 `(1, 80, 50 * 300)` buffer.
#[derive(Debug, Clone)]
struct CausalConditionalCfm {
    estimator: ConditionalDecoder,
    cfg_rate: f32,
    /// Channel-major `[mel, 50 * 300]` noise buffer.
    rand_noise: Vec<f32>,
    mel: usize,
}

impl CausalConditionalCfm {
    fn load(file: &SafetensorsFile, prefix: &str, config: &FlowConfig) -> Result<Self> {
        let mel = config.output_size;
        // The reference generates a (1, 80, 50 * 300) seed-0 buffer at
        // init and real converted checkpoints carry none; the loader
        // accepts any buffer whose length is a whole number of mel
        // rows and refuses empty ones.
        let name = format!("{prefix}.rand_noise");
        let rand_noise = file.load_as_f32(&name).map_err(SpeechError::from)?;
        if rand_noise.is_empty() || rand_noise.len() % mel != 0 {
            return Err(SpeechError::Tensor {
                name,
                why: "noise buffer length must be a whole number of mel rows".to_string(),
            });
        }
        Ok(CausalConditionalCfm {
            estimator: ConditionalDecoder::load(file, &format!("{prefix}.estimator"), config)?,
            cfg_rate: config.inference_cfg_rate,
            rand_noise,
            mel,
        })
    }

    /// The first `frames` noise frames per channel (the reference slices
    /// `rand_noise[:, :, :T]`).
    fn noise_window(&self, frames: usize) -> Vec<f32> {
        let stride = self.rand_noise.len() / self.mel;
        let mut x = vec![0.0f32; self.mel * frames];
        for c in 0..self.mel {
            x[c * frames..(c + 1) * frames]
                .copy_from_slice(&self.rand_noise[c * stride..c * stride + frames]);
        }
        x
    }

    /// `mu` and `cond` channel-major `[mel, T]`, `spks` `[spk_dim]`.
    /// Returns the solved channel-major `[mel, T]`.
    fn forward(&self, mu: &[f32], spks: &[f32], cond: &[f32], n_timesteps: usize) -> Vec<f32> {
        let frames = mu.len() / self.mel;
        let zeros = vec![0.0f32; mu.len()];
        let zeros_spk = vec![0.0f32; spks.len()];
        // t_span = 1 - cos(linspace(0, 1, n + 1) * pi / 2).
        let t_span: Vec<f32> = (0..=n_timesteps)
            .map(|i| 1.0 - ((i as f32 / n_timesteps as f32) * 0.5 * std::f32::consts::PI).cos())
            .collect();
        let mut x = self.noise_window(frames);
        let mut t = t_span[0];
        let mut dt = t_span[1] - t_span[0];
        for step in 1..t_span.len() {
            let t_emb = self.estimator.time_mlp.forward(t);
            let dphi = self.estimator.forward(&x, mu, spks, cond, &t_emb, frames);
            let dphi_uncond = self
                .estimator
                .forward(&x, &zeros, &zeros_spk, &zeros, &t_emb, frames);
            for i in 0..x.len() {
                x[i] += dt * ((1.0 + self.cfg_rate) * dphi[i] - self.cfg_rate * dphi_uncond[i]);
            }
            t += dt;
            if step < t_span.len() - 1 {
                dt = t_span[step + 1] - t;
            }
        }
        x
    }
}

/// The assembled flow model (`flow.py::CausalMaskedDiffWithXvec`).
pub struct ChatterboxFlow {
    config: FlowConfig,
    input_embedding: Vec<f32>,
    spk_embed_affine: SaLinear,
    encoder: UpsampleConformerEncoder,
    encoder_proj: SaLinear,
    decoder: CausalConditionalCfm,
}

impl ChatterboxFlow {
    pub fn load(config: &FlowConfig, file: &SafetensorsFile, prefix: &str) -> Result<Self> {
        let embedding = load_f32_shaped(
            file,
            &format!("{prefix}.input_embedding.weight"),
            &[config.vocab_size, config.input_size],
        )?;
        Ok(ChatterboxFlow {
            config: config.clone(),
            input_embedding: embedding,
            spk_embed_affine: SaLinear::load(
                file,
                &format!("{prefix}.spk_embed_affine_layer"),
                config.spk_embed_dim,
                config.output_size,
            )?,
            encoder: UpsampleConformerEncoder::load(
                file,
                &format!("{prefix}.encoder"),
                config.input_size,
                config.input_size,
                config.attention_heads,
                config.linear_units,
                config.num_blocks,
                config.num_up_blocks,
                config.upsample_stride,
                config.pre_lookahead_len,
            )?,
            encoder_proj: SaLinear::load(
                file,
                &format!("{prefix}.encoder_proj"),
                config.input_size,
                config.output_size,
            )?,
            decoder: CausalConditionalCfm::load(file, &format!("{prefix}.decoder"), config)?,
        })
    }

    /// Mel frames per token.
    pub fn up_rate(&self) -> usize {
        self.config.upsample_stride
    }

    /// The configured Euler step count (the reference `n_timesteps`).
    pub fn n_timesteps(&self) -> usize {
        self.config.n_timesteps
    }

    /// Batch-of-one inference (`flow.inference`, `finalize = true`):
    /// prompt + generated tokens and the length-matched prompt mel in,
    /// generated mel frames out as rows `[mel_len2, num_mels]`.
    pub fn infer(
        &mut self,
        speech_tokens: &[i32],
        prompt_token: &[i32],
        prompt_feat: &[f32],
        embedding: &[f32],
        n_timesteps: usize,
    ) -> Result<Vec<f32>> {
        let mel = self.config.output_size;
        if embedding.len() != self.config.spk_embed_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "speaker embedding has {} dims, expected {}",
                    embedding.len(),
                    self.config.spk_embed_dim
                ),
            });
        }
        if prompt_feat.len() % mel != 0 {
            return Err(SpeechError::Input {
                why: "prompt_feat must be row-major mel frames".to_string(),
            });
        }
        // L2-normalize the speaker embedding (+1e-8) and project.
        let mut spk = embedding.to_vec();
        let norm = spk.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
        for v in &mut spk {
            *v /= norm;
        }
        let spk = self.spk_embed_affine.forward(&spk, 1);

        // Embed the clipped concatenated tokens; the all-valid mask
        // reduces to an identity multiply.
        let mut tokens = prompt_token.to_vec();
        tokens.extend_from_slice(speech_tokens);
        let mut token_rows = vec![0.0f32; tokens.len() * self.config.input_size];
        for (i, &tok) in tokens.iter().enumerate() {
            let idx = tok.clamp(0, self.config.vocab_size as i32 - 1) as usize;
            token_rows[i * self.config.input_size..(i + 1) * self.config.input_size]
                .copy_from_slice(
                    &self.input_embedding
                        [idx * self.config.input_size..(idx + 1) * self.config.input_size],
                );
        }
        let t = tokens.len();
        let h = self.encoder.forward(&token_rows, t)?;
        let mu_rows = self
            .encoder_proj
            .forward(&h, t * self.config.upsample_stride);
        let mu = to_cm(&mu_rows, t * self.config.upsample_stride, mel);

        // conds = [prompt_feat, zeros] over the full length, channel-major.
        let prompt_len = prompt_feat.len() / mel;
        let total = t * self.config.upsample_stride;
        let mut cond_cm = vec![0.0f32; mel * total];
        for (i, &v) in prompt_feat.iter().enumerate() {
            let frame = i / mel;
            let bin = i % mel;
            cond_cm[bin * total + frame] = v;
        }
        let feat = self.decoder.forward(&mu, &spk, &cond_cm, n_timesteps);
        // Slice off the prompt span, back to rows.
        let gen_len = total - prompt_len;
        let mut out = vec![0.0f32; gen_len * mel];
        for bin in 0..mel {
            for f in 0..gen_len {
                out[f * mel + bin] = feat[bin * total + prompt_len + f];
            }
        }
        Ok(out)
    }
}
