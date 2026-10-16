//! The chatterbox_turbo S3Gen flow: speech tokens + speaker prompt ->
//! mel.
//!
//! Reference: `chatterbox_turbo/models/s3gen/s3gen.py`
//! (`S3Token2Mel`), `s3gen/flow_matching.py`
//! (`CausalConditionalCFM` in its meanflow mode), and
//! `s3gen/decoder.py` (`ConditionalDecoder` and its causal blocks), all
//! at the pinned commit recorded in the parent module.
//!
//! Structure: tokens embed (vocab clipped) and, concatenated with the
//! reference prompt tokens, run through the conformer encoder
//! ([`conformer`]) that doubles the frame rate; a linear projects to 80
//! mel dims; and the diffusers-style U-Net estimator (causal
//! conv-LayerNorm-Mish resnet blocks, bidirectional self-attention
//! transformer blocks with exact-erf GELU feed-forwards, one down
//! stage and one up stage at the single reference width, each ending
//! in a `CausalConv1d`) runs a 2-step Euler solve over the LINEAR
//! schedule with the meanflow time mixing: the estimator consumes both
//! the start time `t` and the end time `r` through one shared time MLP
//! and a bias-free mixing matrix. Unlike the sibling family there is no
//! classifier-free guidance and no cosine warp: the distilled meanflow
//! model takes the plain `_basic_euler` path.
//!
//! Batch-of-one with exact lengths: every padding mask in the reference
//! call is all-valid, so the mask multiplies reduce to identity and the
//! decoder attention runs unmasked (the reference mask enters as a
//! `mx.where(mask > 0, scores, -1e9)` that never fires).
//!
//! The flow start point is random: the reference draws fresh normals
//! for the generation span (`noised_mels`) and for the prompt span of
//! the initial state (`z`) at every call. Both windows are explicit
//! [`FlowNoise`] inputs here so a fixture replays the recorded Python
//! stream; `None` draws from the crate RNG (the threefry uniforms are
//! bit-compatible but the normal draws differ by the erfinv
//! implementation, documented in the crate RNG module - the flow start
//! point shifts, the conditioning does not).

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::codec::wnconv::load_f32_shaped;
use crate::music::minimax_music3::rng;
use crate::ops;
use crate::{Result, SpeechError};

use crate::tts::chatterbox::s3gen::{mish, to_cm, to_rows, SaConv1d, SaLayerNorm, SaLinear};

use super::campplus::{CampPlus, CampPlusConfig};
use super::conformer::UpsampleConformerEncoder;

/// Geometry of the flow model, one-to-one with the reference keyword
/// arguments (`S3Token2Mel.__init__` and the estimator it builds).
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Token embedding width and encoder width.
    pub input_size: usize,
    /// Mel bins (fixed by the vocoder at 80).
    pub output_size: usize,
    /// CAMPPlus speaker embedding width feeding `spk_embed_affine_layer`.
    pub spk_embed_dim: usize,
    /// Speech token vocabulary (6561 for the 25 Hz S3 tokenizer).
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub linear_units: usize,
    pub num_blocks: usize,
    /// Mel frames per token: the encoder upsample factor.
    pub upsample_stride: usize,
    pub pre_lookahead_len: usize,
    /// Euler steps of the meanflow solve (the turbo default 2).
    pub n_timesteps: usize,
    /// The constructor default 0.7 is parsed but unused: meanflow mode
    /// takes the `_basic_euler` path without classifier-free guidance.
    pub inference_cfg_rate: f32,
    pub meanflow: bool,
    /// Estimator (U-Net) width; the reference builds one down/up block
    /// each at this width.
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
            upsample_stride: 2,
            pre_lookahead_len: 3,
            n_timesteps: 2,
            inference_cfg_rate: 0.7,
            meanflow: true,
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
        let boolean = |name: &str, fallback: bool| -> Result<bool> {
            match value.get(name) {
                None | Some(serde_json::Value::Null) => Ok(fallback),
                Some(v) => v.as_bool().ok_or_else(|| SpeechError::BadConfig {
                    field: name.to_string(),
                    why: "expected a bool".to_string(),
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
            upsample_stride: int("upsample_stride", defaults.upsample_stride)?,
            pre_lookahead_len: int("pre_lookahead_len", defaults.pre_lookahead_len)?,
            n_timesteps: int("n_timesteps", defaults.n_timesteps)?,
            inference_cfg_rate: float("inference_cfg_rate", defaults.inference_cfg_rate)?,
            meanflow: boolean("meanflow", defaults.meanflow)?,
            decoder_channels: int("decoder_channels", defaults.decoder_channels)?,
            decoder_n_blocks: int("decoder_n_blocks", defaults.decoder_n_blocks)?,
            decoder_num_mid_blocks: int("decoder_num_mid_blocks", defaults.decoder_num_mid_blocks)?,
            decoder_num_heads: int("decoder_num_heads", defaults.decoder_num_heads)?,
            decoder_head_dim: int("decoder_head_dim", defaults.decoder_head_dim)?,
        })
    }
}

/// The explicit flow start windows (`FlowNoise`): the reference draws
/// both at call time.
#[derive(Debug, Clone, Default)]
pub struct FlowNoise {
    /// The generation-span start (`noised_mels`), channel-major
    /// `[mel, 2 * gen_tokens]`.
    pub gen_window: Vec<f32>,
    /// The prompt-span start (the head of the CFM `z`), channel-major
    /// `[mel, 2 * prompt_tokens]`.
    pub prompt_window: Vec<f32>,
}

/// `sinusoidal_pos_emb` (`decoder.py`): half exponential frequencies,
/// `scale * t * freq`, `[sin, cos]` halves.
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

/// `CausalBlock1D` (`decoder.py`): causal conv k3 (kernel-1 left pad) ->
/// LayerNorm over channels -> Mish. Channel-major `[C, T]` in and out.
#[derive(Debug, Clone)]
struct CausalBlock1d {
    channels: usize,
    conv: SaConv1d,
    norm: SaLayerNorm,
}

impl CausalBlock1d {
    /// The turbo checkpoint routes the conv through the `Conv1dPT`
    /// wrapper list: `block.0.conv.conv`, and the norm is `block.1`.
    fn load(file: &SafetensorsFile, prefix: &str, in_ch: usize, out_ch: usize) -> Result<Self> {
        Ok(CausalBlock1d {
            channels: out_ch,
            conv: SaConv1d::load(
                file,
                &format!("{prefix}.block.0.conv.conv"),
                in_ch,
                out_ch,
                3,
                1,
                0,
                1,
                1,
            )?,
            norm: SaLayerNorm::load(file, &format!("{prefix}.block.1"), out_ch)?,
        })
    }

    fn forward(&self, x: &[f32], frames: usize) -> Vec<f32> {
        let in_ch = x.len() / frames.max(1);
        let padded = crate::tts::chatterbox::conformer::pad_left_cm(x, in_ch, frames, 2);
        let h = self.conv.forward(&padded);
        let mut rows = to_rows(&h, self.channels, frames);
        self.norm.forward(&mut rows, frames, 1e-5);
        mish(&mut rows);
        to_cm(&rows, frames, self.channels)
    }
}

/// `ResnetBlock1D`: block1 -> time MLP (Mish first) -> block2 +
/// residual conv. `h` is channel-major, so the time embedding broadcast
/// steps the frame index (`h[c * T + t] += mlp(c)`, not
/// `h[t * C + c]`).
#[derive(Debug, Clone)]
struct ResnetBlock {
    block1: CausalBlock1d,
    block2: CausalBlock1d,
    mlp: SaLinear,
    res_conv: SaConv1d,
    in_ch: usize,
    out_ch: usize,
}

impl ResnetBlock {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        time_emb_dim: usize,
    ) -> Result<Self> {
        Ok(ResnetBlock {
            block1: CausalBlock1d::load(file, &format!("{prefix}.block1"), in_ch, out_ch)?,
            block2: CausalBlock1d::load(file, &format!("{prefix}.block2"), out_ch, out_ch)?,
            mlp: SaLinear::load(file, &format!("{prefix}.mlp.0"), time_emb_dim, out_ch)?,
            res_conv: SaConv1d::load(
                file,
                &format!("{prefix}.res_conv.conv"),
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
        // h += mlp(mish(time_emb)) broadcast over frames.
        let mut cond = t_emb.to_vec();
        mish(&mut cond);
        let projected = self.mlp.forward(&cond, 1);
        for frame in 0..frames {
            for (c, v) in projected.iter().enumerate() {
                h[c * frames + frame] += v;
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

/// `SelfAttention1D`: q/k/v projections to `heads * head_dim` (no
/// bias), out projection back with bias, full bidirectional
/// self-attention (the all-valid reference mask never fires).
#[derive(Debug, Clone)]
struct SelfAttention1d {
    heads: usize,
    head_dim: usize,
    to_q: SaLinear,
    to_k: SaLinear,
    to_v: SaLinear,
    to_out: SaLinear,
}

impl SelfAttention1d {
    fn load(
        file: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let inner = heads * head_dim;
        Ok(SelfAttention1d {
            heads,
            head_dim,
            to_q: SaLinear::load_no_bias(file, &format!("{prefix}.to_q"), dim, inner)?,
            to_k: SaLinear::load_no_bias(file, &format!("{prefix}.to_k"), dim, inner)?,
            to_v: SaLinear::load_no_bias(file, &format!("{prefix}.to_v"), dim, inner)?,
            to_out: SaLinear::load(file, &format!("{prefix}.to_out.0"), inner, dim)?,
        })
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let dh = self.head_dim;
        let q = self.to_q.forward(x, t);
        let k = self.to_k.forward(x, t);
        let v = self.to_v.forward(x, t);
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
        self.to_out.forward(&out, t)
    }
}

/// `TransformerBlock`: bidirectional self-attention and an exact-erf
/// GELU feed-forward, each on its own pre-norm (`norm1`, `norm3`).
#[derive(Debug, Clone)]
struct TransformerBlock {
    norm1: SaLayerNorm,
    attn1: SelfAttention1d,
    norm3: SaLayerNorm,
    ff_proj: SaLinear,
    ff_out: SaLinear,
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
            attn1: SelfAttention1d::load(file, &format!("{prefix}.attn1"), dim, heads, head_dim)?,
            norm3: SaLayerNorm::load(file, &format!("{prefix}.norm3"), dim)?,
            // `ff.net.0.proj` (the GELU Linear) then `ff.net.1` (the
            // output Linear).
            ff_proj: SaLinear::load(file, &format!("{prefix}.ff.net.0.proj"), dim, dim * 4)?,
            ff_out: SaLinear::load(file, &format!("{prefix}.ff.net.1"), dim * 4, dim)?,
        })
    }

    fn forward(&self, x: &mut [f32], t: usize) {
        let mut normed = x.to_vec();
        self.norm1.forward(&mut normed, t, 1e-5);
        let attended = self.attn1.forward(&normed, t);
        for (v, a) in x.iter_mut().zip(&attended) {
            *v += a;
        }
        let mut normed = x.to_vec();
        self.norm3.forward(&mut normed, t, 1e-5);
        let mut hidden = self.ff_proj.forward(&normed, t);
        ops::gelu_erf(&mut hidden);
        let projected = self.ff_out.forward(&hidden, t);
        for (v, p) in x.iter_mut().zip(&projected) {
            *v += p;
        }
    }
}

/// One down/mid/up stage: a resnet followed by transformer blocks. With
/// the single reference width the down and up stages are their list's
/// last entry, so both end in a `CausalConv1d` (`downsample` /
/// `upsample`); mid stages carry no tail.
#[derive(Debug, Clone)]
struct StageBlock {
    resnet: ResnetBlock,
    transformers: Vec<TransformerBlock>,
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
                    &format!("{prefix}.transformer_blocks.{j}"),
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
                &format!("{prefix}.{name}.conv.conv"),
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
            resnet: ResnetBlock::load(
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

/// The U-Net estimator (`decoder.py::ConditionalDecoder`, causal
/// variant at the single-width construction). Channel-major `[C, T]`
/// between stages. Meanflow mode mixes the `t` and `r` time embeddings
/// through the bias-free `time_embed_mixer`.
#[derive(Debug, Clone)]
struct ConditionalDecoder {
    time_mlp: TimeMlp,
    down_block: StageBlock,
    mid_blocks: Vec<StageBlock>,
    up_block: StageBlock,
    final_block: CausalBlock1d,
    final_proj: SaConv1d,
    time_embed_mixer: Option<SaLinear>,
    in_channels: usize,
    out_channels: usize,
    channels: usize,
    spk_width: usize,
    meanflow: bool,
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
        // With `channels = [256]` the down list holds one stage and the
        // up list one stage, each the last of its list: both end in the
        // causal tail conv.
        let down_block = StageBlock::load(
            file,
            &format!("{prefix}.down_blocks.0"),
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
                    &format!("{prefix}.mid_blocks.{i}"),
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
                &format!("{prefix}.up_blocks.0"),
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
                &format!("{prefix}.final_proj.conv"),
                channels,
                config.output_size,
                1,
                1,
                0,
                1,
                1,
            )?,
            time_embed_mixer: if config.meanflow {
                Some(SaLinear::load_no_bias(
                    file,
                    &format!("{prefix}.time_embed_mixer"),
                    time_embed_dim * 2,
                    time_embed_dim,
                )?)
            } else {
                None
            },
            in_channels,
            out_channels: config.output_size,
            channels,
            spk_width,
            meanflow: config.meanflow,
        })
    }

    /// Embeds the (t, r) pair through the shared time MLP and the
    /// meanflow mixer.
    fn time_embed(&self, t: f32, r: f32) -> Result<Vec<f32>> {
        let t_emb = self.time_mlp.forward(t);
        if !self.meanflow {
            return Ok(t_emb);
        }
        let mixer = self
            .time_embed_mixer
            .as_ref()
            .ok_or_else(|| SpeechError::BadConfig {
                field: "meanflow".to_string(),
                why: "meanflow mode requires the time_embed_mixer".to_string(),
            })?;
        let r_emb = self.time_mlp.forward(r);
        let mut concat = t_emb;
        concat.extend_from_slice(&r_emb);
        Ok(mixer.forward(&concat, 1))
    }

    /// `x`, `mu`, `cond` are channel-major `[mel, T]`; `spks` is
    /// `[spk_dim]` broadcast over time; `t_emb` is the mixed embedding
    /// of the current step. Channel-major `[mel, T]` out.
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
            .forward(&crate::tts::chatterbox::conformer::pad_left_cm(
                &skip, ch, frames, 2,
            ));

        // Mid stack: resnet + transformers, no tail.
        for mid in &self.mid_blocks {
            let mut h = mid.resnet.forward(&x, frames, t_emb);
            h = mid.run_transformers(h, frames, ch);
            x = h;
        }

        // Up: truncate x to the skip length (equal here), concat
        // channels ([x, skip]), resnet + transformers + causal tail.
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
            .forward(&crate::tts::chatterbox::conformer::pad_left_cm(
                &h, ch, frames, 2,
            ));

        // Final block + projection.
        let h = self.final_block.forward(&x, frames);
        self.final_proj.forward(&h)
    }
}

/// `CausalConditionalCFM` in its meanflow mode: an n-step Euler solve
/// over the LINEAR schedule `linspace(0, 1, n + 1)` with no
/// classifier-free guidance. The start state is the explicit
/// [`FlowNoise`] windows, or fresh crate-RNG normals in their
/// reference draw order (generation span first, then the full state).
#[derive(Debug, Clone)]
struct CausalConditionalCfm {
    estimator: ConditionalDecoder,
    cfg_rate: f32,
    mel: usize,
}

impl CausalConditionalCfm {
    fn load(file: &SafetensorsFile, prefix: &str, config: &FlowConfig) -> Result<Self> {
        Ok(CausalConditionalCfm {
            estimator: ConditionalDecoder::load(file, &format!("{prefix}.estimator"), config)?,
            cfg_rate: config.inference_cfg_rate,
            mel: config.output_size,
        })
    }

    /// `mu` and `cond` channel-major `[mel, T]`, `spks` `[spk_dim]`.
    /// `gen_frames` is the generation span (twice the generated token
    /// count) whose start window the reference replaces with fresh
    /// normals.
    fn forward(
        &self,
        mu: &[f32],
        spks: &[f32],
        cond: &[f32],
        n_timesteps: usize,
        noise: Option<&FlowNoise>,
        keys: &mut rng::KeySequence,
    ) -> Result<Vec<f32>> {
        let frames = mu.len() / self.mel;
        let prompt_len = frames - noise.map(|n| n.gen_window.len() / self.mel).unwrap_or(0);
        // The start state: prompt span + generation span, spliced PER
        // CHANNEL (`z = concat([z[:, :, :prompt_len], noised_mels],
        // axis=2)` on the channel-major layout).
        let gen_len = frames - prompt_len;
        let mut x = match noise {
            Some(n) => {
                if n.prompt_window.len() != self.mel * prompt_len {
                    return Err(SpeechError::Input {
                        why: format!(
                            "flow noise prompt window has {} entries, expected {}",
                            n.prompt_window.len(),
                            self.mel * prompt_len
                        ),
                    });
                }
                if n.gen_window.len() != self.mel * gen_len {
                    return Err(SpeechError::Input {
                        why: format!(
                            "flow noise generation window has {} entries, expected {}",
                            n.gen_window.len(),
                            self.mel * gen_len
                        ),
                    });
                }
                let mut z = vec![0.0f32; self.mel * frames];
                for c in 0..self.mel {
                    z[c * frames..c * frames + prompt_len]
                        .copy_from_slice(&n.prompt_window[c * prompt_len..(c + 1) * prompt_len]);
                    z[c * frames + prompt_len..(c + 1) * frames]
                        .copy_from_slice(&n.gen_window[c * gen_len..(c + 1) * gen_len]);
                }
                z
            }
            None => {
                // Reference draw order: `noised_mels` (generation span)
                // first, then the full-state normals whose prompt head
                // survives the splice.
                let gen = rng::normal(keys.next(), self.mel * gen_len);
                let full = rng::normal(keys.next(), self.mel * frames);
                let mut z = vec![0.0f32; self.mel * frames];
                for c in 0..self.mel {
                    z[c * frames..c * frames + prompt_len]
                        .copy_from_slice(&full[c * frames..c * frames + prompt_len]);
                    z[c * frames + prompt_len..(c + 1) * frames]
                        .copy_from_slice(&gen[c * gen_len..(c + 1) * gen_len]);
                }
                z
            }
        };
        // t_span = linspace(0, 1, n + 1); meanflow applies no cosine
        // warp.
        let t_span: Vec<f32> = (0..=n_timesteps)
            .map(|i| i as f32 / n_timesteps as f32)
            .collect();
        let _ = self.cfg_rate;
        for step in 0..n_timesteps {
            let t = t_span[step];
            let r = t_span[step + 1];
            let t_emb = self.estimator.time_embed(t, r)?;
            let dxdt = self.estimator.forward(&x, mu, spks, cond, &t_emb, frames);
            let dt = r - t;
            for (v, d) in x.iter_mut().zip(&dxdt) {
                *v += dt * d;
            }
        }
        Ok(x)
    }
}

/// The assembled token-to-mel model (`s3gen.py::S3Token2Mel`): the
/// token embedding, the CAMPPlus speaker encoder, the speaker
/// projection, the conformer encoder, the mel projection, and the
/// meanflow decoder.
pub struct S3GenFlow {
    config: FlowConfig,
    input_embedding: Vec<f32>,
    spk_embed_affine: SaLinear,
    encoder: UpsampleConformerEncoder,
    encoder_proj: SaLinear,
    decoder: CausalConditionalCfm,
    speaker_encoder: CampPlus,
    campplus_config: CampPlusConfig,
}

impl S3GenFlow {
    pub fn load(
        config: &FlowConfig,
        campplus_config: CampPlusConfig,
        file: &SafetensorsFile,
        prefix: &str,
    ) -> Result<Self> {
        let embedding = load_f32_shaped(
            file,
            &format!("{prefix}.input_embedding.weight"),
            &[config.vocab_size, config.input_size],
        )?;
        Ok(S3GenFlow {
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
            speaker_encoder: CampPlus::load(
                file,
                &format!("{prefix}.speaker_encoder."),
                campplus_config.clone(),
            )?,
            campplus_config,
        })
    }

    /// The configured Euler step count (the turbo default 2).
    pub fn n_timesteps(&self) -> usize {
        self.config.n_timesteps
    }

    /// CAMPPlus speaker embedding over a mean-normalized fbank
    /// (`speaker_encoder.inference` on precomputed features).
    pub fn speaker_embed(&self, fbank: &[f32], frames: usize) -> Vec<f32> {
        self.speaker_encoder.inference(fbank, frames)
    }

    pub fn campplus_config(&self) -> &CampPlusConfig {
        &self.campplus_config
    }

    /// L2-normalizes the speaker embedding (+1e-8) and projects it to
    /// the mel width.
    fn project_spk(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if embedding.len() != self.config.spk_embed_dim {
            return Err(SpeechError::Input {
                why: format!(
                    "speaker embedding has {} dims, expected {}",
                    embedding.len(),
                    self.config.spk_embed_dim
                ),
            });
        }
        let mut spk = embedding.to_vec();
        let norm = spk.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
        for v in &mut spk {
            *v /= norm;
        }
        Ok(self.spk_embed_affine.forward(&spk, 1))
    }

    /// Embeds the clipped concatenated tokens and runs the conformer
    /// encoder plus mel projection. Returns the channel-major
    /// `[mel, T * stride]` encoder output.
    fn encode(&mut self, prompt_token: &[i32], speech_tokens: &[i32]) -> Result<Vec<f32>> {
        let mel = self.config.output_size;
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
        let frames = t * self.encoder.up_rate();
        let mu_rows = self.encoder_proj.forward(&h, frames);
        Ok(to_cm(&mu_rows, frames, mel))
    }

    /// `S3Token2Mel.__call__` at batch-of-one exact lengths: prompt +
    /// generated tokens and the length-matched prompt mel in, generated
    /// mel frames out as rows `[2 * gen_tokens, num_mels]`.
    #[allow(clippy::too_many_arguments)]
    pub fn infer(
        &mut self,
        speech_tokens: &[i32],
        prompt_token: &[i32],
        prompt_feat: &[f32],
        embedding: &[f32],
        n_timesteps: usize,
        noise: Option<&FlowNoise>,
        keys: &mut rng::KeySequence,
    ) -> Result<Vec<f32>> {
        let mel = self.config.output_size;
        if prompt_feat.len() % mel != 0 {
            return Err(SpeechError::Input {
                why: "prompt_feat must be row-major mel frames".to_string(),
            });
        }
        let spk = self.project_spk(embedding)?;
        let mu = self.encode(prompt_token, speech_tokens)?;
        let total = mu.len() / mel;
        let prompt_len = prompt_feat.len() / mel;
        if total != prompt_len + speech_tokens.len() * self.config.upsample_stride {
            return Err(SpeechError::Input {
                why: format!(
                    "encoder produced {total} frames for a {}-frame prompt and {} tokens",
                    prompt_len,
                    speech_tokens.len()
                ),
            });
        }
        // conds = [prompt_feat, zeros] over the full length,
        // channel-major.
        let mut cond_cm = vec![0.0f32; mel * total];
        for (i, &v) in prompt_feat.iter().enumerate() {
            let frame = i / mel;
            let bin = i % mel;
            cond_cm[bin * total + frame] = v;
        }
        let feat = self
            .decoder
            .forward(&mu, &spk, &cond_cm, n_timesteps, noise, keys)?;
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
