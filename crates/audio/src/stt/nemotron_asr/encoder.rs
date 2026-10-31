//! Causal FastConformer encoder for the pinned Nemotron ASR checkpoint.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::models::stt::nemotron_asr::NemotronAsrConfig;
use crate::nn::{tensor_error, LayerNorm, Linear};
use crate::ops;
use crate::{Result, SpeechError};

const LN_EPS: f32 = 1e-5;

fn load_tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let desc = file
        .descriptor(name)
        .ok_or_else(|| tensor_error(name, "missing from safetensors"))?;
    if desc.shape != shape {
        return Err(tensor_error(
            name,
            format!("expected shape {shape:?}, found {:?}", desc.shape),
        ));
    }
    if !matches!(desc.dtype.as_str(), "F16" | "BF16" | "F32") {
        return Err(tensor_error(
            name,
            format!("expected floating point weights, found {}", desc.dtype),
        ));
    }
    file.load_as_f32(name)
        .map_err(|error| tensor_error(name, format!("load failed: {error}")))
}

fn load_conv2d(
    file: &SafetensorsFile,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    groups: usize,
) -> Result<Conv2d> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let in_per_group = in_channels / groups;
    // MLX stores Conv2d weights as [out, kernel_h, kernel_w, in_per_group].
    let source = load_tensor(
        file,
        &weight_name,
        &[out_channels, kernel, kernel, in_per_group],
    )?;
    let mut weight = vec![0.0; source.len()];
    for out in 0..out_channels {
        for input in 0..in_per_group {
            for kh in 0..kernel {
                for kw in 0..kernel {
                    let src = (((out * kernel + kh) * kernel + kw) * in_per_group) + input;
                    let dst = (((out * in_per_group + input) * kernel + kh) * kernel) + kw;
                    weight[dst] = source[src];
                }
            }
        }
    }
    Ok(Conv2d {
        weight,
        bias: load_tensor(file, &bias_name, &[out_channels])?,
        in_channels,
        out_channels,
        kernel,
        groups,
    })
}

fn load_conv1d(
    file: &SafetensorsFile,
    prefix: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    groups: usize,
) -> Result<Vec<f32>> {
    let name = format!("{prefix}.weight");
    let in_per_group = in_channels / groups;
    // MLX stores Conv1d weights as [out, kernel, in_per_group].
    let source = load_tensor(file, &name, &[out_channels, kernel, in_per_group])?;
    let mut weight = vec![0.0; source.len()];
    for out in 0..out_channels {
        for input in 0..in_per_group {
            for k in 0..kernel {
                weight[(out * in_per_group + input) * kernel + k] =
                    source[(out * kernel + k) * in_per_group + input];
            }
        }
    }
    Ok(weight)
}

fn load_linear(file: &SafetensorsFile, prefix: &str, out: usize, input: usize) -> Result<Linear> {
    let weight = load_tensor(file, &format!("{prefix}.weight"), &[out, input])?;
    let bias_name = format!("{prefix}.bias");
    let bias = if file.contains_tensor(&bias_name) {
        Some(load_tensor(file, &bias_name, &[out])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, input, out))
}

fn load_bias_free_linear(
    file: &SafetensorsFile,
    prefix: &str,
    out: usize,
    input: usize,
) -> Result<Linear> {
    let linear = load_linear(file, prefix, out, input)?;
    if linear.bias.is_some() {
        return Err(tensor_error(
            &format!("{prefix}.bias"),
            "checkpoint has a bias but this Nemotron layer is configured bias-free",
        ));
    }
    Ok(linear)
}

fn require_absent(file: &SafetensorsFile, name: &str) -> Result<()> {
    if file.contains_tensor(name) {
        return Err(tensor_error(
            name,
            "unexpected bias in the configured bias-free Nemotron layer",
        ));
    }
    Ok(())
}

fn load_norm(file: &SafetensorsFile, prefix: &str, width: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        load_tensor(file, &format!("{prefix}.weight"), &[width])?,
        Some(load_tensor(file, &format!("{prefix}.bias"), &[width])?),
        LN_EPS,
    ))
}

struct Conv2d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    groups: usize,
}

struct Subsampling {
    first: Conv2d,
    stages: Vec<(Conv2d, Conv2d)>,
    output: Linear,
    channels: usize,
    input_mels: usize,
}

struct RelPosAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    pos: Linear,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
}

struct FeedForward {
    input: Linear,
    output: Linear,
}

struct ConformerConvolution {
    pointwise_in: Vec<f32>,
    depthwise: Vec<f32>,
    depthwise_bias: Option<Vec<f32>>,
    norm: LayerNorm,
    pointwise_out: Vec<f32>,
    channels: usize,
    kernel: usize,
}

struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_attn: LayerNorm,
    attention: RelPosAttention,
    norm_conv: LayerNorm,
    convolution: ConformerConvolution,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
    hidden: usize,
    heads: usize,
    expansion: usize,
}

/// Offline Nemotron encoder. The attention mask preserves its trained causal
/// left context and right look-ahead, without carrying chunk cache state.
pub(crate) struct NemotronEncoder {
    subsampling: Subsampling,
    layers: Vec<ConformerLayer>,
    hidden: usize,
    default_context: [usize; 2],
}

impl Subsampling {
    fn load(file: &SafetensorsFile, config: &NemotronAsrConfig) -> Result<Self> {
        let channels = config.subsampling_channels;
        let first = load_conv2d(file, "encoder.pre_encode.conv.0", channels, 1, 3, 1)?;
        let stages = (1..config.subsampling_factor.ilog2() as usize)
            .map(|stage| {
                let depthwise_index = 2 + 3 * (stage - 1);
                let pointwise_index = depthwise_index + 1;
                Ok((
                    load_conv2d(
                        file,
                        &format!("encoder.pre_encode.conv.{depthwise_index}"),
                        channels,
                        channels,
                        3,
                        channels,
                    )?,
                    load_conv2d(
                        file,
                        &format!("encoder.pre_encode.conv.{pointwise_index}"),
                        channels,
                        channels,
                        1,
                        1,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut mel_width = config.num_mels;
        for _ in 0..config.subsampling_factor.ilog2() {
            mel_width = mel_width / 2 + 1;
        }
        let output = load_linear(
            file,
            "encoder.pre_encode.out",
            config.encoder_hidden,
            channels * mel_width,
        )?;
        Ok(Self {
            first,
            stages,
            output,
            channels,
            input_mels: config.num_mels,
        })
    }

    fn forward(&self, features: &[f32], frames: usize) -> Result<(Vec<f32>, usize)> {
        if frames == 0 || features.len() != frames * self.input_mels {
            return Err(SpeechError::Audio(
                "Nemotron mel input must be nonempty [frame, mel]".into(),
            ));
        }
        let mut x = features.to_vec(); // [1, time, mel], channel-major
        let mut time = frames;
        let mut freq = self.input_mels;
        let (padded, padded_time, padded_freq) = pad_causal_2d(&x, 1, time, freq);
        x = self.first.apply(&padded, padded_time, padded_freq, 2);
        time = time / 2 + 1;
        freq = freq / 2 + 1;
        relu(&mut x);
        for (depthwise, pointwise) in &self.stages {
            let (padded, padded_time, padded_freq) = pad_causal_2d(&x, self.channels, time, freq);
            x = depthwise.apply(&padded, padded_time, padded_freq, 2);
            time = time / 2 + 1;
            freq = freq / 2 + 1;
            x = pointwise.apply(&x, time, freq, 1);
            relu(&mut x);
        }
        let mut flat = vec![0.0; time * self.channels * freq];
        for t in 0..time {
            for channel in 0..self.channels {
                let source = channel * time * freq + t * freq;
                let target = t * self.channels * freq + channel * freq;
                flat[target..target + freq].copy_from_slice(&x[source..source + freq]);
            }
        }
        let encoded = self.output.forward(&flat, time);
        Ok((encoded, time))
    }
}

impl Conv2d {
    fn apply(&self, input: &[f32], height: usize, width: usize, stride: usize) -> Vec<f32> {
        ops::conv2d(
            input,
            &self.weight,
            Some(&self.bias),
            self.in_channels,
            self.out_channels,
            height,
            width,
            self.kernel,
            self.kernel,
            stride,
            0,
            self.groups,
        )
    }
}

impl RelPosAttention {
    fn load(file: &SafetensorsFile, prefix: &str, hidden: usize, heads: usize) -> Result<Self> {
        let linear =
            |name: &str| load_bias_free_linear(file, &format!("{prefix}.{name}"), hidden, hidden);
        let head_dim = hidden / heads;
        Ok(Self {
            q: linear("linear_q")?,
            k: linear("linear_k")?,
            v: linear("linear_v")?,
            out: linear("linear_out")?,
            pos: load_bias_free_linear(file, &format!("{prefix}.linear_pos"), hidden, hidden)?,
            bias_u: load_tensor(file, &format!("{prefix}.pos_bias_u"), &[heads, head_dim])?,
            bias_v: load_tensor(file, &format!("{prefix}.pos_bias_v"), &[heads, head_dim])?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        position: &[f32],
        time: usize,
        hidden: usize,
        heads: usize,
        context: [usize; 2],
    ) -> Vec<f32> {
        let head_dim = hidden / heads;
        let pos_len = 2 * time - 1;
        let q = self.q.forward(input, time);
        let k = self.k.forward(input, time);
        let v = self.v.forward(input, time);
        let p = self.pos.forward(position, pos_len);
        let k_heads = split_heads(&k, time, heads, head_dim);
        let v_heads = split_heads(&v, time, heads, head_dim);
        let p_heads = split_heads(&p, pos_len, heads, head_dim);
        let chunk_size = context[1] + 1;
        let left_chunks = context[0] / chunk_size;
        let mut attended = vec![0.0; heads * time * head_dim];
        for head in 0..heads {
            let k_head = &k_heads[head * time * head_dim..(head + 1) * time * head_dim];
            let v_head = &v_heads[head * time * head_dim..(head + 1) * time * head_dim];
            let p_head = &p_heads[head * pos_len * head_dim..(head + 1) * pos_len * head_dim];
            let mut scores = Vec::with_capacity(((left_chunks + 1) * chunk_size).min(time));
            for query in 0..time {
                let q_u: Vec<f32> = (0..head_dim)
                    .map(|dim| {
                        q[query * hidden + head * head_dim + dim]
                            + self.bias_u[head * head_dim + dim]
                    })
                    .collect();
                let q_v: Vec<f32> = (0..head_dim)
                    .map(|dim| {
                        q[query * hidden + head * head_dim + dim]
                            + self.bias_v[head * head_dim + dim]
                    })
                    .collect();
                let query_chunk = query / chunk_size;
                let first = query_chunk.saturating_sub(left_chunks) * chunk_size;
                let end = ((query_chunk + 1) * chunk_size).min(time);
                scores.clear();
                for key in first..end {
                    let key_start = key * head_dim;
                    let relative_start = (time - 1 + key - query) * head_dim;
                    scores.push(scaled_score(
                        dot(&q_u, &k_head[key_start..key_start + head_dim]),
                        dot(&q_v, &p_head[relative_start..relative_start + head_dim]),
                        head_dim,
                    ));
                }
                ops::softmax_row(&mut scores);
                let dst = head * time * head_dim + query * head_dim;
                for (score, key) in scores.iter().zip(first..end) {
                    for dim in 0..head_dim {
                        attended[dst + dim] += score * v_head[key * head_dim + dim];
                    }
                }
            }
        }
        let merged = merge_heads(&attended, time, heads, head_dim);
        self.out.forward(&merged, time)
    }
}

fn scaled_score(content: f32, position: f32, head_dim: usize) -> f32 {
    (content + position) / (head_dim as f32).sqrt()
}

impl FeedForward {
    fn load(file: &SafetensorsFile, prefix: &str, hidden: usize, expanded: usize) -> Result<Self> {
        Ok(Self {
            input: load_bias_free_linear(file, &format!("{prefix}.linear1"), expanded, hidden)?,
            output: load_bias_free_linear(file, &format!("{prefix}.linear2"), hidden, expanded)?,
        })
    }

    fn forward(&self, input: &[f32], rows: usize, hidden: usize, expanded: usize) -> Vec<f32> {
        let mut value = ops::linear(input, &self.input.weight, None, rows, hidden, expanded);
        silu(&mut value);
        ops::linear(&value, &self.output.weight, None, rows, expanded, hidden)
    }
}

impl ConformerConvolution {
    fn load(file: &SafetensorsFile, prefix: &str, channels: usize, kernel: usize) -> Result<Self> {
        require_absent(file, &format!("{prefix}.pointwise_conv1.bias"))?;
        require_absent(file, &format!("{prefix}.depthwise_conv.bias"))?;
        require_absent(file, &format!("{prefix}.pointwise_conv2.bias"))?;
        Ok(Self {
            pointwise_in: load_conv1d(
                file,
                &format!("{prefix}.pointwise_conv1"),
                channels * 2,
                channels,
                1,
                1,
            )?,
            depthwise: load_conv1d(
                file,
                &format!("{prefix}.depthwise_conv"),
                channels,
                channels,
                kernel,
                channels,
            )?,
            depthwise_bias: None,
            norm: load_norm(file, &format!("{prefix}.batch_norm"), channels)?,
            pointwise_out: load_conv1d(
                file,
                &format!("{prefix}.pointwise_conv2"),
                channels,
                channels,
                1,
                1,
            )?,
            channels,
            kernel,
        })
    }

    fn forward(&self, input: &[f32], time: usize) -> Vec<f32> {
        let channels = self.channels;
        let mut channel_major = transpose_time_channels(input, time, channels);
        let pointwise = ops::conv1d(
            &channel_major,
            &self.pointwise_in,
            None,
            channels,
            channels * 2,
            1,
            1,
            0,
            1,
            1,
        );
        let mut gated = vec![0.0; channels * time];
        for channel in 0..channels {
            for frame in 0..time {
                let a = pointwise[channel * time + frame];
                let b = pointwise[(channels + channel) * time + frame];
                gated[channel * time + frame] = a / (1.0 + (-b).exp());
            }
        }
        channel_major = ops::pad_left(&gated, channels, self.kernel - 1);
        let mut depthwise = ops::conv1d(
            &channel_major,
            &self.depthwise,
            self.depthwise_bias.as_deref(),
            channels,
            channels,
            self.kernel,
            1,
            0,
            1,
            channels,
        );
        let mut norm_input = transpose_channels_time(&depthwise, time, channels);
        self.norm.apply(&mut norm_input, time);
        silu(&mut norm_input);
        depthwise = transpose_time_channels(&norm_input, time, channels);
        let pointwise = ops::conv1d(
            &depthwise,
            &self.pointwise_out,
            None,
            channels,
            channels,
            1,
            1,
            0,
            1,
            1,
        );
        transpose_channels_time(&pointwise, time, channels)
    }
}

impl ConformerLayer {
    fn load(
        file: &SafetensorsFile,
        layer: usize,
        hidden: usize,
        heads: usize,
        expansion: usize,
        conv_kernel: usize,
    ) -> Result<Self> {
        let prefix = format!("encoder.layers.{layer}");
        let expanded = hidden * expansion;
        Ok(Self {
            norm_ff1: load_norm(file, &format!("{prefix}.norm_feed_forward1"), hidden)?,
            ff1: FeedForward::load(file, &format!("{prefix}.feed_forward1"), hidden, expanded)?,
            norm_attn: load_norm(file, &format!("{prefix}.norm_self_att"), hidden)?,
            attention: RelPosAttention::load(file, &format!("{prefix}.self_attn"), hidden, heads)?,
            norm_conv: load_norm(file, &format!("{prefix}.norm_conv"), hidden)?,
            convolution: ConformerConvolution::load(
                file,
                &format!("{prefix}.conv"),
                hidden,
                conv_kernel,
            )?,
            norm_ff2: load_norm(file, &format!("{prefix}.norm_feed_forward2"), hidden)?,
            ff2: FeedForward::load(file, &format!("{prefix}.feed_forward2"), hidden, expanded)?,
            norm_out: load_norm(file, &format!("{prefix}.norm_out"), hidden)?,
            hidden,
            heads,
            expansion,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        position: &[f32],
        time: usize,
        context: [usize; 2],
    ) -> Vec<f32> {
        let mut residual = input.to_vec();
        let mut normalized = input.to_vec();
        self.norm_ff1.apply(&mut normalized, time);
        let ff1 = self
            .ff1
            .forward(&normalized, time, self.hidden, self.hidden * self.expansion);
        add_scaled(&mut residual, &ff1, 0.5);

        normalized.clone_from(&residual);
        self.norm_attn.apply(&mut normalized, time);
        let attention = self.attention.forward(
            &normalized,
            position,
            time,
            self.hidden,
            self.heads,
            context,
        );
        add_scaled(&mut residual, &attention, 1.0);

        normalized.clone_from(&residual);
        self.norm_conv.apply(&mut normalized, time);
        let convolution = self.convolution.forward(&normalized, time);
        add_scaled(&mut residual, &convolution, 1.0);

        normalized.clone_from(&residual);
        self.norm_ff2.apply(&mut normalized, time);
        let ff2 = self
            .ff2
            .forward(&normalized, time, self.hidden, self.hidden * self.expansion);
        add_scaled(&mut residual, &ff2, 0.5);
        self.norm_out.apply(&mut residual, time);
        residual
    }
}

impl NemotronEncoder {
    pub(crate) fn load(file: &SafetensorsFile, config: &NemotronAsrConfig) -> Result<Self> {
        let subsampling = Subsampling::load(file, config)?;
        let layers = (0..config.encoder_layers)
            .map(|index| {
                ConformerLayer::load(
                    file,
                    index,
                    config.encoder_hidden,
                    config.encoder_heads,
                    config.encoder_expansion,
                    config.conv_kernel,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling,
            layers,
            hidden: config.encoder_hidden,
            default_context: config.default_attention_context,
        })
    }

    pub(crate) fn encode(
        &self,
        mel: &[f32],
        frames: usize,
        mels: usize,
    ) -> Result<(Vec<f32>, usize)> {
        if mels != self.subsampling.input_mels {
            return Err(SpeechError::Audio(format!(
                "Nemotron expects {} mel bands, received {mels}",
                self.subsampling.input_mels
            )));
        }
        let (mut encoded, time) = self.subsampling.forward(mel, frames)?;
        let position = relative_position_encoding(time, self.hidden);
        for layer in &self.layers {
            encoded = layer.forward(&encoded, &position, time, self.default_context);
        }
        Ok((encoded, time))
    }
}

// =============================================================================
// Cache-aware streaming (NeMo `ConformerEncoder.cache_aware_stream_step`)
// =============================================================================
//
// One step encodes a chunk of pre-encoded rows against per-layer caches, as
// NeMo Speech 218d0b4 does for `att_context_style = chunked_limited`:
// - attention keys and values are `[cache (attn_frames rows) ; chunk]`, where
//   the cache holds the last `attn_frames` `norm_self_att` outputs and only
//   its last `attn_len` rows are valid (NeMo masks the rest);
// - the relative position of chunk query `a` to key `b` is
//   `attn_frames + a - b`, read from one `linear_pos` table per layer;
// - the depthwise conv runs on `[cache (kernel - 1 frames) ; GLU output]`
//   with padded frames zeroed, and keeps the last `kernel - 1` frames;
// - `attn_len` grows by the chunk's row count, capped at `attn_frames`.
// The offline path above is unchanged; the streaming methods only add.

/// Streaming state for one stream (NeMo `cache_last_channel`,
/// `cache_last_time`, `cache_last_channel_len`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EncoderStreamCache {
    /// Per layer `[attn_frames, hidden]`, oldest first.
    pub(crate) attn: Vec<Vec<f32>>,
    /// Valid trailing rows of each `attn` cache.
    pub(crate) attn_len: usize,
    /// Per layer channel-major `[hidden, kernel - 1]`.
    pub(crate) conv: Vec<Vec<f32>>,
}

/// Chunk geometry plus the per-layer relative-position projections, which
/// are the same for every stream and every step.
pub(crate) struct StreamGeometry {
    /// Attention cache length (`att_context_size[0]`).
    pub(crate) attn_frames: usize,
    /// Rows per chunk (`att_context_size[1] + 1`).
    pub(crate) chunk: usize,
    /// Per layer `linear_pos` of the sinusoid table for relative positions
    /// `attn_frames + chunk - 1` down to `-(attn_frames + chunk - 1)`.
    pos: Vec<Vec<f32>>,
}

/// One Conformer layer's weights, borrowed for a device executor of
/// [`NemotronEncoder::stream_layers`]. Linear weights are `[out, in]`;
/// norms are `(weight, bias)`.
pub struct StreamLayerWeights<'a> {
    pub norm_ff1: (&'a [f32], Option<&'a [f32]>),
    pub ff1_in: &'a [f32],
    pub ff1_out: &'a [f32],
    pub norm_attn: (&'a [f32], Option<&'a [f32]>),
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub out: &'a [f32],
    /// `linear_pos` of the relative-position table, `[2 * window - 1,
    /// hidden]` with `window = attn_frames + chunk`.
    pub pos: &'a [f32],
    /// `[heads, head_dim]`.
    pub bias_u: &'a [f32],
    pub bias_v: &'a [f32],
    pub norm_conv: (&'a [f32], Option<&'a [f32]>),
    /// Pointwise conv in, `[2 * hidden, hidden]`.
    pub pointwise_in: &'a [f32],
    /// Depthwise conv, `[hidden, kernel]`.
    pub depthwise: &'a [f32],
    pub depthwise_bias: Option<&'a [f32]>,
    pub conv_norm: (&'a [f32], Option<&'a [f32]>),
    /// Pointwise conv out, `[hidden, hidden]`.
    pub pointwise_out: &'a [f32],
    pub norm_ff2: (&'a [f32], Option<&'a [f32]>),
    pub ff2_in: &'a [f32],
    pub ff2_out: &'a [f32],
    pub norm_out: (&'a [f32], Option<&'a [f32]>),
}

/// The streaming encoder's geometry and weights, borrowed for a device
/// executor that reproduces `stream_layers`. Every norm uses `eps`.
pub struct StreamEncoderWeights<'a> {
    pub hidden: usize,
    pub heads: usize,
    pub ff_hidden: usize,
    pub conv_kernel: usize,
    pub attn_frames: usize,
    pub chunk: usize,
    pub eps: f32,
    pub layers: Vec<StreamLayerWeights<'a>>,
}

fn norm_view(norm: &LayerNorm) -> (&[f32], Option<&[f32]>) {
    (&norm.weight, norm.bias.as_deref())
}

impl NemotronEncoder {
    /// The streaming weights for `geometry`, for a device executor.
    pub(crate) fn stream_weights<'a>(
        &'a self,
        geometry: &'a StreamGeometry,
    ) -> StreamEncoderWeights<'a> {
        let first = self.layers.first();
        StreamEncoderWeights {
            hidden: self.hidden,
            heads: first.map_or(0, |l| l.heads),
            ff_hidden: first.map_or(0, |l| l.hidden * l.expansion),
            conv_kernel: first.map_or(0, |l| l.convolution.kernel),
            attn_frames: geometry.attn_frames,
            chunk: geometry.chunk,
            eps: LN_EPS,
            layers: self
                .layers
                .iter()
                .zip(&geometry.pos)
                .map(|(l, pos)| StreamLayerWeights {
                    norm_ff1: norm_view(&l.norm_ff1),
                    ff1_in: &l.ff1.input.weight,
                    ff1_out: &l.ff1.output.weight,
                    norm_attn: norm_view(&l.norm_attn),
                    q: &l.attention.q.weight,
                    k: &l.attention.k.weight,
                    v: &l.attention.v.weight,
                    out: &l.attention.out.weight,
                    pos,
                    bias_u: &l.attention.bias_u,
                    bias_v: &l.attention.bias_v,
                    norm_conv: norm_view(&l.norm_conv),
                    pointwise_in: &l.convolution.pointwise_in,
                    depthwise: &l.convolution.depthwise,
                    depthwise_bias: l.convolution.depthwise_bias.as_deref(),
                    conv_norm: norm_view(&l.convolution.norm),
                    pointwise_out: &l.convolution.pointwise_out,
                    norm_ff2: norm_view(&l.norm_ff2),
                    ff2_in: &l.ff2.input.weight,
                    ff2_out: &l.ff2.output.weight,
                    norm_out: norm_view(&l.norm_out),
                })
                .collect(),
        }
    }

    /// Streaming geometry for `context = [left, right]` (chunked limited).
    /// Chunks must align with NeMo's chunk index over `[cache ; chunk]`,
    /// which holds when the left context is a whole number of chunks.
    pub(crate) fn stream_geometry(&self, context: [usize; 2]) -> Result<StreamGeometry> {
        let chunk = context[1] + 1;
        let attn_frames = context[0];
        if attn_frames == 0 || attn_frames % chunk != 0 {
            return Err(SpeechError::Unsupported {
                why: format!(
                    "cache-aware streaming needs a left context that is a multiple of the \
                     chunk ({chunk} rows); got {context:?}"
                ),
            });
        }
        let window = attn_frames + chunk;
        let position = relative_position_encoding(window, self.hidden);
        let pos = self
            .layers
            .iter()
            .map(|layer| layer.attention.pos.forward(&position, 2 * window - 1))
            .collect();
        Ok(StreamGeometry {
            attn_frames,
            chunk,
            pos,
        })
    }

    /// Zero caches for a new stream (NeMo `get_initial_cache_state`).
    pub(crate) fn new_stream_cache(&self, geometry: &StreamGeometry) -> EncoderStreamCache {
        let conv_frames = self
            .layers
            .first()
            .map_or(0, |layer| layer.convolution.kernel - 1);
        EncoderStreamCache {
            attn: vec![vec![0.0; geometry.attn_frames * self.hidden]; self.layers.len()],
            attn_len: 0,
            conv: vec![vec![0.0; self.hidden * conv_frames]; self.layers.len()],
        }
    }

    /// Pre-encodes one chunk of frame-major `[frames, mels]` log-mel and
    /// drops the first `drop` rows (NeMo `drop_extra_pre_encoded`).
    /// Returns `(rows, row_count, valid_rows)`, where `valid_rows` follows
    /// NeMo's causal subsampling length rule (`floor(L / 2) + 1` per stage)
    /// for `valid_frames` input frames.
    pub(crate) fn pre_encode(
        &self,
        mel: &[f32],
        frames: usize,
        valid_frames: usize,
        drop: usize,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let (encoded, rows) = self.subsampling.forward(mel, frames)?;
        let stages = self.subsampling.stages.len() + 1;
        let mut valid = valid_frames;
        for _ in 0..stages {
            valid = valid / 2 + 1;
        }
        if drop > rows {
            return Err(SpeechError::Audio(format!(
                "cannot drop {drop} pre-encoded rows from a {rows}-row chunk"
            )));
        }
        let kept = encoded[drop * self.hidden..].to_vec();
        Ok((
            kept,
            rows - drop,
            valid.saturating_sub(drop).min(rows - drop),
        ))
    }

    /// Runs every layer on one chunk of `rows` pre-encoded rows, of which the
    /// first `valid` are real, updating `cache`. Returns the `[rows, hidden]`
    /// layer output; rows at or past `valid` are padding.
    pub(crate) fn stream_layers(
        &self,
        geometry: &StreamGeometry,
        input: &[f32],
        rows: usize,
        valid: usize,
        cache: &mut EncoderStreamCache,
    ) -> Result<Vec<f32>> {
        if rows == 0 || rows > geometry.chunk || valid > rows || input.len() != rows * self.hidden {
            return Err(SpeechError::Audio(format!(
                "streaming chunk of {rows} rows ({valid} valid) does not fit a {}-row chunk",
                geometry.chunk
            )));
        }
        let mut x = input.to_vec();
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(
                &x,
                rows,
                valid,
                &mut cache.attn[index],
                cache.attn_len,
                &mut cache.conv[index],
                &geometry.pos[index],
                geometry,
            );
        }
        cache.attn_len = (cache.attn_len + rows).min(geometry.attn_frames);
        Ok(x)
    }
}

impl ConformerLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        input: &[f32],
        rows: usize,
        valid: usize,
        attn_cache: &mut Vec<f32>,
        attn_len: usize,
        conv_cache: &mut [f32],
        pos: &[f32],
        geometry: &StreamGeometry,
    ) -> Vec<f32> {
        let mut residual = input.to_vec();
        let mut normalized = input.to_vec();
        self.norm_ff1.apply(&mut normalized, rows);
        let ff1 = self
            .ff1
            .forward(&normalized, rows, self.hidden, self.hidden * self.expansion);
        add_scaled(&mut residual, &ff1, 0.5);

        normalized.clone_from(&residual);
        self.norm_attn.apply(&mut normalized, rows);
        let attention = self.attention.forward_cached(
            &normalized,
            rows,
            valid,
            attn_cache,
            attn_len,
            pos,
            geometry,
            self.hidden,
            self.heads,
        );
        add_scaled(&mut residual, &attention, 1.0);

        normalized.clone_from(&residual);
        self.norm_conv.apply(&mut normalized, rows);
        let convolution = self
            .convolution
            .forward_cached(&normalized, rows, valid, conv_cache);
        add_scaled(&mut residual, &convolution, 1.0);

        normalized.clone_from(&residual);
        self.norm_ff2.apply(&mut normalized, rows);
        let ff2 = self
            .ff2
            .forward(&normalized, rows, self.hidden, self.hidden * self.expansion);
        add_scaled(&mut residual, &ff2, 0.5);
        self.norm_out.apply(&mut residual, rows);
        residual
    }
}

impl RelPosAttention {
    /// Chunk attention against `[cache ; chunk]`. Queries at or past `valid`
    /// produce zero rows, as NeMo's fully masked rows do. Replaces `cache`
    /// with the last `attn_frames` rows of `[cache ; input]`.
    #[allow(clippy::too_many_arguments)]
    fn forward_cached(
        &self,
        input: &[f32],
        rows: usize,
        valid: usize,
        cache: &mut Vec<f32>,
        cache_len: usize,
        pos: &[f32],
        geometry: &StreamGeometry,
        hidden: usize,
        heads: usize,
    ) -> Vec<f32> {
        let head_dim = hidden / heads;
        let attn_frames = geometry.attn_frames;
        let window = attn_frames + geometry.chunk;
        let keys = attn_frames + rows;
        let mut kv_input = Vec::with_capacity(keys * hidden);
        kv_input.extend_from_slice(cache);
        kv_input.extend_from_slice(input);

        let q = self.q.forward(input, rows);
        let k = self.k.forward(&kv_input, keys);
        let v = self.v.forward(&kv_input, keys);
        let k_heads = split_heads(&k, keys, heads, head_dim);
        let v_heads = split_heads(&v, keys, heads, head_dim);
        let p_heads = split_heads(pos, 2 * window - 1, heads, head_dim);
        // Valid keys: the trailing `cache_len` cache rows and the valid chunk rows.
        let first = attn_frames - cache_len.min(attn_frames);
        let end = attn_frames + valid;
        let mut attended = vec![0.0; heads * rows * head_dim];
        let mut scores = Vec::with_capacity(end - first);
        for head in 0..heads {
            let k_head = &k_heads[head * keys * head_dim..(head + 1) * keys * head_dim];
            let v_head = &v_heads[head * keys * head_dim..(head + 1) * keys * head_dim];
            let p_head = &p_heads
                [head * (2 * window - 1) * head_dim..(head + 1) * (2 * window - 1) * head_dim];
            for query in 0..valid {
                let q_u: Vec<f32> = (0..head_dim)
                    .map(|dim| {
                        q[query * hidden + head * head_dim + dim]
                            + self.bias_u[head * head_dim + dim]
                    })
                    .collect();
                let q_v: Vec<f32> = (0..head_dim)
                    .map(|dim| {
                        q[query * hidden + head * head_dim + dim]
                            + self.bias_v[head * head_dim + dim]
                    })
                    .collect();
                scores.clear();
                for key in first..end {
                    // Table row of relative position `attn_frames + query - key`.
                    let row = window - 1 + key - attn_frames - query;
                    let key_start = key * head_dim;
                    let pos_start = row * head_dim;
                    scores.push(scaled_score(
                        dot(&q_u, &k_head[key_start..key_start + head_dim]),
                        dot(&q_v, &p_head[pos_start..pos_start + head_dim]),
                        head_dim,
                    ));
                }
                ops::softmax_row(&mut scores);
                let dst = head * rows * head_dim + query * head_dim;
                for (score, key) in scores.iter().zip(first..end) {
                    for dim in 0..head_dim {
                        attended[dst + dim] += score * v_head[key * head_dim + dim];
                    }
                }
            }
        }
        let start = kv_input.len() - attn_frames * hidden;
        cache.clear();
        cache.extend_from_slice(&kv_input[start..]);
        let merged = merge_heads(&attended, rows, heads, head_dim);
        self.out.forward(&merged, rows)
    }
}

impl ConformerConvolution {
    /// Causal conv module on one chunk with its left-context cache. Frames
    /// at or past `valid` are zeroed before the depthwise conv (NeMo's
    /// `pad_mask`). Replaces `cache` with the last `kernel - 1` frames of
    /// `[cache ; GLU output]`.
    fn forward_cached(
        &self,
        input: &[f32],
        rows: usize,
        valid: usize,
        cache: &mut [f32],
    ) -> Vec<f32> {
        let channels = self.channels;
        let context = self.kernel - 1;
        let channel_major = transpose_time_channels(input, rows, channels);
        let pointwise = ops::conv1d(
            &channel_major,
            &self.pointwise_in,
            None,
            channels,
            channels * 2,
            1,
            1,
            0,
            1,
            1,
        );
        let width = context + rows;
        let mut extended = vec![0.0; channels * width];
        for channel in 0..channels {
            extended[channel * width..channel * width + context]
                .copy_from_slice(&cache[channel * context..(channel + 1) * context]);
            for frame in 0..valid {
                let a = pointwise[channel * rows + frame];
                let b = pointwise[(channels + channel) * rows + frame];
                extended[channel * width + context + frame] = a / (1.0 + (-b).exp());
            }
        }
        for channel in 0..channels {
            cache[channel * context..(channel + 1) * context]
                .copy_from_slice(&extended[channel * width + rows..(channel + 1) * width]);
        }
        let depthwise = ops::conv1d(
            &extended,
            &self.depthwise,
            self.depthwise_bias.as_deref(),
            channels,
            channels,
            self.kernel,
            1,
            0,
            1,
            channels,
        );
        let mut norm_input = transpose_channels_time(&depthwise, rows, channels);
        self.norm.apply(&mut norm_input, rows);
        silu(&mut norm_input);
        let activated = transpose_time_channels(&norm_input, rows, channels);
        let pointwise = ops::conv1d(
            &activated,
            &self.pointwise_out,
            None,
            channels,
            channels,
            1,
            1,
            0,
            1,
            1,
        );
        transpose_channels_time(&pointwise, rows, channels)
    }
}

fn pad_causal_2d(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> (Vec<f32>, usize, usize) {
    let pad_before = 2;
    let pad_after = 1;
    let padded_height = height + pad_before + pad_after;
    let padded_width = width + pad_before + pad_after;
    let mut output = vec![0.0; channels * padded_height * padded_width];
    for channel in 0..channels {
        for row in 0..height {
            let src = channel * height * width + row * width;
            let dst = channel * padded_height * padded_width
                + (row + pad_before) * padded_width
                + pad_before;
            output[dst..dst + width].copy_from_slice(&input[src..src + width]);
        }
    }
    (output, padded_height, padded_width)
}

fn relative_position_encoding(time: usize, hidden: usize) -> Vec<f32> {
    let positions = 2 * time - 1;
    let scale = -(10_000f64.ln() / hidden as f64) as f32;
    let div: Vec<f32> = (0..hidden / 2)
        .map(|index| ((2 * index) as f32 * scale).exp())
        .collect();
    let mut output = vec![0.0; positions * hidden];
    for row in 0..positions {
        let position = (time as isize - 1 - row as isize) as f32;
        for (index, &frequency) in div.iter().enumerate() {
            let angle = position * frequency;
            output[row * hidden + index * 2] = angle.sin();
            output[row * hidden + index * 2 + 1] = angle.cos();
        }
    }
    output
}

#[cfg(test)]
fn rel_shift(input: &[f32], heads: usize, time: usize, pos_len: usize) -> Vec<f32> {
    let mut output = vec![0.0; heads * time * time];
    for head in 0..heads {
        for query in 0..time {
            for key in 0..time {
                let source_pos = time - 1 + key - query;
                output[head * time * time + query * time + key] =
                    input[head * time * pos_len + query * pos_len + source_pos];
            }
        }
    }
    output
}

fn split_heads(input: &[f32], time: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let hidden = heads * head_dim;
    let mut output = vec![0.0; input.len()];
    for frame in 0..time {
        for head in 0..heads {
            let src = frame * hidden + head * head_dim;
            let dst = head * time * head_dim + frame * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
    output
}

fn merge_heads(input: &[f32], time: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let hidden = heads * head_dim;
    let mut output = vec![0.0; input.len()];
    for frame in 0..time {
        for head in 0..heads {
            let src = head * time * head_dim + frame * head_dim;
            let dst = frame * hidden + head * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
    output
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn transpose_time_channels(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for t in 0..time {
        for channel in 0..channels {
            output[channel * time + t] = input[t * channels + channel];
        }
    }
    output
}

fn transpose_channels_time(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for t in 0..time {
            output[t * channels + channel] = input[channel * time + t];
        }
    }
    output
}

fn add_scaled(target: &mut [f32], values: &[f32], scale: f32) {
    for (target, value) in target.iter_mut().zip(values) {
        *target += value * scale;
    }
}

fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn silu(values: &mut [f32]) {
    for value in values {
        *value *= 1.0 / (1.0 + (-*value).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::{dot, rel_shift, scaled_score, split_heads, Linear, RelPosAttention};
    use super::{
        ConformerConvolution, ConformerLayer, Conv2d, FeedForward, NemotronEncoder, Subsampling,
    };
    use crate::ops;

    #[test]
    fn relative_shift_maps_each_query_to_key_relative_position() {
        let input: Vec<f32> = (0..15).map(|value| value as f32).collect();
        assert_eq!(
            rel_shift(&input, 1, 3, 5),
            vec![2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn relative_attention_uses_inverse_sqrt_head_scaling() {
        assert_eq!(scaled_score(8.0, 4.0, 16), 3.0);
    }

    #[test]
    fn bounded_attention_matches_dense_masked_reference() {
        let time = 6;
        let hidden = 4;
        let heads = 2;
        let head_dim = 2;
        let pos_len = 2 * time - 1;
        let identity = || {
            Linear::new(
                (0..hidden * hidden)
                    .map(|index| {
                        if index / hidden == index % hidden {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                None,
                hidden,
                hidden,
            )
        };
        let attention = RelPosAttention {
            q: identity(),
            k: identity(),
            v: identity(),
            out: identity(),
            pos: identity(),
            bias_u: vec![0.1, -0.2, 0.3, 0.05],
            bias_v: vec![-0.15, 0.25, 0.1, -0.3],
        };
        let input: Vec<f32> = (0..time * hidden)
            .map(|index| ((index * 7 + 2) % 17) as f32 * 0.07 - 0.4)
            .collect();
        let position: Vec<f32> = (0..pos_len * hidden)
            .map(|index| ((index * 5 + 3) % 23) as f32 * 0.03 - 0.2)
            .collect();
        let k_heads = split_heads(&input, time, heads, head_dim);
        let v_heads = split_heads(&input, time, heads, head_dim);
        let p_heads = split_heads(&position, pos_len, heads, head_dim);
        for context in [[2, 1], [4, 0]] {
            let actual = attention.forward(&input, &position, time, hidden, heads, context);
            let mut ac = vec![0.0; heads * time * time];
            let mut bd = vec![0.0; heads * time * pos_len];
            for head in 0..heads {
                for query in 0..time {
                    let start = query * hidden + head * head_dim;
                    let q_u: Vec<f32> = (0..head_dim)
                        .map(|dim| input[start + dim] + attention.bias_u[head * head_dim + dim])
                        .collect();
                    let q_v: Vec<f32> = (0..head_dim)
                        .map(|dim| input[start + dim] + attention.bias_v[head * head_dim + dim])
                        .collect();
                    for key in 0..time {
                        let offset = (head * time + key) * head_dim;
                        ac[(head * time + query) * time + key] =
                            dot(&q_u, &k_heads[offset..offset + head_dim]);
                    }
                    for index in 0..pos_len {
                        let offset = (head * pos_len + index) * head_dim;
                        bd[(head * time + query) * pos_len + index] =
                            dot(&q_v, &p_heads[offset..offset + head_dim]);
                    }
                }
            }
            let bd = rel_shift(&bd, heads, time, pos_len);
            let chunk = context[1] + 1;
            let left_chunks = context[0] / chunk;
            let mut attended = vec![0.0; heads * time * head_dim];
            for head in 0..heads {
                for query in 0..time {
                    let query_chunk = query / chunk;
                    let mut scores = vec![f32::NEG_INFINITY; time];
                    // The index also feeds offsets or math beyond this buffer, so an iterator is not simpler.
                    #[allow(clippy::needless_range_loop)]
                    for key in 0..time {
                        let key_chunk = key / chunk;
                        if query_chunk >= key_chunk && query_chunk - key_chunk <= left_chunks {
                            let index = (head * time + query) * time + key;
                            scores[key] = scaled_score(ac[index], bd[index], head_dim);
                        }
                    }
                    ops::softmax_row(&mut scores);
                    for key in 0..time {
                        for dim in 0..head_dim {
                            attended[(head * time + query) * head_dim + dim] +=
                                scores[key] * v_heads[(head * time + key) * head_dim + dim];
                        }
                    }
                }
            }
            let expected = super::merge_heads(&attended, time, heads, head_dim);
            for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-5,
                    "context {context:?}, index {index}: bounded {a}, dense {b}"
                );
            }
        }
    }

    /// Deterministic xorshift values in `[-scale, scale]`.
    pub(super) struct Rng(pub(super) u64);

    impl Rng {
        pub(super) fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 8) % 20001) as f32 / 10000.0 * scale - scale
                })
                .collect()
        }

        fn linear(&mut self, out: usize, input: usize) -> Linear {
            Linear::new(
                self.vec(out * input, 1.5 / (input as f32).sqrt()),
                None,
                input,
                out,
            )
        }

        fn norm(&mut self, width: usize) -> super::LayerNorm {
            let weight = self.vec(width, 0.2).iter().map(|w| 1.0 + w).collect();
            super::LayerNorm::new(weight, Some(self.vec(width, 0.1)), super::LN_EPS)
        }

        fn conv2d(&mut self, out: usize, input: usize, kernel: usize, groups: usize) -> Conv2d {
            Conv2d {
                weight: self.vec(out * (input / groups) * kernel * kernel, 0.5),
                bias: self.vec(out, 0.1),
                in_channels: input,
                out_channels: out,
                kernel,
                groups,
            }
        }
    }

    /// Tiny random Nemotron encoder: 16 mels, x8 causal subsampling with 4
    /// channels, 2 layers of width 8 (2 heads), FF x2, depthwise kernel 3.
    pub(super) fn tiny_encoder(seed: u64, context: [usize; 2]) -> NemotronEncoder {
        let (mels, channels, hidden, heads, expansion, kernel) = (16, 4, 8, 2, 2, 3);
        let mut rng = Rng(seed);
        let mut freq = mels;
        for _ in 0..3 {
            freq = freq / 2 + 1;
        }
        let subsampling = Subsampling {
            first: rng.conv2d(channels, 1, 3, 1),
            stages: (0..2)
                .map(|_| {
                    (
                        rng.conv2d(channels, channels, 3, channels),
                        rng.conv2d(channels, channels, 1, 1),
                    )
                })
                .collect(),
            output: rng.linear(hidden, channels * freq),
            channels,
            input_mels: mels,
        };
        let layers = (0..2)
            .map(|_| ConformerLayer {
                norm_ff1: rng.norm(hidden),
                ff1: FeedForward {
                    input: rng.linear(hidden * expansion, hidden),
                    output: rng.linear(hidden, hidden * expansion),
                },
                norm_attn: rng.norm(hidden),
                attention: RelPosAttention {
                    q: rng.linear(hidden, hidden),
                    k: rng.linear(hidden, hidden),
                    v: rng.linear(hidden, hidden),
                    out: rng.linear(hidden, hidden),
                    pos: rng.linear(hidden, hidden),
                    bias_u: rng.vec(hidden, 0.3),
                    bias_v: rng.vec(hidden, 0.3),
                },
                norm_conv: rng.norm(hidden),
                convolution: ConformerConvolution {
                    pointwise_in: rng.vec(2 * hidden * hidden, 0.5),
                    depthwise: rng.vec(hidden * kernel, 0.5),
                    depthwise_bias: None,
                    norm: rng.norm(hidden),
                    pointwise_out: rng.vec(hidden * hidden, 0.5),
                    channels: hidden,
                    kernel,
                },
                norm_ff2: rng.norm(hidden),
                ff2: FeedForward {
                    input: rng.linear(hidden * expansion, hidden),
                    output: rng.linear(hidden, hidden * expansion),
                },
                norm_out: rng.norm(hidden),
                hidden,
                heads,
                expansion,
            })
            .collect();
        NemotronEncoder {
            subsampling,
            layers,
            hidden,
            default_context: context,
        }
    }

    pub(super) fn fnv1a_bits(values: &[f32]) -> u64 {
        values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
            v.to_bits()
                .to_le_bytes()
                .iter()
                .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
        })
    }

    /// Captured on HEAD c4217879 before the cache-aware streaming methods
    /// were added: the offline encode of a tiny random encoder must stay
    /// bit-for-bit unchanged.
    const OFFLINE_DIGEST: (usize, u64) = (27, 3_183_675_627_888_117_202);

    /// Streaming the layers chunk by chunk through the caches must equal the
    /// offline chunked-limited forward over the whole sequence: same keys per
    /// query (cache plus chunk, invalid cache rows excluded), same relative
    /// positions, and the causal conv seeing the same left frames. Covers a
    /// rolling cache (7 chunks against an 8-row cache) and a short final chunk.
    #[test]
    fn stream_layers_match_offline_chunked_forward() {
        let context = [8, 3];
        let encoder = tiny_encoder(0x5151_7a7a_0303_1717, context);
        let hidden = encoder.hidden;
        for total in [28usize, 26] {
            let input = Rng(7).vec(total * hidden, 2.0);
            let position = super::relative_position_encoding(total, hidden);
            let mut offline = input.clone();
            for layer in &encoder.layers {
                offline = layer.forward(&offline, &position, total, context);
            }

            let geometry = encoder.stream_geometry(context).unwrap();
            let mut cache = encoder.new_stream_cache(&geometry);
            let mut streamed = Vec::new();
            let mut start = 0;
            while start < total {
                let rows = geometry.chunk.min(total - start);
                let chunk = &input[start * hidden..(start + rows) * hidden];
                streamed.extend(
                    encoder
                        .stream_layers(&geometry, chunk, rows, rows, &mut cache)
                        .unwrap(),
                );
                start += rows;
            }
            assert_eq!(cache.attn_len, geometry.attn_frames);
            for (index, (a, b)) in streamed.iter().zip(&offline).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-6 * b.abs().max(1.0),
                    "total {total}, row {}, dim {}: streamed {a} vs offline {b}",
                    index / hidden,
                    index % hidden
                );
            }
        }
    }

    /// NeMo's causal dw_striding length rule, `floor(L / 2) + 1` per stage,
    /// minus the dropped rows: a 121-frame Frozen E chunk keeps 14 rows and
    /// a 26-frame final chunk keeps 3.
    #[test]
    fn pre_encode_valid_rows_follow_nemo_lengths() {
        let encoder = tiny_encoder(3, [8, 3]);
        for (frames, rows) in [(121usize, 14usize), (26, 3)] {
            let mel = Rng(frames as u64).vec(frames * 16, 1.0);
            let (out, got_rows, valid) = encoder.pre_encode(&mel, frames, frames, 2).unwrap();
            assert_eq!((got_rows, valid), (rows, rows), "{frames} frames");
            assert_eq!(out.len(), rows * encoder.hidden);
        }
        // A short valid length inside a full chunk only shrinks `valid`.
        let mel = Rng(9).vec(121 * 16, 1.0);
        let (_, rows, valid) = encoder.pre_encode(&mel, 121, 26, 2).unwrap();
        assert_eq!((rows, valid), (14, 3));
    }

    #[test]
    fn offline_encoder_digest_is_frozen() {
        let encoder = tiny_encoder(0x9e37_79b9_7f4a_7c15, [8, 3]);
        let frames = 203;
        let mel = Rng(42).vec(frames * 16, 3.0);
        let (encoded, time) = encoder.encode(&mel, frames, 16).unwrap();
        let got = (time, fnv1a_bits(&encoded));
        eprintln!("OFFLINE_DIGEST {got:?}");
        assert_eq!(got, OFFLINE_DIGEST);
    }
}
