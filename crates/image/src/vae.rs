//! VAE convolution and decode pipeline for Z-Image-Turbo.
//!
//! Reconstructs full-resolution images from latent representations:
//! - Latent rescaling: latent / 0.3611 + 0.1159
//! - Conv2D (3x3 with pad 1, and 1x1 shortcuts)
//! - GroupNorm (32 groups, eps 1e-6)
//! - 2x Nearest spatial upsampling
//! - ResnetBlock2D with skip connections
//! - UNetMidBlock with single-head attention
//! - 4 UpDecoderBlock stages (512->512, 512->512, 512->256, 256->128)
//! - Out norm (GroupNorm 32, 128 channels), SiLU, conv_out (128 -> 3)

#![forbid(unsafe_code)]

use std::io::Cursor;

pub const VAE_LATENT_CHANNELS: usize = 16;
pub const VAE_OUT_CHANNELS: usize = 3;
pub const VAE_SCALE_FACTOR: f32 = 0.3611;
pub const VAE_SHIFT_FACTOR: f32 = 0.1159;
pub const GROUP_NORM_GROUPS: usize = 32;
pub const GROUP_NORM_EPS: f32 = 1e-6;

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// 2D Convolution over NCHW tensor (batch=1).
///
/// Weight shape: `[out_channels, in_channels, kh, kw]`.
/// Bias: optional `[out_channels]`.
#[allow(clippy::too_many_arguments)] // Tensor shape is explicit at this portable reference boundary.
pub fn conv2d(
    x: &[f32],
    c_in: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>, String> {
    if c_in == 0 || c_out == 0 || h == 0 || w == 0 || kh == 0 || kw == 0 || stride == 0 {
        return Err("convolution dimensions and stride must be nonzero".to_string());
    }
    if h + 2 * padding < kh || w + 2 * padding < kw {
        return Err("convolution kernel exceeds padded input".to_string());
    }
    if x.len() != c_in * h * w {
        return Err("convolution input buffer length mismatch".to_string());
    }
    if weight.len() != c_out * c_in * kh * kw {
        return Err("convolution weight buffer length mismatch".to_string());
    }
    if let Some(b) = bias {
        if b.len() != c_out {
            return Err("convolution bias buffer length mismatch".to_string());
        }
    }

    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;
    let mut out = vec![0.0f32; c_out * out_h * out_w];

    for co in 0..c_out {
        let b_val = bias.map_or(0.0f32, |b| b[co]);
        let co_w_offset = co * c_in * kh * kw;

        for oh in 0..out_h {
            let ih_base = (oh * stride) as isize - padding as isize;
            for ow in 0..out_w {
                let iw_base = (ow * stride) as isize - padding as isize;
                let mut sum = b_val;

                for ci in 0..c_in {
                    let ci_x_offset = ci * h * w;
                    let ci_w_offset = co_w_offset + ci * kh * kw;

                    for r in 0..kh {
                        let ih = ih_base + r as isize;
                        if ih >= 0 && (ih as usize) < h {
                            let ih_u = ih as usize;
                            for c in 0..kw {
                                let iw = iw_base + c as isize;
                                if iw >= 0 && (iw as usize) < w {
                                    let iw_u = iw as usize;
                                    let val = x[ci_x_offset + ih_u * w + iw_u];
                                    let w_val = weight[ci_w_offset + r * kw + c];
                                    sum += val * w_val;
                                }
                            }
                        }
                    }
                }

                out[co * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }

    Ok(out)
}

/// GroupNorm over CHW tensor with 32 groups.
///
/// Weight and bias length: `channels`.
#[allow(clippy::too_many_arguments)] // The operation exposes its full tensor contract to tests.
pub fn group_norm(
    x: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
    num_groups: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if channels == 0 || h == 0 || w == 0 || num_groups == 0 || channels % num_groups != 0 {
        return Err("GroupNorm channels, spatial dimensions, and groups must be nonzero with channels divisible by groups".to_string());
    }
    if x.len() != channels * h * w || weight.len() != channels || bias.len() != channels {
        return Err("GroupNorm input or affine parameter shape mismatch".to_string());
    }

    let c_per_group = channels / num_groups;
    let hw = h * w;
    let group_elements = c_per_group * hw;
    let mut out = vec![0.0f32; channels * hw];

    for g in 0..num_groups {
        let ch_start = g * c_per_group;
        let ch_end = ch_start + c_per_group;

        // Compute mean
        let mut sum = 0.0f64;
        for ch in ch_start..ch_end {
            let offset = ch * hw;
            for i in 0..hw {
                sum += x[offset + i] as f64;
            }
        }
        let mean = sum / (group_elements as f64);

        // Compute variance
        let mut var_sum = 0.0f64;
        for ch in ch_start..ch_end {
            let offset = ch * hw;
            for i in 0..hw {
                let diff = (x[offset + i] as f64) - mean;
                var_sum += diff * diff;
            }
        }
        let var = var_sum / (group_elements as f64);
        let inv_std = (1.0 / (var + (eps as f64)).sqrt()) as f32;
        let mean_f = mean as f32;

        // Apply scale and shift
        for ch in ch_start..ch_end {
            let offset = ch * hw;
            let gamma = weight[ch];
            let beta = bias[ch];
            for i in 0..hw {
                let normed = (x[offset + i] - mean_f) * inv_std;
                out[offset + i] = normed * gamma + beta;
            }
        }
    }

    Ok(out)
}

/// 2x Nearest-Neighbor spatial upsampling: [C, H, W] -> [C, 2*H, 2*W].
pub fn upsample_nearest_2x(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    assert_eq!(x.len(), c * h * w);
    let out_h = 2 * h;
    let out_w = 2 * w;
    let mut out = vec![0.0f32; c * out_h * out_w];

    for ch in 0..c {
        let in_plane = ch * h * w;
        let out_plane = ch * out_h * out_w;
        for oh in 0..out_h {
            let ih = oh / 2;
            let in_row = in_plane + ih * w;
            let out_row = out_plane + oh * out_w;
            for ow in 0..out_w {
                let iw = ow / 2;
                out[out_row + ow] = x[in_row + iw];
            }
        }
    }

    out
}

/// Convert a decoded `[3, height, width]` float tensor in `[-1, 1]` to
/// interleaved RGB bytes. Non-finite values are rejected before clamping.
pub fn decoded_to_rgb8(decoded: &[f32], height: usize, width: usize) -> Result<Vec<u8>, String> {
    let plane = height
        .checked_mul(width)
        .ok_or_else(|| "RGB dimensions overflow".to_string())?;
    if height == 0 || width == 0 || decoded.len() != VAE_OUT_CHANNELS * plane {
        return Err(
            "decoded tensor must have shape [3, height, width] with nonzero dimensions".to_string(),
        );
    }
    let mut rgb = Vec::with_capacity(decoded.len());
    for pixel in 0..plane {
        for channel in 0..VAE_OUT_CHANNELS {
            let value = decoded[channel * plane + pixel];
            if !value.is_finite() {
                return Err(format!(
                    "decoded tensor contains non-finite value at channel {channel}, pixel {pixel}"
                ));
            }
            rgb.push((((value * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0).round() as u8);
        }
    }
    Ok(rgb)
}

/// Encode already-validated interleaved RGB data as a PNG.
pub fn encode_rgb8_png(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    let width_u32 = u32::try_from(width).map_err(|_| "PNG width exceeds u32".to_string())?;
    let height_u32 = u32::try_from(height).map_err(|_| "PNG height exceeds u32".to_string())?;
    if width == 0 || height == 0 || rgb.len() != width * height * VAE_OUT_CHANNELS {
        return Err("RGB buffer must exactly match nonzero width * height * 3".to_string());
    }
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(Cursor::new(&mut bytes), width_u32, height_u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("write PNG header: {e}"))?;
        writer
            .write_image_data(rgb)
            .map_err(|e| format!("write PNG pixels: {e}"))?;
    }
    Ok(bytes)
}

/// ResnetBlock2D layer weights.
#[derive(Debug, Clone)]
pub struct ResnetBlock2D {
    pub in_channels: usize,
    pub out_channels: usize,
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub conv1_weight: Vec<f32>,
    pub conv1_bias: Vec<f32>,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub conv2_weight: Vec<f32>,
    pub conv2_bias: Vec<f32>,
    pub conv_shortcut_weight: Option<Vec<f32>>,
    pub conv_shortcut_bias: Option<Vec<f32>>,
}

impl ResnetBlock2D {
    pub fn forward(&self, x: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        // 1. norm1
        let n1 = group_norm(
            x,
            self.in_channels,
            h,
            w,
            &self.norm1_weight,
            &self.norm1_bias,
            GROUP_NORM_GROUPS,
            GROUP_NORM_EPS,
        )?;

        // 2. silu
        let mut act1 = Vec::with_capacity(n1.len());
        for &val in &n1 {
            act1.push(silu(val));
        }

        // 3. conv1 (3x3, pad 1)
        let c1 = conv2d(
            &act1,
            self.in_channels,
            h,
            w,
            &self.conv1_weight,
            Some(&self.conv1_bias),
            self.out_channels,
            3,
            3,
            1,
            1,
        )?;

        // 4. norm2
        let n2 = group_norm(
            &c1,
            self.out_channels,
            h,
            w,
            &self.norm2_weight,
            &self.norm2_bias,
            GROUP_NORM_GROUPS,
            GROUP_NORM_EPS,
        )?;

        // 5. silu
        let mut act2 = Vec::with_capacity(n2.len());
        for &val in &n2 {
            act2.push(silu(val));
        }

        // 6. conv2 (3x3, pad 1)
        let c2 = conv2d(
            &act2,
            self.out_channels,
            h,
            w,
            &self.conv2_weight,
            Some(&self.conv2_bias),
            self.out_channels,
            3,
            3,
            1,
            1,
        )?;

        // 7. shortcut + residual add
        let shortcut = match (&self.conv_shortcut_weight, &self.conv_shortcut_bias) {
            (Some(sw), sb) => conv2d(
                x,
                self.in_channels,
                h,
                w,
                sw,
                sb.as_deref(),
                self.out_channels,
                1,
                1,
                1,
                0,
            )?,
            (None, _) => {
                if self.in_channels != self.out_channels {
                    return Err("channel mismatch requires conv_shortcut".to_string());
                }
                x.to_vec()
            }
        };

        let mut out = Vec::with_capacity(c2.len());
        for (sc, cv) in shortcut.iter().zip(c2.iter()) {
            out.push(sc + cv);
        }

        Ok(out)
    }
}

/// UpSampler layer: 2x nearest upsample followed by 3x3 conv.
#[derive(Debug, Clone)]
pub struct UpSampler {
    pub in_channels: usize,
    pub out_channels: usize,
    pub conv_weight: Vec<f32>,
    pub conv_bias: Vec<f32>,
}

impl UpSampler {
    pub fn forward(&self, x: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        let up = upsample_nearest_2x(x, self.in_channels, h, w);
        conv2d(
            &up,
            self.in_channels,
            2 * h,
            2 * w,
            &self.conv_weight,
            Some(&self.conv_bias),
            self.out_channels,
            3,
            3,
            1,
            1,
        )
    }
}

/// Single-head self-attention used inside the VAE mid-block.
#[derive(Debug, Clone)]
pub struct VaeAttention {
    pub channels: usize,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub q_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
}

impl VaeAttention {
    pub fn forward(&self, x: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        let n = h * w;
        let c = self.channels;

        // GroupNorm
        let normed = group_norm(
            x,
            c,
            h,
            w,
            &self.norm_weight,
            &self.norm_bias,
            GROUP_NORM_GROUPS,
            GROUP_NORM_EPS,
        )?;

        // Permute to (N, C) layout for token-wise projections
        // In normed: element at (ch, i) is at normed[ch * n + i]
        let mut tokens = vec![0.0f32; n * c];
        for ch in 0..c {
            let ch_off = ch * n;
            for i in 0..n {
                tokens[i * c + ch] = normed[ch_off + i];
            }
        }

        // Q, K, V projections: Linear(C, C)
        let mut q = vec![0.0f32; n * c];
        let mut k = vec![0.0f32; n * c];
        let mut v = vec![0.0f32; n * c];

        for i in 0..n {
            let tok = &tokens[i * c..(i + 1) * c];
            for oc in 0..c {
                let w_row = &self.q_weight[oc * c..(oc + 1) * c];
                let mut sum_q = self.q_bias[oc];
                let mut sum_k = self.k_bias[oc];
                let mut sum_v = self.v_bias[oc];
                let kw_row = &self.k_weight[oc * c..(oc + 1) * c];
                let vw_row = &self.v_weight[oc * c..(oc + 1) * c];

                for ci in 0..c {
                    sum_q += tok[ci] * w_row[ci];
                    sum_k += tok[ci] * kw_row[ci];
                    sum_v += tok[ci] * vw_row[ci];
                }
                q[i * c + oc] = sum_q;
                k[i * c + oc] = sum_k;
                v[i * c + oc] = sum_v;
            }
        }

        // Row-chunked single-head attention
        let scale = 1.0 / (c as f32).sqrt();
        let mut attn_out = vec![0.0f32; n * c];

        for i in 0..n {
            let q_vec = &q[i * c..(i + 1) * c];

            // Compute dot product with all keys j
            let mut scores = vec![0.0f32; n];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..n {
                let k_vec = &k[j * c..(j + 1) * c];
                let mut dot = 0.0f32;
                for d in 0..c {
                    dot += q_vec[d] * k_vec[d];
                }
                let s = dot * scale;
                scores[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                let e = (*s - max_s).exp();
                *s = e;
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in &mut scores {
                *s *= inv_sum;
            }

            // Accumulate V
            let out_slice = &mut attn_out[i * c..(i + 1) * c];
            for j in 0..n {
                let w = scores[j];
                let v_vec = &v[j * c..(j + 1) * c];
                for d in 0..c {
                    out_slice[d] += w * v_vec[d];
                }
            }
        }

        // Output projection and transpose back to CHW + residual add
        let mut out = vec![0.0f32; c * n];
        for i in 0..n {
            let tok = &attn_out[i * c..(i + 1) * c];
            for oc in 0..c {
                let w_row = &self.out_weight[oc * c..(oc + 1) * c];
                let mut sum = self.out_bias[oc];
                for ci in 0..c {
                    sum += tok[ci] * w_row[ci];
                }
                // Add residual: x[oc * n + i]
                out[oc * n + i] = x[oc * n + i] + sum;
            }
        }

        Ok(out)
    }
}

/// UNetMidBlock: resnet[0] -> attention -> resnet[1].
#[derive(Debug, Clone)]
pub struct UNetMidBlock {
    pub resnet0: ResnetBlock2D,
    pub attention: VaeAttention,
    pub resnet1: ResnetBlock2D,
}

impl UNetMidBlock {
    pub fn forward(&self, x: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        let r0 = self.resnet0.forward(x, h, w)?;
        let att = self.attention.forward(&r0, h, w)?;
        self.resnet1.forward(&att, h, w)
    }
}

/// UpDecoderBlock: sequence of ResnetBlock2D layers + optional UpSampler.
#[derive(Debug, Clone)]
pub struct UpDecoderBlock {
    pub resnets: Vec<ResnetBlock2D>,
    pub upsampler: Option<UpSampler>,
}

impl UpDecoderBlock {
    pub fn forward(
        &self,
        x: &[f32],
        mut h: usize,
        mut w: usize,
    ) -> Result<(Vec<f32>, usize, usize), String> {
        let mut cur = x.to_vec();
        for resnet in &self.resnets {
            cur = resnet.forward(&cur, h, w)?;
        }
        if let Some(up) = &self.upsampler {
            cur = up.forward(&cur, h, w)?;
            h *= 2;
            w *= 2;
        }
        Ok((cur, h, w))
    }
}

/// Full VAE Decoder network.
#[derive(Debug, Clone)]
pub struct VaeDecoder {
    pub conv_in_weight: Vec<f32>,
    pub conv_in_bias: Vec<f32>,
    pub mid_block: UNetMidBlock,
    pub up_blocks: Vec<UpDecoderBlock>,
    pub conv_norm_out_weight: Vec<f32>,
    pub conv_norm_out_bias: Vec<f32>,
    pub conv_out_weight: Vec<f32>,
    pub conv_out_bias: Vec<f32>,
}

impl VaeDecoder {
    /// Load full VAE decoder network from safetensors.
    pub fn from_safetensors(sf: &model_io::safetensors::SafetensorsFile) -> Result<Self, String> {
        let load = |name: &str| -> Result<Vec<f32>, String> {
            sf.load_as_f32(name)
                .map_err(|e| format!("failed to load {name}: {e}"))
        };
        let load_opt = |name: &str| -> Result<Option<Vec<f32>>, String> {
            if sf.contains_tensor(name) {
                sf.load_as_f32(name)
                    .map(Some)
                    .map_err(|e| format!("failed to load {name}: {e}"))
            } else {
                Ok(None)
            }
        };

        let conv_in_weight = load("decoder.conv_in.weight")?;
        let conv_in_bias = load("decoder.conv_in.bias")?;

        let load_resnet =
            |prefix: &str, in_c: usize, out_c: usize| -> Result<ResnetBlock2D, String> {
                Ok(ResnetBlock2D {
                    in_channels: in_c,
                    out_channels: out_c,
                    norm1_weight: load(&format!("{prefix}.norm1.weight"))?,
                    norm1_bias: load(&format!("{prefix}.norm1.bias"))?,
                    conv1_weight: load(&format!("{prefix}.conv1.weight"))?,
                    conv1_bias: load(&format!("{prefix}.conv1.bias"))?,
                    norm2_weight: load(&format!("{prefix}.norm2.weight"))?,
                    norm2_bias: load(&format!("{prefix}.norm2.bias"))?,
                    conv2_weight: load(&format!("{prefix}.conv2.weight"))?,
                    conv2_bias: load(&format!("{prefix}.conv2.bias"))?,
                    conv_shortcut_weight: load_opt(&format!("{prefix}.conv_shortcut.weight"))?,
                    conv_shortcut_bias: load_opt(&format!("{prefix}.conv_shortcut.bias"))?,
                })
            };

        let mid_block = UNetMidBlock {
            resnet0: load_resnet("decoder.mid_block.resnets.0", 512, 512)?,
            attention: VaeAttention {
                channels: 512,
                norm_weight: load("decoder.mid_block.attentions.0.group_norm.weight")?,
                norm_bias: load("decoder.mid_block.attentions.0.group_norm.bias")?,
                q_weight: load("decoder.mid_block.attentions.0.to_q.weight")?,
                q_bias: load("decoder.mid_block.attentions.0.to_q.bias")?,
                k_weight: load("decoder.mid_block.attentions.0.to_k.weight")?,
                k_bias: load("decoder.mid_block.attentions.0.to_k.bias")?,
                v_weight: load("decoder.mid_block.attentions.0.to_v.weight")?,
                v_bias: load("decoder.mid_block.attentions.0.to_v.bias")?,
                out_weight: load("decoder.mid_block.attentions.0.to_out.0.weight")?,
                out_bias: load("decoder.mid_block.attentions.0.to_out.0.bias")?,
            },
            resnet1: load_resnet("decoder.mid_block.resnets.1", 512, 512)?,
        };

        let stage_specs = [
            (512, 512, true),
            (512, 512, true),
            (512, 256, true),
            (256, 128, false),
        ];

        let mut up_blocks = Vec::with_capacity(4);
        for (b, &(in_c, out_c, has_up)) in stage_specs.iter().enumerate() {
            let mut resnets = Vec::with_capacity(3);
            for r in 0..3 {
                let r_in = if r == 0 { in_c } else { out_c };
                let prefix = format!("decoder.up_blocks.{b}.resnets.{r}");
                resnets.push(load_resnet(&prefix, r_in, out_c)?);
            }
            let upsampler = if has_up {
                Some(UpSampler {
                    in_channels: out_c,
                    out_channels: out_c,
                    conv_weight: load(&format!("decoder.up_blocks.{b}.upsamplers.0.conv.weight"))?,
                    conv_bias: load(&format!("decoder.up_blocks.{b}.upsamplers.0.conv.bias"))?,
                })
            } else {
                None
            };
            up_blocks.push(UpDecoderBlock { resnets, upsampler });
        }

        let conv_norm_out_weight = load("decoder.conv_norm_out.weight")?;
        let conv_norm_out_bias = load("decoder.conv_norm_out.bias")?;
        let conv_out_weight = load("decoder.conv_out.weight")?;
        let conv_out_bias = load("decoder.conv_out.bias")?;

        Ok(Self {
            conv_in_weight,
            conv_in_bias,
            mid_block,
            up_blocks,
            conv_norm_out_weight,
            conv_norm_out_bias,
            conv_out_weight,
            conv_out_bias,
        })
    }

    /// Decode latents into RGB pixels.
    ///
    /// Input `latents`: shape `[16, h, w]`.
    /// Output: shape `[3, 8*h, 8*w]`.
    pub fn decode(&self, latents: &[f32], h: usize, w: usize) -> Result<Vec<f32>, String> {
        if latents.len() != VAE_LATENT_CHANNELS * h * w {
            return Err(format!(
                "latent length {} does not match 16 * {h} * {w} = {}",
                latents.len(),
                VAE_LATENT_CHANNELS * h * w
            ));
        }

        // 1. Rescale latents: latent / 0.3611 + 0.1159
        let inv_scale = 1.0 / VAE_SCALE_FACTOR;
        let mut scaled = Vec::with_capacity(latents.len());
        for &val in latents {
            scaled.push(val * inv_scale + VAE_SHIFT_FACTOR);
        }

        // 2. conv_in (16 -> 512, 3x3, pad 1)
        let mut cur = conv2d(
            &scaled,
            VAE_LATENT_CHANNELS,
            h,
            w,
            &self.conv_in_weight,
            Some(&self.conv_in_bias),
            512,
            3,
            3,
            1,
            1,
        )?;
        let mut cur_h = h;
        let mut cur_w = w;

        // 3. mid_block
        cur = self.mid_block.forward(&cur, cur_h, cur_w)?;

        // 4. up_blocks
        for up_block in &self.up_blocks {
            let (next_cur, next_h, next_w) = up_block.forward(&cur, cur_h, cur_w)?;
            cur = next_cur;
            cur_h = next_h;
            cur_w = next_w;
        }

        // 5. conv_norm_out (GroupNorm 32, 128 channels)
        let normed = group_norm(
            &cur,
            128,
            cur_h,
            cur_w,
            &self.conv_norm_out_weight,
            &self.conv_norm_out_bias,
            GROUP_NORM_GROUPS,
            GROUP_NORM_EPS,
        )?;

        // 6. silu
        let mut act = Vec::with_capacity(normed.len());
        for &val in &normed {
            act.push(silu(val));
        }

        // 7. conv_out (128 -> 3, 3x3, pad 1)
        conv2d(
            &act,
            128,
            cur_h,
            cur_w,
            &self.conv_out_weight,
            Some(&self.conv_out_bias),
            VAE_OUT_CHANNELS,
            3,
            3,
            1,
            1,
        )
    }
}
