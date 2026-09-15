//! Native Metal backend for the Z-Image-Turbo IG2 envelope.
//!
//! The backend owns one queue and maps only the component currently being
//! executed. Packed tensor rows stay in their verified `tensors.bin` mapping;
//! the image shaders decode FP32, BF16, and the image-specific interleaved
//! INT4 representation at the point of use. There is no CPU fallback in this
//! type. The opt-in parity and resource gates remain separate from this
//! device implementation because no source checkpoint is copied into tests.

use std::path::{Path, PathBuf};

use tokenizer::MfTokenizer;

use crate::conditioning::{frame_prompt, tokenize_prompt};
use crate::install::ImageManifest;
use crate::metal_ops::{self, Component};
use crate::patchify::{
    create_coordinate_grid, patchify_image, unpatchify, DEFAULT_F_PATCH_SIZE, DEFAULT_PATCH_SIZE,
    LATENT_CHANNELS, SEQ_MULTI_OF,
};
use crate::pipeline::{
    timestep_embedding, Z_IMAGE_CAP_DIM, Z_IMAGE_DIM, Z_IMAGE_FFN_DIM, Z_IMAGE_HEADS,
    Z_IMAGE_HEAD_DIM, Z_IMAGE_TIME_DIM,
};
use crate::rope::RopeEmbedder;
use crate::runtime::{CancellationToken, ImageBackend, ImageRequest};
use crate::scheduler::FlowMatchEulerScheduler;

use super::text_encoder::{EXTRACT_LAYER_COUNT, HEAD_DIM, HIDDEN_SIZE, NUM_KV_HEADS, NUM_Q_HEADS};

const TEXT_COMPONENT: &str = "components/text_encoder";
const TRANSFORMER_COMPONENT: &str = "components/transformer";
const VAE_COMPONENT: &str = "components/vae_decoder";
const IMAGE_CANCELLED: &str = "image generation cancelled";
const ROPE_THETA: f32 = 1_000_000.0;
const TRANSFORMER_EPS: f32 = 1e-5;

/// Native Metal implementation of [`ImageBackend`].
///
/// `open` verifies the complete image receipt before creating a resident
/// mapping. Each stage then opens and drops its own packed payload, keeping
/// the lifetime contract explicit instead of retaining all three heavy
/// components for the duration of a request.
pub struct MetalImageBackend {
    context: gpu::MetalContext,
    root: PathBuf,
    tokenizer: MfTokenizer,
    manifest: ImageManifest,
}

impl std::fmt::Debug for MetalImageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalImageBackend")
            .field("root", &self.root)
            .field("model_id", &self.manifest.source)
            .finish_non_exhaustive()
    }
}

impl MetalImageBackend {
    /// Open and verify a packed image install on a Metal-capable macOS host.
    pub fn open(root: &Path) -> Result<Self, String> {
        let context = gpu::MetalContext::new().map_err(|e| {
            format!(
                "native image generation requires a Metal device: {e}; use the explicit reference backend for diagnostics"
            )
        })?;
        let manifest = ImageManifest::load(root)?;
        manifest.validate()?;
        manifest.verify_files(root)?;
        let tokenizer = crate::conditioning::load_tokenizer(&root.join("components/tokenizer"))
            .map_err(|e| format!("failed to load image tokenizer: {e}"))?;
        Ok(Self {
            context,
            root: root.to_path_buf(),
            tokenizer,
            manifest,
        })
    }

    /// Number of Metal allocations made by this backend, for the IG2 memory
    /// oracle. Resident mapped payloads are intentionally not counted as
    /// copied buffers by `MetalContext`.
    pub fn buffer_allocation_count(&self) -> u64 {
        self.context.buffer_allocation_count()
    }

    fn component(&self, relative: &str) -> Result<Component, String> {
        Component::open(&self.context, &self.root.join(relative))
    }

    fn cancelled(cancellation: &CancellationToken) -> Result<(), String> {
        if cancellation.is_cancelled() {
            Err(IMAGE_CANCELLED.to_string())
        } else {
            Ok(())
        }
    }

    fn freqs(&self, rows: usize) -> metal::Buffer {
        let values = metal_ops::frequencies(rows, HEAD_DIM, ROPE_THETA);
        self.context.new_buffer_with_data(&values)
    }

    fn cis_freqs(&self, ids: &[[i32; 3]]) -> Result<metal::Buffer, String> {
        let pairs = RopeEmbedder::default().embed_ids(ids)?;
        let values: Vec<f32> = pairs
            .into_iter()
            .flat_map(|(cos, sin)| [cos, sin])
            .collect();
        Ok(self.context.new_buffer_with_data(&values))
    }

    fn text_weight(
        component: &Component,
        name: &str,
        shape: &[usize],
    ) -> Result<metal_ops::WeightRef, String> {
        component.weight(name, shape)
    }

    fn text_layer(
        &mut self,
        component: &Component,
        x: metal_ops::GpuTensor,
        layer: usize,
        seq_len: usize,
        freqs: &metal::Buffer,
    ) -> Result<metal_ops::GpuTensor, String> {
        let prefix = format!("model.layers.{layer}");
        let input_norm = Self::text_weight(
            component,
            &format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN_SIZE],
        )?;
        let q_proj = Self::text_weight(
            component,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE],
        )?;
        let k_proj = Self::text_weight(
            component,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
        )?;
        let v_proj = Self::text_weight(
            component,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE],
        )?;
        let q_norm = Self::text_weight(
            component,
            &format!("{prefix}.self_attn.q_norm.weight"),
            &[HEAD_DIM],
        )?;
        let k_norm = Self::text_weight(
            component,
            &format!("{prefix}.self_attn.k_norm.weight"),
            &[HEAD_DIM],
        )?;
        let o_proj = Self::text_weight(
            component,
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM],
        )?;
        let post_norm = Self::text_weight(
            component,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN_SIZE],
        )?;
        let gate_proj = Self::text_weight(
            component,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[crate::text_encoder::INTERMEDIATE_SIZE, HIDDEN_SIZE],
        )?;
        let up_proj = Self::text_weight(
            component,
            &format!("{prefix}.mlp.up_proj.weight"),
            &[crate::text_encoder::INTERMEDIATE_SIZE, HIDDEN_SIZE],
        )?;
        let down_proj = Self::text_weight(
            component,
            &format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN_SIZE, crate::text_encoder::INTERMEDIATE_SIZE],
        )?;

        let normed = metal_ops::rms_norm(
            &mut self.context,
            component,
            input_norm,
            &x,
            seq_len,
            HIDDEN_SIZE,
            crate::text_encoder::RMS_NORM_EPS,
        )?;
        let q = metal_ops::linear(
            &mut self.context,
            component,
            q_proj,
            None,
            &normed,
            seq_len,
            HIDDEN_SIZE,
            NUM_Q_HEADS * HEAD_DIM,
        )?;
        let k = metal_ops::linear(
            &mut self.context,
            component,
            k_proj,
            None,
            &normed,
            seq_len,
            HIDDEN_SIZE,
            NUM_KV_HEADS * HEAD_DIM,
        )?;
        let v = metal_ops::linear(
            &mut self.context,
            component,
            v_proj,
            None,
            &normed,
            seq_len,
            HIDDEN_SIZE,
            NUM_KV_HEADS * HEAD_DIM,
        )?;
        let q = metal_ops::rope(
            &mut self.context,
            component,
            q_norm,
            &q,
            freqs,
            seq_len,
            NUM_Q_HEADS,
            HEAD_DIM,
            crate::text_encoder::RMS_NORM_EPS,
        )?;
        let k = metal_ops::rope(
            &mut self.context,
            component,
            k_norm,
            &k,
            freqs,
            seq_len,
            NUM_KV_HEADS,
            HEAD_DIM,
            crate::text_encoder::RMS_NORM_EPS,
        )?;
        let attention = metal_ops::attention(
            &mut self.context,
            &q,
            &k,
            &v,
            seq_len,
            seq_len,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            true,
        )?;
        let attention = metal_ops::linear(
            &mut self.context,
            component,
            o_proj,
            None,
            &attention,
            seq_len,
            NUM_Q_HEADS * HEAD_DIM,
            HIDDEN_SIZE,
        )?;
        let x = metal_ops::add(&mut self.context, &x, &attention)?;
        let post = metal_ops::rms_norm(
            &mut self.context,
            component,
            post_norm,
            &x,
            seq_len,
            HIDDEN_SIZE,
            crate::text_encoder::RMS_NORM_EPS,
        )?;
        let gate = metal_ops::linear(
            &mut self.context,
            component,
            gate_proj,
            None,
            &post,
            seq_len,
            HIDDEN_SIZE,
            crate::text_encoder::INTERMEDIATE_SIZE,
        )?;
        let up = metal_ops::linear(
            &mut self.context,
            component,
            up_proj,
            None,
            &post,
            seq_len,
            HIDDEN_SIZE,
            crate::text_encoder::INTERMEDIATE_SIZE,
        )?;
        let activation = metal_ops::silu_mul(&mut self.context, &gate, &up)?;
        let down = metal_ops::linear(
            &mut self.context,
            component,
            down_proj,
            None,
            &activation,
            seq_len,
            crate::text_encoder::INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )?;
        metal_ops::add(&mut self.context, &x, &down)
    }

    fn time_embedding(
        &mut self,
        component: &Component,
        scaled_timestep: f32,
    ) -> Result<metal_ops::GpuTensor, String> {
        let frequencies = timestep_embedding(scaled_timestep, Z_IMAGE_TIME_DIM);
        let input = metal_ops::upload(&self.context, &frequencies);
        let first_weight =
            component.weight("t_embedder.mlp.0.weight", &[1024, Z_IMAGE_TIME_DIM])?;
        let first_bias = component.weight("t_embedder.mlp.0.bias", &[1024])?;
        let first = metal_ops::linear(
            &mut self.context,
            component,
            first_weight,
            Some(first_bias),
            &input,
            1,
            Z_IMAGE_TIME_DIM,
            1024,
        )?;
        let first_values = metal_ops::read(&first)
            .into_iter()
            .map(crate::transformer::silu)
            .collect::<Vec<_>>();
        let activated = metal_ops::upload(&self.context, &first_values);
        let second_weight =
            component.weight("t_embedder.mlp.2.weight", &[Z_IMAGE_TIME_DIM, 1024])?;
        let second_bias = component.weight("t_embedder.mlp.2.bias", &[Z_IMAGE_TIME_DIM])?;
        metal_ops::linear(
            &mut self.context,
            component,
            second_weight,
            Some(second_bias),
            &activated,
            1,
            1024,
            Z_IMAGE_TIME_DIM,
        )
    }

    fn modulation(
        &mut self,
        component: &Component,
        prefix: &str,
        timestep: &metal_ops::GpuTensor,
    ) -> Result<
        (
            metal_ops::GpuTensor,
            metal_ops::GpuTensor,
            metal_ops::GpuTensor,
            metal_ops::GpuTensor,
        ),
        String,
    > {
        let weight = component.weight(
            &format!("{prefix}.adaLN_modulation.0.weight"),
            &[4 * Z_IMAGE_DIM, Z_IMAGE_TIME_DIM],
        )?;
        let bias = component.weight(
            &format!("{prefix}.adaLN_modulation.0.bias"),
            &[4 * Z_IMAGE_DIM],
        )?;
        let raw = metal_ops::linear(
            &mut self.context,
            component,
            weight,
            Some(bias),
            timestep,
            1,
            Z_IMAGE_TIME_DIM,
            4 * Z_IMAGE_DIM,
        )?;
        let values = metal_ops::read(&raw);
        let scale_msa = metal_ops::upload(&self.context, &values[..Z_IMAGE_DIM]);
        let gate_msa = metal_ops::upload(
            &self.context,
            &values[Z_IMAGE_DIM..2 * Z_IMAGE_DIM]
                .iter()
                .map(|value| value.tanh())
                .collect::<Vec<_>>(),
        );
        let scale_mlp = metal_ops::upload(&self.context, &values[2 * Z_IMAGE_DIM..3 * Z_IMAGE_DIM]);
        let gate_mlp = metal_ops::upload(
            &self.context,
            &values[3 * Z_IMAGE_DIM..]
                .iter()
                .map(|value| value.tanh())
                .collect::<Vec<_>>(),
        );
        Ok((scale_msa, gate_msa, scale_mlp, gate_mlp))
    }

    fn transformer_block(
        &mut self,
        component: &Component,
        mut x: metal_ops::GpuTensor,
        prefix: &str,
        rows: usize,
        freqs: &metal::Buffer,
        timestep: Option<&metal_ops::GpuTensor>,
    ) -> Result<metal_ops::GpuTensor, String> {
        let modulation = match timestep {
            Some(timestep) => Some(self.modulation(component, prefix, timestep)?),
            None => None,
        };
        let weight =
            |suffix: &str, shape: &[usize]| component.weight(&format!("{prefix}.{suffix}"), shape);
        let attention_norm1 = weight("attention_norm1.weight", &[Z_IMAGE_DIM])?;
        let to_q = weight("attention.to_q.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?;
        let to_k = weight("attention.to_k.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?;
        let to_v = weight("attention.to_v.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?;
        let norm_q = weight("attention.norm_q.weight", &[Z_IMAGE_HEAD_DIM])?;
        let norm_k = weight("attention.norm_k.weight", &[Z_IMAGE_HEAD_DIM])?;
        let to_out = weight("attention.to_out.0.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?;
        let attention_norm2 = weight("attention_norm2.weight", &[Z_IMAGE_DIM])?;
        let ffn_norm1 = weight("ffn_norm1.weight", &[Z_IMAGE_DIM])?;
        let w1 = weight("feed_forward.w1.weight", &[Z_IMAGE_FFN_DIM, Z_IMAGE_DIM])?;
        let w2 = weight("feed_forward.w2.weight", &[Z_IMAGE_DIM, Z_IMAGE_FFN_DIM])?;
        let w3 = weight("feed_forward.w3.weight", &[Z_IMAGE_FFN_DIM, Z_IMAGE_DIM])?;
        let ffn_norm2 = weight("ffn_norm2.weight", &[Z_IMAGE_DIM])?;

        let normalized = metal_ops::rms_norm(
            &mut self.context,
            component,
            attention_norm1,
            &x,
            rows,
            Z_IMAGE_DIM,
            TRANSFORMER_EPS,
        )?;
        let normalized = match &modulation {
            Some((scale, _, _, _)) => {
                metal_ops::scale_rows(&mut self.context, &normalized, scale, rows, Z_IMAGE_DIM)?
            }
            None => normalized,
        };
        let q = metal_ops::linear(
            &mut self.context,
            component,
            to_q,
            None,
            &normalized,
            rows,
            Z_IMAGE_DIM,
            Z_IMAGE_DIM,
        )?;
        let k = metal_ops::linear(
            &mut self.context,
            component,
            to_k,
            None,
            &normalized,
            rows,
            Z_IMAGE_DIM,
            Z_IMAGE_DIM,
        )?;
        let v = metal_ops::linear(
            &mut self.context,
            component,
            to_v,
            None,
            &normalized,
            rows,
            Z_IMAGE_DIM,
            Z_IMAGE_DIM,
        )?;
        let q = metal_ops::adjacent_rope(
            &mut self.context,
            component,
            norm_q,
            &q,
            freqs,
            rows,
            Z_IMAGE_HEADS,
            Z_IMAGE_HEAD_DIM,
            TRANSFORMER_EPS,
        )?;
        let k = metal_ops::adjacent_rope(
            &mut self.context,
            component,
            norm_k,
            &k,
            freqs,
            rows,
            Z_IMAGE_HEADS,
            Z_IMAGE_HEAD_DIM,
            TRANSFORMER_EPS,
        )?;
        let attention = metal_ops::attention(
            &mut self.context,
            &q,
            &k,
            &v,
            rows,
            rows,
            Z_IMAGE_HEADS,
            Z_IMAGE_HEADS,
            Z_IMAGE_HEAD_DIM,
            false,
        )?;
        let projected = metal_ops::linear(
            &mut self.context,
            component,
            to_out,
            None,
            &attention,
            rows,
            Z_IMAGE_DIM,
            Z_IMAGE_DIM,
        )?;
        let projected = metal_ops::rms_norm(
            &mut self.context,
            component,
            attention_norm2,
            &projected,
            rows,
            Z_IMAGE_DIM,
            TRANSFORMER_EPS,
        )?;
        x = match &modulation {
            Some((_, gate, _, _)) => {
                metal_ops::gate_add(&mut self.context, &x, &projected, gate, rows, Z_IMAGE_DIM)?
            }
            None => metal_ops::add(&mut self.context, &x, &projected)?,
        };

        let ffn_input = metal_ops::rms_norm(
            &mut self.context,
            component,
            ffn_norm1,
            &x,
            rows,
            Z_IMAGE_DIM,
            TRANSFORMER_EPS,
        )?;
        let ffn_input = match &modulation {
            Some((_, _, scale, _)) => {
                metal_ops::scale_rows(&mut self.context, &ffn_input, scale, rows, Z_IMAGE_DIM)?
            }
            None => ffn_input,
        };
        let left = metal_ops::linear(
            &mut self.context,
            component,
            w1,
            None,
            &ffn_input,
            rows,
            Z_IMAGE_DIM,
            Z_IMAGE_FFN_DIM,
        )?;
        let right = metal_ops::linear(
            &mut self.context,
            component,
            w3,
            None,
            &ffn_input,
            rows,
            Z_IMAGE_DIM,
            Z_IMAGE_FFN_DIM,
        )?;
        let activation = metal_ops::silu_mul(&mut self.context, &left, &right)?;
        let ffn_output = metal_ops::linear(
            &mut self.context,
            component,
            w2,
            None,
            &activation,
            rows,
            Z_IMAGE_FFN_DIM,
            Z_IMAGE_DIM,
        )?;
        let ffn_output = metal_ops::rms_norm(
            &mut self.context,
            component,
            ffn_norm2,
            &ffn_output,
            rows,
            Z_IMAGE_DIM,
            TRANSFORMER_EPS,
        )?;
        match &modulation {
            Some((_, _, _, gate)) => {
                metal_ops::gate_add(&mut self.context, &x, &ffn_output, gate, rows, Z_IMAGE_DIM)
            }
            None => metal_ops::add(&mut self.context, &x, &ffn_output),
        }
    }

    // These arguments mirror the VAE block's separate convolution, norm, and
    // spatial contracts; a config struct would make the kernel shapes less
    // visible at the call site.
    #[allow(clippy::too_many_arguments)]
    fn vae_resnet(
        &mut self,
        component: &Component,
        x: metal_ops::GpuTensor,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    ) -> Result<metal_ops::GpuTensor, String> {
        let weight =
            |suffix: &str, shape: &[usize]| component.weight(&format!("{prefix}.{suffix}"), shape);
        let norm1_weight = weight("norm1.weight", &[in_channels])?;
        let norm1_bias = weight("norm1.bias", &[in_channels])?;
        let conv1_weight = weight("conv1.weight", &[out_channels, in_channels, 3, 3])?;
        let conv1_bias = weight("conv1.bias", &[out_channels])?;
        let norm2_weight = weight("norm2.weight", &[out_channels])?;
        let norm2_bias = weight("norm2.bias", &[out_channels])?;
        let conv2_weight = weight("conv2.weight", &[out_channels, out_channels, 3, 3])?;
        let conv2_bias = weight("conv2.bias", &[out_channels])?;
        let normalized = metal_ops::group_norm(
            &mut self.context,
            component,
            norm1_weight,
            norm1_bias,
            &x,
            in_channels,
            height,
            width,
            crate::vae::GROUP_NORM_GROUPS,
            crate::vae::GROUP_NORM_EPS,
        )?;
        let activated = metal_ops::silu(&mut self.context, component, &normalized)?;
        let conv1 = metal_ops::conv2d(
            &mut self.context,
            component,
            conv1_weight,
            Some(conv1_bias),
            &activated,
            in_channels,
            height,
            width,
            out_channels,
            3,
            1,
            1,
        )?;
        let normalized = metal_ops::group_norm(
            &mut self.context,
            component,
            norm2_weight,
            norm2_bias,
            &conv1,
            out_channels,
            height,
            width,
            crate::vae::GROUP_NORM_GROUPS,
            crate::vae::GROUP_NORM_EPS,
        )?;
        let activated = metal_ops::silu(&mut self.context, component, &normalized)?;
        let conv2 = metal_ops::conv2d(
            &mut self.context,
            component,
            conv2_weight,
            Some(conv2_bias),
            &activated,
            out_channels,
            height,
            width,
            out_channels,
            3,
            1,
            1,
        )?;
        let shortcut = if component
            .store
            .contains_tensor(&format!("{prefix}.conv_shortcut.weight"))
        {
            let shortcut_weight =
                weight("conv_shortcut.weight", &[out_channels, in_channels, 1, 1])?;
            let shortcut_bias = weight("conv_shortcut.bias", &[out_channels])?;
            metal_ops::conv2d(
                &mut self.context,
                component,
                shortcut_weight,
                Some(shortcut_bias),
                &x,
                in_channels,
                height,
                width,
                out_channels,
                1,
                1,
                0,
            )?
        } else {
            if in_channels != out_channels {
                return Err(format!(
                    "VAE resnet {prefix} needs a shortcut for {in_channels}->{out_channels}"
                ));
            }
            x
        };
        metal_ops::add(&mut self.context, &shortcut, &conv2)
    }

    fn vae_attention(
        &mut self,
        component: &Component,
        x: metal_ops::GpuTensor,
        height: usize,
        width: usize,
    ) -> Result<metal_ops::GpuTensor, String> {
        let channels = 512;
        let norm = metal_ops::group_norm(
            &mut self.context,
            component,
            component.weight(
                "decoder.mid_block.attentions.0.group_norm.weight",
                &[channels],
            )?,
            component.weight(
                "decoder.mid_block.attentions.0.group_norm.bias",
                &[channels],
            )?,
            &x,
            channels,
            height,
            width,
            crate::vae::GROUP_NORM_GROUPS,
            crate::vae::GROUP_NORM_EPS,
        )?;
        let projection =
            |name: &str| -> Result<(metal_ops::WeightRef, metal_ops::WeightRef), String> {
                Ok((
                    component.weight(
                        &format!("decoder.mid_block.attentions.0.{name}.weight"),
                        &[channels, channels, 1, 1],
                    )?,
                    component.weight(
                        &format!("decoder.mid_block.attentions.0.{name}.bias"),
                        &[channels],
                    )?,
                ))
            };
        let (q_weight, q_bias) = projection("to_q")?;
        let (k_weight, k_bias) = projection("to_k")?;
        let (v_weight, v_bias) = projection("to_v")?;
        let q = metal_ops::conv2d(
            &mut self.context,
            component,
            q_weight,
            Some(q_bias),
            &norm,
            channels,
            height,
            width,
            channels,
            1,
            1,
            0,
        )?;
        let k = metal_ops::conv2d(
            &mut self.context,
            component,
            k_weight,
            Some(k_bias),
            &norm,
            channels,
            height,
            width,
            channels,
            1,
            1,
            0,
        )?;
        let v = metal_ops::conv2d(
            &mut self.context,
            component,
            v_weight,
            Some(v_bias),
            &norm,
            channels,
            height,
            width,
            channels,
            1,
            1,
            0,
        )?;
        let attended = metal_ops::vae_attention(
            &mut self.context,
            component,
            &q,
            &k,
            &v,
            channels,
            height,
            width,
        )?;
        let out = metal_ops::conv2d(
            &mut self.context,
            component,
            component.weight(
                "decoder.mid_block.attentions.0.to_out.0.weight",
                &[channels, channels, 1, 1],
            )?,
            Some(component.weight("decoder.mid_block.attentions.0.to_out.0.bias", &[channels])?),
            &attended,
            channels,
            height,
            width,
            channels,
            1,
            1,
            0,
        )?;
        metal_ops::add(&mut self.context, &x, &out)
    }

    /// Run native denoising while retaining each of the nine scheduler
    /// outputs for the packed parity gate. The production `ImageBackend`
    /// call uses the same path with a no-op observer, so this cannot create a
    /// second implementation of the update order.
    pub fn denoise_steps(
        &mut self,
        conditioning: &[f32],
        request: &ImageRequest,
        scheduler: &FlowMatchEulerScheduler,
        cancellation: &CancellationToken,
    ) -> Result<Vec<Vec<f32>>, String> {
        if scheduler.timesteps.len() != request.scheduler_steps as usize
            || scheduler.sigmas.len() != request.scheduler_steps as usize + 1
        {
            return Err("native image scheduler shape does not match the request".to_string());
        }
        let mut steps = Vec::with_capacity(request.scheduler_steps as usize);
        self.denoise_impl(
            conditioning,
            request,
            scheduler,
            cancellation,
            &mut |_, latent| steps.push(latent.to_vec()),
            &mut |_, _| {},
        )?;
        Ok(steps)
    }

    fn final_velocity(
        &mut self,
        component: &Component,
        unified: &metal_ops::GpuTensor,
        image_padded_len: usize,
        image_len: usize,
        token_size: (usize, usize, usize),
        timestep: &metal_ops::GpuTensor,
    ) -> Result<metal_ops::GpuTensor, String> {
        let values = metal_ops::read(unified);
        let image = metal_ops::upload(&self.context, &values[..image_padded_len * Z_IMAGE_DIM]);
        let timestep_values = metal_ops::read(timestep)
            .into_iter()
            .map(crate::transformer::silu)
            .collect::<Vec<_>>();
        let timestep_values = metal_ops::upload(&self.context, &timestep_values);
        let final_scale = metal_ops::linear(
            &mut self.context,
            component,
            component.weight(
                "all_final_layer.2-1.adaLN_modulation.1.weight",
                &[Z_IMAGE_DIM, Z_IMAGE_TIME_DIM],
            )?,
            Some(component.weight(
                "all_final_layer.2-1.adaLN_modulation.1.bias",
                &[Z_IMAGE_DIM],
            )?),
            &timestep_values,
            1,
            Z_IMAGE_TIME_DIM,
            Z_IMAGE_DIM,
        )?;
        let mut scale_values = metal_ops::read(&final_scale);
        for value in &mut scale_values {
            *value += 1.0;
        }
        let final_scale = metal_ops::upload(&self.context, &scale_values);
        let normalized = metal_ops::layer_norm(
            &mut self.context,
            &image,
            &final_scale,
            image_padded_len,
            Z_IMAGE_DIM,
            1e-6,
        )?;
        let projected = metal_ops::linear(
            &mut self.context,
            component,
            component.weight("all_final_layer.2-1.linear.weight", &[64, Z_IMAGE_DIM])?,
            Some(component.weight("all_final_layer.2-1.linear.bias", &[64])?),
            &normalized,
            image_padded_len,
            Z_IMAGE_DIM,
            64,
        )?;
        let projected = metal_ops::read(&projected);
        let mut output = unpatchify(
            &projected[..image_len * 64],
            token_size.0,
            token_size.1,
            token_size.2,
            DEFAULT_F_PATCH_SIZE,
            DEFAULT_PATCH_SIZE,
            LATENT_CHANNELS,
        )?;
        for value in &mut output {
            *value = -*value;
        }
        Ok(metal_ops::upload(&self.context, &output))
    }
}

impl ImageBackend for MetalImageBackend {
    fn encode_conditioning(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        Self::cancelled(cancellation)?;
        let framed = frame_prompt(prompt, &self.tokenizer)
            .map_err(|e| format!("failed to frame image prompt: {e}"))?;
        let (padded_ids, mask) = tokenize_prompt(&framed, &self.tokenizer, max_tokens);
        let ids: Vec<u32> = padded_ids
            .into_iter()
            .zip(mask)
            .filter_map(|(id, active)| (active != 0).then_some(id as u32))
            .collect();
        if ids.is_empty() {
            return Err("image prompt produced no active tokens".to_string());
        }
        let seq_len = ids.len();
        let component = self.component(TEXT_COMPONENT)?;
        let embedding_shape = component
            .store
            .shape("model.embed_tokens.weight")
            .ok_or_else(|| "image text encoder is missing model.embed_tokens.weight".to_string())?
            .to_vec();
        if embedding_shape.len() != 2 || embedding_shape[1] != HIDDEN_SIZE {
            return Err(format!(
                "image embedding has shape {embedding_shape:?}, expected [vocab, {HIDDEN_SIZE}]"
            ));
        }
        let embedding =
            Self::text_weight(&component, "model.embed_tokens.weight", &embedding_shape)?;
        let mut x = metal_ops::lookup(
            &mut self.context,
            &component,
            embedding,
            &ids,
            embedding_shape[0],
            HIDDEN_SIZE,
        )?;
        let freqs = self.freqs(seq_len);
        let total = EXTRACT_LAYER_COUNT as u32;
        for layer in 0..EXTRACT_LAYER_COUNT {
            Self::cancelled(cancellation)?;
            x = self.text_layer(&component, x, layer, seq_len, &freqs)?;
            progress((layer + 1) as u32, total);
        }
        Self::cancelled(cancellation)?;
        Ok(metal_ops::read(&x))
    }

    fn denoise(
        &mut self,
        conditioning: &[f32],
        request: &ImageRequest,
        scheduler: &FlowMatchEulerScheduler,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        let mut step_observer = |_: usize, _: &[f32]| {};
        self.denoise_impl(
            conditioning,
            request,
            scheduler,
            cancellation,
            &mut step_observer,
            progress,
        )
    }

    fn decode(
        &mut self,
        latents: &[f32],
        width: u32,
        height: u32,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        self.decode_impl(latents, width, height, cancellation, progress)
    }
}

impl MetalImageBackend {
    fn denoise_impl(
        &mut self,
        conditioning: &[f32],
        request: &ImageRequest,
        scheduler: &FlowMatchEulerScheduler,
        cancellation: &CancellationToken,
        on_step: &mut dyn FnMut(usize, &[f32]),
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        if conditioning.is_empty() || conditioning.len() % Z_IMAGE_CAP_DIM != 0 {
            return Err("image conditioning is not a nonzero whole-token sequence".to_string());
        }
        let component = self.component(TRANSFORMER_COMPONENT)?;
        let cap_len = conditioning.len() / Z_IMAGE_CAP_DIM;
        let cap_padded_len = round_up(cap_len, SEQ_MULTI_OF)?;
        let latent_count = LATENT_CHANNELS * 128 * 128;
        let mut latent = seeded_noise(request.seed, latent_count);
        let (patches, _, token_size) = patchify_image(
            &latent,
            LATENT_CHANNELS,
            1,
            128,
            128,
            DEFAULT_PATCH_SIZE,
            DEFAULT_F_PATCH_SIZE,
        )?;
        let image_len = token_size.0 * token_size.1 * token_size.2;
        let image_padded_len = round_up(image_len, SEQ_MULTI_OF)?;
        let image_patch_buffer = metal_ops::upload(&self.context, &patches);
        let image_embed_weight =
            component.weight("all_x_embedder.2-1.weight", &[Z_IMAGE_DIM, 64])?;
        let image_embed_bias = component.weight("all_x_embedder.2-1.bias", &[Z_IMAGE_DIM])?;
        let image_embed = metal_ops::linear(
            &mut self.context,
            &component,
            image_embed_weight,
            Some(image_embed_bias),
            &image_patch_buffer,
            image_len,
            64,
            Z_IMAGE_DIM,
        )?;
        let mut image_values = metal_ops::read(&image_embed);
        if image_padded_len > image_len {
            let pad = component.store.load_tensor("x_pad_token")?;
            if pad.len() != Z_IMAGE_DIM {
                return Err("x_pad_token has an unexpected shape".to_string());
            }
            for _ in image_len..image_padded_len {
                image_values.extend_from_slice(&pad);
            }
        }
        let image = metal_ops::upload(&self.context, &image_values);

        let cap_input = metal_ops::upload(&self.context, conditioning);
        let cap_norm = component.weight("cap_embedder.0.weight", &[Z_IMAGE_CAP_DIM])?;
        let cap_weight =
            component.weight("cap_embedder.1.weight", &[Z_IMAGE_DIM, Z_IMAGE_CAP_DIM])?;
        let cap_bias = component.weight("cap_embedder.1.bias", &[Z_IMAGE_DIM])?;
        let cap_input = metal_ops::rms_norm(
            &mut self.context,
            &component,
            cap_norm,
            &cap_input,
            cap_len,
            Z_IMAGE_CAP_DIM,
            TRANSFORMER_EPS,
        )?;
        let caption_embed = metal_ops::linear(
            &mut self.context,
            &component,
            cap_weight,
            Some(cap_bias),
            &cap_input,
            cap_len,
            Z_IMAGE_CAP_DIM,
            Z_IMAGE_DIM,
        )?;
        let mut caption_values = metal_ops::read(&caption_embed);
        if cap_padded_len > cap_len {
            let pad = component.store.load_tensor("cap_pad_token")?;
            if pad.len() != Z_IMAGE_DIM {
                return Err("cap_pad_token has an unexpected shape".to_string());
            }
            for _ in cap_len..cap_padded_len {
                caption_values.extend_from_slice(&pad);
            }
        }
        let caption = metal_ops::upload(&self.context, &caption_values);

        let image_ids = create_coordinate_grid(token_size, (cap_padded_len + 1, 0, 0));
        let image_freqs = self.cis_freqs(&image_ids)?;
        let caption_ids = create_coordinate_grid((cap_padded_len, 1, 1), (1, 0, 0));
        let caption_freqs = self.cis_freqs(&caption_ids)?;
        let mut unified_ids = image_ids;
        unified_ids.extend_from_slice(&caption_ids);
        let unified_freqs = self.cis_freqs(&unified_ids)?;

        let mut timestep =
            self.time_embedding(&component, scheduler.normalized_time(0) * 1000.0)?;
        let mut image = image;
        for index in 0..2 {
            image = self.transformer_block(
                &component,
                image,
                &format!("noise_refiner.{index}"),
                image_padded_len,
                &image_freqs,
                Some(&timestep),
            )?;
        }
        let mut caption = caption;
        for index in 0..2 {
            caption = self.transformer_block(
                &component,
                caption,
                &format!("context_refiner.{index}"),
                cap_padded_len,
                &caption_freqs,
                None,
            )?;
        }
        let image_values = metal_ops::read(&image);
        // The context refiner has no timestep modulation and always starts
        // from the same caption embedding. Keep its output as the immutable
        // conditioning input for every denoising step instead of rebuilding
        // the two context-refiner blocks after each scheduler update.
        let refined_caption_values = metal_ops::read(&caption);
        let mut unified_values =
            Vec::with_capacity((image_padded_len + cap_padded_len) * Z_IMAGE_DIM);
        unified_values.extend_from_slice(&image_values);
        unified_values.extend_from_slice(&refined_caption_values);
        let mut unified = metal_ops::upload(&self.context, &unified_values);
        for step in 0..request.scheduler_steps as usize {
            Self::cancelled(cancellation)?;
            if step != 0 {
                // Each denoising step uses a new modulation embedding, while
                // image/caption embeddings and positional frequencies remain
                // stage-local immutable inputs.
                timestep =
                    self.time_embedding(&component, scheduler.normalized_time(step) * 1000.0)?;
            }
            for index in 0..30 {
                unified = self.transformer_block(
                    &component,
                    unified,
                    &format!("layers.{index}"),
                    image_padded_len + cap_padded_len,
                    &unified_freqs,
                    Some(&timestep),
                )?;
            }
            let velocity = self.final_velocity(
                &component,
                &unified,
                image_padded_len,
                image_len,
                token_size,
                &timestep,
            )?;
            let sample = metal_ops::upload(&self.context, &latent);
            let next = metal_ops::scheduler_step(
                &mut self.context,
                &sample,
                &velocity,
                scheduler.sigmas[step + 1] - scheduler.sigmas[step],
            )?;
            latent = metal_ops::read(&next);
            on_step(step, &latent);
            progress((step + 1) as u32, request.scheduler_steps);
            if step + 1 < request.scheduler_steps as usize {
                let (next_patches, _, _) = patchify_image(
                    &latent,
                    LATENT_CHANNELS,
                    1,
                    128,
                    128,
                    DEFAULT_PATCH_SIZE,
                    DEFAULT_F_PATCH_SIZE,
                )?;
                let next_input = metal_ops::upload(&self.context, &next_patches);
                unified = metal_ops::linear(
                    &mut self.context,
                    &component,
                    component.weight("all_x_embedder.2-1.weight", &[Z_IMAGE_DIM, 64])?,
                    Some(component.weight("all_x_embedder.2-1.bias", &[Z_IMAGE_DIM])?),
                    &next_input,
                    image_len,
                    64,
                    Z_IMAGE_DIM,
                )?;
                let mut next_image = metal_ops::read(&unified);
                if image_padded_len > image_len {
                    let pad = component.store.load_tensor("x_pad_token")?;
                    for _ in image_len..image_padded_len {
                        next_image.extend_from_slice(&pad);
                    }
                }

                // The noise refiner sees only image tokens. It must use the
                // next scheduler timestep, then the refined caption is
                // appended for the main transformer sequence.
                let next_timestep =
                    self.time_embedding(&component, scheduler.normalized_time(step + 1) * 1000.0)?;
                let mut next_image = metal_ops::upload(&self.context, &next_image);
                for index in 0..2 {
                    next_image = self.transformer_block(
                        &component,
                        next_image,
                        &format!("noise_refiner.{index}"),
                        image_padded_len,
                        &image_freqs,
                        Some(&next_timestep),
                    )?;
                }
                let next_image = metal_ops::read(&next_image);
                let mut next_unified =
                    Vec::with_capacity((image_padded_len + cap_padded_len) * Z_IMAGE_DIM);
                next_unified.extend_from_slice(&next_image);
                next_unified.extend_from_slice(&refined_caption_values);
                unified = metal_ops::upload(&self.context, &next_unified);
            }
        }

        Ok(latent)
    }

    fn decode_impl(
        &mut self,
        latents: &[f32],
        width: u32,
        height: u32,
        cancellation: &CancellationToken,
        progress: &mut dyn FnMut(u32, u32),
    ) -> Result<Vec<f32>, String> {
        if (width, height) != (1024, 1024) {
            return Err("native image VAE only supports 1024x1024".to_string());
        }
        if latents.len() != LATENT_CHANNELS * 128 * 128 {
            return Err("native image VAE expects a [16, 128, 128] latent".to_string());
        }
        Self::cancelled(cancellation)?;
        let component = self.component(VAE_COMPONENT)?;
        let scaled = latents
            .iter()
            .map(|value| value / crate::vae::VAE_SCALE_FACTOR + crate::vae::VAE_SHIFT_FACTOR)
            .collect::<Vec<_>>();
        let input = metal_ops::upload(&self.context, &scaled);
        let mut current = metal_ops::conv2d(
            &mut self.context,
            &component,
            component.weight("decoder.conv_in.weight", &[512, 16, 3, 3])?,
            Some(component.weight("decoder.conv_in.bias", &[512])?),
            &input,
            16,
            128,
            128,
            512,
            3,
            1,
            1,
        )?;
        current = self.vae_resnet(
            &component,
            current,
            "decoder.mid_block.resnets.0",
            512,
            512,
            128,
            128,
        )?;
        current = self.vae_attention(&component, current, 128, 128)?;
        current = self.vae_resnet(
            &component,
            current,
            "decoder.mid_block.resnets.1",
            512,
            512,
            128,
            128,
        )?;

        let mut cur_height = 128;
        let mut cur_width = 128;
        for (block, &(in_channels, out_channels, has_upsampler)) in [
            (512, 512, true),
            (512, 512, true),
            (512, 256, true),
            (256, 128, false),
        ]
        .iter()
        .enumerate()
        {
            for resnet in 0..3 {
                let resnet_in = if resnet == 0 {
                    in_channels
                } else {
                    out_channels
                };
                current = self.vae_resnet(
                    &component,
                    current,
                    &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
                    resnet_in,
                    out_channels,
                    cur_height,
                    cur_width,
                )?;
                Self::cancelled(cancellation)?;
            }
            if has_upsampler {
                current = metal_ops::upsample(
                    &mut self.context,
                    &component,
                    &current,
                    out_channels,
                    cur_height,
                    cur_width,
                )?;
                cur_height *= 2;
                cur_width *= 2;
                current = metal_ops::conv2d(
                    &mut self.context,
                    &component,
                    component.weight(
                        &format!("decoder.up_blocks.{block}.upsamplers.0.conv.weight"),
                        &[out_channels, out_channels, 3, 3],
                    )?,
                    Some(component.weight(
                        &format!("decoder.up_blocks.{block}.upsamplers.0.conv.bias"),
                        &[out_channels],
                    )?),
                    &current,
                    out_channels,
                    cur_height,
                    cur_width,
                    out_channels,
                    3,
                    1,
                    1,
                )?;
            }
        }
        let normalized = metal_ops::group_norm(
            &mut self.context,
            &component,
            component.weight("decoder.conv_norm_out.weight", &[128])?,
            component.weight("decoder.conv_norm_out.bias", &[128])?,
            &current,
            128,
            cur_height,
            cur_width,
            crate::vae::GROUP_NORM_GROUPS,
            crate::vae::GROUP_NORM_EPS,
        )?;
        let activated = metal_ops::silu(&mut self.context, &component, &normalized)?;
        let output = metal_ops::conv2d(
            &mut self.context,
            &component,
            component.weight("decoder.conv_out.weight", &[3, 128, 3, 3])?,
            Some(component.weight("decoder.conv_out.bias", &[3])?),
            &activated,
            128,
            cur_height,
            cur_width,
            3,
            3,
            1,
            1,
        )?;
        Self::cancelled(cancellation)?;
        progress(1, 1);
        Ok(metal_ops::read(&output))
    }
}

fn round_up(value: usize, multiple: usize) -> Result<usize, String> {
    value
        .checked_add(multiple - 1)
        .map(|v| v / multiple * multiple)
        .ok_or_else(|| "image sequence length overflowed".to_string())
}

fn seeded_noise(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut output = Vec::with_capacity(count);
    while output.len() < count {
        let u1 = uniform(&mut state).max(f64::MIN_POSITIVE);
        let u2 = uniform(&mut state);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        output.push((radius * angle.cos()) as f32);
        if output.len() < count {
            output.push((radius * angle.sin()) as f32);
        }
    }
    output
}

fn uniform(state: &mut u64) -> f64 {
    *state ^= *state << 7;
    *state ^= *state >> 9;
    *state ^= *state << 8;
    (*state as f64) / (u64::MAX as f64)
}
