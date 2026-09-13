//! Checkpoint-backed FP32 CPU reference for the Z-Image-Turbo DiT.
//!
//! This is deliberately a streamed reference implementation. It holds only
//! one transformer block's decoded FP32 weights at a time, which makes it a
//! parity gate rather than a resident or quantized runtime.

use std::path::Path;

use crate::patchify::{
    create_coordinate_grid, patchify_image, unpatchify, DEFAULT_F_PATCH_SIZE, DEFAULT_PATCH_SIZE,
    LATENT_CHANNELS, SEQ_MULTI_OF,
};
use crate::rope::RopeEmbedder;
use crate::text_encoder::ShardedSafetensors;
use crate::transformer::{
    linear_forward, rms_norm, silu, AdaLnModulation, FinalLayer, ZImageTransformerBlock,
};

pub const Z_IMAGE_DIM: usize = 3840;
pub const Z_IMAGE_HEADS: usize = 30;
pub const Z_IMAGE_HEAD_DIM: usize = 128;
pub const Z_IMAGE_CAP_DIM: usize = 2560;
pub const Z_IMAGE_TIME_DIM: usize = 256;
const Z_IMAGE_FFN_DIM: usize = 10240;
const Z_IMAGE_REFINER_BLOCKS: usize = 2;
const Z_IMAGE_MAIN_BLOCKS: usize = 30;

/// Pinned Z-Image transformer checkpoint, backed by memory-mapped shards.
pub struct ZImageTransformer {
    weights: ShardedSafetensors,
}

impl ZImageTransformer {
    /// Open the canonical transformer component directory.
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        Ok(Self {
            weights: ShardedSafetensors::open_indexed(
                model_dir,
                "diffusion_pytorch_model.safetensors.index.json",
            )?,
        })
    }

    /// Predict the scheduler velocity for one `[16, 1, height, width]` latent.
    pub fn forward(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
        timestep: f32,
        conditioning: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.forward_with_progress(latent, height, width, timestep, conditioning, |_| {})
    }

    /// Predict one scheduler velocity and report each completed transformer block.
    ///
    /// The callback exists for the opt-in checkpoint gate, where a single
    /// scalar CPU forward can take long enough that silent execution is not
    /// useful evidence.
    pub fn forward_with_progress<F>(
        &self,
        latent: &[f32],
        height: usize,
        width: usize,
        timestep: f32,
        conditioning: &[f32],
        mut completed: F,
    ) -> Result<Vec<f32>, String>
    where
        F: FnMut(&str),
    {
        if !timestep.is_finite() {
            return Err("timestep must be finite".to_string());
        }
        if conditioning.is_empty() || conditioning.len() % Z_IMAGE_CAP_DIM != 0 {
            return Err(format!(
                "conditioning length {} is not a nonzero multiple of {Z_IMAGE_CAP_DIM}",
                conditioning.len()
            ));
        }
        let (patches, _, token_size) = patchify_image(
            latent,
            LATENT_CHANNELS,
            1,
            height,
            width,
            DEFAULT_PATCH_SIZE,
            DEFAULT_F_PATCH_SIZE,
        )?;
        let cap_len = conditioning.len() / Z_IMAGE_CAP_DIM;
        let cap_padded_len = round_up(cap_len, SEQ_MULTI_OF)?;
        let image_len = token_size.0 * token_size.1 * token_size.2;
        let image_padded_len = round_up(image_len, SEQ_MULTI_OF)?;
        let t_emb = self.time_embedding(timestep * 1000.0)?;

        let mut image = self.embed_image(&patches)?;
        image.resize(image_padded_len * Z_IMAGE_DIM, 0.0);
        let x_pad = self.load("x_pad_token", &[1, Z_IMAGE_DIM])?;
        for token in image_len..image_padded_len {
            image[token * Z_IMAGE_DIM..(token + 1) * Z_IMAGE_DIM].copy_from_slice(&x_pad);
        }
        let image_ids = padded_image_ids(token_size, cap_padded_len, image_len, image_padded_len);
        let rope = RopeEmbedder::default();
        let image_freqs = rope.embed_ids(&image_ids)?;
        let attend_image = vec![true; image_padded_len];
        for index in 0..Z_IMAGE_REFINER_BLOCKS {
            let name = format!("noise_refiner.{index}");
            let block = self.load_block(&name, true)?;
            image = block.forward(&image, Some(&attend_image), &image_freqs, Some(&t_emb))?;
            completed(&name);
        }

        let mut caption = self.embed_caption(conditioning)?;
        caption.resize(cap_padded_len * Z_IMAGE_DIM, 0.0);
        let cap_pad = self.load("cap_pad_token", &[1, Z_IMAGE_DIM])?;
        for token in cap_len..cap_padded_len {
            caption[token * Z_IMAGE_DIM..(token + 1) * Z_IMAGE_DIM].copy_from_slice(&cap_pad);
        }
        let caption_ids = create_coordinate_grid((cap_padded_len, 1, 1), (1, 0, 0));
        let caption_freqs = rope.embed_ids(&caption_ids)?;
        let attend_caption = vec![true; cap_padded_len];
        for index in 0..Z_IMAGE_REFINER_BLOCKS {
            let name = format!("context_refiner.{index}");
            let block = self.load_block(&name, false)?;
            caption = block.forward(&caption, Some(&attend_caption), &caption_freqs, None)?;
            completed(&name);
        }

        let mut unified = Vec::with_capacity((image_padded_len + cap_padded_len) * Z_IMAGE_DIM);
        unified.extend_from_slice(&image);
        unified.extend_from_slice(&caption);
        let mut unified_freqs = image_freqs;
        unified_freqs.extend_from_slice(&caption_freqs);
        let attend_unified = vec![true; image_padded_len + cap_padded_len];
        for index in 0..Z_IMAGE_MAIN_BLOCKS {
            let name = format!("layers.{index}");
            let block = self.load_block(&name, true)?;
            unified = block.forward(
                &unified,
                Some(&attend_unified),
                &unified_freqs,
                Some(&t_emb),
            )?;
            completed(&name);
        }

        let final_layer = FinalLayer::new(
            Z_IMAGE_DIM,
            64,
            self.load("all_final_layer.2-1.linear.weight", &[64, Z_IMAGE_DIM])?,
            self.load("all_final_layer.2-1.linear.bias", &[64])?,
            self.load(
                "all_final_layer.2-1.adaLN_modulation.1.weight",
                &[Z_IMAGE_DIM, Z_IMAGE_TIME_DIM],
            )?,
            self.load(
                "all_final_layer.2-1.adaLN_modulation.1.bias",
                &[Z_IMAGE_DIM],
            )?,
        );
        let projected = final_layer.forward(&unified[..image_padded_len * Z_IMAGE_DIM], &t_emb)?;
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
        Ok(output)
    }

    fn load(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        self.weights.load_tensor_shape(name, shape)
    }

    fn time_embedding(&self, scaled_timestep: f32) -> Result<Vec<f32>, String> {
        let frequencies = timestep_embedding(scaled_timestep, Z_IMAGE_TIME_DIM);
        let first = linear_forward(
            &frequencies,
            &self.load("t_embedder.mlp.0.weight", &[1024, Z_IMAGE_TIME_DIM])?,
            Some(&self.load("t_embedder.mlp.0.bias", &[1024])?),
            1024,
            Z_IMAGE_TIME_DIM,
        );
        let activated: Vec<f32> = first.into_iter().map(silu).collect();
        Ok(linear_forward(
            &activated,
            &self.load("t_embedder.mlp.2.weight", &[Z_IMAGE_TIME_DIM, 1024])?,
            Some(&self.load("t_embedder.mlp.2.bias", &[Z_IMAGE_TIME_DIM])?),
            Z_IMAGE_TIME_DIM,
            1024,
        ))
    }

    fn embed_image(&self, patches: &[f32]) -> Result<Vec<f32>, String> {
        project_rows(
            patches,
            64,
            &self.load("all_x_embedder.2-1.weight", &[Z_IMAGE_DIM, 64])?,
            Some(&self.load("all_x_embedder.2-1.bias", &[Z_IMAGE_DIM])?),
            Z_IMAGE_DIM,
        )
    }

    fn embed_caption(&self, conditioning: &[f32]) -> Result<Vec<f32>, String> {
        let norm = self.load("cap_embedder.0.weight", &[Z_IMAGE_CAP_DIM])?;
        let weight = self.load("cap_embedder.1.weight", &[Z_IMAGE_DIM, Z_IMAGE_CAP_DIM])?;
        let bias = self.load("cap_embedder.1.bias", &[Z_IMAGE_DIM])?;
        let mut out = Vec::with_capacity(conditioning.len() / Z_IMAGE_CAP_DIM * Z_IMAGE_DIM);
        for token in conditioning.chunks_exact(Z_IMAGE_CAP_DIM) {
            let normalized = rms_norm(token, &norm, 1e-5);
            out.extend(linear_forward(
                &normalized,
                &weight,
                Some(&bias),
                Z_IMAGE_DIM,
                Z_IMAGE_CAP_DIM,
            ));
        }
        Ok(out)
    }

    fn load_block(&self, prefix: &str, modulation: bool) -> Result<ZImageTransformerBlock, String> {
        let load = |suffix: &str, shape: &[usize]| self.load(&format!("{prefix}.{suffix}"), shape);
        Ok(ZImageTransformerBlock {
            dim: Z_IMAGE_DIM,
            num_heads: Z_IMAGE_HEADS,
            head_dim: Z_IMAGE_HEAD_DIM,
            norm_eps: 1e-5,
            attention_norm1: load("attention_norm1.weight", &[Z_IMAGE_DIM])?,
            to_q: load("attention.to_q.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?,
            to_k: load("attention.to_k.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?,
            to_v: load("attention.to_v.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?,
            norm_q: load("attention.norm_q.weight", &[Z_IMAGE_HEAD_DIM])?,
            norm_k: load("attention.norm_k.weight", &[Z_IMAGE_HEAD_DIM])?,
            to_out: load("attention.to_out.0.weight", &[Z_IMAGE_DIM, Z_IMAGE_DIM])?,
            attention_norm2: load("attention_norm2.weight", &[Z_IMAGE_DIM])?,
            ffn_norm1: load("ffn_norm1.weight", &[Z_IMAGE_DIM])?,
            w1: load("feed_forward.w1.weight", &[Z_IMAGE_FFN_DIM, Z_IMAGE_DIM])?,
            w2: load("feed_forward.w2.weight", &[Z_IMAGE_DIM, Z_IMAGE_FFN_DIM])?,
            w3: load("feed_forward.w3.weight", &[Z_IMAGE_FFN_DIM, Z_IMAGE_DIM])?,
            ffn_norm2: load("ffn_norm2.weight", &[Z_IMAGE_DIM])?,
            modulation: if modulation {
                Some(AdaLnModulation::try_new(
                    Z_IMAGE_DIM,
                    load(
                        "adaLN_modulation.0.weight",
                        &[4 * Z_IMAGE_DIM, Z_IMAGE_TIME_DIM],
                    )?,
                    load("adaLN_modulation.0.bias", &[4 * Z_IMAGE_DIM])?,
                )?)
            } else {
                None
            },
        })
    }
}

/// Diffusers-compatible sinusoidal embedding for a single scaled timestep.
pub fn timestep_embedding(timestep: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = Vec::with_capacity(dim);
    for index in 0..half {
        let frequency = (-10000.0f32.ln() * index as f32 / half as f32).exp();
        out.push((timestep * frequency).cos());
    }
    for index in 0..half {
        let frequency = (-10000.0f32.ln() * index as f32 / half as f32).exp();
        out.push((timestep * frequency).sin());
    }
    if dim % 2 == 1 {
        out.push(0.0);
    }
    out
}

fn project_rows(
    input: &[f32],
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
) -> Result<Vec<f32>, String> {
    if input.is_empty() || input.len() % input_dim != 0 {
        return Err("projection input must contain whole nonempty rows".to_string());
    }
    let mut output = Vec::with_capacity(input.len() / input_dim * output_dim);
    for row in input.chunks_exact(input_dim) {
        output.extend(linear_forward(row, weight, bias, output_dim, input_dim));
    }
    Ok(output)
}

fn round_up(value: usize, multiple: usize) -> Result<usize, String> {
    value
        .checked_add(multiple - 1)
        .map(|v| v / multiple * multiple)
        .ok_or_else(|| "sequence length overflow".to_string())
}

fn padded_image_ids(
    token_size: (usize, usize, usize),
    cap_padded_len: usize,
    image_len: usize,
    image_padded_len: usize,
) -> Vec<[i32; 3]> {
    let mut ids = create_coordinate_grid(token_size, (cap_padded_len + 1, 0, 0));
    debug_assert_eq!(ids.len(), image_len);
    ids.resize(image_padded_len, [0, 0, 0]);
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_starts_with_cosine_then_sine() {
        let embedding = timestep_embedding(0.0, 256);
        assert_eq!(embedding.len(), 256);
        assert!(embedding[..128]
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-6));
        assert!(embedding[128..].iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn padding_keeps_image_tokens_attendable() {
        let ids = padded_image_ids((1, 1, 3), 32, 3, 32);
        assert_eq!(ids[..3], [[33, 0, 0], [33, 0, 1], [33, 0, 2]]);
        assert!(ids[3..].iter().all(|id| *id == [0, 0, 0]));
    }
}
