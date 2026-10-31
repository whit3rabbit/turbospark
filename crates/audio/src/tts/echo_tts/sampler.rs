//! The Echo-TTS euler flow sampler (`sampling.py`): independent
//! text/speaker classifier-free guidance over a tripled batch, optional
//! temporal score rescaling, optional speaker-KV scaling with a
//! mid-trajectory restore, and the blockwise (continuation) variant
//! over latent-prefix KV caches.
//!
//! Divergence by design: the reference seeds `mx.random` for the
//! initial noise; the draws here come from the injected [`GaussianNoise`]
//! source in reference draw order, so a recorded replay reproduces the
//! reference trajectory exactly and a seeded crate RNG gives
//! deterministic standalone runs.

use super::dit::{concat_kv, multiply_kv, EchoDiT, KvCache};

/// The gaussian draw source of the sampler (one flat draw per call, in
/// reference order).
pub trait GaussianNoise {
    fn normal(&mut self, n: usize) -> Vec<f32>;
}

/// Crate-local threefry seeded draws for standalone runs.
pub struct SeededGaussian {
    key: crate::music::minimax_music3::rng::Key,
}

impl SeededGaussian {
    pub fn new(seed: u64) -> Self {
        SeededGaussian {
            key: crate::music::minimax_music3::rng::Key::new(seed),
        }
    }
}

impl GaussianNoise for SeededGaussian {
    fn normal(&mut self, n: usize) -> Vec<f32> {
        crate::music::minimax_music3::rng::normal(self.key, n)
    }
}

/// The sampler controls (the loaded `SamplerConfig` values; the
/// `sequence_length` field stays on the generate path).
#[derive(Debug, Clone)]
pub struct SamplerParams {
    pub num_steps: usize,
    pub cfg_scale_text: f32,
    pub cfg_scale_speaker: f32,
    pub cfg_min_t: f32,
    pub cfg_max_t: f32,
    pub truncation_factor: Option<f32>,
    pub rescale_k: Option<f32>,
    pub rescale_sigma: Option<f32>,
    pub speaker_kv_scale: Option<f32>,
    pub speaker_kv_max_layers: Option<usize>,
    pub speaker_kv_min_t: Option<f32>,
}

/// `DEFAULT_TRUNCATION_FACTOR`.
pub const DEFAULT_TRUNCATION_FACTOR: f32 = 0.96;

/// The `np.linspace(0.999, 0.0, num_steps + 1)` schedule over f32.
pub fn t_schedule(num_steps: usize) -> Vec<f32> {
    let start = 0.999f32;
    (0..=num_steps)
        .map(|i| {
            if num_steps == 0 {
                start
            } else {
                start + (0.0 - start) * (i as f32 / num_steps as f32)
            }
        })
        .collect()
}

/// `_temporal_score_rescale`.
pub fn temporal_score_rescale(
    v_pred: &[f32],
    x_t: &[f32],
    t: f32,
    rescale_k: f32,
    rescale_sigma: f32,
) -> Vec<f32> {
    if t < 1.0 {
        let snr = (1.0 - t).powi(2) / (t * t);
        let ratio = (snr * rescale_sigma * rescale_sigma + 1.0)
            / (snr * rescale_sigma * rescale_sigma / rescale_k + 1.0);
        v_pred
            .iter()
            .zip(x_t)
            .map(|(&v, &x)| (1.0 / (1.0 - t)) * (ratio * ((1.0 - t) * v + x) - x))
            .collect()
    } else {
        v_pred.to_vec()
    }
}

/// Guidance and bookkeeping shared by both samplers.
struct Guidance<'a> {
    params: &'a SamplerParams,
    text_mask: &'a [bool],
    speaker_mask: &'a [bool],
    full_text_mask: Vec<bool>,
    full_speaker_mask: Vec<bool>,
}

impl<'a> Guidance<'a> {
    /// Text `[cond, uncond, cond]`, speaker `[cond, cond, uncond]`.
    fn new(params: &'a SamplerParams, text_mask: &'a [bool], speaker_mask: &'a [bool]) -> Self {
        let mut full_text_mask = Vec::with_capacity(text_mask.len() * 3);
        full_text_mask.extend_from_slice(text_mask);
        full_text_mask.extend(std::iter::repeat_n(false, text_mask.len()));
        full_text_mask.extend_from_slice(text_mask);
        let mut full_speaker_mask = Vec::with_capacity(speaker_mask.len() * 3);
        full_speaker_mask.extend_from_slice(speaker_mask);
        full_speaker_mask.extend_from_slice(speaker_mask);
        full_speaker_mask.extend(std::iter::repeat_n(false, speaker_mask.len()));
        Guidance {
            params,
            text_mask,
            speaker_mask,
            full_text_mask,
            full_speaker_mask,
        }
    }

    /// One euler step. Returns the guided (or plain) velocity prediction.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        model: &EchoDiT,
        x_t: &[f32],
        t: f32,
        batch: usize,
        kv_text_cond: &[KvCache],
        kv_speaker_cond: &[KvCache],
        kv_text_full: &[KvCache],
        kv_speaker_full: &[KvCache],
        kv_latent_full: Option<&[KvCache]>,
        kv_latent_cond: Option<&[KvCache]>,
        start_pos: usize,
    ) -> crate::Result<Vec<f32>> {
        let has_cfg = self.params.cfg_min_t <= t && t <= self.params.cfg_max_t;
        let v_pred = if has_cfg {
            let mut x_full = vec![0.0f32; x_t.len() * 3];
            for b in 0..3 {
                x_full[b * x_t.len()..(b + 1) * x_t.len()].copy_from_slice(x_t);
            }
            let t_full = vec![t; batch * 3];
            let out = model.forward(
                &x_full,
                &t_full,
                &self.full_text_mask,
                &self.full_speaker_mask,
                kv_text_full,
                kv_speaker_full,
                start_pos,
                kv_latent_full,
            )?;
            let third = out.len() / 3;
            let (v_cond, rest) = out.split_at(third);
            let (v_uncond_text, v_uncond_speaker) = rest.split_at(third);
            let mut guided = vec![0.0f32; third];
            for j in 0..third {
                guided[j] = v_cond[j]
                    + self.params.cfg_scale_text * (v_cond[j] - v_uncond_text[j])
                    + self.params.cfg_scale_speaker * (v_cond[j] - v_uncond_speaker[j]);
            }
            guided
        } else {
            let t_cond = vec![t; batch];
            model.forward(
                x_t,
                &t_cond,
                self.text_mask,
                self.speaker_mask,
                kv_text_cond,
                kv_speaker_cond,
                start_pos,
                kv_latent_cond,
            )?
        };
        Ok(v_pred)
    }
}

/// The speaker-KV crossing test: `t_next < min_t <= t`.
fn kv_crossing(params: &SamplerParams, t: f32, t_next: f32) -> bool {
    match (params.speaker_kv_scale, params.speaker_kv_min_t) {
        (Some(_), Some(min_t)) => t_next < min_t && min_t <= t,
        _ => false,
    }
}

/// `sample_euler_cfg_independent_guidances`: the whole-sequence sampler.
/// `sequence_length` is the latent frames generated in one pass.
#[allow(clippy::too_many_arguments)]
pub fn sample_euler_cfg_independent_guidances(
    model: &EchoDiT,
    speaker_latent: &[f32],
    speaker_latent_rows: usize,
    speaker_mask: &[bool],
    text_ids: &[i32],
    text_mask: &[bool],
    params: &SamplerParams,
    sequence_length: usize,
    noise: &mut dyn GaussianNoise,
) -> crate::Result<Vec<f32>> {
    let batch = 1usize;
    let schedule = t_schedule(params.num_steps);
    let guidance = Guidance::new(params, text_mask, speaker_mask);

    let kv_text_cond = model.get_kv_cache_text(text_ids, text_mask)?;
    let mut kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent, speaker_latent_rows);
    if let Some(scale) = params.speaker_kv_scale {
        kv_speaker_cond = multiply_kv(&kv_speaker_cond, scale, params.speaker_kv_max_layers);
    }
    let kv_text_full: Vec<KvCache> = (0..model.num_blocks())
        .map(|i| concat_kv(&[&kv_text_cond[i], &kv_text_cond[i], &kv_text_cond[i]]))
        .collect();
    let mut kv_speaker_full: Vec<KvCache> = (0..model.num_blocks())
        .map(|i| {
            concat_kv(&[
                &kv_speaker_cond[i],
                &kv_speaker_cond[i],
                &kv_speaker_cond[i],
            ])
        })
        .collect();

    let trunc = params
        .truncation_factor
        .unwrap_or(DEFAULT_TRUNCATION_FACTOR);
    let latent = model.latent_size();
    let mut x_t = noise.normal(batch * sequence_length * latent);
    for value in &mut x_t {
        *value *= trunc;
    }

    for i in 0..params.num_steps {
        let t = schedule[i];
        let t_next = schedule[i + 1];
        let mut v = guidance.step(
            model,
            &x_t,
            t,
            batch,
            &kv_text_cond,
            &kv_speaker_cond,
            &kv_text_full,
            &kv_speaker_full,
            None,
            None,
            0,
        )?;
        if let (Some(rescale_k), Some(rescale_sigma)) = (params.rescale_k, params.rescale_sigma) {
            v = temporal_score_rescale(&v, &x_t, t, rescale_k, rescale_sigma);
        }
        if kv_crossing(params, t, t_next) {
            if let Some(scale) = params.speaker_kv_scale {
                kv_speaker_cond =
                    multiply_kv(&kv_speaker_cond, 1.0 / scale, params.speaker_kv_max_layers);
                kv_speaker_full = (0..model.num_blocks())
                    .map(|b| {
                        concat_kv(&[
                            &kv_speaker_cond[b],
                            &kv_speaker_cond[b],
                            &kv_speaker_cond[b],
                        ])
                    })
                    .collect();
            }
        }
        for (x, value) in x_t.iter_mut().zip(&v) {
            *x += value * (t_next - t);
        }
    }
    Ok(x_t)
}

/// `sample_blockwise_euler_cfg_independent_guidances`: the continuation
/// sampler over `block_sizes` latent blocks with the latent-prefix KV.
/// `continuation_latent` is `(rows x latent, rows)`.
#[allow(clippy::too_many_arguments)]
pub fn sample_blockwise_euler_cfg_independent_guidances(
    model: &EchoDiT,
    speaker_latent: &[f32],
    speaker_latent_rows: usize,
    speaker_mask: &[bool],
    text_ids: &[i32],
    text_mask: &[bool],
    block_sizes: &[usize],
    params: &SamplerParams,
    continuation_latent: Option<(&[f32], usize)>,
    noise: &mut dyn GaussianNoise,
) -> crate::Result<Vec<f32>> {
    let batch = 1usize;
    let schedule = t_schedule(params.num_steps);
    let guidance = Guidance::new(params, text_mask, speaker_mask);

    let kv_text_cond = model.get_kv_cache_text(text_ids, text_mask)?;
    let mut kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent, speaker_latent_rows);
    let kv_text_full: Vec<KvCache> = (0..model.num_blocks())
        .map(|i| concat_kv(&[&kv_text_cond[i], &kv_text_cond[i], &kv_text_cond[i]]))
        .collect();

    let mut generated: Vec<f32> = Vec::new();
    let mut generated_rows = 0usize;
    if let Some((cont, rows)) = continuation_latent {
        generated.extend_from_slice(cont);
        generated_rows = rows;
    }
    let trunc = params
        .truncation_factor
        .unwrap_or(DEFAULT_TRUNCATION_FACTOR);
    let latent = model.latent_size();
    let mut start_pos = generated_rows;

    for &block_size in block_sizes {
        // The cond caches persist and re-scale across blocks exactly
        // like the reference loop (a crossing restores them inside the
        // euler loop of the block that crosses).
        if let Some(scale) = params.speaker_kv_scale {
            kv_speaker_cond = multiply_kv(&kv_speaker_cond, scale, params.speaker_kv_max_layers);
        }
        let kv_speaker_full: Vec<KvCache> = (0..model.num_blocks())
            .map(|i| {
                concat_kv(&[
                    &kv_speaker_cond[i],
                    &kv_speaker_cond[i],
                    &kv_speaker_cond[i],
                ])
            })
            .collect();

        // Latent-prefix KV over the tripled prefix batch; the cond copy
        // is the first batch slice (`kv_latent_cond`).
        let tripled = batch * 3;
        let mut prefix = generated.clone();
        prefix.extend_from_slice(&generated);
        prefix.extend_from_slice(&generated);
        let kv_latent_full = model.get_kv_cache_latent(&prefix, tripled, generated_rows)?;
        let kv_latent_cond: Vec<KvCache> = kv_latent_full
            .iter()
            .map(|layer| {
                let heads_dim = if layer.keys > 0 && !layer.k.is_empty() {
                    layer.k.len() / (layer.keys * layer.batch)
                } else {
                    0
                };
                let span = layer.keys * heads_dim;
                KvCache {
                    k: layer.k[..span].to_vec(),
                    v: layer.v[..span].to_vec(),
                    keys: layer.keys,
                    batch,
                }
            })
            .collect();

        let mut x_t = noise.normal(batch * block_size * latent);
        for value in &mut x_t {
            *value *= trunc;
        }

        let mut speaker_full = kv_speaker_full;
        for i in 0..params.num_steps {
            let t = schedule[i];
            let t_next = schedule[i + 1];
            let mut v = guidance.step(
                model,
                &x_t,
                t,
                batch,
                &kv_text_cond,
                &kv_speaker_cond,
                &kv_text_full,
                &speaker_full,
                Some(&kv_latent_full),
                Some(&kv_latent_cond),
                start_pos,
            )?;
            if let (Some(rescale_k), Some(rescale_sigma)) = (params.rescale_k, params.rescale_sigma)
            {
                v = temporal_score_rescale(&v, &x_t, t, rescale_k, rescale_sigma);
            }
            if kv_crossing(params, t, t_next) {
                if let Some(scale) = params.speaker_kv_scale {
                    kv_speaker_cond =
                        multiply_kv(&kv_speaker_cond, 1.0 / scale, params.speaker_kv_max_layers);
                    speaker_full = (0..model.num_blocks())
                        .map(|b| {
                            concat_kv(&[
                                &kv_speaker_cond[b],
                                &kv_speaker_cond[b],
                                &kv_speaker_cond[b],
                            ])
                        })
                        .collect();
                }
            }
            for (x, value) in x_t.iter_mut().zip(&v) {
                *x += value * (t_next - t);
            }
        }
        generated.extend_from_slice(&x_t);
        generated_rows += block_size;
        start_pos += block_size;
    }
    Ok(generated)
}
