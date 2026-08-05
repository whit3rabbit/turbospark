//! A real (not scripted) [`LogitProducer`]: runs an actual dense
//! transformer forward pass through the real GPU kernels
//! `mrefrust_gpu` wires (`rmsnorm_no_scale`, `rope_proportional_neox`,
//! `dequant_int4_gemv_simd`, `logit_softcap_softmax`, and the two-pass
//! split-KV decode `attention_decode`) against real, resident,
//! INT4-affine-quantized weights loaded through `mrefrust_model_io`.
//! macOS/Metal only, matching `mrefrust_gpu`'s own platform gate.
//!
//! What this is NOT: production parity with the Swift `RealForwardRunner`.
//! No GPU MoE/FFN kernel is vendored (see `DEVIATIONS.md`), so both the
//! dense gated FFN and the MoE routed-expert FFN are bridged on the CPU
//! with the already-tested `mrefrust_compute` reference (`run_ffn`)
//! instead of being skipped. For MoE layers, only the router GEMV and each
//! selected expert's gate/up/down GEMVs run on the GPU (the same
//! `dequant_int4_gemv_simd` kernel every other projection uses); top-k
//! expert selection and softmax weighting are plain host arithmetic (no
//! kernel needed — `num_experts` is small enough that this is not a
//! meaningful cost center in a reference implementation), and the
//! weighted combine is a simple accumulate, not
//! `compute::apply_streamed_routed`'s residual-fused form (this runner
//! adds the residual itself, after the same sandwich-norm step the dense
//! path uses). Every other op — embedding lookup, both RMSNorms per
//! layer, both projections' RoPE, causal attention, every INT4 GEMV
//! projection, and the final softcapped-softmax — runs for real on the
//! GPU. KV history is kept in host `Vec<f16>` per layer rather than
//! `gpu::KvCacheManager`'s GPU-resident buffers (a performance
//! simplification, not a correctness one: `gpu::attention_decode` reads
//! this history the same way regardless of where it lives; wiring
//! `KvCacheManager` in is future work).
//!
//! Only all-full-attention architectures are supported (no sliding-window/
//! linear/compressed-attention layers, no hyper-connection residual), with
//! either `num_experts == 0` (dense FFN every layer) or `num_experts > 0`
//! (routed-expert FFN every layer, no separate dense/shared branch summed
//! in alongside it — real Gemma 4 sums both; this runner does only one or
//! the other per architecture). `mrefrust_repack::build_synthetic_gemma4_install`
//! (dense) and `build_synthetic_gemma4_moe_install` (routed) are the two
//! shapes this runner is exercised against, since no trained `.gturbo`
//! checkpoint exists in this environment. A real checkpoint of either exact
//! shape would run through unmodified.

use std::collections::HashMap;
use std::path::Path;

use foundation::LogitValue;
use half::f16;
use model_io::{ArchConfig, ResidentBuffer, ResidentIndex};

use crate::producer::LogitProducer;

#[derive(Debug)]
pub enum RealForwardError {
    Model(model_io::ModelError),
    Gpu(gpu::GpuError),
    MissingTensor(String),
    Unsupported(String),
}

impl std::fmt::Display for RealForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealForwardError::Model(e) => write!(f, "{e}"),
            RealForwardError::Gpu(e) => write!(f, "{e}"),
            RealForwardError::MissingTensor(name) => {
                write!(f, "missing resident tensor: {name}")
            }
            RealForwardError::Unsupported(detail) => write!(f, "unsupported: {detail}"),
        }
    }
}

impl std::error::Error for RealForwardError {}

const RMS_EPS: f32 = 1e-6;

struct CachedScales {
    scales: Vec<u16>,
    biases: Vec<u16>,
}

pub struct RealForwardRunner {
    context: gpu::MetalContext,
    buffer: ResidentBuffer,
    index: ResidentIndex,
    arch: ArchConfig,
    scales: HashMap<String, CachedScales>,
    kv_k: Vec<Vec<f16>>,
    kv_v: Vec<Vec<f16>>,
}

impl RealForwardRunner {
    /// Opens a `.gturbo` install directory whose `manifest.json` matches
    /// `expecting` field-by-field, and whose weights are all resident
    /// (no packed experts): every layer must be full attention
    /// (`full_attention_layer_mask` all `1`). `num_experts` may be `0`
    /// (dense FFN) or positive (routed-expert FFN); see module docs for
    /// what MoE support here does and does not cover.
    pub fn open(dir: &Path, expecting: ArchConfig) -> Result<Self, RealForwardError> {
        if expecting
            .full_attention_layer_mask
            .iter()
            .any(|&kind| kind != 1)
        {
            return Err(RealForwardError::Unsupported(
                "only all-full-attention (dense) architectures are supported".to_string(),
            ));
        }

        model_io::load_manifest(dir, &expecting, model_io::DEFAULT_MAX_BYTES)
            .map_err(RealForwardError::Model)?;
        let index = model_io::load_resident_index(&dir.join("model_weights.bin"))
            .map_err(RealForwardError::Model)?;
        let buffer = ResidentBuffer::map(
            &dir.join("model_weights.bin"),
            index.header.index_size,
            index.header.resident_size,
        )
        .map_err(RealForwardError::Model)?;
        let context = gpu::MetalContext::new().map_err(RealForwardError::Gpu)?;

        let mut scales = HashMap::with_capacity(index.entries.len());
        for (name, entry) in &index.entries {
            let scale_local = (entry.scale_offset - index.header.index_size) as usize;
            let bias_local = (entry.bias_offset - index.header.index_size) as usize;
            let scale_bytes = &buffer.data()[scale_local..scale_local + entry.scale_size as usize];
            let bias_bytes = &buffer.data()[bias_local..bias_local + entry.bias_size as usize];
            scales.insert(
                name.clone(),
                CachedScales {
                    scales: le_bytes_to_u16(scale_bytes),
                    biases: le_bytes_to_u16(bias_bytes),
                },
            );
        }

        let num_layers = expecting.num_layers as usize;
        Ok(Self {
            context,
            buffer,
            index,
            arch: expecting,
            scales,
            kv_k: vec![Vec::new(); num_layers],
            kv_v: vec![Vec::new(); num_layers],
        })
    }
}

fn le_bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn f32_to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn f16_to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

fn tensor_bytes<'a>(
    index: &'a ResidentIndex,
    data: &'a [u8],
    name: &str,
) -> Result<&'a [u8], RealForwardError> {
    let entry = index
        .entries
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    let local = (entry.file_offset - index.header.index_size) as usize;
    Ok(&data[local..local + entry.size_bytes as usize])
}

#[allow(clippy::too_many_arguments)]
fn gpu_rows<'a>(
    index: &'a ResidentIndex,
    data: &'a [u8],
    scales: &'a HashMap<String, CachedScales>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<gpu::Int4AffineRowGpu<'a>>, RealForwardError> {
    let packed_all = tensor_bytes(index, data, name)?;
    let cached = scales
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    let row_bytes = cols / 2;
    let groups = cols / 64;
    Ok((0..rows)
        .map(|r| gpu::Int4AffineRowGpu {
            packed: &packed_all[r * row_bytes..(r + 1) * row_bytes],
            scales: &cached.scales[r * groups..(r + 1) * groups],
            biases: &cached.biases[r * groups..(r + 1) * groups],
        })
        .collect())
}

fn owned_rows(
    index: &ResidentIndex,
    data: &[u8],
    scales: &HashMap<String, CachedScales>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<compute::quant::Int4AffineRow>, RealForwardError> {
    let packed_all = tensor_bytes(index, data, name)?;
    let cached = scales
        .get(name)
        .ok_or_else(|| RealForwardError::MissingTensor(name.to_string()))?;
    let row_bytes = cols / 2;
    let groups = cols / 64;
    Ok((0..rows)
        .map(|r| compute::quant::Int4AffineRow {
            packed: packed_all[r * row_bytes..(r + 1) * row_bytes].to_vec(),
            scales: cached.scales[r * groups..(r + 1) * groups].to_vec(),
            biases: cached.biases[r * groups..(r + 1) * groups].to_vec(),
        })
        .collect())
}

fn layer_name(prefix: &str, layer: usize) -> String {
    format!("layer{layer}.{prefix}")
}

/// Softmax over all `logits`, then the top-`k` entries with their
/// probabilities renormalized to sum to `1` over just the survivors
/// (Gemma 4's `router_scaled` convention). Plain host arithmetic — cheap
/// enough at any real `num_experts` count that no kernel is warranted.
fn topk_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let selected: Vec<usize> = order.into_iter().take(k).collect();

    let selected_sum: f32 = selected.iter().map(|&i| probs[i]).sum();
    let weights: Vec<f32> = selected
        .iter()
        .map(|&i| {
            if selected_sum > 0.0 {
                probs[i] / selected_sum
            } else {
                0.0
            }
        })
        .collect();
    (selected, weights)
}

/// The routed-expert FFN: a real GPU router GEMV, host-side top-k
/// selection, then each selected expert's gate/up/down GEMVs and gated
/// activation via `compute::run_ffn` (see module docs for why this stays
/// CPU-bridged), weighted and summed. Only the selected experts' weights
/// are ever read, not the full expert table.
#[allow(clippy::too_many_arguments)]
fn moe_ffn(
    context: &mut gpu::MetalContext,
    index: &ResidentIndex,
    data: &[u8],
    scale_cache: &HashMap<String, CachedScales>,
    layer: usize,
    x: &[f32],
    hidden: usize,
    moe_inter: usize,
    num_experts: usize,
    top_k: usize,
) -> Result<Vec<f32>, RealForwardError> {
    let x16 = f32_to_f16(x);
    let router_rows = gpu_rows(
        index,
        data,
        scale_cache,
        &layer_name("router", layer),
        num_experts,
        hidden,
    )?;
    let logits16 = gpu::dequant_int4_gemv(context, &router_rows, &x16, hidden)
        .map_err(RealForwardError::Gpu)?;
    let (selected, weights) = topk_softmax(&f16_to_f32(&logits16), top_k);

    let mut combined = vec![0f32; hidden];
    for (&e, &w) in selected.iter().zip(weights.iter()) {
        let gate_rows = owned_rows(
            index,
            data,
            scale_cache,
            &format!("layer{layer}.expert{e}.gate_proj"),
            moe_inter,
            hidden,
        )?;
        let up_rows = owned_rows(
            index,
            data,
            scale_cache,
            &format!("layer{layer}.expert{e}.up_proj"),
            moe_inter,
            hidden,
        )?;
        let down_rows = owned_rows(
            index,
            data,
            scale_cache,
            &format!("layer{layer}.expert{e}.down_proj"),
            hidden,
            moe_inter,
        )?;
        let out = compute::run_ffn(&gate_rows, &up_rows, &down_rows, x, hidden, moe_inter);
        for (c, o) in combined.iter_mut().zip(out.iter()) {
            *c += w * o;
        }
    }
    Ok(combined)
}

impl LogitProducer for RealForwardRunner {
    fn reset(&mut self) {
        for k in &mut self.kv_k {
            k.clear();
        }
        for v in &mut self.kv_v {
            v.clear();
        }
    }

    fn produce(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), String> {
        self.produce_inner(token, position, logits)
            .map_err(|e| e.to_string())
    }
}

impl RealForwardRunner {
    fn produce_inner(
        &mut self,
        token: i32,
        position: usize,
        logits: &mut [LogitValue],
    ) -> Result<(), RealForwardError> {
        let hidden = self.arch.hidden_size as usize;
        let inter = self.arch.intermediate_size as usize;
        let num_heads = self.arch.num_heads as usize;
        let num_kv_heads = self.arch.num_full_kv_heads as usize;
        let head_dim = self.arch.full_head_dim as usize;
        let qk_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let vocab = self.arch.vocab_size as usize;
        let rotated_pairs =
            ((head_dim as f64 * self.arch.partial_rotary_factor) / 2.0).round() as u32;
        let theta = self.arch.full_rope_theta as f32;
        let attn_scale = self.arch.attention_scale as f32;
        let softcap = self.arch.final_logit_softcap as f32;
        let embed_scale = if self.arch.embedding_scaled_by_sqrt_hidden {
            (hidden as f32).sqrt()
        } else {
            1.0
        };

        let context = &mut self.context;
        let data = self.buffer.data();
        let index = &self.index;
        let scale_cache = &self.scales;

        let embed_bytes = tensor_bytes(index, data, "embed_lm_head")?;
        let embed_cached = scale_cache
            .get("embed_lm_head")
            .ok_or_else(|| RealForwardError::MissingTensor("embed_lm_head".to_string()))?;
        let mut x = compute::quant::embed_lookup_int4(
            embed_bytes,
            &embed_cached.scales,
            &embed_cached.biases,
            token as usize,
            hidden,
            embed_scale,
        );

        for layer in 0..self.arch.num_layers as usize {
            let x16 = f32_to_f16(&x);
            let normed16 =
                gpu::rms_norm_no_scale(context, &x16, RMS_EPS).map_err(RealForwardError::Gpu)?;

            let q_rows = gpu_rows(
                index,
                data,
                scale_cache,
                &layer_name("q_proj", layer),
                qk_dim,
                hidden,
            )?;
            let mut q16 = gpu::dequant_int4_gemv(context, &q_rows, &normed16, hidden)
                .map_err(RealForwardError::Gpu)?;
            let k_rows = gpu_rows(
                index,
                data,
                scale_cache,
                &layer_name("k_proj", layer),
                kv_dim,
                hidden,
            )?;
            let mut k16 = gpu::dequant_int4_gemv(context, &k_rows, &normed16, hidden)
                .map_err(RealForwardError::Gpu)?;

            q16 = gpu::rope_proportional_neox(
                context,
                &q16,
                position as u32,
                1,
                num_heads as u32,
                head_dim as u32,
                rotated_pairs,
                theta,
            )
            .map_err(RealForwardError::Gpu)?;
            k16 = gpu::rope_proportional_neox(
                context,
                &k16,
                position as u32,
                1,
                num_kv_heads as u32,
                head_dim as u32,
                rotated_pairs,
                theta,
            )
            .map_err(RealForwardError::Gpu)?;

            let v16 = if self.arch.attention_k_eq_v {
                k16.clone()
            } else {
                return Err(RealForwardError::Unsupported(
                    "only attention_k_eq_v architectures are supported".to_string(),
                ));
            };

            self.kv_k[layer].extend_from_slice(&k16);
            self.kv_v[layer].extend_from_slice(&v16);
            let seq_len = position + 1;

            let attn16 = gpu::attention_decode(
                context,
                &q16,
                &self.kv_k[layer],
                &self.kv_v[layer],
                head_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                seq_len as u32,
                attn_scale,
            )
            .map_err(RealForwardError::Gpu)?;
            let o_rows = gpu_rows(
                index,
                data,
                scale_cache,
                &layer_name("o_proj", layer),
                hidden,
                qk_dim,
            )?;
            let o16 = gpu::dequant_int4_gemv(context, &o_rows, &attn16, qk_dim)
                .map_err(RealForwardError::Gpu)?;
            let post_attn16 = if self.arch.ffn_sandwich_norms {
                gpu::rms_norm_no_scale(context, &o16, RMS_EPS).map_err(RealForwardError::Gpu)?
            } else {
                o16
            };
            let post_attn32 = f16_to_f32(&post_attn16);
            for i in 0..hidden {
                x[i] += post_attn32[i];
            }

            let x16b = f32_to_f16(&x);
            let pre_ffn16 =
                gpu::rms_norm_no_scale(context, &x16b, RMS_EPS).map_err(RealForwardError::Gpu)?;
            let pre_ffn32 = f16_to_f32(&pre_ffn16);

            let ffn_out32 = if self.arch.num_experts > 0 {
                moe_ffn(
                    context,
                    index,
                    data,
                    scale_cache,
                    layer,
                    &pre_ffn32,
                    hidden,
                    self.arch.moe_intermediate_size as usize,
                    self.arch.num_experts as usize,
                    self.arch.top_k_experts as usize,
                )?
            } else {
                let gate_rows = owned_rows(
                    index,
                    data,
                    scale_cache,
                    &layer_name("gate_proj", layer),
                    inter,
                    hidden,
                )?;
                let up_rows = owned_rows(
                    index,
                    data,
                    scale_cache,
                    &layer_name("up_proj", layer),
                    inter,
                    hidden,
                )?;
                let down_rows = owned_rows(
                    index,
                    data,
                    scale_cache,
                    &layer_name("down_proj", layer),
                    hidden,
                    inter,
                )?;
                compute::run_ffn(&gate_rows, &up_rows, &down_rows, &pre_ffn32, hidden, inter)
            };

            let ffn16 = f32_to_f16(&ffn_out32);
            let post_ffn16 = if self.arch.ffn_sandwich_norms {
                gpu::rms_norm_no_scale(context, &ffn16, RMS_EPS).map_err(RealForwardError::Gpu)?
            } else {
                ffn16
            };
            let post_ffn32 = f16_to_f32(&post_ffn16);
            for i in 0..hidden {
                x[i] += post_ffn32[i];
            }
        }

        let x16 = f32_to_f16(&x);
        let normed_final16 =
            gpu::rms_norm_no_scale(context, &x16, RMS_EPS).map_err(RealForwardError::Gpu)?;
        let lm_rows = gpu_rows(index, data, scale_cache, "embed_lm_head", vocab, hidden)?;
        let raw_logits16 = gpu::dequant_int4_gemv(context, &lm_rows, &normed_final16, hidden)
            .map_err(RealForwardError::Gpu)?;
        let probs16 = gpu::logit_softcap_softmax(context, &raw_logits16, softcap)
            .map_err(RealForwardError::Gpu)?;

        if probs16.len() != logits.len() {
            return Err(RealForwardError::Unsupported(format!(
                "vocab mismatch: model has {}, caller expected {}",
                probs16.len(),
                logits.len()
            )));
        }
        logits.copy_from_slice(&probs16);
        Ok(())
    }
}
