//! `RealUnlimitedOcr2State`: what the runner allocates once for an
//! `unlimited_ocr2` install, plus the router's per-generation host state.
//!
//! One decoder layer:
//!
//! ```text
//! a      = rms_norm(x, input_layernorm)              // eps 1e-6; commit, read
//! q_i,k_i= router(a), rope                           // host, FP32 (compute::dash)
//! keys   = dash_select(q_i, summaries, candidates)   // host; k_i joins its block
//! q      = rope(a @ q_proj)     k = rope(a @ k_proj) // page2d angles, written
//! v      = a @ v_proj                                //   straight into the cache
//! o      = biased_attn(q, K[keys], V[keys], bias)    // scale head_dim^-0.5
//! x      = x + o_proj(o)
//! f      = rms_norm(x, post_attention_layernorm)
//! layer 0:  x += dense SwiGLU(f)                     // dense lead
//! else:     h = softmax(f @ gate) top-6, NO renorm   // BF16 gate; commit, read
//!           x += routed_sum + shared SwiGLU(f)       // fused 2-expert shexp
//! ```
//!
//! Final: `rms_norm(x, model.norm)`, raw-logit head (untied).
//!
//! The router's state cannot be rebuilt from token ids, because an image row's
//! key depends on pixels. So there is no prefix reuse and no rollback here:
//! `reset` rewinds to an empty context and every generation prefills.

use compute::dash::{
    page2d_inv_freq, DashDecodeState, DashParams, DashSelection, PromptPages, RouterWeights,
};
use model_io::{ArchConfig, DashConfig, ModelFamily, ResidentIndex};

use crate::real_forward::RealForwardError;
use crate::real_forward_utils::read_bf16_host;

use super::layer_tensor;

/// One layer's router state: the completed blocks' summaries and the index
/// keys of the block still filling.
#[derive(Default)]
pub(crate) struct LayerRouter {
    pub(crate) summaries: Vec<f32>,
    pub(crate) pending: Vec<f32>,
}

pub(crate) struct RealUnlimitedOcr2State {
    pub(crate) rms_eps: f32,
    pub(crate) dir: std::path::PathBuf,
    /// Opened on the first image, so a text prompt never reads the file.
    pub(crate) tower: Option<crate::vision::unlimited_ocr2::Uocr2Tower>,
    pub(crate) dash: DashConfig,
    pub(crate) params: DashParams,
    pub(crate) routers: Vec<RouterWeights>,
    pub(crate) layers: Vec<LayerRouter>,

    /// Page bookkeeping for the current prompt. Survives `reset`, the way
    /// `prompt_vision` does: a caller sets it just before the generation
    /// loop's own reset. Empty means a text prompt.
    pub(crate) pages: PromptPages,
    pub(crate) prompt_ids: Vec<i32>,
    /// `[image positions, hidden]` FP16 rows, in position order, and which
    /// row each prompt position takes (`None` for a text token).
    pub(crate) image_rows: Vec<half::f16>,
    pub(crate) image_row_at: Vec<Option<u32>>,
    pub(crate) decode: DashDecodeState,

    pub(crate) inv_tok: Vec<f32>,
    pub(crate) inv_page: Vec<f32>,
    /// The 1-D yarn frequencies a prompt with no page marker rotates by.
    pub(crate) yarn: Vec<f32>,

    // ---- host-written per layer; each is read inside a pass that finishes
    // before the next write, because every layer commits and waits ----
    pub(crate) angles: Vec<f32>,
    pub(crate) angles_buffer: gpu::MetalBuffer,
    pub(crate) selection: DashSelection,
    pub(crate) candidates: Vec<bool>,
    pub(crate) positions: gpu::MetalBuffer,
    pub(crate) bias: gpu::MetalBuffer,
    pub(crate) count: gpu::MetalBuffer,

    // ---- FFN scratch, the deepseek2 shapes ----
    pub(crate) ffn_a: gpu::MetalBuffer,
    pub(crate) ffn_b: gpu::MetalBuffer,
    pub(crate) shared_out: gpu::MetalBuffer,
    pub(crate) moe_x: gpu::MetalBuffer,
    pub(crate) h2: gpu::MetalBuffer,
    pub(crate) lead: usize,
    pub(crate) dense_inter: usize,
    pub(crate) shared_inter: usize,
}

fn empty_prompt(params: &DashParams) -> PromptPages {
    PromptPages::new(&[], params)
}

impl RealUnlimitedOcr2State {
    pub(crate) fn build(
        context: &mut gpu::MetalContext,
        weights: &gpu::ResidentGpuWeights,
        index: &ResidentIndex,
        arch: &ArchConfig,
        dir: &std::path::Path,
    ) -> Result<Self, RealForwardError> {
        let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
        if arch.family != ModelFamily::UnlimitedOcr2 {
            return unsupported("unlimited_ocr2 state built for another family".into());
        }
        if arch.ffn_sandwich_norms
            || arch.router_scaled
            || arch.embedding_scaled_by_sqrt_hidden
            || arch.attn_output_gate
            || arch.shared_expert_gated
            || arch.attention_k_eq_v
            || arch.final_logit_softcap != 0.0
        {
            return unsupported(
                "the unlimited_ocr2 flow takes none of ffnSandwichNorms / routerScaled / \
                 embeddingScaledBySqrtHidden / attnOutputGate / sharedExpertGated / \
                 attentionKEqV / a logit softcap"
                    .to_string(),
            );
        }
        if arch.full_attention_layer_mask.iter().any(|&m| m != 1) {
            return unsupported("every unlimited_ocr2 layer keeps a full KV (mask 1)".into());
        }
        if arch.num_heads != arch.num_kv_heads || arch.head_dim != arch.full_head_dim {
            return unsupported("unlimited_ocr2 needs one attention geometry".into());
        }
        let lead = validate_dense_lead(arch)?;
        let num_layers = arch.num_layers as usize;
        let top_k = arch.top_k_experts as usize;
        if top_k == 0 || top_k > gpu::MAX_STREAMED_EXPERTS {
            return unsupported(format!(
                "top_k {top_k} exceeds the {}-slot MoE kernels",
                gpu::MAX_STREAMED_EXPERTS
            ));
        }
        let head_dim = arch.head_dim as usize;
        let hidden = arch.hidden_size as usize;
        let dash =
            DashConfig::load(dir, num_layers, head_dim).map_err(RealForwardError::Unsupported)?;
        let params = DashParams {
            block_size: dash.block_size,
            local_blocks: dash.local_blocks,
            index_dim: dash.index_dim,
            topk: dash.topk,
            topm: model_io::DASH_INFER_TOPM,
            sigma: dash.sigma,
            image_token_id: dash.image_token_id,
            page_token_id: dash.page_token_id,
            per_page_image_tokens: dash.per_page_image_tokens,
            rope_page_mod: dash.rope_page_mod,
        };

        // Name every tensor the flow reads, so a gap fails at open.
        for (layer, suffix) in [
            (0usize, "mlp.gate_proj.weight"),
            (0, "mlp.up_proj.weight"),
            (0, "mlp.down_proj.weight"),
            (lead, "mlp.gate.weight"),
            (lead, "mlp.shared_expert.gate_proj.weight"),
            (lead, "mlp.shared_expert.up_proj.weight"),
            (lead, "mlp.shared_expert.down_proj.weight"),
        ] {
            let name = layer_tensor(layer, suffix);
            if !index.entries.contains_key(&name) {
                return Err(RealForwardError::MissingTensor(name));
            }
        }

        // Widened once: the router runs on the host, so these are FP32 copies
        // of small dense tensors, about 660 KiB per layer.
        let mut routers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let read = |suffix: &str, want: usize| -> Result<Vec<f32>, RealForwardError> {
                let name = layer_tensor(layer, &format!("self_attn.dash_router.{suffix}"));
                let values = read_bf16_host(weights, index, &name)?;
                if values.len() != want {
                    return Err(RealForwardError::Unsupported(format!(
                        "tensor {name}: {} elements, expected {want}",
                        values.len()
                    )));
                }
                Ok(values)
            };
            let d = dash.index_dim;
            routers.push(RouterWeights {
                q_proj: read("q_proj.weight", d * hidden)?,
                k_proj: read("k_proj.weight", d * hidden)?,
                q_norm: read("q_norm.weight", d)?,
                k_norm: read("k_norm.weight", d)?,
                head_cls: read("head_cls", d)?,
            });
        }

        let (inv_tok, inv_page) = page2d_inv_freq(
            head_dim,
            dash.rope_tok_theta,
            dash.rope_page_theta,
            dash.rope_page_pairs,
        );
        let yarn = compute::yarn_frequencies(
            head_dim,
            arch.full_rope_theta as f32,
            arch.rope_scaling.factor as f32,
            arch.rope_scaling.original_context,
            arch.rope_scaling.beta_fast as f32,
            arch.rope_scaling.beta_slow as f32,
        )
        .frequencies;

        let bf16 = |n: usize| (n * 2) as u64;
        let ffn_width = arch
            .dense_lead_intermediate_size
            .max(arch.intermediate_size) as usize;
        let max_keys = params.max_keys() as u64;
        let pages = empty_prompt(&params);
        let decode = DashDecodeState::start(&pages);
        Ok(Self {
            rms_eps: 1e-6,
            dir: dir.to_path_buf(),
            tower: None,
            layers: (0..num_layers).map(|_| LayerRouter::default()).collect(),
            routers,
            pages,
            prompt_ids: Vec::new(),
            image_rows: Vec::new(),
            image_row_at: Vec::new(),
            decode,
            inv_tok,
            inv_page,
            yarn,
            angles: vec![0.0; head_dim / 2],
            angles_buffer: context.new_output_buffer((head_dim / 2 * 4) as u64),
            selection: DashSelection::default(),
            candidates: Vec::new(),
            positions: context.new_output_buffer(max_keys * 4),
            bias: context.new_output_buffer(max_keys * 4),
            count: context.new_output_buffer(4),
            ffn_a: context.new_output_buffer(bf16(ffn_width)),
            ffn_b: context.new_output_buffer(bf16(ffn_width)),
            shared_out: context.new_output_buffer(bf16(hidden)),
            moe_x: context.new_output_buffer(bf16(hidden)),
            h2: context.new_output_buffer(bf16(hidden)),
            lead,
            dense_inter: arch.dense_lead_intermediate_size as usize,
            shared_inter: arch.intermediate_size as usize,
            dash,
            params,
        })
    }

    /// Rewind to an empty context, keeping the prompt it was given.
    pub(crate) fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.summaries.clear();
            layer.pending.clear();
        }
        self.decode = DashDecodeState::start(&self.pages);
    }

    /// Install the next generation's prompt. `image_rows` holds one `hidden`
    /// row per image token, in order.
    pub(crate) fn set_prompt(
        &mut self,
        ids: &[i32],
        image_rows: Vec<half::f16>,
        hidden: usize,
    ) -> Result<(), RealForwardError> {
        let images = ids
            .iter()
            .filter(|&&id| id == self.params.image_token_id)
            .count();
        if image_rows.len() != images * hidden {
            return Err(RealForwardError::Unsupported(format!(
                "prompt has {images} image positions but {} embedding rows",
                image_rows.len() / hidden.max(1)
            )));
        }
        let mut next = 0u32;
        self.image_row_at = ids
            .iter()
            .map(|&id| {
                (id == self.params.image_token_id).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        self.image_rows = image_rows;
        self.prompt_ids = ids.to_vec();
        self.pages = PromptPages::new(ids, &self.params);
        self.reset();
        Ok(())
    }

    /// Back to a text prompt.
    pub(crate) fn clear_prompt(&mut self) {
        self.image_rows = Vec::new();
        self.image_row_at = Vec::new();
        self.prompt_ids = Vec::new();
        self.pages = empty_prompt(&self.params);
        self.reset();
    }

    /// The per-pair angles for the token entering `position`, advancing the
    /// decode page state past the prompt.
    pub(crate) fn angles_for(&mut self, token: i32, position: usize) {
        use compute::dash::page2d_angles;
        let coords = if position < self.pages.length {
            self.pages
                .coords
                .as_ref()
                .map(|(tok, page)| (tok[position], page[position]))
        } else {
            self.decode.advance(token, &self.pages, &self.params)
        };
        match coords {
            Some((tok, page)) => page2d_angles(
                tok,
                page,
                self.params.rope_page_mod,
                &self.inv_tok,
                &self.inv_page,
                &mut self.angles,
            ),
            None => {
                for (a, f) in self.angles.iter_mut().zip(&self.yarn) {
                    *a = position as f32 * f;
                }
            }
        }
    }
}

/// Check the dense-lead / MoE split and return the lead count.
///
/// The flow needs at least one dense lead layer: the open-time tensor probe
/// and the forward pass both treat layer 0 as dense (`mlp.gate_proj.weight`),
/// and the real checkpoint has a lead of 1 (`unlimited_ocr2_12l`). A lead of 0
/// would otherwise pass the old `lead >= num_layers` test and only fail later
/// as a `MissingTensor` on layer 0, or worse, mis-slice the expert blobs.
fn validate_dense_lead(arch: &ArchConfig) -> Result<usize, RealForwardError> {
    let unsupported = |detail: String| Err(RealForwardError::Unsupported(detail));
    let lead = arch.num_dense_leading_layers.max(0) as usize;
    let num_layers = arch.num_layers.max(0) as usize;
    if lead == 0 {
        return unsupported(format!(
            "unlimited_ocr2 needs a dense lead: num_dense_leading_layers is {}, expected 1..{num_layers}",
            arch.num_dense_leading_layers
        ));
    }
    if lead >= num_layers || arch.dense_lead_intermediate_size <= 0 {
        return unsupported("unlimited_ocr2 needs a dense lead and MoE layers".into());
    }
    Ok(lead)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch(lead: i64, layers: i64, dense_inter: i64) -> ArchConfig {
        let mut a = model_io::unlimited_ocr2_12l();
        a.num_dense_leading_layers = lead;
        a.num_layers = layers;
        a.dense_lead_intermediate_size = dense_inter;
        a
    }

    fn err(a: &ArchConfig) -> String {
        match validate_dense_lead(a) {
            Err(RealForwardError::Unsupported(m)) => m,
            other => panic!(
                "expected Unsupported, got {:?}",
                other.map_err(|e| e.to_string())
            ),
        }
    }

    #[test]
    fn real_baseline_lead_is_accepted() {
        assert_eq!(
            validate_dense_lead(&model_io::unlimited_ocr2_12l()).unwrap(),
            1
        );
    }

    #[test]
    fn zero_dense_lead_is_refused_naming_the_field() {
        let m = err(&arch(0, 12, 6848));
        assert!(m.contains("num_dense_leading_layers"), "{m}");
    }

    #[test]
    fn lead_covering_every_layer_or_zero_width_is_refused() {
        err(&arch(12, 12, 6848));
        err(&arch(1, 12, 0));
    }
}
