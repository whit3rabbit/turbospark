//! Tensor classification and within-layer slot ordering across families.

use model_io::ModelFamily;

/// Classification bucket for a Gemma 4 source checkpoint tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4Bucket {
    /// Text model resident tensor loaded permanently into RAM/VRAM.
    LmResident,
    /// Routed expert tensor with role ('gate', 'up', 'down') and layer index.
    RoutedExpert {
        /// Role of the expert tensor ('gate', 'up', or 'down').
        role: &'static str,
        /// Layer index containing the expert.
        layer: usize,
    },
    /// Multimodal vision or audio tensor excluded from the text language model.
    ExcludedMultimodal,
    /// Tensor not matching known language model or multimodal patterns.
    Unknown,
}

/// Extracts the layer index from a layer-scoped tensor name (e.g. `...layers.12...`).
pub fn layer_index(name: &str) -> Option<usize> {
    let tail = &name[name.find(".layers.")? + ".layers.".len()..];
    tail[..tail.find('.')?].parse().ok()
}

/// The container path a family's routed (per-expert) weights live under.
pub fn routed_marker(family: ModelFamily) -> &'static str {
    match family {
        // `qwen3_5` is DENSE -- one `mlp.{gate,up,down}_proj` per layer and
        // no routed tensors at all -- so this marker never fires on it. It
        // takes Qwen 3.6's rather than Gemma's because it is that family's
        // safetensors sibling, and because a marker that could only ever
        // match the wrong thing is worse than one that cannot match.
        ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense => ".mlp.switch_mlp.",
        // A GGUF-derived Llama or Qwen3-MoE never reaches this classifier
        // (the GGUF walk maps routed tensors by NAME, in `gguf_names.rs`),
        // and neither has a safetensors path. DeepSeek V4 has no repack path
        // either; Gemma's marker is the default for all of them.
        ModelFamily::Gemma4
        | ModelFamily::DeepseekV4Flash
        | ModelFamily::Llama
        | ModelFamily::Qwen3Moe
        | ModelFamily::GptOss => ".experts.switch_glu.",
    }
}

/// Classifies source tensor name under specified model family contract.
pub fn classify_for_family(name: &str, num_layers: usize, family: ModelFamily) -> Gemma4Bucket {
    if name.starts_with("language_model.") {
        if name.contains(routed_marker(family)) {
            let role = if name.contains(".gate_proj.") {
                Some("gate")
            } else if name.contains(".up_proj.") {
                Some("up")
            } else if name.contains(".down_proj.") {
                Some("down")
            } else {
                None
            };
            if let (Some(role), Some(layer)) = (role, layer_index(name)) {
                if layer < num_layers {
                    return Gemma4Bucket::RoutedExpert { role, layer };
                }
            }
        }
        return Gemma4Bucket::LmResident;
    }
    if name.starts_with("vision_tower.")
        || name.starts_with("embed_vision.")
        || name.starts_with("audio_tower.")
    {
        return Gemma4Bucket::ExcludedMultimodal;
    }
    Gemma4Bucket::Unknown
}

/// Classify a source tensor name under the Gemma 4 family contract: the
/// text tower lives under `language_model.`, routed experts under
/// `.experts.switch_glu.`, and the vision/audio towers are excluded.
pub fn classify_gemma4(name: &str, num_layers: usize) -> Gemma4Bucket {
    classify_for_family(name, num_layers, ModelFamily::Gemma4)
}

/// Within-layer slot order, mirroring `RepackPlanner.swift`'s Gemma table.
pub fn slot_rank(n: &str) -> usize {
    const CONTAINS: [&str; 12] = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
        ".router.proj.weight",
        ".router.scale",
        ".router.per_expert_scale",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ];
    for (rank, pat) in CONTAINS.iter().enumerate() {
        if n.contains(pat) {
            return rank;
        }
    }
    const SUFFIXES: [&str; 8] = [
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".pre_feedforward_layernorm.weight",
        ".pre_feedforward_layernorm_2.weight",
        ".post_feedforward_layernorm.weight",
        ".post_feedforward_layernorm_1.weight",
        ".post_feedforward_layernorm_2.weight",
        ".layer_scalar",
    ];
    for (j, pat) in SUFFIXES.iter().enumerate() {
        if n.ends_with(pat) {
            return CONTAINS.len() + j;
        }
    }
    100
}

/// Stable resident order: embedding first, then per-layer groups in layer
/// order (slot-ranked within a layer), then top-level extras, the final
/// norm, and `lm_head` last.
pub fn lm_order_key(n: &str) -> (usize, usize, usize, &str) {
    match n {
        "language_model.model.embed_tokens.weight" => (0, 0, 0, n),
        "language_model.model.norm.weight" => (3, 0, 0, n),
        "language_model.lm_head.weight" => (4, 0, 0, n),
        _ => match layer_index(n) {
            Some(li) => (1, li, slot_rank(n), n),
            None => (2, 0, 0, n),
        },
    }
}
