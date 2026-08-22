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
    /// A multi-token-prediction head tensor (`mtp.*`), ingested as a
    /// speculative drafter (`docs/MTP_SPECULATIVE.md`).
    MtpHead,
    /// A DFlash2 block-diffusion drafter tensor (`dflash.*`), the second
    /// speculative drafter this walk knows (`docs/DFLASH2.md`).
    DflashDrafter,
    /// Tensor not matching known language model or multimodal patterns.
    Unknown,
}

/// The prefix the multi-token-prediction head's tensors carry.
pub const MTP_PREFIX: &str = "mtp.";

/// The prefix a DFlash2 drafter's tensors carry once they reach a walk.
///
/// **The published repository spells its tensors BARE** (`layers.0.*`,
/// `fc.weight`, `candidate_selector.*`), which no `classify_for_family` arm
/// could ever match: the CALLER renames the shard header's names onto this
/// prefix before handing the pair to `Gemma4Shards`, for the same reason the
/// official Qwen shard's `model.language_model.` spelling is renamed by being
/// read through the conversion instead. Keeping the namespace HERE rather
/// than classifying bare names is what keeps `fc.weight` -- a name any model
/// could carry -- from colliding with a future trunk tensor.
pub const DFLASH_PREFIX: &str = "dflash.";

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
        // `muse_glimmer` is DENSE too, and unlike `qwen3_5` it has no MoE
        // sibling whose marker it could borrow, so it takes the default for
        // the same reason the GGUF-only families do: the marker can never
        // match, and one that could only ever match the wrong thing is worse
        // than one that cannot match.
        ModelFamily::Gemma4
        | ModelFamily::DeepseekV4Flash
        | ModelFamily::Llama
        | ModelFamily::Qwen3Moe
        | ModelFamily::GptOss
        | ModelFamily::MuseGlimmer => ".experts.switch_glu.",
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
    // THE MULTI-TOKEN-PREDICTION HEAD, gated on the family rather than
    // accepted everywhere: an `mtp.` tensor under a family that has no
    // drafter is a checkpoint this walk has never seen, and falling
    // through to `Unknown` refuses it by name rather than ingesting a head
    // no decode flow would look for.
    //
    // **BOTH HALVES OF `qwen3_5`, not just the dense one.** This read
    // `== QwenGdnDense` until ROADMAP Phase 3, on the true observation
    // that `Qwen/Qwen3.8-27B` was the only published carrier. It is
    // AGENTS.md Gotcha 61's shape -- one arm of a shared architecture
    // naming one family -- and the MoE half is a real carrier now:
    // Ornith-1.5-35B-A3B's BF16 repo ships a head in its last shard.
    //
    // Widening admits nothing that exists today. Every mlx conversion of
    // the MoE half DROPS `mtp.*` (verified off the published indexes:
    // `Qwen3.6-35B-A3B-4bit` and `Ornith-1.5-35B-A3B-MLX-4bit` carry zero),
    // so no install on disk grows a head by this arm existing. What it
    // enables is the fixture the batched routed verify is gated by, and
    // eventually the real head. NOTE the head it would ingest from Ornith
    // is itself MoE where `MtpState::REQUIRED` names DENSE FFN tensors, so
    // that stream still fails at open naming the tensor it wanted -- which
    // is the loud failure, and better than reporting no head at all.
    if name.starts_with(MTP_PREFIX)
        && matches!(family, ModelFamily::QwenGdnDense | ModelFamily::QwenGdnMoe)
    {
        return Gemma4Bucket::MtpHead;
    }
    // THE DFLASH2 DRAFTER, gated the same way and for the same reason: the
    // one published checkpoint (`incoai/Qwen3.8-27B-DFlash2`) targets this
    // family's 27B model, and a `dflash.` tensor under any other family is a
    // pairing this walk has never seen, which `Unknown` refuses by name
    // rather than ingesting a drafter no decode flow would look for.
    if name.starts_with(DFLASH_PREFIX) && family == ModelFamily::QwenGdnDense {
        return Gemma4Bucket::DflashDrafter;
    }
    // THIS LIST IS READ OFF REAL CHECKPOINT HEADERS, one prefix per
    // publisher's naming, and it is not guesswork: an unlisted prefix falls
    // through to `Gemma4Bucket::Unknown`, which the walk refuses. That is the
    // right failure mode and it is an EXPENSIVE one to discover, because
    // nothing sees it until a multi-GB stream reaches the shard the tensor
    // lives in.
    //
    // The last three are `mlx-community/Muse-Glimmer-30B-4bit`'s, enumerated
    // from its `model.safetensors.index.json` before any bytes moved: it
    // splits its vision side three ways where Gemma keeps it under one
    // prefix (806 `vision_tower.` tensors, 6 `vision_adapter.`, 3
    // `vision_projection.`). `perception_emb_norm` is deliberately NOT here
    // -- the reference makes it a no-scale RMSNorm, which has no weight, so
    // it appears in no checkpoint.
    if name.starts_with("vision_tower.")
        || name.starts_with("embed_vision.")
        || name.starts_with("audio_tower.")
        || name.starts_with("vision_adapter.")
        || name.starts_with("vision_projection.")
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
