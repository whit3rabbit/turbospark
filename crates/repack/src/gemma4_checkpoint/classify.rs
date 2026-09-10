//! Tensor classification and within-layer slot ordering across families.

use model_io::ModelFamily;

use crate::safetensors_header::SafetensorsHeader;

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
    /// A `qwen3_5` vision-tower tensor (`vision_tower.*`), ingested rather
    /// than dropped (ROADMAP M-V3). Distinct from [`Self::ExcludedMultimodal`]
    /// by FAMILY, not by prefix: the same `vision_tower.` string is a Gemma or
    /// `muse_glimmer` tower this port still has no kernels for.
    VisionTower,
    /// One plane of one shard of `qwen4_exp`'s hashed n-gram PLE table.
    ///
    /// **THE REASON THIS BUCKET EXISTS IS THAT THE FALLBACK IS 32 GB IN THE
    /// WRONG FILE.** These names sit under `language_model.`, so without an
    /// arm they take [`Self::LmResident`] and the whole table lands in
    /// `model_weights.bin` -- which is `crates/repack` Gotcha 15's
    /// unrecognized-marker failure ("loads and generates fine, just with the
    /// whole expert table pinned") at a size that cannot be pinned at all:
    /// 30.8% of the checkpoint, against a 3.8 GB resident core.
    ///
    /// 128 shards x three planes = 384 tensors, of 2,500,012 rows each.
    NgramShard {
        /// `weight`, `scales` or `biases`.
        role: &'static str,
        /// Which of `split_ngram_parts` shards.
        shard: usize,
    },
    /// One of the three int64 BUFFERS describing the n-gram table's hashing:
    /// `layer_multipliers`, `ngram_heads_offsets`, `ngram_heads_vocab_sizes`.
    ///
    /// **SEPARATE FROM [`Self::LmResident`] BECAUSE THE RESIDENT WALK NARROWS
    /// TO BF16** (Gotcha 9), and these are int64. A 20-million-entry prime
    /// vocabulary size rounded through BF16 is not a near miss, it is a
    /// different modulus, so every hash lands on the wrong row -- with no
    /// error, because the bytes are the right width for the tensor they were
    /// written as.
    ///
    /// They are the table's own statement of its hashing, and the reference
    /// recomputes them from `seed` only as a fallback. Carrying them means
    /// this port never has to derive a convention it could get wrong, which
    /// is the control-vector lesson (Gotcha 11) applied before it costs
    /// anything.
    NgramMeta {
        /// The buffer's leaf name.
        field: &'static str,
    },
    /// Tensor not matching known language model or multimodal patterns.
    Unknown,
}

/// The prefix the multi-token-prediction head's tensors carry.
pub const MTP_PREFIX: &str = "mtp.";

/// The prefix the vision tower's tensors carry in the SOURCE checkpoint.
///
/// Both `qwen3_5` publishers use it, and so do Gemma 4 and `muse_glimmer` --
/// which is exactly why the arm reading it is family-gated. It is NOT the
/// prefix the tower's resident tensors get in the INSTALL: that is
/// `VISION_INSTALL_PREFIX`, and the two differ on purpose (see there).
pub const VISION_PREFIX: &str = "vision_tower.";

/// The prefix the tower's resident tensors carry in the WRITTEN install.
///
/// Shorter than the source's, and the rename is load-bearing twice over.
/// `crates/runtime`'s FP16 exception is scoped to this exact string, so it has
/// to be one nothing else can collide with; and `lm_order_key` sorts resident
/// names on `layer_index`, which finds `.layers.` in anything spelled like a
/// block -- the tower is kept out of `resident_bases` for that reason
/// (`ClassifiedNames::vision_bases`), and a distinct namespace makes the
/// separation visible to every later reader of the index.
pub const VISION_INSTALL_PREFIX: &str = "vision.";

/// The HF-native spelling of [`VISION_PREFIX`], as an HF-native repository or
/// a standalone `vision.safetensors` export keeps the module path intact
/// (`model.visual.blocks.0.norm1.weight` rather than
/// `vision_tower.blocks.0.norm1.weight`). Every checkpoint this walk has
/// actually read is an MLX conversion under [`VISION_PREFIX`], so this second
/// spelling exists for the vision SIDECAR path (which reads an HF-native
/// repository directly, with no MLX conversion step in between) rather than
/// for anything the combined-install walk has needed so far.
pub const VISION_SOURCE_PREFIXES: [&str; 2] = [VISION_PREFIX, "model.visual."];

/// Renames every `model.visual.*` tensor in `header` onto [`VISION_PREFIX`]'s
/// naming, in place, so [`super::vision::read_vision_entries`] -- which is
/// keyed on [`VISION_PREFIX`] alone -- never has to know a second spelling
/// exists. A no-op on a header that carries only the canonical prefix
/// already, which is every checkpoint this walk has read so far.
///
/// Refuses, NAMING BOTH PREFIXES, a header that carries tensors under both
/// spellings at once. That header is not a checkpoint this walk has ever
/// seen: it reads either as two different towers accidentally concatenated,
/// or as a file already partway through some other tool's rename, and
/// silently preferring one prefix over the other would produce an install
/// that is missing half a tower or duplicates roles under two names, neither
/// of which fails until a dispatch four layers in.
pub fn canonicalize_vision_header(
    header: &mut SafetensorsHeader,
) -> Result<(), super::config::Gemma4Error> {
    const HF_PREFIX: &str = "model.visual.";
    let has_canonical = header.tensors.keys().any(|k| k.starts_with(VISION_PREFIX));
    let has_hf = header.tensors.keys().any(|k| k.starts_with(HF_PREFIX));
    if has_canonical && has_hf {
        return Err(super::config::Gemma4Error::Config(format!(
            "checkpoint carries vision tensors under both {VISION_PREFIX:?} and \
             {HF_PREFIX:?}; refusing an ambiguous or already-mixed source"
        )));
    }
    if !has_hf {
        return Ok(());
    }
    let mut renamed = std::collections::BTreeMap::new();
    for (name, info) in std::mem::take(&mut header.tensors) {
        match name.strip_prefix(HF_PREFIX) {
            Some(tail) => {
                renamed.insert(format!("{VISION_PREFIX}{tail}"), info);
            }
            None => {
                renamed.insert(name, info);
            }
        }
    }
    header.tensors = renamed;
    Ok(())
}

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
        // `qwen4_exp` spells its routed container identically, which is one
        // of the things that made most of its expert half free here: the
        // checkpoint's per-expert tensors are
        // `model.layers.N.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`.
        ModelFamily::QwenGdnMoe | ModelFamily::QwenGdnDense | ModelFamily::Qwen4Exp => {
            ".mlp.switch_mlp."
        }
        // A GGUF-derived Llama or Qwen3-MoE never reaches this classifier
        // (the GGUF walk maps routed tensors by NAME, in `gguf_names.rs`),
        // and neither has a safetensors path. DeepSeek V4 has no repack path
        // either; Gemma's marker is the default for all of them.
        // `muse_glimmer` is DENSE too, and unlike `qwen3_5` it has no MoE
        // sibling whose marker it could borrow, so it takes the default for
        // the same reason the GGUF-only families do: the marker can never
        // match, and one that could only ever match the wrong thing is worse
        // than one that cannot match. `spark2_5` is dense in the same way
        // and is GGUF-intake-only this pass, so it rides along.
        ModelFamily::Gemma4
        | ModelFamily::DeepseekV4Flash
        | ModelFamily::Llama
        | ModelFamily::Qwen3Moe
        | ModelFamily::GptOss
        | ModelFamily::MuseGlimmer
        // Dense `qwen3` is GGUF-intake-only for the same reason
        // (`docs/QWEN3_PHASE0.md`) and has no MoE sibling marker to borrow
        // either, so it rides along too.
        | ModelFamily::Spark25
        | ModelFamily::Qwen3Dense
        | ModelFamily::MiniMaxM2 => ".experts.switch_glu.",
    }
}

/// The container `qwen4_exp`'s sharded n-gram embedding lives under.
///
/// Every name below it is one of two things and NOTHING else, which is what
/// makes an exhaustive match here safe: 128 x 3 shard planes, and three int64
/// buffers. Read off the published index of
/// `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`, all 3,215 names.
pub const NGRAM_CONTAINER: &str = ".ple.ple_embedding.";

/// The shard sub-container inside [`NGRAM_CONTAINER`], singular with an
/// underscore before the index (`shard_37.weight`).
const NGRAM_SHARD_MARKER: &str = "ngram_embedding.shard_";
/// The SAME sub-container, spelled plural with a dot before the index
/// (`shards.37.weight`) -- `sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`'s
/// own convention, confirmed against a real streamed install: a name under
/// this spelling that missed both markers fell through to `LmResident` and
/// reached `pass_through_packed` with the trunk's group size (64) instead of
/// the table's own (32, `ngram.rs`'s `NGRAM_GROUP_SIZE`), refusing with a
/// shape-mismatch error that named a group-size arithmetic problem rather
/// than the real classification miss. Checked FIRST below, because it is the
/// one an actual install needs; both are kept because the singular form's
/// own doc cites a different publisher as its source and nothing here
/// confirms that publisher stopped using it.
const NGRAM_SHARDS_MARKER: &str = "ngram_embedding.shards.";

/// `qwen4_exp`'s n-gram table, split into its two shapes.
///
/// `None` for any other `ple.` tensor -- `key_proj`, `value_proj`, the three
/// norms and `conv1d` are ordinary per-layer weights and belong in the
/// resident index like every other small tensor.
///
/// **AN UNRECOGNIZED NAME UNDER THIS CONTAINER RETURNS `None` AND SO BECOMES
/// RESIDENT, WHICH IS THE WRONG DIRECTION**, so the two arms below are written
/// to cover the container exhaustively rather than to match what was expected.
/// `every_real_ngram_name_is_classified` walks the published index's own name
/// patterns and asserts nothing under it falls through.
fn classify_qwen4_ngram(name: &str) -> Option<Gemma4Bucket> {
    let tail = name.split_once(NGRAM_CONTAINER)?.1;

    let rest = tail
        .strip_prefix(NGRAM_SHARDS_MARKER)
        .or_else(|| tail.strip_prefix(NGRAM_SHARD_MARKER));
    if let Some(rest) = rest {
        // `<index>.<role>`, e.g. `37.scales`.
        let (index, role) = rest.split_once('.')?;
        let shard = index.parse().ok()?;
        let role = match role {
            "weight" => "weight",
            "scales" => "scales",
            "biases" => "biases",
            // A fourth plane would be a layout this walk has never seen.
            // `None` here files it resident, so it is refused by name instead.
            _ => return None,
        };
        return Some(Gemma4Bucket::NgramShard { role, shard });
    }

    // The three int64 buffers, matched EXACTLY rather than by prefix: they
    // describe the hashing, and a near-miss name silently becoming resident
    // would leave the table addressed by recomputed values while the
    // checkpoint's own sat unread (`Gemma4Bucket::NgramMeta`).
    let field = match tail {
        "layer_multipliers" => "layer_multipliers",
        "ngram_heads_offsets" => "ngram_heads_offsets",
        "ngram_heads_vocab_sizes" => "ngram_heads_vocab_sizes",
        _ => return None,
    };
    Some(Gemma4Bucket::NgramMeta { field })
}

/// Classifies source tensor name under specified model family contract.
pub fn classify_for_family(name: &str, num_layers: usize, family: ModelFamily) -> Gemma4Bucket {
    if name.starts_with("language_model.") {
        // THE N-GRAM TABLE, before the routed check and long before the
        // `LmResident` fallback, and family-gated so no other checkpoint can
        // reach it on a string collision.
        //
        // It is checked FIRST among the `language_model.` arms because it is
        // the only one whose fallback is unrecoverable: a routed expert
        // misfiled as resident makes a fat install, and this makes one that
        // cannot open. Order is cheap insurance rather than a requirement --
        // `ngram_embedding.` shares no substring with the routed marker.
        if family == ModelFamily::Qwen4Exp {
            if let Some(bucket) = classify_qwen4_ngram(name) {
                return bucket;
            }
        }
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
    // THE VISION TOWER (ROADMAP M-V3), and it must come BEFORE the drop list
    // below, which matches the same prefix for every family.
    //
    // `matches!` over two variants rather than `==` one, and that is AGENTS.md
    // Gotcha 61 applied rather than quoted: `qwen35` and `qwen35moe` share
    // this file, and three separate `== ModelFamily::QwenGdnMoe` conditions
    // elsewhere were each a latent bug for exactly as long as no dense
    // checkpoint existed to exercise them. Both halves of this architecture
    // ship the identical 333-tensor tower.
    //
    // Gemma 4 and `muse_glimmer` keep `ExcludedMultimodal` on the same string:
    // this port has no kernels for their towers, and ingesting one because the
    // prefix matched would write an install carrying weights nothing can
    // dispatch.
    if name.starts_with(VISION_PREFIX)
        && matches!(family, ModelFamily::QwenGdnDense | ModelFamily::QwenGdnMoe)
    {
        return Gemma4Bucket::VisionTower;
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
