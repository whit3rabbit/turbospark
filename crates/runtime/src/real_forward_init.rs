//! Helper initialization routines for opening `.gturbo` model installs,
//! validating architectural constraints, and instantiating expert streamers.

use std::path::Path;

use model_io::ArchConfig;

use crate::real_forward_types::RealForwardError;
use model_io::ExpertCacheSlots;

pub(crate) fn validate_arch_config(expecting: &ArchConfig) -> Result<(), RealForwardError> {
    for (name, value) in [
        ("num_kv_heads", expecting.num_kv_heads),
        ("num_full_kv_heads", expecting.num_full_kv_heads),
        ("head_dim", expecting.head_dim),
        ("full_head_dim", expecting.full_head_dim),
    ] {
        if value <= 0 {
            return Err(RealForwardError::Unsupported(format!(
                "{name} must be positive, got {value}"
            )));
        }
    }
    let max_kind = expecting
        .full_attention_layer_mask
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if max_kind > 2 {
        return Err(RealForwardError::Unsupported(
            "compressed (DeepSeek CSA/HCA) attention layers are not supported yet".to_string(),
        ));
    }
    // A mask-2 layer is gated DeltaNet, and the three families whose flow can
    // encode one are the two `families/qwen/` serves (`qwen36` and, ROADMAP's
    // 1-bit entry, the dense `qwen3_5`) plus `families/qwen4/`'s own GDN
    // branch (`qwen4_exp`, whose layer mask is GDN on every non-attention
    // layer by construction). Listed rather than defaulted for the reason
    // `encode_gemv_any`'s catch-all is: a family added later that declares
    // linear layers and has no GDN flow would otherwise reach `RealQwenState`
    // and bind a neighbour's tensors.
    if max_kind == 2
        && !matches!(
            expecting.family,
            model_io::ModelFamily::QwenGdnMoe
                | model_io::ModelFamily::QwenGdnDense
                | model_io::ModelFamily::Qwen4Exp
        )
    {
        return Err(RealForwardError::Unsupported(format!(
            "linear-attention layers need a qwen36, qwen35 or qwen4exp family, not {}",
            expecting.family.as_str()
        )));
    }
    if expecting.full_attention_layer_mask.contains(&0) && expecting.sliding_window <= 0 {
        return Err(RealForwardError::Unsupported(
            "sliding-window layers require a positive sliding_window".to_string(),
        ));
    }
    Ok(())
}

/// Why `--kv-bits` cannot open this install, or `None` when it can.
///
/// [`gpu::KvQuantTables::new`] and [`gpu::KvCacheManager::new_with_kv_quant`]
/// both PANIC on an unsupported head_dim rather than returning a `Result`
/// (see their own docs): they are lower-level contracts that assume the
/// caller already checked, and this is that check, translated into the
/// `RealForwardError::Unsupported` every other open-time refusal in this
/// file uses.
pub(crate) fn kv_quant_unsupported_reason(
    expecting: &ArchConfig,
    kv_quant: model_io::KvQuant,
) -> Option<String> {
    if !kv_quant.is_on() {
        return None;
    }
    if !model_io::rht_supported(expecting.full_head_dim) {
        return Some(format!(
            "--kv-bits needs a full head_dim that is a power of two in 32..=512; this install's \
             is {} (see docs/TRUBOQUANT.md)",
            expecting.full_head_dim
        ));
    }
    let num_layers = expecting.num_layers as usize;
    let has_eligible_layer = expecting
        .full_attention_layer_mask
        .iter()
        .enumerate()
        .any(|(layer, &mask)| model_io::layer_is_quantized(kv_quant, mask, layer, num_layers));
    if !has_eligible_layer {
        return Some(
            "--kv-bits has nothing to quantize on this install: every full-attention layer is \
             excluded by the last-layer rule, or there are no full-attention layers at all"
                .to_string(),
        );
    }
    None
}

pub(crate) type ExpertStreamersResult = (
    Vec<Option<streaming::PreadExpertStreamer>>,
    Vec<Vec<gpu::MetalBuffer>>,
    Option<model_io::PackedExpertsLayout>,
    // The RESOLVED slot count. Handed back rather than recomputed by the
    // caller because `Auto` reads the machine, so a second evaluation is not
    // guaranteed to agree with the one the buffers were allocated against.
    usize,
    // MAPPED residency, `None` per layer unless the seam is on. Carries the
    // Metal buffers FIRST so they drop before the mappings they alias, the
    // same declaration-order contract `slot_buffers` has against `streamers`.
    MappedResidency,
);

/// The routed experts read in place out of an `mmap` rather than `pread`-copied
/// into pinned slots (`streaming::MappedExpertLayer`).
///
/// Empty unless `TURBOSPARK_EXPERT_RESIDENCY=mapped`. When it is on, NO streamer
/// is opened and no slot is allocated -- which is the entire point, since the
/// slot cache is 70-90% of the measured peak of every MoE install. Both halves
/// stay `Vec`s indexed by layer so the decode path branches on
/// `buffers[layer].is_some()` and never on a mode flag it could disagree with.
#[derive(Default)]
pub(crate) struct MappedResidency {
    /// Declared BEFORE `layers` so the buffers drop first: each aliases its
    /// mapping with no deallocator, so the mapping must outlive it
    /// (`gpu::resident_metal`'s module docs).
    pub(crate) buffers: Vec<Option<gpu::MetalBuffer>>,
    pub(crate) layers: Vec<Option<streaming::MappedExpertLayer>>,
}

/// Reads the residency seam. UNSET is the `pread` streamer, so an install that
/// opened before this existed opens the same way and no frozen footprint row
/// moves without someone asking for it (AGENTS.md Gotcha 35's discipline: the
/// harnesses that measure this must not sense it).
///
/// Deliberately an env seam rather than a CLI flag for now, matching how
/// `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` landed: both arms must
/// produce identical tokens, so it is an A/B seam first and a feature second.
pub(crate) fn mapped_residency_requested() -> bool {
    std::env::var("TURBOSPARK_EXPERT_RESIDENCY")
        .map(|v| v.eq_ignore_ascii_case("mapped"))
        .unwrap_or(false)
}

/// Resolves the residency REQUEST (flag or default) to the mode this open
/// takes. THE ONE RESOLVER FOR EVERY CALLER: the open itself, the CLI's and
/// server's pre-open `committed_breakdown` sizing and the startup lines all
/// go through here, so the budget arithmetic and the allocation cannot
/// disagree about which mode was chosen -- which is ROADMAP P1 item 3's
/// "pick residency mode FIRST, then slot count only if streamed" as one
/// function rather than as an ordering convention.
///
/// `Auto` defers to the `TURBOSPARK_EXPERT_RESIDENCY=mapped` seam when it is
/// set -- that seam predates the flag and every mapped test and probe drives
/// it -- and otherwise to the memory-headroom rule, which TODAY is "always
/// stream" (see `ExpertResidency::Auto`'s doc for why that is deliberate
/// and what has to be measured before it flips).
pub fn resolve_expert_residency(
    requested: model_io::ExpertResidency,
) -> model_io::ResolvedExpertResidency {
    use model_io::{ExpertResidency, ResolvedExpertResidency};
    match requested {
        ExpertResidency::Mapped => ResolvedExpertResidency::Mapped,
        ExpertResidency::Streamed => ResolvedExpertResidency::Streamed,
        ExpertResidency::Auto => {
            if mapped_residency_requested() {
                ResolvedExpertResidency::Mapped
            } else {
                ResolvedExpertResidency::Streamed
            }
        }
    }
}

/// REFUSED BY NAME, NEVER IGNORED, AND NEVER LEFT TO FAIL DOWNSTREAM.
///
/// The mapped arm exists at exactly one dispatch site
/// (`families/gemma4/moe.rs`), and `open_expert_streamers` nulls EVERY
/// streamer when this mode engages -- so an unwired family would reach its own
/// `.ok_or_else` and report "layer N has no packed-expert streamer", blaming
/// the INSTALL for a mode the caller chose. It is also one code change away
/// from the silent-ignore failure `TURBOSPARK_ROUTED_BATCH` actually shipped
/// with on the MoE `llama` family, where a caller measured the per-token
/// engine and would have reported it under the batched label (Gotcha 22).
/// Refuse where the request is MEANINGFUL and unserved.
///
/// A pure function of the family rather than an inline check, so the rule is
/// testable without an env var, a GPU or an install -- and so WIDENING it is
/// one edit here plus the family's own dispatch arm, with the test that names
/// every unwired family reddening until both are done.
pub(crate) fn mapped_residency_refusal(
    family: model_io::ModelFamily,
) -> Result<(), RealForwardError> {
    if matches!(
        family,
        model_io::ModelFamily::Gemma4
            | model_io::ModelFamily::QwenGdnMoe
            | model_io::ModelFamily::Llama
            | model_io::ModelFamily::Qwen3Moe
            | model_io::ModelFamily::GptOss
    ) {
        return Ok(());
    }
    Err(RealForwardError::Unsupported(format!(
        "TURBOSPARK_EXPERT_RESIDENCY=mapped is not wired for {}; \
         unset it to use the pread expert streamer. Widening it is ROADMAP item 9, and \
         each family REPLACES this refusal rather than adding a branch to a silent path",
        family.as_str()
    )))
}

pub(crate) fn open_expert_streamers(
    dir: &Path,
    expecting: &ArchConfig,
    expert_cache_slots: ExpertCacheSlots,
    residency: model_io::ExpertResidency,
    resident_bytes: u64,
    max_bytes: u64,
    context: &mut gpu::MetalContext,
) -> Result<ExpertStreamersResult, RealForwardError> {
    let layout =
        model_io::load_packed_experts_layout(dir, max_bytes).map_err(RealForwardError::Model)?;
    let num_layers = expecting.num_layers as usize;
    let resolved_residency = resolve_expert_residency(residency);

    // THE SLOT CACHE IS SIZED `slots x layers x expert_stride`, AND THAT
    // PRODUCT IS A PROPERTY OF THE MODEL'S EXPERT GRANULARITY, NOT OF ITS
    // SIZE (ROADMAP Phase M2). A fine-grained MoE has many small experts --
    // Gemma 4 26B-A4B is 128 of ~3.2 MiB, so 16 slots over 30 layers pin
    // 1.5 GiB and the engine's ~2 GiB result follows. A COARSE one has few
    // large ones: Mixtral 8x7B is 8 experts of 108.9 MiB, so the same 16
    // slots over 32 layers want 54.5 GiB, and even `slots == num_experts`
    // pins the entire 27.2 GiB expert table, which is the opposite of
    // streaming.
    //
    // Two things follow, and both are cheap. A slot count ABOVE the expert
    // count can never help, so it is capped rather than allocated. And the
    // working set is reported in the error when the streamer cannot get its
    // memory, because "cannot allocate" without the number sends the reader
    // looking for a leak instead of at the arithmetic.
    // MAPPED RESIDENCY SHORT-CIRCUITS ALL OF THE ARITHMETIC BELOW, because
    // there is no slot cache to size: the kernels read each expert in place
    // out of its layer's mapping. Measured on the real Gemma 4 install, that
    // mapping costs 2.9 MiB of `phys_footprint` for the 30 buffer objects and
    // 0.1 MiB for the pages the GPU actually reads, against the 1.5-3.0 GiB
    // the slot cache pins for the same model (AGENTS.md Gotcha 19).
    //
    // The resolved slot count is still reported as the policy's answer rather
    // than 0, because it is what a caller printing a startup line has always
    // shown and a 0 there would read as "the cache is broken" rather than
    // "there is no cache".
    if resolved_residency == model_io::ResolvedExpertResidency::Mapped && layout.num_layers > 0 {
        mapped_residency_refusal(expecting.family)?;
        // The OTHER refusal this mode owes -- mapped residency against the
        // batched routed pair -- lives at that driver's own entry
        // (`families/gemma4/moe_batch.rs`) rather than here, because
        // `set_routed_batch_prefill` can flip that seam after open and an
        // env read here would miss the setter.
        let mut mapped = MappedResidency::default();
        for layer in 0..num_layers {
            let entry = layout
                .layers
                .iter()
                .find(|l| l.layer == layer)
                .ok_or_else(|| {
                    RealForwardError::Unsupported(format!(
                        "packed_experts layout missing layer {layer}"
                    ))
                })?;
            let stream_layout = streaming::StreamLayout::from_packed_experts_layer(entry, dir);
            let mapped_layer = streaming::MappedExpertLayer::open(stream_layout).map_err(|e| {
                RealForwardError::Unsupported(format!("mapped expert layer {layer}: {e}"))
            })?;
            let bytes = mapped_layer.page_aligned_bytes();
            let buffer =
                gpu::wrap_page_aligned_no_copy(context.device(), bytes.as_ptr(), bytes.len())
                    .map_err(RealForwardError::Gpu)?;
            mapped.buffers.push(Some(buffer));
            mapped.layers.push(Some(mapped_layer));
        }
        let resolved = expert_cache_slots
            .resolve(
                gpu::physical_memory(),
                resident_bytes,
                layout.layers.iter().map(|l| l.expert_stride).sum::<u64>(),
            )
            .min(layout.experts_per_layer.max(1));
        let mut streamers: Vec<Option<streaming::PreadExpertStreamer>> = Vec::new();
        streamers.resize_with(num_layers, || None);
        let slot_buffers = vec![Vec::new(); num_layers];
        return Ok((streamers, slot_buffers, Some(layout), resolved, mapped));
    }

    let experts_per_layer = layout.experts_per_layer.max(1);
    // ONE additional slot costs this much across the whole model, which is
    // the quantity the `Auto` policy divides its budget by. Summed over the
    // real per-layer strides rather than `layers * max(stride)`, because
    // ROADMAP Phase S's candidate has a layer 29 at 1.6x its siblings and
    // the model-wide maximum over-states the cost by 35% there.
    let bytes_per_slot = layout.layers.iter().map(|l| l.expert_stride).sum::<u64>();
    let expert_cache_slots = expert_cache_slots
        .resolve(gpu::physical_memory(), resident_bytes, bytes_per_slot)
        .min(experts_per_layer);
    let working_set = bytes_per_slot * expert_cache_slots as u64;
    let mut streamers: Vec<Option<streaming::PreadExpertStreamer>> = Vec::new();
    let experts_layout = if layout.num_layers > 0 {
        for layer in 0..num_layers {
            let entry = layout
                .layers
                .iter()
                .find(|l| l.layer == layer)
                .ok_or_else(|| {
                    RealForwardError::Unsupported(format!(
                        "packed_experts layout missing layer {layer}"
                    ))
                })?;
            let stream_layout = streaming::StreamLayout::from_packed_experts_layer(entry, dir);
            let streamer = streaming::PreadExpertStreamer::open(
                stream_layout,
                expert_cache_slots,
                streaming::ExpertCachePolicy::DEFAULT,
            )
            .map_err(|e| {
                RealForwardError::Unsupported(format!(
                    "expert streamer: {e} (this install's slot cache wants {:.1} GiB of pinned \
                     host memory: {expert_cache_slots} slots x {num_layers} layers x \
                     {:.1} MiB per expert. That product is set by expert GRANULARITY -- a \
                     coarse MoE like Mixtral 8x7B has 8 experts of ~109 MiB where Gemma 4 has \
                     128 of ~3.2 MiB -- so lower `--expert-cache-slots`, or use a \
                     fine-grained checkpoint)",
                    working_set as f64 / (1024.0 * 1024.0 * 1024.0),
                    entry.expert_stride as f64 / (1024.0 * 1024.0),
                ))
            })?;
            streamers.push(Some(streamer));
        }
        Some(layout)
    } else {
        streamers.resize_with(num_layers, || None);
        None
    };

    let mut slot_buffers: Vec<Vec<gpu::MetalBuffer>> = Vec::with_capacity(streamers.len());
    for streamer in &streamers {
        match streamer {
            Some(s) => {
                let mut wrapped = Vec::with_capacity(expert_cache_slots);
                for slot in 0..expert_cache_slots {
                    let (ptr, len) = s.slot_allocation(slot);
                    wrapped.push(
                        gpu::wrap_page_aligned_no_copy(context.device(), ptr, len)
                            .map_err(RealForwardError::Gpu)?,
                    );
                }
                slot_buffers.push(wrapped);
            }
            None => slot_buffers.push(Vec::new()),
        }
    }

    Ok((
        streamers,
        slot_buffers,
        experts_layout,
        expert_cache_slots,
        MappedResidency::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{mapped_residency_refusal, validate_arch_config};
    use model_io::ModelFamily;

    /// EVERY family is listed, not a sample, so adding a `ModelFamily`
    /// variant without deciding this question leaves it out of the sweep and
    /// the count assertion below reddens. The alternative -- a wildcard or a
    /// three-family sample -- is what lets a new family inherit an answer
    /// nobody gave it (AGENTS.md Gotchas 24, 37 and 39, and `crates/bench`
    /// Gotcha 16's no-wildcard-arm rule).
    const EVERY_FAMILY: &[ModelFamily] = &[
        ModelFamily::Gemma4,
        ModelFamily::QwenGdnMoe,
        ModelFamily::DeepseekV4Flash,
        ModelFamily::Llama,
        ModelFamily::Qwen3Moe,
        ModelFamily::GptOss,
        ModelFamily::QwenGdnDense,
        ModelFamily::MuseGlimmer,
        ModelFamily::Qwen4Exp,
        ModelFamily::Spark25,
        ModelFamily::Qwen3Dense,
        ModelFamily::MiniMaxM2,
        ModelFamily::Qwen2Dense,
    ];

    fn assert_invalid_kv_dimension(name: &str, value: i64, mutate: fn(&mut model_io::ArchConfig)) {
        let mut arch = model_io::gemma4_26b_a4b();
        mutate(&mut arch);
        let error = validate_arch_config(&arch).expect_err(name);
        assert_eq!(
            error.to_string(),
            format!("unsupported: {name} must be positive, got {value}")
        );
    }

    #[test]
    fn negative_swa_kv_heads_are_refused_before_gpu_allocation() {
        assert_invalid_kv_dimension("num_kv_heads", -4_294_967_294, |arch| {
            arch.num_kv_heads = -4_294_967_294
        });
    }

    #[test]
    fn zero_full_kv_heads_are_refused_before_gpu_allocation() {
        assert_invalid_kv_dimension("num_full_kv_heads", 0, |arch| arch.num_full_kv_heads = 0);
    }

    #[test]
    fn zero_swa_head_dim_is_refused_before_gpu_allocation() {
        assert_invalid_kv_dimension("head_dim", 0, |arch| arch.head_dim = 0);
    }

    #[test]
    fn negative_full_head_dim_is_refused_before_gpu_allocation() {
        assert_invalid_kv_dimension("full_head_dim", -1, |arch| arch.full_head_dim = -1);
    }

    #[test]
    fn mapped_residency_is_served_on_gemma4_and_refused_by_name_everywhere_else() {
        assert_eq!(
            EVERY_FAMILY.len(),
            ModelFamily::ALL.len(),
            "a ModelFamily variant was added without deciding whether it serves \
             mapped expert residency; add it to EVERY_FAMILY and to the dispatch \
             site, or leave it refused deliberately"
        );
        for &family in EVERY_FAMILY {
            match mapped_residency_refusal(family) {
                Ok(()) => assert!(
                    matches!(
                        family,
                        ModelFamily::Gemma4
                            | ModelFamily::QwenGdnMoe
                            | ModelFamily::Llama
                            | ModelFamily::Qwen3Moe
                            | ModelFamily::GptOss
                    ),
                    "{} accepts mapped residency but has no mapped arm at its \
                     dispatch site; a family that accepts it and does not serve it \
                     runs the streamed engine under the mapped label",
                    family.as_str()
                ),
                Err(err) => {
                    assert_ne!(family, ModelFamily::Gemma4);
                    let text = err.to_string();
                    // The seam the caller actually SET has to appear, or the
                    // message cannot be connected to the thing that caused it
                    // -- which is the whole difference between this and the
                    // downstream "layer N has no packed-expert streamer".
                    assert!(
                        text.contains("TURBOSPARK_EXPERT_RESIDENCY"),
                        "the refusal must name the seam; got {text}"
                    );
                    assert!(
                        text.contains(family.as_str()),
                        "the refusal must name the family it declined; got {text}"
                    );
                }
            }
        }
    }
}
