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
    // The gpt-oss attention flow uses the full-attention dimensions for every
    // layer. Its sliding and full pairs are equal in canonical installs, and
    // the KV cache relies on that invariant when sizing a sliding layer.
    // Refuse a hand-built manifest before any cache buffers are allocated.
    if expecting.family == model_io::ModelFamily::GptOss
        && (expecting.num_kv_heads != expecting.num_full_kv_heads
            || expecting.head_dim != expecting.full_head_dim)
    {
        return Err(RealForwardError::Unsupported(format!(
            "gpt-oss requires identical sliding and full KV dimensions; got sliding {}x{} and \
             full {}x{}",
            expecting.num_kv_heads,
            expecting.head_dim,
            expecting.num_full_kv_heads,
            expecting.full_head_dim
        )));
    }
    let mask = &expecting.full_attention_layer_mask;
    let max_kind = mask.iter().copied().max().unwrap_or(0);
    // Each kind is checked by membership rather than by the maximum, because
    // mask 6 (`lfm3`'s short convolution) sits above the MLA code and a
    // max-based test would read it as one of the compressed kinds.
    //
    // Mask 5 is multi-head latent attention (`deepseek2`), whose compressed
    // cache and absorbed kernels live in `families/deepseek2/`; masks 3 and
    // 4 remain the DeepSeek V4 compressed-attention kinds nothing runs yet.
    // Gated BY FAMILY rather than by kind alone so a future family declaring
    // mask 5 without an MLA flow is refused here instead of reaching for a
    // neighbour's kernel. Mask 6 is gated the same way.
    if mask.contains(&5) && expecting.family != model_io::ModelFamily::Deepseek2 {
        return Err(RealForwardError::Unsupported(format!(
            "MLA (mask 5) layers need the deepseek2 family, not {}",
            expecting.family.as_str()
        )));
    }
    if mask.contains(&6) && expecting.family != model_io::ModelFamily::Lfm3 {
        return Err(RealForwardError::Unsupported(format!(
            "short-conv (mask 6) layers need the lfm3 family, not {}",
            expecting.family.as_str()
        )));
    }
    if mask.iter().any(|&k| k == 3 || k == 4) {
        return Err(RealForwardError::Unsupported(
            "compressed (DeepSeek CSA/HCA) attention layers are not supported yet".to_string(),
        ));
    }
    if max_kind > 6 {
        return Err(RealForwardError::Unsupported(format!(
            "layer mask kind {max_kind} is not a kind this port knows"
        )));
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
    // The RESOLVED residency mode, carried with the buffers for the same
    // reason as the slot count. The runner must report what actually opened.
    model_io::ResolvedExpertResidency,
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
/// it -- and otherwise to the memory-headroom rule. The caller supplies the
/// already-read layout cost so this decision is identical before and during
/// open.
pub fn resolve_expert_residency(
    requested: model_io::ExpertResidency,
    physical_bytes: u64,
    resident_bytes: u64,
    bytes_per_slot: u64,
    mapped_supported: bool,
) -> model_io::ResolvedExpertResidency {
    use model_io::{ExpertResidency, ResolvedExpertResidency};
    match requested {
        ExpertResidency::Mapped => ResolvedExpertResidency::Mapped,
        ExpertResidency::Streamed => ResolvedExpertResidency::Streamed,
        ExpertResidency::Auto => {
            let should_map = mapped_residency_requested()
                || (mapped_supported
                    && model_io::auto_residency_prefers_mapped(
                        physical_bytes,
                        resident_bytes,
                        bytes_per_slot,
                    ));
            if should_map {
                ResolvedExpertResidency::Mapped
            } else {
                ResolvedExpertResidency::Streamed
            }
        }
    }
}

/// Resolve the same policy before Metal allocation, when only an install path
/// and its architecture are available. The runtime open repeats the layout
/// read and carries its resolved answer out of `open_expert_streamers`; this
/// helper makes the CLI, server, FFI and app admission checks use the same
/// inputs rather than assuming every `Auto` open is streamed.
#[cfg(target_os = "macos")]
pub fn resolve_expert_residency_for_install(
    dir: &Path,
    family: model_io::ModelFamily,
    requested: model_io::ExpertResidency,
    physical_bytes: u64,
) -> Result<model_io::ResolvedExpertResidency, String> {
    let resident_bytes = std::fs::metadata(dir.join("model_weights.bin"))
        .map(|m| m.len())
        .unwrap_or(0);
    let (bytes_per_slot, table_bytes) = model_io::load_packed_experts_layout(
        dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .map(|layout| expert_table_bytes(&layout))
    .unwrap_or((0, 0));
    let resolved = resolve_expert_residency(
        requested,
        physical_bytes,
        resident_bytes,
        bytes_per_slot,
        mapped_residency_supported(family),
    );
    guard_mapped_residency(
        mapped_explicitly(requested),
        resolved,
        mapped_working_set(resolved),
        resident_bytes,
        table_bytes,
    )
}

/// One slot's bytes across every layer, and the whole expert table's.
fn expert_table_bytes(layout: &model_io::PackedExpertsLayout) -> (u64, u64) {
    let per_slot = layout.layers.iter().map(|l| l.expert_stride).sum::<u64>();
    (
        per_slot,
        per_slot.saturating_mul(layout.experts_per_layer as u64),
    )
}

/// Whether mapped residency was asked for by name (the flag or the
/// `TURBOSPARK_EXPERT_RESIDENCY=mapped` seam) rather than picked by `Auto`.
fn mapped_explicitly(requested: model_io::ExpertResidency) -> bool {
    match requested {
        model_io::ExpertResidency::Mapped => true,
        model_io::ExpertResidency::Auto => mapped_residency_requested(),
        model_io::ExpertResidency::Streamed => false,
    }
}

/// Metal's recommended working set, asked only when mapped residency is on
/// the table. Zero when there is no device, which the guard reads as unknown.
#[cfg(target_os = "macos")]
fn mapped_working_set(resolved: model_io::ResolvedExpertResidency) -> u64 {
    if resolved != model_io::ResolvedExpertResidency::Mapped {
        return 0;
    }
    gpu::recommended_max_working_set().map_or(0, |(bytes, _)| bytes)
}

/// Mapped residency only where its mappings fit what the GPU keeps wired
/// (`model_io::mapped_residency_fits`; `docs/EXPERT_RESIDENCY.md` has the
/// measurement). Past that, mapped asked for by name is refused with the
/// reason, and `Auto` resolves down to the streamer, as it does everywhere.
pub(crate) fn guard_mapped_residency(
    explicit: bool,
    resolved: model_io::ResolvedExpertResidency,
    working_set_bytes: u64,
    resident_bytes: u64,
    table_bytes: u64,
) -> Result<model_io::ResolvedExpertResidency, String> {
    use model_io::ResolvedExpertResidency;
    if resolved != ResolvedExpertResidency::Mapped
        || model_io::mapped_residency_fits(working_set_bytes, resident_bytes, table_bytes)
    {
        return Ok(resolved);
    }
    if !explicit {
        return Ok(ResolvedExpertResidency::Streamed);
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    Err(format!(
        "mapped expert residency refused: the GPU wires the whole {:.2} GiB expert table and \
         the {:.2} GiB resident weights, and with {:.0} GiB kept for the rest of the process \
         and the machine that exceeds Metal's {:.2} GiB recommended working set. Past it the \
         driver unwires layers and every routed pass re-reads a whole layer (measured 70 to \
         180 times slower, docs/EXPERT_RESIDENCY.md). Use streamed residency",
        gib(table_bytes),
        gib(resident_bytes),
        gib(model_io::MAPPED_WIRED_RESERVE_BYTES),
        gib(working_set_bytes)
    ))
}

fn mapped_residency_supported(family: model_io::ModelFamily) -> bool {
    matches!(
        family,
        model_io::ModelFamily::Gemma4
            | model_io::ModelFamily::QwenGdnMoe
            | model_io::ModelFamily::Llama
            | model_io::ModelFamily::Qwen3Moe
            | model_io::ModelFamily::GptOss
    )
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
    if layout.num_layers > 0 && layout.experts_per_layer != expecting.num_experts as usize {
        return Err(RealForwardError::Unsupported(format!(
            "packed_experts layout declares {} experts per layer, but the architecture declares {}",
            layout.experts_per_layer, expecting.num_experts
        )));
    }
    let (bytes_per_slot, table_bytes) = expert_table_bytes(&layout);
    let resolved_residency = resolve_expert_residency(
        residency,
        gpu::physical_memory(),
        resident_bytes,
        bytes_per_slot,
        mapped_residency_supported(expecting.family),
    );
    // The admission resolver above runs this same guard on the same inputs,
    // so admission and the open cannot disagree about the mode.
    let resolved_residency = guard_mapped_residency(
        mapped_explicitly(residency),
        resolved_residency,
        mapped_working_set(resolved_residency),
        resident_bytes,
        table_bytes,
    )
    .map_err(RealForwardError::Unsupported)?;

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
        // Router indices span the architecture's expert count. Every mapped
        // index must therefore name a complete blob in this layout before it
        // can be forwarded as a Metal buffer offset.
        if !crate::expert_layout_validation::mapped_expert_count_matches(
            expecting.num_experts,
            layout.experts_per_layer,
        ) {
            return Err(RealForwardError::Unsupported(format!(
                "mapped expert residency needs packed_experts/layout.json expertsPerLayer \
                 ({}) to match the architecture's num_experts ({})",
                layout.experts_per_layer, expecting.num_experts
            )));
        }
        // The OTHER refusal this mode owes -- mapped residency against the
        // batched routed pair -- lives at that driver's own entry
        // (`families/gemma4/moe_batch.rs`) rather than here, because
        // `set_routed_batch_prefill` can flip that seam after open and an
        // env read here would miss the setter.
        let mut mapped = MappedResidency::default();
        // Blob positions are sequential from the first ROUTED layer, so
        // checkpoint layer L reads layout position `L - lead` (`deepseek2`'s
        // dense lead; zero on every family that routes all its layers). The
        // dense lead layers get None holes, which is safe because a dense
        // FFN layer never consults the routed path.
        let lead = expecting.num_dense_leading_layers.max(0) as usize;
        mapped.buffers.resize_with(num_layers, || None);
        mapped.layers.resize_with(num_layers, || None);
        for layer in lead..num_layers {
            let entry = layout
                .layers
                .iter()
                .find(|l| l.layer == layer - lead)
                .ok_or_else(|| {
                    RealForwardError::Unsupported(format!(
                        "packed_experts layout missing blob position {} (checkpoint layer {layer})",
                        layer - lead
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
            mapped.buffers[layer] = Some(buffer);
            mapped.layers[layer] = Some(mapped_layer);
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
        return Ok((
            streamers,
            slot_buffers,
            Some(layout),
            resolved,
            resolved_residency,
            mapped,
        ));
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
    // Read once per open, like the other streamer seams: every layer's
    // cache must run the same policy or a run measures a mixture.
    let cache_policy =
        streaming::ExpertCachePolicy::from_env().map_err(RealForwardError::Unsupported)?;
    let experts_layout = if layout.num_layers > 0 {
        // Dense lead layers carry no expert blob: they get None holes, and
        // the flow (which indexes by CHECKPOINT layer) never consults them.
        let lead = expecting.num_dense_leading_layers.max(0) as usize;
        for layer in 0..num_layers {
            if layer < lead {
                streamers.push(None);
                continue;
            }
            let entry = layout
                .layers
                .iter()
                .find(|l| l.layer == layer - lead)
                .ok_or_else(|| {
                    RealForwardError::Unsupported(format!(
                        "packed_experts layout missing blob position {} (checkpoint layer \
                         {layer})",
                        layer - lead
                    ))
                })?;
            let stream_layout = streaming::StreamLayout::from_packed_experts_layer(entry, dir);
            let streamer = streaming::PreadExpertStreamer::open(
                stream_layout,
                expert_cache_slots,
                cache_policy,
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
        resolved_residency,
        MappedResidency::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        guard_mapped_residency, mapped_residency_refusal, resolve_expert_residency,
        validate_arch_config,
    };
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
        ModelFamily::Qwen3Vl,
        // Refused deliberately: no mapped arm in the flint dispatch's
        // mapped residency yet (it is wired, untested, in `moe.rs`).
        ModelFamily::Flint,
        // Refused deliberately: mapped residency over the compressed MLA
        // cache has no mapped arm in the deepseek2 dispatch, and the family
        // opens fine on the streamed path.
        ModelFamily::Deepseek2,
        // Refused deliberately: DENSE, no routed experts to map.
        ModelFamily::Lfm3,
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

    #[test]
    fn auto_residency_uses_memory_headroom_only_for_wired_families() {
        let bytes_per_slot = 30 * 3_355_443;
        assert_eq!(
            resolve_expert_residency(
                model_io::ExpertResidency::Auto,
                16 * 1024 * 1024 * 1024,
                13 * 1024 * 1024 * 1024,
                bytes_per_slot,
                true,
            ),
            model_io::ResolvedExpertResidency::Mapped
        );
        assert_eq!(
            resolve_expert_residency(
                model_io::ExpertResidency::Auto,
                16 * 1024 * 1024 * 1024,
                13 * 1024 * 1024 * 1024,
                bytes_per_slot,
                false,
            ),
            model_io::ResolvedExpertResidency::Streamed
        );
        assert_eq!(
            resolve_expert_residency(
                model_io::ExpertResidency::Streamed,
                16 * 1024 * 1024 * 1024,
                13 * 1024 * 1024 * 1024,
                bytes_per_slot,
                true,
            ),
            model_io::ResolvedExpertResidency::Streamed
        );
    }

    /// Fish Audio Drama 3's trunk (resident, expert table) against the
    /// 36 GiB M4 Max's 28.08 GiB working set: decoded 180 times slower mapped.
    const DRAMA3: (u64, u64) = (2_558_993_664, 18_124_701_696);
    const WORKING_SET: u64 = 30_150_000_000;

    #[test]
    fn mapped_residency_past_the_wired_budget_is_refused_by_name_or_resolved_down() {
        use model_io::ResolvedExpertResidency::{Mapped, Streamed};
        let (resident, table) = DRAMA3;
        let err = guard_mapped_residency(true, Mapped, WORKING_SET, resident, table)
            .expect_err("asked for by name, a table past the budget is refused");
        assert!(
            err.contains("16.88 GiB expert table")
                && err.contains("EXPERT_RESIDENCY.md")
                && err.contains("streamed"),
            "{err}"
        );
        assert_eq!(
            guard_mapped_residency(false, Mapped, WORKING_SET, resident, table),
            Ok(Streamed),
            "Auto resolves down to the streamer"
        );
        assert_eq!(
            guard_mapped_residency(true, Streamed, WORKING_SET, resident, table),
            Ok(Streamed),
            "a streamed open is never touched"
        );
        // A table that fits, and the same table with no working set known.
        assert_eq!(
            guard_mapped_residency(true, Mapped, WORKING_SET, resident, table / 2),
            Ok(Mapped)
        );
        assert!(guard_mapped_residency(true, Mapped, 0, resident, table / 2).is_err());
    }
}
