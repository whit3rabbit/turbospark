//! Helper initialization routines for opening `.gturbo` model installs,
//! validating architectural constraints, and instantiating expert streamers.

use std::path::Path;

use model_io::ArchConfig;

use crate::expert_cache_policy::ExpertCacheSlots;
use crate::real_forward_types::RealForwardError;

pub(crate) fn validate_arch_config(expecting: &ArchConfig) -> Result<(), RealForwardError> {
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
    // A mask-2 layer is gated DeltaNet, and the two families whose flow can
    // encode one are the two `families/qwen/` serves: `qwen36` and (ROADMAP's
    // 1-bit entry) the dense `qwen3_5`. Listed rather than defaulted for the
    // reason `encode_gemv_any`'s catch-all is: a family added later that
    // declares linear layers and has no GDN flow would otherwise reach
    // `RealQwenState` and bind a neighbour's tensors.
    if max_kind == 2
        && !matches!(
            expecting.family,
            model_io::ModelFamily::QwenGdnMoe | model_io::ModelFamily::QwenGdnDense
        )
    {
        return Err(RealForwardError::Unsupported(format!(
            "linear-attention layers need a qwen36 or qwen35 family, not {}",
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

pub(crate) type ExpertStreamersResult = (
    Vec<Option<streaming::PreadExpertStreamer>>,
    Vec<Vec<gpu::MetalBuffer>>,
    Option<model_io::PackedExpertsLayout>,
    // The RESOLVED slot count. Handed back rather than recomputed by the
    // caller because `Auto` reads the machine, so a second evaluation is not
    // guaranteed to agree with the one the buffers were allocated against.
    usize,
);

pub(crate) fn open_expert_streamers(
    dir: &Path,
    expecting: &ArchConfig,
    expert_cache_slots: ExpertCacheSlots,
    resident_bytes: u64,
    max_bytes: u64,
    context: &mut gpu::MetalContext,
) -> Result<ExpertStreamersResult, RealForwardError> {
    let layout =
        model_io::load_packed_experts_layout(dir, max_bytes).map_err(RealForwardError::Model)?;
    let num_layers = expecting.num_layers as usize;

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

    Ok((streamers, slot_buffers, experts_layout, expert_cache_slots))
}
