//! Helper initialization routines for opening `.gturbo` model installs,
//! validating architectural constraints, and instantiating expert streamers.

use std::path::Path;

use model_io::ArchConfig;

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
    if max_kind == 2 && expecting.family != model_io::ModelFamily::Qwen36 {
        return Err(RealForwardError::Unsupported(format!(
            "linear-attention layers need the qwen36 family, not {}",
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
);

pub(crate) fn open_expert_streamers(
    dir: &Path,
    expecting: &ArchConfig,
    expert_cache_slots: usize,
    max_bytes: u64,
    context: &mut gpu::MetalContext,
) -> Result<ExpertStreamersResult, RealForwardError> {
    let layout =
        model_io::load_packed_experts_layout(dir, max_bytes).map_err(RealForwardError::Model)?;
    let num_layers = expecting.num_layers as usize;
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
            .map_err(|e| RealForwardError::Unsupported(format!("expert streamer: {e}")))?;
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

    Ok((streamers, slot_buffers, experts_layout))
}
