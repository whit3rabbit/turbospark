//! Byte layout of one routed-expert layer file. Ported from
//! `Infrastructure/Streaming/ExpertStreamer.swift` (`StreamLayout`).

#[derive(Debug, Clone, PartialEq)]
pub struct StreamLayout {
    pub path: String,
    pub stream_offset: u64,
    pub stream_size: u64,
    pub experts_per_layer: usize,
    pub expert_stride: u64,
    /// Explicit per-expert offsets for layer 0 (a packed, non-uniform
    /// layout); `None` falls back to the uniform `layer * per_layer +
    /// expert * expert_stride` formula for every layer.
    pub expert_offsets: Option<Vec<u64>>,
}

impl StreamLayout {
    /// Builds the streaming layout for one packed-expert layer file from the
    /// decoded `packed_experts/layout.json` (`mrefrust-model-io`'s
    /// `PackedExpertsLayout`). Per-expert offsets are carried explicitly
    /// (rather than relying on the uniform `expert * expert_stride`
    /// formula) since the writer may pack experts out of stride order.
    pub fn from_packed_experts_layer(
        layer: &model_io::LayerLayout,
        layout_dir: &std::path::Path,
        expert_stride: u64,
    ) -> Self {
        let path = layout_dir.join("packed_experts").join(&layer.file);
        let expert_offsets: Vec<u64> = layer.experts.iter().map(|e| e.offset).collect();
        let stream_size = expert_offsets.len() as u64 * expert_stride;
        Self {
            path: path.display().to_string(),
            stream_offset: 0,
            stream_size,
            experts_per_layer: layer.experts.len(),
            expert_stride,
            expert_offsets: Some(expert_offsets),
        }
    }

    pub fn expert_offset(&self, layer: usize, expert: usize) -> u64 {
        if layer == 0 {
            if let Some(offsets) = &self.expert_offsets {
                if let Some(&offset) = offsets.get(expert) {
                    return offset;
                }
            }
        }
        let per_layer = self.experts_per_layer as u64 * self.expert_stride;
        layer as u64 * per_layer + expert as u64 * self.expert_stride
    }
}
