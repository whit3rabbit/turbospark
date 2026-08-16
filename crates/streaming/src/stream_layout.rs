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
    /// decoded `packed_experts/layout.json` (`turbospark-model-io`'s
    /// `PackedExpertsLayout`). Per-expert offsets are carried explicitly
    /// (rather than relying on the uniform `expert * expert_stride`
    /// formula) since the writer may pack experts out of stride order.
    ///
    /// The stride comes from the LAYER, not from the model-wide value, and
    /// used to be a parameter the caller filled in from the latter. A mixed
    /// sub-4-bit install (ROADMAP Phase S) is not uniform across layers, and
    /// since one streamer serves one layer there is nothing to be gained by
    /// sizing it for the widest layer in the model: that only over-reads and
    /// over-allocates. Taking it off the layer removes the chance of the
    /// caller passing the wrong one.
    pub fn from_packed_experts_layer(
        layer: &model_io::LayerLayout,
        layout_dir: &std::path::Path,
    ) -> Self {
        let expert_stride = layer.expert_stride;
        let path = layout_dir.join("packed_experts").join(&layer.file);
        let expert_offsets: Vec<u64> = layer.experts.iter().map(|e| e.offset).collect();
        // Spans the HIGHEST offset rather than `count * stride`. The two
        // agree only while the writer emits dense `e * stride` offsets, and
        // the paragraph above is explicit that it need not -- which is the
        // whole reason the offsets are carried. A header, padding, or any
        // permutation makes the count-based window too small, and what
        // fails then is `PreadExpertStreamer`'s per-load bounds check
        // rejecting the last expert: an `OffsetOutOfRange` a long way from
        // the layout that produced it.
        let stream_size = expert_offsets
            .iter()
            .copied()
            .max()
            .map_or(0, |highest| highest + expert_stride);
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
