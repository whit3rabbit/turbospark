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
        Self::from_packed_layer_in(layer, layout_dir, model_io::PACKED_EXPERTS_DIR)
    }

    /// [`Self::from_packed_experts_layer`] against an arbitrary subdirectory.
    ///
    /// The vision tower (ROADMAP M-V3) passes `model_io::PACKED_VISION_DIR`
    /// and gets a streamer over its 27 BLOCKS with nothing else changed --
    /// which is the whole reuse claim, and it holds because this struct
    /// interprets none of what it addresses. `expert` is an index into a
    /// fixed-stride file and `layer` is which file; neither word means
    /// anything about routing here.
    pub fn from_packed_layer_in(
        layer: &model_io::LayerLayout,
        layout_dir: &std::path::Path,
        subdir: &str,
    ) -> Self {
        let expert_stride = layer.expert_stride;
        // The decoder refuses anything but a bare name, but `LayerLayout`'s
        // fields are public and this join is what would leave the install.
        // Only the final component joins; a name with none (`..`) resolves to
        // the directory, whose open or size check then fails closed.
        let file = std::path::Path::new(&layer.file)
            .file_name()
            .unwrap_or_default();
        let path = layout_dir.join(subdir).join(file);
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
            // Saturating, not wrapping: `LayerLayout`'s fields are public, and
            // a wrapped window would be tiny and pass `open`'s size check on a
            // short file. `u64::MAX` fails that check instead.
            .map_or(0, |highest| highest.saturating_add(expert_stride));
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
        // Saturating so an out-of-range `layer` or `expert` cannot wrap back
        // onto a real expert: `u64::MAX` fails every caller's checked
        // `offset + stride` bound.
        let per_layer = (self.experts_per_layer as u64).saturating_mul(self.expert_stride);
        (layer as u64)
            .saturating_mul(per_layer)
            .saturating_add((expert as u64).saturating_mul(self.expert_stride))
    }
}
