//! The routed experts read IN PLACE out of an `mmap`, instead of `pread`-copied
//! into a pinned slot. The sibling of [`crate::PreadExpertStreamer`], not its
//! replacement: see the trade at the bottom of this comment.
//!
//! # Why this can exist at all
//!
//! The MoE decode kernels never required a slot. `moe_decode.rs`'s own header
//! says they read expert weights in place from "streamer slots **or any other
//! page of memory**" through the `RoutedBlobs` argument buffer, and
//! `RoutedBlobsBuffer::bind` has always taken `(buffer, offset)` pairs. The
//! slot cache exists because the STREAMER copies; nothing downstream of it
//! asked for a copy. So a routed blob pointer can be
//! `mapped_layer_buffer + expert_offset` and the kernels do not change.
//!
//! # What it costs, measured rather than assumed
//!
//! The obvious objection is that a `newBufferWithBytesNoCopy` over a
//! multi-gigabyte mapping would pin it, which AGENTS.md Gotcha 19 asserted for
//! months. It does not. Measured 2026-08-23 on the real Gemma 4 install's
//! 12.3 GB expert table (`crates/bench/tests/mapped_expert_probe.rs`): the
//! `mmap` charges 0.0 MiB of `phys_footprint`, wrapping all 30 layer files
//! charges 2.9 MiB (the buffer objects), and reading one expert on each of the
//! 30 layers through the GPU -- about 96 MiB of fresh file-backed pages --
//! charges a further 0.1 MiB. Clean file-backed pages are excluded whoever
//! reads them.
//!
//! # The trade, which is why the streamer stays
//!
//! Pages nobody is charged for are pages the OS may EVICT. The slot cache's
//! virtue is that it PINS a bounded working set; a mapping hands residency to
//! the kernel, which is the right answer on a machine that can hold the table
//! and the wrong one on a machine that cannot. `crates/streaming`'s Gotcha 3
//! is explicit that on a cold or memory-tight machine this path is genuinely
//! disk-bound. So this is a MODE, resolved against the machine, and it
//! resolves DOWN to the streamer rather than up.
//!
//! # Layering
//!
//! One instance per LAYER, exactly as one `PreadExpertStreamer` is opened per
//! layer file, and it reuses that streamer's [`StreamLayout`] rather than
//! restating the offset arithmetic -- including the per-expert offset table,
//! which exists because the writer need not emit dense `expert * stride`
//! offsets. Nothing here is Metal: this crate does not depend on `gpu`, so it
//! hands out page-aligned bytes and the caller wraps them.

use model_io::ResidentBuffer;

use crate::error::StreamerError;
use crate::stream_layout::StreamLayout;

/// One layer's expert file, mapped and addressable by expert index.
pub struct MappedExpertLayer {
    layout: StreamLayout,
    mapping: ResidentBuffer,
    /// Byte offset of logical position 0 inside [`Self::page_aligned_bytes`].
    ///
    /// `ResidentBuffer` rounds the requested file offset DOWN to a page
    /// boundary so the mapping base is page-aligned, which is what
    /// `newBufferWithBytesNoCopy` requires, and reports the difference. Every
    /// expert offset this type hands out is relative to that aligned base,
    /// because that is the pointer the caller wrapped. Adding the shift at the
    /// point of use rather than trusting it to be zero is the whole reason a
    /// non-zero `stream_offset` cannot silently read the wrong expert.
    shift: u64,
}

impl MappedExpertLayer {
    /// Maps `layout`'s file window read-only.
    ///
    /// The size check mirrors `PreadExpertStreamer::open`'s and for its
    /// reason: a short file is the case where a bad layout is most likely,
    /// and the alternative to failing here is an out-of-bounds expert offset
    /// discovered a long way downstream.
    pub fn open(layout: StreamLayout) -> Result<Self, StreamerError> {
        let path = std::path::PathBuf::from(&layout.path);
        let meta = std::fs::metadata(&path).map_err(|e| StreamerError::OpenFailed {
            path: layout.path.clone(),
            detail: e.to_string(),
        })?;
        let required = layout.stream_offset + layout.stream_size;
        if meta.len() < required {
            return Err(StreamerError::SizeMismatch {
                expected: required,
                actual: meta.len(),
            });
        }
        let mapping = ResidentBuffer::map(&path, layout.stream_offset, layout.stream_size)
            .map_err(|e| StreamerError::OpenFailed {
                path: layout.path.clone(),
                detail: format!("{e:?}"),
            })?;
        let shift = mapping.slice_shift() as u64;
        Ok(Self {
            layout,
            mapping,
            shift,
        })
    }

    /// The page-aligned mapping the caller wraps zero-copy. Stays valid for
    /// this value's whole lifetime; the wrap aliases it and creates no
    /// deallocator, so the wrapper must not outlive this.
    pub fn page_aligned_bytes(&self) -> &[u8] {
        self.mapping.mapped_bytes()
    }

    /// Byte offset of `expert`'s blob inside [`Self::page_aligned_bytes`].
    ///
    /// Layer 0 semantics, matching how a per-layer streamer is opened: the
    /// layout's explicit per-expert table is consulted first and the uniform
    /// `expert * stride` formula is the fallback.
    pub fn expert_offset(&self, expert: usize) -> u64 {
        self.shift + self.layout.expert_offset(0, expert)
    }

    /// Bytes per expert blob in this layer. Per LAYER and never the
    /// model-wide maximum, for the reason `LayerLayout::expert_stride`
    /// records: a mixed sub-4-bit install is not uniform across layers.
    pub fn expert_stride(&self) -> u64 {
        self.layout.expert_stride
    }

    /// How many experts this layer holds.
    pub fn experts_per_layer(&self) -> usize {
        self.layout.experts_per_layer
    }

    /// The layout this was built from.
    pub fn layout(&self) -> &StreamLayout {
        &self.layout
    }

    /// The expert's bytes, for a host-side reader. The GPU path does not use
    /// this -- it addresses the mapping by offset -- but a test proving the
    /// mapped bytes equal what the streamer would have copied does.
    pub fn expert_bytes(&self, expert: usize) -> Option<&[u8]> {
        let start = self.expert_offset(expert) as usize;
        let end = start.checked_add(self.expert_stride() as usize)?;
        self.mapping.mapped_bytes().get(start..end)
    }
}
