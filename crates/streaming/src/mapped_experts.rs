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

/// Preparation changes page access only, never expert weights or slot order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappedPagePreparationMode {
    /// Ask the OS to fetch selected ranges; failure is advisory only.
    Advice,
    /// Read one byte per selected VM page before the GPU consumes it.
    Touch,
}

/// Work performed by one demand-selected preparation call.
///
/// Byte counts describe mapped spans, not physical disk I/O or guaranteed
/// residency. `selected_bytes` sums logical bytes of unique expert IDs;
/// `bytes_prepared` counts the union of their admitted page spans, clipped at
/// the mapping end. Advisory errors do not change inference behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MappedPagePreparationStats {
    pub requested_experts: usize,
    pub unique_experts: usize,
    pub duplicate_experts: usize,
    pub prepared_experts: usize,
    pub skipped_budget_experts: usize,
    pub selected_bytes: u64,
    pub bytes_prepared: u64,
    pub pages_prepared: u64,
    pub page_touches: u64,
    /// Number of merged ranges passed to madvise, including failed calls.
    pub advisory_ranges: u64,
    pub advisory_errors: u64,
    pub page_size: u64,
    pub elapsed_ms: f64,
}

// A routed batch is small. Bound stack work and refuse larger selections
// rather than allocating on every layer's inference path.
const MAX_PREPARATION_EXPERTS: usize = 64;

#[derive(Clone, Copy, Default)]
struct PageSpan {
    start: u64,
    end: u64,
}

fn merge_page_spans(spans: &mut [PageSpan]) -> usize {
    spans.sort_unstable_by_key(|span| span.start);
    let mut merged = 0;
    for index in 0..spans.len() {
        let span = spans[index];
        if merged > 0 && span.start <= spans[merged - 1].end {
            spans[merged - 1].end = spans[merged - 1].end.max(span.end);
        } else {
            spans[merged] = span;
            merged += 1;
        }
    }
    merged
}

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
    /// Refuses indices absent from the layout and ranges that do not fit in
    /// the mapped buffer. This is the GPU-facing boundary, so it must not use
    /// [`StreamLayout::expert_offset`]'s uniform fallback for a missing entry.
    pub fn expert_offset(&self, expert: usize) -> Result<u64, StreamerError> {
        let offset = if expert < self.layout.experts_per_layer {
            self.layout.expert_offset(0, expert)
        } else {
            return Err(StreamerError::OffsetOutOfRange {
                offset: expert as u64,
            });
        };
        let start = self
            .shift
            .checked_add(offset)
            .ok_or(StreamerError::OffsetOutOfRange { offset })?;
        let end = start
            .checked_add(self.layout.expert_stride)
            .ok_or(StreamerError::OffsetOutOfRange { offset: start })?;
        if end > self.mapping.mapped_bytes().len() as u64 {
            return Err(StreamerError::OffsetOutOfRange { offset: start });
        }
        Ok(start)
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
        let start = self.expert_offset(expert).ok()? as usize;
        let end = start.checked_add(self.expert_stride() as usize)?;
        self.mapping.mapped_bytes().get(start..end)
    }

    /// Touches each VM page in one expert's range without changing its bytes.
    /// Returns page-touch count, not disk bytes or a page-residency guarantee.
    pub fn prefetch_expert(&self, expert: usize) -> Result<u64, StreamerError> {
        if self
            .layout
            .expert_offsets
            .as_ref()
            .is_some_and(|offsets| expert >= offsets.len())
        {
            return Err(StreamerError::OffsetOutOfRange {
                offset: expert as u64,
            });
        }
        let start = self.expert_offset(expert)? as usize;
        let count =
            usize::try_from(self.expert_stride()).map_err(|_| StreamerError::OffsetOutOfRange {
                offset: start as u64,
            })?;
        if count == 0 {
            return Ok(0);
        }
        // SAFETY: sysconf reads a process constant and borrows no memory.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page_size = if page_size > 0 {
            page_size as usize
        } else {
            4096
        };
        let bytes = self.mapping.mapped_bytes();
        let end = start + count; // expert_offset checked this whole range.
        std::hint::black_box(bytes[start]);
        let mut touches = 1;
        let mut next = start.saturating_add(page_size - start % page_size);
        while next < end {
            std::hint::black_box(bytes[next]);
            touches += 1;
            next = next.saturating_add(page_size);
        }
        Ok(touches)
    }

    /// Prepares only the experts already selected for this layer.
    ///
    /// All IDs and byte ranges are validated before any read or advice.
    /// Duplicate IDs and overlapping pages are prepared once. Whole experts
    /// are admitted in caller order while the union of page spans fits
    /// `max_bytes`; a skipped expert does not prevent later smaller spans
    /// from fitting. A zero budget still validates all selections. This
    /// method does not change routing, bind order, or expert contents.
    pub fn prepare_selected_experts(
        &self,
        experts: &[usize],
        mode: MappedPagePreparationMode,
        max_bytes: u64,
    ) -> Result<MappedPagePreparationStats, StreamerError> {
        let started = std::time::Instant::now();
        if experts.len() > MAX_PREPARATION_EXPERTS {
            return Err(StreamerError::OpenFailed {
                path: self.layout.path.clone(),
                detail: format!(
                    "mapped preparation supports at most {MAX_PREPARATION_EXPERTS} selected experts"
                ),
            });
        }
        let bytes = self.mapping.mapped_bytes();
        let mut unique_ids = [0usize; MAX_PREPARATION_EXPERTS];
        let mut selected = [PageSpan::default(); MAX_PREPARATION_EXPERTS];
        let mut stats = MappedPagePreparationStats {
            requested_experts: experts.len(),
            ..Default::default()
        };
        for &expert in experts {
            // Explicit offset tables are authoritative, including missing
            // entries. Checked arithmetic also covers uniform layouts.
            if expert >= self.layout.experts_per_layer {
                return Err(StreamerError::OffsetOutOfRange {
                    offset: expert as u64,
                });
            }
            let offset = if let Some(offsets) = &self.layout.expert_offsets {
                *offsets.get(expert).ok_or(StreamerError::OffsetOutOfRange {
                    offset: expert as u64,
                })?
            } else {
                (expert as u64)
                    .checked_mul(self.layout.expert_stride)
                    .ok_or(StreamerError::OffsetOutOfRange {
                        offset: expert as u64,
                    })?
            };
            let start = self
                .shift
                .checked_add(offset)
                .ok_or(StreamerError::OffsetOutOfRange { offset })?;
            let end = start
                .checked_add(self.layout.expert_stride)
                .filter(|&end| end <= bytes.len() as u64)
                .ok_or(StreamerError::OffsetOutOfRange { offset: start })?;
            if unique_ids[..stats.unique_experts].contains(&expert) {
                stats.duplicate_experts += 1;
                continue;
            }
            stats.selected_bytes = stats
                .selected_bytes
                .checked_add(self.layout.expert_stride)
                .ok_or(StreamerError::OffsetOutOfRange { offset: start })?;
            unique_ids[stats.unique_experts] = expert;
            selected[stats.unique_experts] = PageSpan { start, end };
            stats.unique_experts += 1;
        }
        // SAFETY: sysconf reads a process constant and borrows no memory.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(StreamerError::OpenFailed {
                path: self.layout.path.clone(),
                detail: "cannot determine VM page size for mapped preparation".to_string(),
            });
        }
        stats.page_size = page_size as u64;
        let mut admitted = [PageSpan::default(); MAX_PREPARATION_EXPERTS];
        let mut admitted_count = 0;
        for &span in &selected[..stats.unique_experts] {
            if span.start == span.end {
                continue;
            }
            if max_bytes == 0 {
                stats.skipped_budget_experts += 1;
                continue;
            }
            let start = span.start / stats.page_size * stats.page_size;
            let tail = (stats.page_size - span.end % stats.page_size) % stats.page_size;
            let end = span.end.saturating_add(tail).min(bytes.len() as u64);
            let mut candidate = admitted;
            candidate[admitted_count] = PageSpan { start, end };
            let count = merge_page_spans(&mut candidate[..admitted_count + 1]);
            let candidate_bytes: u64 = candidate[..count]
                .iter()
                .map(|span| span.end - span.start)
                .sum();
            if candidate_bytes > max_bytes {
                stats.skipped_budget_experts += 1;
                continue;
            }
            admitted = candidate;
            admitted_count = count;
            stats.bytes_prepared = candidate_bytes;
            stats.prepared_experts += 1;
        }
        for span in &admitted[..admitted_count] {
            stats.pages_prepared += (span.end - span.start).div_ceil(stats.page_size);
            match mode {
                MappedPagePreparationMode::Advice => {
                    stats.advisory_ranges += 1;
                    // SAFETY: mmap's base and span.start are page-aligned;
                    // validation and clipping keep this nonempty range
                    // inside the immutable mapping for this whole call.
                    let result = unsafe {
                        libc::madvise(
                            bytes.as_ptr().add(span.start as usize).cast_mut().cast(),
                            (span.end - span.start) as usize,
                            libc::MADV_WILLNEED,
                        )
                    };
                    if result != 0 {
                        stats.advisory_errors += 1;
                    }
                }
                MappedPagePreparationMode::Touch => {
                    let mut page = span.start;
                    while page < span.end {
                        // SAFETY: each page address is inside the validated
                        // read-only mapping. A volatile read forces the CPU
                        // access even though the weight value is not used.
                        std::hint::black_box(unsafe {
                            bytes.as_ptr().add(page as usize).read_volatile()
                        });
                        stats.page_touches += 1;
                        page = page.saturating_add(stats.page_size);
                    }
                }
            }
        }
        stats.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(stats)
    }
}
