//! `pread`-based routed-expert streamer with a fixed per-layer slot cache.
//! Ported from `Infrastructure/Streaming/PreadExpertStreamer.swift`. Slot
//! memory matches the Swift original's shape: one `posix_memalign`(2 MiB)
//! page-rounded allocation per slot, made once at open and reused forever
//! (the decode hot path never allocates), so a GPU backend can wrap each
//! slot zero-copy with `newBufferWithBytesNoCopy` (the `MTLBuffer`
//! wrapping itself stays in the `gpu` crate).
//!
//! Cache-plan misses are read in parallel. The Swift original's shape was
//! `DispatchQueue.concurrentPerform` over the misses, one task per miss;
//! this port splits each miss into chunks and runs them on the shared
//! `read_pool`, because one thread per miss collapses to a single-threaded
//! copy on the common warm-cache layer that misses exactly once. See
//! [`PreadExpertStreamer::execute_expert_cache_plan`] for the measurement.

use std::collections::HashSet;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;

use crate::aligned_slot::AlignedSlot;
use crate::disk_io::{self, ExpertIoStats};
use crate::error::StreamerError;
use crate::expert_cache::{
    coalesced_adjacent_advice_ranges, ExpertCache, ExpertCachePlan, ExpertCachePolicy,
    ExpertIoAdviceResult,
};
use crate::rdadvice;
use crate::read_pool::{self, ReadChunk};
use crate::stream_layout::StreamLayout;

/// Bytes of one expert's blob read per parallel chunk. At the real Gemma 4
/// stride (3,358,720 B) this is a 4-way split of a single miss.
const MISS_READ_CHUNK_BYTES: usize = 840 * 1024;

pub struct PreadExpertStreamer {
    layout: StreamLayout,
    file: File,
    slot_count: usize,
    slots: Vec<AlignedSlot>,
    cache: ExpertCache,
    next_slot: usize,
    io_stats: ExpertIoStats,
    /// Whether this streamer's descriptor is actually bypassing the unified
    /// buffer cache. Distinct from `disk_io::nocache_requested()`: the
    /// `fcntl` can be refused, and a run that silently measured the warm
    /// path under `MFERENCE_EXPERT_NOCACHE=1` would be the worst outcome
    /// available (a published disk-bound row that is not one).
    nocache: bool,
}

impl PreadExpertStreamer {
    /// Opens expert streamer with layout, pre-allocated slots, and cache policy.
    pub fn open(
        layout: StreamLayout,
        slot_count: usize,
        cache_policy: ExpertCachePolicy,
    ) -> Result<Self, StreamerError> {
        assert!(slot_count > 0, "slot_count must be positive");
        let file = File::open(&layout.path).map_err(|e| StreamerError::OpenFailed {
            path: layout.path.clone(),
            detail: e.to_string(),
        })?;
        // Propagated rather than swallowed: an `if let Ok(meta)` here skips
        // the size check silently on a failing stat, which is the one case
        // where the file is most likely to be wrong.
        let meta = file.metadata().map_err(|e| StreamerError::OpenFailed {
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
        let slots = (0..slot_count)
            .map(|_| AlignedSlot::allocate(layout.expert_stride as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let cache = ExpertCache::new(slot_count, cache_policy, layout.experts_per_layer);
        // The experimental disk-bound condition, established once per open
        // rather than per read: `F_NOCACHE` is a property of the
        // DESCRIPTOR, so it cannot be toggled between batches, and reading
        // the env once matches the `MFERENCE_READ_QOS` seam next door.
        let nocache = disk_io::nocache_requested() && disk_io::set_nocache(file.as_raw_fd());
        Ok(Self {
            layout,
            file,
            slot_count,
            slots,
            cache,
            next_slot: 0,
            io_stats: ExpertIoStats::default(),
            nocache,
        })
    }

    /// Cumulative miss-read byte accounting for this streamer. Zero
    /// `samples` means the physical-I/O probe was off or unavailable, so
    /// `bytes_physical` says nothing at all rather than saying "no disk
    /// reads" (`disk_io::ExpertIoStats`).
    pub fn io_stats(&self) -> ExpertIoStats {
        self.io_stats
    }

    /// Whether this streamer's reads are actually bypassing the buffer
    /// cache. False when the seam was never asked for AND when the `fcntl`
    /// was refused; the caller reporting a disk-bound arm must print this
    /// rather than the env variable.
    pub fn nocache_active(&self) -> bool {
        self.nocache
    }

    /// The stream layout this streamer was built with (expert stride,
    /// per-expert offsets, file window).
    pub fn layout(&self) -> &StreamLayout {
        &self.layout
    }

    /// Returns reference to slot bytes slice for a given slot index.
    pub fn slot_data(&self, slot: usize) -> &[u8] {
        &self.slots[slot].as_slice()[..self.layout.expert_stride as usize]
    }

    /// The slot's page-aligned backing allocation, for a GPU backend to
    /// wrap zero-copy (`newBufferWithBytesNoCopy`). The pointer stays
    /// valid for this streamer's whole lifetime; slot contents change as
    /// experts stream through the cache.
    pub fn slot_allocation(&self, slot: usize) -> (*const u8, usize) {
        (self.slots[slot].as_ptr(), self.slots[slot].len())
    }

    /// Loads an expert into the next slot index in round-robin fashion.
    pub fn load_expert(&mut self, layer: usize, expert: usize) -> Result<usize, StreamerError> {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slot_count;
        self.load_expert_into_slot(layer, expert, slot)?;
        Ok(slot)
    }

    /// Loads an expert directly into a specific target slot index,
    /// BYPASSING the cache plan.
    ///
    /// The slot's residency record is dropped first, so a later
    /// [`Self::plan_experts_cached`] re-reads whatever used to live there
    /// rather than reporting a hit on bytes this call overwrote. Before the
    /// slot's bytes, not after: a failed read leaves the slot half-written
    /// too, and a half-written slot is not the expert the cache thinks it
    /// is either. Mixing this with the cached path was safe only by
    /// convention, and the failure it invited is Gotcha 27's shape --
    /// correct-looking output computed from the wrong expert.
    pub fn load_expert_into_slot(
        &mut self,
        layer: usize,
        expert: usize,
        slot: usize,
    ) -> Result<(), StreamerError> {
        if slot >= self.slot_count {
            return Err(StreamerError::SlotOutOfRange { slot });
        }
        self.cache.invalidate_slot(slot);
        let region_offset = self.layout.expert_offset(layer, expert);
        if region_offset + self.layout.expert_stride > self.layout.stream_size {
            return Err(StreamerError::OffsetOutOfRange {
                offset: region_offset,
            });
        }
        let file_offset = self.layout.stream_offset + region_offset;
        let stride = self.layout.expert_stride as usize;
        read_full(
            &self.file,
            &mut self.slots[slot].as_mut_slice()[..stride],
            file_offset,
        )
    }

    /// Generates an expert cache plan for requested expert indices.
    pub fn plan_experts_cached(
        &mut self,
        experts: &[usize],
        avoiding_slots: &HashSet<usize>,
    ) -> ExpertCachePlan {
        self.cache.plan(experts, avoiding_slots)
    }

    /// Generates cache plan if possible without evicting pinned slots.
    pub fn plan_experts_cached_if_possible(
        &mut self,
        experts: &[usize],
        avoiding_slots: &HashSet<usize>,
    ) -> Option<ExpertCachePlan> {
        self.cache.plan_if_possible(experts, avoiding_slots)
    }

    /// Loads every miss in `plan` (layer 0, matching the Swift original's
    /// cache which is scoped to one layer at a time) and commits the plan.
    /// Returns the assigned slot for each requested expert, in request
    /// order.
    ///
    /// Misses are read in parallel, but the unit of parallelism is a
    /// CHUNK of one expert's blob, not a whole expert. The Swift original
    /// (and this port until 2026-08-06) ran one thread per miss, which
    /// silently degrades to a single-threaded copy whenever a layer has
    /// only one miss -- the common case once the cache is warm. On the
    /// real 26B install at 32 slots the average is 1.3 misses per layer,
    /// and the measured read rate was 23.8 GiB/s against 44.8 GiB/s on the
    /// same machine and code path when 8-slot runs forced ~5 misses per
    /// layer. Splitting each miss decouples thread count from miss count,
    /// so a lone miss gets the same width as a busy layer.
    ///
    /// Note what this is really optimizing: with the install's expert
    /// files in page cache, `pread` here is a memcpy, not disk I/O
    /// (125 MiB per token in 5.26 ms is far past any SSD). On a cold or
    /// memory-tight machine this path is disk-bound instead and the
    /// chunking buys much less.
    pub fn execute_expert_cache_plan(
        &mut self,
        plan: &ExpertCachePlan,
    ) -> Result<Vec<usize>, StreamerError> {
        if plan.misses.is_empty() {
            self.cache.commit_plan(plan);
            return Ok(plan.assigned_slots.clone());
        }

        // Distinct-slot guarantee backs the disjoint parallel writes.
        let mut seen = HashSet::new();
        for &index in &plan.misses {
            assert!(
                seen.insert(plan.assigned_slots[index]),
                "cache plan assigned one slot to two misses"
            );
        }
        for &index in &plan.misses {
            let slot = plan.assigned_slots[index];
            if slot >= self.slot_count {
                return Err(StreamerError::SlotOutOfRange { slot });
            }
            let region_offset = self.layout.expert_offset(0, plan.experts[index]);
            if region_offset + self.layout.expert_stride > self.layout.stream_size {
                return Err(StreamerError::OffsetOutOfRange {
                    offset: region_offset,
                });
            }
        }

        let stride = self.layout.expert_stride as usize;
        // Derived, not a constant: a fixed 4 is Gemma's stride against this
        // chunk size and nobody else's. A Mixtral-class 108.9 MiB expert is
        // 133 chunks (Gotcha 36's granularity axis), which under-reserves.
        let chunks_per_miss = stride.div_ceil(MISS_READ_CHUNK_BYTES).max(1);
        let mut chunks: Vec<ReadChunk> = Vec::with_capacity(plan.misses.len() * chunks_per_miss);
        for &index in &plan.misses {
            let file_offset =
                self.layout.stream_offset + self.layout.expert_offset(0, plan.experts[index]);
            let base = self.slots[plan.assigned_slots[index]].as_ptr() as *mut u8;
            let mut start = 0usize;
            while start < stride {
                let len = MISS_READ_CHUNK_BYTES.min(stride - start);
                chunks.push(ReadChunk {
                    // SAFETY: `start` stays below `stride`, which is
                    // within the slot's page-rounded allocation.
                    #[allow(unsafe_code)]
                    dest: unsafe { base.add(start) },
                    len,
                    file_offset: file_offset + start as u64,
                });
                start += len;
            }
        }

        // Requested bytes are free to count and always counted: misses
        // only, since a hit reads nothing. Physical bytes cost a syscall
        // pair per batch, so they are sampled only under the seam.
        self.io_stats.bytes_requested += (plan.misses.len() * stride) as u64;
        self.io_stats.batches += 1;
        let physical_before = disk_io::measure_physical_io()
            .then(disk_io::process_disk_bytes_read)
            .flatten();

        let result = read_pool::run_batch(&self.file, &chunks);

        // Sampled even when the batch FAILED: the bytes a partial read
        // pulled off the device were still pulled, and a probe that only
        // fires on the happy path is the shape AGENTS.md Gotcha 38 calls
        // close to no guard at all.
        if let (Some(before), Some(after)) = (physical_before, disk_io::process_disk_bytes_read()) {
            self.io_stats.bytes_physical += after.saturating_sub(before);
            self.io_stats.samples += 1;
        }
        result?;

        self.cache.commit_plan(plan);
        Ok(plan.assigned_slots.clone())
    }

    /// Plans and executes expert caching for requested expert indices.
    pub fn load_experts_cached(&mut self, experts: &[usize]) -> Result<Vec<usize>, StreamerError> {
        let plan = self.plan_experts_cached(experts, &HashSet::new());
        self.execute_expert_cache_plan(&plan)
    }

    /// Identifies requested expert indices that are not currently resident in cache.
    pub fn non_resident_experts(&self, experts: &[usize]) -> Vec<usize> {
        self.cache.non_resident_experts(experts)
    }

    /// Reserves slots for speculative expert loading.
    pub fn reserve_speculative_slots(
        &mut self,
        experts: &[usize],
        keep_evictable: usize,
    ) -> Vec<(usize, usize)> {
        self.cache
            .reserve_speculative_slots(experts, keep_evictable)
    }

    /// Runs a reservation from [`Self::reserve_speculative_slots`] and
    /// publishes the results. Returns the bytes read.
    pub fn execute_speculative_reservation(&mut self, reservation: &[(usize, usize)]) -> u64 {
        let mut loaded = Vec::with_capacity(reservation.len());
        for (index, &(expert, slot)) in reservation.iter().enumerate() {
            if self.load_expert_into_slot(0, expert, slot).is_ok() {
                loaded.push(index);
            }
        }
        self.cache
            .publish_speculative_reservation(reservation, &loaded);
        loaded.len() as u64 * self.layout.expert_stride
    }

    /// Returns snapshot of expert index per slot.
    pub fn resident_experts_snapshot(&self) -> Vec<Option<usize>> {
        self.cache.resident_experts_snapshot()
    }

    /// Issues OS I/O advice for misses in an expert cache plan.
    pub fn advise_expert_cache_plan_misses(&self, plan: &ExpertCachePlan) -> ExpertIoAdviceResult {
        let experts: Vec<usize> = plan.misses.iter().map(|&i| plan.experts[i]).collect();
        self.advise_ranges(&self.expert_advice_ranges(&experts), experts.len())
    }

    /// Issues OS I/O advice for requested expert indices.
    pub fn advise_experts(&self, experts: &[usize]) -> ExpertIoAdviceResult {
        self.advise_ranges(&self.expert_advice_ranges(experts), experts.len())
    }

    /// Issues OS I/O advice for non-resident expert indices.
    pub fn advise_expert_misses(&self, experts: &[usize]) -> ExpertIoAdviceResult {
        let misses = self.cache.non_resident_experts(experts);
        self.advise_ranges(&self.expert_advice_ranges(&misses), misses.len())
    }

    /// Byte ranges to advise for `experts`. Out-of-range experts are
    /// dropped here, which is why every caller hands the ORIGINAL count to
    /// [`Self::advise_ranges`] as well: the difference is what gets
    /// reported as `skipped`.
    fn expert_advice_ranges(&self, experts: &[usize]) -> Vec<(u64, u64)> {
        experts
            .iter()
            .filter_map(|&expert| {
                let region_offset = self.layout.expert_offset(0, expert);
                if region_offset + self.layout.expert_stride > self.layout.stream_size {
                    return None;
                }
                Some((
                    self.layout.stream_offset + region_offset,
                    self.layout.expert_stride,
                ))
            })
            .collect()
    }

    fn advise_ranges(&self, ranges: &[(u64, u64)], requested: usize) -> ExpertIoAdviceResult {
        // Readahead and cache-bypass are contradictory instructions about
        // the same descriptor: `F_RDADVISE` asks the kernel to populate the
        // buffer cache and `F_NOCACHE` asks it not to keep anything there.
        // A run issuing both measures neither condition cleanly, so the
        // disk-bound arm reports every range as SKIPPED rather than
        // quietly warming the cache it was supposed to be bypassing.
        if self.nocache {
            let bytes = ranges
                .iter()
                .map(|&(_, count)| rdadvice::clipped_byte_count(count))
                .sum();
            return ExpertIoAdviceResult::skipped(requested, bytes);
        }
        let coalesced = coalesced_adjacent_advice_ranges(ranges);
        let mut failed = 0;
        let mut bytes = 0u64;
        let mut max_call_nanos = 0u64;
        let fd = self.file.as_raw_fd();
        for &(offset, count) in &coalesced {
            let result = rdadvice::call(fd, offset, count);
            if !result.succeeded {
                failed += 1;
            }
            bytes += result.requested_bytes;
            max_call_nanos = max_call_nanos.max(result.elapsed_nanos);
        }
        ExpertIoAdviceResult {
            requested,
            failed,
            calls: coalesced.len(),
            bytes,
            // Hardcoded 0 before, which made the struct fail to add up:
            // `expert_advice_ranges` silently drops experts whose blob falls
            // outside the stream window, so a caller comparing `requested`
            // against what was covered saw the shortfall attributed nowhere.
            skipped: requested.saturating_sub(ranges.len()),
            max_call_nanos,
        }
    }
}

pub(crate) fn read_full(
    file: &File,
    destination: &mut [u8],
    file_offset: u64,
) -> Result<(), StreamerError> {
    let mut filled = 0usize;
    while filled < destination.len() {
        let got = file
            .read_at(&mut destination[filled..], file_offset + filled as u64)
            .map_err(|e| StreamerError::PreadFailed {
                detail: e.to_string(),
            })?;
        if got == 0 {
            return Err(StreamerError::SizeMismatch {
                expected: destination.len() as u64,
                actual: filled as u64,
            });
        }
        filled += got;
    }
    Ok(())
}
