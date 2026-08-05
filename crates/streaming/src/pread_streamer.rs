//! `pread`-based routed-expert streamer with a fixed per-layer slot cache.
//! Ported from `Infrastructure/Streaming/PreadExpertStreamer.swift`. Slot
//! memory matches the Swift original's shape: one `posix_memalign`(2 MiB)
//! page-rounded allocation per slot, made once at open and reused forever
//! (the decode hot path never allocates), so a GPU backend can wrap each
//! slot zero-copy with `newBufferWithBytesNoCopy` (the `MTLBuffer`
//! wrapping itself stays in the `gpu` crate). Cache-plan misses are read
//! in parallel, one thread per miss into its own disjoint slot — the
//! `DispatchQueue.concurrentPerform` equivalent.

use std::collections::HashSet;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;

use crate::error::StreamerError;
use crate::expert_cache::{
    coalesced_adjacent_advice_ranges, ExpertCache, ExpertCachePlan, ExpertCachePolicy,
    ExpertIoAdviceResult,
};
use crate::rdadvice;
use crate::stream_layout::StreamLayout;

/// The Swift original's `scratchAlignment`: slot bases are 2 MiB-aligned
/// (comfortably page-aligned on any page size), sized up to whole pages.
const SLOT_ALIGNMENT: usize = 2 * 1024 * 1024;

/// One expert slot's backing memory: page-aligned, page-rounded, allocated
/// once. Exposes its base pointer so a GPU backend can wrap it no-copy.
pub struct AlignedSlot {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the allocation is plain heap memory; the streamer alone decides
// which threads write which slot (disjointly, during plan execution).
#[allow(unsafe_code)]
unsafe impl Send for AlignedSlot {}
#[allow(unsafe_code)]
unsafe impl Sync for AlignedSlot {}

impl AlignedSlot {
    fn allocate(len: usize) -> Result<Self, StreamerError> {
        let page = page_size();
        let rounded = len.div_ceil(page) * page;
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: standard posix_memalign call; alignment is a power of
        // two and a multiple of pointer size; failure is checked below.
        #[allow(unsafe_code)]
        let rc = unsafe { libc::posix_memalign(&mut raw, SLOT_ALIGNMENT, rounded.max(page)) };
        if rc != 0 || raw.is_null() {
            return Err(StreamerError::PreadFailed {
                detail: format!("posix_memalign failed with {rc}"),
            });
        }
        // SAFETY: freshly allocated, at least `rounded` bytes.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::write_bytes(raw as *mut u8, 0, rounded.max(page));
        }
        Ok(Self {
            ptr: raw as *mut u8,
            len: rounded.max(page),
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is a live allocation of `len` bytes owned by self.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, with exclusive access through &mut self.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }
}

impl Drop for AlignedSlot {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from posix_memalign and is freed exactly once.
        #[allow(unsafe_code)]
        unsafe {
            libc::free(self.ptr as *mut std::ffi::c_void);
        }
    }
}

fn page_size() -> usize {
    // SAFETY: reads a process constant.
    #[allow(unsafe_code)]
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page > 0 {
        page as usize
    } else {
        4096
    }
}

pub struct PreadExpertStreamer {
    layout: StreamLayout,
    file: File,
    slot_count: usize,
    slots: Vec<AlignedSlot>,
    cache: ExpertCache,
    next_slot: usize,
}

impl PreadExpertStreamer {
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
        if let Ok(meta) = file.metadata() {
            let required = layout.stream_offset + layout.stream_size;
            if meta.len() < required {
                return Err(StreamerError::SizeMismatch {
                    expected: required,
                    actual: meta.len(),
                });
            }
        }
        let slots = (0..slot_count)
            .map(|_| AlignedSlot::allocate(layout.expert_stride as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let cache = ExpertCache::new(slot_count, cache_policy, layout.experts_per_layer);
        Ok(Self {
            layout,
            file,
            slot_count,
            slots,
            cache,
            next_slot: 0,
        })
    }

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

    pub fn load_expert(&mut self, layer: usize, expert: usize) -> Result<usize, StreamerError> {
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slot_count;
        self.load_expert_into_slot(layer, expert, slot)?;
        Ok(slot)
    }

    pub fn load_expert_into_slot(
        &mut self,
        layer: usize,
        expert: usize,
        slot: usize,
    ) -> Result<(), StreamerError> {
        if slot >= self.slot_count {
            return Err(StreamerError::SlotOutOfRange { slot });
        }
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

    pub fn plan_experts_cached(
        &mut self,
        experts: &[usize],
        avoiding_slots: &HashSet<usize>,
    ) -> ExpertCachePlan {
        self.cache.plan(experts, avoiding_slots)
    }

    pub fn plan_experts_cached_if_possible(
        &mut self,
        experts: &[usize],
        avoiding_slots: &HashSet<usize>,
    ) -> Option<ExpertCachePlan> {
        self.cache.plan_if_possible(experts, avoiding_slots)
    }

    /// Loads every miss in `plan` (layer 0, matching the Swift original's
    /// cache which is scoped to one layer at a time) and commits the plan.
    /// Misses are read in parallel — one thread per miss, each writing its
    /// own disjoint slot, matching the Swift original's
    /// `DispatchQueue.concurrentPerform` shape. Returns the assigned slot
    /// for each requested expert, in request order.
    pub fn execute_expert_cache_plan(
        &mut self,
        plan: &ExpertCachePlan,
    ) -> Result<Vec<usize>, StreamerError> {
        if plan.misses.len() <= 1 {
            for &index in &plan.misses {
                self.load_expert_into_slot(0, plan.experts[index], plan.assigned_slots[index])?;
            }
        } else {
            // Distinct-slot guarantee backs the disjoint parallel writes.
            let mut seen = HashSet::new();
            for &index in &plan.misses {
                assert!(
                    seen.insert(plan.assigned_slots[index]),
                    "cache plan assigned one slot to two misses"
                );
            }

            struct MissRead {
                expert: usize,
                slot: usize,
            }
            let reads: Vec<MissRead> = plan
                .misses
                .iter()
                .map(|&index| MissRead {
                    expert: plan.experts[index],
                    slot: plan.assigned_slots[index],
                })
                .collect();
            for read in &reads {
                if read.slot >= self.slot_count {
                    return Err(StreamerError::SlotOutOfRange { slot: read.slot });
                }
                let region_offset = self.layout.expert_offset(0, read.expert);
                if region_offset + self.layout.expert_stride > self.layout.stream_size {
                    return Err(StreamerError::OffsetOutOfRange {
                        offset: region_offset,
                    });
                }
            }

            let file = &self.file;
            let layout = &self.layout;
            let stride = layout.expert_stride as usize;
            let slots = &self.slots;
            let first_error = std::sync::Mutex::new(None::<StreamerError>);
            std::thread::scope(|scope| {
                for read in &reads {
                    let first_error = &first_error;
                    scope.spawn(move || {
                        let region_offset = layout.expert_offset(0, read.expert);
                        let file_offset = layout.stream_offset + region_offset;
                        let slot = &slots[read.slot];
                        // SAFETY: each spawned read owns a distinct slot
                        // (asserted above), so these mutable views never
                        // alias; the underlying allocations outlive the
                        // scope (owned by self).
                        #[allow(unsafe_code)]
                        let dest = unsafe {
                            std::slice::from_raw_parts_mut(slot.as_ptr() as *mut u8, stride)
                        };
                        if let Err(e) = read_full(file, dest, file_offset) {
                            let mut guard = first_error.lock().unwrap();
                            if guard.is_none() {
                                *guard = Some(e);
                            }
                        }
                    });
                }
            });
            if let Some(e) = first_error.into_inner().unwrap() {
                return Err(e);
            }
        }
        self.cache.commit_plan(plan);
        Ok(plan.assigned_slots.clone())
    }

    pub fn load_experts_cached(&mut self, experts: &[usize]) -> Result<Vec<usize>, StreamerError> {
        let plan = self.plan_experts_cached(experts, &HashSet::new());
        self.execute_expert_cache_plan(&plan)
    }

    pub fn non_resident_experts(&self, experts: &[usize]) -> Vec<usize> {
        self.cache.non_resident_experts(experts)
    }

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

    pub fn resident_experts_snapshot(&self) -> Vec<Option<usize>> {
        self.cache.resident_experts_snapshot()
    }

    pub fn advise_expert_cache_plan_misses(&self, plan: &ExpertCachePlan) -> ExpertIoAdviceResult {
        let experts: Vec<usize> = plan.misses.iter().map(|&i| plan.experts[i]).collect();
        self.advise_ranges(&self.expert_advice_ranges(&experts), experts.len())
    }

    pub fn advise_experts(&self, experts: &[usize]) -> ExpertIoAdviceResult {
        self.advise_ranges(&self.expert_advice_ranges(experts), experts.len())
    }

    pub fn advise_expert_misses(&self, experts: &[usize]) -> ExpertIoAdviceResult {
        let misses = self.cache.non_resident_experts(experts);
        self.advise_ranges(&self.expert_advice_ranges(&misses), misses.len())
    }

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
            skipped: 0,
            max_call_nanos,
        }
    }
}

fn read_full(file: &File, destination: &mut [u8], file_offset: u64) -> Result<(), StreamerError> {
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
