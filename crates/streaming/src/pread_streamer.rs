//! `pread`-based routed-expert streamer with a fixed per-layer slot cache.
//! Ported from `Infrastructure/Streaming/PreadExpertStreamer.swift`, minus
//! the `MTLBuffer` wrapping (GPU-specific, deferred to the `gpu` crate) and
//! the concurrent-read parallelism (reads run sequentially here; still
//! correct, just not pipelined — a later phase can reintroduce a thread
//! pool once there is a GPU consumer to pipeline against).

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

pub struct PreadExpertStreamer {
    layout: StreamLayout,
    file: File,
    slot_count: usize,
    slots: Vec<Vec<u8>>,
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
            .map(|_| vec![0u8; layout.expert_stride as usize])
            .collect();
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
        &self.slots[slot]
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
        read_full(&self.file, &mut self.slots[slot], file_offset)
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
    /// Returns the assigned slot for each requested expert, in request order.
    pub fn execute_expert_cache_plan(
        &mut self,
        plan: &ExpertCachePlan,
    ) -> Result<Vec<usize>, StreamerError> {
        for &index in &plan.misses {
            self.load_expert_into_slot(0, plan.experts[index], plan.assigned_slots[index])?;
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
