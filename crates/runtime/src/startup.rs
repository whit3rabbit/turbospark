//! Startup measurements and bounded, caller-selected expert preparation.

use std::collections::HashSet;
use std::time::Instant;

use crate::{RealForwardError, RealForwardRunner};

/// Disjoint wall-clock phases of a successful runner open, in milliseconds.
/// Compilation counters are a separate axis and overlap these phases.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StartupStats {
    pub total_open_ms: f64,
    pub manifest_index_ms: f64,
    pub resident_mapping_ms: f64,
    pub kv_scratch_ms: f64,
    pub expert_setup_ms: f64,
    pub family_state_ms: f64,
    pub session_state_ms: f64,
    pub kernel_warmup: gpu::KernelWarmupStats,
}

/// Logical ranges prepared, not a guarantee that the OS retains their pages.
/// Mapped ranges are page-touched; streamed ranges are read into bounded scratch.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExpertPrefetchStats {
    pub requested_experts: usize,
    pub prepared_experts: usize,
    pub duplicate_experts: usize,
    pub skipped_budget_experts: usize,
    pub bytes_prepared: u64,
    pub mapped_page_touches: u64,
    pub elapsed_ms: f64,
}

impl RealForwardRunner {
    pub fn startup_stats(&self) -> StartupStats {
        self.startup_stats
    }

    /// Includes open-time and subsequent lazy compilation in this context.
    pub fn metal_compilation_stats(&self) -> gpu::MetalCompilationStats {
        self.context.compilation_stats()
    }

    /// Persists prepared pipelines; cache-write failures remain nonfatal.
    pub fn flush_pipeline_cache(&mut self) {
        self.context.flush_pipeline_cache();
    }

    /// Prepares caller-selected packed experts without changing slot residency,
    /// routing, or model state. Nothing calls this on the default open path.
    ///
    /// The budget covers whole logical expert ranges. Duplicate selections are
    /// read once, in first-selection order. All selections are checked before
    /// I/O, even those that do not fit. Count this call's elapsed time in TTFT.
    /// OS readahead can read more physical bytes than the selected ranges, and
    /// the OS may evict pages immediately after the call returns.
    pub fn prefetch_experts(
        &self,
        selections: &[(usize, usize)],
        byte_budget: u64,
    ) -> Result<ExpertPrefetchStats, RealForwardError> {
        let started = Instant::now();
        let (plan, mut stats) = prefetch_plan(selections, byte_budget, |layer, expert| {
            let layout = if let Some(Some(mapped)) = self.mapped.layers.get(layer) {
                mapped.layout()
            } else if let Some(Some(streamer)) = self.streamers.get(layer) {
                if streamer.nocache_active() {
                    return Err(RealForwardError::Unsupported(format!(
                        "expert prefetch cannot warm layer {layer} while F_NOCACHE is active"
                    )));
                }
                streamer.layout()
            } else {
                return Err(RealForwardError::Unsupported(format!(
                    "expert prefetch selection names layer {layer}, which has no packed experts"
                )));
            };
            validate_range(layout, expert).map_err(|detail| {
                RealForwardError::Unsupported(format!(
                    "expert prefetch layer {layer}, expert {expert}: {detail}"
                ))
            })?;
            Ok(layout.expert_stride)
        })?;
        // Allocation is bounded independently of model size and trace length.
        let mut scratch = vec![0u8; if plan.is_empty() { 0 } else { 64 * 1024 }];
        for (layer, expert, bytes) in plan {
            if let Some(Some(mapped)) = self.mapped.layers.get(layer) {
                stats.mapped_page_touches += mapped
                    .prefetch_expert(expert)
                    .map_err(|e| RealForwardError::Unsupported(format!("expert prefetch: {e}")))?;
            } else {
                self.streamers[layer]
                    .as_ref()
                    .expect("prefetch plan validated the streamer")
                    .prefetch_expert(expert, &mut scratch)
                    .map_err(|e| RealForwardError::Unsupported(format!("expert prefetch: {e}")))?;
            }
            stats.prepared_experts += 1;
            stats.bytes_prepared += bytes;
        }
        stats.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(stats)
    }
}

fn validate_range(layout: &streaming::StreamLayout, expert: usize) -> Result<(), &'static str> {
    if expert >= layout.experts_per_layer {
        return Err("expert index is outside the layout");
    }
    let offset = match &layout.expert_offsets {
        Some(offsets) => *offsets.get(expert).ok_or("expert offset is missing")?,
        None => (expert as u64)
            .checked_mul(layout.expert_stride)
            .ok_or("expert offset overflows")?,
    };
    let end = offset
        .checked_add(layout.expert_stride)
        .ok_or("expert range overflows")?;
    if end > layout.stream_size {
        return Err("expert range is outside the stream window");
    }
    layout
        .stream_offset
        .checked_add(end)
        .ok_or("expert file offset overflows")?;
    Ok(())
}

type ExpertPrefetchPlan = Vec<(usize, usize, u64)>;

fn prefetch_plan(
    selections: &[(usize, usize)],
    byte_budget: u64,
    mut validate: impl FnMut(usize, usize) -> Result<u64, RealForwardError>,
) -> Result<(ExpertPrefetchPlan, ExpertPrefetchStats), RealForwardError> {
    let mut stats = ExpertPrefetchStats {
        requested_experts: selections.len(),
        ..ExpertPrefetchStats::default()
    };
    let mut seen = HashSet::new();
    let mut remaining = byte_budget;
    let mut plan = Vec::new();
    for &(layer, expert) in selections {
        let bytes = validate(layer, expert)?;
        if !seen.insert((layer, expert)) {
            stats.duplicate_experts += 1;
        } else if bytes > remaining {
            stats.skipped_budget_experts += 1;
        } else {
            remaining -= bytes;
            plan.push((layer, expert, bytes));
        }
    }
    Ok((plan, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_preserves_trace_order_deduplicates_and_never_exceeds_budget() {
        let (plan, stats) = prefetch_plan(&[(1, 2), (1, 2), (0, 1), (2, 0)], 11, |layer, _| {
            Ok(if layer == 2 { 3 } else { 8 })
        })
        .unwrap();
        assert_eq!(plan, vec![(1, 2, 8), (2, 0, 3)]);
        assert_eq!(stats.requested_experts, 4);
        assert_eq!(stats.duplicate_experts, 1);
        assert_eq!(stats.skipped_budget_experts, 1);
    }

    #[test]
    fn zero_budget_still_validates_the_whole_trace() {
        let result = prefetch_plan(&[(0, 0), (9, 1)], 0, |layer, _| {
            if layer == 9 {
                Err(RealForwardError::Unsupported("invalid layer".into()))
            } else {
                Ok(8)
            }
        });
        assert!(result.is_err());
        let (plan, stats) = prefetch_plan(&[(0, 0)], 0, |_, _| Ok(8)).unwrap();
        assert!(plan.is_empty());
        assert_eq!(stats.skipped_budget_experts, 1);
    }

    #[test]
    fn range_validation_rejects_missing_offsets_and_overflow() {
        let mut layout = streaming::StreamLayout {
            path: String::new(),
            stream_offset: 0,
            stream_size: 16,
            experts_per_layer: 2,
            expert_stride: 8,
            expert_offsets: Some(vec![0]),
        };
        assert!(validate_range(&layout, 1).is_err());
        layout.expert_offsets = Some(vec![u64::MAX]);
        assert!(validate_range(&layout, 0).is_err());
        layout.expert_offsets = None;
        layout.stream_offset = u64::MAX - 4;
        assert!(validate_range(&layout, 0).is_err());
    }
}
