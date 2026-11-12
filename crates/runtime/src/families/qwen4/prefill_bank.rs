use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use crate::real_forward::RealForwardRunner;
use crate::real_forward_dispatch::router_topk_gemma4;
use crate::real_forward_types::{PhaseCounters, RealForwardError, MAX_PREFILL_BATCH};

use super::RealQwen4State;

pub(super) const WIDE_ROWS: usize = 128;
static PREFILL_ADMISSION: Mutex<()> = Mutex::new(());

pub(super) struct PrefillResources {
    pub(super) buffers: [gpu::MetalBuffer; 5],
    pub(super) bank: Option<PrefillBank>,
    // Keep admission exclusive until the temporary allocations are gone.
    _admission: MutexGuard<'static, ()>,
}

#[derive(Clone, Copy)]
pub(crate) enum PrefillExperiment {
    Wide,
    Bank,
}

impl RealForwardRunner {
    /// Whether the loaded configuration can use the shared expert bank on a text prefill.
    pub fn supports_moe_prefill_bank(&self) -> bool {
        self.real_qwen4
            .as_ref()
            .is_some_and(|state| state.tap_capture.is_none())
            && self.resolved_residency == model_io::ResolvedExpertResidency::Streamed
            && !self.routed_batch_prefill
            && !self.batched_gemv_prefill
    }

    /// Configure the session before serving it. An omitted request preserves the CLI experiment.
    pub fn configure_moe_prefill_bank(
        &mut self,
        enabled: Option<bool>,
        guard: model_io::LoadGuard,
        temporary_limit: Option<u64>,
    ) {
        let supported = self.supports_moe_prefill_bank();
        if let Some(state) = self.real_qwen4.as_mut() {
            state.prefill_temporary_limit = temporary_limit;
            state.prefill_load_guard = guard;
            if let Some(enabled) = enabled {
                state.prefill_experiment =
                    (supported && enabled).then_some(PrefillExperiment::Bank);
            }
        }
    }

    /// Session configuration only. An individual prefill still needs memory admission.
    pub fn moe_prefill_bank_enabled(&self) -> bool {
        self.supports_moe_prefill_bank()
            && self.real_qwen4.as_ref().is_some_and(|state| {
                matches!(state.prefill_experiment, Some(PrefillExperiment::Bank))
            })
    }
}

impl PrefillExperiment {
    pub(super) fn from_env() -> Result<Option<Self>, RealForwardError> {
        match std::env::var("TURBOSPARK_QWEN4_PREFILL").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("off") => Ok(None),
            Ok("wide") => Ok(Some(Self::Wide)),
            Ok("bank") => Ok(Some(Self::Bank)),
            _ => Err(RealForwardError::Unsupported(
                "TURBOSPARK_QWEN4_PREFILL expects off, wide, or bank".into(),
            )),
        }
    }

    pub(super) fn allocate(
        self,
        runner: &mut RealForwardRunner,
        rows: usize,
    ) -> Result<PrefillResources, String> {
        if !runner.supports_moe_prefill_bank() || runner.prompt_vision.is_some() {
            return Err("requires streamed text prefill with the per-token routed kernels".into());
        }
        let admission = PREFILL_ADMISSION
            .try_lock()
            .map_err(|_| "another experimental prefill is active")?;
        let bank_bytes = match self {
            Self::Wide => 0,
            Self::Bank => runner
                .streamers
                .iter()
                .flatten()
                .map(|streamer| streamer.layout())
                .map(|layout| layout.expert_stride * layout.experts_per_layer as u64)
                .max()
                .filter(|&bytes| bytes > 0)
                .ok_or("packed expert layout is missing")?,
        };
        let state = runner.real_qwen4.as_mut().expect("Qwen4 prefill");
        let lengths = row_buffers(state)
            .map(|buffer| buffer.length() / MAX_PREFILL_BATCH as u64 * rows as u64);
        // Charge both the bank and temporary rows before touching pages. The
        // ordinary decode cache stays in the measured resident footprint.
        let additional = bank_bytes
            + lengths
                .iter()
                .map(|n| n.next_multiple_of(16384))
                .sum::<u64>();
        let resident = gpu::process_footprint().ok_or("could not read process footprint")?;
        let physical = gpu::physical_memory();
        let available = model_io::LoadGuard::default()
            .available(physical, resident)
            .min(state.prefill_load_guard.available(physical, resident))
            .min(state.prefill_temporary_limit.unwrap_or(u64::MAX));
        if gpu::memory_pressure_raw() != 1 || additional > available {
            return Err(format!("{additional} prefill bytes exceed available headroom or host is under memory pressure"));
        }
        let buffers = lengths.map(|len| runner.context.new_output_buffer(len));
        if buffers
            .iter()
            .zip(lengths)
            .any(|(buf, len)| buf.length() != len)
        {
            return Err("Metal refused wide prefill scratch".into());
        }
        let bank = if bank_bytes > 0 {
            // Unmap the bank on drop so allocator retention cannot charge
            // the released pages to the subsequent decode.
            let storage = memmap2::MmapOptions::new()
                .len(bank_bytes as usize)
                .map_anon()
                .map_err(|error| error.to_string())?;
            let buffer = gpu::wrap_page_aligned_no_copy(
                runner.context.device(),
                storage.as_ptr(),
                storage.len(),
            )
            .map_err(|error| error.to_string())?;
            Some(PrefillBank { buffer, storage })
        } else {
            None
        };
        Ok(PrefillResources {
            buffers,
            bank,
            _admission: admission,
        })
    }
}

pub(super) fn row_buffers(state: &mut RealQwen4State) -> [&mut gpu::MetalBuffer; 5] {
    [
        &mut state.wide_x,
        &mut state.hc_inject,
        &mut state.router_logits_f32,
        &mut state.moe_x,
        &mut state.ngram_emb,
    ]
}

pub(super) struct PrefillBank {
    // Drop the no-copy Metal view before freeing the host allocation.
    buffer: gpu::MetalBuffer,
    storage: memmap2::MmapMut,
}

pub(super) struct LoadedLayer<'a> {
    pub(super) buffer: &'a gpu::MetalBuffer,
    pub(super) stride: u64,
}

impl PrefillBank {
    pub(super) fn load_layer(
        &mut self,
        state: &RealQwen4State,
        streamer: &mut streaming::PreadExpertStreamer,
        phases: &mut PhaseCounters,
        rows: usize,
        top_k: usize,
    ) -> Result<LoadedLayer<'_>, RealForwardError> {
        let t_router = Instant::now();
        let num_experts = streamer.layout().experts_per_layer;
        let logits = gpu::read_f32_buffer_at(&state.router_logits_f32, 0, rows * num_experts);
        let mut selected = vec![false; num_experts];
        for row in logits.chunks_exact(num_experts) {
            let (ids, _) = router_topk_gemma4(row, top_k, &state.per_expert_ones);
            for id in ids {
                selected[id] = true;
            }
        }
        let experts: Vec<_> = selected
            .iter()
            .enumerate()
            .filter_map(|(id, &needed)| needed.then_some(id))
            .collect();
        let requests = (rows * top_k) as u64;
        phases.expert_requests += requests;
        phases.expert_hits += requests - experts.len() as u64;
        phases.router_nanos += t_router.elapsed().as_nanos() as u64;

        let t_io = Instant::now();
        streamer
            .read_experts_into_bank(&experts, &mut self.storage)
            .map_err(|error| {
                RealForwardError::Unsupported(format!("prefill expert bank: {error}"))
            })?;
        phases.expert_io_nanos += t_io.elapsed().as_nanos() as u64;
        Ok(LoadedLayer {
            buffer: &self.buffer,
            stride: streamer.layout().expert_stride,
        })
    }
}
