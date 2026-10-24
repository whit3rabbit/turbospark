//! MEASUREMENT ONLY, NOT FOR COMMIT: the qwen38 Neural Engine split
//! prototype (`docs/MOE_PRIOR_ART.md`, "Phase 1, if taken"). Compiled only
//! with the `ane-split-probe` feature and taken only when
//! `TURBOSPARK_PROBE_ANE_SPLIT` asks for it. Steel stays the default.
//!
//! It splits each trunk layer's MLP gate and up projections (17408 x 5120
//! on Qwen3.8-27B) by output rows inside the dense chunked prefill. One
//! Core ML prediction runs rows `[0, MA)` of both on the ANE while Steel
//! computes rows `[MA, inter)` straight into `ffn_gate` and `ffn_up`.
//!
//! The default arm encodes a micro-batch as ONE command buffer. Core ML is
//! driven from the host, so the buffer has to break at every split. Per
//! layer L:
//!
//! ```text
//! seg(L)   [GPU waits for E >= base + L] convert(L-1), FFN tail(L-1),
//!          layer L up to moe_x, prep(L): weights to F16, scaled activations
//! rows(L)  Steel rows [MA, inter) of gate and up
//! host     wait seg(L) -> Core ML predict(L) -> signal E = base + L + 1
//! ```
//!
//! Every buffer is encoded before any is committed, the default arm's
//! encode-then-run shape, so overlapping the encode with GPU work does not
//! flatter the split. `break` keeps the same buffers, event waits and host
//! loop, but Steel takes every row and no prediction runs: the cost of the
//! breaks alone.
//!
//! Env (all read per micro-batch, echoed once per value):
//! - `TURBOSPARK_PROBE_ANE_SPLIT`: `ane` or `break`. Anything else is off.
//! - `TURBOSPARK_PROBE_ANE_SPLIT_PKGS`: directory of
//!   `gateup_ma{MA}_k{hidden}_n{N}.mlpackage` (`gen_gateup.py`).
//! - `TURBOSPARK_PROBE_ANE_SPLIT_ROWS`: MA, default 8192. Either one value
//!   for every program or `N:MA` pairs (`2048:7680,1024:5632`), so each
//!   token count can take its own share. Unlisted programs take 8192.
//! - `TURBOSPARK_PROBE_ANE_SPLIT_TOKENS`: the programs' token counts,
//!   default `512,1024,2048`. A micro-batch of m rows takes the smallest
//!   N >= m, its activations padded. Below the smallest, or above the
//!   largest, the micro-batch runs the default arm.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use model_io::{ArchConfig, ResidentIndex};
use turbospark_coreml::{ComputeUnits, CoreMlModel, LoadOptions, SharedF16Plane};

use super::batched_layers::encode_dense_ffn_tail_batched;
use super::prefill_layers::{
    encode_qwen_dense_layer_batched_post_ffn, encode_qwen_dense_layer_batched_pre_ffn,
};
use super::{layer_tensor, RealQwenState, TRUNK_PREFIX};
use crate::real_forward_types::{DecodeScratch, RealForwardError};
use crate::real_forward_utils::{norm_view, resident_matrix};

use gpu::ane_split as ane;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SplitMode {
    Ane,
    Break,
}

fn echo_once(what: &str, value: &str) {
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let line = format!("{what} = {value}");
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    if !seen.contains(&line) {
        eprintln!("[probe] ane split {line}");
        seen.push(line);
    }
}

pub(crate) fn mode() -> Option<SplitMode> {
    let v = std::env::var("TURBOSPARK_PROBE_ANE_SPLIT").unwrap_or_default();
    let mode = match v.as_str() {
        "ane" => Some(SplitMode::Ane),
        "break" => Some(SplitMode::Break),
        _ => None,
    };
    echo_once("mode", &format!("{mode:?}"));
    mode
}

fn ane_rows(tokens: usize) -> usize {
    let spec = std::env::var("TURBOSPARK_PROBE_ANE_SPLIT_ROWS").unwrap_or_default();
    let rows = match spec.trim().parse::<usize>() {
        Ok(all) => all,
        Err(_) => spec
            .split(',')
            .filter_map(|pair| pair.split_once(':'))
            .filter_map(|(n, ma)| Some((n.trim().parse::<usize>().ok()?, ma.trim().parse().ok()?)))
            .find(|&(n, _)| n == tokens)
            .map_or(8192, |(_, ma)| ma),
    };
    echo_once(&format!("rows for n={tokens}"), &rows.to_string());
    rows
}

fn token_slots() -> Vec<usize> {
    let mut slots: Vec<usize> = std::env::var("TURBOSPARK_PROBE_ANE_SPLIT_TOKENS")
        .unwrap_or_else(|_| "512,1024,2048".into())
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .filter(|&n| n > 0)
        .collect();
    slots.sort_unstable();
    slots.dedup();
    echo_once("token slots", &format!("{slots:?}"));
    slots
}

/// The program token count a micro-batch of `m` rows runs at, if any.
pub(crate) fn slot_tokens(m: usize) -> Option<usize> {
    token_slots().into_iter().find(|&n| n >= m && m >= 512)
}

fn pkg_dir() -> Result<PathBuf, RealForwardError> {
    let dir = std::env::var_os("TURBOSPARK_PROBE_ANE_SPLIT_PKGS").ok_or_else(|| {
        RealForwardError::Unsupported("TURBOSPARK_PROBE_ANE_SPLIT_PKGS is not set".into())
    })?;
    echo_once("pkgs", &format!("{dir:?}"));
    Ok(PathBuf::from(dir))
}

fn coreml_err(e: turbospark_coreml::CoreMlError) -> RealForwardError {
    RealForwardError::Unsupported(format!("ane split Core ML: {e}"))
}

/// An IOSurface plane and the no-copy Metal buffer over it. The buffer is
/// declared first so it drops before the memory it wraps.
pub(crate) struct Plane {
    buffer: gpu::MetalBuffer,
    plane: SharedF16Plane,
}

impl Plane {
    fn new(
        context: &gpu::MetalContext,
        rows: usize,
        cols: usize,
    ) -> Result<Self, RealForwardError> {
        let plane = SharedF16Plane::new(rows, cols).map_err(coreml_err)?;
        let buffer =
            gpu::wrap_page_aligned_no_copy(context.device(), plane.base_ptr(), plane.alloc_size())
                .map_err(RealForwardError::Gpu)?;
        Ok(Self { buffer, plane })
    }
}

/// One compiled program and the planes it uses: activations and outputs
/// carry the token count in their shape, and the weight planes this
/// program's ANE row count.
struct Slot {
    tokens: usize,
    ma: usize,
    wg: Plane,
    wu: Plane,
    a: Plane,
    og: Plane,
    ou: Plane,
    model: CoreMlModel,
}

/// The ANE half: per-token inverse scales and one slot per program.
struct GateUp {
    inv: gpu::MetalBuffer,
    slots: Vec<Slot>,
}

pub(crate) struct AneSplitState {
    event: ane::HostEvent,
    gate_up: Option<GateUp>,
}

/// Allocates the split state the first time `mode` needs it, loading every
/// program (and timing that) for [`SplitMode::Ane`].
pub(crate) fn ensure(
    state: &mut Option<AneSplitState>,
    context: &mut gpu::MetalContext,
    arch: &ArchConfig,
    mode: SplitMode,
) -> Result<(), RealForwardError> {
    if state.is_none() {
        *state = Some(AneSplitState {
            event: ane::HostEvent::new(context),
            gate_up: None,
        });
    }
    let st = state.as_mut().expect("set above");
    if mode != SplitMode::Ane || st.gate_up.is_some() {
        return Ok(());
    }
    let (hidden, inter) = (arch.hidden_size as usize, arch.intermediate_size as usize);
    let dir = pkg_dir()?;
    let cache = dir.join("compiled");
    let started = Instant::now();
    let max_tokens = token_slots().last().copied().unwrap_or(0);
    let inv = context.new_output_buffer((max_tokens.max(1) * 4) as u64);
    let mut slots = Vec::new();
    for tokens in token_slots() {
        let ma = ane_rows(tokens);
        if ma == 0 || ma >= inter || ma % 32 != 0 || (inter - ma) % 32 != 0 {
            return Err(RealForwardError::Unsupported(format!(
                "ane split rows {ma} must be a multiple of 32 inside (0, {inter})"
            )));
        }
        let pkg = dir.join(format!("gateup_ma{ma}_k{hidden}_n{tokens}.mlpackage"));
        let t0 = Instant::now();
        let model = CoreMlModel::open(
            &pkg,
            &cache,
            &LoadOptions::new(ComputeUnits::CpuAndNeuralEngine),
        )
        .map_err(coreml_err)?;
        let load_s = t0.elapsed().as_secs_f64();
        let slot = Slot {
            tokens,
            ma,
            wg: Plane::new(context, ma, hidden)?,
            wu: Plane::new(context, ma, hidden)?,
            a: Plane::new(context, tokens, hidden)?,
            og: Plane::new(context, ma, tokens)?,
            ou: Plane::new(context, ma, tokens)?,
            model,
        };
        // The first prediction pays the Neural Engine program load.
        let t1 = Instant::now();
        predict(&slot)?;
        let first_s = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        predict(&slot)?;
        eprintln!(
            "[probe] ane split slot n={tokens} ma={ma}: open {load_s:.3} s, first predict \
             {first_s:.3} s, warm predict {:.2} ms, strides a={} og={} w={}",
            t2.elapsed().as_secs_f64() * 1e3,
            slot.a.plane.stride(),
            slot.og.plane.stride(),
            slot.wg.plane.stride()
        );
        slots.push(slot);
    }
    eprintln!(
        "[probe] ane split ready in {:.3} s ({} slots)",
        started.elapsed().as_secs_f64(),
        slots.len()
    );
    st.gate_up = Some(GateUp { inv, slots });
    Ok(())
}

fn predict(slot: &Slot) -> Result<(), RealForwardError> {
    slot.model
        .predict_shared(
            &[
                ("a", &slot.a.plane),
                ("wg", &slot.wg.plane),
                ("wu", &slot.wu.plane),
            ],
            &[("og", &slot.og.plane), ("ou", &slot.ou.plane)],
        )
        .map_err(coreml_err)
}

/// What one split micro-batch cost, in seconds.
pub(crate) struct SplitTimes {
    /// GPU busy time: each buffer's GPU interval, with the stretch an
    /// event-gated buffer spent waiting for the host left out.
    pub(crate) gpu_busy: f64,
    /// First commit to the last buffer's completion, on the host.
    pub(crate) wall: f64,
}

/// Runs one dense micro-batch of `m` rows with the split. `first` already
/// holds the embedding lookups; `head` encodes (or skips) the LM head into
/// the last segment. Returns once every buffer has completed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_micro_batch(
    context: &mut gpu::MetalContext,
    first: gpu::PassEncoder,
    weights: &gpu::ResidentGpuWeights,
    index: &ResidentIndex,
    scratch: &DecodeScratch,
    kv: &mut gpu::KvCacheManager,
    qwen: &RealQwenState,
    arch: &ArchConfig,
    resid_capture: Option<&crate::resid_capture::ResidCapture>,
    steering: Option<&crate::steering::SteeringState>,
    mode: SplitMode,
    use_silu: bool,
    start_position: usize,
    m: usize,
    head: impl FnOnce(&mut gpu::MetalContext, &gpu::PassEncoder) -> Result<(), RealForwardError>,
) -> Result<SplitTimes, RealForwardError> {
    let (hidden, inter) = (arch.hidden_size as usize, arch.intermediate_size as usize);
    let layers = arch.num_layers as usize;
    let state = qwen
        .ane_split
        .as_ref()
        .ok_or_else(|| RealForwardError::Unsupported("ane split state missing".into()))?;
    let batched = qwen.batched_prefill();
    let (gate_up, slot) = match mode {
        SplitMode::Ane => {
            let g = state
                .gate_up
                .as_ref()
                .ok_or_else(|| RealForwardError::Unsupported("ane split not loaded".into()))?;
            let tokens = slot_tokens(m).ok_or_else(|| {
                RealForwardError::Unsupported(format!("no ane split program for {m} rows"))
            })?;
            let slot = g.slots.iter().find(|s| s.tokens == tokens).ok_or_else(|| {
                RealForwardError::Unsupported(format!("ane split slot {tokens} not loaded"))
            })?;
            (Some(g), Some(slot))
        }
        SplitMode::Break => (None, None),
    };
    // Every buffer is created before any is committed; past the queue's
    // capacity, creating one more blocks forever.
    if 2 * layers + 1 > ane::MAX_COMMAND_BUFFERS {
        return Err(RealForwardError::Unsupported(format!(
            "ane split needs {} command buffers in flight, the queue allows {}",
            2 * layers + 1,
            ane::MAX_COMMAND_BUFFERS
        )));
    }
    let row0 = slot.map_or(0, |s| s.ma);
    let base = state.event.value();
    let gpu_err = RealForwardError::Gpu;

    let t_encode = Instant::now();
    let mut segs: Vec<gpu::PassEncoder> = Vec::with_capacity(layers + 1);
    let mut rows: Vec<gpu::PassEncoder> = Vec::with_capacity(layers);
    let mut pass = first;
    for layer in 0..layers {
        let input_norm = norm_view(
            weights,
            index,
            &layer_tensor(layer, "input_layernorm.weight"),
            hidden,
        )?;
        let post_attn_norm = norm_view(
            weights,
            index,
            &layer_tensor(layer, "post_attention_layernorm.weight"),
            hidden,
        )?;
        encode_qwen_dense_layer_batched_pre_ffn(
            context,
            &pass,
            weights,
            index,
            scratch,
            kv,
            qwen,
            arch,
            input_norm,
            post_attn_norm,
            layer,
            hidden,
            start_position,
            m,
        )?;
        let gate = resident_matrix(
            weights,
            index,
            &layer_tensor(layer, "mlp.gate_proj.weight"),
            inter,
            hidden,
        )?;
        let up = resident_matrix(
            weights,
            index,
            &layer_tensor(layer, "mlp.up_proj.weight"),
            inter,
            hidden,
        )?;
        if let (Some(g), Some(s)) = (gate_up, slot) {
            ane::encode_ane_dequant_rows(
                context,
                &pass,
                &gate,
                s.ma,
                (&s.wg.buffer, 0),
                s.wg.plane.stride(),
            )
            .map_err(gpu_err)?;
            ane::encode_ane_dequant_rows(
                context,
                &pass,
                &up,
                s.ma,
                (&s.wu.buffer, 0),
                s.wu.plane.stride(),
            )
            .map_err(gpu_err)?;
            ane::encode_ane_scale_rows(
                context,
                &pass,
                (&batched.moe_x, 0),
                hidden,
                m,
                (&s.a.buffer, 0),
                s.a.plane.stride(),
                &g.inv,
            )
            .map_err(gpu_err)?;
        }
        let steel = context.begin_pass_labeled("qwen split steel rows");
        for (w, out) in [(&gate, &batched.ffn_gate), (&up, &batched.ffn_up)] {
            ane::encode_dequant_int4_gemm_steel_rows(
                context,
                &steel,
                w,
                row0,
                (&batched.moe_x, 0),
                (out, 0),
                inter,
                m,
            )
            .map_err(gpu_err)?;
        }
        let next = ane::begin_pass_after(
            context,
            "qwen split segment",
            &state.event,
            base + layer as u64 + 1,
        );
        if let (Some(g), Some(s)) = (gate_up, slot) {
            for (o, out) in [(&s.og, &batched.ffn_gate), (&s.ou, &batched.ffn_up)] {
                ane::encode_ane_convert_out(
                    context,
                    &next,
                    (&o.buffer, 0),
                    s.ma,
                    m,
                    o.plane.stride(),
                    &g.inv,
                    (out, 0),
                    inter,
                )
                .map_err(gpu_err)?;
            }
        }
        encode_dense_ffn_tail_batched(
            context,
            &next,
            weights,
            index,
            scratch,
            batched,
            TRUNK_PREFIX,
            layer,
            hidden,
            inter,
            use_silu,
            m,
        )?;
        encode_qwen_dense_layer_batched_post_ffn(
            context,
            &next,
            scratch,
            resid_capture,
            steering,
            layer,
            hidden,
            m,
        )?;
        segs.push(std::mem::replace(&mut pass, next));
        rows.push(steel);
    }
    head(context, &pass)?;
    segs.push(pass);
    let encode_s = t_encode.elapsed().as_secs_f64();

    // Commit in queue order: seg(0), rows(0), seg(1), rows(1), ..., seg(L).
    let t_commit = ane::host_seconds();
    let mut seg_done = Vec::with_capacity(layers + 1);
    let mut rows_done = Vec::with_capacity(layers);
    let mut rows_iter = rows.into_iter();
    for seg in segs {
        seg_done.push(seg.commit());
        if let Some(r) = rows_iter.next() {
            rows_done.push(r.commit());
        }
    }
    let seg_done_committed = ane::host_seconds();

    // The host half. A failed prediction still signals every value so the
    // GPU drains before the error is returned.
    let mut ready = vec![0.0f64; layers];
    let mut predicted = vec![0.0f64; layers];
    let mut signaled = vec![0.0f64; layers];
    let mut failure = None;
    for layer in 0..layers {
        seg_done[layer].wait_ref();
        ready[layer] = ane::host_seconds();
        if let (Some(s), None) = (slot, failure.as_ref()) {
            if let Err(e) = predict(s) {
                failure = Some(e);
            }
        }
        predicted[layer] = ane::host_seconds();
        state.event.signal(base + layer as u64 + 1);
        signaled[layer] = ane::host_seconds();
    }
    seg_done[layers].wait_ref();
    let wall = ane::host_seconds() - t_commit;
    if let Some(e) = failure {
        return Err(e);
    }

    let seg_t: Vec<(f64, f64)> = seg_done.iter().map(|c| c.wait_gpu_start_end()).collect();
    let rows_t: Vec<(f64, f64)> = rows_done.iter().map(|c| c.wait_gpu_start_end()).collect();
    if std::env::var_os("TURBOSPARK_PROBE_ANE_SPLIT_TIMELINE").is_some() {
        // Per layer, ms from the first commit: seg GPU start/end, host saw
        // it, prediction end, signal, Steel rows GPU start/end.
        let at = |t: f64| (t - t_commit) * 1e3;
        eprintln!("[probe-tl] commit_all_ms={:.3}", at(seg_done_committed));
        for layer in 0..layers {
            eprintln!(
                "[probe-tl] L={layer} seg={:.2}..{:.2} ready={:.2} pred={:.2} sig={:.2} \
                 rows={:.2}..{:.2}",
                at(seg_t[layer].0),
                at(seg_t[layer].1),
                at(ready[layer]),
                at(predicted[layer]),
                at(signaled[layer]),
                at(rows_t[layer].0),
                at(rows_t[layer].1),
            );
        }
        eprintln!(
            "[probe-tl] final seg={:.2}..{:.2}",
            at(seg_t[layers].0),
            at(seg_t[layers].1)
        );
    }
    let mut gpu_raw = 0.0;
    let mut gpu_busy = 0.0;
    for (i, &(s, e)) in seg_t.iter().enumerate() {
        gpu_raw += e - s;
        // seg(i) for i >= 1 waits on the signal sent after layer i - 1.
        let from = if i == 0 { s } else { s.max(signaled[i - 1]) };
        gpu_busy += (e - from).max(0.0);
    }
    let mut steel_rows = 0.0;
    let (mut ane_s, mut wake, mut gpu_waits_ane, mut resume, mut ane_waits_gpu) =
        (0.0, 0.0, 0.0, 0.0, 0.0);
    for layer in 0..layers {
        let (rs, re) = rows_t[layer];
        gpu_raw += re - rs;
        gpu_busy += re - rs;
        steel_rows += re - rs;
        ane_s += predicted[layer] - ready[layer];
        // GPU end of seg(L) to the host seeing it.
        wake += ready[layer] - seg_t[layer].1;
        // The Steel rows finished before the host's signal: GPU idle.
        gpu_waits_ane += (signaled[layer] - re).max(0.0);
        // The prediction finished before the Steel rows: ANE idle.
        ane_waits_gpu += (re - predicted[layer]).max(0.0);
        // Signal (or Steel end, if later) to the next segment's first work.
        resume += seg_t[layer + 1].0 - signaled[layer].max(re);
    }
    eprintln!(
        "[probe-split] mode={mode:?} start={start_position} m={m} slot={} ma={row0} \
         encode_ms={:.3} wall_ms={:.3} gpu_busy_ms={:.3} gpu_raw_ms={:.3} \
         steel_rows_ms={:.3} ane_ms={:.3} wake_ms={:.3} gpu_waits_ane_ms={:.3} \
         ane_waits_gpu_ms={:.3} resume_ms={:.3}",
        slot.map_or(0, |s| s.tokens),
        encode_s * 1e3,
        wall * 1e3,
        gpu_busy * 1e3,
        gpu_raw * 1e3,
        steel_rows * 1e3,
        ane_s * 1e3,
        wake * 1e3,
        gpu_waits_ane * 1e3,
        ane_waits_gpu * 1e3,
        resume * 1e3,
    );
    eprintln!(
        "[probe-mb] start={start_position} m={m} batched=true encode_ms={:.3} wait_ms={:.3} \
         gpu_ms={:.3}",
        encode_s * 1e3,
        wall * 1e3,
        gpu_busy * 1e3
    );
    Ok(SplitTimes { gpu_busy, wall })
}

#[cfg(test)]
mod tests {
    //! The split's arithmetic on synthetic INT4 weights at the real gate/up
    //! shape, against one-shot Steel. Needs the Neural Engine and the
    //! generated programs:
    //!
    //!   TURBOSPARK_PROBE_ANE_SPLIT_PKGS=~/models/ane-split/pkgs cargo test \
    //!     -p turbospark-runtime --features ane-split-probe --release \
    //!     ane_split -- --ignored --nocapture
    use super::*;
    use half::{bf16, f16};

    const K: usize = 5120;
    const ROWS: usize = 17408;

    /// The synthetic recipe of `qwen38_int4_prefill_gemm_probe.rs`, with
    /// per-group scales that vary so a group-index bug cannot hide.
    fn blob(seed: u64) -> (Vec<u8>, u64, u64) {
        let mut state = seed | 1;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let groups = ROWS * K / 64;
        let mut out: Vec<u8> = (0..ROWS * K / 2).map(|_| next() as u8).collect();
        let scales = out.len() as u64;
        for g in 0..groups {
            let s = 0.004 + 0.002 * (g % 7) as f32;
            out.extend_from_slice(&bf16::from_f32(s).to_bits().to_le_bytes());
        }
        let biases = out.len() as u64;
        for g in 0..groups {
            let b = -0.03 - 0.005 * (g % 5) as f32;
            out.extend_from_slice(&bf16::from_f32(b).to_bits().to_le_bytes());
        }
        (out, scales, biases)
    }

    #[test]
    #[ignore = "needs the Neural Engine and TURBOSPARK_PROBE_ANE_SPLIT_PKGS"]
    fn ane_split_matches_one_shot_steel() {
        let Some(dir) = std::env::var_os("TURBOSPARK_PROBE_ANE_SPLIT_PKGS") else {
            eprintln!("TURBOSPARK_PROBE_ANE_SPLIT_PKGS is not set; skipping");
            return;
        };
        let dir = PathBuf::from(dir);
        let (ma, n, m) = (8192usize, 512usize, 500usize);
        let mut context = gpu::MetalContext::new().expect("Metal");
        let (data, so, bo) = blob(0xC0FFEE);
        let wbuf = context.new_buffer_with_data(&data);
        let w = gpu::Int4ResidentMatrix {
            buffer: &wbuf,
            weights_offset: 0,
            scales_offset: so,
            biases_offset: bo,
            rows: ROWS,
            cols: K,
        };
        // Rows of very different magnitudes, so the per-token scale matters:
        // without it the 1e-3 rows sit in F16 subnormals on the ANE.
        let mags = [1.0f32, 1e-3, 30.0, 0.05, 4.0];
        let x: Vec<u16> = (0..m * K)
            .map(|i| {
                let (t, k) = (i / K, i % K);
                let v = (((t * 7 + k * 13) % 29) as f32 - 14.0) / 64.0 * mags[t % mags.len()];
                f16::from_f32(v).to_bits()
            })
            .collect();
        let xbuf = context.new_buffer_with_data(&x);
        let y_ref = context.new_output_buffer((m * ROWS * 2) as u64);
        let y = context.new_output_buffer((m * ROWS * 2) as u64);

        let pass = context.begin_pass();
        gpu::encode_dequant_int4_gemm_steel(
            &mut context,
            &pass,
            &w,
            (&xbuf, 0),
            (&y_ref, 0),
            m,
            gpu::SteelTile::M32K32N32,
        )
        .unwrap();
        pass.commit_and_wait();

        let pkg = dir.join(format!("gateup_ma{ma}_k{K}_n{n}.mlpackage"));
        let model = CoreMlModel::open(
            &pkg,
            &dir.join("compiled"),
            &LoadOptions::new(ComputeUnits::CpuAndNeuralEngine),
        )
        .expect("program");
        let slot = Slot {
            tokens: n,
            ma,
            wg: Plane::new(&context, ma, K).unwrap(),
            wu: Plane::new(&context, ma, K).unwrap(),
            a: Plane::new(&context, n, K).unwrap(),
            og: Plane::new(&context, ma, n).unwrap(),
            ou: Plane::new(&context, ma, n).unwrap(),
            model,
        };
        let inv = context.new_output_buffer((n * 4) as u64);

        // Gate and up are the same matrix here, so both outputs are checked
        // against one reference.
        let pass = context.begin_pass();
        for p in [&slot.wg, &slot.wu] {
            ane::encode_ane_dequant_rows(
                &mut context,
                &pass,
                &w,
                ma,
                (&p.buffer, 0),
                p.plane.stride(),
            )
            .unwrap();
        }
        ane::encode_ane_scale_rows(
            &mut context,
            &pass,
            (&xbuf, 0),
            K,
            m,
            (&slot.a.buffer, 0),
            slot.a.plane.stride(),
            &inv,
        )
        .unwrap();
        ane::encode_dequant_int4_gemm_steel_rows(
            &mut context,
            &pass,
            &w,
            ma,
            (&xbuf, 0),
            (&y, 0),
            ROWS,
            m,
        )
        .unwrap();
        pass.commit_and_wait();
        predict(&slot).expect("predict");
        let y_up = context.new_output_buffer((m * ROWS * 2) as u64);
        let pass = context.begin_pass();
        for (o, out) in [(&slot.og, &y), (&slot.ou, &y_up)] {
            ane::encode_ane_convert_out(
                &mut context,
                &pass,
                (&o.buffer, 0),
                ma,
                m,
                o.plane.stride(),
                &inv,
                (out, 0),
                ROWS,
            )
            .unwrap();
        }
        pass.commit_and_wait();

        let r = gpu::read_buffer_f16(&y_ref, 0, m * ROWS);
        let s = gpu::read_buffer_f16(&y, 0, m * ROWS);
        let u = gpu::read_buffer_f16(&y_up, 0, m * ROWS);
        // Steel's rows land at their offset, bit for bit.
        let mut steel_diff = 0usize;
        for t in 0..m {
            for row in ma..ROWS {
                steel_diff +=
                    usize::from(r[t * ROWS + row].to_bits() != s[t * ROWS + row].to_bits());
            }
        }
        assert_eq!(steel_diff, 0, "Steel rows differ from one-shot Steel");
        // The ANE rows: worst per-token max |d| / max |ref| (Phase 0's metric).
        let mut worst = 0.0f64;
        for (label, got) in [("gate", &s), ("up", &u)] {
            let mut worst_here = 0.0f64;
            for t in 0..m {
                let (mut mx, mut md) = (0.0f64, 0.0f64);
                for row in 0..ma {
                    let (a, b) = (r[t * ROWS + row].to_f64(), got[t * ROWS + row].to_f64());
                    assert!(b.is_finite(), "{label}: non-finite at token {t} row {row}");
                    mx = mx.max(a.abs());
                    md = md.max((a - b).abs());
                }
                worst_here = worst_here.max(if mx > 0.0 { md / mx } else { md });
            }
            eprintln!("{label}: ANE rows worst normalized error {worst_here:.5}");
            worst = worst.max(worst_here);
        }
        assert!(worst < 5e-3, "ANE rows off by {worst}");
    }
}

#[cfg(test)]
mod cost {
    //! GPU seconds of the split's conversions at the 2048-row micro-batch,
    //! per layer (gate and up), median of 9 after a warmup. Diagnostic.
    use super::*;

    #[test]
    #[ignore = "Metal timing probe; run explicitly"]
    fn ane_split_conversion_gpu_cost() {
        let (k, rows, ma, n) = (5120usize, 17408usize, 8192usize, 2048usize);
        let mut context = gpu::MetalContext::new().expect("Metal");
        // Two weight copies so neither stays cache-resident between samples.
        let blobs: Vec<_> = (0..2)
            .map(|i| {
                let bytes = rows * k / 2 + 2 * (rows * k / 64) * 2;
                let data: Vec<u8> = (0..bytes).map(|j| (j * 31 + i * 7) as u8).collect();
                context.new_buffer_with_data(&data)
            })
            .collect();
        let x = context.new_buffer_with_data(&vec![0x3c00u16; n * k]);
        let wg = Plane::new(&context, ma, k).unwrap();
        let wu = Plane::new(&context, ma, k).unwrap();
        let a = Plane::new(&context, n, k).unwrap();
        let og = Plane::new(&context, ma, n).unwrap();
        let inv = context.new_output_buffer((n * 4) as u64);
        let y = context.new_output_buffer((n * rows * 2) as u64);
        let (mut prep, mut conv) = (Vec::new(), Vec::new());
        for rep in 0..10 {
            let wbuf = &blobs[rep % 2];
            let w = gpu::Int4ResidentMatrix {
                buffer: wbuf,
                weights_offset: 0,
                scales_offset: (rows * k / 2) as u64,
                biases_offset: (rows * k / 2 + rows * k / 64 * 2) as u64,
                rows,
                cols: k,
            };
            let pass = context.begin_pass();
            for p in [&wg, &wu] {
                ane::encode_ane_dequant_rows(&mut context, &pass, &w, ma, (&p.buffer, 0), k)
                    .unwrap();
            }
            ane::encode_ane_scale_rows(
                &mut context,
                &pass,
                (&x, 0),
                k,
                n,
                (&a.buffer, 0),
                a.plane.stride(),
                &inv,
            )
            .unwrap();
            let t_prep = pass.commit_and_wait_with_gpu_time();
            let pass = context.begin_pass();
            for _ in 0..2 {
                ane::encode_ane_convert_out(
                    &mut context,
                    &pass,
                    (&og.buffer, 0),
                    ma,
                    n,
                    og.plane.stride(),
                    &inv,
                    (&y, 0),
                    rows,
                )
                .unwrap();
            }
            let t_conv = pass.commit_and_wait_with_gpu_time();
            if rep > 0 {
                prep.push(t_prep * 1e3);
                conv.push(t_conv * 1e3);
            }
        }
        prep.sort_by(f64::total_cmp);
        conv.sort_by(f64::total_cmp);
        eprintln!(
            "per layer at {n} rows: prep (2 dequant + scale) {:.3} ms, convert x2 {:.3} ms; \
             x64 layers = {:.1} ms",
            prep[4],
            conv[4],
            (prep[4] + conv[4]) * 64.0
        );
    }
}
