//! A gated-DeltaNet recurrence run for many steps through two kernel routes,
//! with the state hashed after every step (evaluator item LAB-34). One-step
//! parity (`gdn_tests.rs`) can pass while a recurrent state compounds a
//! difference; this runs the state forward.
//!
//! Both arms start from one prefilled state and conv tail, cloned. Each
//! step runs every layer's conv, q/k norm and delta rule through each arm's
//! route into the arm's own buffers, and then compares the recurrent state,
//! the conv tail and the outputs bit for bit. The whole trajectory runs
//! twice, A encoded before B and B before A, so a dependence on dispatch
//! order shows as two different digests.
//!
//! Before the real pairs, the same loop runs arm A against itself with one
//! ULP flipped in one layer's state at one step, and must report exactly
//! that layer and step. A comparison that cannot fail is no result, so the
//! bench errors out if the injection is missed.
//!
//! Inputs are synthetic and fixed (a pool of distinct rows walked in order),
//! and layers do not feed each other: this tests each route's arithmetic on
//! the state, not a model's forward. Outputs are hashes; nothing is timed.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_copy_rows_fp16_strided, encode_gdn_conv_decode,
    encode_gdn_conv_prefill, encode_gdn_conv_tail_update, encode_gdn_delta_decode,
    encode_gdn_delta_prefill, encode_gdn_delta_prefill_lazy, encode_gdn_qk_norm,
    encode_gdn_state_fold, read_buffer_bytes, write_buffer_bytes, GdnShape, MetalBuffer,
    MetalContext,
};
use half::{bf16, f16};
use serde_json::json;

use super::gdn_delta::layer_count;
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{ArmCurve, Axis, BenchResult, Point};

pub const NAME: &str = "gdn_trajectory";
const DEFAULT_STEPS: usize = 256;
const DEFAULT_PREFILL: usize = 128;
/// The step the self-test injects at, when the trajectory is that long.
const FAULT_STEP: usize = 100;
/// Distinct input rows the trajectory walks.
const POOL: usize = 256;
/// Rows per eager prefill dispatch while building the shared start state.
const SEED_CHUNK: usize = 64;
/// The widest `--chunk`: a chunk is read contiguously from the pool.
pub const MAX_CHUNK: usize = 64;

/// A route through the delta rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The one-row decode kernels, one row at a time.
    Decode,
    /// The eager multi-row prefill kernel over the chunk.
    Prefill,
    /// The lazy kernel (records a tape) and then the state fold, as a
    /// verify round that is accepted whole.
    LazyFold,
}

impl Route {
    pub fn label(self) -> &'static str {
        match self {
            Route::Decode => "decode",
            Route::Prefill => "prefill_eager",
            Route::LazyFold => "lazy_fold",
        }
    }
}

const PAIRS: [(&str, Route, Route); 2] = [
    ("decode_vs_prefill", Route::Decode, Route::Prefill),
    ("decode_vs_lazy_fold", Route::Decode, Route::LazyFold),
];

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub shape: GdnShape,
    pub layers: usize,
    pub steps: usize,
    pub prefill: usize,
    /// Rows a chunked route takes at once; state is compared per chunk.
    pub chunk: usize,
}

/// One flipped ULP in arm B's state of `layer`, after the chunk holding
/// `step` ran and before it is compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub layer: usize,
    pub step: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    pub layer: usize,
    /// 0-based decode step. For `state` and `conv_tail` it is the last step
    /// of the chunk they were compared after.
    pub step: usize,
    /// `state`, `conv_tail` or `y`.
    pub what: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// The earliest step that differs (the lowest layer at that step).
    pub first: Option<Mismatch>,
    /// Per arm, per layer: every step's state, tail and output hashes folded.
    pub digests: [Vec<u64>; 2],
}

/// A word-at-a-time 64-bit mix: the states are megabytes per layer and are
/// hashed every step, which the byte-wise FNV would make the whole run.
fn hash_words(mut h: u64, bytes: &[u8]) -> u64 {
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().expect("8 bytes"));
        h = (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    }
    for &b in chunks.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Deterministic values in `[-scale, scale]`.
fn stream(seed: u64, n: usize, scale: f32) -> impl Iterator<Item = f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n).map(move |_| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (((s % 2000) as f32 / 1000.0) - 1.0) * scale
    })
}

fn halves(context: &MetalContext, seed: u64, n: usize, scale: f32) -> MetalBuffer {
    let bytes: Vec<u8> = stream(seed, n, scale)
        .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    context.new_buffer_with_data(&bytes)
}

fn bfloats(context: &MetalContext, seed: u64, n: usize, scale: f32, center: f32) -> MetalBuffer {
    let bytes: Vec<u8> = stream(seed, n, scale)
        .flat_map(|v| bf16::from_f32(center + v).to_bits().to_le_bytes())
        .collect();
    context.new_buffer_with_data(&bytes)
}

fn zeros(context: &MetalContext, bytes: usize) -> MetalBuffer {
    context.new_buffer_with_data(&vec![0u8; bytes])
}

struct Sizes {
    c: usize,
    hv: usize,
    vd: usize,
    k_width: usize,
    state_bytes: usize,
    tail_bytes: usize,
}

impl Sizes {
    fn of(s: GdnShape) -> Self {
        let (c, vd) = (s.qkv_dim() as usize, s.value_dim() as usize);
        Self {
            c,
            hv: s.num_v_heads as usize,
            vd,
            k_width: (s.num_k_heads * s.key_head_dim) as usize,
            state_bytes: vd * s.key_head_dim as usize * 4,
            tail_bytes: (s.conv_kernel_size.max(2) as usize - 1) * c * 2,
        }
    }
}

/// What both arms read and neither writes.
struct LayerInputs {
    qkv: MetalBuffer,
    a: MetalBuffer,
    b: MetalBuffer,
    conv_w: MetalBuffer,
    a_log: MetalBuffer,
    dt_bias: MetalBuffer,
}

impl LayerInputs {
    fn new(context: &MetalContext, shape: GdnShape, z: &Sizes, layer: usize) -> Self {
        let seed = 0x6D00_0000 + 16 * layer as u64;
        // POOL rows, plus room for a chunk that starts at the last one.
        let rows = POOL + SEED_CHUNK.max(MAX_CHUNK);
        Self {
            qkv: halves(context, seed + 1, rows * z.c, 0.8),
            a: halves(context, seed + 2, rows * z.hv, 1.5),
            b: halves(context, seed + 3, rows * z.hv, 1.0),
            conv_w: bfloats(
                context,
                seed + 4,
                z.c * shape.conv_kernel_size as usize,
                0.4,
                0.1,
            ),
            a_log: bfloats(context, seed + 5, z.hv, 1.25, 0.25),
            dt_bias: bfloats(context, seed + 6, z.hv, 0.5, 0.0),
        }
    }
}

/// One arm's own buffers for one layer.
struct ArmLayer {
    state: MetalBuffer,
    tail: MetalBuffer,
    conv_out: MetalBuffer,
    y: MetalBuffer,
    k_tape: MetalBuffer,
    delta_tape: MetalBuffer,
}

impl ArmLayer {
    fn new(context: &MetalContext, z: &Sizes, rows: usize) -> Self {
        Self {
            state: zeros(context, z.state_bytes),
            tail: zeros(context, z.tail_bytes),
            conv_out: zeros(context, rows * z.c * 2),
            y: zeros(context, rows * z.vd * 2),
            k_tape: zeros(context, rows * z.k_width * 2),
            delta_tape: zeros(context, rows * z.vd * 4),
        }
    }
}

/// Run `n` rows starting at pool row `start` through `route`.
#[allow(clippy::too_many_arguments)]
fn encode(
    context: &mut MetalContext,
    pass: &gpu::PassEncoder,
    route: Route,
    shape: GdnShape,
    z: &Sizes,
    i: &LayerInputs,
    m: &ArmLayer,
    start: usize,
    n: usize,
) -> Result<(), String> {
    let e = |r: Result<(), gpu::GpuError>| r.map_err(|e| e.to_string());
    let qkv = |row: usize| (&i.qkv, (row * z.c * 2) as u64);
    let a = |row: usize| (&i.a, (row * z.hv * 2) as u64);
    let b = |row: usize| (&i.b, (row * z.hv * 2) as u64);
    let rows = n as u32;
    if route == Route::Decode {
        for r in 0..n {
            let conv = (&m.conv_out, (r * z.c * 2) as u64);
            e(encode_gdn_conv_decode(
                context,
                pass,
                shape,
                (&m.tail, 0),
                qkv(start + r),
                (&i.conv_w, 0),
                conv,
                1,
            ))?;
            e(encode_gdn_qk_norm(context, pass, shape, conv, 1))?;
            e(encode_gdn_delta_decode(
                context,
                pass,
                shape,
                conv,
                a(start + r),
                b(start + r),
                (&i.a_log, 0),
                (&i.dt_bias, 0),
                &m.state,
                (&m.y, (r * z.vd * 2) as u64),
            ))?;
        }
        return Ok(());
    }
    e(encode_gdn_conv_prefill(
        context,
        pass,
        shape,
        (&m.tail, 0),
        qkv(start),
        (&i.conv_w, 0),
        (&m.conv_out, 0),
        rows,
        1,
    ))?;
    e(encode_gdn_conv_tail_update(
        context,
        pass,
        shape,
        (&m.tail, 0),
        qkv(start),
        rows,
        1,
    ))?;
    e(encode_gdn_qk_norm(
        context,
        pass,
        shape,
        (&m.conv_out, 0),
        rows,
    ))?;
    if route == Route::Prefill {
        return e(encode_gdn_delta_prefill(
            context,
            pass,
            shape,
            (&m.conv_out, 0),
            a(start),
            b(start),
            (&i.a_log, 0),
            (&i.dt_bias, 0),
            &m.state,
            (&m.y, 0),
            rows,
        ));
    }
    // The fold needs each row's normed k slice, which sits after q.
    e(encode_copy_rows_fp16_strided(
        context,
        pass,
        (&m.conv_out, (z.k_width * 2) as u64),
        (&m.k_tape, 0),
        rows,
        z.k_width as u32,
        shape.qkv_dim(),
        z.k_width as u32,
    ))?;
    e(encode_gdn_delta_prefill_lazy(
        context,
        pass,
        shape,
        (&m.conv_out, 0),
        a(start),
        b(start),
        (&i.a_log, 0),
        (&i.dt_bias, 0),
        &m.state,
        (&m.y, 0),
        (&m.delta_tape, 0),
        rows,
    ))?;
    e(encode_gdn_state_fold(
        context,
        pass,
        shape,
        (&m.k_tape, 0),
        (&m.delta_tape, 0),
        a(start),
        (&i.a_log, 0),
        (&i.dt_bias, 0),
        &m.state,
        rows,
    ))
}

/// Flip the lowest bit of the first f32 of a state buffer.
fn flip_one_ulp(state: &MetalBuffer) {
    let mut word: [u8; 4] = read_buffer_bytes(state, 0, 4)
        .try_into()
        .expect("four bytes");
    word[0] ^= 1;
    write_buffer_bytes(state, 0, &word);
}

/// The state and conv tail every arm starts from: `prefill` rows through the
/// eager kernels.
fn seed_state(
    context: &mut MetalContext,
    cfg: &Config,
    z: &Sizes,
    inputs: &[LayerInputs],
) -> Result<Vec<(Vec<u8>, Vec<u8>)>, String> {
    let mut out = Vec::with_capacity(cfg.layers);
    for i in inputs {
        let m = ArmLayer::new(context, z, SEED_CHUNK);
        let mut done = 0;
        while done < cfg.prefill {
            let start = done % POOL;
            let n = SEED_CHUNK.min(cfg.prefill - done).min(POOL - start);
            autorelease_pool(|| -> Result<(), String> {
                let pass = context.begin_pass();
                encode(
                    context,
                    &pass,
                    Route::Prefill,
                    cfg.shape,
                    z,
                    i,
                    &m,
                    start,
                    n,
                )?;
                pass.commit_and_wait();
                Ok(())
            })?;
            done += n;
        }
        out.push((
            read_buffer_bytes(&m.state, 0, z.state_bytes),
            read_buffer_bytes(&m.tail, 0, z.tail_bytes),
        ));
    }
    Ok(out)
}

/// Run both routes `cfg.steps` steps from the shared start state. Stops at
/// the first step that differs.
pub fn run_pair(
    context: &mut MetalContext,
    cfg: &Config,
    routes: [Route; 2],
    b_first: bool,
    fault: Option<Fault>,
) -> Result<Trajectory, String> {
    if cfg.layers == 0 || cfg.steps == 0 || cfg.chunk == 0 || cfg.chunk > MAX_CHUNK {
        return Err(format!(
            "gdn_trajectory needs layers and steps above 0 and a chunk of 1..={MAX_CHUNK}"
        ));
    }
    let z = Sizes::of(cfg.shape);
    let inputs: Vec<LayerInputs> = (0..cfg.layers)
        .map(|l| LayerInputs::new(context, cfg.shape, &z, l))
        .collect();
    let seed = seed_state(context, cfg, &z, &inputs)?;
    let arms: [Vec<ArmLayer>; 2] = std::array::from_fn(|_| {
        seed.iter()
            .map(|(state, tail)| {
                let m = ArmLayer::new(context, &z, cfg.chunk);
                write_buffer_bytes(&m.state, 0, state);
                write_buffer_bytes(&m.tail, 0, tail);
                m
            })
            .collect()
    });
    let order = if b_first { [1, 0] } else { [0, 1] };
    let mut digests = [vec![0u64; cfg.layers], vec![0u64; cfg.layers]];
    let mut step = 0;
    while step < cfg.steps {
        let start = (cfg.prefill + step) % POOL;
        let n = cfg.chunk.min(cfg.steps - step);
        autorelease_pool(|| -> Result<(), String> {
            let pass = context.begin_pass();
            for (l, i) in inputs.iter().enumerate() {
                for a in order {
                    encode(
                        context,
                        &pass,
                        routes[a],
                        cfg.shape,
                        &z,
                        i,
                        &arms[a][l],
                        start,
                        n,
                    )?;
                }
            }
            pass.commit_and_wait();
            Ok(())
        })?;
        let last = step + n - 1;
        if let Some(f) = fault {
            if (step..=last).contains(&f.step) {
                flip_one_ulp(&arms[1][f.layer].state);
            }
        }
        let mut first: Option<Mismatch> = None;
        for l in 0..cfg.layers {
            let read = |a: usize| {
                let m = &arms[a][l];
                [
                    read_buffer_bytes(&m.state, 0, z.state_bytes),
                    read_buffer_bytes(&m.tail, 0, z.tail_bytes),
                    read_buffer_bytes(&m.y, 0, n * z.vd * 2),
                ]
            };
            let (left, right) = (read(0), read(1));
            for (a, bytes) in [&left, &right].into_iter().enumerate() {
                for part in bytes {
                    digests[a][l] = hash_words(digests[a][l], part);
                }
            }
            // Outputs name the exact row; state and tail the chunk's end.
            let differs = left[2]
                .chunks_exact(z.vd * 2)
                .zip(right[2].chunks_exact(z.vd * 2))
                .position(|(x, y)| x != y)
                .map(|row| (step + row, "y"))
                .or((left[0] != right[0]).then_some((last, "state")))
                .or((left[1] != right[1]).then_some((last, "conv_tail")));
            if let Some((at, what)) = differs {
                if first.as_ref().is_none_or(|f| at < f.step) {
                    first = Some(Mismatch {
                        layer: l,
                        step: at,
                        what,
                    });
                }
            }
        }
        if first.is_some() {
            return Ok(Trajectory { first, digests });
        }
        step += n;
    }
    Ok(Trajectory {
        first: None,
        digests,
    })
}

/// Arm `route` against itself with one flipped ULP: the comparison must
/// report that layer at that step's chunk. Returns what it injected.
pub fn fault_self_test(
    context: &mut MetalContext,
    cfg: &Config,
    route: Route,
) -> Result<Fault, String> {
    let fault = Fault {
        layer: 1.min(cfg.layers.saturating_sub(1)),
        step: FAULT_STEP.min(cfg.steps.saturating_sub(1)),
    };
    let found = run_pair(context, cfg, [route, route], false, Some(fault))?.first;
    let chunk_end = (fault.step / cfg.chunk * cfg.chunk + cfg.chunk).min(cfg.steps) - 1;
    let expected = Mismatch {
        layer: fault.layer,
        step: chunk_end,
        what: "state",
    };
    if found.as_ref() != Some(&expected) {
        return Err(format!(
            "gdn_trajectory self-test failed: one ULP injected at layer {} step {} should read {expected:?}, got {found:?}",
            fault.layer, fault.step
        ));
    }
    Ok(fault)
}

pub struct GdnTrajectory;

impl KernelBench for GdnTrajectory {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> &'static str {
        "gated-DeltaNet state run forward through two routes, state hashed per layer and step; --steps, --prefill, --chunk, --layers"
    }

    fn flags(&self) -> &'static [&'static str] {
        &["steps", "prefill", "chunk", "layers"]
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let arch = subject.arch()?;
        if !arch.has_linear_attention_layers() {
            return Err("no gated-DeltaNet layer".into());
        }
        let l = &arch.linear_attention;
        Ok(PAIRS
            .iter()
            .map(|(label, _, _)| Case {
                label: (*label).into(),
                params: json!({
                    "num_k_heads": l.num_k_heads, "num_v_heads": l.num_v_heads,
                    "key_head_dim": l.key_head_dim, "value_head_dim": l.value_head_dim,
                    "conv_kernel_size": l.conv_kernel_size, "count": layer_count(arch),
                }),
            })
            .collect())
    }

    fn run(
        &self,
        context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        _opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let (_, a, b) = PAIRS
            .iter()
            .find(|p| p.0 == case.label)
            .copied()
            .ok_or_else(|| format!("gdn_trajectory has no case {:?}", case.label))?;
        let dim = |k: &str| {
            case.params[k]
                .as_u64()
                .map(|v| v as u32)
                .ok_or_else(|| format!("gdn_trajectory case lacks {k}"))
        };
        let shape = GdnShape {
            num_k_heads: dim("num_k_heads")?,
            num_v_heads: dim("num_v_heads")?,
            key_head_dim: dim("key_head_dim")?,
            value_head_dim: dim("value_head_dim")?,
            conv_kernel_size: dim("conv_kernel_size")?,
        };
        shape.validate().map_err(|e| e.to_string())?;
        // A command-line value arrives as a string.
        let given = |k: &str, default: usize| -> Result<usize, String> {
            match case.params[k].as_str() {
                None => Ok(default),
                Some(v) => v.parse().map_err(|_| format!("--{k} {v:?}")),
            }
        };
        let all = dim("count")? as usize;
        let cfg = Config {
            shape,
            layers: given("layers", all)?.min(all),
            steps: given("steps", DEFAULT_STEPS)?,
            prefill: given("prefill", DEFAULT_PREFILL)?,
            chunk: given("chunk", 1)?,
        };
        let fault = fault_self_test(context, &cfg, a)?;
        let forward = run_pair(context, &cfg, [a, b], false, None)?;
        let reverse = run_pair(context, &cfg, [a, b], true, None)?;
        let order_stable = forward == reverse;
        let bit_exact = order_stable && forward.first.is_none();
        let mismatch = |t: &Trajectory| {
            t.first
                .as_ref()
                .map(|m| json!({"layer": m.layer, "step": m.step, "what": m.what}))
        };
        let mut subject = BTreeMap::new();
        for (k, v) in [
            ("rows", json!(cfg.prefill + cfg.steps)),
            ("prefill", json!(cfg.prefill)),
            ("steps", json!(cfg.steps)),
            ("chunk", json!(cfg.chunk)),
            ("layers", json!(cfg.layers)),
            ("bit_exact", json!(bit_exact)),
            ("order_stable", json!(order_stable)),
            ("first_mismatch", json!(mismatch(&forward))),
            ("first_mismatch_b_first", json!(mismatch(&reverse))),
            (
                "fault_self_test",
                json!({"layer": fault.layer, "step": fault.step, "detected": true}),
            ),
        ] {
            subject.insert(k.to_string(), v);
        }
        let verdict = match &forward.first {
            None => format!("no mismatch over {} steps", cfg.steps),
            Some(m) => format!(
                "first mismatch at layer {} step {} ({})",
                m.layer, m.step, m.what
            ),
        };
        let arms = [a, b]
            .iter()
            .zip(&forward.digests)
            .map(|(route, digests)| ArmCurve {
                label: route.label().into(),
                points: digests
                    .iter()
                    .enumerate()
                    .map(|(layer, d)| {
                        let mut p = Point::new(layer as f64, 0.0);
                        p.out_fnv = Some(format!("{d:016x}"));
                        p
                    })
                    .collect(),
                fit: None,
                unsupported: None,
            })
            .collect();
        Ok(BenchResult {
            bench: NAME.into(),
            case: case.label.clone(),
            subject,
            x: Axis::new("config", "index", "linear"),
            layout: "gdn_f16_inputs_f32_state".into(),
            arms,
            break_even: Vec::new(),
            notes: vec![
                format!(
                    "{verdict}; order stable: {order_stable}; one-ULP self-test detected at layer {} step {}",
                    fault.layer, fault.step
                ),
                format!(
                    "prefill {} rows (eager), chunk {}; synthetic inputs, layers do not feed each other; points are per-layer trajectory digests",
                    cfg.prefill, cfg.chunk
                ),
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_word_hash_sees_every_byte_and_the_order() {
        let base: Vec<u8> = (0..21).collect();
        let h = hash_words(0, &base);
        for i in 0..base.len() {
            let mut changed = base.clone();
            changed[i] ^= 1;
            assert_ne!(hash_words(0, &changed), h, "byte {i}");
        }
        let mut swapped = base.clone();
        swapped[..16].rotate_left(8);
        assert_ne!(hash_words(0, &swapped), h);
    }
}
