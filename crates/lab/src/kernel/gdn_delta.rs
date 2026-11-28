//! Gated-DeltaNet delta kernels at the install's own geometry: the one-row
//! decode step, the eager multi-row prefill, the lazy verify kernel that
//! records a tape, and the state fold that commits it.
//!
//! The port of `gdn::gdn_tests::baked_dk128_speed_at_qwen38_shapes`, with the
//! head counts, head dims and GDN layer count taken from `ArchConfig`. It is
//! the bench that would have priced `TURBOSPARK_GDN_DK128` before the model
//! A/B: a kernel that loops a runtime count shows up here as a large ratio
//! between two runs of the same bench.
//!
//! One build per process. Whether the Dk-baked pipelines are used is a
//! process-cached environment read in the gpu crate (`TURBOSPARK_GDN_DK128`),
//! and the in-process override the unit test uses is test-only. So a mode is
//! compared across two experiments (`TURBOSPARK_GDN_DK128=0 tslab kernel ...`
//! against the default), and `tslab hash-diff` or `tslab project
//! --default-from A --mode-from B` reads the pair. The environment value is
//! recorded in the case's subject so a ledger says which build it timed.
//!
//! Inputs are deterministic and fixed, never random: the hashes under
//! `--hash` are a function of the arithmetic alone.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_gdn_delta_decode, encode_gdn_delta_prefill,
    encode_gdn_delta_prefill_lazy, encode_gdn_state_fold, read_buffer_bytes, GdnShape, MetalBuffer,
    MetalContext,
};
use half::{bf16, f16};
use serde_json::json;

use super::fit::{c_of_m, fit_affine};
use super::support::median;
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, Point};

pub const NAME: &str = "gdn_delta";
/// The arms, in the order they run. `decode` has one point (one row).
pub const ARMS: [&str; 4] = ["decode", "prefill_eager", "prefill_lazy", "state_fold"];
/// The largest verify or chunk width timed. A speculative block is at most 16
/// rows, and the buffers are sized for it.
pub const MAX_ROWS: usize = 32;
/// Dispatches per timed pass. One pass per sample keeps the GPU timestamps
/// above the command buffer's own overhead (the in-tree bench's 64).
const PER_PASS: usize = 32;
/// Distinct state and output buffer sets the dispatches rotate through, so a
/// dispatch does not find the previous one's state in the cache.
const COPIES: usize = 8;

pub struct GdnDelta;

fn shape_of(arch: &model_io::ArchConfig) -> Result<GdnShape, String> {
    if !arch.has_linear_attention_layers() {
        return Err("no gated-DeltaNet layer".into());
    }
    let l = &arch.linear_attention;
    let shape = GdnShape {
        num_k_heads: l.num_k_heads as u32,
        num_v_heads: l.num_v_heads as u32,
        key_head_dim: l.key_head_dim as u32,
        value_head_dim: l.value_head_dim as u32,
        conv_kernel_size: l.conv_kernel_size as u32,
    };
    shape.validate().map_err(|e| e.to_string())?;
    Ok(shape)
}

/// GDN layers one forward runs (mask value 2).
pub fn layer_count(arch: &model_io::ArchConfig) -> usize {
    arch.full_attention_layer_mask
        .iter()
        .filter(|&&m| m == 2)
        .count()
}

impl KernelBench for GdnDelta {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> &'static str {
        "gated-DeltaNet delta kernels at the install's geometry: decode, eager and lazy prefill, state fold, over rows"
    }

    /// `--oracle on`: also check decode and eager prefill against FP64.
    fn flags(&self) -> &'static [&'static str] {
        &["oracle"]
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let arch = subject.arch()?;
        let shape = shape_of(arch)?;
        let layers = layer_count(arch);
        Ok(vec![Case {
            label: "gdn".into(),
            params: json!({
                "num_k_heads": shape.num_k_heads, "num_v_heads": shape.num_v_heads,
                "key_head_dim": shape.key_head_dim, "value_head_dim": shape.value_head_dim,
                "conv_kernel_size": shape.conv_kernel_size, "count": layers,
            }),
        }])
    }

    fn run(
        &self,
        context: &mut MetalContext,
        subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let arch = subject.arch()?;
        let shape = shape_of(arch)?;
        let layers = layer_count(arch);
        let mut result = run_shape(context, &case.label, shape, layers, opts)?;
        match case.params["oracle"].as_str() {
            None | Some("off") => {}
            Some("on") => {
                let arms = oracle_shape(context, shape)?;
                if let Some(blind) = arms.iter().find(|a| !a.caught) {
                    return Err(format!(
                        "gdn_delta oracle cannot see a dt_bias shift of {ORACLE_DT_SHIFT} on {} \
                         (perturbed y {:.3e}, state {:.3e}; tolerance {:.3e}, {:.3e}): \
                         its tolerance is too loose to judge a candidate",
                        blind.arm,
                        blind.perturbed_y.max_abs,
                        blind.perturbed_state.max_abs,
                        blind.tol_y,
                        blind.tol_state
                    ));
                }
                result.subject.insert("oracle".into(), json!(arms));
                result.notes.push(
                    "oracle: decode and prefill_eager against FP64 on the pooled seeds; \
                     tolerance max(floor x |ref|, 3 x the incumbent's error), and a dt_bias shift must exceed it"
                        .into(),
                );
            }
            Some(other) => return Err(format!("--oracle {other:?}: expected on or off")),
        }
        Ok(result)
    }
}

/// The deterministic fill every input uses: an xorshift stream mapped to
/// `[-scale, scale]`. Same bytes on every run and build.
fn stream(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s % 2000) as f32 / 1000.0) - 1.0) * scale
        })
        .collect()
}

fn f16_bytes(seed: u64, n: usize, scale: f32) -> Vec<u8> {
    stream(seed, n, scale)
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

fn bf16_bytes(seed: u64, n: usize, scale: f32, center: f32) -> Vec<u8> {
    stream(seed, n, scale)
        .into_iter()
        .flat_map(|v| bf16::from_f32(center + v).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(seed: u64, n: usize, scale: f32) -> Vec<u8> {
    stream(seed, n, scale)
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// One set of buffers a dispatch writes: state, output, delta tape.
struct Mutable {
    state: MetalBuffer,
    y: MetalBuffer,
    delta_tape: MetalBuffer,
}

/// Everything a kernel reads and does not change.
struct Inputs {
    conv: MetalBuffer,
    a: MetalBuffer,
    b: MetalBuffer,
    a_log: MetalBuffer,
    dt_bias: MetalBuffer,
    k_tape: MetalBuffer,
}

struct Dims {
    shape: GdnShape,
    vd: usize,
    state_len: usize,
    k_width: usize,
}

impl Dims {
    fn new(shape: GdnShape) -> Self {
        Self {
            vd: shape.value_dim() as usize,
            state_len: shape.value_dim() as usize * shape.key_head_dim as usize,
            k_width: (shape.num_k_heads * shape.key_head_dim) as usize,
            shape,
        }
    }
}

fn inputs(context: &MetalContext, d: &Dims, max_rows: usize) -> Inputs {
    inputs_shifted(context, d, max_rows, 0.0)
}

/// [`inputs`] with `dt_shift` added to every `dt_bias` before its bf16
/// rounding: the oracle's deliberately perturbed gate. 0.0 gives the bench's
/// own bytes.
fn inputs_shifted(context: &MetalContext, d: &Dims, max_rows: usize, dt_shift: f32) -> Inputs {
    let hv = d.shape.num_v_heads as usize;
    let c = d.shape.qkv_dim() as usize;
    Inputs {
        conv: context.new_buffer_with_data(&f16_bytes(0xD1CE, max_rows * c, 1.0)),
        a: context.new_buffer_with_data(&f16_bytes(0xA000, max_rows * hv, 1.5)),
        b: context.new_buffer_with_data(&f16_bytes(0xB000, max_rows * hv, 1.0)),
        a_log: context.new_buffer_with_data(&bf16_bytes(0xA100, hv, 1.25, 0.25)),
        dt_bias: context.new_buffer_with_data(&bf16_bytes(0xA200, hv, 0.5, dt_shift)),
        k_tape: context.new_buffer_with_data(&f16_bytes(0xC000, max_rows * d.k_width, 0.5)),
    }
}

fn mutable(context: &MetalContext, d: &Dims, max_rows: usize, seed: u64) -> Mutable {
    Mutable {
        state: context.new_buffer_with_data(&f32_bytes(seed, d.state_len, 0.05)),
        y: context.new_output_buffer((max_rows * d.vd * 2) as u64),
        delta_tape: context.new_buffer_with_data(&f32_bytes(seed ^ 0x77, max_rows * d.vd, 0.1)),
    }
}

/// Encode one dispatch of `arm` at `rows` rows into `pass`.
fn encode(
    context: &mut MetalContext,
    pass: &gpu::PassEncoder,
    arm: &str,
    d: &Dims,
    i: &Inputs,
    m: &Mutable,
    rows: u32,
) -> Result<(), String> {
    let r = match arm {
        "decode" => encode_gdn_delta_decode(
            context,
            pass,
            d.shape,
            (&i.conv, 0),
            (&i.a, 0),
            (&i.b, 0),
            (&i.a_log, 0),
            (&i.dt_bias, 0),
            &m.state,
            (&m.y, 0),
        ),
        "prefill_eager" => encode_gdn_delta_prefill(
            context,
            pass,
            d.shape,
            (&i.conv, 0),
            (&i.a, 0),
            (&i.b, 0),
            (&i.a_log, 0),
            (&i.dt_bias, 0),
            &m.state,
            (&m.y, 0),
            rows,
        ),
        "prefill_lazy" => encode_gdn_delta_prefill_lazy(
            context,
            pass,
            d.shape,
            (&i.conv, 0),
            (&i.a, 0),
            (&i.b, 0),
            (&i.a_log, 0),
            (&i.dt_bias, 0),
            &m.state,
            (&m.y, 0),
            (&m.delta_tape, 0),
            rows,
        ),
        "state_fold" => encode_gdn_state_fold(
            context,
            pass,
            d.shape,
            (&i.k_tape, 0),
            (&m.delta_tape, 0),
            (&i.a, 0),
            (&i.a_log, 0),
            (&i.dt_bias, 0),
            &m.state,
            rows,
        ),
        other => return Err(format!("gdn_delta has no arm {other:?}")),
    };
    r.map_err(|e| e.to_string())
}

/// Which rows an arm runs at: `decode` is the one-row step only.
fn runs_at(arm: &str, rows: usize) -> bool {
    arm != "decode" || rows == 1
}

/// The bytes and first-row bytes one fresh dispatch of `arm` leaves, for
/// `--hash`: the output `y`, or the folded state for `state_fold`.
fn hashes(
    context: &mut MetalContext,
    arm: &str,
    d: &Dims,
    i: &Inputs,
    rows: usize,
    max_rows: usize,
) -> Result<(String, Option<String>), String> {
    let m = mutable(context, d, max_rows, 0x5EED);
    autorelease_pool(|| -> Result<(), String> {
        let pass = context.begin_pass();
        encode(context, &pass, arm, d, i, &m, rows as u32)?;
        pass.commit_and_wait_with_gpu_time();
        Ok(())
    })?;
    if arm == "state_fold" {
        let bytes = read_buffer_bytes(&m.state, 0, d.state_len * 4);
        return Ok((fnv_hex(&bytes), None));
    }
    let n = if arm == "decode" { 1 } else { rows };
    let all = read_buffer_bytes(&m.y, 0, n * d.vd * 2);
    let first = read_buffer_bytes(&m.y, 0, d.vd * 2);
    Ok((fnv_hex(&all), Some(fnv_hex(&first))))
}

pub fn run_shape(
    context: &mut MetalContext,
    label: &str,
    shape: GdnShape,
    layers: usize,
    opts: &BenchOpts,
) -> Result<BenchResult, String> {
    let rows_list: Vec<usize> = opts
        .rows
        .iter()
        .copied()
        .filter(|&r| r <= MAX_ROWS)
        .collect();
    if rows_list.is_empty() {
        return Err(format!(
            "gdn_delta times 1..={MAX_ROWS} rows; --rows has nothing in that range"
        ));
    }
    let max_rows = *rows_list.iter().max().unwrap_or(&1);
    let d = Dims::new(shape);
    let ins = inputs(context, &d, max_rows);
    let sets: Vec<Mutable> = (0..COPIES)
        .map(|c| mutable(context, &d, max_rows, 0xFACE + c as u64))
        .collect();

    // (arm index, rows) -> median ms per dispatch
    let mut samples: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
    for &rows in &rows_list {
        for rep in 0..=opts.reps {
            for (a, arm) in ARMS.iter().enumerate() {
                if !runs_at(arm, rows) {
                    continue;
                }
                let t = autorelease_pool(|| -> Result<f64, String> {
                    let pass = context.begin_pass();
                    for k in 0..PER_PASS {
                        encode(
                            context,
                            &pass,
                            arm,
                            &d,
                            &ins,
                            &sets[k % COPIES],
                            rows as u32,
                        )?;
                    }
                    Ok(pass.commit_and_wait_with_gpu_time())
                })?;
                if rep > 0 {
                    samples
                        .entry((a, rows))
                        .or_default()
                        .push(t * 1e3 / PER_PASS as f64);
                }
            }
        }
    }
    // One fresh dispatch per point for the hashes.
    let mut hash_at: BTreeMap<(usize, usize), (String, Option<String>)> = BTreeMap::new();
    if opts.hash {
        for &rows in &rows_list {
            for (a, arm) in ARMS.iter().enumerate() {
                if runs_at(arm, rows) {
                    hash_at.insert((a, rows), hashes(context, arm, &d, &ins, rows, max_rows)?);
                }
            }
        }
    }

    // The c(M) unit: one decode row.
    let one_row = samples
        .get_mut(&(0, 1))
        .map(|s| median(s))
        .ok_or("gdn_delta needs rows to include 1: the decode step is its c(M) unit")?;
    let mut arms = Vec::new();
    for (a, arm) in ARMS.iter().enumerate() {
        let mut points = Vec::new();
        for &rows in &rows_list {
            let Some(s) = samples.get_mut(&(a, rows)) else {
                continue;
            };
            let ms = median(s);
            let mut p = Point::new(rows as f64, ms);
            p.c_of_m = Some(c_of_m(ms, one_row, rows));
            if let Some((all, first)) = hash_at.get(&(a, rows)) {
                p.out_fnv = Some(all.clone());
                p.row_fnv = first.clone();
            }
            points.push(p);
        }
        let fit = fit_affine(&points.iter().map(|p| (p.x, p.ms)).collect::<Vec<_>>());
        arms.push(ArmCurve {
            label: arm.to_string(),
            points,
            fit,
            unsupported: None,
        });
    }

    let mut subject = BTreeMap::new();
    subject.insert("num_k_heads".into(), json!(shape.num_k_heads));
    subject.insert("num_v_heads".into(), json!(shape.num_v_heads));
    subject.insert("key_head_dim".into(), json!(shape.key_head_dim));
    subject.insert("value_head_dim".into(), json!(shape.value_head_dim));
    // Per forward, so a projection can weight a per-dispatch delta.
    subject.insert("count".into(), json!(layers));
    subject.insert("one_row_decode_ms".into(), json!(one_row));
    // Which build this process timed (see the module doc).
    subject.insert(
        "gdn_dk128_env".into(),
        json!(std::env::var("TURBOSPARK_GDN_DK128").ok()),
    );
    let mut notes = vec![
        "one build per process: set TURBOSPARK_GDN_DK128=0 on the tslab process for the runtime-Dk build; \
         state_fold's hash is the folded state, not a row of y"
            .to_string(),
    ];
    if opts.rows.iter().any(|&r| r > MAX_ROWS) {
        notes.push(format!("rows above {MAX_ROWS} are not timed"));
    }
    Ok(BenchResult {
        bench: NAME.into(),
        case: label.into(),
        subject,
        x: Axis::new("m", "rows", "log2"),
        layout: "gdn_f16_inputs_f32_state".into(),
        arms,
        break_even: Vec::new(),
        notes,
    })
}

// ---- FP64 oracle (KERN-13) ----

/// The pooled fixture set: initial-state seeds the oracle runs every arm on.
pub const ORACLE_SEEDS: [u64; 4] = [0x5EED, 0x0A11, 0xBEEF, 0xF00D];
/// Rows the eager prefill arm runs in the oracle.
pub const ORACLE_ROWS: usize = 8;
/// The perturbed arm's `dt_bias` shift: a small gate bug the oracle must see.
pub const ORACLE_DT_SHIFT: f32 = 0.01;
/// Floors, relative to the largest reference magnitude, set 2026-11-26 from
/// the incumbent's pooled error at Qwen3.6 and a small geometry (test
/// `oracle_records_the_incumbent_error_and_catches_a_perturbed_gate`): y
/// measured 2.3e-4 to 3.8e-4, the `half` store of y, so its floor is one half
/// rounding step (2^-11); the f32 state measured 1.3e-7 to 3.6e-7, so its
/// floor is 8 f32 rounding steps (8 x 2^-23).
pub const ORACLE_FLOOR_REL_Y: f64 = 4.8828125e-4;
pub const ORACLE_FLOOR_REL_STATE: f64 = 9.5367431640625e-7;

/// The largest error and reference magnitude over the pooled seeds.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct Pooled {
    pub max_abs: f64,
    pub max_ref: f64,
}

impl Pooled {
    fn add(&mut self, s: compute::oracle::ErrorStats) {
        self.max_abs = self.max_abs.max(s.max_abs);
        self.max_ref = self.max_ref.max(s.max_ref);
    }

    fn rel(&self) -> f64 {
        if self.max_ref > 0.0 {
            self.max_abs / self.max_ref
        } else {
            self.max_abs
        }
    }
}

/// One arm's oracle reading: the shipped kernel and the perturbed run, both
/// against FP64, and the tolerance a candidate must meet.
#[derive(Debug, Clone, serde::Serialize)]
pub struct OracleArm {
    pub arm: String,
    pub rows: usize,
    pub y: Pooled,
    pub state: Pooled,
    pub y_rel: f64,
    pub state_rel: f64,
    pub perturbed_y: Pooled,
    pub perturbed_state: Pooled,
    pub tol_y: f64,
    pub tol_state: f64,
    /// The perturbed run exceeded the tolerance on y or state.
    pub caught: bool,
}

fn f16_vals(seed: u64, n: usize, scale: f32) -> Vec<f64> {
    stream(seed, n, scale)
        .into_iter()
        .map(|v| f16::from_f32(v).to_f64())
        .collect()
}

fn bf16_vals(seed: u64, n: usize, scale: f32, center: f32) -> Vec<f64> {
    stream(seed, n, scale)
        .into_iter()
        .map(|v| bf16::from_f32(center + v).to_f64())
        .collect()
}

/// The FP64 reference for `rows` rows from the state `seed` gives, on the
/// unperturbed inputs: `(y, final state)`.
fn reference(d: &Dims, seed: u64, rows: usize, max_rows: usize) -> (Vec<f64>, Vec<f64>) {
    let hv = d.shape.num_v_heads as usize;
    let c = d.shape.qkv_dim() as usize;
    let dims = compute::gdn::GdnDims {
        num_k_heads: d.shape.num_k_heads as usize,
        num_v_heads: hv,
        key_head_dim: d.shape.key_head_dim as usize,
        value_head_dim: d.shape.value_head_dim as usize,
        conv_kernel_size: d.shape.conv_kernel_size as usize,
    };
    let conv = f16_vals(0xD1CE, max_rows * c, 1.0);
    let a = f16_vals(0xA000, max_rows * hv, 1.5);
    let b = f16_vals(0xB000, max_rows * hv, 1.0);
    let a_log = bf16_vals(0xA100, hv, 1.25, 0.25);
    let dt_bias = bf16_vals(0xA200, hv, 0.5, 0.0);
    let mut state: Vec<f64> = stream(seed, d.state_len, 0.05)
        .into_iter()
        .map(f64::from)
        .collect();
    let mut y = Vec::with_capacity(rows * d.vd);
    for r in 0..rows {
        y.extend(compute::oracle::gdn_delta_step_f64(
            dims,
            &conv[r * c..][..c],
            &a[r * hv..][..hv],
            &b[r * hv..][..hv],
            &a_log,
            &dt_bias,
            &mut state,
        ));
    }
    (y, state)
}

/// One GPU dispatch of `arm` from the state `seed` gives: `(y, state)`.
fn gpu_once(
    context: &mut MetalContext,
    arm: &str,
    d: &Dims,
    ins: &Inputs,
    seed: u64,
    rows: usize,
    max_rows: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let m = mutable(context, d, max_rows, seed);
    autorelease_pool(|| -> Result<(), String> {
        let pass = context.begin_pass();
        encode(context, &pass, arm, d, ins, &m, rows as u32)?;
        pass.commit_checked().map_err(|e| e.to_string())?.wait();
        Ok(())
    })?;
    let y = gpu::read_buffer_f16(&m.y, 0, rows * d.vd)
        .into_iter()
        .map(f16::to_f32)
        .collect();
    Ok((y, gpu::read_f32_buffer(&m.state, d.state_len)))
}

/// Runs the oracle for `decode` and `prefill_eager` at `shape`.
pub fn oracle_shape(context: &mut MetalContext, shape: GdnShape) -> Result<Vec<OracleArm>, String> {
    use compute::oracle::{error_stats, oracle_tolerance};
    let d = Dims::new(shape);
    let max_rows = ORACLE_ROWS;
    let ins = inputs(context, &d, max_rows);
    let bent = inputs_shifted(context, &d, max_rows, ORACLE_DT_SHIFT);
    let mut out = Vec::new();
    for (arm, rows) in [("decode", 1), ("prefill_eager", ORACLE_ROWS)] {
        let (mut y, mut state, mut py, mut ps) = Default::default();
        for &seed in &ORACLE_SEEDS {
            let (want_y, want_state) = reference(&d, seed, rows, max_rows);
            let (gy, gs) = gpu_once(context, arm, &d, &ins, seed, rows, max_rows)?;
            Pooled::add(&mut y, error_stats(&gy, &want_y));
            Pooled::add(&mut state, error_stats(&gs, &want_state));
            let (by, bs) = gpu_once(context, arm, &d, &bent, seed, rows, max_rows)?;
            Pooled::add(&mut py, error_stats(&by, &want_y));
            Pooled::add(&mut ps, error_stats(&bs, &want_state));
        }
        let (y, state, py, ps): (Pooled, Pooled, Pooled, Pooled) = (y, state, py, ps);
        let tol_y = oracle_tolerance(ORACLE_FLOOR_REL_Y * y.max_ref, y.max_abs);
        let tol_state = oracle_tolerance(ORACLE_FLOOR_REL_STATE * state.max_ref, state.max_abs);
        out.push(OracleArm {
            arm: arm.into(),
            rows,
            y_rel: y.rel(),
            state_rel: state.rel(),
            y,
            state,
            perturbed_y: py,
            perturbed_state: ps,
            tol_y,
            tol_state,
            caught: py.max_abs > tol_y || ps.max_abs > tol_state,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_runs_at_one_row_only() {
        assert!(runs_at("decode", 1));
        assert!(!runs_at("decode", 8));
        for arm in &ARMS[1..] {
            assert!(runs_at(arm, 8), "{arm}");
        }
    }

    #[test]
    fn the_stream_is_deterministic_and_bounded() {
        assert_eq!(stream(7, 64, 0.5), stream(7, 64, 0.5));
        assert_ne!(stream(7, 64, 0.5), stream(8, 64, 0.5));
        assert!(stream(7, 4096, 0.5).iter().all(|v| v.abs() <= 0.5));
    }

    /// Qwen3.6 GDN geometry and a small one.
    fn oracle_shapes() -> [GdnShape; 2] {
        let shape = |hk, hv, dk, dv| GdnShape {
            num_k_heads: hk,
            num_v_heads: hv,
            key_head_dim: dk,
            value_head_dim: dv,
            conv_kernel_size: 4,
        };
        [shape(16, 32, 128, 128), shape(2, 4, 64, 32)]
    }

    #[test]
    fn oracle_records_the_incumbent_error_and_catches_a_perturbed_gate() {
        let mut context = MetalContext::new().expect("Metal device");
        for shape in oracle_shapes() {
            let arms = oracle_shape(&mut context, shape).expect("oracle");
            for a in &arms {
                eprintln!(
                    "{:?} {}: y {:.3e} rel {:.3e}, state {:.3e} rel {:.3e}; perturbed y {:.3e}, state {:.3e}",
                    (shape.num_k_heads, shape.num_v_heads, shape.key_head_dim, shape.value_head_dim),
                    a.arm, a.y.max_abs, a.y_rel, a.state.max_abs, a.state_rel,
                    a.perturbed_y.max_abs, a.perturbed_state.max_abs
                );
                assert!(a.caught, "{a:?}");
                // The shipped kernel agrees with FP64 to storage rounding;
                // a wrong reference (or a regressed kernel) fails here.
                assert!(a.y_rel <= ORACLE_FLOOR_REL_Y, "{a:?}");
                assert!(a.state_rel <= ORACLE_FLOOR_REL_STATE, "{a:?}");
            }
        }
    }

    #[test]
    fn arm_names_are_unique() {
        let mut names = ARMS.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ARMS.len());
    }
}
