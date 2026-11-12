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
        run_shape(context, &case.label, shape, layers, opts)
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
    let hv = d.shape.num_v_heads as usize;
    let c = d.shape.qkv_dim() as usize;
    Inputs {
        conv: context.new_buffer_with_data(&f16_bytes(0xD1CE, max_rows * c, 1.0)),
        a: context.new_buffer_with_data(&f16_bytes(0xA000, max_rows * hv, 1.5)),
        b: context.new_buffer_with_data(&f16_bytes(0xB000, max_rows * hv, 1.0)),
        a_log: context.new_buffer_with_data(&bf16_bytes(0xA100, hv, 1.25, 0.25)),
        dt_bias: context.new_buffer_with_data(&bf16_bytes(0xA200, hv, 0.5, 0.0)),
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

    #[test]
    fn arm_names_are_unique() {
        let mut names = ARMS.to_vec();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ARMS.len());
    }
}
