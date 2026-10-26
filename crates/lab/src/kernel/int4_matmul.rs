//! INT4 affine matmul at the install's own shapes: M one-row GEMVs against
//! the scalar batched GEMM, both Steel tiles and the 8-row MMA tile, over a
//! sweep of M.
//!
//! The port of `crates/gpu/tests/qwen38_int4_prefill_gemm_probe.rs` with the
//! shapes taken from `ArchConfig` instead of Qwen3.8's constants, and the
//! one-row GEMV added as the unit that c(M) is measured against.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_dequant_int4_gemm_resident, encode_dequant_int4_gemm_steel,
    encode_dequant_int4_gemv_resident, encode_dequant_int4_mma8_resident, read_buffer_bytes,
    steel_supports, Int4ResidentMatrix, MetalContext, SteelTile, INT4_MMA8_MAX_ROWS,
    MAX_BATCH_ROWS,
};
use model_io::ArchConfig;
use serde_json::json;

use super::fit::{break_even, c_of_m, fit_affine};
use super::support::{copies_for, f16_activations, int4_blob, median};
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, BreakEven, Point};

pub use crate::bench::INT4_AFFINE_LAYOUT as LAYOUT;

/// The engine serves at most this many rows through the tile per call
/// sequence (`encode_gemm_any` batches up to 16 rows in tiles of 8), so the
/// bench does not time the tile at M where nothing would route to it.
const MMA8_BENCH_MAX_M: usize = 2 * INT4_MMA8_MAX_ROWS;

/// Dispatches a timed pass holds at least (see `sweeps` in [`run_shape`]).
const MIN_DISPATCHES_PER_PASS: usize = 64;

pub struct Int4Matmul;

impl KernelBench for Int4Matmul {
    fn name(&self) -> &'static str {
        "int4_matmul"
    }

    fn describe(&self) -> &'static str {
        "INT4 projections at the install's shapes: per-row GEMV, scalar GEMM, Steel tiles, MMA8 tile over M"
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let arch = subject.arch()?;
        let (mut shapes, source) =
            super::matmul_shapes(subject, crate::inventory::DTYPE_INT4_AFFINE, || {
                shapes(arch)
            });
        // Routed experts live in packed_experts/, outside the resident index.
        for (rows, cols) in routed_shapes(arch) {
            if !shapes.iter().any(|s| (s.rows, s.cols) == (rows, cols)) {
                shapes.push(super::MatShape {
                    rows,
                    cols,
                    roles: vec!["routed_expert".into()],
                    count: arch.top_k_experts.max(0) as usize,
                });
            }
        }
        if shapes.is_empty() {
            return Err("no INT4 affine projection with a width divisible by 64".into());
        }
        Ok(shapes.iter().map(|s| s.case(source)).collect())
    }

    fn run(
        &self,
        context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let dim = |k: &str| {
            case.params[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("int4_matmul case {} lacks {k}", case.label))
        };
        let mut r = run_shape(context, &case.label, dim("rows")?, dim("cols")?, opts)?;
        for k in ["roles", "count", "shape_source"] {
            if !case.params[k].is_null() {
                r.subject.insert(k.into(), case.params[k].clone());
            }
        }
        Ok(r)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arm {
    /// M separate one-row GEMV dispatches: what M tokens cost unbatched.
    GemvPerRow,
    /// The scalar batched GEMM, `MAX_BATCH_ROWS` rows per dispatch.
    Scalar,
    Steel(SteelTile),
    /// The small-M matrix-unit tile (`TURBOSPARK_INT4_MMA8`), 8 rows a call.
    Mma8,
}

impl Arm {
    pub fn label(&self) -> String {
        match self {
            Arm::GemvPerRow => "gemv_per_row".into(),
            Arm::Scalar => "scalar_gemm".into(),
            Arm::Steel(SteelTile::M32K32N32) => "steel_m32".into(),
            Arm::Steel(SteelTile::M64K32N64) => "steel_m64".into(),
            Arm::Mma8 => "mma8".into(),
        }
    }
}

pub const ARMS: [Arm; 5] = [
    Arm::GemvPerRow,
    Arm::Scalar,
    Arm::Steel(SteelTile::M32K32N32),
    Arm::Steel(SteelTile::M64K32N64),
    Arm::Mma8,
];

/// The config-derived INT4 shapes, the fallback when the resident index
/// cannot be read: (rows, cols) pairs, deduplicated, restricted to the
/// layout the resident kernels take (cols a multiple of 64), routed-expert
/// shapes included for MoE installs.
pub fn shapes(arch: &ArchConfig) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = arch
        .decode_int4_gemv_shapes()
        .into_iter()
        .map(|(m, n)| (m as usize, n as usize))
        .collect();
    let (h, nh) = (arch.hidden_size, arch.num_heads);
    if arch.has_mla_layers() {
        // MLA projects through latent ranks, so the q/k/v/o rows the
        // generic list computes from `full_head_dim` are not real tensors.
        // Drop them when no layer runs plain attention, and add the latent
        // projections (`docs/DEEPSEEK2.md` shapes, from `MlaConfig`).
        if !arch.full_attention_layer_mask.iter().any(|&m| m <= 1) {
            let fhd = arch.full_head_dim;
            let generic = [
                (nh * fhd, h),
                (arch.num_full_kv_heads * fhd, h),
                (h, nh * fhd),
            ];
            out.retain(|&(r, c)| !generic.contains(&(r as i64, c as i64)));
        }
        let m = &arch.mla;
        let qk = m.nope_head_dim + m.rope_head_dim;
        let mut mla = vec![
            (m.kv_lora_rank + m.rope_head_dim, h),
            (nh * (m.nope_head_dim + m.v_head_dim), m.kv_lora_rank),
            (h, nh * m.v_head_dim),
        ];
        if m.q_lora_rank > 0 {
            mla.push((m.q_lora_rank, h));
            mla.push((nh * qk, m.q_lora_rank));
        } else {
            mla.push((nh * qk, h));
        }
        out.extend(mla.into_iter().map(|(r, c)| (r as usize, c as usize)));
    }
    if arch.num_dense_leading_layers > 0 && arch.dense_lead_intermediate_size > 0 {
        let f = arch.dense_lead_intermediate_size;
        out.push((f as usize, h as usize));
        out.push((h as usize, f as usize));
    }
    // The head: one GEMV per token, the largest one in most installs.
    out.push((arch.vocab_size as usize, h as usize));
    out.extend(routed_shapes(arch));
    if arch.has_linear_attention_layers() {
        // The gated-DeltaNet gate projections (`in_proj_a` and `in_proj_b`):
        // `num_v_heads` rows, so a handful of output rows against the full
        // hidden width. They are absent from `decode_int4_gemv_shapes`, and
        // they are exactly the shape where an 8-row verify pays launch and
        // occupancy latency instead of bandwidth (docs/KAPITSA_KERNELS.md,
        // lever 9), so a bench that skips them cannot see that cost.
        out.push((
            arch.linear_attention.num_v_heads as usize,
            arch.hidden_size as usize,
        ));
    }
    out.retain(|&(r, c)| r > 0 && c > 0 && c % 64 == 0);
    out.sort_unstable();
    out.dedup();
    out
}

/// A routed expert's gate|up and down shapes, for MoE installs.
pub fn routed_shapes(arch: &ArchConfig) -> Vec<(usize, usize)> {
    if arch.num_experts > 0 && arch.moe_intermediate_size > 0 {
        let (h, f) = (
            arch.hidden_size as usize,
            arch.moe_intermediate_size as usize,
        );
        vec![(2 * f, h), (h, f)]
    } else {
        Vec::new()
    }
}

/// Time every arm at every M for one shape.
pub fn run_shape(
    context: &mut MetalContext,
    label: &str,
    rows: usize,
    cols: usize,
    opts: &BenchOpts,
) -> Result<BenchResult, String> {
    let (ms, reps) = (&opts.rows[..], opts.reps);
    let weight_bytes = rows * cols / 2 + rows * cols / 64 * 4;
    let copies = copies_for(weight_bytes, 256 << 20);
    // A tiny shape's pass is a handful of microsecond dispatches, and the
    // command buffer's own fixed cost then dominates the time per dispatch
    // (the 48-row gate read 157 us against 46 us in the 64-copy bench). Walk
    // the copies more than once per pass until it holds at least this many
    // dispatches; the weights stay distinct within each walk.
    let sweeps = MIN_DISPATCHES_PER_PASS.div_ceil(copies).max(1);
    let mats: Vec<_> = (0..copies)
        .map(|i| {
            let (data, scales, biases) = int4_blob(rows, cols, 0xC0FF_EE00 + i as u64);
            (context.new_buffer_with_data(&data), scales, biases)
        })
        .collect();
    // A Steel tile needs the row count to fit its tile, and some of the
    // install's real projections do not (the 48-row gated-DeltaNet gates):
    // callers of the engine fall back to the scalar GEMM there. The bench
    // records the arm as unsupported for that shape and times the rest.
    let supported: Vec<bool> = {
        let (buffer, scales, biases) = &mats[0];
        let w = Int4ResidentMatrix {
            buffer,
            weights_offset: 0,
            scales_offset: *scales,
            biases_offset: *biases,
            rows,
            cols,
        };
        ARMS.iter()
            .map(|arm| match arm {
                Arm::Steel(tile) => steel_supports(&w, *tile),
                _ => true,
            })
            .collect()
    };
    let max_m = *ms.iter().max().unwrap_or(&1);
    let x = context.new_buffer_with_data(&f16_activations(max_m * cols, 7));
    let y = context.new_output_buffer((max_m * rows * 2) as u64);
    // Whether an arm has a point at M: the tile only where the engine would
    // route to it.
    let runs_at =
        |a: usize, m: usize| supported[a] && (ARMS[a] != Arm::Mma8 || m <= MMA8_BENCH_MAX_M);
    // (m, per-arm median ms, per-arm output hash, per-arm first-row hash)
    #[allow(clippy::type_complexity)]
    let mut table: Vec<(usize, Vec<f64>, Vec<Option<String>>, Vec<Option<String>>)> = Vec::new();
    for &m in ms {
        let mut samples = vec![Vec::new(); ARMS.len()];
        let mut hashes = vec![None; ARMS.len()];
        let mut row_hashes = vec![None; ARMS.len()];
        for rep in 0..=reps {
            for (a, &arm) in ARMS.iter().enumerate() {
                if !runs_at(a, m) {
                    continue;
                }
                let t = autorelease_pool(|| -> Result<f64, String> {
                    let pass = context.begin_pass();
                    for _ in 0..sweeps {
                        for (buffer, scales, biases) in &mats {
                            let w = Int4ResidentMatrix {
                                buffer,
                                weights_offset: 0,
                                scales_offset: *scales,
                                biases_offset: *biases,
                                rows,
                                cols,
                            };
                            encode_arm(context, &pass, arm, &w, &x, &y, m)?;
                        }
                    }
                    Ok(pass.commit_and_wait_with_gpu_time())
                })?;
                if rep > 0 {
                    samples[a].push(t * 1e3 / (copies * sweeps) as f64);
                }
                // The output of the last copy after the last repetition:
                // every copy runs the same seeded weights family, so the
                // bytes are a function of the arm's arithmetic alone.
                if opts.hash && rep == reps {
                    hashes[a] = Some(fnv_hex(&read_buffer_bytes(&y, 0, m * rows * 2)));
                    // Row 0 alone: the same bytes whatever M the arm ran at
                    // when a row's bits do not depend on its batch.
                    row_hashes[a] = Some(fnv_hex(&read_buffer_bytes(&y, 0, rows * 2)));
                }
            }
        }
        let medians: Vec<f64> = samples
            .iter_mut()
            .map(|s| if s.is_empty() { f64::NAN } else { median(s) })
            .collect();
        table.push((m, medians, hashes, row_hashes));
    }

    // One-row GEMV time is the c(M) unit.
    let one_row = table
        .iter()
        .find(|(m, _, _, _)| *m == 1)
        .map(|(_, t, _, _)| t[0])
        .unwrap_or(table[0].1[0] / table[0].0 as f64);
    let flops = |m: usize| 2.0 * m as f64 * rows as f64 * cols as f64;
    let mut arms = Vec::new();
    for (a, arm) in ARMS.iter().enumerate() {
        if !supported[a] {
            arms.push(ArmCurve::unsupported(
                &arm.label(),
                "this shape does not fit the tile",
            ));
            continue;
        }
        let points: Vec<Point> = table
            .iter()
            .filter(|(m, _, _, _)| runs_at(a, *m))
            .map(|(m, t, h, rh)| Point {
                x: *m as f64,
                ms: t[a],
                gb_s: Some(weight_bytes as f64 / (t[a] * 1e-3) / 1e9),
                tflops: Some(flops(*m) / (t[a] * 1e-3) / 1e12),
                c_of_m: Some(c_of_m(t[a], one_row, *m)),
                out_fnv: h[a].clone(),
                row_fnv: rh[a].clone(),
            })
            .collect();
        let fit = fit_affine(&points.iter().map(|p| (p.x, p.ms)).collect::<Vec<_>>());
        arms.push(ArmCurve {
            label: arm.label(),
            points,
            fit,
            unsupported: None,
        });
    }
    let reference = arms[0].fit;
    let break_even_rows: Vec<BreakEven> = arms
        .iter()
        .skip(1)
        .filter_map(|c| {
            Some(BreakEven {
                arm: c.label.clone(),
                against: arms[0].label.clone(),
                above_x: break_even(&reference?, &c.fit?),
            })
        })
        .collect();
    let mut subject = BTreeMap::new();
    subject.insert("rows".into(), json!(rows));
    subject.insert("cols".into(), json!(cols));
    subject.insert("weight_bytes".into(), json!(weight_bytes));
    subject.insert("copies".into(), json!(copies));
    subject.insert("one_row_gemv_ms".into(), json!(one_row));
    let mut notes = Vec::new();
    if ms.iter().any(|&m| m > MMA8_BENCH_MAX_M) {
        notes.push(format!(
            "mma8 is timed only up to M = {MMA8_BENCH_MAX_M}, the rows the engine routes to it"
        ));
    }
    Ok(BenchResult {
        bench: "int4_matmul".into(),
        case: label.to_string(),
        subject,
        x: Axis::new("m", "rows", "log2"),
        layout: LAYOUT.into(),
        arms,
        break_even: break_even_rows,
        notes,
    })
}

/// Encode M rows of one arm for one weight copy.
fn encode_arm(
    context: &mut MetalContext,
    pass: &gpu::PassEncoder,
    arm: Arm,
    w: &Int4ResidentMatrix<'_>,
    x: &gpu::MetalBuffer,
    y: &gpu::MetalBuffer,
    m: usize,
) -> Result<(), String> {
    let (rows, cols) = (w.rows, w.cols);
    let e = |r: Result<(), gpu::GpuError>| r.map_err(|e| e.to_string());
    match arm {
        Arm::GemvPerRow => {
            for r in 0..m {
                e(encode_dequant_int4_gemv_resident(
                    context,
                    pass,
                    w,
                    (x, (r * cols * 2) as u64),
                    (y, (r * rows * 2) as u64),
                ))?;
            }
        }
        Arm::Scalar | Arm::Mma8 => {
            let step = if arm == Arm::Scalar {
                MAX_BATCH_ROWS
            } else {
                INT4_MMA8_MAX_ROWS
            };
            let mut done = 0;
            while done < m {
                let take = (m - done).min(step);
                let (xo, yo) = ((done * cols * 2) as u64, (done * rows * 2) as u64);
                if arm == Arm::Scalar {
                    e(encode_dequant_int4_gemm_resident(
                        context,
                        pass,
                        w,
                        (x, xo),
                        (y, yo),
                        take,
                    ))?;
                } else {
                    e(encode_dequant_int4_mma8_resident(
                        context,
                        pass,
                        w,
                        (x, xo),
                        (y, yo),
                        take,
                    ))?;
                }
                done += take;
            }
        }
        Arm::Steel(tile) => {
            e(encode_dequant_int4_gemm_steel(
                context,
                pass,
                w,
                (x, 0),
                (y, 0),
                m,
                tile,
            ))?;
        }
    }
    Ok(())
}
