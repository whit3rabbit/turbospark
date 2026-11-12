//! INT8 affine matmul at the install's INT8 shapes (routers, shared-expert
//! gates, INT8 installs and the Qwen3-ASR decoder): M one-row GEMVs against
//! the batched INT8 GEMM, over a sweep of M.
//!
//! Same sample discipline as `int4_matmul`: distinct weight copies per
//! sample, arms alternated inside each repetition, the first discarded.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_dequant_int8_gemm_resident, encode_dequant_int8_gemv_resident,
    read_buffer_bytes, Int8ResidentMatrix, MetalContext, INT8_MAX_BATCH_ROWS,
};
use serde_json::json;

use super::fit::{break_even, c_of_m, fit_affine};
use super::support::{copies_for, f16_activations, int8_blob, median};
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, BreakEven, Point};

pub const LAYOUT: &str = "mlx_affine_int8_g64_bf16";

pub struct Int8Matmul;

pub const ARMS: [&str; 2] = ["gemv_per_row", "batched_gemm"];

impl KernelBench for Int8Matmul {
    fn name(&self) -> &'static str {
        "int8_matmul"
    }

    fn describe(&self) -> &'static str {
        "INT8 projections at the install's shapes (routers, shared gates): per-row GEMV against the batched GEMM over M"
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let arch = subject.arch()?;
        let (shapes, source) =
            super::matmul_shapes(subject, crate::inventory::DTYPE_INT8_AFFINE, || {
                arch.decode_int8_gemv_shapes()
                    .into_iter()
                    .filter(|(r, _)| *r > 0)
                    .map(|(r, c)| (r as usize, c as usize))
                    .collect()
            });
        if shapes.is_empty() {
            return Err("no INT8 affine matmul in this install".into());
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
                .ok_or_else(|| format!("int8_matmul case {} lacks {k}", case.label))
        };
        let (rows, cols) = (dim("rows")?, dim("cols")?);
        let mut r = run_shape(context, &case.label, rows, cols, opts)?;
        for k in ["roles", "count", "shape_source"] {
            if !case.params[k].is_null() {
                r.subject.insert(k.into(), case.params[k].clone());
            }
        }
        Ok(r)
    }
}

pub fn run_shape(
    context: &mut MetalContext,
    label: &str,
    rows: usize,
    cols: usize,
    opts: &BenchOpts,
) -> Result<BenchResult, String> {
    let weight_bytes = rows * cols + rows * cols / 64 * 4;
    let copies = copies_for(weight_bytes, 256 << 20);
    let mats: Vec<_> = (0..copies)
        .map(|i| {
            let (data, scales, biases) = int8_blob(rows, cols, 0x1A7E_0800 + i as u64);
            (context.new_buffer_with_data(&data), scales, biases)
        })
        .collect();
    let max_m = *opts.rows.iter().max().unwrap_or(&1);
    let x = context.new_buffer_with_data(&f16_activations(max_m * cols, 7));
    let y = context.new_output_buffer((max_m * rows * 2) as u64);
    let mut table: Vec<(usize, [f64; 2], [Option<String>; 2])> = Vec::new();
    for &m in &opts.rows {
        let mut samples = [Vec::new(), Vec::new()];
        let mut hashes = [None, None];
        for rep in 0..=opts.reps {
            for (a, sample) in samples.iter_mut().enumerate() {
                let t = autorelease_pool(|| -> Result<f64, String> {
                    let pass = context.begin_pass();
                    for (buffer, scales, biases) in &mats {
                        let w = Int8ResidentMatrix {
                            buffer,
                            weights_offset: 0,
                            scales_offset: *scales,
                            biases_offset: *biases,
                            rows,
                            cols,
                        };
                        let e = |r: Result<(), gpu::GpuError>| r.map_err(|e| e.to_string());
                        if a == 0 {
                            for r in 0..m {
                                e(encode_dequant_int8_gemv_resident(
                                    context,
                                    &pass,
                                    &w,
                                    (&x, (r * cols * 2) as u64),
                                    (&y, (r * rows * 2) as u64),
                                ))?;
                            }
                        } else {
                            let mut done = 0;
                            while done < m {
                                let take = (m - done).min(INT8_MAX_BATCH_ROWS);
                                e(encode_dequant_int8_gemm_resident(
                                    context,
                                    &pass,
                                    &w,
                                    (&x, (done * cols * 2) as u64),
                                    (&y, (done * rows * 2) as u64),
                                    take,
                                ))?;
                                done += take;
                            }
                        }
                    }
                    Ok(pass.commit_and_wait_with_gpu_time())
                })?;
                if rep > 0 {
                    sample.push(t * 1e3 / copies as f64);
                }
                if opts.hash && rep == opts.reps {
                    hashes[a] = Some(fnv_hex(&read_buffer_bytes(&y, 0, m * rows * 2)));
                }
            }
        }
        table.push((
            m,
            [median(&mut samples[0]), median(&mut samples[1])],
            hashes,
        ));
    }
    let one_row = table
        .iter()
        .find(|(m, _, _)| *m == 1)
        .map(|(_, t, _)| t[0])
        .unwrap_or(table[0].1[0] / table[0].0 as f64);
    let flops = |m: usize| 2.0 * m as f64 * rows as f64 * cols as f64;
    let arms: Vec<ArmCurve> = ARMS
        .iter()
        .enumerate()
        .map(|(a, label)| {
            let points: Vec<Point> = table
                .iter()
                .map(|(m, t, h)| Point {
                    x: *m as f64,
                    ms: t[a],
                    gb_s: Some(weight_bytes as f64 / (t[a] * 1e-3) / 1e9),
                    tflops: Some(flops(*m) / (t[a] * 1e-3) / 1e12),
                    c_of_m: Some(c_of_m(t[a], one_row, *m)),
                    out_fnv: h[a].clone(),
                    row_fnv: None,
                    j: None,
                })
                .collect();
            let fit = fit_affine(&points.iter().map(|p| (p.x, p.ms)).collect::<Vec<_>>());
            ArmCurve {
                label: label.to_string(),
                points,
                fit,
                unsupported: None,
            }
        })
        .collect();
    let break_even_rows = match (arms[0].fit, arms[1].fit) {
        (Some(r), Some(c)) => vec![BreakEven {
            arm: arms[1].label.clone(),
            against: arms[0].label.clone(),
            above_x: break_even(&r, &c),
        }],
        _ => Vec::new(),
    };
    let mut subject = BTreeMap::new();
    subject.insert("rows".into(), json!(rows));
    subject.insert("cols".into(), json!(cols));
    subject.insert("weight_bytes".into(), json!(weight_bytes));
    subject.insert("copies".into(), json!(copies));
    subject.insert("one_row_gemv_ms".into(), json!(one_row));
    Ok(BenchResult {
        bench: "int8_matmul".into(),
        case: label.to_string(),
        subject,
        x: Axis::new("m", "rows", "log2"),
        layout: LAYOUT.into(),
        arms,
        break_even: break_even_rows,
        notes: Vec::new(),
    })
}
