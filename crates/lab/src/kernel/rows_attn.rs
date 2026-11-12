//! Multi-row prefill attention at the install's full-attention shape: one
//! `attention_decode_gqa_rows` dispatch for M query rows against the per-row
//! GQA loop it replaced, over M, one case per context length.
//!
//! The port of `qwen38_prefill_rows_attention_time_by_context` in
//! `crates/gpu/tests/qwen38_attention_decode_probe.rs`, with the shape from
//! `ArchConfig`. It also sweeps the kernel's layout as arms (rows per
//! threadgroup, Q heads per simdgroup) through the launch-explicit
//! `encode_attention_decode_gqa_rows_with`: evaluator item L6
//! (`docs/MLX_SERVE_KERNELS.md`), the axis that found "one Q head per
//! simdgroup" by hand.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_attention_decode, encode_attention_decode_gqa_rows,
    encode_attention_decode_gqa_rows_with, read_buffer_bytes, AttentionScratch,
    DecodeAttentionKernel, GqaRowsScratch, MetalContext, GQA_ROWS_SCRATCH_CHUNKS,
};
use serde_json::json;

use super::attn_decode::shapes as attn_shapes;
use super::support::{f16_bytes, median};
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, Point};

/// The production micro-batch: the rows kernel never sees more.
pub const MAX_ROWS: usize = 64;
/// Chunks per row the sweep's scratch holds, so layout arms that want more
/// chunks than production still fit.
const SWEEP_CHUNKS: u32 = 8;
/// The production launch's unroll and tile (`gqa_rows_launch`).
const UNROLL: u32 = 3;
const TILE: u32 = 12;

pub struct RowsAttn;

/// A layout arm: rows per threadgroup and Q heads per simdgroup.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arm {
    /// One-row GQA kernel per row: the loop decode and verify still run.
    Loop,
    /// The production rows launch.
    Rows,
    Layout {
        rows: u32,
        heads: u32,
    },
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Loop => "per_row_loop".into(),
            Arm::Rows => "rows".into(),
            Arm::Layout { rows, heads } => format!("rows_r{rows}_h{heads}"),
        }
    }
}

fn arms(group: u32) -> Vec<Arm> {
    let mut out = vec![Arm::Loop, Arm::Rows];
    for rows in [1u32, 2, 4] {
        for heads in [1u32, group] {
            // (2, 1) is the production layout, already the `rows` arm.
            let dup = rows == 2 && heads == 1;
            let fits = group % heads == 0 && rows * (group / heads) <= 32;
            if !dup && fits && !out.contains(&Arm::Layout { rows, heads }) {
                out.push(Arm::Layout { rows, heads });
            }
        }
    }
    out
}

impl KernelBench for RowsAttn {
    fn name(&self) -> &'static str {
        "rows_attn"
    }

    fn describe(&self) -> &'static str {
        "multi-row prefill attention against the per-row GQA loop, with rows/heads layout arms, per context"
    }

    fn cases(&self, subject: &Subject, opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let arch = subject.arch()?;
        let full = attn_shapes(arch)
            .into_iter()
            .find(|(label, _)| *label == "full_attn")
            .map(|(_, s)| s)
            .ok_or("no full-attention layers")?;
        if !gpu::attention_decode_gqa_supported(full.head_dim, full.q_heads, full.kv_heads) {
            return Err(format!(
                "the rows kernel serves the GQA-shared shapes only (head_dim 128 or 256, group <= {}); this is head_dim {} with {} q over {} kv heads",
                gpu::MAX_GQA_GROUP, full.head_dim, full.q_heads, full.kv_heads
            ));
        }
        if full.sinks {
            return Err("the rows kernel takes no attention sinks".into());
        }
        // One case per context: attention cost moves with the span, so each
        // context is its own rows curve.
        Ok(opts
            .contexts
            .iter()
            .filter(|&&c| c as usize >= MAX_ROWS)
            .map(|&c| Case {
                label: format!("full_attn@{c}"),
                params: json!({"q_heads": full.q_heads, "kv_heads": full.kv_heads,
                               "head_dim": full.head_dim, "layers": full.layers,
                               "context": c}),
            })
            .collect())
    }

    fn run(
        &self,
        context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let u = |k: &str| case.params[k].as_u64().unwrap_or(0) as u32;
        let (nq, nkv, head_dim, layers) = (u("q_heads"), u("kv_heads"), u("head_dim"), u("layers"));
        let rows: Vec<usize> = opts
            .rows
            .iter()
            .copied()
            .filter(|&m| m <= MAX_ROWS)
            .collect();
        if rows.is_empty() {
            return Err(format!("rows_attn needs --rows values up to {MAX_ROWS}"));
        }
        run_at(
            context,
            nq,
            nkv,
            head_dim,
            layers as usize,
            u("context"),
            &rows,
            opts,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_at(
    context: &mut MetalContext,
    nq: u32,
    nkv: u32,
    head_dim: u32,
    layers: usize,
    seq: u32,
    rows: &[usize],
    opts: &BenchOpts,
) -> Result<BenchResult, String> {
    let max_m = *rows.iter().max().unwrap() as u32;
    if (seq as usize) < MAX_ROWS {
        return Err(format!("context {seq} is shorter than the micro-batch"));
    }
    let group = nq / nkv;
    let kv_width = (nkv * head_dim) as usize;
    let q_width = (nq * head_dim) as usize;
    let q = context.new_buffer_with_data(&f16_bytes(
        (0..max_m as usize * q_width).map(|i| ((i as f32) * 0.019).sin() * 0.5),
    ));
    let kv: Vec<_> = (0..layers)
        .map(|layer| {
            let phase = layer as f32 * 0.37;
            let k = context.new_buffer_with_data(&f16_bytes(
                (0..seq as usize * kv_width).map(|i| ((i as f32) * 0.0007 + phase).cos() * 0.3),
            ));
            let v = context.new_buffer_with_data(&f16_bytes(
                (0..seq as usize * kv_width)
                    .map(|i| 0.7 + ((i as f32) * 0.0011 + phase).sin() * 0.3),
            ));
            (k, v)
        })
        .collect();
    let out = context.new_output_buffer((max_m as usize * q_width * 2) as u64);
    let one = AttentionScratch::new(context, nq, head_dim)
        .with_decode_kernel(DecodeAttentionKernel::GqaShared);
    let prod = GqaRowsScratch::new(context, max_m, nq, head_dim, GQA_ROWS_SCRATCH_CHUNKS);
    let sweep = GqaRowsScratch::new(context, max_m, nq, head_dim, SWEEP_CHUNKS);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let arm_list = arms(group);
    let mut curves: Vec<Vec<Point>> = vec![Vec::new(); arm_list.len()];
    for &m in rows {
        let m32 = m as u32;
        // The batch ends at `seq`: row r attends over [0, first + r].
        let first = seq - m32;
        let mut samples = vec![Vec::new(); arm_list.len()];
        let mut hashes = vec![None; arm_list.len()];
        for rep in 0..=opts.reps {
            for (a, &arm) in arm_list.iter().enumerate() {
                let t = autorelease_pool(|| -> Result<f64, String> {
                    let pass = context.begin_pass();
                    let e = |r: Result<(), gpu::GpuError>| r.map_err(|e| e.to_string());
                    for (k, v) in &kv {
                        match arm {
                            Arm::Loop => {
                                for r in 0..m32 {
                                    let off = (r as usize * q_width * 2) as u64;
                                    e(encode_attention_decode(
                                        context,
                                        &pass,
                                        (&q, off),
                                        k,
                                        v,
                                        &one,
                                        (&out, off),
                                        head_dim,
                                        nq,
                                        nkv,
                                        first + r + 1,
                                        0,
                                        0,
                                        scale,
                                        None,
                                    ))?;
                                }
                            }
                            Arm::Rows => e(encode_attention_decode_gqa_rows(
                                context,
                                &pass,
                                (&q, 0),
                                k,
                                v,
                                &prod,
                                (&out, 0),
                                head_dim,
                                nq,
                                nkv,
                                first,
                                m32,
                                0,
                                0,
                                scale,
                            ))?,
                            Arm::Layout { rows: rpg, heads } => {
                                // The production chunk rule, with this
                                // layout's rows per threadgroup.
                                let groups = m32.div_ceil(rpg) * nkv;
                                let want = 64u32.div_ceil(groups).clamp(2, SWEEP_CHUNKS);
                                let fit = max_m * SWEEP_CHUNKS / m32;
                                e(encode_attention_decode_gqa_rows_with(
                                    context,
                                    &pass,
                                    (&q, 0),
                                    k,
                                    v,
                                    &sweep,
                                    (&out, 0),
                                    head_dim,
                                    nq,
                                    nkv,
                                    first,
                                    m32,
                                    0,
                                    0,
                                    scale,
                                    rpg,
                                    heads,
                                    want.min(fit).max(1),
                                    UNROLL,
                                    TILE,
                                ))?
                            }
                        }
                    }
                    Ok(pass.commit_and_wait_with_gpu_time())
                })?;
                if rep > 0 {
                    samples[a].push(t * 1e3);
                }
                if opts.hash && rep == opts.reps {
                    hashes[a] = Some(fnv_hex(&read_buffer_bytes(&out, 0, m * q_width * 2)));
                }
            }
        }
        let loop_ms = median(&mut samples[0].clone());
        for (a, s) in samples.iter_mut().enumerate() {
            let ms = median(s);
            let mut p = Point::new(m as f64, ms);
            // Against the loop at the same M: below 1 the batch kernel wins.
            p.c_of_m = Some(ms / loop_ms);
            p.out_fnv = hashes[a].take();
            curves[a].push(p);
        }
    }
    let arms_out: Vec<ArmCurve> = arm_list
        .iter()
        .zip(curves)
        .map(|(a, points)| ArmCurve {
            label: a.label(),
            points,
            fit: None,
            unsupported: None,
        })
        .collect();
    let mut subject = BTreeMap::new();
    for (k, v) in [
        ("q_heads", json!(nq)),
        ("kv_heads", json!(nkv)),
        ("head_dim", json!(head_dim)),
        ("layers", json!(layers)),
        ("context", json!(seq)),
    ] {
        subject.insert(k.to_string(), v);
    }
    Ok(BenchResult {
        bench: "rows_attn".into(),
        case: format!("full_attn@{seq}"),
        subject,
        x: Axis::new("m", "rows", "log2"),
        layout: "fp16_kv_linear".into(),
        arms: arms_out,
        break_even: Vec::new(),
        notes: vec![
            "c(M) here is each arm's time over the per-row loop's at the same M".into(),
            "not bit-exact against the loop: chunk boundaries differ (docs/SPEEDUP_PLAYBOOK.md)"
                .into(),
        ],
    })
}
