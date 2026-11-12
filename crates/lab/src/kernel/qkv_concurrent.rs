//! The attention q/k/v projection triple in a serial pass against the same
//! three dispatches in a concurrent compute encoder
//! (`MetalContext::begin_pass_concurrent_labeled`, `docs/KAPITSA_KERNELS.md`
//! item 2).
//!
//! This prices a mode that has no runtime switch yet: the concurrent encoder
//! is plumbed and semantically tested (`concurrent_pass_semantics.rs`) but no
//! decode flow uses it. The three GEMVs share one input and write three
//! separate outputs, so they need no barrier between them. One point only
//! (a decode token's triple, one row); `c_of_m` here is the arm's time over
//! the serial arm's, so a value under 1 is what concurrency buys per layer.
//! The per-forward count is the install's full-attention layer count.
//!
//! The triple comes from the resident index's roles (`self_attn.q_proj`,
//! `self_attn.k_proj`, `self_attn.v_proj`, INT4 affine), so an install
//! without a plain attention triple says why it cannot run. On an install
//! with a drafter the index also counts the drafter's projections; time the
//! plain install for a trunk-only count.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_dequant_int4_gemv_resident, read_buffer_bytes, Int4ResidentMatrix,
    MetalBuffer, MetalContext,
};
use serde_json::json;

use super::fit::fit_affine;
use super::support::{copies_for, f16_activations, int4_blob, median};
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, Point, INT4_AFFINE_LAYOUT};
use crate::inventory::DTYPE_INT4_AFFINE;

pub const NAME: &str = "qkv_concurrent";
pub const ARMS: [&str; 2] = ["serial", "concurrent"];
/// Triples a timed pass holds at least, so the command buffer's fixed cost
/// does not dominate a few-microsecond dispatch.
const MIN_TRIPLES_PER_PASS: usize = 24;

pub struct QkvConcurrent;

/// (q rows, k and v rows, cols, layers) of the install's attention triple.
pub fn triple(subject: &Subject) -> Result<(usize, usize, usize, usize), String> {
    let inv = subject.inventory.as_ref().map_err(|e| e.clone())?;
    let find = |role: &str| {
        inv.iter()
            .find(|s| s.dtype == DTYPE_INT4_AFFINE && s.roles.iter().any(|r| r.ends_with(role)))
    };
    let (q, k, v) = (
        find("self_attn.q_proj").ok_or("no INT4 self_attn.q_proj in the resident index")?,
        find("self_attn.k_proj").ok_or("no INT4 self_attn.k_proj in the resident index")?,
        find("self_attn.v_proj").ok_or("no INT4 self_attn.v_proj in the resident index")?,
    );
    if k.rows != v.rows || q.cols != k.cols || k.cols != v.cols {
        return Err(format!(
            "q {}x{}, k {}x{} and v {}x{} are not a shared-input triple",
            q.rows, q.cols, k.rows, k.cols, v.rows, v.cols
        ));
    }
    // One q projection per full-attention layer (the drafter's, if any, are
    // counted too; see the module doc).
    Ok((q.rows, k.rows, q.cols, q.count.max(1)))
}

impl KernelBench for QkvConcurrent {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> &'static str {
        "attention q/k/v projection triple: serial pass against a concurrent encoder (no barrier, shared input)"
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let (q, k, cols, layers) = triple(subject)?;
        if cols % 64 != 0 {
            return Err(format!("width {cols} is not a multiple of 64"));
        }
        Ok(vec![Case {
            label: "qkv".into(),
            params: json!({"q_rows": q, "kv_rows": k, "cols": cols, "count": layers}),
        }])
    }

    fn run(
        &self,
        context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let dim = |key: &str| {
            case.params[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("qkv_concurrent case lacks {key}"))
        };
        run_triple(
            context,
            dim("q_rows")?,
            dim("kv_rows")?,
            dim("cols")?,
            dim("count")?,
            opts,
        )
    }
}

/// One weight copy of the triple and its three outputs.
struct Copy {
    w: Vec<(MetalBuffer, u64, u64)>,
    y: Vec<MetalBuffer>,
}

fn encode_triple(
    context: &mut MetalContext,
    pass: &gpu::PassEncoder,
    c: &Copy,
    dims: &[(usize, usize); 3],
    x: &MetalBuffer,
) -> Result<(), String> {
    for (i, (rows, cols)) in dims.iter().enumerate() {
        let (buffer, scales, biases) = &c.w[i];
        let w = Int4ResidentMatrix {
            buffer,
            weights_offset: 0,
            scales_offset: *scales,
            biases_offset: *biases,
            rows: *rows,
            cols: *cols,
        };
        encode_dequant_int4_gemv_resident(context, pass, &w, (x, 0), (&c.y[i], 0))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn run_triple(
    context: &mut MetalContext,
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    layers: usize,
    opts: &BenchOpts,
) -> Result<BenchResult, String> {
    let dims = [(q_rows, cols), (kv_rows, cols), (kv_rows, cols)];
    let bytes: usize = dims.iter().map(|(r, c)| r * c / 2 + r * c / 64 * 4).sum();
    let copies = copies_for(bytes, 256 << 20);
    let sweeps = MIN_TRIPLES_PER_PASS.div_ceil(copies).max(1);
    let pool: Vec<Copy> = (0..copies)
        .map(|ci| Copy {
            w: dims
                .iter()
                .enumerate()
                .map(|(i, (r, c))| {
                    let (data, s, b) = int4_blob(*r, *c, 0xA11CE + (ci * 3 + i) as u64);
                    (context.new_buffer_with_data(&data), s, b)
                })
                .collect(),
            y: dims
                .iter()
                .map(|(r, _)| context.new_output_buffer((r * 2) as u64))
                .collect(),
        })
        .collect();
    let x = context.new_buffer_with_data(&f16_activations(cols, 7));

    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ARMS.len()];
    let mut hashes: Vec<Option<String>> = vec![None; ARMS.len()];
    for rep in 0..=opts.reps {
        for (a, arm) in ARMS.iter().enumerate() {
            let t = autorelease_pool(|| -> Result<f64, String> {
                let pass = if *arm == "concurrent" {
                    context.begin_pass_concurrent_labeled("qkv")
                } else {
                    context.begin_pass()
                };
                for _ in 0..sweeps {
                    for c in &pool {
                        encode_triple(context, &pass, c, &dims, &x)?;
                    }
                }
                Ok(pass.commit_and_wait_with_gpu_time())
            })?;
            if rep > 0 {
                samples[a].push(t * 1e3 / (copies * sweeps) as f64);
            }
            if opts.hash && rep == opts.reps {
                // The q output of the last copy: bits must not depend on the
                // encoder kind.
                let last = &pool[copies - 1];
                hashes[a] = Some(fnv_hex(&read_buffer_bytes(&last.y[0], 0, q_rows * 2)));
            }
        }
    }
    let medians: Vec<f64> = samples.iter_mut().map(|s| median(s)).collect();
    let serial = medians[0];
    let mut arms = Vec::new();
    for (a, arm) in ARMS.iter().enumerate() {
        let mut p = Point::new(1.0, medians[a]);
        p.c_of_m = Some(medians[a] / serial);
        p.out_fnv = hashes[a].clone();
        let points = vec![p];
        let fit = fit_affine(&[]);
        arms.push(ArmCurve {
            label: arm.to_string(),
            points,
            fit,
            unsupported: None,
        });
    }
    let mut subject = BTreeMap::new();
    subject.insert("q_rows".into(), json!(q_rows));
    subject.insert("kv_rows".into(), json!(kv_rows));
    subject.insert("cols".into(), json!(cols));
    subject.insert("count".into(), json!(layers));
    subject.insert("copies".into(), json!(copies));
    subject.insert("serial_triple_ms".into(), json!(serial));
    Ok(BenchResult {
        bench: NAME.into(),
        case: "qkv".into(),
        subject,
        x: Axis::new("m", "rows", "log2"),
        layout: INT4_AFFINE_LAYOUT.into(),
        arms,
        break_even: Vec::new(),
        notes: vec![
            "c(M) here is each arm's time over the serial arm's: under 1, the concurrent encoder pays".into(),
            "one point (a decode token's triple): the projection prices a count of layers x the per-triple delta".into(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_names_are_serial_then_concurrent() {
        assert_eq!(ARMS, ["serial", "concurrent"]);
    }
}
