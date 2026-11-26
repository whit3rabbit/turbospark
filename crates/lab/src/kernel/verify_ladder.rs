//! Evaluator item L2 (`docs/MLX_SERVE_KERNELS.md`): the whole-forward
//! verify-width ladder. `int4_matmul` times each shape alone; a speculative
//! block is decided on the whole model, so this sums every projection the
//! install binds (the resident index's per-shape counts, routed experts at
//! top-k per MoE layer) at each verify width M, under the arm the engine
//! routes that width to, into one GEMM row per M.
//!
//! Routes, from `real_forward_dispatch::encode_gemv_any`, `encode_gemm_any`
//! and `encode_gemm_rows`: one row runs the GEMV, 2..=16 rows the scalar
//! batched GEMM, and a wider batch is that GEMM in `gpu::MAX_BATCH_ROWS`
//! slices; under `TURBOSPARK_INT4_MMA8=1` every shape of at least 4096 rows
//! runs the 8-row tile, in 8-row slices, at any M. Steel serves prefill
//! micro-batches of 32+ rows and never a verify, so it is not on the ladder.
//!
//! Widths above 16 price that sliced route; no speculative verify takes
//! them today (`ALLOWED_SPECULATION_BLOCKS` is 1..=15 and the Qwen batched
//! head calls `encode_gemm_any`, which refuses a batch above 16). The ladder
//! is what such a width would cost in GEMM time, not a width that runs.
//!
//! With `--verify-from EXPERIMENT`, the speculative runs in that ledger
//! supply measured verify ms per round at their width; the difference from
//! the ladder at the same width and flag is the non-GEMM marginal
//! (attention, GDN, glue), the decomposition that told mlx-serve its 16-row
//! marginal was 69% qmm.

use std::collections::BTreeMap;

use gpu::{MetalContext, INT4_MMA8_MAX_ROWS, MAX_BATCH_ROWS};
use serde_json::{json, Value};

use super::{int4_matmul, int8_matmul, BenchOpts, Case, KernelBench, Subject};
use crate::bench::{ArmCurve, Axis, BenchResult, Point};
use crate::inventory::{DTYPE_INT4_AFFINE, DTYPE_INT8_AFFINE};

pub struct VerifyLadder;

/// The shared tile's row threshold (`int4_mma8_min_rows`'s default).
pub const MMA8_MIN_ROWS: usize = 4096;

/// The widest logits a ladder width may imply: `m` rows of `vocab` entries,
/// F16 on the GPU and f32 on the host (6 bytes). A width past it would not
/// be a width anyone runs, so the ladder refuses it instead of timing it.
pub const LOGITS_BUDGET_BYTES: u64 = 256 << 20;

/// The default ladder: every width 1..=64, then 96.
pub fn ladder_widths() -> Vec<usize> {
    (1..=64).chain([96]).collect()
}

/// What `--rows` means here: the command's generic default sweep is
/// replaced by [`ladder_widths`], anything else is taken as written.
pub fn requested_widths(rows: &[usize]) -> Vec<usize> {
    let mut w = if rows == BenchOpts::default().rows.as_slice() {
        ladder_widths()
    } else {
        rows.to_vec()
    };
    w.sort_unstable();
    w.dedup();
    w
}

pub fn logits_bytes(vocab: usize, m: usize) -> u64 {
    m as u64 * vocab as u64 * 6
}

/// Split `widths` into those inside the logits budget and those refused.
pub fn admit_widths(widths: &[usize], vocab: usize) -> (Vec<usize>, Vec<usize>) {
    widths
        .iter()
        .copied()
        .partition(|&m| logits_bytes(vocab, m) <= LOGITS_BUDGET_BYTES)
}

/// A step of the measured curve where `slice` rows more start a new
/// dispatch: the cost added between the two swept widths around it.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// The last swept width of the old slice count.
    pub from: usize,
    /// The first swept width of the new one.
    pub to: usize,
    pub add_ms: f64,
    /// `add_ms` per added row.
    pub per_row_ms: f64,
}

/// Where the slice count (`ceil(m / slice)`) grows along `points`, which
/// are (width, ms) in ascending width.
pub fn step_edges(points: &[(usize, f64)], slice: usize) -> Vec<Step> {
    points
        .windows(2)
        .filter(|w| w[1].0.div_ceil(slice) > w[0].0.div_ceil(slice))
        .map(|w| {
            let add_ms = w[1].1 - w[0].1;
            Step {
                from: w[0].0,
                to: w[1].0,
                add_ms,
                per_row_ms: add_ms / (w[1].0 - w[0].0) as f64,
            }
        })
        .collect()
}

/// The median added ms per row between neighbours that stay inside one
/// slice: what a step edge is read against. `None` with no such pair.
pub fn within_slice_per_row(points: &[(usize, f64)], slice: usize) -> Option<f64> {
    let mut v: Vec<f64> = points
        .windows(2)
        .filter(|w| w[1].0.div_ceil(slice) == w[0].0.div_ceil(slice))
        .map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0) as f64)
        .collect();
    (!v.is_empty()).then(|| super::support::median(&mut v))
}

/// The arm label `int4_matmul` / `int8_matmul` time that the engine runs
/// for a `rows`-row matrix at verify width `m`.
pub fn routed_arm(dtype: u8, rows: usize, m: usize, mma8: bool) -> &'static str {
    match (dtype, m) {
        (DTYPE_INT8_AFFINE, 1) => "gemv_per_row",
        (DTYPE_INT8_AFFINE, _) => "batched_gemm",
        _ if mma8 && rows >= MMA8_MIN_ROWS => "mma8",
        (_, 1) => "gemv_per_row",
        _ => "scalar_gemm",
    }
}

/// One shape's role in the ladder.
#[derive(Debug, Clone)]
struct Term {
    dtype: u8,
    rows: usize,
    cols: usize,
    /// Times one forward runs it.
    count: usize,
    label: String,
}

fn terms(subject: &Subject) -> Result<Vec<Term>, String> {
    let arch = subject.arch()?;
    // The config fallback has no per-shape counts, and a ladder without
    // counts is not a whole-model sum.
    if let Err(e) = &subject.inventory {
        return Err(format!(
            "the ladder needs per-shape counts from the resident index: {e}"
        ));
    }
    let mut out = Vec::new();
    for dtype in [DTYPE_INT4_AFFINE, DTYPE_INT8_AFFINE] {
        let (shapes, _) = super::matmul_shapes(subject, dtype, Vec::new);
        for s in shapes {
            out.push(Term {
                dtype,
                rows: s.rows,
                cols: s.cols,
                count: s.count,
                label: s.roles.join("+"),
            });
        }
    }
    // Routed experts: top-k experts per MoE layer per row. At M rows a
    // layer binds at most min(M*k, E) distinct experts; the ladder counts
    // M*k expert-rows through the per-expert GEMVs as an upper end, and the
    // record says so.
    let moe_layers = (arch.num_layers - arch.num_dense_leading_layers).max(0) as usize;
    if arch.num_experts > 0 && moe_layers > 0 {
        for (rows, cols) in int4_matmul::routed_shapes(arch) {
            out.push(Term {
                dtype: DTYPE_INT4_AFFINE,
                rows,
                cols,
                count: moe_layers * arch.top_k_experts.max(1) as usize,
                label: "routed_expert".into(),
            });
        }
    }
    out.retain(|t| t.cols % 64 == 0 && t.count > 0);
    Ok(out)
}

impl KernelBench for VerifyLadder {
    fn name(&self) -> &'static str {
        "verify_ladder"
    }

    fn describe(&self) -> &'static str {
        "L2: whole-model GEMM time per verify width, summed over the inventory under the engine's routes"
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let t = terms(subject)?;
        if t.is_empty() {
            return Err("no INT4 or INT8 affine projections to sum".into());
        }
        Ok(vec![Case {
            label: "whole_model".into(),
            params: json!({"terms": t.len()}),
        }])
    }

    fn run(
        &self,
        context: &mut MetalContext,
        subject: &Subject,
        _case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let terms = terms(subject)?;
        let vocab = subject.arch()?.vocab_size.max(0) as usize;
        let (widths, refused) = admit_widths(&requested_widths(&opts.rows), vocab);
        for &m in &refused {
            crate::say!(
                "verify_ladder: width {m} refused: {} MiB of logits (vocab {vocab}) is past the {} MiB budget",
                logits_bytes(vocab, m) >> 20,
                LOGITS_BUDGET_BYTES >> 20
            );
        }
        if !widths.contains(&1) {
            return Err("the ladder needs width 1 in --rows (its c(M) unit)".into());
        }
        let shape_opts = BenchOpts {
            rows: widths.clone(),
            ..opts.clone()
        };
        // Time each distinct (dtype, rows, cols) once, every arm.
        let mut timed: BTreeMap<(u8, usize, usize), BenchResult> = BTreeMap::new();
        for t in &terms {
            let key = (t.dtype, t.rows, t.cols);
            if timed.contains_key(&key) {
                continue;
            }
            let label = format!("{}x{}", t.rows, t.cols);
            let r = if t.dtype == DTYPE_INT8_AFFINE {
                int8_matmul::run_shape(context, &label, t.rows, t.cols, &shape_opts)?
            } else {
                // Only the arms the two routes read: a 65-width sweep of
                // all six arms would spend its time on the ones it drops.
                let shared = t.rows >= MMA8_MIN_ROWS;
                let want = |arm: int4_matmul::Arm, m: usize| match arm {
                    int4_matmul::Arm::GemvPerRow => m == 1,
                    int4_matmul::Arm::Scalar => m > 1,
                    int4_matmul::Arm::Mma8 => shared,
                    _ => false,
                };
                int4_matmul::run_shape_where(
                    context,
                    &label,
                    t.rows,
                    t.cols,
                    &shape_opts,
                    &want,
                    true,
                )?
            };
            eprintln!(
                "  verify_ladder: timed {} {label}",
                crate::inventory::dtype_name(t.dtype)
            );
            timed.insert(key, r);
        }
        let ms_at = |t: &Term, arm: &str, m: usize| -> Option<f64> {
            timed[&(t.dtype, t.rows, t.cols)]
                .arm(arm)?
                .points
                .iter()
                .find(|p| p.x == m as f64)
                .map(|p| p.ms)
        };
        let ladder = |m: usize, mma8: bool| -> Result<f64, String> {
            let mut sum = 0.0;
            for t in &terms {
                let arm = routed_arm(t.dtype, t.rows, m, mma8);
                let ms = ms_at(t, arm, m)
                    .ok_or_else(|| format!("{} has no {arm} point at M={m}", t.label))?;
                sum += ms * t.count as f64;
            }
            Ok(sum)
        };
        let mut arms = Vec::new();
        let mut totals: BTreeMap<&str, BTreeMap<usize, f64>> = BTreeMap::new();
        for (label, mma8) in [("default_route", false), ("mma8_route", true)] {
            let one = ladder(1, mma8)?;
            let mut points = Vec::new();
            for &m in &widths {
                let ms = ladder(m, mma8)?;
                totals.entry(label).or_default().insert(m, ms);
                let mut p = Point::new(m as f64, ms);
                p.c_of_m = Some(ms / (one * m as f64));
                points.push(p);
            }
            arms.push(ArmCurve {
                label: label.into(),
                points,
                fit: None,
                unsupported: None,
            });
        }
        // What M separate decode steps would pay: the no-batching line.
        let one = ladder(1, false)?;
        arms.push(ArmCurve {
            label: "m_decode_steps".into(),
            points: widths
                .iter()
                .map(|&m| {
                    let mut p = Point::new(m as f64, one * m as f64);
                    p.c_of_m = Some(1.0);
                    p
                })
                .collect(),
            fit: None,
            unsupported: None,
        });
        // Each term's share of the default ladder at the widest width.
        let top = *widths.iter().max().unwrap();
        let top_total = totals["default_route"][&top];
        let shares: Vec<Value> = terms
            .iter()
            .filter_map(|t| {
                let ms = ms_at(t, routed_arm(t.dtype, t.rows, top, false), top)? * t.count as f64;
                Some(json!({"term": t.label, "rows": t.rows, "cols": t.cols,
                            "dtype": crate::inventory::dtype_name(t.dtype),
                            "count": t.count, "ms": ms, "share": ms / top_total}))
            })
            .collect();
        // Where the sliced route steps, from the measured sums.
        let curve = |route: &str| -> Vec<(usize, f64)> {
            totals[route].iter().map(|(&m, &ms)| (m, ms)).collect()
        };
        let mut edges = Vec::new();
        let mut edge_notes = Vec::new();
        for (route, slice) in [
            ("default_route", MAX_BATCH_ROWS),
            ("mma8_route", INT4_MMA8_MAX_ROWS),
        ] {
            let points = curve(route);
            let within = within_slice_per_row(&points, slice);
            let steps = step_edges(&points, slice);
            if !steps.is_empty() {
                edge_notes.push(format!(
                    "{route} step edges ({slice}-row slices), added ms: {}; median ms per row inside a slice: {}",
                    steps
                        .iter()
                        .map(|s| format!("{}->{} +{:.3}", s.from, s.to, s.add_ms))
                        .collect::<Vec<_>>()
                        .join(", "),
                    within.map_or("unmeasured".into(), |w| format!("{w:.3}"))
                ));
            }
            for s in steps {
                edges.push(json!({"route": route, "slice": slice, "from": s.from,
                                  "to": s.to, "add_ms": s.add_ms,
                                  "per_row_ms": s.per_row_ms,
                                  "within_slice_per_row_ms": within}));
            }
        }
        let mut subject_map = BTreeMap::new();
        subject_map.insert("step_edges".into(), json!(edges));
        subject_map.insert("vocab".into(), json!(vocab));
        subject_map.insert("refused_widths".into(), json!(refused));
        subject_map.insert("logits_budget_bytes".into(), json!(LOGITS_BUDGET_BYTES));
        subject_map.insert("terms".into(), json!(terms.len()));
        subject_map.insert(format!("shares_at_m{top}"), json!(shares));
        let mut notes = vec![
            "GEMM time only: attention, GDN, norms and glue are not on the ladder".into(),
            "routed experts count M*top_k expert rows per MoE layer, an upper end".into(),
            "c(M) is the ladder at M over M one-row ladders".into(),
            format!(
                "widths above {MAX_BATCH_ROWS} price the sliced GEMM route; no speculative verify accepts them today"
            ),
        ];
        notes.extend(edge_notes);
        if !refused.is_empty() {
            notes.push(format!(
                "widths {refused:?} refused: logits past {} MiB",
                LOGITS_BUDGET_BYTES >> 20
            ));
        }
        if let Some(exp) = &opts.verify_from {
            let measured = measured_verify(exp, &totals)?;
            notes.push(format!(
                "{} measured verify widths from {exp}; non_gemm_ms = verify ms per round minus the ladder",
                measured.len()
            ));
            subject_map.insert("measured_verify".into(), json!(measured));
        }
        Ok(BenchResult {
            bench: "verify_ladder".into(),
            case: "whole_model".into(),
            subject: subject_map,
            x: Axis::new("m", "rows", "linear"),
            layout: "per term (int4_matmul, int8_matmul)".into(),
            arms,
            break_even: Vec::new(),
            notes,
        })
    }
}

/// Verify ms per round from an experiment's speculative runs, set against
/// the ladder at the same width and flag. A round of block k verifies k + 1
/// rows. The first `[spec-stats]` call of a run is its warm pass.
fn measured_verify(
    experiment: &str,
    totals: &BTreeMap<&str, BTreeMap<usize, f64>>,
) -> Result<Vec<Value>, String> {
    let lab = crate::paths::lab_dir();
    let ledger = crate::ledger::Ledger::open(&crate::paths::experiment_dir(&lab, experiment)?)?;
    let mut out = Vec::new();
    for r in ledger.of_kind("run")? {
        if r["eligible"] != true || r["warmup"] == true {
            continue;
        }
        let Some(block) = r["speculation"]["block"].as_u64() else {
            continue;
        };
        let width = block as usize + 1;
        let verify: Vec<f64> = r["spec_stats"]
            .as_array()
            .into_iter()
            .flatten()
            .skip(1)
            .filter_map(|c| c["ms_per_round"]["verify"].as_f64())
            .collect();
        if verify.is_empty() {
            continue;
        }
        let mma8 = r["arm_env"]["TURBOSPARK_INT4_MMA8"] == "1";
        let route = if mma8 { "mma8_route" } else { "default_route" };
        let gemm = totals.get(route).and_then(|t| t.get(&width)).copied();
        let v = crate::stats::median(&verify);
        out.push(
            json!({"run": r["id"], "arm": r["arm"], "width": width, "mma8": mma8,
                        "verify_ms": v, "gemm_ms": gemm,
                        "non_gemm_ms": gemm.map(|g| v - g)}),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_sweep_becomes_the_ladder_and_a_named_one_is_kept() {
        let d = requested_widths(&BenchOpts::default().rows);
        assert_eq!(d.len(), 65);
        assert_eq!((d[0], d[63], d[64]), (1, 64, 96));
        assert_eq!(requested_widths(&[33, 1, 17, 1]), [1, 17, 33]);
    }

    #[test]
    fn widths_past_the_logits_budget_are_refused() {
        // 1 MiB-entry vocab: 6 MiB per row, 42 rows fit in 256 MiB.
        let vocab = 1 << 20;
        let (kept, refused) = admit_widths(&[1, 42, 43, 96], vocab);
        assert_eq!(kept, [1, 42]);
        assert_eq!(refused, [43, 96]);
        // A Qwen3.8-sized vocabulary keeps the whole default ladder.
        assert!(admit_widths(&ladder_widths(), 248_320).1.is_empty());
        assert!(!admit_widths(&[512], 201_024).1.is_empty());
    }

    #[test]
    fn step_edges_sit_where_a_new_slice_starts() {
        // 1 ms per slice plus 0.1 ms per row.
        let pts: Vec<(usize, f64)> = (1..=40usize)
            .map(|m| (m, m.div_ceil(16) as f64 + 0.1 * m as f64))
            .collect();
        let e = step_edges(&pts, 16);
        assert_eq!(
            e.iter().map(|s| (s.from, s.to)).collect::<Vec<_>>(),
            [(16, 17), (32, 33)]
        );
        assert!((e[0].add_ms - 1.1).abs() < 1e-9, "{}", e[0].add_ms);
        assert!((within_slice_per_row(&pts, 16).unwrap() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn routes_follow_the_dispatcher() {
        assert_eq!(
            routed_arm(DTYPE_INT4_AFFINE, 17408, 1, false),
            "gemv_per_row"
        );
        assert_eq!(
            routed_arm(DTYPE_INT4_AFFINE, 17408, 3, false),
            "scalar_gemm"
        );
        assert_eq!(routed_arm(DTYPE_INT4_AFFINE, 17408, 1, true), "mma8");
        assert_eq!(routed_arm(DTYPE_INT4_AFFINE, 1024, 3, true), "scalar_gemm");
        assert_eq!(routed_arm(DTYPE_INT8_AFFINE, 256, 3, true), "batched_gemm");
        // Past one slice the route is the same kernel, sliced.
        assert_eq!(
            routed_arm(DTYPE_INT4_AFFINE, 17408, 65, false),
            "scalar_gemm"
        );
        assert_eq!(routed_arm(DTYPE_INT4_AFFINE, 17408, 96, true), "mma8");
    }
}
