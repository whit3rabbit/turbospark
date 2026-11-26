//! The routed-expert pair at the install's MoE shape: M sequential decode
//! pairs (`encode_moe_phase1`/`2`, what M tokens cost one at a time) against
//! one route-list pair (`encode_moe_prefill_phase1` and
//! `_phase2_fused`), over M tokens.
//!
//! The port of the affine arm of `crates/gpu/tests/moe_prefill_batch_bench.rs`
//! with D, F and top-k from `ArchConfig`. `--routes` picks where the
//! unions come from (`super::routes`): the default `uniform` models the
//! expected number of distinct experts, `E * (1 - (1 - k/E)^M)`, capped at
//! the binding limit, and the record says so (real routing is skewed, so the
//! batched arm's cost is then an upper end); `trace:<capture.json>` takes
//! the median-union window of a router trace, `zipf:<s>` of a seeded draw.
//!
//! The route-list pair runs twice: with phase 1's grid in `(token, rank)`
//! order, and with it sorted by expert. Host work (bind, route build, the
//! sort) is outside every arm, as in the test: it belongs to the end-to-end
//! A/B. Neither phase takes a rows-per-threadgroup or threadgroup-size
//! parameter in the gpu crate, so there are no tile arms here.

use std::collections::BTreeMap;

use gpu::{
    autorelease_pool, encode_moe_phase1, encode_moe_phase2, encode_moe_prefill_phase1,
    encode_moe_prefill_phase2_fused, read_buffer_bytes, MetalBuffer, MetalContext,
    MoeExpertOffsets, MoePrefillRoute, RoutedBlobsBuffer, RoutedBlobsWideBuffer,
    MAX_PREFILL_EXPERT_BINDINGS, MAX_STREAMED_EXPERTS, PHASE2_FIXED_SLOTS,
};
use serde_json::json;

use super::routes::{self, RouteSpec, Window};
use super::support::{f16_bytes, median};
use super::tile_pin::{self, Bucket};
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, BreakEven, Point};

pub struct MoeRouted;

/// Expected distinct experts for `m` tokens of top-`k` routing over `e`
/// experts, if routing were uniform.
pub fn uniform_union(e: usize, k: usize, m: usize) -> usize {
    let p = 1.0 - (1.0 - k as f64 / e as f64).powi(m as i32);
    ((e as f64 * p).round() as usize).clamp(k.min(e), e)
}

pub const ARMS: [&str; 3] = [
    "sequential_decode_pairs",
    "route_list_pair",
    "route_list_pair_sorted",
];

/// The modeled batch of the uniform source: `union` slots dealt round-robin.
fn uniform_window(e: usize, k: usize, m: usize) -> Window {
    let union = uniform_union(e, k, m);
    Window {
        slots: (0..m)
            .map(|t| (0..k).map(|r| ((t * k + r) % union) as u32).collect())
            .collect(),
        union,
    }
}

/// The batch to time at each M and what the source's windows looked like.
/// `None` for an M the source cannot serve, with the reason in the record.
fn plan(
    spec: &RouteSpec,
    trace: Option<&routes::Trace>,
    e: usize,
    k: usize,
    m: usize,
) -> (Option<Window>, serde_json::Value) {
    let (window, mut record) = match (spec, trace) {
        (RouteSpec::Uniform, _) => {
            let w = uniform_window(e, k, m);
            let mut hist = BTreeMap::new();
            routes::rows_per_expert(&w, &mut hist);
            let record = json!({"m": m, "union": w.union, "rows_per_expert": hist});
            (Some(w), record)
        }
        (_, Some(t)) => match routes::windows(t, m, MAX_PREFILL_EXPERT_BINDINGS) {
            Some(w) => {
                let record = json!({"m": m, "union": w.typical.union, "windows": w.count,
                    "union_min": w.union_min, "union_median": w.union_median,
                    "union_max": w.union_max, "windows_over_cap": w.over_cap,
                    "rows_per_expert": w.rows_per_expert});
                (Some(w.typical), record)
            }
            None => (None, json!({"m": m, "skipped": "no layer holds m passes"})),
        },
        (_, None) => (None, json!({"m": m, "skipped": "no trace"})),
    };
    if window
        .as_ref()
        .is_some_and(|w| w.union > MAX_PREFILL_EXPERT_BINDINGS)
    {
        record["skipped"] = json!("union past the binding cap");
        return (None, record);
    }
    (window, record)
}

/// Gate, up and down planes per expert, as the parity test builds a blob:
/// per role the packed INT4 run, then BF16 scales, then BF16 biases.
fn blob_layout(d: usize, f: usize) -> (MoeExpertOffsets, usize) {
    let mut at = 0usize;
    let mut place = |len: usize| {
        let off = at;
        at += len;
        off as u32
    };
    let (gp, gs) = (f * d / 2, f * (d / 64) * 2);
    let (dp, ds) = (d * f / 2, d * (f / 64) * 2);
    let gate_w = place(gp);
    let gate_s = place(gs);
    let gate_b = place(gs);
    let up_w = place(gp);
    let up_s = place(gs);
    let up_b = place(gs);
    let down_w = place(dp);
    let down_s = place(ds);
    let down_b = place(ds);
    (
        MoeExpertOffsets {
            gate_w,
            gate_s,
            gate_b,
            up_w,
            up_s,
            up_b,
            down_w,
            down_s,
            down_b,
        },
        at,
    )
}

/// Why this top-k cannot run, or `None`. Both pairs must take it: the
/// decode pair binds at most `MAX_STREAMED_EXPERTS`, and the route-list
/// pair asserts `top_k <= PHASE2_FIXED_SLOTS` (a panic, not an error), so
/// qwen4exp's top-10 has to be refused here before `run` reaches it.
fn top_k_refusal(k: usize) -> Option<String> {
    if k == 0 || k > MAX_STREAMED_EXPERTS {
        return Some(format!(
            "top-k {k} outside the decode pair's 1..={MAX_STREAMED_EXPERTS}"
        ));
    }
    if k > PHASE2_FIXED_SLOTS {
        return Some(format!(
            "top-k {k} above the route-list pair's {PHASE2_FIXED_SLOTS} fixed slots"
        ));
    }
    None
}

fn routed_scheme(subject: &Subject) -> Option<String> {
    let text = std::fs::read_to_string(subject.dir.join("manifest.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let q = &v["quant"]["routedExpert"];
    Some(format!(
        "{}{}g{}",
        q["scheme"].as_str()?,
        q["weightBits"].as_u64()?,
        q["groupSize"].as_u64()?
    ))
}

impl KernelBench for MoeRouted {
    fn name(&self) -> &'static str {
        "moe_routed"
    }

    fn describe(&self) -> &'static str {
        "routed-expert pair: M sequential decode pairs against one route-list pair (pair-order and expert-sorted grids), over M tokens; --routes uniform|zipf:<s>|trace:<capture.json>"
    }

    fn flags(&self) -> &'static [&'static str] {
        &["routes"]
    }

    fn cases(&self, subject: &Subject, _opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let arch = subject.arch()?;
        if arch.num_experts == 0 || arch.moe_intermediate_size == 0 {
            return Err("the install routes no experts".into());
        }
        let (d, f, k) = (
            arch.hidden_size as usize,
            arch.moe_intermediate_size as usize,
            arch.top_k_experts as usize,
        );
        if d % 64 != 0 || f % 64 != 0 {
            return Err(format!("D={d} or F={f} is not a multiple of 64"));
        }
        if let Some(why) = top_k_refusal(k) {
            return Err(why);
        }
        // The synthetic blob is MLX affine INT4; an install whose experts are
        // MXFP4 or GGUF runs other kernels, which this bench does not time.
        match routed_scheme(subject).as_deref() {
            None | Some("affine4g64") => {}
            Some(other) => {
                return Err(format!(
                    "routed experts are {other}; this bench synthesizes affine INT4 group 64"
                ))
            }
        }
        Ok(vec![Case {
            label: format!("d{d}_f{f}_k{k}"),
            params: json!({"d": d, "f": f, "top_k": k, "experts": arch.num_experts,
                           "silu": arch.hidden_activation.contains("silu")}),
        }])
    }

    fn run(
        &self,
        context: &mut MetalContext,
        _subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let u = |k: &str| case.params[k].as_u64().unwrap_or(0) as usize;
        let (d, f, k, e) = (u("d"), u("f"), u("top_k"), u("experts"));
        let silu = case.params["silu"] == true;
        let routes_arg = case.params["routes"].as_str().unwrap_or("uniform");
        let spec = RouteSpec::parse(routes_arg)?;
        let mut capture_fnv = None;
        let trace = match &spec {
            RouteSpec::Uniform => None,
            RouteSpec::Zipf(s) => Some(routes::zipf_default(e, k, *s)),
            RouteSpec::Trace(path) => {
                let text = std::fs::read_to_string(path)
                    .map_err(|err| format!("{}: {err}", path.display()))?;
                capture_fnv = Some(fnv_hex(text.as_bytes()));
                Some(routes::parse_capture(&text, e, k)?)
            }
        };
        let (offsets, blob_len) = blob_layout(d, f);
        // Finite in every plane (the parity test's choice): packed nibbles,
        // BF16 scales and biases all read as small finite values.
        let blob = vec![0x11u8; blob_len];
        let mut points: [Vec<Point>; 3] = Default::default();
        let mut routing_rows = Vec::new();
        let mut skipped = Vec::new();
        for &m in &opts.rows {
            let (window, record) = plan(&spec, trace.as_ref(), e, k, m);
            routing_rows.push(record);
            let Some(window) = window else {
                skipped.push(m);
                continue;
            };
            let union = window.union;
            let blobs: Vec<MetalBuffer> = (0..union)
                .map(|_| context.new_buffer_with_data(&blob))
                .collect();
            let x = context.new_buffer_with_data(&f16_bytes(
                (0..m * d).map(|i| ((i as f32) * 0.13).sin() * 0.7),
            ));
            let weights: Vec<f32> = (0..m * k).map(|i| 0.05 + 0.08 * ((i % 7) as f32)).collect();
            let routing = context.new_buffer_with_data(&f16_bytes(weights.iter().copied()));
            // Per-token decode routing, padded to the decode pair's width.
            let decode_routing: Vec<MetalBuffer> = (0..m)
                .map(|t| {
                    let mut w = vec![0.0f32; MAX_STREAMED_EXPERTS];
                    w[..k].copy_from_slice(&weights[t * k..t * k + k]);
                    context.new_buffer_with_data(&f16_bytes(w.into_iter()))
                })
                .collect();
            let decode = RoutedBlobsBuffer::new(context, silu).map_err(|e| e.to_string())?;
            let seq_acts = context.new_output_buffer((MAX_STREAMED_EXPERTS * f * 2) as u64);
            let seq_y = context.new_output_buffer((d * 2) as u64);
            let seq_zero = context.new_buffer_with_data(&vec![0u8; d * 2]);
            let (pair_order, sorted_order) = routes::route_lists(&window.slots);
            let route_buffer = |list: &[(u32, u32, u32)]| {
                let encoded: Vec<MoePrefillRoute> = list
                    .iter()
                    .map(|&(token, rank, slot)| MoePrefillRoute { token, rank, slot })
                    .collect();
                context.new_buffer_with_data(&MoePrefillRoute::bytes(&encoded))
            };
            let routes_buf = route_buffer(&pair_order);
            let sorted_buf = route_buffer(&sorted_order);
            let wide = RoutedBlobsWideBuffer::new(context, silu).map_err(|e| e.to_string())?;
            let refs: Vec<(&MetalBuffer, u64)> = blobs.iter().map(|b| (b, 0u64)).collect();
            wide.bind(context, silu, &refs).map_err(|e| e.to_string())?;
            let bat_acts = context.new_output_buffer((m * k * f * 2) as u64);
            let bat_y = context.new_output_buffer((m * d * 2) as u64);
            let bat_zero = context.new_buffer_with_data(&vec![0u8; m * d * 2]);

            let mut samples: [Vec<f64>; 3] = Default::default();
            let mut hashes: [Option<String>; 3] = Default::default();
            for rep in 0..=opts.reps {
                // Every token's sequential output, read after its own pass
                // so the next one does not overwrite it.
                let mut seq_bytes = Vec::new();
                // Sequential: one decode pair per token, each its own pass,
                // as the decode path commits them. GPU time summed.
                let mut seq = 0.0;
                for t in 0..m {
                    let tok_refs: Vec<(&MetalBuffer, u64)> = window.slots[t]
                        .iter()
                        .map(|&slot| (&blobs[slot as usize], 0u64))
                        .collect();
                    decode
                        .bind(context, silu, &tok_refs)
                        .map_err(|e| e.to_string())?;
                    seq += autorelease_pool(|| -> Result<f64, String> {
                        let pass = context.begin_pass();
                        for b in &blobs {
                            pass.use_read_buffer(b);
                        }
                        let e = |r: Result<(), gpu::GpuError>| r.map_err(|e| e.to_string());
                        e(encode_moe_phase1(
                            context,
                            &pass,
                            &decode,
                            &offsets,
                            (&x, (t * d * 2) as u64),
                            (&seq_acts, 0),
                            d as u32,
                            f as u32,
                            k as u32,
                            silu,
                        ))?;
                        e(encode_moe_phase2(
                            context,
                            &pass,
                            &decode,
                            &offsets,
                            (&seq_acts, 0),
                            (&decode_routing[t], 0),
                            (&seq_zero, 0),
                            (&seq_y, 0),
                            d as u32,
                            f as u32,
                            k as u32,
                            silu,
                        ))?;
                        Ok(pass.commit_and_wait_with_gpu_time())
                    })?;
                    if opts.hash && rep == opts.reps {
                        seq_bytes.extend(read_buffer_bytes(&seq_y, 0, d * 2));
                    }
                }
                // The two grid orders alternate which goes first, so neither
                // always inherits the other's warm weights.
                let orders = if rep % 2 == 0 { [1, 2] } else { [2, 1] };
                let mut batched = [0.0f64; 3];
                for arm in orders {
                    let phase1_routes = if arm == 1 { &routes_buf } else { &sorted_buf };
                    batched[arm] = autorelease_pool(|| -> Result<f64, String> {
                        let pass = context.begin_pass();
                        for b in &blobs {
                            pass.use_read_buffer(b);
                        }
                        let e = |r: Result<(), gpu::GpuError>| r.map_err(|e| e.to_string());
                        e(encode_moe_prefill_phase1(
                            context,
                            &pass,
                            &wide,
                            &offsets,
                            (&x, 0),
                            (&bat_acts, 0),
                            (phase1_routes, 0),
                            d as u32,
                            f as u32,
                            k as u32,
                            (m * k) as u32,
                            silu,
                        ))?;
                        e(encode_moe_prefill_phase2_fused(
                            context,
                            &pass,
                            &wide,
                            &offsets,
                            (&bat_acts, 0),
                            (&routing, 0),
                            (&routes_buf, 0),
                            (&bat_zero, 0),
                            (&bat_y, 0),
                            d as u32,
                            f as u32,
                            k as u32,
                            m as u32,
                            silu,
                        ))?;
                        Ok(pass.commit_and_wait_with_gpu_time())
                    })?;
                    if opts.hash && rep == opts.reps {
                        hashes[arm] = Some(fnv_hex(&read_buffer_bytes(&bat_y, 0, m * d * 2)));
                    }
                }
                if rep > 0 {
                    samples[0].push(seq * 1e3);
                    samples[1].push(batched[1] * 1e3);
                    samples[2].push(batched[2] * 1e3);
                }
                if opts.hash && rep == opts.reps {
                    hashes[0] = Some(fnv_hex(&seq_bytes));
                }
            }
            let seq_ms = median(&mut samples[0]);
            for (a, s) in samples.iter_mut().enumerate() {
                let ms = median(s);
                let mut p = Point::new(m as f64, ms);
                p.c_of_m = Some(ms / seq_ms);
                p.out_fnv = hashes[a].take();
                points[a].push(p);
            }
        }
        if points[0].is_empty() {
            return Err(format!(
                "no M can run under --routes {routes_arg}: unions past {MAX_PREFILL_EXPERT_BINDINGS} experts or too few passes"
            ));
        }
        let mut subject = BTreeMap::new();
        for (key, v) in [
            ("d", json!(d)),
            ("f", json!(f)),
            ("top_k", json!(k)),
            ("experts", json!(e)),
            ("routes", json!(routes_arg)),
            ("capture_fnv", json!(capture_fnv)),
            ("routing", json!(routing_rows)),
        ] {
            subject.insert(key.to_string(), v);
        }
        let curve = |a: usize| -> Vec<(usize, f64)> {
            points[a].iter().map(|p| (p.x as usize, p.ms)).collect()
        };
        let break_even: Vec<BreakEven> = [1, 2]
            .iter()
            .map(|&a| BreakEven {
                arm: ARMS[a].into(),
                against: ARMS[0].into(),
                above_x: routes::wins_above(&curve(0), &curve(a)),
            })
            .collect();
        let source = match &spec {
            RouteSpec::Uniform => {
                "unions are modeled (uniform), not measured; skewed real routing binds fewer experts"
                    .to_string()
            }
            RouteSpec::Zipf(s) => format!(
                "routes are a seeded zipf({s}) draw; each M times its median-union window"
            ),
            RouteSpec::Trace(path) => format!(
                "routes are from {}; each M times its median-union window of consecutive passes in one layer, with synthetic routing weights",
                path.display()
            ),
        };
        let mut notes = vec![
            "c(M) here is each arm over M sequential decode pairs".to_string(),
            source,
            "break-even is the largest swept M at which an arm does not beat the sequential pairs"
                .to_string(),
        ];
        if !skipped.is_empty() {
            notes.push(format!(
                "M = {skipped:?} skipped (see subject.routing: union past the {MAX_PREFILL_EXPERT_BINDINGS}-binding cap, or too few passes)"
            ));
        }
        // A non-default source is its own case, so no ledger reader joins
        // its curve to a uniform one.
        let case_label = match spec {
            RouteSpec::Uniform => format!("d{d}_f{f}_k{k}"),
            RouteSpec::Zipf(s) => format!("d{d}_f{f}_k{k}@zipf{s}"),
            RouteSpec::Trace(_) => format!(
                "d{d}_f{f}_k{k}@trace_{}",
                capture_fnv.as_deref().unwrap_or("")
            ),
        };
        let mut result = BenchResult {
            bench: "moe_routed".into(),
            case: case_label,
            subject,
            x: Axis::new("m", "tokens", "log2"),
            layout: crate::bench::INT4_AFFINE_LAYOUT.into(),
            arms: points
                .into_iter()
                .zip(ARMS)
                .map(|(points, label)| ArmCurve {
                    label: label.into(),
                    points,
                    fit: None,
                    unsupported: None,
                })
                .collect(),
            break_even,
            notes,
        };
        tile_pin::attach(
            &mut result,
            &tile_pin::Plan {
                buckets: [(1, 1), (2, 4), (5, 8), (9, 32)]
                    .iter()
                    .map(|&(lo, hi)| (Bucket { lo, hi }, ARMS[1]))
                    .collect(),
                candidates: vec![ARMS[2]],
            },
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_union_is_bounded_and_monotone() {
        assert_eq!(uniform_union(128, 8, 1), 8);
        let u: Vec<usize> = [1, 2, 4, 8, 16]
            .iter()
            .map(|&m| uniform_union(128, 8, m))
            .collect();
        assert!(u.windows(2).all(|w| w[0] <= w[1]), "{u:?}");
        assert!(uniform_union(8, 4, 64) <= 8);
    }

    #[test]
    fn top_k_above_the_fixed_slots_is_refused_not_run() {
        // qwen4exp routes top-10; the route-list pair panicked on it.
        let why = top_k_refusal(10).expect("top-10 must be refused");
        assert!(why.contains("fixed slots"), "{why}");
        assert_eq!(top_k_refusal(PHASE2_FIXED_SLOTS), None);
        assert_eq!(top_k_refusal(1), None);
        assert!(top_k_refusal(0).is_some());
    }
}
