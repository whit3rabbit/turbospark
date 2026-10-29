//! The Neural Engine against Steel at the install's shapes: Steel alone, a
//! Core ML matmul on the ANE (its Metal conversions and the host handoff
//! included), and the concurrent row split at a few shares, per shape and
//! token count, with modeled energy beside the time.
//!
//! `main-npu`'s `scripts/ane-sweep.py` in this engine (`docs/MOE_PRIOR_ART.md`,
//! "The Neural Engine work"), so the per-matmul question Phase 0 answered by
//! hand once is a ledgered curve any chip can re-run. The ANE arms use the
//! shipped public path (`SharedF16Plane`, `predict_shared`, a host-signaled
//! `MTLSharedEvent`), not the fork's private framework, so what they measure
//! is what `TURBOSPARK_ANE_PREFILL` would pay. Each point records where Core
//! ML planned the program: a program planned on the CPU keeps its timing and
//! is marked, since "supported on the ANE" is not "placed there".
//!
//! The bench needs one Core ML program per (K, ANE rows, tokens), written by
//! `crates/coreml/tools/gen_ane_sweep_programs.py` into a directory given
//! as `--programs`. Token counts come from those programs; `--rows` does not
//! apply here. For an install that routes experts, the manifest's packed
//! expert programs (`E` experts of `g` rows as one batched matmul) run
//! against `E` Steel dispatches: the "one multi-expert Core ML call" form
//! `docs/SPEEDUP_PLAYBOOK.md` names as the only reopen path for ANE expert
//! work.
//!
//! Compiled only with the lab's `ane-prefill` feature; without it `cases`
//! returns the coverage answer.

use std::path::Path;

use serde_json::Value;

use super::Subject;

pub struct AneMatmul;

/// A program whose ops Core ML plans on the ANE below this cost share is
/// "planned on CPU": timed, marked, and left out of the gate fit.
pub const ANE_PLACEMENT_MIN_SHARE: f64 = 0.5;

/// Weight copies rotate through at most this many F16 planes: the big dense
/// shapes are 170 MiB each as F16, and the IOSurface budget is bounded.
#[cfg(feature = "ane-prefill")]
const MAX_PLANE_COPIES: usize = 4;

/// A program whose weights are INT4 constants inside it, not an F16 input.
pub const CONST_INT4: &str = "const_int4";

/// One program of a `sweep.json` manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// `""` for the F16 weight input form, [`CONST_INT4`] for baked weights.
    pub weights: String,
    /// The install tensor a baked program was built from (`""` for the
    /// synthetic-weight programs). Programs with one make a `real` case.
    pub tensor: String,
    pub kind: String,
    pub k: usize,
    pub rows: usize,
    pub ane_rows: usize,
    pub tokens: usize,
    pub share: f64,
    pub package: String,
    pub role: String,
    pub experts: usize,
    pub group: usize,
}

/// Reads `DIR/sweep.json` (`ane-sweep-v1`).
pub fn read_manifest(dir: &Path) -> Result<Vec<Program>, String> {
    let path = dir.join("sweep.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if v["format"].as_str() != Some("ane-sweep-v1") {
        return Err(format!("{}: not an ane-sweep-v1 manifest", path.display()));
    }
    let n = |p: &Value, k: &str| p[k].as_u64().unwrap_or(0) as usize;
    Ok(v["programs"]
        .as_array()
        .ok_or_else(|| format!("{}: programs must be an array", path.display()))?
        .iter()
        .map(|p| Program {
            weights: p["weights"].as_str().unwrap_or("").to_string(),
            tensor: p["tensor"].as_str().unwrap_or("").to_string(),
            kind: p["kind"].as_str().unwrap_or("").to_string(),
            k: n(p, "k"),
            rows: n(p, "rows"),
            ane_rows: n(p, "ane_rows"),
            tokens: n(p, "tokens"),
            share: p["share"].as_f64().unwrap_or(1.0),
            package: p["package"].as_str().unwrap_or("").to_string(),
            role: p["role"].as_str().unwrap_or("").to_string(),
            experts: n(p, "experts"),
            group: n(p, "group"),
        })
        .collect())
}

/// The install's INT4 shapes the planes can hold: rows and K multiples of 32.
#[cfg(feature = "ane-prefill")]
fn install_shapes(subject: &Subject) -> Result<Vec<super::MatShape>, String> {
    let arch = subject.arch()?;
    let (shapes, _) = super::matmul_shapes(subject, crate::inventory::DTYPE_INT4_AFFINE, || {
        super::int4_matmul::shapes(arch)
    });
    Ok(shapes
        .into_iter()
        .filter(|s| s.rows % 32 == 0 && s.cols % 32 == 0)
        .collect())
}

/// `(hidden, intermediate)` for an install that routes experts.
#[cfg(feature = "ane-prefill")]
fn expert_dims(arch: &model_io::ArchConfig) -> Option<(usize, usize)> {
    (arch.num_experts > 0 && arch.moe_intermediate_size > 0).then_some((
        arch.hidden_size as usize,
        arch.moe_intermediate_size as usize,
    ))
}

/// The generator line that writes this install's programs.
pub fn generator_line(
    shapes: &[super::MatShape],
    dims: Option<(usize, usize)>,
    out: &str,
) -> String {
    let mut line = format!(
        "uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 python -I \
         crates/coreml/tools/gen_ane_sweep_programs.py {out} --tokens 32,128,512,1024,2048"
    );
    if !shapes.is_empty() {
        let list: Vec<String> = shapes
            .iter()
            .map(|s| format!("{}x{}", s.rows, s.cols))
            .collect();
        line.push_str(&format!(" --shapes {}", list.join(",")));
    }
    if let Some((h, f)) = dims {
        line.push_str(&format!(" --experts 4:32,8:64,16:64 --expert-dims {h},{f}"));
    }
    line
}

#[cfg(not(feature = "ane-prefill"))]
impl super::KernelBench for AneMatmul {
    fn name(&self) -> &'static str {
        "ane_matmul"
    }

    fn describe(&self) -> &'static str {
        "Steel against the Neural Engine per shape and token count, with the row split and modeled energy (needs --features ane-prefill and --programs DIR)"
    }

    fn cases(
        &self,
        _subject: &Subject,
        _opts: &super::BenchOpts,
    ) -> Result<Vec<super::Case>, String> {
        Err(
            "built without the ane-prefill feature: cargo build --release -p turbospark-lab \
             --features ane-prefill"
                .into(),
        )
    }

    fn run(
        &self,
        _context: &mut gpu::MetalContext,
        _subject: &Subject,
        _case: &super::Case,
        _opts: &super::BenchOpts,
    ) -> Result<crate::bench::BenchResult, String> {
        Err("built without the ane-prefill feature".into())
    }
}

#[cfg(feature = "ane-prefill")]
pub use enabled::*;

#[cfg(feature = "ane-prefill")]
mod enabled {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::Barrier;
    use std::time::Instant;

    use gpu::ane_split::{
        begin_pass_after, encode_ane_convert_out, encode_ane_dequant_rows, encode_ane_scale_rows,
        encode_dequant_int4_gemm_steel_rows, HostEvent,
    };
    use gpu::{
        autorelease_pool, read_buffer_bytes, steel_supports, Int4ResidentMatrix, MetalContext,
        SteelTile,
    };
    use half::f16;
    use serde_json::{json, Value};
    use turbospark_coreml::{
        compile_if_needed, compute_plan, supports, Capability, ComputeUnits, CoreMlModel,
        LoadOptions, SharedF16Plane,
    };

    use super::super::support::{copies_for, f16_activations, int4_blob, median};
    use super::super::{BenchOpts, Case, KernelBench, Subject};
    use super::{
        expert_dims, generator_line, install_shapes, read_manifest, AneMatmul, Program,
        ANE_PLACEMENT_MIN_SHARE, CONST_INT4, MAX_PLANE_COPIES,
    };
    use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, BreakEven, Point};
    use crate::fit::{ane_gate_fit, break_even, fit_affine, GateCase, GateObjective};

    pub use crate::bench::INT4_AFFINE_LAYOUT as LAYOUT;

    /// One arm at one token count: wall ms samples, joule samples, and the
    /// output hash under `--hash`.
    type Samples = (Vec<f64>, Vec<f64>, Option<String>);

    /// Wall ms samples and joule samples of one arm at one token count.
    type Timed = (Vec<f64>, Vec<f64>);

    /// An F16 plane both devices read in place: the IOSurface and the
    /// no-copy Metal buffer over it. `metal` is declared first so it drops
    /// before the surface it points into.
    struct Plane {
        metal: gpu::MetalBuffer,
        plane: SharedF16Plane,
    }

    impl Plane {
        fn new(context: &MetalContext, rows: usize, cols: usize) -> Result<Self, String> {
            let plane = SharedF16Plane::new(rows, cols).map_err(|e| e.to_string())?;
            let metal = gpu::wrap_page_aligned_no_copy(
                context.device(),
                plane.base_ptr(),
                plane.alloc_size(),
            )
            .map_err(|e| e.to_string())?;
            Ok(Self { metal, plane })
        }

        fn stride(&self) -> usize {
            self.plane.stride()
        }
    }

    /// A compiled, loaded program and where Core ML planned it.
    struct Loaded {
        model: CoreMlModel,
        /// Cost share of its ops on the ANE, NaN when plans are unavailable.
        ane_share: f64,
    }

    fn load_program(dir: &Path, package: &str) -> Result<Loaded, String> {
        let path = dir.join(package);
        let cache = dir.join(".compiled");
        let compiled = compile_if_needed(&path, &cache).map_err(|e| format!("{package}: {e}"))?;
        let opts = LoadOptions::new(ComputeUnits::CpuAndNeuralEngine);
        let ane_share = if supports(Capability::ComputePlan) {
            let plan = compute_plan(&compiled, &opts).map_err(|e| format!("{package}: {e}"))?;
            crate::ane::summarize(&plan)
                .as_array()
                .and_then(|fs| fs.first())
                .and_then(|f| f["cost_share"]["ane"].as_f64())
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let model = CoreMlModel::load(&compiled, &opts).map_err(|e| format!("{package}: {e}"))?;
        Ok(Loaded { model, ane_share })
    }

    fn placed_on_ane(share: f64) -> bool {
        share.is_nan() || share >= ANE_PLACEMENT_MIN_SHARE
    }

    /// The arm a const-weight program times, apart from the F16-plane `ane`.
    const CONST_INT4_ARM: &str = "ane_c4";

    /// The arm that splits tokens, not rows, between the ANE and Steel.
    const TOKEN_SPLIT_ARM: &str = "tsplit";

    /// The token count the ANE takes in the token split of `n` tokens: half,
    /// when a whole-matrix program exists for it.
    pub(super) fn tsplit_half(n: usize, tokens: &[usize]) -> Option<usize> {
        let m = n / 2;
        (n % 2 == 0 && tokens.contains(&m)).then_some(m)
    }

    /// The arm label of a program: const-weight programs get their own.
    pub(super) fn label_of(p: &Program) -> String {
        match (p.weights == CONST_INT4, p.share >= 1.0) {
            (true, true) => CONST_INT4_ARM.into(),
            (true, false) => format!("split_c4@{}", p.share),
            (false, _) => arm_label(p.share),
        }
    }

    /// Whether an arm runs a const-weight program: no F16 plane to fill, no
    /// dequant pass, and only the activation as an input.
    pub(super) fn is_const_arm(label: &str) -> bool {
        label.starts_with(CONST_INT4_ARM) || label.starts_with("split_c4")
    }

    fn arm_label(share: f64) -> String {
        if share >= 1.0 {
            "ane".into()
        } else {
            format!("split@{share}")
        }
    }

    fn programs_of(case: &Case) -> Vec<Program> {
        let n = |p: &Value, k: &str| p[k].as_u64().unwrap_or(0) as usize;
        case.params["programs"]
            .as_array()
            .map(|ps| {
                ps.iter()
                    .map(|p| Program {
                        weights: p["weights"].as_str().unwrap_or("").to_string(),
                        tensor: p["tensor"].as_str().unwrap_or("").to_string(),
                        kind: p["kind"].as_str().unwrap_or("").to_string(),
                        k: n(p, "k"),
                        rows: n(p, "rows"),
                        ane_rows: n(p, "ane_rows"),
                        tokens: n(p, "tokens"),
                        share: p["share"].as_f64().unwrap_or(1.0),
                        package: p["package"].as_str().unwrap_or("").to_string(),
                        role: p["role"].as_str().unwrap_or("").to_string(),
                        experts: n(p, "experts"),
                        group: n(p, "group"),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn program_json(p: &Program) -> Value {
        json!({"weights": p.weights, "tensor": p.tensor, "kind": p.kind, "k": p.k, "rows": p.rows, "ane_rows": p.ane_rows,
               "tokens": p.tokens, "share": p.share, "package": p.package,
               "role": p.role, "experts": p.experts, "group": p.group})
    }

    /// Predictions per queue-depth sample. Even, so depth 2 splits it evenly.
    const DEPTH_BATCH: usize = 8;

    /// Per token count: ms and joule samples per prediction at queue depth
    /// 1 and at depth 2.
    #[derive(Default)]
    struct DepthSamples {
        x1: (Vec<f64>, Vec<f64>),
        x2: (Vec<f64>, Vec<f64>),
    }

    /// The three planes one prediction binds: activation, weights, output.
    /// `Plane` wraps a surface in a Metal buffer so the GPU can fill it; the
    /// worker threads need the surface alone, which is `Send` (the Metal
    /// wrapper is not), so the wrappers are dropped once the fill is done.
    fn depth_planes(
        context: &mut MetalContext,
        w: &Int4ResidentMatrix,
        x: &gpu::MetalBuffer,
        inv: &gpu::MetalBuffer,
        p: &Program,
        cols: usize,
    ) -> Result<[SharedF16Plane; 3], String> {
        let ap = Plane::new(context, p.tokens, cols)?;
        let wp = Plane::new(context, w.rows, cols)?;
        let out = SharedF16Plane::new(p.ane_rows, p.tokens).map_err(|e| e.to_string())?;
        let prep = context.begin_pass();
        encode_ane_dequant_rows(context, &prep, w, p.ane_rows, (&wp.metal, 0), wp.stride())
            .map_err(|e| e.to_string())?;
        encode_ane_scale_rows(
            context,
            &prep,
            (x, 0),
            cols,
            p.tokens,
            (&ap.metal, 0),
            ap.stride(),
            inv,
        )
        .map_err(|e| e.to_string())?;
        prep.commit().wait();
        let (Plane { plane: a, .. }, Plane { plane: wt, .. }) = (ap, wp);
        Ok([a, wt, out])
    }

    /// One worker of the queue-depth measurement. Per rep it meets the
    /// driver at four barriers: depth 1 (only the first worker predicts)
    /// start and end, then depth 2 (both predict half the batch) start and
    /// end. After a failed prediction it keeps meeting the barriers so the
    /// driver never deadlocks, and reports the first error at the end.
    fn depth_worker(
        first: bool,
        model: &CoreMlModel,
        planes: [SharedF16Plane; 3],
        bar: &Barrier,
        reps: usize,
    ) -> Result<(), String> {
        let [a, w, o] = planes;
        let mut err: Option<String> = None;
        let mut run = |count: usize| {
            for _ in 0..count {
                if err.is_some() {
                    return;
                }
                if let Err(e) = model.predict_shared(&[("a", &a), ("w0", &w)], &[("o0", &o)]) {
                    err = Some(e.to_string());
                }
            }
        };
        for _ in 0..=reps {
            bar.wait();
            if first {
                run(DEPTH_BATCH);
            }
            bar.wait();
            bar.wait();
            run(DEPTH_BATCH / 2);
            bar.wait();
        }
        err.map_or(Ok(()), Err)
    }

    /// Raw prediction cost at queue depth 1 against 2: the same program on
    /// two separately loaded models, each with its own planes, driven by two
    /// threads (a model serializes predictions on a mutex, so depth 2 needs
    /// two instances and two callers). No conversions and no handoff, so it
    /// answers one question: does the ANE finish more predictions per second
    /// when a second is already waiting.
    #[allow(clippy::too_many_arguments)]
    fn measure_depth(
        context: &mut MetalContext,
        w: &Int4ResidentMatrix,
        x: &gpu::MetalBuffer,
        inv: &gpu::MetalBuffer,
        p: &Program,
        cols: usize,
        models: [&CoreMlModel; 2],
        reps: usize,
        joules: &Joules,
    ) -> Result<DepthSamples, String> {
        let sets = [
            depth_planes(context, w, x, inv, p, cols)?,
            depth_planes(context, w, x, inv, p, cols)?,
        ];
        let bar = Barrier::new(3);
        let mut out = DepthSamples::default();
        let per = |t: f64| t * 1e3 / DEPTH_BATCH as f64;
        let driven = std::thread::scope(|s| -> Result<(), String> {
            let [set0, set1] = sets;
            let h0 = s.spawn(|| depth_worker(true, models[0], set0, &bar, reps));
            let h1 = s.spawn(|| depth_worker(false, models[1], set1, &bar, reps));
            for rep in 0..=reps {
                for depth in 0..2 {
                    let (t, j) = joules.around(|| {
                        let t0 = Instant::now();
                        bar.wait();
                        bar.wait();
                        Ok(t0.elapsed().as_secs_f64())
                    })?;
                    if rep > 0 {
                        let slot = if depth == 0 { &mut out.x1 } else { &mut out.x2 };
                        slot.0.push(per(t));
                        if let Some(j) = j {
                            slot.1.push(j / DEPTH_BATCH as f64);
                        }
                    }
                }
            }
            h0.join()
                .map_err(|_| "depth worker 0 panicked".to_string())??;
            h1.join()
                .map_err(|_| "depth worker 1 panicked".to_string())??;
            Ok(())
        });
        driven.map(|()| out)
    }

    /// A raw-prediction curve over token counts for the queue-depth arms.
    fn depth_curve(
        label: &str,
        rows: usize,
        cols: usize,
        samples: Vec<(usize, Vec<f64>, Vec<f64>)>,
    ) -> ArmCurve {
        let points: Vec<Point> = samples
            .into_iter()
            .map(|(n, mut ms, mut js)| {
                let t = median(&mut ms);
                Point {
                    x: n as f64,
                    ms: t,
                    gb_s: None,
                    tflops: Some(2.0 * n as f64 * rows as f64 * cols as f64 / (t * 1e-3) / 1e12),
                    c_of_m: None,
                    out_fnv: None,
                    row_fnv: None,
                    j: (!js.is_empty()).then(|| median(&mut js)),
                }
            })
            .collect();
        let fit_points: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.ms)).collect();
        ArmCurve {
            label: label.into(),
            points,
            fit: fit_affine(&fit_points),
            unsupported: None,
        }
    }

    impl KernelBench for AneMatmul {
        fn name(&self) -> &'static str {
            "ane_matmul"
        }

        fn describe(&self) -> &'static str {
            "Steel against the Neural Engine per shape and token count, with the row split and modeled energy (--programs DIR)"
        }

        fn cases(&self, subject: &Subject, opts: &BenchOpts) -> Result<Vec<Case>, String> {
            let arch = subject.arch()?;
            let shapes = install_shapes(subject)?;
            let dims = expert_dims(arch);
            let Some(dir) = &opts.ane_programs else {
                return Err(format!(
                    "needs --programs DIR holding sweep.json; write this install's programs with:\n  {}",
                    generator_line(&shapes, dims, "~/models/ane-sweep/<install>")
                ));
            };
            let manifest = read_manifest(dir)?;
            let mut cases = Vec::new();
            let mut missing = Vec::new();
            for s in &shapes {
                let progs: Vec<Value> = manifest
                    .iter()
                    .filter(|p| {
                        p.kind == "dense"
                            && p.tensor.is_empty()
                            && p.rows == s.rows
                            && p.k == s.cols
                    })
                    .map(program_json)
                    .collect();
                if progs.is_empty() {
                    missing.push(s.clone());
                    continue;
                }
                cases.push(Case {
                    label: format!("{}x{}", s.rows, s.cols),
                    params: json!({"kind": "dense", "rows": s.rows, "cols": s.cols,
                                   "roles": s.roles, "count": s.count,
                                   "programs": progs, "dir": dir.display().to_string()}),
                });
            }
            if let Some((h, f)) = dims {
                for p in manifest.iter().filter(|p| p.kind == "experts") {
                    let fits = match p.role.as_str() {
                        "gateup" => p.k == h && p.rows == f,
                        "down" => p.k == f && p.rows == h,
                        _ => false,
                    };
                    if !fits {
                        continue;
                    }
                    cases.push(Case {
                        label: format!("experts_{}_e{}_g{}", p.role, p.experts, p.group),
                        params: json!({"kind": "experts", "programs": [program_json(p)],
                                       "dir": dir.display().to_string()}),
                    });
                }
            }
            // One `real` case per install tensor a baked program was built
            // from: its F16-plane twin (same shape and ANE rows) rides along.
            let mut tensors: Vec<&str> = manifest
                .iter()
                .filter(|p| !p.tensor.is_empty())
                .map(|p| p.tensor.as_str())
                .collect();
            tensors.sort_unstable();
            tensors.dedup();
            for t in tensors {
                let baked: Vec<&Program> = manifest.iter().filter(|p| p.tensor == t).collect();
                let (rows, cols) = (baked[0].rows, baked[0].k);
                let progs: Vec<Value> = manifest
                    .iter()
                    .filter(|p| {
                        p.tensor == t
                            || (p.kind == "dense"
                                && p.tensor.is_empty()
                                && p.weights.is_empty()
                                && p.rows == rows
                                && p.k == cols
                                && baked.iter().any(|b| b.ane_rows == p.ane_rows))
                    })
                    .map(program_json)
                    .collect();
                cases.push(Case {
                    label: format!(
                        "real:{}",
                        t.split("layers.")
                            .last()
                            .unwrap_or(t)
                            .trim_end_matches(".weight")
                    ),
                    params: json!({"kind": "real", "tensor": t, "rows": rows, "cols": cols,
                                   "programs": progs, "dir": dir.display().to_string()}),
                });
            }
            if cases.is_empty() {
                return Err(format!(
                    "no program in {} matches this install; write them with:\n  {}",
                    dir.display(),
                    generator_line(&shapes, dims, &dir.display().to_string())
                ));
            }
            if !missing.is_empty() {
                eprintln!(
                    "ane_matmul: {} of this install's shapes have no program in {} (add them with the --shapes list: {})",
                    missing.len(),
                    dir.display(),
                    missing
                        .iter()
                        .map(|s| format!("{}x{}", s.rows, s.cols))
                        .collect::<Vec<_>>()
                        .join(",")
                );
            }
            Ok(cases)
        }

        fn run(
            &self,
            context: &mut MetalContext,
            subject: &Subject,
            case: &Case,
            opts: &BenchOpts,
        ) -> Result<BenchResult, String> {
            let dir = Path::new(case.params["dir"].as_str().unwrap_or(""));
            match case.params["kind"].as_str() {
                Some("real") => run_real(context, subject, case, dir, opts),
                Some("experts") => run_experts(context, case, dir, opts),
                _ => run_dense(context, case, dir, opts),
            }
        }
    }

    /// Reads the meter around one arm's rep: joules per matmul, or None.
    struct Joules {
        meter: Option<crate::energy::EnergyMeter>,
    }

    impl Joules {
        fn open() -> Self {
            Self {
                meter: crate::energy::EnergyMeter::open().ok(),
            }
        }

        fn around<T>(
            &self,
            f: impl FnOnce() -> Result<T, String>,
        ) -> Result<(T, Option<f64>), String> {
            let e0 = self.meter.as_ref().and_then(|m| m.read().ok());
            let out = f()?;
            let e1 = self.meter.as_ref().and_then(|m| m.read().ok());
            let j = match (e0, e1) {
                (Some(a), Some(b)) => crate::energy::delta(&a, &b).map(|d| d.total_j),
                _ => None,
            };
            Ok((out, j))
        }
    }

    /// Estimated bytes moved by the ANE path for one matmul: the resident
    /// INT4 read, the F16 weight write and read, the activation read, the
    /// scaled copy's write and read, the output's write and read, and the
    /// final F16 write. The intensity the gate is fitted on.
    fn ane_bytes(weight_bytes: usize, rows: usize, ma: usize, cols: usize, n: usize) -> f64 {
        (weight_bytes + 4 * ma * cols + 2 * n * cols + 4 * n * cols + 4 * n * ma + 2 * n * rows)
            as f64
    }

    fn run_dense(
        context: &mut MetalContext,
        case: &Case,
        dir: &Path,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let dim = |k: &str| {
            case.params[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("ane_matmul case {} lacks {k}", case.label))
        };
        let (rows, cols) = (dim("rows")?, dim("cols")?);
        let programs = programs_of(case);
        let mut tokens: Vec<usize> = programs.iter().map(|p| p.tokens).collect();
        tokens.sort_unstable();
        tokens.dedup();
        // Arm labels by descending share, a const-weight arm after the
        // F16-plane arm of the same share.
        let mut ordered: Vec<(f64, bool, String)> = programs
            .iter()
            .map(|p| (p.share, p.weights == CONST_INT4, label_of(p)))
            .collect();
        ordered.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut ane_labels: Vec<String> = ordered.into_iter().map(|(_, _, l)| l).collect();
        ane_labels.dedup();
        let reps = opts.reps;
        let weight_bytes = rows * cols / 2 + rows * cols / 64 * 4;
        let copies = copies_for(weight_bytes, 128 << 20).min(MAX_PLANE_COPIES);
        let max_n = *tokens.iter().max().ok_or("no token counts")?;

        let mats: Vec<_> = (0..copies)
            .map(|i| {
                let (data, scales, biases) = int4_blob(rows, cols, 0xA11E_0000 + i as u64);
                (context.new_buffer_with_data(&data), scales, biases)
            })
            .collect();
        let steel_ok = {
            let (buffer, scales, biases) = &mats[0];
            let w = Int4ResidentMatrix {
                buffer,
                weights_offset: 0,
                scales_offset: *scales,
                biases_offset: *biases,
                rows,
                cols,
            };
            steel_supports(&w, SteelTile::M32K32N32)
        };
        let wplanes: Vec<Plane> = (0..copies)
            .map(|_| Plane::new(context, rows, cols))
            .collect::<Result<_, _>>()?;
        let x = context.new_buffer_with_data(&f16_activations(max_n * cols, 7));
        let y = context.new_output_buffer((max_n * rows * 2) as u64);
        let inv = context.new_output_buffer((max_n * 4) as u64);
        let nonfinite = context.new_buffer_with_data(&[0u32]);
        let mut aplanes: BTreeMap<usize, Plane> = BTreeMap::new();
        for &n in &tokens {
            aplanes.insert(n, Plane::new(context, n, cols)?);
        }
        let mut oplanes: BTreeMap<(usize, usize), Plane> = BTreeMap::new();
        let mut loaded: BTreeMap<(String, usize), Loaded> = BTreeMap::new();
        let mut notes = Vec::new();
        let mut placement = serde_json::Map::new();
        for p in &programs {
            let label = label_of(p);
            match load_program(dir, &p.package) {
                Ok(l) => {
                    placement.insert(
                        format!("{label}@{}", p.tokens),
                        json!({"ane_cost_share": l.ane_share, "ane_rows": p.ane_rows,
                               "planned_on": if placed_on_ane(l.ane_share) { "ane" } else { "cpu" }}),
                    );
                    if let std::collections::btree_map::Entry::Vacant(slot) =
                        oplanes.entry((p.ane_rows, p.tokens))
                    {
                        slot.insert(Plane::new(context, p.ane_rows, p.tokens)?);
                    }
                    loaded.insert((label, p.tokens), l);
                }
                Err(e) => notes.push(format!("{label} at {} tokens not loaded: {e}", p.tokens)),
            }
        }
        let event = HostEvent::new(context);
        let mut seq = event.value();
        let joules = Joules::open();
        if joules.meter.is_none() {
            notes.push("no energy meter on this host: j columns absent".into());
        }

        // Per arm label: per token count, (ms samples, j samples, hash).
        let mut arms: Vec<String> = vec!["steel".into()];
        arms.extend(ane_labels);
        // The token split, when a whole-matrix program exists for half of
        // some token count: the ANE takes the first half of the tokens
        // through every row while Steel takes the rest.
        let half_loaded = |n: usize| {
            tsplit_half(n, &tokens).filter(|m| loaded.contains_key(&("ane".to_string(), *m)))
        };
        if steel_ok && tokens.iter().any(|&n| half_loaded(n).is_some()) {
            arms.push(TOKEN_SPLIT_ARM.into());
        }
        let mut table: BTreeMap<String, BTreeMap<usize, Samples>> = BTreeMap::new();
        for &n in &tokens {
            for rep in 0..=reps {
                for label in &arms {
                    let program = programs
                        .iter()
                        .find(|p| label_of(p) == *label && p.tokens == n);
                    let ma = program.map_or(rows, |p| p.ane_rows);
                    if label == "steel" && !steel_ok {
                        continue;
                    }
                    let split_m = if label == TOKEN_SPLIT_ARM {
                        match half_loaded(n) {
                            Some(m) => Some(m),
                            None => continue,
                        }
                    } else {
                        None
                    };
                    if label != "steel"
                        && label != TOKEN_SPLIT_ARM
                        && !loaded.contains_key(&(label.clone(), n))
                    {
                        continue;
                    }
                    let (wall, j) = joules.around(|| -> Result<f64, String> {
                        let t0 = Instant::now();
                        for (c, (buffer, scales, biases)) in mats.iter().enumerate() {
                            let w = Int4ResidentMatrix {
                                buffer,
                                weights_offset: 0,
                                scales_offset: *scales,
                                biases_offset: *biases,
                                rows,
                                cols,
                            };
                            autorelease_pool(|| -> Result<(), String> {
                                if label == "steel" {
                                    let pass = context.begin_pass();
                                    encode_dequant_int4_gemm_steel_rows(
                                        context,
                                        &pass,
                                        &w,
                                        0,
                                        (&x, 0),
                                        (&y, 0),
                                        rows,
                                        n,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    pass.commit_and_wait();
                                    return Ok(());
                                }
                                if let Some(m) = split_m {
                                    // Tokens [0, m) through the ANE program
                                    // for every row, tokens [m, n) on Steel.
                                    let wp = &wplanes[c];
                                    let ap = &aplanes[&m];
                                    let op = &oplanes[&(rows, m)];
                                    let model = &loaded[&("ane".to_string(), m)].model;
                                    let prep = context.begin_pass();
                                    encode_ane_dequant_rows(
                                        context,
                                        &prep,
                                        &w,
                                        rows,
                                        (&wp.metal, 0),
                                        wp.stride(),
                                    )
                                    .map_err(|e| e.to_string())?;
                                    encode_ane_scale_rows(
                                        context,
                                        &prep,
                                        (&x, 0),
                                        cols,
                                        m,
                                        (&ap.metal, 0),
                                        ap.stride(),
                                        &inv,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    prep.commit().wait();
                                    seq += 1;
                                    // Steel's half is committed ahead of the
                                    // pass that waits on the host: a queue
                                    // runs its buffers in commit order, so a
                                    // waiting buffer committed first would
                                    // hold Steel back until the prediction
                                    // was done and the halves would add.
                                    let steel = context.begin_pass();
                                    encode_dequant_int4_gemm_steel_rows(
                                        context,
                                        &steel,
                                        &w,
                                        0,
                                        (&x, (m * cols * 2) as u64),
                                        (&y, (m * rows * 2) as u64),
                                        rows,
                                        n - m,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    let steel = steel.commit();
                                    let conv =
                                        begin_pass_after(context, "ane convert", &event, seq);
                                    encode_ane_convert_out(
                                        context,
                                        &conv,
                                        (&op.metal, 0),
                                        rows,
                                        m,
                                        op.stride(),
                                        &inv,
                                        (&y, 0),
                                        rows,
                                        &nonfinite,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    let conv = conv.commit();
                                    let predicted = model.predict_shared(
                                        &[("a", &ap.plane), ("w0", &wp.plane)],
                                        &[("o0", &op.plane)],
                                    );
                                    // Release the GPU whatever the prediction
                                    // did, or the convert pass waits forever.
                                    event.signal(seq);
                                    conv.wait();
                                    steel.wait();
                                    predicted.map_err(|e| e.to_string())?;
                                    return Ok(());
                                }
                                let wp = &wplanes[c];
                                let ap = &aplanes[&n];
                                let op = &oplanes[&(ma, n)];
                                let model = &loaded[&(label.clone(), n)].model;
                                let prep = context.begin_pass();
                                // A const-weight program carries its INT4
                                // weights inside, so there is no F16 plane to
                                // fill and no dequant pass to pay for.
                                if !is_const_arm(label) {
                                    encode_ane_dequant_rows(
                                        context,
                                        &prep,
                                        &w,
                                        ma,
                                        (&wp.metal, 0),
                                        wp.stride(),
                                    )
                                    .map_err(|e| e.to_string())?;
                                }
                                encode_ane_scale_rows(
                                    context,
                                    &prep,
                                    (&x, 0),
                                    cols,
                                    n,
                                    (&ap.metal, 0),
                                    ap.stride(),
                                    &inv,
                                )
                                .map_err(|e| e.to_string())?;
                                prep.commit().wait();
                                let mut inputs = vec![("a", &ap.plane)];
                                if !is_const_arm(label) {
                                    inputs.push(("w0", &wp.plane));
                                }
                                let outputs = [("o0", &op.plane)];
                                if ma < rows {
                                    // The split: the convert pass waits on the
                                    // host's signal, Steel's rows run meanwhile.
                                    seq += 1;
                                    // Steel's rows are committed ahead of the
                                    // pass that waits on the host: a queue
                                    // runs its buffers in commit order, so a
                                    // waiting buffer committed first holds
                                    // Steel back until the prediction is done
                                    // and the two halves add instead of
                                    // overlapping.
                                    let steel = context.begin_pass();
                                    encode_dequant_int4_gemm_steel_rows(
                                        context,
                                        &steel,
                                        &w,
                                        ma,
                                        (&x, 0),
                                        (&y, 0),
                                        rows,
                                        n,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    let steel = steel.commit();
                                    let conv =
                                        begin_pass_after(context, "ane convert", &event, seq);
                                    encode_ane_convert_out(
                                        context,
                                        &conv,
                                        (&op.metal, 0),
                                        ma,
                                        n,
                                        op.stride(),
                                        &inv,
                                        (&y, 0),
                                        rows,
                                        &nonfinite,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    let conv = conv.commit();
                                    let predicted = model.predict_shared(&inputs, &outputs);
                                    // Release the GPU whatever the prediction
                                    // did, or the convert pass waits forever.
                                    event.signal(seq);
                                    conv.wait();
                                    steel.wait();
                                    predicted.map_err(|e| e.to_string())?;
                                } else {
                                    model
                                        .predict_shared(&inputs, &outputs)
                                        .map_err(|e| e.to_string())?;
                                    let conv = context.begin_pass();
                                    encode_ane_convert_out(
                                        context,
                                        &conv,
                                        (&op.metal, 0),
                                        ma,
                                        n,
                                        op.stride(),
                                        &inv,
                                        (&y, 0),
                                        rows,
                                        &nonfinite,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    conv.commit_and_wait();
                                }
                                Ok(())
                            })?;
                        }
                        Ok(t0.elapsed().as_secs_f64())
                    })?;
                    let entry = table
                        .entry(label.clone())
                        .or_default()
                        .entry(n)
                        .or_default();
                    if rep > 0 {
                        entry.0.push(wall * 1e3 / copies as f64);
                        if let Some(j) = j {
                            entry.1.push(j / copies as f64);
                        }
                    }
                    if opts.hash && rep == reps {
                        entry.2 = Some(fnv_hex(&read_buffer_bytes(&y, 0, n * rows * 2)));
                    }
                }
            }
        }

        // Queue depth: the whole-matrix `ane` program as raw predictions on
        // one model, then on two. Only programs Core ML placed on the ANE.
        let mut depth: BTreeMap<usize, DepthSamples> = BTreeMap::new();
        for &n in &tokens {
            let Some(p) = programs
                .iter()
                .find(|p| label_of(p) == "ane" && p.tokens == n)
            else {
                continue;
            };
            let Some(first) = loaded.get(&("ane".to_string(), n)) else {
                continue;
            };
            if !placement
                .get(&format!("ane@{n}"))
                .is_some_and(|v| v["planned_on"] == "ane")
            {
                continue;
            }
            let second = match load_program(dir, &p.package) {
                Ok(l) => l,
                Err(e) => {
                    notes.push(format!(
                        "queue depth at {n} tokens: second model not loaded: {e}"
                    ));
                    continue;
                }
            };
            let (buffer, scales, biases) = &mats[0];
            let w = Int4ResidentMatrix {
                buffer,
                weights_offset: 0,
                scales_offset: *scales,
                biases_offset: *biases,
                rows,
                cols,
            };
            match measure_depth(
                context,
                &w,
                &x,
                &inv,
                p,
                cols,
                [&first.model, &second.model],
                reps,
                &joules,
            ) {
                Ok(d) => {
                    depth.insert(n, d);
                }
                Err(e) => notes.push(format!("queue depth at {n} tokens: {e}")),
            }
        }

        let flops = |n: usize| 2.0 * n as f64 * rows as f64 * cols as f64;
        let mut curves = Vec::new();
        for label in &arms {
            if label == "steel" && !steel_ok {
                curves.push(ArmCurve::unsupported(
                    "steel",
                    "this shape does not fit the Steel tile",
                ));
                continue;
            }
            let Some(rows_of) = table.get(label) else {
                curves.push(ArmCurve::unsupported(
                    label,
                    "no program loaded for this arm",
                ));
                continue;
            };
            let points: Vec<Point> = rows_of
                .iter()
                .map(|(&n, (ms, js, hash))| {
                    let mut ms = ms.clone();
                    let mut js = js.clone();
                    let t = median(&mut ms);
                    Point {
                        x: n as f64,
                        ms: t,
                        gb_s: None,
                        tflops: Some(flops(n) / (t * 1e-3) / 1e12),
                        c_of_m: None,
                        out_fnv: hash.clone(),
                        row_fnv: None,
                        j: (!js.is_empty()).then(|| median(&mut js)),
                    }
                })
                .collect();
            // Points whose program Core ML planned on the CPU stay out of the fit.
            let fitted: Vec<(f64, f64)> = points
                .iter()
                .filter(|p| {
                    // The token split runs the `ane` program of half the
                    // tokens, so it is placed where that program is.
                    let key = if label == TOKEN_SPLIT_ARM {
                        format!("ane@{}", p.x as usize / 2)
                    } else {
                        format!("{label}@{}", p.x as usize)
                    };
                    label == "steel"
                        || placement
                            .get(&key)
                            .is_some_and(|v| v["planned_on"] == "ane")
                })
                .map(|p| (p.x, p.ms))
                .collect();
            let all_cpu = label != "steel" && fitted.is_empty() && !points.is_empty();
            curves.push(ArmCurve {
                label: label.clone(),
                points,
                fit: fit_affine(&fitted),
                unsupported: all_cpu.then(|| "planned on CPU at every token count".to_string()),
            });
        }
        let steel_fit = curves[0].fit;
        let break_even_rows: Vec<BreakEven> = curves
            .iter()
            .skip(1)
            .filter_map(|c| {
                Some(BreakEven {
                    arm: c.label.clone(),
                    against: "steel".into(),
                    above_x: break_even(&steel_fit?, &c.fit?),
                })
            })
            .collect();

        // The queue-depth arms come after the break-evens: they leave out
        // the conversions and the handoff, so a crossing with Steel would
        // mean nothing.
        if !depth.is_empty() {
            for (label, second) in [("ane_pred_x1", false), ("ane_pred_x2", true)] {
                let samples = depth
                    .iter()
                    .map(|(&n, d)| {
                        let (ms, js) = if second { &d.x2 } else { &d.x1 };
                        (n, ms.clone(), js.clone())
                    })
                    .collect();
                curves.push(depth_curve(label, rows, cols, samples));
            }
            notes.push(format!(
                "ane_pred_x1 / ane_pred_x2: ms per raw prediction ({DEPTH_BATCH} per sample) on one model, then two models driven by two threads; no conversions, no handoff, so they are not comparable with the steel or ane arms"
            ));
        }

        // The gate: whole matmuls on the ANE, fitted over the `ane` arm's
        // points that Core ML planned there.
        if let (Some(steel), Some(ane)) = (table.get("steel"), table.get("ane")) {
            let cases: Vec<GateCase> = ane
                .iter()
                .filter(|(n, _)| {
                    placement
                        .get(&format!("ane@{n}"))
                        .is_some_and(|v| v["planned_on"] == "ane")
                })
                .filter_map(|(&n, (ms, js, _))| {
                    let s = steel.get(&n)?;
                    let mut ms = ms.clone();
                    let mut js = js.clone();
                    let mut sms = s.0.clone();
                    let mut sjs = s.1.clone();
                    let macs = (n * rows * cols) as f64;
                    Some(GateCase {
                        macs,
                        intensity: macs / ane_bytes(weight_bytes, rows, rows, cols, n),
                        steel_ms: median(&mut sms),
                        ane_ms: median(&mut ms),
                        steel_j: (!sjs.is_empty()).then(|| median(&mut sjs)),
                        ane_j: (!js.is_empty()).then(|| median(&mut js)),
                    })
                })
                .collect();
            for objective in [
                GateObjective::Speed,
                GateObjective::Energy,
                GateObjective::Edp,
            ] {
                if let Some(g) = ane_gate_fit(&cases, objective) {
                    notes.push(format!(
                        "ane gate ({:?}): min_macs {:.3e}, min_intensity {:.1} routes {} of {} points with {} regressions; steel {:.3}, routed {:.3}, oracle {:.3}",
                        objective, g.min_macs, g.min_intensity, g.selected, cases.len(),
                        g.regressions, g.steel_total, g.routed_total, g.oracle_total
                    ));
                }
            }
        }
        notes.push(
            "ms is wall per matmul over the copies (conversions, host handoff and the final wait included); steel is wall too"
                .into(),
        );
        notes.push("ANE outputs are F16-summed: their hashes differ from steel's by design".into());
        if arms.iter().any(|a| a == TOKEN_SPLIT_ARM) {
            notes.push(
                "tsplit: the ANE runs the first half of the tokens through every row (all rows dequantized to F16) while Steel runs the other half; compare with steel at the same x. One matrix at a time, so the handoff count of a fused block is not priced"
                    .into(),
            );
        }

        let mut subject = BTreeMap::new();
        subject.insert("rows".into(), json!(rows));
        subject.insert("cols".into(), json!(cols));
        subject.insert("weight_bytes".into(), json!(weight_bytes));
        subject.insert("copies".into(), json!(copies));
        subject.insert("placement".into(), Value::Object(placement));
        subject.insert("energy_model".into(), json!(crate::energy::ENERGY_MODEL));
        for k in ["roles", "count"] {
            if !case.params[k].is_null() {
                subject.insert(k.into(), case.params[k].clone());
            }
        }
        Ok(BenchResult {
            bench: "ane_matmul".into(),
            case: case.label.clone(),
            subject,
            x: Axis::new("tokens", "tokens", "log2"),
            layout: LAYOUT.into(),
            arms: curves,
            break_even: break_even_rows,
            notes,
        })
    }

    /// An install tensor's three planes in the layout the INT4 kernels read:
    /// codes, then BF16 scales, then BF16 biases. Returns the bytes and the
    /// scales and biases offsets.
    fn read_install_tensor(
        dir: &Path,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<u8>, u64, u64), String> {
        use std::os::unix::fs::FileExt;
        let path = dir.join("model_weights.bin");
        let index =
            model_io::load_resident_index(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let e = index
            .entries
            .get(name)
            .ok_or_else(|| format!("{name}: not in the resident index"))?;
        if e.dtype != 4 || e.shape.0 as usize != rows || e.shape.1 as usize != cols {
            return Err(format!(
                "{name}: the install has dtype {} {}x{}, the programs were built for INT4 {rows}x{cols}",
                e.dtype, e.shape.0, e.shape.1
            ));
        }
        let file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let read = |offset: u64, size: u64| -> Result<Vec<u8>, String> {
            let mut buf = vec![0u8; size as usize];
            file.read_exact_at(&mut buf, offset)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(buf)
        };
        let mut out = read(e.file_offset, e.size_bytes)?;
        let scales = out.len() as u64;
        out.extend(read(e.scale_offset, e.scale_size)?);
        let biases = out.len() as u64;
        out.extend(read(e.bias_offset, e.bias_size)?);
        Ok((out, scales, biases))
    }

    /// F16 activations with unit variance (a sum of four uniforms), closer to
    /// a normalized hidden state than the bench's small repeating pattern.
    fn gaussian_f16(count: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        (0..count)
            .flat_map(|_| {
                let g = (uniform() + uniform() + uniform() + uniform()) * 3.0f32.sqrt();
                f16::from_f32(g).to_bits().to_le_bytes()
            })
            .collect()
    }

    /// How a split output differs from Steel's on the same weights.
    pub(super) struct Accuracy {
        pub(super) max_abs: f64,
        /// `max_abs` over the rms of Steel's ANE-row outputs.
        pub(super) max_over_rms: f64,
        /// Root of the summed squared difference over the summed squared
        /// Steel output, across the ANE rows.
        pub(super) rel_rms: f64,
        /// Whether every row Steel computed in the split matches bit for bit.
        pub(super) steel_rows_identical: bool,
        pub(super) nonfinite: bool,
    }

    /// Compares token-major `[n][rows]` F16 outputs: rows `[0, ma)` came from
    /// the ANE, the rest from the same Steel kernel as the reference.
    pub(super) fn compare_outputs(
        reference: &[u8],
        got: &[u8],
        rows: usize,
        ma: usize,
        n: usize,
    ) -> Accuracy {
        let at = |b: &[u8], i: usize| f16::from_bits(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]));
        let (mut max_abs, mut d2, mut r2) = (0.0f64, 0.0f64, 0.0f64);
        let (mut identical, mut nonfinite) = (true, false);
        for t in 0..n {
            for r in 0..rows {
                let i = t * rows + r;
                let g = at(got, i);
                if !g.to_f32().is_finite() {
                    nonfinite = true;
                }
                if r < ma {
                    let (a, b) = (at(reference, i).to_f64(), g.to_f64());
                    max_abs = max_abs.max((a - b).abs());
                    d2 += (a - b) * (a - b);
                    r2 += a * a;
                } else if at(reference, i).to_bits() != g.to_bits() {
                    identical = false;
                }
            }
        }
        let rms_ref = (r2 / (n * ma).max(1) as f64).sqrt();
        Accuracy {
            max_abs,
            max_over_rms: max_abs / rms_ref.max(1e-30),
            rel_rms: (d2 / r2.max(1e-30)).sqrt(),
            steel_rows_identical: identical,
            nonfinite,
        }
    }

    /// One split matmul: the ANE takes rows `[0, ma)` while Steel takes the
    /// rest, into `y`. `w_plane` is the F16 weight plane an F16-input program
    /// reads (filled here from `w`), `None` for a baked program. Steel is
    /// committed ahead of the pass that waits on the host so the two overlap.
    #[allow(clippy::too_many_arguments)]
    fn split_once(
        context: &mut MetalContext,
        w: &Int4ResidentMatrix,
        x: &gpu::MetalBuffer,
        y: &gpu::MetalBuffer,
        inv: &gpu::MetalBuffer,
        nonfinite: &gpu::MetalBuffer,
        event: &HostEvent,
        seq: &mut u64,
        model: &CoreMlModel,
        a_plane: &Plane,
        w_plane: Option<&Plane>,
        o_plane: &Plane,
        ma: usize,
        n: usize,
    ) -> Result<(), String> {
        let (rows, cols) = (w.rows, w.cols);
        let prep = context.begin_pass();
        if let Some(wp) = w_plane {
            encode_ane_dequant_rows(context, &prep, w, ma, (&wp.metal, 0), wp.stride())
                .map_err(|e| e.to_string())?;
        }
        encode_ane_scale_rows(
            context,
            &prep,
            (x, 0),
            cols,
            n,
            (&a_plane.metal, 0),
            a_plane.stride(),
            inv,
        )
        .map_err(|e| e.to_string())?;
        prep.commit().wait();
        *seq += 1;
        let steel = context.begin_pass();
        encode_dequant_int4_gemm_steel_rows(context, &steel, w, ma, (x, 0), (y, 0), rows, n)
            .map_err(|e| e.to_string())?;
        let steel = steel.commit();
        let conv = begin_pass_after(context, "ane convert", event, *seq);
        encode_ane_convert_out(
            context,
            &conv,
            (&o_plane.metal, 0),
            ma,
            n,
            o_plane.stride(),
            inv,
            (y, 0),
            rows,
            nonfinite,
        )
        .map_err(|e| e.to_string())?;
        let conv = conv.commit();
        let mut inputs = vec![("a", &a_plane.plane)];
        if let Some(wp) = w_plane {
            inputs.push(("w0", &wp.plane));
        }
        let predicted = model.predict_shared(&inputs, &[("o0", &o_plane.plane)]);
        // Release the GPU whatever the prediction did.
        event.signal(*seq);
        conv.wait();
        steel.wait();
        predicted.map_err(|e| e.to_string())?;
        if read_buffer_bytes(nonfinite, 0, 4) != [0u8; 4] {
            return Err("non-finite ANE rows".into());
        }
        Ok(())
    }

    /// A baked program (and its F16-plane twin) on an install tensor's real
    /// bytes, against Steel on the same bytes: time, and how far the ANE
    /// rows land from Steel's. The question P1 of the baked-weights plan asks.
    fn run_real(
        context: &mut MetalContext,
        subject: &Subject,
        case: &Case,
        dir: &Path,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let tensor = case.params["tensor"].as_str().unwrap_or("");
        let dim = |k: &str| {
            case.params[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| format!("ane_matmul case {} lacks {k}", case.label))
        };
        let (rows, cols) = (dim("rows")?, dim("cols")?);
        let programs = programs_of(case);
        let (blob, scales, biases) = read_install_tensor(&subject.dir, tensor, rows, cols)?;
        let weights = context.new_buffer_with_data(&blob);
        let w = Int4ResidentMatrix {
            buffer: &weights,
            weights_offset: 0,
            scales_offset: scales,
            biases_offset: biases,
            rows,
            cols,
        };
        let mut tokens: Vec<usize> = programs.iter().map(|p| p.tokens).collect();
        tokens.sort_unstable();
        tokens.dedup();
        let reps = opts.reps;
        let event = HostEvent::new(context);
        let mut seq = event.value();
        let joules = Joules::open();
        let mut notes = vec![format!(
            "real weights: {tensor} ({rows}x{cols}) read from {}; activations are unit-variance F16",
            subject.dir.display()
        )];
        let mut placement = serde_json::Map::new();
        let mut accuracy = serde_json::Map::new();
        // label -> tokens -> (ms samples, j samples)
        let mut table: BTreeMap<String, BTreeMap<usize, Timed>> = BTreeMap::new();

        for &n in &tokens {
            let x = context.new_buffer_with_data(&gaussian_f16(n * cols, 0x5EED + n as u64));
            let y = context.new_output_buffer((n * rows * 2) as u64);
            let inv = context.new_output_buffer((n * 4) as u64);
            let nonfinite = context.new_buffer_with_data(&[0u32]);
            let a_plane = Plane::new(context, n, cols)?;

            // Steel on every row: the reference.
            let mut steel_ms = Vec::new();
            let mut steel_j = Vec::new();
            for rep in 0..=reps {
                let (wall, j) = joules.around(|| -> Result<f64, String> {
                    let t0 = Instant::now();
                    let pass = context.begin_pass();
                    encode_dequant_int4_gemm_steel_rows(
                        context,
                        &pass,
                        &w,
                        0,
                        (&x, 0),
                        (&y, 0),
                        rows,
                        n,
                    )
                    .map_err(|e| e.to_string())?;
                    pass.commit_and_wait();
                    Ok(t0.elapsed().as_secs_f64())
                })?;
                if rep > 0 {
                    steel_ms.push(wall * 1e3);
                    steel_j.extend(j);
                }
            }
            let reference = read_buffer_bytes(&y, 0, n * rows * 2);
            table
                .entry("steel".into())
                .or_default()
                .insert(n, (steel_ms, steel_j));

            let mut here: Vec<&Program> = programs.iter().filter(|p| p.tokens == n).collect();
            here.sort_by_key(|p| p.weights == CONST_INT4);
            for p in here {
                let label = label_of(p);
                let loaded = match load_program(dir, &p.package) {
                    Ok(l) => l,
                    Err(e) => {
                        notes.push(format!("{label} at {n} tokens not loaded: {e}"));
                        continue;
                    }
                };
                placement.insert(
                    format!("{label}@{n}"),
                    json!({"ane_cost_share": loaded.ane_share, "ane_rows": p.ane_rows,
                           "planned_on": if placed_on_ane(loaded.ane_share) { "ane" } else { "cpu" }}),
                );
                let ma = p.ane_rows;
                let o_plane = Plane::new(context, ma, n)?;
                let w_plane = if is_const_arm(&label) {
                    None
                } else {
                    Some(Plane::new(context, rows, cols)?)
                };
                let mut ms = Vec::new();
                let mut js = Vec::new();
                for rep in 0..=reps {
                    let (wall, j) = joules.around(|| -> Result<f64, String> {
                        let t0 = Instant::now();
                        split_once(
                            context,
                            &w,
                            &x,
                            &y,
                            &inv,
                            &nonfinite,
                            &event,
                            &mut seq,
                            &loaded.model,
                            &a_plane,
                            w_plane.as_ref(),
                            &o_plane,
                            ma,
                            n,
                        )?;
                        Ok(t0.elapsed().as_secs_f64())
                    })?;
                    if rep > 0 {
                        ms.push(wall * 1e3);
                        js.extend(j);
                    }
                }
                let got = read_buffer_bytes(&y, 0, n * rows * 2);
                let acc = compare_outputs(&reference, &got, rows, ma, n);
                notes.push(format!(
                    "accuracy {label}@{n}: ANE rows [0, {ma}) max abs {:.3e} ({:.3}x the rms of Steel's output), relative rms {:.3e}; Steel rows identical: {}; non-finite: {}",
                    acc.max_abs, acc.max_over_rms, acc.rel_rms, acc.steel_rows_identical, acc.nonfinite
                ));
                accuracy.insert(
                    format!("{label}@{n}"),
                    json!({"ane_rows": ma, "max_abs": acc.max_abs,
                           "max_over_rms": acc.max_over_rms, "rel_rms": acc.rel_rms,
                           "steel_rows_identical": acc.steel_rows_identical,
                           "nonfinite": acc.nonfinite}),
                );
                table.entry(label).or_default().insert(n, (ms, js));
            }
        }

        let flops = |n: usize| 2.0 * n as f64 * rows as f64 * cols as f64;
        let mut labels: Vec<String> = table.keys().cloned().collect();
        labels.sort_by_key(|l| (l != "steel", is_const_arm(l), l.clone()));
        let curves: Vec<ArmCurve> = labels
            .iter()
            .map(|label| {
                let points: Vec<Point> = table[label]
                    .iter()
                    .map(|(&n, (ms, js))| {
                        let (mut ms, mut js) = (ms.clone(), js.clone());
                        let t = median(&mut ms);
                        Point {
                            x: n as f64,
                            ms: t,
                            gb_s: None,
                            tflops: Some(flops(n) / (t * 1e-3) / 1e12),
                            c_of_m: None,
                            out_fnv: None,
                            row_fnv: None,
                            j: (!js.is_empty()).then(|| median(&mut js)),
                        }
                    })
                    .collect();
                let fit_points: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.ms)).collect();
                ArmCurve {
                    label: label.clone(),
                    points,
                    fit: fit_affine(&fit_points),
                    unsupported: None,
                }
            })
            .collect();
        notes.push(
            "ms is wall per matmul (prep, the overlapped halves, conversion and the final wait); steel is Steel on every row"
                .into(),
        );
        let mut subject_map = BTreeMap::new();
        subject_map.insert("rows".into(), json!(rows));
        subject_map.insert("cols".into(), json!(cols));
        subject_map.insert("tensor".into(), json!(tensor));
        subject_map.insert("placement".into(), Value::Object(placement));
        subject_map.insert("accuracy".into(), Value::Object(accuracy));
        subject_map.insert("energy_model".into(), json!(crate::energy::ENERGY_MODEL));
        Ok(BenchResult {
            bench: "ane_matmul".into(),
            case: case.label.clone(),
            subject: subject_map,
            x: Axis::new("tokens", "tokens", "log2"),
            layout: LAYOUT.into(),
            arms: curves,
            break_even: Vec::new(),
            notes,
        })
    }

    /// `E` experts of `g` rows: `E` Steel dispatches against one batched
    /// Core ML program over the stacked planes.
    fn run_experts(
        context: &mut MetalContext,
        case: &Case,
        dir: &Path,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let p = programs_of(case)
            .into_iter()
            .next()
            .ok_or("experts case without a program")?;
        let (e, g, k, rows) = (p.experts, p.group, p.k, p.rows);
        if e == 0 || g == 0 || k % 64 != 0 || rows % 32 != 0 {
            return Err(format!(
                "experts program {}: E={e} g={g} K={k} rows={rows}",
                p.package
            ));
        }
        let reps = opts.reps;
        let expert_bytes = rows * k / 2 + rows * k / 64 * 4;
        let copies = copies_for(e * expert_bytes, 64 << 20).min(MAX_PLANE_COPIES);
        // copies x experts resident blobs.
        let mats: Vec<Vec<_>> = (0..copies)
            .map(|c| {
                (0..e)
                    .map(|i| {
                        let (data, scales, biases) =
                            int4_blob(rows, k, 0xE7_0000 + (c * e + i) as u64);
                        (context.new_buffer_with_data(&data), scales, biases)
                    })
                    .collect()
            })
            .collect();
        let steel_ok = {
            let (buffer, scales, biases) = &mats[0][0];
            steel_supports(
                &Int4ResidentMatrix {
                    buffer,
                    weights_offset: 0,
                    scales_offset: *scales,
                    biases_offset: *biases,
                    rows,
                    cols: k,
                },
                SteelTile::M32K32N32,
            )
        };
        let wplanes: Vec<Plane> = (0..copies)
            .map(|_| Plane::new(context, e * rows, k))
            .collect::<Result<_, _>>()?;
        let aplane = Plane::new(context, e * g, k)?;
        let oplane = Plane::new(context, e * rows, g)?;
        let x = context.new_buffer_with_data(&f16_activations(e * g * k, 11));
        let y = context.new_output_buffer((e * g * rows * 2) as u64);
        let inv = context.new_output_buffer((e * g * 4) as u64);
        let nonfinite = context.new_buffer_with_data(&[0u32]);
        let mut notes = Vec::new();
        let loaded = match load_program(dir, &p.package) {
            Ok(l) => Some(l),
            Err(err) => {
                notes.push(format!("ane not loaded: {err}"));
                None
            }
        };
        let joules = Joules::open();
        let mut samples: BTreeMap<&str, Samples> = BTreeMap::new();
        for rep in 0..=reps {
            for label in ["steel", "ane"] {
                if label == "steel" && !steel_ok {
                    continue;
                }
                if label == "ane" && loaded.is_none() {
                    continue;
                }
                let (wall, j) = joules.around(|| -> Result<f64, String> {
                    let t0 = Instant::now();
                    for (c, experts) in mats.iter().enumerate() {
                        autorelease_pool(|| -> Result<(), String> {
                            let w_of = |i: usize| {
                                let (buffer, scales, biases) = &experts[i];
                                Int4ResidentMatrix {
                                    buffer,
                                    weights_offset: 0,
                                    scales_offset: *scales,
                                    biases_offset: *biases,
                                    rows,
                                    cols: k,
                                }
                            };
                            if label == "steel" {
                                let pass = context.begin_pass();
                                for i in 0..e {
                                    encode_dequant_int4_gemm_steel_rows(
                                        context,
                                        &pass,
                                        &w_of(i),
                                        0,
                                        (&x, (i * g * k * 2) as u64),
                                        (&y, (i * g * rows * 2) as u64),
                                        rows,
                                        g,
                                    )
                                    .map_err(|e| e.to_string())?;
                                }
                                pass.commit_and_wait();
                                return Ok(());
                            }
                            let wp = &wplanes[c];
                            let prep = context.begin_pass();
                            for i in 0..e {
                                encode_ane_dequant_rows(
                                    context,
                                    &prep,
                                    &w_of(i),
                                    rows,
                                    (&wp.metal, (i * rows * wp.stride() * 2) as u64),
                                    wp.stride(),
                                )
                                .map_err(|e| e.to_string())?;
                            }
                            encode_ane_scale_rows(
                                context,
                                &prep,
                                (&x, 0),
                                k,
                                e * g,
                                (&aplane.metal, 0),
                                aplane.stride(),
                                &inv,
                            )
                            .map_err(|e| e.to_string())?;
                            prep.commit().wait();
                            let l = loaded.as_ref().expect("checked");
                            l.model
                                .predict_shared(
                                    &[("a", &aplane.plane), ("w", &wp.plane)],
                                    &[("o0", &oplane.plane)],
                                )
                                .map_err(|e| e.to_string())?;
                            let conv = context.begin_pass();
                            encode_ane_convert_out(
                                context,
                                &conv,
                                (&oplane.metal, 0),
                                e * rows,
                                g,
                                oplane.stride(),
                                &inv,
                                (&y, 0),
                                e * rows,
                                &nonfinite,
                            )
                            .map_err(|e| e.to_string())?;
                            conv.commit_and_wait();
                            Ok(())
                        })?;
                    }
                    Ok(t0.elapsed().as_secs_f64())
                })?;
                let entry = samples.entry(label).or_default();
                if rep > 0 {
                    entry.0.push(wall * 1e3 / copies as f64);
                    if let Some(j) = j {
                        entry.1.push(j / copies as f64);
                    }
                }
                if opts.hash && rep == reps {
                    entry.2 = Some(fnv_hex(&read_buffer_bytes(&y, 0, e * g * rows * 2)));
                }
            }
        }
        let flops = 2.0 * (e * g * rows * k) as f64;
        let mut arms = Vec::new();
        for label in ["steel", "ane"] {
            match samples.get(label) {
                Some((ms, js, hash)) if !ms.is_empty() => {
                    let mut ms = ms.clone();
                    let mut js = js.clone();
                    let t = median(&mut ms);
                    arms.push(ArmCurve {
                        label: label.into(),
                        points: vec![Point {
                            x: (e * g) as f64,
                            ms: t,
                            gb_s: None,
                            tflops: Some(flops / (t * 1e-3) / 1e12),
                            c_of_m: None,
                            out_fnv: hash.clone(),
                            row_fnv: None,
                            j: (!js.is_empty()).then(|| median(&mut js)),
                        }],
                        fit: None,
                        unsupported: None,
                    });
                }
                _ => arms.push(ArmCurve::unsupported(
                    label,
                    if label == "steel" {
                        "this shape does not fit the Steel tile"
                    } else {
                        "program not loaded"
                    },
                )),
            }
        }
        let placement = loaded.as_ref().map(|l| {
            json!({"ane_cost_share": l.ane_share,
                   "planned_on": if placed_on_ane(l.ane_share) { "ane" } else { "cpu" }})
        });
        if let Some(l) = &loaded {
            if !placed_on_ane(l.ane_share) {
                notes.push(format!(
                    "Core ML planned the packed program on the CPU (ANE cost share {:.2}): the timing is a CPU timing",
                    l.ane_share.max(0.0)
                ));
            }
        }
        notes.push("steel is E dispatches of g rows; ane is one batched program over stacked planes; the two output layouts differ, so their hashes are not comparable".into());
        let mut subject = BTreeMap::new();
        subject.insert("experts".into(), json!(e));
        subject.insert("group".into(), json!(g));
        subject.insert("k".into(), json!(k));
        subject.insert("rows".into(), json!(rows));
        subject.insert("role".into(), json!(p.role));
        subject.insert("copies".into(), json!(copies));
        subject.insert("placement".into(), placement.unwrap_or(Value::Null));
        subject.insert("energy_model".into(), json!(crate::energy::ENERGY_MODEL));
        Ok(BenchResult {
            bench: "ane_matmul".into(),
            case: case.label.clone(),
            subject,
            x: Axis::new("tokens", "tokens", "log2"),
            layout: LAYOUT.into(),
            arms,
            break_even: Vec::new(),
            notes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ane-prefill")]
    #[test]
    fn output_comparison_separates_ane_rows_from_steel_rows() {
        use super::enabled::compare_outputs;
        use half::f16;
        let (rows, ma, n) = (4, 2, 3);
        let bytes = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes())
                .collect()
        };
        let reference: Vec<f32> = (0..rows * n).map(|i| 1.0 + i as f32).collect();
        let same = compare_outputs(&bytes(&reference), &bytes(&reference), rows, ma, n);
        assert_eq!((same.max_abs, same.rel_rms), (0.0, 0.0));
        assert!(same.steel_rows_identical && !same.nonfinite);
        // Row 1 is an ANE row: it shows up as error, not as a Steel mismatch.
        let mut ane = reference.clone();
        ane[1] += 0.5;
        let a = compare_outputs(&bytes(&reference), &bytes(&ane), rows, ma, n);
        assert_eq!(a.max_abs, 0.5);
        assert!(a.rel_rms > 0.0 && a.steel_rows_identical);
        // Row 3 is a Steel row: any change there must be flagged.
        let mut steel = reference.clone();
        steel[3] += 0.5;
        let s = compare_outputs(&bytes(&reference), &bytes(&steel), rows, ma, n);
        assert!(!s.steel_rows_identical && s.max_abs == 0.0);
        let mut bad = reference;
        bad[0] = f32::INFINITY;
        assert!(compare_outputs(&bytes(&bad), &bytes(&bad), rows, ma, n).nonfinite);
    }

    #[cfg(feature = "ane-prefill")]
    #[test]
    fn arm_labels_separate_f16_const_and_split_programs() {
        use super::enabled::{is_const_arm, label_of, tsplit_half};
        let p = |weights: &str, share: f64| Program {
            weights: weights.into(),
            tensor: String::new(),
            kind: "dense".into(),
            k: 5120,
            rows: 17408,
            ane_rows: 17408,
            tokens: 512,
            share,
            package: String::new(),
            role: String::new(),
            experts: 0,
            group: 0,
        };
        assert_eq!(label_of(&p("", 1.0)), "ane");
        assert_eq!(label_of(&p("", 0.5)), "split@0.5");
        assert_eq!(label_of(&p(CONST_INT4, 1.0)), "ane_c4");
        assert_eq!(label_of(&p(CONST_INT4, 0.6)), "split_c4@0.6");
        for l in ["ane_c4", "split_c4@0.6"] {
            assert!(is_const_arm(l), "{l}");
        }
        for l in ["ane", "split@0.5", "steel", "tsplit"] {
            assert!(!is_const_arm(l), "{l}");
        }
        // The token split needs an even count whose half has a program.
        assert_eq!(tsplit_half(2048, &[512, 1024, 2048]), Some(1024));
        assert_eq!(tsplit_half(512, &[512, 1024, 2048]), None);
        assert_eq!(tsplit_half(1025, &[512, 1024, 2048]), None);
    }

    #[test]
    fn manifest_marks_const_weight_programs() {
        let dir = std::env::temp_dir().join(format!("tslab-ane-c4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("sweep.json"),
            r#"{"format":"ane-sweep-v1","programs":[
                {"kind":"dense","k":5120,"rows":17408,"ane_rows":17408,"tokens":512,"share":1.0,"package":"a.mlpackage"},
                {"kind":"dense","weights":"const_int4","tensor":"layers.0.mlp.gate_proj.weight","k":5120,"rows":17408,"ane_rows":17408,"tokens":512,"share":1.0,"package":"c.mlpackage"}]}"#,
        )
        .unwrap();
        let m = read_manifest(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(m[0].weights, "");
        assert_eq!(m[1].weights, CONST_INT4);
        assert_eq!(
            (m[0].tensor.as_str(), m[1].tensor.as_str()),
            ("", "layers.0.mlp.gate_proj.weight")
        );
    }

    #[test]
    fn manifest_reads_and_generator_line_names_the_shapes() {
        let dir = std::env::temp_dir().join(format!("tslab-ane-sweep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("sweep.json"),
            r#"{"format":"ane-sweep-v1","programs":[
                {"kind":"dense","k":5120,"rows":17408,"ane_rows":17408,"plane_rows":17408,"tokens":512,"share":1.0,"package":"a.mlpackage"},
                {"kind":"experts","role":"gateup","experts":8,"group":64,"k":2048,"rows":512,"package":"e.mlpackage"}]}"#,
        )
        .unwrap();
        let m = read_manifest(&dir).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(
            (m[0].rows, m[0].k, m[0].tokens, m[0].share),
            (17408, 5120, 512, 1.0)
        );
        assert_eq!(
            (m[1].experts, m[1].group, m[1].role.as_str()),
            (8, 64, "gateup")
        );
        std::fs::write(dir.join("sweep.json"), r#"{"format":"other"}"#).unwrap();
        assert!(read_manifest(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let shapes = vec![super::super::MatShape {
            rows: 17408,
            cols: 5120,
            roles: vec![],
            count: 0,
        }];
        let line = generator_line(&shapes, Some((2048, 512)), "OUT");
        assert!(line.contains("--shapes 17408x5120"));
        assert!(line.contains("--expert-dims 2048,512"));
        assert!(line.contains("gen_ane_sweep_programs.py OUT"));
    }
}
