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

/// One program of a `sweep.json` manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
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
    use std::time::Instant;

    use gpu::ane_split::{
        begin_pass_after, encode_ane_convert_out, encode_ane_dequant_rows, encode_ane_scale_rows,
        encode_dequant_int4_gemm_steel_rows, HostEvent,
    };
    use gpu::{
        autorelease_pool, read_buffer_bytes, steel_supports, Int4ResidentMatrix, MetalContext,
        SteelTile,
    };
    use serde_json::{json, Value};
    use turbospark_coreml::{
        compile_if_needed, compute_plan, supports, Capability, ComputeUnits, CoreMlModel,
        LoadOptions, SharedF16Plane,
    };

    use super::super::support::{copies_for, f16_activations, int4_blob, median};
    use super::super::{BenchOpts, Case, KernelBench, Subject};
    use super::{
        expert_dims, generator_line, install_shapes, read_manifest, AneMatmul, Program,
        ANE_PLACEMENT_MIN_SHARE, MAX_PLANE_COPIES,
    };
    use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, BreakEven, Point};
    use crate::fit::{ane_gate_fit, break_even, fit_affine, GateCase, GateObjective};

    pub use crate::bench::INT4_AFFINE_LAYOUT as LAYOUT;

    /// One arm at one token count: wall ms samples, joule samples, and the
    /// output hash under `--hash`.
    type Samples = (Vec<f64>, Vec<f64>, Option<String>);

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
        json!({"kind": p.kind, "k": p.k, "rows": p.rows, "ane_rows": p.ane_rows,
               "tokens": p.tokens, "share": p.share, "package": p.package,
               "role": p.role, "experts": p.experts, "group": p.group})
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
                    .filter(|p| p.kind == "dense" && p.rows == s.rows && p.k == s.cols)
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
            _subject: &Subject,
            case: &Case,
            opts: &BenchOpts,
        ) -> Result<BenchResult, String> {
            let dir = Path::new(case.params["dir"].as_str().unwrap_or(""));
            match case.params["kind"].as_str() {
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
        let mut shares: Vec<f64> = programs.iter().map(|p| p.share).collect();
        shares.sort_by(|a, b| b.total_cmp(a));
        shares.dedup();
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
            let label = arm_label(p.share);
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
        arms.extend(shares.iter().map(|&s| arm_label(s)));
        let mut table: BTreeMap<String, BTreeMap<usize, Samples>> = BTreeMap::new();
        for &n in &tokens {
            for rep in 0..=reps {
                for label in &arms {
                    let program = programs
                        .iter()
                        .find(|p| arm_label(p.share) == *label && p.tokens == n);
                    let ma = program.map_or(rows, |p| p.ane_rows);
                    if label == "steel" && !steel_ok {
                        continue;
                    }
                    if label != "steel" && !loaded.contains_key(&(label.clone(), n)) {
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
                                let wp = &wplanes[c];
                                let ap = &aplanes[&n];
                                let op = &oplanes[&(ma, n)];
                                let model = &loaded[&(label.clone(), n)].model;
                                let prep = context.begin_pass();
                                encode_ane_dequant_rows(
                                    context,
                                    &prep,
                                    &w,
                                    ma,
                                    (&wp.metal, 0),
                                    wp.stride(),
                                )
                                .map_err(|e| e.to_string())?;
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
                                let inputs = [("a", &ap.plane), ("w0", &wp.plane)];
                                let outputs = [("o0", &op.plane)];
                                if ma < rows {
                                    // The split: the convert pass waits on the
                                    // host's signal, Steel's rows run meanwhile.
                                    seq += 1;
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
                    label == "steel"
                        || placement
                            .get(&format!("{label}@{}", p.x as usize))
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
