//! The host handoff of one ANE+Steel split point, quiet and under CPU
//! contention.
//!
//! A split point is a GPU segment that prepares the planes, a Core ML
//! prediction, and a GPU segment that waits on the host's signal to consume
//! the output (`crates/runtime/src/families/qwen/ane_split.rs`). The host
//! sits between both: it has to see the first segment finish and then send
//! the signal. That cost is `wake + resume`, named as the split's own
//! `TURBOSPARK_ANE_PREFILL_STATS` line names them. This bench times it for
//! one of the install's real programs, in two arms:
//!
//! - `wait`: the split's path, `waitUntilCompleted` on the segment.
//! - `poll`: the segment signals a shared event at its end and the host
//!   spins on its value (`gpu::ane_split::commit_signaling`).
//!
//! `SPEEDUP_PLAYBOOK.md` reopens the asynchronous Core ML and private
//! shared-event ideas only if a controlled-contention run shows handoff
//! tails dominating. The end-to-end reading of that question is the
//! `~hog:N` A/B on a real split; this bench is the microbench that prices
//! the poll alternative against it. `--contend N` is the control: the bench
//! itself spins N threads for the contended points, so the load is known
//! and in-process (the host guard still refuses a foreign neighbour). The
//! x axis is that thread count.
//!
//! The producer runs only the activation scale, not the weight dequant, so
//! its GPU time is not the split's. Whether the handoff depends on the
//! producer's length is not measured here.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::BenchResult;

pub struct AneHandoff;

/// The family whose prefill the split serves (the `ane_prefill` knob's).
const FAMILY: &str = "qwen35";

/// The directories the runtime searches when `TURBOSPARK_ANE_PREFILL_DIR`
/// is unset (`ensure` in the runtime's `ane_split.rs`), in its order.
pub fn default_dirs() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let home = PathBuf::from(home);
    [
        ".turbospark/models/ane-prefill/qwen38",
        ".turbospark/ane-prefill/qwen38",
        "models/ane-prefill/qwen38",
    ]
    .iter()
    .map(|p| home.join(p))
    .collect()
}

/// One program of an ANE prefill `manifest.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub class: String,
    pub k: usize,
    pub tokens: usize,
    pub package: String,
    pub ane_rows: Vec<usize>,
}

impl Program {
    pub fn label(&self) -> String {
        format!("{}_n{}", self.class, self.tokens)
    }
}

/// The manifest's preset and programs. Only what this bench reads; the
/// runtime's own parser checks the shapes against the install.
pub fn parse_manifest(text: &str) -> Result<(String, Vec<Program>), String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("manifest.json: {e}"))?;
    if v["format"] != 1 {
        return Err(format!("manifest format {} is not 1", v["format"]));
    }
    let preset = v["preset"].as_str().unwrap_or("unknown").to_string();
    let programs = v["programs"]
        .as_array()
        .ok_or("manifest.json has no programs")?
        .iter()
        .map(|p| {
            let n = |key: &str| {
                p[key]
                    .as_u64()
                    .map(|x| x as usize)
                    .ok_or(format!("program {key} missing"))
            };
            let ane_rows = p["matrices"]
                .as_array()
                .ok_or("program matrices missing")?
                .iter()
                .map(|m| m["ane_rows"].as_u64().map(|x| x as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or("matrix ane_rows missing")?;
            Ok(Program {
                class: p["class"].as_str().ok_or("program class missing")?.into(),
                k: n("k")?,
                tokens: n("tokens")?,
                package: p["package"]
                    .as_str()
                    .ok_or("program package missing")?
                    .into(),
                ane_rows,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if programs.is_empty() {
        return Err("manifest.json lists no programs".into());
    }
    Ok((preset, programs))
}

/// Split points one micro-batch handles: every layer splits its first
/// projection (q with gate, or GDN qkv and z), o, gate/up and down.
pub fn points_per_microbatch(layers: usize) -> usize {
    4 * layers
}

/// Nearest-rank `q` quantile of samples sorted ascending.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let rank = ((q * n as f64).ceil() as usize).clamp(1, n);
    sorted[rank - 1]
}

fn program_dir(opts: &BenchOpts) -> Result<PathBuf, String> {
    if let Some(d) = &opts.ane_dir {
        return Ok(d.clone());
    }
    default_dirs()
        .into_iter()
        .find(|d| d.join("manifest.json").is_file())
        .ok_or_else(|| {
            "no ANE prefill programs (pass --ane-dir; generate them with \
             crates/coreml/tools/gen_ane_prefill_programs.py)"
                .into()
        })
}

fn read_manifest(dir: &Path) -> Result<(String, Vec<Program>), String> {
    let path = dir.join("manifest.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_manifest(&text)
}

impl KernelBench for AneHandoff {
    fn name(&self) -> &'static str {
        "ane_handoff"
    }

    fn describe(&self) -> &'static str {
        "ANE split point host handoff (wake + resume), wait vs poll, quiet vs --contend N"
    }

    fn cases(&self, subject: &Subject, opts: &BenchOpts) -> Result<Vec<Case>, String> {
        if !cfg!(feature = "ane-prefill") {
            return Err("tslab was built without the ane-prefill feature".into());
        }
        let arch = subject.arch()?;
        if subject.family != FAMILY {
            return Err(format!("the ANE prefill split serves {FAMILY} only"));
        }
        let dir = program_dir(opts)?;
        let (preset, programs) = read_manifest(&dir)?;
        // The runtime refuses a manifest whose shapes are not the install's;
        // this checks the two widths every program set uses.
        let (hidden, inter) = (arch.hidden_size as usize, arch.intermediate_size as usize);
        let ks: Vec<usize> = programs.iter().map(|p| p.k).collect();
        if !ks.contains(&hidden) || !ks.contains(&inter) {
            return Err(format!(
                "{} ({preset}) is not for this install: program K {ks:?}, install D={hidden} F={inter}",
                dir.display()
            ));
        }
        Ok(programs
            .iter()
            .map(|p| Case {
                label: p.label(),
                params: json!({"dir": dir.display().to_string(), "preset": preset,
                               "class": p.class, "k": p.k, "tokens": p.tokens,
                               "package": p.package, "ane_rows": p.ane_rows,
                               "layers": arch.num_layers}),
            })
            .collect())
    }

    #[cfg(feature = "ane-prefill")]
    fn run(
        &self,
        context: &mut gpu::MetalContext,
        _subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        run::run(context, case, opts)
    }

    #[cfg(not(feature = "ane-prefill"))]
    fn run(
        &self,
        _context: &mut gpu::MetalContext,
        _subject: &Subject,
        _case: &Case,
        _opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        Err("tslab was built without the ane-prefill feature".into())
    }
}

#[cfg(feature = "ane-prefill")]
mod run {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use gpu::ane_split::{
        begin_pass_after, commit_signaling, encode_ane_convert_out, encode_ane_scale_rows,
        host_seconds, HostEvent,
    };
    use gpu::{read_buffer_bytes, write_buffer_bytes, MetalBuffer, MetalContext};
    use serde_json::{json, Value};
    use turbospark_coreml::{
        ComputeUnits, CoreMlModel, FeatureKind, FeatureSpec, LoadOptions, SharedF16Plane,
    };

    use super::super::support::{f16_activations, f16_bytes, median};
    use super::super::{BenchOpts, Case};
    use super::{points_per_microbatch, quantile};
    use crate::bench::{ArmCurve, Axis, BenchResult, Point};

    /// Samples per arm per contention level in one repetition: `--reps 5`
    /// gives 200 per point, enough for a nearest-rank p99.
    const SAMPLES_PER_REP: usize = 40;

    /// How long a poll waits for the GPU before calling the run failed.
    const POLL_TIMEOUT: Duration = Duration::from_secs(10);

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        Wait,
        Poll,
    }

    const ARMS: [(Arm, &str); 2] = [(Arm::Wait, "wait"), (Arm::Poll, "poll")];

    /// An IOSurface plane and the no-copy Metal buffer over it. The buffer
    /// is declared first so it drops before the memory it wraps.
    struct Plane {
        buffer: MetalBuffer,
        plane: SharedF16Plane,
    }

    impl Plane {
        fn new(context: &MetalContext, rows: usize, cols: usize) -> Result<Self, String> {
            let plane = SharedF16Plane::new(rows, cols).map_err(|e| format!("Core ML: {e}"))?;
            let buffer = gpu::wrap_page_aligned_no_copy(
                context.device(),
                plane.base_ptr(),
                plane.alloc_size(),
            )
            .map_err(|e| format!("Metal: {e}"))?;
            Ok(Self { buffer, plane })
        }
    }

    fn fixed_shape(spec: &FeatureSpec) -> Result<(usize, usize), String> {
        match &spec.kind {
            FeatureKind::MultiArray { shape: Some(s), .. } if s.len() == 4 => Ok((s[2], s[3])),
            other => Err(format!(
                "{}: not a fixed [1,1,R,C] array ({other:?})",
                spec.name
            )),
        }
    }

    /// CPU spinners for the contended points. Thread `j` spins while
    /// `j < active`, else sleeps, so one set serves every level.
    struct Spinners {
        active: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        threads: Vec<JoinHandle<()>>,
    }

    impl Spinners {
        fn new(n: usize) -> Self {
            let active = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let threads = (0..n)
                .map(|j| {
                    let (active, stop) = (active.clone(), stop.clone());
                    std::thread::spawn(move || {
                        let mut x = j as u64 + 1;
                        while !stop.load(Ordering::Relaxed) {
                            if j < active.load(Ordering::Relaxed) {
                                for _ in 0..10_000 {
                                    x = std::hint::black_box(
                                        x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1),
                                    );
                                }
                            } else {
                                std::thread::sleep(Duration::from_millis(1));
                            }
                        }
                    })
                })
                .collect();
            Self {
                active,
                stop,
                threads,
            }
        }

        /// Sets the number of spinning threads and lets the scheduler see it.
        fn set(&self, n: usize) {
            self.active.store(n, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    impl Drop for Spinners {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            for t in self.threads.drain(..) {
                let _ = t.join();
            }
        }
    }

    /// One split point's host timeline, in seconds.
    #[derive(Default, Clone, Copy)]
    struct Sample {
        /// Producer GPU end to the host seeing it.
        wake: f64,
        predict: f64,
        /// Host signal to the consumer's GPU start.
        resume: f64,
        /// Producer GPU start to consumer GPU end.
        wall: f64,
    }

    struct Rig<'a> {
        model: &'a CoreMlModel,
        a: Plane,
        ws: Vec<(String, Plane)>,
        os: Vec<(String, Plane)>,
        x: MetalBuffer,
        inv: MetalBuffer,
        y: MetalBuffer,
        flag: MetalBuffer,
        k: usize,
        m: usize,
        ready: HostEvent,
        go: HostEvent,
        value: u64,
    }

    impl Rig<'_> {
        fn point(&mut self, context: &mut MetalContext, arm: Arm) -> Result<Sample, String> {
            self.value += 1;
            let v = self.value;
            let g = |r: Result<(), gpu::GpuError>| r.map_err(|e| e.to_string());
            gpu::autorelease_pool(|| -> Result<Sample, String> {
                // Producer first: the consumer waits on the host's signal,
                // and a buffer committed ahead of it on this queue would
                // never start.
                let producer = context.begin_pass();
                g(encode_ane_scale_rows(
                    context,
                    &producer,
                    (&self.x, 0),
                    self.k,
                    self.m,
                    (&self.a.buffer, 0),
                    self.a.plane.stride(),
                    &self.inv,
                ))?;
                let producer = match arm {
                    Arm::Wait => producer.commit(),
                    Arm::Poll => commit_signaling(producer, &self.ready, v),
                };
                let o = &self.os[0].1;
                let consumer = begin_pass_after(context, "ane_handoff.consumer", &self.go, v);
                g(encode_ane_convert_out(
                    context,
                    &consumer,
                    (&o.buffer, 0),
                    o.plane.rows(),
                    self.m,
                    o.plane.stride(),
                    &self.inv,
                    (&self.y, 0),
                    o.plane.rows(),
                    &self.flag,
                ))?;
                let consumer = consumer.commit().waiting_on_drop();

                match arm {
                    Arm::Wait => producer.wait_ref(),
                    Arm::Poll => {
                        let started = Instant::now();
                        while self.ready.value() < v {
                            if started.elapsed() > POLL_TIMEOUT {
                                // Release the consumer so the queue drains.
                                self.go.signal(v);
                                return Err("poll: the producer never signaled".into());
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
                let seen = host_seconds();
                let inputs: Vec<(&str, &SharedF16Plane)> = std::iter::once(("a", &self.a.plane))
                    .chain(self.ws.iter().map(|(n, p)| (n.as_str(), &p.plane)))
                    .collect();
                let outputs: Vec<(&str, &SharedF16Plane)> = self
                    .os
                    .iter()
                    .map(|(n, p)| (n.as_str(), &p.plane))
                    .collect();
                let predicted = self.model.predict_shared(&inputs, &outputs);
                let done = host_seconds();
                // Signal even on failure, so the consumer drains.
                self.go.signal(v);
                let signaled = host_seconds();
                predicted.map_err(|e| format!("prediction: {e}"))?;
                let (p_start, p_end) = producer.wait_gpu_start_end();
                let (c_start, c_end) = consumer.wait_gpu_start_end();
                Ok(Sample {
                    wake: seen - p_end,
                    predict: done - seen,
                    resume: c_start - signaled,
                    wall: c_end - p_start,
                })
            })
        }
    }

    fn summary(samples: &[Sample], points: usize) -> Value {
        let sorted = |f: fn(&Sample) -> f64| {
            let mut v: Vec<f64> = samples.iter().map(f).collect();
            v.sort_by(f64::total_cmp);
            v
        };
        let ms = |v: &[f64], q: f64| quantile(v, q) * 1e3;
        let handoff = sorted(|s| s.wake + s.resume);
        let wake = sorted(|s| s.wake);
        let resume = sorted(|s| s.resume);
        let predict = sorted(|s| s.predict);
        let wall = sorted(|s| s.wall);
        json!({
            "samples": samples.len(),
            "handoff_ms": {"p50": ms(&handoff, 0.5), "p90": ms(&handoff, 0.9),
                           "p99": ms(&handoff, 0.99), "max": ms(&handoff, 1.0)},
            "wake_ms": {"p50": ms(&wake, 0.5), "p99": ms(&wake, 0.99)},
            "resume_ms": {"p50": ms(&resume, 0.5), "p99": ms(&resume, 0.99)},
            "predict_ms_p50": ms(&predict, 0.5),
            "wall_ms_p50": ms(&wall, 0.5),
            // Every point of a micro-batch paying this point's handoff: the
            // mean is the honest projection, p99 an upper bound.
            "per_microbatch_ms": {
                "points": points,
                "at_mean": samples.iter().map(|s| s.wake + s.resume).sum::<f64>()
                    / samples.len() as f64 * points as f64 * 1e3,
                "at_p99": ms(&handoff, 0.99) * points as f64,
            },
        })
    }

    pub(super) fn run(
        context: &mut MetalContext,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let p = &case.params;
        let u = |key: &str| p[key].as_u64().unwrap_or(0) as usize;
        let s = |key: &str| p[key].as_str().unwrap_or("").to_string();
        let (k, tokens, layers) = (u("k"), u("tokens"), u("layers"));
        let dir = std::path::PathBuf::from(s("dir"));
        let package = s("package");
        let model = CoreMlModel::open(
            &dir.join(&package),
            &dir.join("compiled"),
            &LoadOptions::new(ComputeUnits::CpuAndNeuralEngine),
        )
        .map_err(|e| format!("{package}: Core ML: {e}"))?;
        let rig = build_rig(context, &model, k, tokens)?;
        measure(context, rig, case, opts, layers, &dir)
    }

    fn build_rig<'a>(
        context: &MetalContext,
        model: &'a CoreMlModel,
        k: usize,
        tokens: usize,
    ) -> Result<Rig<'a>, String> {
        let mut a = None;
        let mut ws = Vec::new();
        for spec in model.inputs() {
            let (rows, cols) = fixed_shape(spec)?;
            let plane = Plane::new(context, rows, cols)?;
            if spec.name == "a" {
                if (rows, cols) != (tokens, k) {
                    return Err(format!("input a is {rows}x{cols}, want {tokens}x{k}"));
                }
                a = Some(plane);
            } else {
                // Small finite weights; the values do not matter to the
                // handoff, only that the output stays finite.
                let bytes = f16_bytes(
                    (0..plane.plane.alloc_size() / 2).map(|i| ((i % 17) as f32 - 8.0) / 4096.0),
                );
                write_buffer_bytes(&plane.buffer, 0, &bytes);
                ws.push((spec.name.clone(), plane));
            }
        }
        let a = a.ok_or("the program has no input a")?;
        let mut os = Vec::new();
        for spec in model.outputs() {
            let (rows, cols) = fixed_shape(spec)?;
            os.push((spec.name.clone(), Plane::new(context, rows, cols)?));
        }
        os.sort_by(|x, y| x.0.cmp(&y.0));
        let first_rows = os
            .first()
            .ok_or("the program has no output")?
            .1
            .plane
            .rows();
        Ok(Rig {
            model,
            a,
            ws,
            os,
            x: context.new_buffer_with_data(&f16_activations(tokens * k, 7)),
            inv: context.new_output_buffer((tokens * 4) as u64),
            y: context.new_output_buffer((tokens * first_rows * 2) as u64),
            flag: context.new_buffer_with_data(&[0u8; 4]),
            k,
            m: tokens,
            ready: HostEvent::new(context),
            go: HostEvent::new(context),
            value: 0,
        })
    }

    fn measure(
        context: &mut MetalContext,
        mut rig: Rig<'_>,
        case: &Case,
        opts: &BenchOpts,
        layers: usize,
        dir: &Path,
    ) -> Result<BenchResult, String> {
        let levels: Vec<usize> = if opts.contend > 0 {
            vec![0, opts.contend]
        } else {
            vec![0]
        };
        let spinners = Spinners::new(opts.contend);
        // samples[arm][level]
        let mut samples = vec![vec![Vec::new(); levels.len()]; ARMS.len()];
        for rep in 0..=opts.reps {
            // Alternate the level order across repetitions so neither level
            // owns the start of the run.
            let order: Vec<usize> = if rep % 2 == 0 {
                (0..levels.len()).collect()
            } else {
                (0..levels.len()).rev().collect()
            };
            for li in order {
                spinners.set(levels[li]);
                for i in 0..SAMPLES_PER_REP {
                    for j in 0..ARMS.len() {
                        let ai = if i % 2 == 0 { j } else { ARMS.len() - 1 - j };
                        let sample = rig.point(context, ARMS[ai].0)?;
                        // The first repetition warms the program and the
                        // queue, and is discarded.
                        if rep > 0 {
                            samples[ai][li].push(sample);
                        }
                    }
                }
            }
        }
        spinners.set(0);
        drop(spinners);
        if read_buffer_bytes(&rig.flag, 0, 4) != [0u8; 4] {
            return Err("the program wrote non-finite rows".into());
        }
        let points = points_per_microbatch(layers);
        let mut detail = serde_json::Map::new();
        let mut arms = Vec::new();
        for (ai, (_, label)) in ARMS.iter().enumerate() {
            let mut curve = Vec::new();
            for (li, &level) in levels.iter().enumerate() {
                let s = &samples[ai][li];
                let mut handoffs: Vec<f64> = s.iter().map(|x| (x.wake + x.resume) * 1e3).collect();
                curve.push(Point::new(level as f64, median(&mut handoffs)));
                detail.insert(format!("{label}@{level}"), summary(s, points));
            }
            arms.push(ArmCurve {
                label: (*label).into(),
                points: curve,
                fit: None,
                unsupported: None,
            });
        }

        let mut subject = BTreeMap::new();
        for key in [
            "preset", "class", "k", "tokens", "package", "ane_rows", "layers",
        ] {
            subject.insert(key.to_string(), case.params[key].clone());
        }
        subject.insert("dir".into(), json!(dir.display().to_string()));
        subject.insert("contend_threads".into(), json!(opts.contend));
        subject.insert(
            "cpus".into(),
            json!(std::thread::available_parallelism().map_or(0, |n| n.get())),
        );
        subject.insert("detail".into(), Value::Object(detail));
        Ok(BenchResult {
            bench: "ane_handoff".into(),
            case: case.label.clone(),
            subject,
            x: Axis::new("contend", "count", "linear"),
            layout: "coreml_f16_iosurface".into(),
            arms,
            break_even: Vec::new(),
            notes: vec![
                "ms is the median handoff (wake + resume) of one split point, not GPU time".into(),
                "x is spinner threads this process ran during the point; the load is self-made"
                    .into(),
                "per_microbatch_ms multiplies one point by the install's split points; \
                 compare it with a measured split micro-batch, not with this bench's wall"
                    .into(),
                "the producer runs the activation scale only, not the weight dequant".into(),
                "the arms' outputs are not compared, so --hash does nothing here".into(),
                "the host monitor samples around the case, not during the spinners, so \
                 `eligible` does not describe the contended points"
                    .into(),
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"{"format": 1, "preset": "qwen38-27b", "programs": [
        {"class": "down", "k": 17408, "tokens": 2048,
         "matrices": [{"rows": 5120, "ane_rows": 2272, "plane_rows": 2272}],
         "package": "split_k17408_n2048_m2272w2272.mlpackage"},
        {"class": "gateup", "k": 5120, "tokens": 1024,
         "matrices": [{"rows": 17408, "ane_rows": 7168, "plane_rows": 8192},
                      {"rows": 17408, "ane_rows": 7168, "plane_rows": 8192}],
         "package": "g.mlpackage"}]}"#;

    #[test]
    fn manifest_parses_labels_and_rows() {
        let (preset, programs) = parse_manifest(MANIFEST).unwrap();
        assert_eq!(preset, "qwen38-27b");
        let labels: Vec<String> = programs.iter().map(Program::label).collect();
        assert_eq!(labels, ["down_n2048", "gateup_n1024"]);
        assert_eq!(programs[1].ane_rows, [7168, 7168]);
        assert_eq!(programs[0].k, 17408);
    }

    #[test]
    fn manifest_refuses_other_formats_and_gaps() {
        assert!(parse_manifest(&MANIFEST.replace("\"format\": 1", "\"format\": 2")).is_err());
        assert!(parse_manifest(&MANIFEST.replace("\"ane_rows\": 2272, ", "")).is_err());
        assert!(parse_manifest(r#"{"format": 1, "programs": []}"#).is_err());
    }

    #[test]
    fn quantile_is_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(quantile(&v, 0.5), 50.0);
        assert_eq!(quantile(&v, 0.99), 99.0);
        assert_eq!(quantile(&v, 1.0), 100.0);
        assert_eq!(quantile(&[3.0], 0.99), 3.0);
        // Rank never falls below the first sample.
        assert_eq!(quantile(&v, 0.0), 1.0);
    }

    #[test]
    fn a_microbatch_splits_four_points_per_layer() {
        // qwen38-27b: 64 layers, 256 points (SPEEDUP_PLAYBOOK.md).
        assert_eq!(points_per_microbatch(64), 256);
    }

    /// One repetition of both arms at both levels on the real program set:
    /// the plumbing, not a measurement. Needs a dense qwen install and its
    /// programs:
    ///
    /// ```sh
    /// TSLAB_ANE_MODEL=~/models/qwen38-27b.gturbo \
    ///   cargo test -p turbospark-lab --features ane-prefill --lib \
    ///   ane_handoff -- --ignored
    /// ```
    #[cfg(feature = "ane-prefill")]
    #[test]
    #[ignore = "needs a dense qwen install (TSLAB_ANE_MODEL) and its ANE programs"]
    fn runs_both_arms_on_the_real_programs() {
        let model = std::env::var("TSLAB_ANE_MODEL").expect("TSLAB_ANE_MODEL");
        let subject = Subject::open(&model).unwrap();
        let opts = BenchOpts {
            reps: 1,
            contend: 2,
            ..BenchOpts::default()
        };
        let cases = AneHandoff.cases(&subject, &opts).unwrap();
        let case = cases
            .iter()
            .find(|c| c.label == "o_n1024")
            .expect("o_n1024");
        let mut ctx = gpu::MetalContext::new().unwrap();
        let r = AneHandoff.run(&mut ctx, &subject, case, &opts).unwrap();
        for arm in ["wait", "poll"] {
            let pts = &r.arm(arm).unwrap().points;
            assert_eq!(pts.iter().map(|p| p.x).collect::<Vec<_>>(), [0.0, 2.0]);
            assert!(pts.iter().all(|p| p.ms.is_finite()));
        }
        assert_eq!(r.subject["detail"]["wait@0"]["samples"], 40);
    }
}
