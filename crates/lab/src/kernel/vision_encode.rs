//! The vision tower, timed per stage: `RealForwardRunner::encode_image_with_stages`
//! over three synthetic pages, with block-stream reads kept apart from GPU
//! compute and the peak `phys_footprint` beside each page.
//!
//! The text phases (`tslab profile`) count text tokens only, so an image
//! prompt's tower time hides inside TTFT there. This bench answers the two
//! questions an accelerator decision needs first: how long the tower takes
//! per page, and whether that time is compute (`block_gpu`) or the block
//! stream (`block_read`, the host `pread` of each block from
//! `packed_vision/` into one of two slots, which does not overlap compute
//! today). A tower whose `block_read` share is large gains nothing from a
//! faster block kernel, on the GPU or the Neural Engine.
//!
//! An exception to "shapes from the install", said here as the rules ask:
//! the three page sizes are a fixed corpus, the sizes
//! `crates/bench/tests/vision_memory_oracle.rs` reads (1536x1536, 1024x1280,
//! 512x640). The grid each becomes is the install's: preprocessing takes
//! the patch and merge geometry from `preprocessor_config.json` when the
//! install ships one, else from the manifest's vision config with the
//! oracle's pixel bounds. Pages are deterministic noise over a ruled
//! pattern, not text: the tower's cost depends on the grid, not the pixels.
//!
//! Each rep runs every page (largest first, then rotated), and the first
//! rep, which also opens the tower and warms the block files into the page
//! cache, is discarded and reported apart as `cold`. Warm reads are page
//! cache copies; a cold read off the SSD is the first rep's number, one
//! sample, and says so. The stage readbacks this path adds are measured and
//! left out of `total`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use gpu::MetalContext;
use serde_json::{json, Value};

use super::support::median;
use super::{BenchOpts, Case, KernelBench, Subject};
use crate::bench::{fnv_hex, ArmCurve, Axis, BenchResult, Point};

pub const NAME: &str = "vision_encode";

/// `(width, height)` of the corpus pages, largest first: the
/// `vision_memory_oracle` sizes.
pub const PAGES: [(usize, usize); 3] = [(1536, 1536), (1024, 1280), (512, 640)];

/// Context the runner opens with: the tower needs no KV, and a small
/// window keeps the trunk's KV out of the footprint reading.
const OPEN_CONTEXT: usize = 512;

/// Footprint sampling period while a page encodes.
const SAMPLE_EVERY: std::time::Duration = std::time::Duration::from_millis(2);

/// Arms in print order: the whole call less the diagnostic readbacks, then
/// its parts.
pub const ARMS: [&str; 10] = [
    "total",
    "block_read",
    "block_gpu",
    "block_host",
    "patch_embed_gpu",
    "deepstack_gpu",
    "merger_gpu",
    "host_prep",
    "scratch",
    "readback",
];

pub struct VisionEncode;

impl KernelBench for VisionEncode {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> &'static str {
        "the vision tower per stage on three synthetic pages: block stream read against GPU compute, patches/s, peak phys_footprint"
    }

    fn cases(&self, subject: &Subject, opts: &BenchOpts) -> Result<Vec<Case>, String> {
        let v = tower_config(subject, opts)?;
        let mut params = json!({"depth": v.depth, "hidden": v.hidden_size,
                                "intermediate": v.intermediate_size, "heads": v.num_heads,
                                "out_hidden": v.out_hidden_size,
                                "deepstack": v.deepstack_visual_indexes.len(),
                                "pages": PAGES.iter().map(|(w, h)| format!("{w}x{h}")).collect::<Vec<_>>()});
        if let Some(dir) = &opts.vision_sidecar {
            params["sidecar"] = json!(dir.display().to_string());
        }
        Ok(vec![Case {
            label: "tower".into(),
            params,
        }])
    }

    fn run(
        &self,
        _context: &mut MetalContext,
        subject: &Subject,
        case: &Case,
        opts: &BenchOpts,
    ) -> Result<BenchResult, String> {
        let arch = subject.arch()?.clone();
        let t_open = Instant::now();
        let mut runner =
            runtime::RealForwardRunner::open_with_max_context(&subject.dir, arch, OPEN_CONTEXT)
                .map_err(|e| format!("open {}: {e}", subject.dir.display()))?;
        // A sidecar's preprocessor_config.json describes its tower; the
        // trunk's, when it ships one, describes no tower at all.
        let mut preprocess_dir = subject.dir.clone();
        if let Some(dir) = &opts.vision_sidecar {
            runner
                .attach_vision_sidecar(dir)
                .map_err(|e| format!("attach {}: {e}", dir.display()))?;
            if dir.join("preprocessor_config.json").is_file() {
                preprocess_dir = dir.clone();
            }
        }
        let open_ms = t_open.elapsed().as_secs_f64() * 1e3;
        let (params, params_source) = preprocess_params(&preprocess_dir, runner.vision_config())?;

        let mut pages = Vec::new();
        for (i, (w, h)) in page_sizes(case)?.into_iter().enumerate() {
            let t = Instant::now();
            let image =
                turbospark_vision_io::preprocess(&synthetic_page(w, h, 41 + i as u64), &params)
                    .map_err(|e| format!("preprocess {w}x{h}: {e}"))?;
            pages.push(Page {
                label: format!("{w}x{h}"),
                preprocess_ms: t.elapsed().as_secs_f64() * 1e3,
                image,
                samples: Vec::new(),
                peak: 0,
                cold: None,
                out_fnv: None,
            });
        }

        let mut probe = turbospark_bench::memory::AppMemorySampler::new();
        let footprint_open = probe.sample().unwrap_or(0);
        // Rep 0 opens the tower (lazily, on the first image) and is discarded.
        let mut tower_open_ms = 0.0;
        for rep in 0..=opts.reps {
            for k in 0..pages.len() {
                let i = (k + rep) % pages.len();
                let page = &mut pages[i];
                let sampler = PeakSampler::start();
                let t = Instant::now();
                let (embedding, stages) = runner
                    .encode_image_with_stages(&page.image, &params)
                    .map_err(|e| format!("encode {}: {e}", page.label))?;
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let peak = sampler.stop();
                if rep == 0 && k == 0 {
                    tower_open_ms = (wall - stages.timing.total_ms).max(0.0);
                }
                page.peak = page.peak.max(peak);
                if opts.hash && page.out_fnv.is_none() {
                    let bytes: Vec<u8> = embedding
                        .rows
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect();
                    page.out_fnv = Some(fnv_hex(&bytes));
                }
                if rep == 0 {
                    page.cold = Some(stages.timing);
                } else {
                    page.samples.push(stages.timing);
                }
            }
        }

        let v = runner.vision_config().clone();
        let shape = TowerShape {
            hidden: v.hidden_size as usize,
            intermediate: v.intermediate_size as usize,
            depth: v.depth as usize,
        };
        let slot_bytes = runner.vision_slot_bytes().unwrap_or(0);
        Ok(build_result(
            case,
            &pages,
            &shape,
            Context {
                params_source,
                open_ms,
                tower_open_ms,
                footprint_open,
                slot_bytes,
                mapped: pages
                    .first()
                    .and_then(|p| p.samples.first())
                    .is_some_and(|t| t.mapped),
            },
        ))
    }
}

/// The case's `pages` (`WxH` strings), the corpus unless a caller built
/// the case with others (the synthetic-install test does: its tower's
/// 4-pixel patches would make the corpus pages ~147k patches).
pub fn page_sizes(case: &Case) -> Result<Vec<(usize, usize)>, String> {
    let list = case.params["pages"]
        .as_array()
        .ok_or_else(|| format!("{NAME} case {} lacks pages", case.label))?;
    list.iter()
        .map(|v| {
            let s = v.as_str().unwrap_or("");
            let (w, h) = s
                .split_once('x')
                .ok_or_else(|| format!("{NAME} page {s:?}: WxH"))?;
            match (w.parse(), h.parse()) {
                (Ok(w), Ok(h)) if w > 0 && h > 0 => Ok((w, h)),
                _ => Err(format!("{NAME} page {s:?}: WxH")),
            }
        })
        .collect()
}

/// One corpus page and what its reps measured.
struct Page {
    label: String,
    preprocess_ms: f64,
    image: turbospark_vision_io::PreprocessedImage,
    samples: Vec<runtime::vision::VisionTiming>,
    cold: Option<runtime::vision::VisionTiming>,
    /// Highest sampled `phys_footprint` over every rep of this page.
    peak: u64,
    out_fnv: Option<String>,
}

struct TowerShape {
    hidden: usize,
    intermediate: usize,
    depth: usize,
}

impl TowerShape {
    /// FLOPs of the block stack for `seq` patches: qkv, proj, fc1, fc2, and
    /// the two attention matmuls over the whole page (one window).
    fn block_flops(&self, seq: usize) -> f64 {
        let (h, i, s) = (self.hidden as f64, self.intermediate as f64, seq as f64);
        let dense = 2.0 * s * (3.0 * h * h + h * h + 2.0 * h * i);
        let attn = 4.0 * s * s * h;
        self.depth as f64 * (dense + attn)
    }
}

struct Context {
    params_source: &'static str,
    open_ms: f64,
    tower_open_ms: f64,
    footprint_open: u64,
    slot_bytes: u64,
    mapped: bool,
}

/// The stage of `t` an arm reads, in ms.
pub fn stage_ms(t: &runtime::vision::VisionTiming, arm: &str) -> f64 {
    match arm {
        "total" => t.total_ms - t.capture_ms,
        "block_read" => t.block_read_ms,
        "block_gpu" => t.block_gpu_ms,
        // Encode, commit and the wait's host side: wall less GPU.
        "block_host" => (t.block_wall_ms - t.block_gpu_ms).max(0.0),
        "patch_embed_gpu" => t.patch_embed_gpu_ms,
        "deepstack_gpu" => t.deepstack_gpu_ms,
        "merger_gpu" => t.merger_gpu_ms,
        "host_prep" => t.host_prep_ms,
        "scratch" => t.scratch_ms,
        "readback" => t.readback_ms,
        _ => f64::NAN,
    }
}

fn med(samples: &[runtime::vision::VisionTiming], arm: &str) -> f64 {
    let mut v: Vec<f64> = samples.iter().map(|t| stage_ms(t, arm)).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    median(&mut v)
}

fn build_result(case: &Case, pages: &[Page], shape: &TowerShape, ctx: Context) -> BenchResult {
    let mut ordered: Vec<&Page> = pages.iter().collect();
    ordered.sort_by_key(|p| p.image.grid.patches());
    let arms = ARMS
        .iter()
        .map(|&arm| ArmCurve {
            label: arm.into(),
            points: ordered
                .iter()
                .map(|p| {
                    let seq = p.image.grid.patches();
                    let mut pt = Point::new(seq as f64, med(&p.samples, arm));
                    if arm == "block_gpu" {
                        pt.tflops = Some(shape.block_flops(seq) / (pt.ms * 1e-3) / 1e12);
                    }
                    if arm == "total" {
                        pt.out_fnv = p.out_fnv.clone();
                    }
                    pt
                })
                .collect(),
            fit: None,
            unsupported: None,
        })
        .collect();

    let page_rows: Vec<Value> = ordered.iter().map(|p| page_record(p)).collect();
    let mut notes: Vec<String> = ordered
        .iter()
        .map(|p| {
            let total = med(&p.samples, "total");
            let read = med(&p.samples, "block_read");
            let gpu = med(&p.samples, "block_gpu");
            format!(
                "{} ({} patches): total {:.1} ms, {:.0} patches/s, blocks: read {:.1} ms against gpu {:.1} ms ({:.0}% stream), peak {:.0} MiB",
                p.label,
                p.image.grid.patches(),
                total,
                p.image.grid.patches() as f64 / (total * 1e-3),
                read,
                gpu,
                100.0 * stream_share(read, gpu),
                p.peak as f64 / 1048576.0
            )
        })
        .collect();
    notes.push(format!(
        "block reads are {}; the first rep is discarded and recorded as cold",
        if ctx.mapped {
            "mapped (no host read)"
        } else {
            "pread into two slots, serialized with compute"
        }
    ));

    let mut subject: BTreeMap<String, Value> = BTreeMap::new();
    if let Some(o) = case.params.as_object() {
        for (k, v) in o {
            subject.insert(k.clone(), v.clone());
        }
    }
    subject.insert("params_source".into(), json!(ctx.params_source));
    subject.insert("runner_open_ms".into(), json!(ctx.open_ms));
    subject.insert("tower_open_ms".into(), json!(ctx.tower_open_ms));
    subject.insert(
        "footprint_after_open_bytes".into(),
        json!(ctx.footprint_open),
    );
    subject.insert("slot_bytes".into(), json!(ctx.slot_bytes));
    subject.insert(
        "residency".into(),
        json!(if ctx.mapped { "mapped" } else { "pread" }),
    );
    subject.insert("vision_pages".into(), Value::Array(page_rows));

    BenchResult {
        bench: NAME.into(),
        case: case.label.clone(),
        subject,
        x: Axis::new("patches", "patches", "log2"),
        layout: "fp16_vision_tower".into(),
        arms,
        break_even: Vec::new(),
        notes,
    }
}

/// Block read over block read plus block GPU: the share of the block loop a
/// faster kernel cannot touch.
pub fn stream_share(read_ms: f64, gpu_ms: f64) -> f64 {
    let sum = read_ms + gpu_ms;
    if sum > 0.0 {
        read_ms / sum
    } else {
        0.0
    }
}

fn page_record(p: &Page) -> Value {
    let stages: BTreeMap<&str, f64> = ARMS.iter().map(|&a| (a, med(&p.samples, a))).collect();
    let total = stages["total"];
    let cold = p.cold.as_ref().map(|t| {
        json!({"total_ms": stage_ms(t, "total"), "block_read_ms": t.block_read_ms,
               "block_gpu_ms": t.block_gpu_ms})
    });
    json!({
        "page": p.label,
        "resized": [p.image.resized.0, p.image.resized.1],
        "grid": [p.image.grid.t, p.image.grid.h, p.image.grid.w],
        "patches": p.image.grid.patches(),
        "merged_tokens": p.image.merged_tokens,
        "reps": p.samples.len(),
        "preprocess_ms": p.preprocess_ms,
        "stages_ms": stages,
        "patches_per_s": p.image.grid.patches() as f64 / (total * 1e-3),
        "stream_share": stream_share(stages["block_read"], stages["block_gpu"]),
        "peak_footprint_bytes": p.peak,
        "cold": cold,
    })
}

/// The `vision_encode` ledger payload of a result: the per-page stage table
/// `tslab kernel` appends beside the curve. `None` for any other bench.
pub fn ledger_payload(result: &BenchResult) -> Option<Value> {
    if result.bench != NAME {
        return None;
    }
    let mut s = result.subject.clone();
    let pages = s.remove("vision_pages")?;
    Some(json!({"case": result.case, "tower": s, "pages": pages}))
}

/// The tower this run times: the sidecar's when `--vision-sidecar` names one
/// (the trunk must declare none), else the install's own.
fn tower_config(subject: &Subject, opts: &BenchOpts) -> Result<model_io::VisionConfig, String> {
    let arch = subject.arch()?;
    if let Some(dir) = &opts.vision_sidecar {
        if arch.vision.is_active() {
            return Err(format!(
                "{} already declares a vision tower; --vision-sidecar is for a text-only trunk",
                subject.dir.display()
            ));
        }
        let (_, vision) = model_io::load_vision_sidecar(dir)
            .map_err(|e| format!("--vision-sidecar {}: {e}", dir.display()))?;
        return Ok(vision);
    }
    if !arch.vision.is_active() {
        return Err(
            "this install declares no vision tower (stream it with its vision_tower.* tensors, \
             or pass --vision-sidecar DIR)"
                .into(),
        );
    }
    Ok(arch.vision.clone())
}

/// Preprocessing for this install: its `preprocessor_config.json` when it
/// ships one (checked against the manifest's geometry, as the CLI does),
/// else the manifest's geometry with the memory oracle's pixel bounds.
fn preprocess_params(
    dir: &std::path::Path,
    v: &model_io::VisionConfig,
) -> Result<(turbospark_vision_io::PreprocessParams, &'static str), String> {
    let path = dir.join("preprocessor_config.json");
    if let Ok(text) = std::fs::read_to_string(&path) {
        let p = turbospark_vision_io::PreprocessParams::from_preprocessor_config_json(&text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if p.patch_size != v.patch_size as usize
            || p.merge_size != v.spatial_merge_size as usize
            || p.temporal_patch_size != v.temporal_patch_size as usize
        {
            return Err(format!(
                "{}: patch geometry disagrees with the manifest's vision config",
                path.display()
            ));
        }
        return Ok((p, "preprocessor_config"));
    }
    Ok((
        turbospark_vision_io::PreprocessParams {
            patch_size: v.patch_size as usize,
            temporal_patch_size: v.temporal_patch_size as usize,
            merge_size: v.spatial_merge_size as usize,
            in_channels: v.in_channels as usize,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            rescale_factor: 1.0 / 255.0,
        },
        "manifest_oracle_bounds",
    ))
}

/// A deterministic `width x height` page: dark rules every 24 rows over
/// seeded noise, so no two pages share content.
pub fn synthetic_page(width: usize, height: usize, seed: u64) -> turbospark_vision_io::Rgb8Image {
    let mut state = seed | 1;
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for _ in 0..width {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (state >> 59) as u8;
            let base = if y % 24 < 3 { 40 } else { 230 };
            let v = base - noise;
            data.extend_from_slice(&[v, v, v]);
        }
    }
    turbospark_vision_io::Rgb8Image {
        width,
        height,
        data,
    }
}

/// Samples `phys_footprint` on a thread until stopped; returns the peak.
struct PeakSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let (s, p) = (stop.clone(), peak.clone());
        let handle = std::thread::spawn(move || {
            let mut sampler = turbospark_bench::memory::AppMemorySampler::new();
            loop {
                if let Some(b) = sampler.sample() {
                    p.fetch_max(b, Ordering::Relaxed);
                }
                if s.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(SAMPLE_EVERY);
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    fn stop(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.peak.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_share_is_the_read_fraction_of_the_block_loop() {
        assert_eq!(stream_share(0.0, 0.0), 0.0);
        assert!((stream_share(1.0, 3.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn total_leaves_out_the_diagnostic_readbacks() {
        let t = runtime::vision::VisionTiming {
            total_ms: 10.0,
            capture_ms: 2.5,
            block_wall_ms: 4.0,
            block_gpu_ms: 3.0,
            ..Default::default()
        };
        assert_eq!(stage_ms(&t, "total"), 7.5);
        assert_eq!(stage_ms(&t, "block_host"), 1.0);
    }

    #[test]
    fn synthetic_pages_differ_by_seed_and_fill_the_frame() {
        let a = synthetic_page(64, 48, 41);
        let b = synthetic_page(64, 48, 42);
        assert_eq!(a.data.len(), 64 * 48 * 3);
        assert_ne!(a.data, b.data);
    }
}
