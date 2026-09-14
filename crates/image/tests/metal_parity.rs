//! Opt-in packed Metal gates for Z-Image-Turbo IG2.
//!
//! Set `TURBOSPARK_IMAGE_INSTALL_DIR` to a complete install produced by the
//! image packer, then run these tests with `--ignored --nocapture`. Keeping
//! the real install outside the repository is deliberate: the pinned source
//! is tens of gigabytes and the checked packed copy needs its own disk budget.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::Instant;

use turbospark_image::{
    generate, CancellationToken, FlowMatchEulerScheduler, ImageBackend, ImageRequest, ImageStage,
    MetalImageBackend, IMAGE_GUIDANCE, IMAGE_HEIGHT, IMAGE_QUANTIZATION, IMAGE_STEPS, IMAGE_WIDTH,
};

const PROMPT: &str =
    "A lighthouse in winter at dusk, warm windows reflected on wet snow, cold blue shadows.";
const MODEL_REVISION: &str = "f332072aa78be7aecdf3ee76d5c247082da564a6";
const TEXT_REL_L2_LIMIT: f32 = 0.015;
const ROLLOUT_REL_L2_LIMIT: f32 = 0.196;
const VAE_MAX_ABS_LIMIT: f32 = 6e-5;
const VAE_REL_L2_LIMIT: f32 = 3e-6;

#[test]
#[ignore = "opt-in complete packed Metal image parity gate"]
fn packed_native_matches_conditioning_all_updates_and_decoded_output() {
    let root = image_install();
    let mut backend = MetalImageBackend::open(&root).expect("open packed Metal image install");
    let cancellation = CancellationToken::new();
    let mut conditioning_progress = |_, _| {};
    let conditioning = backend
        .encode_conditioning(PROMPT, 512, &cancellation, &mut conditioning_progress)
        .expect("native conditioning");
    let expected_conditioning = read_fixture("conditioning.npy");
    assert_eq!(conditioning.len(), expected_conditioning.len());
    assert!(
        relative_l2(&conditioning, &expected_conditioning) <= TEXT_REL_L2_LIMIT,
        "native packed conditioning drift exceeds {TEXT_REL_L2_LIMIT}"
    );

    let request = request();
    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(IMAGE_STEPS as usize);
    let steps = backend
        .denoise_steps(&conditioning, &request, &scheduler, &cancellation)
        .expect("native nine-step denoise");
    assert_eq!(steps.len(), 9, "native denoise must perform nine updates");
    for (step, latent) in steps.iter().enumerate() {
        let expected = read_fixture(&format!("latent_{step:02}.npy"));
        assert_eq!(latent.len(), expected.len());
        let error = relative_l2(latent, &expected);
        assert!(
            error <= ROLLOUT_REL_L2_LIMIT,
            "native packed rollout step {} relative L2 {error} exceeds {ROLLOUT_REL_L2_LIMIT}",
            step + 1
        );
    }
    let final_latents = steps.last().expect("nine-step trace has a final latent");
    let expected_final = read_fixture("final_latents.npy");
    assert!(relative_l2(final_latents, &expected_final) <= ROLLOUT_REL_L2_LIMIT);

    let decoded = backend
        .decode(
            final_latents,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            &cancellation,
            &mut |_, _| {},
        )
        .expect("native VAE decode");
    let expected_pixels = read_fixture("decoded_pixels.npy");
    assert_eq!(decoded.len(), expected_pixels.len());
    let max_abs = decoded
        .iter()
        .zip(&expected_pixels)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let rel_l2 = relative_l2(&decoded, &expected_pixels);
    assert!(max_abs <= VAE_MAX_ABS_LIMIT, "VAE max abs error {max_abs}");
    assert!(rel_l2 <= VAE_REL_L2_LIMIT, "VAE relative L2 error {rel_l2}");
}

#[test]
#[ignore = "opt-in complete packed Metal PNG metadata gate"]
fn packed_native_png_contains_the_frozen_request_metadata() {
    let root = image_install();
    let mut backend = MetalImageBackend::open(&root).expect("open packed Metal image install");
    let cancellation = CancellationToken::new();
    let result = generate(&mut backend, &request(), &cancellation, |event| {
        eprintln!(
            "native image {:?}: {}/{}",
            event.stage, event.completed, event.total
        );
    })
    .expect("native image generation");
    assert_eq!(result.metadata.seed, 42);
    assert_eq!(result.metadata.scheduler_steps, IMAGE_STEPS);
    assert_eq!(result.metadata.transformer_forwards, 9);
    assert_eq!(result.metadata.guidance_scale, IMAGE_GUIDANCE);
    assert_eq!(result.metadata.model_revision, MODEL_REVISION);
    assert!(result.png.starts_with(b"\x89PNG\r\n\x1a\n"));
    let metadata = serde_json::to_vec(&result.metadata).expect("serialize returned metadata");
    assert!(
        result
            .png
            .windows(metadata.len())
            .any(|window| window == metadata),
        "PNG must carry the exact serialized request metadata"
    );
}

#[test]
#[ignore = "opt-in quiet-machine packed Metal quality and resource oracle"]
fn packed_native_quality_and_resource_oracle() {
    let root = image_install();
    let mut backend = MetalImageBackend::open(&root).expect("open packed Metal image install");
    let expected_pixels = read_fixture("decoded_pixels.npy");
    let expected_rgb = turbospark_image::vae::decoded_to_rgb8(&expected_pixels, 1024, 1024)
        .expect("expected pixels convert to RGB");
    let mut measurements = Vec::new();
    for label in ["cold", "warm"] {
        let measurement = measure_generation(&mut backend, label);
        let actual_pixels = decode_png_rgb(&measurement.result.png);
        let quality_error = relative_l2_u8(&actual_pixels, &expected_rgb);
        println!(
            "resource report: arm={label} latency_ms={:.3} forward_count=9 phys_footprint_bytes={:?} stage_peak_phys_footprint={:?} managed_allocations={} retained_buffers=0_at_idle physical_reads_pageins={:?} swap_used_before={:?} swap_used_after={:?} swap_delta={:?} png_relative_l2={quality_error:.6e}",
            measurement.elapsed_ms,
            measurement.peak,
            measurement.stage_peaks,
            measurement.allocations,
            measurement.physical_reads,
            measurement.before.swap_used,
            measurement.after.swap_used,
            measurement.swap_delta,
        );
        if let Some(limit) = optional_u64("TURBOSPARK_IMAGE_MAX_PHYS_FOOTPRINT") {
            assert!(
                measurement.peak.unwrap_or(u64::MAX) <= limit,
                "{label} phys_footprint exceeds {limit}"
            );
        }
        if let Some(limit) = optional_u64("TURBOSPARK_IMAGE_MAX_ALLOCATIONS") {
            assert!(
                measurement.allocations <= limit,
                "{label} Metal allocations {} exceed {limit}",
                measurement.allocations
            );
        }
        if let Some(limit) = optional_f32("TURBOSPARK_IMAGE_MAX_PNG_REL_L2") {
            assert!(
                quality_error <= limit,
                "{label} PNG quality error {quality_error} exceeds {limit}"
            );
        }
        measurements.push((measurement, quality_error));
    }
    assert_eq!(
        measurements[0].0.result.png, measurements[1].0.result.png,
        "cold and warm native generations must be deterministic"
    );
}

struct GenerationMeasurement {
    result: turbospark_image::ImageResult,
    elapsed_ms: f64,
    peak: Option<u64>,
    stage_peaks: [Option<u64>; 4],
    allocations: u64,
    before: ResourceSnapshot,
    after: ResourceSnapshot,
    physical_reads: Option<u64>,
    swap_delta: Option<i128>,
}

fn measure_generation(backend: &mut MetalImageBackend, _label: &str) -> GenerationMeasurement {
    let before = resource_snapshot();
    let allocations_before = backend.buffer_allocation_count();
    let monitor = PeakMonitor::start(before.footprint);
    let started = Instant::now();
    let cancellation = CancellationToken::new();
    let generated = generate(backend, &request(), &cancellation, |event| {
        monitor.set_stage(event.stage);
    });
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let after = resource_snapshot();
    let (peak, stage_peaks) = monitor.finish();
    let result = generated.expect("native image");
    GenerationMeasurement {
        result,
        elapsed_ms,
        peak: peak.or(after.footprint).or(before.footprint),
        stage_peaks,
        allocations: backend.buffer_allocation_count() - allocations_before,
        before,
        after,
        physical_reads: delta(before.pageins, after.pageins),
        swap_delta: signed_delta(before.swap_used, after.swap_used),
    }
}

fn request() -> ImageRequest {
    ImageRequest {
        model_id: "Tongyi-MAI/Z-Image-Turbo".to_string(),
        model_revision: MODEL_REVISION.to_string(),
        component_revisions: BTreeMap::new(),
        prompt: PROMPT.to_string(),
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
        batch: 1,
        scheduler_steps: IMAGE_STEPS,
        guidance_scale: IMAGE_GUIDANCE,
        seed: 42,
        quantization: IMAGE_QUANTIZATION.to_string(),
        noise_provenance: "zimage_metal_xorshift_box_muller_v1".to_string(),
    }
}

fn image_install() -> PathBuf {
    std::env::var_os("TURBOSPARK_IMAGE_INSTALL_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .unwrap_or_else(|| {
            panic!("set TURBOSPARK_IMAGE_INSTALL_DIR to a complete packed image install")
        })
}

fn read_fixture(name: &str) -> Vec<f32> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/ig0/runs/lighting")
        .join(name);
    turbospark_image::fixtures::read_npy_file_f32(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
        .data
}

fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let (diff, norm) =
        actual
            .iter()
            .zip(expected)
            .fold((0.0f64, 0.0f64), |(diff, norm), (actual, expected)| {
                let delta = *actual as f64 - *expected as f64;
                (
                    diff + delta * delta,
                    norm + (*expected as f64) * (*expected as f64),
                )
            });
    (diff.sqrt() / norm.sqrt().max(1e-30)) as f32
}

fn relative_l2_u8(actual: &[u8], expected: &[u8]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let (diff, norm) =
        actual
            .iter()
            .zip(expected)
            .fold((0.0f64, 0.0f64), |(diff, norm), (actual, expected)| {
                let delta = *actual as f64 - *expected as f64;
                (
                    diff + delta * delta,
                    norm + (*expected as f64) * (*expected as f64),
                )
            });
    (diff.sqrt() / norm.sqrt().max(1e-30)) as f32
}

fn decode_png_rgb(png: &[u8]) -> Vec<u8> {
    let decoder = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = decoder.read_info().expect("read generated PNG info");
    let mut pixels = vec![0; reader.output_buffer_size().expect("PNG output size")];
    let info = reader
        .next_frame(&mut pixels)
        .expect("decode generated PNG");
    assert_eq!(info.color_type, png::ColorType::Rgb);
    pixels[..info.buffer_size()].to_vec()
}

fn optional_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn optional_f32(name: &str) -> Option<f32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

#[derive(Clone, Copy, Debug, Default)]
struct ResourceSnapshot {
    footprint: Option<u64>,
    pageins: Option<u64>,
    swap_used: Option<u64>,
}

#[repr(C)]
#[derive(Default)]
struct TaskVmInfo {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
}

#[repr(C)]
#[derive(Default)]
struct TaskEventsInfo {
    faults: i32,
    pageins: i32,
    cow_faults: i32,
    messages_sent: i32,
    messages_received: i32,
    syscalls_mach: i32,
    syscalls_unix: i32,
    csw: i32,
}

#[repr(C)]
#[derive(Default)]
struct XswUsage {
    total: u64,
    available: u64,
    used: u64,
    page_size: u32,
    encrypted: i32,
}

const TASK_VM_INFO: u32 = 22;
const TASK_EVENTS_INFO: u32 = 2;

unsafe extern "C" {
    static mach_task_self_: u32;
    fn task_info(task: u32, flavor: u32, info: *mut i32, count: *mut u32) -> i32;
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

fn resource_snapshot() -> ResourceSnapshot {
    ResourceSnapshot {
        footprint: footprint(),
        pageins: pageins(),
        swap_used: swap_used(),
    }
}

fn footprint() -> Option<u64> {
    unsafe {
        let mut info = TaskVmInfo::default();
        let mut count = (std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<i32>()) as u32;
        (task_info(
            mach_task_self_,
            TASK_VM_INFO,
            (&mut info as *mut TaskVmInfo).cast(),
            &mut count,
        ) == 0)
            .then_some(info.phys_footprint)
    }
}

fn pageins() -> Option<u64> {
    unsafe {
        let mut info = TaskEventsInfo::default();
        let mut count = (std::mem::size_of::<TaskEventsInfo>() / std::mem::size_of::<i32>()) as u32;
        (task_info(
            mach_task_self_,
            TASK_EVENTS_INFO,
            (&mut info as *mut TaskEventsInfo).cast(),
            &mut count,
        ) == 0)
            .then_some(info.pageins.max(0) as u64)
    }
}

fn swap_used() -> Option<u64> {
    unsafe {
        let mut usage = XswUsage::default();
        let mut length = std::mem::size_of::<XswUsage>();
        (sysctlbyname(
            c"vm.swapusage".as_ptr(),
            (&mut usage as *mut XswUsage).cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        ) == 0)
            .then_some(usage.used)
    }
}

fn delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    after?.checked_sub(before?)
}

fn signed_delta(before: Option<u64>, after: Option<u64>) -> Option<i128> {
    Some(i128::from(after?) - i128::from(before?))
}

struct PeakMonitor {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    stage: Arc<AtomicUsize>,
    stage_peaks: Arc<[AtomicU64; 4]>,
    thread: Option<thread::JoinHandle<()>>,
}

impl PeakMonitor {
    fn start(initial: Option<u64>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(initial.unwrap_or(0)));
        let stage = Arc::new(AtomicUsize::new(0));
        let stage_peaks = Arc::new(std::array::from_fn(|_| AtomicU64::new(0)));
        let thread_stop = Arc::clone(&stop);
        let thread_peak = Arc::clone(&peak);
        let thread_stage = Arc::clone(&stage);
        let thread_stage_peaks = Arc::clone(&stage_peaks);
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                if let Some(value) = footprint() {
                    thread_peak.fetch_max(value, Ordering::Relaxed);
                    thread_stage_peaks[thread_stage.load(Ordering::Relaxed)]
                        .fetch_max(value, Ordering::Relaxed);
                }
                thread::sleep(std::time::Duration::from_millis(100));
            }
        });
        Self {
            stop,
            peak,
            stage,
            stage_peaks,
            thread: Some(thread),
        }
    }

    fn set_stage(&self, stage: ImageStage) {
        self.stage.store(stage_index(stage), Ordering::Release);
    }

    fn finish(mut self) -> (Option<u64>, [Option<u64>; 4]) {
        self.stop.store(true, Ordering::Release);
        let joined = self
            .thread
            .take()
            .map(|thread| thread.join().is_ok())
            .unwrap_or(false);
        let peak = (joined && self.peak.load(Ordering::Relaxed) != 0)
            .then_some(self.peak.load(Ordering::Relaxed));
        let stage_peaks =
            std::array::from_fn(
                |index| match self.stage_peaks[index].load(Ordering::Relaxed) {
                    0 => None,
                    value => Some(value),
                },
            );
        (peak, stage_peaks)
    }
}

fn stage_index(stage: ImageStage) -> usize {
    match stage {
        ImageStage::TextEncoder => 0,
        ImageStage::Transformer => 1,
        ImageStage::VaeDecoder => 2,
        ImageStage::PngEncode => 3,
    }
}

impl Drop for PeakMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
