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
    generate, read_npy_f32, read_npz_file, CancellationToken, FlowMatchEulerScheduler,
    ImageBackend, ImageMemoryBudget, ImageRequest, ImageResidency, ImageStage, MetalImageBackend,
    IMAGE_GUIDANCE, IMAGE_HEIGHT, IMAGE_QUANTIZATION, IMAGE_STEPS, IMAGE_WIDTH,
};

const PROMPT: &str =
    "A lighthouse in winter at dusk, warm windows reflected on wet snow, cold blue shadows.";
const MODEL_REVISION: &str = "f332072aa78be7aecdf3ee76d5c247082da564a6";
// These are the frozen INT4-emulation envelopes from IG0, rounded upward to
// the next 0.001. They are quality bounds against the higher-precision
// fixtures, not implementation-parity tolerances.
const PACKED_CONDITIONING_REL_L2_LIMIT: f32 = 0.084;
const PACKED_ROLLOUT_REL_L2_LIMIT: f32 = 0.923;
const VAE_MAX_ABS_LIMIT: f32 = 6e-5;
const VAE_REL_L2_LIMIT: f32 = 3e-6;

#[test]
#[ignore = "opt-in packed Metal quality envelope and VAE parity gate"]
fn packed_native_matches_quality_envelope_and_vae_parity() {
    let root = image_install();
    let mut backend = MetalImageBackend::open(&root).expect("open packed Metal image install");
    let cancellation = CancellationToken::new();
    let mut conditioning_progress = |_, _| {};
    let conditioning = backend
        .encode_conditioning(PROMPT, 512, &cancellation, &mut conditioning_progress)
        .expect("native conditioning");
    let expected_conditioning = read_fixture("conditioning.npy");
    assert_eq!(conditioning.len(), expected_conditioning.len());
    let conditioning_error = relative_l2(&conditioning, &expected_conditioning);
    assert!(
        conditioning_error <= PACKED_CONDITIONING_REL_L2_LIMIT,
        "native packed conditioning quality error {conditioning_error} exceeds {PACKED_CONDITIONING_REL_L2_LIMIT}"
    );

    let request = request();
    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(IMAGE_STEPS as usize);
    // The 0.923 envelope is a matched-noise bound from the IG0 INT4
    // emulation, so the rollout must start from the captured initial noise.
    // The backend's native seeded noise is an independent realization and
    // reads about sqrt(2) relative L2 against these fixtures before any
    // weight is touched.
    let expected_initial = read_fixture("initial_noise.npy");
    let steps = backend
        .denoise_steps_from_noise(
            &conditioning,
            &request,
            &scheduler,
            &cancellation,
            &expected_initial,
        )
        .expect("native nine-step denoise");
    assert_eq!(steps.len(), 9, "native denoise must perform nine updates");
    for (step, latent) in steps.iter().enumerate() {
        let expected = read_fixture(&format!("latent_{step:02}.npy"));
        assert_eq!(latent.len(), expected.len());
        let error = relative_l2(latent, &expected);
        assert!(
            error <= PACKED_ROLLOUT_REL_L2_LIMIT,
            "native packed rollout step {} quality error {error} exceeds {PACKED_ROLLOUT_REL_L2_LIMIT}",
            step + 1
        );
    }
    let final_latents = steps.last().expect("nine-step trace has a final latent");
    let expected_final = read_fixture("final_latents.npy");
    let final_error = relative_l2(final_latents, &expected_final);
    assert!(
        final_error <= PACKED_ROLLOUT_REL_L2_LIMIT,
        "native packed final-latent quality error {final_error} exceeds {PACKED_ROLLOUT_REL_L2_LIMIT}"
    );

    // Decode the frozen high-precision latent as the isolated VAE
    // implementation gate. The packed denoiser's quantization drift is
    // deliberately not folded into this comparison.
    let expected_pixels = read_fixture("decoded_pixels.npy");
    let vae_decoded = backend
        .decode(
            &expected_final,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            &cancellation,
            &mut |_, _| {},
        )
        .expect("native VAE decode");
    assert_eq!(vae_decoded.len(), expected_pixels.len());
    let max_abs = vae_decoded
        .iter()
        .zip(&expected_pixels)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let rel_l2 = relative_l2(&vae_decoded, &expected_pixels);
    assert!(max_abs <= VAE_MAX_ABS_LIMIT, "VAE max abs error {max_abs}");
    assert!(rel_l2 <= VAE_REL_L2_LIMIT, "VAE relative L2 error {rel_l2}");

    // Also exercise the actual packed end-to-end latent. Its decoded pixels
    // are checked by the separate PNG quality oracle, not against the
    // higher-precision pixels with an invalid exact-parity tolerance.
    let decoded = backend
        .decode(
            final_latents,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            &cancellation,
            &mut |_, _| {},
        )
        .expect("native packed VAE decode");
    assert_eq!(decoded.len(), expected_pixels.len());
    assert!(
        decoded.iter().all(|value| value.is_finite()),
        "native packed decoded pixels must be finite"
    );
}

#[test]
#[ignore = "opt-in bounded packed first-step divergence trace"]
fn packed_native_first_step_trace_localizes_divergent_boundary() {
    let root = image_install();
    let mut backend = MetalImageBackend::open(&root).expect("open packed Metal image install");
    let cancellation = CancellationToken::new();
    let conditioning = backend
        .encode_conditioning(PROMPT, 512, &cancellation, &mut |_, _| {})
        .expect("native conditioning");
    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(IMAGE_STEPS as usize);
    // The trace rolls out from the captured reference noise. This backend's
    // native seeded noise is an independent realization that already reads
    // about sqrt(2) relative L2 against the captured initial_noise array, so
    // seeding it here is the difference between measuring packed drift and
    // measuring two different noise draws.
    let expected_initial = read_fixture("initial_noise.npy");
    let trace = backend
        .denoise_first_step_trace(
            &conditioning,
            &request(),
            &scheduler,
            &cancellation,
            Some(&expected_initial),
        )
        .expect("native first-step trace");

    let expected_conditioning = read_fixture("conditioning.npy");
    let expected_block_input = read_fixture("block_00_input.npy");
    let expected_main = read_fixture("block_29_output.npy");
    let expected_main_checkpoints = [
        (0, read_fixture("block_00_output.npy")),
        (15, read_fixture("block_15_output.npy")),
        (16, read_fixture("block_16_output.npy")),
        (20, read_fixture("block_20_output.npy")),
        (24, read_fixture("block_24_output.npy")),
        (28, read_fixture("block_28_output.npy")),
        (29, expected_main.clone()),
    ];
    let expected_latent = read_fixture("latent_00.npy");
    let dt = scheduler.sigmas[1] - scheduler.sigmas[0];
    let expected_velocity: Vec<f32> = expected_latent
        .iter()
        .zip(&expected_initial)
        .map(|(next, current)| (next - current) / dt)
        .collect();

    assert_eq!(trace.conditioning.len(), expected_conditioning.len());
    assert_eq!(trace.noise_refiner.len(), 4096 * 3840);
    assert_eq!(trace.main_transformer.len(), expected_main.len());
    assert_eq!(trace.velocity.len(), expected_velocity.len());
    assert_eq!(trace.scheduler_latent.len(), expected_latent.len());

    let noise_refiner_expected = &expected_block_input[..trace.noise_refiner.len()];
    assert!(trace.patchification.iter().all(|value| value.is_finite()));
    let rows = [
        (
            "conditioning",
            relative_l2(&trace.conditioning, &expected_conditioning),
        ),
        (
            "noise_refiner",
            relative_l2(&trace.noise_refiner, noise_refiner_expected),
        ),
        (
            "main_transformer",
            relative_l2(&trace.main_transformer, &expected_main),
        ),
        ("velocity", relative_l2(&trace.velocity, &expected_velocity)),
        (
            "scheduler_latent",
            relative_l2(&trace.scheduler_latent, &expected_latent),
        ),
    ];
    for (boundary, error) in rows {
        assert!(error.is_finite(), "{boundary} trace error is not finite");
        eprintln!("first-step trace boundary={boundary} relative-L2={error:.8e}");
    }
    for (index, expected) in expected_main_checkpoints {
        let actual = trace
            .main_transformer_checkpoints
            .iter()
            .find(|(checkpoint, _)| *checkpoint == index)
            .map(|(_, values)| values)
            .expect("native first-step trace missing main-transformer checkpoint");
        assert_eq!(actual.len(), expected.len());
        let error = relative_l2(actual, &expected);
        assert!(
            error.is_finite(),
            "main block {index} trace error is not finite"
        );
        eprintln!(
            "first-step trace boundary=main_transformer.block_{index:02} relative-L2={error:.8e}"
        );
    }
    eprintln!(
        "first-step trace seeded from the captured initial_noise fixture; conditioning_within_frozen_envelope={}",
        rows[0].1 <= PACKED_CONDITIONING_REL_L2_LIMIT
    );
    let patches_path = fixture_path("patches.npy");
    if patches_path.exists() {
        let expected_patches = turbospark_image::fixtures::read_npy_file_f32(&patches_path)
            .expect("read captured patchification fixture");
        assert_eq!(trace.patchification.len(), expected_patches.data.len());
        eprintln!(
            "first-step trace boundary=patchification relative-L2={:.8e}",
            relative_l2(&trace.patchification, &expected_patches.data)
        );
    } else {
        eprintln!(
            "first-step trace boundary=patchification reference=unavailable; rerun z_image_capture.py denoise"
        );
    }
}

#[test]
#[ignore = "opt-in isolated intra-block BF16 precision experiment"]
fn packed_native_intra_block_bf16_trace_reports_operation_deltas() {
    let root = image_install();
    let mut backend = MetalImageBackend::open(&root).expect("open packed Metal image install");
    let input = read_fixture("block_28_input.npy");
    let timestep = read_fixture("block_28_modulation.npy");
    let expected = read_fixture("block_28_output.npy");

    let fp32 = backend
        .intra_block_trace(28, &input, &timestep, false)
        .expect("FP32 intra-block trace");
    let bf16 = backend
        .intra_block_trace(28, &input, &timestep, true)
        .expect("BF16 intra-block trace");
    let labels: Vec<&str> = fp32
        .operations
        .iter()
        .map(|(label, _)| label.as_str())
        .collect();
    assert_eq!(labels.first(), Some(&"input"));
    assert_eq!(labels.last(), Some(&"ffn.residual"));
    assert_eq!(
        labels,
        bf16.operations
            .iter()
            .map(|(label, _)| label.as_str())
            .collect::<Vec<_>>()
    );

    for ((label, fp32_values), (_, bf16_values)) in fp32.operations.iter().zip(&bf16.operations) {
        let insertion_delta = relative_l2(bf16_values, fp32_values);
        assert!(
            bf16_values.iter().all(|value| value.is_finite()),
            "BF16 operation {label} produced a non-finite value"
        );
        eprintln!(
            "intra-block block=28 operation={label} bf16_vs_fp32_relative_l2={insertion_delta:.8e}"
        );
    }
    let fp32_output = &fp32.operations.last().expect("FP32 final operation").1;
    let bf16_output = &bf16.operations.last().expect("BF16 final operation").1;
    eprintln!(
        "intra-block block=28 final fp32_relative_l2={:.8e} bf16_relative_l2={:.8e}",
        relative_l2(fp32_output, &expected),
        relative_l2(bf16_output, &expected),
    );

    if let Ok(path) = std::env::var("TURBOSPARK_IMAGE_REFERENCE_INTRA_TRACE") {
        let entries = read_npz_file(Path::new(&path)).expect("read reference intra-block trace");
        for (label, native_values) in &fp32.operations {
            let entry = format!("{label}.npy");
            let bytes = entries
                .get(&entry)
                .unwrap_or_else(|| panic!("reference trace is missing {entry}"));
            let reference = read_npy_f32(bytes).expect("read reference operation array");
            assert_eq!(
                native_values.len(),
                reference.data.len(),
                "shape mismatch for {label}"
            );
            assert!(
                reference.data.iter().all(|value| value.is_finite()),
                "reference operation {label} produced a non-finite value"
            );
            let reference_error = relative_l2(native_values, &reference.data);
            eprintln!(
                "intra-block block=28 operation={label} native_vs_reference_relative_l2={reference_error:.8e}"
            );
            if matches!(label.as_str(), "attention.q_rope" | "attention.k_rope") {
                assert!(
                    reference_error <= 2e-2,
                    "native {label} diverged from the pinned reference: {reference_error}"
                );
            }
            if matches!(label.as_str(), "attention.q_rope" | "attention.k_rope") {
                let image_end = 4096 * 3840;
                eprintln!(
                    "intra-block block=28 operation={label} image_relative_l2={:.8e} caption_relative_l2={:.8e}",
                    relative_l2(
                        &native_values[..image_end],
                        &reference.data[..image_end]
                    ),
                    relative_l2(&native_values[image_end..], &reference.data[image_end..]),
                );
                eprintln!(
                    "intra-block block=28 operation={label} native_first={:?} reference_first={:?}",
                    &native_values[..8],
                    &reference.data[..8]
                );
            }
        }
    }
}

#[test]
#[ignore = "opt-in packed Metal VAE decode parity against the frozen latent"]
fn packed_native_vae_decodes_the_frozen_latent() {
    let root = image_install();
    let mut backend = MetalImageBackend::open(&root).expect("open packed Metal image install");
    let cancellation = CancellationToken::new();
    // Decode the frozen higher-precision final latent so this isolates the
    // packed VAE implementation from the packed denoiser's quantization
    // drift, exactly like the VAE section of the complete quality gate but
    // without the nine-step denoise in front of it.
    let expected_final = read_fixture("final_latents.npy");
    let expected_pixels = read_fixture("decoded_pixels.npy");
    let vae_decoded = backend
        .decode(
            &expected_final,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            &cancellation,
            &mut |_, _| {},
        )
        .expect("native packed VAE decode");
    assert_eq!(vae_decoded.len(), expected_pixels.len());
    let max_abs = vae_decoded
        .iter()
        .zip(&expected_pixels)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let rel_l2 = relative_l2(&vae_decoded, &expected_pixels);
    assert!(max_abs <= VAE_MAX_ABS_LIMIT, "VAE max abs error {max_abs}");
    assert!(rel_l2 <= VAE_REL_L2_LIMIT, "VAE relative L2 error {rel_l2}");
    eprintln!("packed VAE decode of the frozen latent: max_abs={max_abs} rel_l2={rel_l2}");
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
    let plan = backend.memory_plan().expect("build image memory plan");
    println!(
        "memory plan: resident_component_weights={} streamed_component_weights={} conditioning={} latents={} activations={} scratch={} staging={} in_flight_gpu={} allocator_retention={} largest_block={} block_workspace={} resident_lower_bound={} streamed_lower_bound={}",
        plan.component_weights,
        plan.streamed_component_weights,
        plan.conditioning,
        plan.latents,
        plan.activations,
        plan.scratch,
        plan.staging,
        plan.in_flight_gpu,
        plan.allocator_retention,
        plan.largest_block,
        plan.block_workspace,
        plan.resident_lower_bound(),
        plan.streamed_lower_bound(),
    );
    let arms: &[&str] = if std::env::var_os("TURBOSPARK_IMAGE_RESOURCE_WARM_ONLY").is_some() {
        &["warm"]
    } else {
        &["cold", "warm"]
    };
    let repeats = optional_u64("TURBOSPARK_IMAGE_RESOURCE_REPEATS")
        .unwrap_or(2)
        .max(1) as usize;
    let repeat_idle_growth_limit =
        optional_u64("TURBOSPARK_IMAGE_MAX_REPEAT_IDLE_GROWTH").unwrap_or(256 * 1024 * 1024);
    for label in arms {
        let mut first_idle = None;
        let mut first_peak = None;
        let mut previous_png = None;
        for repeat in 0..repeats {
            let measurement = measure_generation(&mut backend, label);
            let actual_pixels = decode_png_rgb(&measurement.result.png);
            let quality_error = relative_l2_u8(&actual_pixels, &expected_rgb);
            println!(
                "resource report: arm={label} repeat={} latency_ms={:.3} forward_count=9 phys_footprint_bytes={:?} idle_phys_footprint_bytes={:?} stage_peak_phys_footprint={:?} managed_allocations={} rust_owned_retained_buffers=0_at_idle physical_reads_pageins={:?} swap_used_before={:?} swap_used_after={:?} swap_delta={:?} png_relative_l2={quality_error:.6e}",
                repeat + 1,
                measurement.elapsed_ms,
                measurement.peak,
                measurement.after.footprint,
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
                    "{label} repeat {} phys_footprint exceeds {limit}",
                    repeat + 1
                );
            }
            if let Some(limit) = optional_u64("TURBOSPARK_IMAGE_MAX_ALLOCATIONS") {
                assert!(
                    measurement.allocations <= limit,
                    "{label} repeat {} Metal allocations {} exceed {limit}",
                    repeat + 1,
                    measurement.allocations
                );
            }
            if let Some(limit) = optional_f32("TURBOSPARK_IMAGE_MAX_PNG_REL_L2") {
                assert!(
                    quality_error <= limit,
                    "{label} repeat {} PNG quality error {quality_error} exceeds {limit}",
                    repeat + 1
                );
            }
            if repeats > 1 {
                let current_peak = measurement
                    .peak
                    .expect("phys_footprint is required for repeated image peak oracle");
                if let Some(first) = first_peak {
                    let growth = current_peak.saturating_sub(first);
                    assert!(
                        growth <= repeat_idle_growth_limit,
                        "{label} repeat {} peak footprint grew by {growth} bytes, limit {repeat_idle_growth_limit}",
                        repeat + 1
                    );
                }
                first_peak = Some(current_peak);
                let current = measurement
                    .after
                    .footprint
                    .expect("phys_footprint is required for repeated image storage oracle");
                if let Some(first) = first_idle {
                    let growth = current.saturating_sub(first);
                    assert!(
                        growth <= repeat_idle_growth_limit,
                        "{label} repeat {} idle footprint grew by {growth} bytes, limit {repeat_idle_growth_limit}",
                        repeat + 1
                    );
                }
                first_idle = Some(current);
            }
            if let Some(previous) = &previous_png {
                assert_eq!(
                    previous, &measurement.result.png,
                    "repeated {label} native generations must be deterministic"
                );
            }
            previous_png = Some(measurement.result.png.clone());
        }
    }

    let denoise_repeats = optional_u64("TURBOSPARK_IMAGE_RESOURCE_DENOISE_REPEATS")
        .unwrap_or(2)
        .max(1) as usize;
    let cancellation = CancellationToken::new();
    let conditioning = backend
        .encode_conditioning(PROMPT, 512, &cancellation, &mut |_, _| {})
        .expect("conditioning for repeated denoise oracle");
    let initial_noise = read_fixture("initial_noise.npy");
    let mut previous_final_latent = None;
    let mut first_denoise_idle = None;
    let mut first_denoise_peak = None;
    for cycle in 0..denoise_repeats {
        let before = resource_snapshot();
        let allocations_before = backend.buffer_allocation_count();
        let monitor = PeakMonitor::start(before.footprint);
        let started = Instant::now();
        let steps = backend
            .denoise_steps_from_noise(
                &conditioning,
                &request(),
                &scheduler_for_request(),
                &cancellation,
                &initial_noise,
            )
            .expect("repeated native denoise cycle");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let (peak, _) = monitor.finish();
        let after = resource_snapshot();
        assert_eq!(steps.len(), IMAGE_STEPS as usize);
        assert!(steps.iter().flatten().all(|value| value.is_finite()));
        let final_latent = steps.last().expect("denoise cycle has a final latent");
        if let Some(previous) = &previous_final_latent {
            assert_eq!(previous, final_latent, "repeated denoise cycles must agree");
        }
        previous_final_latent = Some(final_latent.to_vec());
        if denoise_repeats > 1 {
            let current_peak =
                peak.expect("phys_footprint is required for repeated denoise peak oracle");
            if let Some(first) = first_denoise_peak {
                let growth = current_peak.saturating_sub(first);
                assert!(
                    growth <= repeat_idle_growth_limit,
                    "denoise cycle {} peak footprint grew by {growth} bytes, limit {repeat_idle_growth_limit}",
                    cycle + 1
                );
            }
            first_denoise_peak = Some(current_peak);
            let current = after
                .footprint
                .expect("phys_footprint is required for repeated denoise oracle");
            if let Some(first) = first_denoise_idle {
                let growth = current.saturating_sub(first);
                assert!(
                    growth <= repeat_idle_growth_limit,
                    "denoise cycle {} idle footprint grew by {growth} bytes, limit {repeat_idle_growth_limit}",
                    cycle + 1
                );
            }
            first_denoise_idle = Some(current);
        }
        println!(
            "denoise cycle report: cycle={} latency_ms={elapsed_ms:.3} forwards={} final_latent_values={} peak_phys_footprint_bytes={peak:?} idle_phys_footprint_bytes={:?} managed_allocations={} physical_reads_pageins={:?} swap_delta={:?}",
            cycle + 1,
            steps.len(),
            steps.last().map_or(0, Vec::len),
            after.footprint,
            backend.buffer_allocation_count() - allocations_before,
            delta(before.pageins, after.pageins),
            signed_delta(before.swap_used, after.swap_used),
        );
    }
}

#[test]
#[ignore = "opt-in resident versus bounded streamed image comparison"]
fn packed_native_resident_agrees_with_two_slot_streamed() {
    let root = image_install();
    let expected_pixels = read_fixture("decoded_pixels.npy");
    let expected_rgb = turbospark_image::vae::decoded_to_rgb8(&expected_pixels, 1024, 1024)
        .expect("expected pixels convert to RGB");
    let mut resident = MetalImageBackend::open_with_residency(&root, ImageResidency::Resident)
        .expect("open resident packed Metal image install");
    let plan = resident.memory_plan().expect("build image memory plan");
    println!(
        "memory plan: resident_component_weights={} streamed_component_weights={} conditioning={} latents={} activations={} scratch={} staging={} in_flight_gpu={} allocator_retention={} largest_block={} block_workspace={} resident_lower_bound={} streamed_lower_bound={}",
        plan.component_weights,
        plan.streamed_component_weights,
        plan.conditioning,
        plan.latents,
        plan.activations,
        plan.scratch,
        plan.staging,
        plan.in_flight_gpu,
        plan.allocator_retention,
        plan.largest_block,
        plan.block_workspace,
        plan.resident_lower_bound(),
        plan.streamed_lower_bound(),
    );
    let allocations_before_refusal = resident.buffer_allocation_count();
    let refusal = turbospark_image::generate_with_memory_budget(
        &mut resident,
        &request(),
        &CancellationToken::new(),
        plan,
        ImageMemoryBudget {
            max_bytes: plan
                .largest_block
                .saturating_add(plan.block_workspace)
                .saturating_sub(1),
        },
        true,
        |_| {},
    )
    .expect_err("a too-small streamed budget must refuse before execution");
    assert!(
        refusal.contains("refused before execution"),
        "unexpected budget refusal: {refusal}"
    );
    assert_eq!(
        resident.buffer_allocation_count(),
        allocations_before_refusal,
        "budget refusal must not allocate Metal buffers"
    );
    let resident_measurement = measure_generation(&mut resident, "resident");
    drop(resident);

    let mut streamed = MetalImageBackend::open_with_residency(&root, ImageResidency::Streamed)
        .expect("open streamed packed Metal image install");
    let streamed_measurement = measure_generation(&mut streamed, "streamed");
    let metrics = streamed.stream_metrics();

    assert_eq!(
        resident_measurement.result.png, streamed_measurement.result.png,
        "resident and synchronous streamed generation must agree"
    );
    assert!(
        metrics.peak_slot_bytes > 0,
        "streamed path allocated no slots"
    );
    assert!(
        metrics.read_count > 0,
        "streamed path performed no payload reads"
    );
    assert!(
        metrics.read_bytes > 0,
        "streamed path read no payload bytes"
    );
    assert!(
        metrics.slot_reuse_waits > 0,
        "streamed path did not exercise a fenced slot reuse"
    );
    assert!(
        metrics.peak_slot_bytes <= plan.streamed_component_weights,
        "streamed slot capacity exceeded the admission plan"
    );

    for (label, measurement) in [
        ("resident", &resident_measurement),
        ("streamed", &streamed_measurement),
    ] {
        let actual_rgb = decode_png_rgb(&measurement.result.png);
        let quality_error = relative_l2_u8(&actual_rgb, &expected_rgb);
        assert!(
            quality_error <= PACKED_ROLLOUT_REL_L2_LIMIT,
            "{label} PNG quality error {quality_error} exceeds {PACKED_ROLLOUT_REL_L2_LIMIT}"
        );
    }
    println!(
        "resident_vs_streamed report: resident_latency_ms={:.3} streamed_latency_ms={:.3} latency_ratio={:.6} resident_peak_phys_footprint={:?} streamed_peak_phys_footprint={:?} streamed_peak_slot_bytes={} streamed_read_count={} streamed_read_bytes={} streamed_slot_reuse_waits={}",
        resident_measurement.elapsed_ms,
        streamed_measurement.elapsed_ms,
        streamed_measurement.elapsed_ms / resident_measurement.elapsed_ms,
        resident_measurement.peak,
        streamed_measurement.peak,
        metrics.peak_slot_bytes,
        metrics.read_count,
        metrics.read_bytes,
        metrics.slot_reuse_waits,
    );
}

#[test]
#[ignore = "opt-in packed image memory plan and pre-execution refusal gate"]
fn packed_native_memory_plan_refuses_before_execution() {
    let root = image_install();
    let mut backend = MetalImageBackend::open_with_residency(&root, ImageResidency::Streamed)
        .expect("open streamed packed Metal image install");
    let plan = backend.memory_plan().expect("build image memory plan");
    assert!(plan.component_weights > plan.streamed_component_weights);
    assert!(plan.largest_block > 0);
    assert!(plan.block_workspace > 0);
    let budget = ImageMemoryBudget {
        max_bytes: plan
            .largest_block
            .saturating_add(plan.block_workspace)
            .saturating_sub(1),
    };
    let allocations_before = backend.buffer_allocation_count();
    let error = turbospark_image::generate_with_memory_budget(
        &mut backend,
        &request(),
        &CancellationToken::new(),
        plan,
        budget,
        true,
        |_| {},
    )
    .expect_err("budget refusal must happen before streamed execution");
    assert!(error.contains("refused before execution"));
    let metrics = backend.stream_metrics();
    assert_eq!(backend.buffer_allocation_count(), allocations_before);
    assert_eq!(metrics.read_count, 0);
    assert_eq!(metrics.read_bytes, 0);
}

fn scheduler_for_request() -> FlowMatchEulerScheduler {
    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(IMAGE_STEPS as usize);
    scheduler
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
        if std::env::var_os("TURBOSPARK_IMAGE_STREAM_PROGRESS").is_some() {
            eprintln!(
                "image generation progress: stage={:?} completed={} total={}",
                event.stage, event.completed, event.total
            );
        }
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

fn fixture_path(name: &str) -> PathBuf {
    let root = std::env::var_os("TURBOSPARK_IMAGE_TRACE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/ig0/runs/lighting")
        });
    root.join(name)
}

fn read_fixture(name: &str) -> Vec<f32> {
    let path = fixture_path(name);
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
