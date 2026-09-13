//! Opt-in full 1024 by 1024 DiT reference gate.
//!
//! These tests are intentionally ignored: they execute all 34 transformer
//! blocks in batched FP32 Rust and use the pinned local checkpoint.

use turbospark_image::fixtures::{model_subpath, read_npy_file_f32, run_array_path};
use turbospark_image::{FlowMatchEulerScheduler, ZImageTransformer};

const CAPTURED_INPUT_SCHEDULER_REL_L2_LIMIT: f32 = 0.02;
const CUMULATIVE_BF16_REL_L2_LIMIT: f32 = 0.196;
const STEP_FIXTURES: [(&str, &str); 9] = [
    ("initial_noise.npy", "latent_00.npy"),
    ("latent_00.npy", "latent_01.npy"),
    ("latent_01.npy", "latent_02.npy"),
    ("latent_02.npy", "latent_03.npy"),
    ("latent_03.npy", "latent_04.npy"),
    ("latent_04.npy", "latent_05.npy"),
    ("latent_05.npy", "latent_06.npy"),
    ("latent_06.npy", "latent_07.npy"),
    ("latent_07.npy", "latent_08.npy"),
];

fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let (diff, norm) = actual
        .iter()
        .zip(expected)
        .fold((0.0f64, 0.0f64), |(d, n), (a, b)| {
            let delta = (*a as f64) - (*b as f64);
            (d + delta * delta, n + (*b as f64) * (*b as f64))
        });
    (diff.sqrt() / norm.sqrt().max(1e-30)) as f32
}

#[test]
#[ignore = "opt-in full 1024 checkpoint forward (very slow CPU reference gate)"]
fn test_z_image_full_nine_step_checkpoint_parity() {
    let transformer_dir = model_subpath("transformer");
    let initial_path = run_array_path("lighting", "initial_noise.npy");
    let conditioning_path = run_array_path("lighting", "conditioning.npy");
    assert!(
        transformer_dir.exists(),
        "missing transformer checkout at {}",
        transformer_dir.display()
    );
    assert!(
        initial_path.exists(),
        "missing initial noise at {}",
        initial_path.display()
    );
    assert!(
        conditioning_path.exists(),
        "missing conditioning at {}",
        conditioning_path.display()
    );

    let initial = read_npy_file_f32(&initial_path).expect("read initial noise");
    let conditioning = read_npy_file_f32(&conditioning_path).expect("read conditioning");
    assert_eq!(initial.shape, vec![1, 16, 128, 128]);
    assert_eq!(conditioning.shape[1], 2560);

    let transformer =
        ZImageTransformer::open(&transformer_dir).expect("open transformer checkpoint");
    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(9);
    let mut sample = initial.data;

    let mut accumulated_errors = Vec::with_capacity(STEP_FIXTURES.len());
    for (step, &(input_name, expected_name)) in STEP_FIXTURES.iter().enumerate() {
        eprintln!("step {}/9: starting transformer forward", step + 1);
        let model_output = transformer
            .forward_with_progress(
                &sample,
                128,
                128,
                scheduler.normalized_time(step),
                &conditioning.data,
                |name| eprintln!("step {}/9: completed {name}", step + 1),
            )
            .expect("native DiT forward");
        sample = scheduler.step(&model_output, step, &sample);
        let expected_path = run_array_path("lighting", expected_name);
        let expected = read_npy_file_f32(&expected_path).expect("read captured latent");
        assert_eq!(sample.len(), expected.data.len());
        let error = relative_l2(&sample, &expected.data);
        eprintln!(
            "rollout update {}/9: input={input_name} expected={expected_name} accumulated-rollout-relative-L2={error:.8e}",
            step + 1
        );
        accumulated_errors.push(error);
    }

    let max_error = accumulated_errors.iter().copied().fold(0.0f32, f32::max);
    assert!(
        accumulated_errors
            .iter()
            .all(|&error| error <= CUMULATIVE_BF16_REL_L2_LIMIT),
        "maximum accumulated rollout relative L2 {max_error} exceeds frozen BF16 envelope {CUMULATIVE_BF16_REL_L2_LIMIT}"
    );

    let final_latents_path = run_array_path("lighting", "final_latents.npy");
    let final_latents = read_npy_file_f32(&final_latents_path).expect("read final_latents");
    let final_capture_path = run_array_path("lighting", "latent_08.npy");
    let final_capture = read_npy_file_f32(&final_capture_path).expect("read latent_08");
    assert_eq!(final_capture.data, final_latents.data);
}

#[test]
#[ignore = "opt-in full 1024 checkpoint forward from every captured timestep"]
fn test_z_image_all_steps_from_captured_input_parity() {
    let transformer_dir = model_subpath("transformer");
    let conditioning_path = run_array_path("lighting", "conditioning.npy");
    for path in [&transformer_dir, &conditioning_path] {
        assert!(
            path.exists(),
            "missing checkpoint prerequisite at {}",
            path.display()
        );
    }

    let conditioning = read_npy_file_f32(&conditioning_path).expect("read conditioning");
    assert_eq!(conditioning.shape[1], 2560);

    let transformer =
        ZImageTransformer::open(&transformer_dir).expect("open transformer checkpoint");
    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(9);

    for (step, &(input_name, expected_name)) in STEP_FIXTURES.iter().enumerate() {
        let input_path = run_array_path("lighting", input_name);
        let expected_path = run_array_path("lighting", expected_name);
        for path in [&input_path, &expected_path] {
            assert!(
                path.exists(),
                "missing captured-input prerequisite at {}",
                path.display()
            );
        }

        let input = read_npy_file_f32(&input_path).expect("read captured input");
        let expected = read_npy_file_f32(&expected_path).expect("read captured output");
        assert_eq!(input.shape, vec![1, 16, 128, 128]);
        assert_eq!(input.shape, expected.shape);

        let model_output = transformer
            .forward_with_progress(
                &input.data,
                128,
                128,
                scheduler.normalized_time(step),
                &conditioning.data,
                |name| eprintln!("captured-input update {}/9: completed {name}", step + 1),
            )
            .expect("native DiT forward");
        let actual = scheduler.step(&model_output, step, &input.data);
        let scheduler_error = relative_l2(&actual, &expected.data);
        let dt = scheduler.sigmas[step + 1] - scheduler.sigmas[step];
        let expected_model_output: Vec<f32> = expected
            .data
            .iter()
            .zip(&input.data)
            .map(|(next, current)| (next - current) / dt)
            .collect();
        let model_error = relative_l2(&model_output, &expected_model_output);
        eprintln!(
            "captured-input update {}/9: input={input_name} expected={expected_name} transformer-output-relative-L2={model_error:.8e} scheduler-output-relative-L2={scheduler_error:.8e}",
            step + 1
        );
        assert!(
            scheduler_error <= CAPTURED_INPUT_SCHEDULER_REL_L2_LIMIT,
            "captured-input update {} scheduler relative L2 {scheduler_error} exceeds provisional BF16-reference tolerance {CAPTURED_INPUT_SCHEDULER_REL_L2_LIMIT}",
            step + 1
        );
    }
}
