//! Opt-in full 1024 by 1024 DiT reference gate.
//!
//! This test is intentionally ignored: it executes all 34 transformer blocks
//! nine times in scalar FP32 Rust and uses the pinned local checkpoint.

use turbospark_image::fixtures::{model_subpath, read_npy_file_f32, run_array_path};
use turbospark_image::{FlowMatchEulerScheduler, ZImageTransformer};

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

    for step in 0..9 {
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
        let expected_path = if step == 8 {
            run_array_path("lighting", "final_latents.npy")
        } else {
            run_array_path("lighting", &format!("latent_{step:02}.npy"))
        };
        let expected = read_npy_file_f32(&expected_path).expect("read captured latent");
        assert_eq!(sample.len(), expected.data.len());
        let error = relative_l2(&sample, &expected.data);
        eprintln!("step {}/9: scheduler relative L2 {error:.8e}", step + 1);
        assert!(
            error <= 0.02,
            "step {step} relative L2 {error} exceeds provisional BF16-reference tolerance"
        );
    }
}
