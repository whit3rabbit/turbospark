use std::fs;
use turbospark_image::fixtures::{contracts_json_path, read_npy_file_f32, run_array_path};
use turbospark_image::scheduler::FlowMatchEulerScheduler;

#[test]
fn test_scheduler_schedules_exact_contract() {
    let path = contracts_json_path();
    assert!(
        path.exists(),
        "contracts json must exist at {}",
        path.display()
    );
    let content = fs::read_to_string(&path).expect("read contracts json");
    let json: serde_json::Value = serde_json::from_str(&content).expect("parse contracts json");

    let schedules = json
        .get("schedules")
        .and_then(|v| v.as_array())
        .expect("schedules array in contracts");

    let mut sched = FlowMatchEulerScheduler::default();

    for row in schedules {
        let requested_steps = row["requested_steps"].as_u64().unwrap() as usize;
        let expected_sigmas: Vec<f32> = row["sigmas"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let expected_timesteps: Vec<f32> = row["timesteps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        sched.set_timesteps(requested_steps);

        assert_eq!(sched.timesteps.len(), expected_timesteps.len());
        assert_eq!(sched.sigmas.len(), expected_sigmas.len());

        for (i, (&actual, &expected)) in sched.timesteps.iter().zip(&expected_timesteps).enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "step {requested_steps} timestep {i} bit mismatch: actual={actual}, expected={expected}"
            );
        }

        for (i, (&actual, &expected)) in sched.sigmas.iter().zip(&expected_sigmas).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "step {requested_steps} sigma {i} bit mismatch: actual={actual}, expected={expected}"
            );
        }
    }
}

#[test]
fn test_scheduler_captured_timesteps_sigmas_exact() {
    let t_path = run_array_path("lighting", "timesteps.npy");
    let s_path = run_array_path("lighting", "sigmas.npy");
    if !t_path.exists() || !s_path.exists() {
        eprintln!(
            "NOTE: skipping test_scheduler_captured_timesteps_sigmas_exact; captured files missing: {} or {}",
            t_path.display(),
            s_path.display()
        );
        return;
    }

    let captured_timesteps = read_npy_file_f32(&t_path).expect("read captured timesteps.npy");
    let captured_sigmas = read_npy_file_f32(&s_path).expect("read captured sigmas.npy");

    let mut sched = FlowMatchEulerScheduler::default();
    sched.set_timesteps(9);

    assert_eq!(sched.timesteps.len(), captured_timesteps.data.len());
    for (i, (&actual, &expected)) in sched
        .timesteps
        .iter()
        .zip(&captured_timesteps.data)
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "timestep {i} mismatch: actual={actual}, expected={expected}"
        );
    }

    assert_eq!(sched.sigmas.len(), captured_sigmas.data.len());
    for (i, (&actual, &expected)) in sched.sigmas.iter().zip(&captured_sigmas.data).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "sigma {i} mismatch: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn test_scheduler_euler_step_parity_latents() {
    let mut latents = Vec::new();
    for i in 0..9 {
        let path = run_array_path("lighting", &format!("latent_{i:02}.npy"));
        if !path.exists() {
            eprintln!(
                "NOTE: skipping test_scheduler_euler_step_parity_latents; latent file missing: {}",
                path.display()
            );
            return;
        }
        let arr = read_npy_file_f32(&path).expect("read latent array");
        latents.push(arr.data);
    }

    let final_path = run_array_path("lighting", "final_latents.npy");
    if !final_path.exists() {
        eprintln!(
            "NOTE: skipping test_scheduler_euler_step_parity_latents; final_latents missing: {}",
            final_path.display()
        );
        return;
    }
    let final_arr = read_npy_file_f32(&final_path).expect("read final_latents");
    latents.push(final_arr.data);

    let s_path = run_array_path("lighting", "sigmas.npy");
    let captured_sigmas = read_npy_file_f32(&s_path).expect("read sigmas");

    let mut sched = FlowMatchEulerScheduler::default();
    sched.set_timesteps(9);

    for (i, (&actual, &expected)) in sched.sigmas.iter().zip(&captured_sigmas.data).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "sigma {i} mismatch");
    }

    let mut max_err = 0.0f32;
    for i in 0..9 {
        let dt = sched.sigmas[i + 1] - sched.sigmas[i];
        let cur = &latents[i];
        let next = &latents[i + 1];

        // Invert step to derive model_output: next = cur + dt * model_output
        let mut model_output = Vec::with_capacity(cur.len());
        for k in 0..cur.len() {
            model_output.push((next[k] - cur[k]) / dt);
        }

        // Recompute forward Euler step
        let recomputed = sched.step(&model_output, i, cur);
        assert_eq!(recomputed.len(), next.len());

        for k in 0..next.len() {
            let err = (recomputed[k] - next[k]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    assert!(
        max_err < 1e-6,
        "Euler step recomputation max_err={max_err} exceeds 1e-6 roundoff bound"
    );
}
