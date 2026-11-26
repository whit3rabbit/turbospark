//! The teacher-forced stage gate for image denoise (LAB-36): each scheduler
//! step takes the REFERENCE's latent as input and is compared on its own.
//!
//! A free-running rollout bound cannot see a kernel change that moves one
//! step by one percent. The reference latents come from
//! `scripts/z_image_capture.py` (`conditioning.npy`, `initial_noise.npy`,
//! `latent_NN.npy`).
//!
//! The command reports per step and, against an earlier record, fails when
//! any step's output bits changed. It holds no quality bound: one must come
//! from a measured reference-against-reference floor, and none exists yet.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use turbospark_image::{
    fixtures::read_npy_file_f32, CancellationToken, FlowMatchEulerScheduler, ImageRequest,
    MetalImageBackend, IMAGE_GUIDANCE, IMAGE_HEIGHT, IMAGE_QUANTIZATION, IMAGE_STEPS, IMAGE_WIDTH,
};

use crate::args::Args;
use crate::ledger::Ledger;
use crate::schema::CmdOutput;

/// What the reference capture holds.
#[derive(Debug, Clone, Default)]
pub struct Fixtures {
    pub conditioning: Vec<f32>,
    pub initial_noise: Vec<f32>,
    /// The latent after each step.
    pub latents: Vec<Vec<f32>>,
}

impl Fixtures {
    pub fn load(dir: &Path, steps: usize) -> Result<Self, String> {
        let read = |name: &str| -> Result<Vec<f32>, String> {
            let path = dir.join(name);
            read_npy_file_f32(&path)
                .map(|a| a.data)
                .map_err(|e| format!("{}: {e}", path.display()))
        };
        let out = Self {
            conditioning: read("conditioning.npy")?,
            initial_noise: read("initial_noise.npy")?,
            latents: (0..steps)
                .map(|i| read(&format!("latent_{i:02}.npy")))
                .collect::<Result<_, _>>()?,
        };
        if out
            .latents
            .iter()
            .any(|l| l.len() != out.initial_noise.len())
        {
            return Err("the captured latents differ in size from the initial noise".into());
        }
        Ok(out)
    }

    /// The reference's input to step `step`.
    pub fn input(&self, step: usize) -> &[f32] {
        if step == 0 {
            &self.initial_noise
        } else {
            &self.latents[step - 1]
        }
    }

    pub fn digest(&self) -> String {
        let mut bytes = f32_bytes(&self.conditioning);
        bytes.extend(f32_bytes(&self.initial_noise));
        for l in &self.latents {
            bytes.extend(f32_bytes(l));
        }
        model_io::hash_data(&bytes)
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step: usize,
    /// Relative L2 of the output latent against the reference's.
    pub rel_l2: f64,
    /// Relative L2 of the UPDATE (output minus input). The latent barely
    /// moves in one step, so this is where a model error shows.
    pub update_rel_l2: f64,
    pub update_cosine: f64,
    pub output_sha256: String,
}

pub fn step_result(
    step: usize,
    input: &[f32],
    got: &[f32],
    want: &[f32],
) -> Result<StepResult, String> {
    if got.len() != want.len() || input.len() != want.len() {
        return Err(format!(
            "step {step}: {} values in, {} out, {} expected",
            input.len(),
            got.len(),
            want.len()
        ));
    }
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
    let diff = |a: &[f32], b: &[f32]| {
        let mut it = a.iter().zip(b).map(|(&x, &y)| f64::from(x) - f64::from(y));
        norm(&mut it)
    };
    let update = |out: &[f32]| -> Vec<f64> {
        out.iter()
            .zip(input)
            .map(|(&o, &i)| f64::from(o) - f64::from(i))
            .collect()
    };
    let (g, w) = (update(got), update(want));
    let size = |v: &[f64]| norm(&mut v.iter().copied());
    let dot: f64 = g.iter().zip(&w).map(|(a, b)| a * b).sum();
    let delta: Vec<f64> = g.iter().zip(&w).map(|(a, b)| a - b).collect();
    let reference = norm(&mut want.iter().map(|&x| f64::from(x)));
    // A zero denominator reads as NaN, which no comparison calls good.
    Ok(StepResult {
        step,
        rel_l2: diff(got, want) / reference,
        update_rel_l2: size(&delta) / size(&w),
        update_cosine: dot / (size(&g) * size(&w)),
        output_sha256: model_io::hash_data(&f32_bytes(got)),
    })
}

/// Run every step from the reference's input through `step_fn`.
pub fn run_steps(
    fixtures: &Fixtures,
    mut step_fn: impl FnMut(usize, &[f32]) -> Result<Vec<f32>, String>,
) -> Result<Vec<StepResult>, String> {
    (0..fixtures.latents.len())
        .map(|i| {
            let input = fixtures.input(i);
            let got = step_fn(i, input)?;
            step_result(i, input, &got, &fixtures.latents[i])
        })
        .collect()
}

/// The steps whose output bits differ from an earlier record of the same
/// fixtures, a missing step included.
pub fn changed_steps(earlier: &[StepResult], now: &[StepResult]) -> Vec<usize> {
    (0..earlier.len().max(now.len()))
        .filter(|&i| {
            earlier.get(i).map(|s| &s.output_sha256) != now.get(i).map(|s| &s.output_sha256)
        })
        .collect()
}

fn request() -> ImageRequest {
    ImageRequest {
        model_id: "Tongyi-MAI/Z-Image-Turbo".into(),
        // The step reads only the geometry and the step count.
        model_revision: "capture".into(),
        component_revisions: Default::default(),
        prompt: "capture".into(),
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
        batch: 1,
        scheduler_steps: IMAGE_STEPS,
        guidance_scale: IMAGE_GUIDANCE,
        seed: 0,
        quantization: IMAGE_QUANTIZATION.into(),
        noise_provenance: "capture".into(),
    }
}

/// `tslab image-stage --model INSTALL --fixtures DIR [--against EXPERIMENT]`.
/// Run it once as the reference, change the knob in the environment, run it
/// again with `--against` that experiment. Exit 3 when any step changed.
pub fn image_stage_cmd(mut args: Args) -> Result<CmdOutput, String> {
    let model = args.required("--model")?;
    let dir = args.required("--fixtures")?;
    let experiment = args
        .take("--experiment")?
        .unwrap_or_else(|| "image_stage".into());
    let against = args.take("--against")?;
    args.finish()?;
    let steps = IMAGE_STEPS as usize;
    let fixtures = Fixtures::load(Path::new(&dir), steps)?;
    let digest = fixtures.digest();
    let lab = crate::paths::lab_dir();
    // Read the comparison first, so a missing one fails before the model loads.
    let earlier: Option<Vec<StepResult>> = against
        .as_deref()
        .map(|name| {
            let ledger = Ledger::open(&crate::paths::experiment_dir(&lab, name)?)?;
            let record = ledger
                .of_kind("image_stage")?
                .into_iter()
                .rev()
                .find(|r| r["fixtures_sha256"] == digest.as_str())
                .ok_or_else(|| format!("no image_stage record of these fixtures in {name}"))?;
            serde_json::from_value(record["steps"].clone()).map_err(|e| e.to_string())
        })
        .transpose()?;
    let _lock = crate::lock::MachineLock::acquire(&lab, "tslab image-stage")?;
    let mut backend = MetalImageBackend::open(&catalog::resolve_image_arg(&model))?;
    let request = request();
    let mut scheduler = FlowMatchEulerScheduler::default();
    scheduler.set_timesteps(steps);
    let cancellation = CancellationToken::new();
    let results = run_steps(&fixtures, |step, latent| {
        backend.denoise_one_step(
            &fixtures.conditioning,
            &request,
            &scheduler,
            step,
            latent,
            &cancellation,
        )
    })?;
    for r in &results {
        crate::say!(
            "  step {}: update rel L2 {:.3e}, cosine {:.6}, latent rel L2 {:.3e}",
            r.step,
            r.update_rel_l2,
            r.update_cosine,
            r.rel_l2
        );
    }
    let changed = earlier.as_deref().map(|e| changed_steps(e, &results));
    match &changed {
        Some(c) if c.is_empty() => crate::say!("  every step is bit-identical to the earlier run"),
        Some(c) => crate::say!("  steps that changed: {c:?}"),
        None => crate::say!("  recorded; no bound is applied"),
    }
    let env: Value = json!(crate::provenance::turbospark_env());
    let rec = json!({"model": model, "fixtures": dir, "fixtures_sha256": digest,
                     "steps": results, "against": against, "changed_steps": changed,
                     "provenance": crate::provenance::collect(), "turbospark_env": env,
                     "quotable": false});
    Ledger::open(&crate::paths::experiment_dir(&lab, &experiment)?)?
        .append("image_stage", rec.clone())?;
    let code = if changed.is_some_and(|c| !c.is_empty()) {
        3
    } else {
        0
    };
    Ok(CmdOutput::with_code(code, rec))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> Fixtures {
        Fixtures {
            conditioning: vec![1.0],
            initial_noise: vec![1.0, 0.0],
            latents: vec![vec![1.0, 1.0], vec![1.0, 2.0]],
        }
    }

    #[test]
    fn every_step_takes_the_reference_input_not_its_own_output() {
        let f = fixtures();
        let mut seen = Vec::new();
        // A step that is wrong at step 0 must not move step 1's input.
        let results = run_steps(&f, |i, input| {
            seen.push(input.to_vec());
            Ok(if i == 0 {
                vec![9.0, 9.0]
            } else {
                f.latents[i].clone()
            })
        })
        .unwrap();
        assert_eq!(seen, [vec![1.0, 0.0], vec![1.0, 1.0]]);
        assert!(results[0].update_rel_l2 > 1.0);
        assert_eq!((results[1].rel_l2, results[1].update_rel_l2), (0.0, 0.0));
        assert!((results[1].update_cosine - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_small_update_error_shows_in_the_update_not_the_latent() {
        // The latent moves 1 from 100: a 10% error of the update is 0.1% of
        // the latent.
        let r = step_result(0, &[100.0], &[101.1], &[101.0]).unwrap();
        assert!((r.update_rel_l2 - 0.1).abs() < 1e-3, "{}", r.update_rel_l2);
        assert!(r.rel_l2 < 0.002);
        assert!(step_result(0, &[1.0], &[1.0, 2.0], &[1.0]).is_err());
        // A step that should not move has no relative update: NaN, not zero.
        assert!(step_result(0, &[1.0], &[1.0], &[1.0])
            .unwrap()
            .update_rel_l2
            .is_nan());
    }

    #[test]
    fn changed_steps_are_found_by_bits() {
        let f = fixtures();
        let exact = |off: f32| {
            run_steps(&f, |i, _| {
                Ok(f.latents[i].iter().map(|v| v + off * i as f32).collect())
            })
            .unwrap()
        };
        let a = exact(0.0);
        assert!(changed_steps(&a, &exact(0.0)).is_empty());
        assert_eq!(changed_steps(&a, &exact(1e-6)), [1]);
        assert_eq!(changed_steps(&a, &a[..1]), [1]);
    }

    #[test]
    fn a_capture_loads_and_a_short_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("tslab-image-stage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A version 1.0 npy of two little-endian f32 values.
        let npy = |values: [f32; 2]| {
            let mut header =
                "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }".to_string();
            while (10 + header.len() + 1) % 64 != 0 {
                header.push(' ');
            }
            header.push('\n');
            let mut out = b"\x93NUMPY\x01\x00".to_vec();
            out.extend((header.len() as u16).to_le_bytes());
            out.extend(header.as_bytes());
            out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            out
        };
        for name in ["conditioning", "initial_noise", "latent_00"] {
            std::fs::write(dir.join(format!("{name}.npy")), npy([1.0, 2.0])).unwrap();
        }
        let f = Fixtures::load(&dir, 1).unwrap();
        assert_eq!(f.latents, [vec![1.0, 2.0]]);
        assert_eq!(f.digest().len(), 64);
        let err = Fixtures::load(&dir, 2).unwrap_err();
        assert!(err.contains("latent_01.npy"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
