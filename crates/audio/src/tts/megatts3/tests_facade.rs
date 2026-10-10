//! Tests for the `MegaTts3` pipeline.
//!
//! The whole-pipeline gate is env-gated because it needs the real checkpoints:
//! TURBOSPARK_MEGATTS3_DIR (the converted directory with `aligner_lm/`,
//! `diffusion_transformer/`, `duration_lm/`, `wavvae/`, `g2p/`) and
//! TURBOSPARK_MEGATTS3_E2E_GOLDEN (the output of
//! `tools/gen_megatts3_e2e_fixtures.py`). The golden composes the reference's own
//! `align`, `make_dur_prompt`, `g2p`, `dur_pred`, `prepare_inputs_for_dit`,
//! `Diffusion.inference` and WavVAE with every random draw recorded; the Rust
//! pipeline replays those draws in the same order. Skipped when unset. Tests
//! never download.

use std::path::{Path, PathBuf};

use super::facade::{dit_inputs, MegaTts3, SynthParams, VoicePrompt};
use super::rng::{ReplayRng, SeededRng};
use super::tests::{f32s, i32s, max_diff};
use super::tests_duration::{load_tiny, prompt_for};

fn tiny_prompt(phone: Vec<i32>, tone: Vec<i32>, mel2ph: Vec<i32>) -> VoicePrompt {
    let lm = load_tiny();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/megatts3_duration");
    VoicePrompt {
        phone,
        tone,
        mel2ph,
        latent: vec![0.0; 32 * 2],
        duration: prompt_for(&lm, &dir),
        loudness: -20.0,
    }
}

#[test]
fn dit_inputs_concatenate_offset_and_truncate() {
    // Prompt of 2 phones over 8 frames; chunk of 2 phones over 11 frames.
    let prompt = tiny_prompt(vec![10, 11], vec![3, 4], vec![1, 1, 2, 2, 1, 1, 2, 2]);
    let chunk_mel2ph = vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
    let (phone, tone, mel2ph) = dit_inputs(&prompt, &[20, 21], &[1, 12], &chunk_mel2ph);
    assert_eq!(phone, vec![10, 11, 20, 21]);
    // Tone 1 is outside {0, 4, 11..=15} and becomes the English tone 3; 4 and 12 stay.
    assert_eq!(tone, vec![3, 4, 3, 12]);
    // The chunk's phone indices shift past the prompt's 2 phones; 19 frames cut to 16.
    assert_eq!(mel2ph.len(), 16);
    assert_eq!(&mel2ph[..8], &[1, 1, 2, 2, 1, 1, 2, 2]);
    assert_eq!(&mel2ph[8..], &[3, 3, 3, 3, 3, 4, 4, 4]);
}

#[test]
fn tone_rule_keeps_the_chinese_tone_ids() {
    let prompt = tiny_prompt(vec![5], vec![0], vec![1; 8]);
    let tones_in: Vec<i32> = (0..=16).collect();
    let phones = vec![7; tones_in.len()];
    let (_, tone, _) = dit_inputs(&prompt, &phones, &tones_in, &[1; 8]);
    let kept: Vec<i32> = tone[1..]
        .iter()
        .zip(&tones_in)
        .filter(|(o, i)| o == i)
        .map(|(o, _)| *o)
        .collect();
    assert_eq!(tone[0], 0);
    assert!(kept.contains(&0) && kept.contains(&4) && kept.contains(&11) && kept.contains(&15));
    for t in &tone[1..] {
        assert!(
            *t == 0 || *t == 4 || (11..=15).contains(t) || *t == 3,
            "unexpected tone {t}"
        );
    }
}

#[test]
fn default_params_are_the_space_defaults() {
    let p = SynthParams::default();
    assert_eq!((p.steps, p.p_w, p.t_w), (32, 1.4, 3.0));
    assert_eq!(
        (p.dur_disturb, p.dur_alpha, p.max_chunk_bytes),
        (0.1, 1.0, 130)
    );
}

fn replay_from(dir: &Path) -> ReplayRng {
    let info: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("info.json")).unwrap()).unwrap();
    let normals = (0..info["normals"].as_u64().unwrap())
        .map(|i| f32s(dir, &format!("n_{i:03}.npy")))
        .collect();
    let uniforms = (0..info["uniforms"].as_u64().unwrap())
        .map(|i| f32s(dir, &format!("u_{i:03}.npy")))
        .collect();
    ReplayRng::new(normals, uniforms)
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / a.len() as f64,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / b.len() as f64,
    );
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x) - ma, f64::from(*y) - mb);
        sab += x * y;
        saa += x * x;
        sbb += y * y;
    }
    sab / (saa.sqrt() * sbb.sqrt())
}

/// Whole pipeline on real weights against the reference's own functions.
#[test]
fn real_pipeline_matches_the_reference() {
    let (Some(model_dir), Some(golden)) = (
        std::env::var_os("TURBOSPARK_MEGATTS3_DIR"),
        std::env::var_os("TURBOSPARK_MEGATTS3_E2E_GOLDEN"),
    ) else {
        eprintln!("skipped: TURBOSPARK_MEGATTS3_DIR / TURBOSPARK_MEGATTS3_E2E_GOLDEN not set");
        return;
    };
    let g = PathBuf::from(golden);
    let info: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(g.join("info.json")).unwrap()).unwrap();
    let mut tts = MegaTts3::open(Path::new(&model_dir)).unwrap();
    assert!(tts.can_clone());
    let mut rng = replay_from(&g);

    // The reference's own 16 kHz aligner input isolates everything past the resampler.
    let wav24 = f32s(&g, "e_wav24.npy");
    let wav16 = f32s(&g, "e_wav16.npy");
    let prompt = tts
        .prepare_prompt_with(&wav24, Some(&wav16), &mut rng)
        .unwrap();
    assert_eq!(
        prompt.phone,
        i32s(&g, "e_prompt_phone.npy"),
        "prompt phones"
    );
    assert_eq!(prompt.tone, i32s(&g, "e_prompt_tone.npy"), "prompt tones");
    assert_eq!(
        prompt.mel2ph,
        i32s(&g, "e_prompt_mel2ph.npy"),
        "prompt mel2ph"
    );
    let latent_diff = max_diff(&prompt.latent, &f32s(&g, "e_prompt_latent.npy"));
    let loud_diff = (prompt.loudness - info["loudness_prompt"].as_f64().unwrap()).abs();
    println!("e2e prompt: {} phones, {} latent frames, latent diff {latent_diff:e}, loudness diff {loud_diff:e}",
             prompt.phone.len(), prompt.latent_frames());
    assert!(latent_diff < 5e-4, "prompt latent diff {latent_diff}");
    assert!(loud_diff < 1e-6, "prompt loudness diff {loud_diff}");

    let tol: f32 = std::env::var("TURBOSPARK_MEGATTS3_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    for (i, case) in info["cases"].as_array().unwrap().iter().enumerate() {
        let params = SynthParams {
            steps: case["steps"].as_u64().unwrap() as usize,
            p_w: case["p_w"].as_f64().unwrap() as f32,
            t_w: case["t_w"].as_f64().unwrap() as f32,
            max_chunk_bytes: case["max_chars"].as_u64().unwrap() as usize,
            ..SynthParams::default()
        };
        let audio = tts
            .synthesize(&prompt, case["text"].as_str().unwrap(), &params, &mut rng)
            .unwrap();
        let want = f32s(&g, &format!("e_audio_{i}.npy"));
        assert_eq!(audio.len(), want.len(), "case {i} length");
        let diff = max_diff(&audio, &want);
        let corr = correlation(&audio, &want);
        println!(
            "e2e case {i}: {} samples ({} chunks), max diff {diff:e}, correlation {corr:.6}",
            audio.len(),
            case["chunks"].as_array().unwrap().len()
        );
        assert!(corr > 0.9999, "case {i} correlation {corr}");
        assert!(diff < tol, "case {i} max diff {diff}");
    }
    assert!(
        rng.is_exhausted(),
        "the pipeline drew fewer random values than the reference"
    );

    // Production path: the crate resampler and a seeded RNG. The alignment can
    // shift by a frame, so this checks sanity rather than equality.
    let mut seeded = SeededRng::new(2024);
    let p2 = tts.prepare_prompt(&wav24, &mut seeded).unwrap();
    println!(
        "production path: {} phones (reference {}), {} latent frames",
        p2.phone.len(),
        prompt.phone.len(),
        p2.latent_frames()
    );
    assert_eq!(p2.phone, prompt.phone, "phones under the crate resampler");
    let params = SynthParams {
        steps: 4,
        ..SynthParams::default()
    };
    let audio = tts
        .synthesize(
            &p2,
            "Hello world, this is a short test.",
            &params,
            &mut seeded,
        )
        .unwrap();
    assert!(audio.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    let want = info["cases"][0]["samples"].as_u64().unwrap() as f64;
    assert!(
        (audio.len() as f64 - want).abs() / want < 0.25,
        "length {} vs {want}",
        audio.len()
    );
}
