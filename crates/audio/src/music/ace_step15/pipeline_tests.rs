//! End-to-end text2music against `tools/gen_ace_step15_e2e_reference.py`.
//! The unit tests run by default; the real run is ignored and needs
//! TURBOSPARK_ACE_STEP15_ROOT (checkpoint root, with
//! `silence_latent.safetensors` converted) and TURBOSPARK_ACE_STEP15_E2E_REF.

use std::path::PathBuf;

use super::pipeline::{gaussian, AceStep15, Stage, Text2Music, SAMPLE_RATE};
use super::tests::{read_npy, rel_err};

#[test]
fn gaussian_is_deterministic_and_standard() {
    let a = gaussian(7, 100_001);
    assert_eq!(a, gaussian(7, 100_001));
    assert_ne!(a[..16], gaussian(8, 16)[..]);
    let mean = a.iter().map(|&v| v as f64).sum::<f64>() / a.len() as f64;
    let var = a.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
    assert!(
        mean.abs() < 0.01 && (var - 1.0).abs() < 0.02,
        "mean {mean} var {var}"
    );
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
fn frames_follow_upstream_sizing() {
    assert_eq!(AceStep15::frames_for(10.0), 250);
    assert_eq!(AceStep15::frames_for(30.0), 750);
    // int(duration * 48000) // 1920, floored, then max(128, ...).
    assert_eq!(AceStep15::frames_for(47.9), 1197);
    assert_eq!(AceStep15::frames_for(2.0), 128);
    assert_eq!(AceStep15::frames_for(600.0), 15_000);
}

/// One end-to-end run against a reference directory; `dcw` must match
/// how the reference was made (its request.json records it).
fn run_e2e(refdir_var: &str, dcw: Option<super::dcw::Dcw>, wav: &str, audio_tol: f32) {
    let root = PathBuf::from(std::env::var("TURBOSPARK_ACE_STEP15_ROOT").unwrap());
    let refdir = PathBuf::from(std::env::var(refdir_var).unwrap());
    let req: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(refdir.join("request.json")).unwrap())
            .unwrap();
    let (_, noise) = read_npy(&refdir.join("noise.npy"));
    let mut model = AceStep15::load(&root, None).unwrap();
    let mut stages = Vec::new();
    let out = model
        .text2music(
            &Text2Music {
                caption: req["caption"].as_str().unwrap(),
                lyrics: req["lyrics"].as_str().unwrap(),
                language: req["language"].as_str().unwrap(),
                bpm: req["bpm"].as_u64().map(|b| b as u32),
                key_scale: req["key_scale"].as_str().unwrap(),
                time_signature: req["time_signature"].as_str().unwrap(),
                duration_seconds: req["duration"].as_f64().unwrap(),
                shift: req["shift"].as_f64().unwrap() as f32,
                seed: 0,
                noise: Some(&noise),
                dcw,
            },
            |s| {
                stages.push(s);
                true
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(out.frames, req["frames"].as_u64().unwrap() as usize);
    assert_eq!(
        req["dcw"].as_bool().unwrap_or(false),
        dcw.is_some(),
        "reference DCW setting"
    );
    assert_eq!(stages.first(), Some(&Stage::Text));
    assert_eq!(stages.last(), Some(&Stage::Decode));
    let (_, want_latents) = read_npy(&refdir.join("latents.npy"));
    let (_, want_audio) = read_npy(&refdir.join("audio.npy"));
    let (le, ae) = (
        rel_err(&out.latents, &want_latents),
        rel_err(&out.audio, &want_audio),
    );
    eprintln!(
        "e2e cpu ({wav}): latents {le:.2e} audio {ae:.2e}; text {:.1} s, condition {:.1} s, sampling {:.1} s, decode {:.1} s",
        out.timings.text.as_secs_f64(),
        out.timings.condition.as_secs_f64(),
        out.timings.sampling.as_secs_f64(),
        out.timings.decode.as_secs_f64()
    );
    // Interleave and write for listening (not part of the gate).
    let mut interleaved = Vec::with_capacity(out.audio.len());
    for i in 0..out.samples {
        interleaved.push(out.audio[i]);
        interleaved.push(out.audio[out.samples + i]);
    }
    let wave = crate::waveform::Waveform::new(SAMPLE_RATE, 2, interleaved).unwrap();
    std::fs::write(refdir.join(wav), crate::wav::write_wav_f32(&wave)).unwrap();
    // Measured on M4 Max at 10 s (250 frames): latents 1.3e-5 with or
    // without DCW. The latent gap grows with length (5.8e-6 at 48 frames in
    // the DiT gate); the bound keeps about 3x headroom.
    assert!(le <= 4e-5, "latents: relative error {le:.3e}");
    assert!(
        ae <= audio_tol,
        "audio: relative error {ae:.3e} > {audio_tol:.1e}"
    );
}

#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_ROOT and TURBOSPARK_ACE_STEP15_E2E_REF; CPU, minutes"]
fn real_text2music_matches_reference() {
    // Audio measured 9.7e-6.
    run_e2e("TURBOSPARK_ACE_STEP15_E2E_REF", None, "rust_cpu.wav", 4e-5);
}

/// Upstream's turbo default sampler (DCW double 0.05 / 0.02, haar),
/// reference from `gen_ace_step15_e2e_reference.py --dcw`.
#[test]
#[ignore = "needs TURBOSPARK_ACE_STEP15_ROOT and TURBOSPARK_ACE_STEP15_E2E_DCW_REF; CPU, minutes"]
fn real_text2music_dcw_matches_reference() {
    // Audio measured 1.06e-4. The decoder is 10x more sensitive at this
    // louder DCW operating point: a ~1e-4 latent perturbation moves the
    // diffusers decode by 1.1e-4 here against 1.1e-5 without DCW, so the
    // unchanged 1.3e-5 latent gap accounts for it. About 3x headroom.
    run_e2e(
        "TURBOSPARK_ACE_STEP15_E2E_DCW_REF",
        Some(super::dcw::Dcw::TURBO),
        "rust_cpu_dcw.wav",
        3e-4,
    );
}
