//! The Nemotron-3 diarization encoder on Metal against the CPU encoder and
//! NeMo Speech 218d0b4, on real hardware with the pinned checkpoint.
//! Gated on `TURBOSPEECH_NEMOTRON_MODEL` (the f667ed73 conversion); the
//! features and NeMo chunk predictions are `crates/audio`'s coupled
//! fixture (`testdata/nemotron_diarization/coupled`).
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use audio::vad::nemotron_diarization::{EncoderExecutor, NemotronDiarization};
use turbospark_runtime::nemotron_diar_metal::NemotronDiarMetal;

fn pinned_model_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("TURBOSPEECH_NEMOTRON_MODEL").ok()?);
    let config = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config).ok()?;
    (config["source_revision"] == "f667ed73aee57d40cc39428eb768b4fd87a0a29e").then_some(dir)
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../audio/testdata/nemotron_diarization/coupled")
        .join(name)
}

/// A little-endian f32, C-order `.npy`: shape and values.
fn read_npy(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"));
    let start = header.find('(').unwrap() + 1;
    let end = header[start..].find(')').unwrap() + start;
    let shape = header[start..end]
        .split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect();
    let data = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

/// `(max |a - b|, max |a - b| / rms(b))`.
fn errors(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let rms = (b.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
    (
        worst,
        (f64::from(worst) / rms.max(f64::MIN_POSITIVE)) as f32,
    )
}

/// Channel-major `[n_mels, frames]` slice of the fixture features.
fn mel_window(feats: &[f32], n_mels: usize, total: usize, first: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_mels * frames];
    for m in 0..n_mels {
        out[m * frames..(m + 1) * frames]
            .copy_from_slice(&feats[m * total + first..m * total + first + frames]);
    }
    out
}

/// The encoder stack on a full 542-row window of real pre-encoded
/// features (the spkcache + FIFO + chunk size), with every key valid and
/// with the trailing keys masked.
#[test]
fn metal_encoder_matches_cpu_on_a_full_window() {
    let Some(dir) = pinned_model_dir() else {
        eprintln!("skipping: TURBOSPEECH_NEMOTRON_MODEL unset or not the f667ed73 conversion");
        return;
    };
    let model = NemotronDiarization::load(&dir).unwrap();
    let (shape, feats) = read_npy(&fixture("features.npy"));
    let (n_mels, total) = (shape[0], shape[1]);
    let frames = 542 * 8;
    assert!(total >= frames);
    let mel = mel_window(&feats, n_mels, total, 0, frames);
    let (embedded, valid) = model.pre_encode(&mel, frames, frames);
    let d = model.encoder_weights().d_model;
    let rows = embedded.len() / d;
    assert_eq!((rows, valid), (542, 542));

    let mut metal = NemotronDiarMetal::new(&model).unwrap();
    for lengths in [rows, 500] {
        let mut cpu = embedded.clone();
        model.encoder_forward(&mut cpu, rows, lengths);
        let mut gpu = embedded.clone();
        metal.encoder_forward(&mut gpu, rows, lengths).unwrap();
        let (abs, scaled) = errors(&gpu, &cpu);
        eprintln!("encoder rows {rows} keys {lengths}: max |d| {abs:e}, scaled {scaled:e}");
        assert!(scaled <= 1e-4, "keys {lengths}: scaled error {scaled:e}");
    }
}

/// The coupled diarizer with the Metal encoder over all 54 fixture steps,
/// against NeMo's chunk predictions and the CPU session: identical cache
/// and FIFO lengths, small probability drift, and no 0.5 decision flips
/// away from the threshold.
#[test]
fn metal_coupled_chunk_preds_track_nemo_and_cpu() {
    let Some(dir) = pinned_model_dir() else {
        eprintln!("skipping: TURBOSPEECH_NEMOTRON_MODEL unset or not the f667ed73 conversion");
        return;
    };
    let model = NemotronDiarization::load(&dir).unwrap();
    let (fshape, feats) = read_npy(&fixture("features.npy"));
    let (pshape, golden) = read_npy(&fixture("chunk_preds.npy"));
    let (n_mels, total) = (fshape[0], fshape[1]);
    let (steps, rows, n_spk) = (pshape[0], pshape[1], pshape[2]);
    assert_eq!(total, steps * 112);

    let modules = model.config().modules.multitalker_coupled(14);
    let mut cpu = model.coupled(modules.clone()).unwrap();
    let mut gpu = model
        .coupled(modules)
        .unwrap()
        .with_executor(Box::new(NemotronDiarMetal::new(&model).unwrap()));
    let (mut vs_nemo, mut vs_cpu, mut flips) = (0.0f32, 0.0f32, 0usize);
    for step in 0..steps {
        let mel = mel_window(&feats, n_mels, total, step * 112, 112);
        let want_cpu = cpu.step(&mel, 112, 112).unwrap().to_vec();
        let got = gpu.step(&mel, 112, 112).unwrap().to_vec();
        let want = &golden[step * rows * n_spk..(step + 1) * rows * n_spk];
        for ((a, b), c) in got.iter().zip(want).zip(&want_cpu) {
            vs_nemo = vs_nemo.max((a - b).abs());
            vs_cpu = vs_cpu.max((a - c).abs());
            if (b - 0.5).abs() >= 1e-3 && (*a > 0.5) != (*b > 0.5) {
                flips += 1;
            }
        }
        assert_eq!(
            (gpu.spkcache_frames(), gpu.fifo_frames()),
            (cpu.spkcache_frames(), cpu.fifo_frames()),
            "step {step}: cache/FIFO lengths"
        );
    }
    eprintln!(
        "metal coupled over {steps} steps: max |dp| vs NeMo {vs_nemo:e}, vs CPU {vs_cpu:e}, \
         decision flips {flips}"
    );
    assert_eq!(flips, 0, "0.5 decisions away from the threshold flipped");
    // The CPU path's own bound against NeMo. Measured on an M4 Max:
    // 2.2e-6 against NeMo and 1.7e-6 against the CPU session.
    assert!(
        vs_nemo <= 1e-4 && vs_cpu <= 1e-4,
        "vs NeMo {vs_nemo:e}, vs CPU {vs_cpu:e}"
    );
}
