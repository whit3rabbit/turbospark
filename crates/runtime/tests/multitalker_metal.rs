//! The Multitalker Parakeet streaming encoder on Metal against NeMo Speech
//! 218d0b4 and the CPU path, on real hardware with the F32 conversion.
//! Gated on `TURBOSPEECH_MULTITALKER_MODEL`; the features, targets and NeMo
//! encoder outputs are `crates/audio`'s `testdata/multitalker_asr` fixture.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use audio::models::stt::multitalker_asr::{
    Features, MultitalkerAsr, PreEncoded, SpeakerEncoder, SpeakerStream,
};
use serde_json::Value;
use turbospark_runtime::multitalker_metal::MultitalkerEncoderMetal;

fn gated_model() -> Option<MultitalkerAsr> {
    let dir = PathBuf::from(std::env::var("TURBOSPEECH_MULTITALKER_MODEL").ok()?);
    if !dir.join("config.json").exists() {
        eprintln!("skipping: no checkpoint at {}", dir.display());
        return None;
    }
    Some(MultitalkerAsr::open(&dir).unwrap())
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../audio/testdata/multitalker_asr")
        .join(name)
}

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

/// `max |a - b| / rms(b)`.
fn scaled(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let rms = (b.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
    (f64::from(worst) / rms.max(f64::MIN_POSITIVE)) as f32
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// The fixture's features and per-step chunks, as the CPU NeMo test reads
/// them: `(pre-encoded chunk, target, background)` per step.
fn fixture_steps(asr: &MultitalkerAsr) -> Vec<(PreEncoded, Vec<f32>, Vec<f32>)> {
    let (fshape, feats) = read_npy(&fixture("features.npy"));
    let (mels, total) = (fshape[0], fshape[1]);
    let feats = &feats;
    let features = Features {
        frames: total,
        mels,
        data: (0..total)
            .flat_map(|t| (0..mels).map(move |m| feats[m * total + t]))
            .collect(),
    };
    let steps: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("steps.json")).unwrap()).unwrap();
    let (shift, cache) = (asr.shift_frames(), asr.pre_encode_cache_frames());
    steps
        .as_array()
        .unwrap()
        .iter()
        .map(|step| {
            let k = step["step"].as_u64().unwrap() as usize;
            let frames = step["asr_chunk_frames"].as_u64().unwrap() as usize;
            let valid = step["asr_chunk_len"].as_u64().unwrap() as usize;
            let drop = step["drop_extra_pre_encoded"].as_u64().unwrap() as usize;
            let chunk = features.frame_major((k * shift) as isize - cache as isize, frames);
            let pre = asr.pre_encode(&chunk, frames, valid, drop).unwrap();
            (
                pre,
                floats(&step["spk_targets"][0]),
                floats(&step["bg_targets"][0]),
            )
        })
        .collect()
}

/// NeMo's channel-major `[1, hidden, rows]` encoder output as `[rows, hidden]`.
fn nemo_encoded(k: usize, hidden: usize) -> Vec<f32> {
    let (shape, data) = read_npy(&fixture(&format!("step_{k:04}_encoded.npy")));
    assert_eq!(shape[1], hidden);
    let rows = shape[2];
    let data = &data;
    (0..rows)
        .flat_map(|t| (0..hidden).map(move |c| data[c * rows + t]))
        .collect()
}

/// Six single-speaker steps through the device encoder, against NeMo's
/// encoder output and the CPU step, while the cache fills to 70 rows.
#[test]
fn metal_encoder_steps_match_nemo_and_cpu() {
    let Some(asr) = gated_model() else { return };
    let hidden = asr.stream_weights().hidden;
    let mut metal = MultitalkerEncoderMetal::new(&asr).unwrap();
    let mut cpu_stream = asr.new_stream();
    let mut cache_len = 0usize;
    let (mut vs_nemo, mut vs_cpu) = (0.0f32, 0.0f32);
    for (k, (pre, target, background)) in fixture_steps(&asr).into_iter().enumerate() {
        let cpu = asr
            .step(&pre, &target, &background, &mut cpu_stream, false)
            .unwrap();
        let input = asr.apply_speaker_kernels(&pre, &target, &background);
        let gpu = metal
            .stream_layers(&[0], &[&input], pre.rows, pre.valid, &[cache_len])
            .unwrap()
            .remove(0);
        cache_len = (cache_len + pre.rows).min(70);
        let nemo = nemo_encoded(k, hidden);
        let (a, b) = (scaled(&gpu, &nemo), scaled(&gpu, &cpu.encoded));
        eprintln!("step {k}: scaled vs NeMo {a:e}, vs CPU {b:e}");
        vs_nemo = vs_nemo.max(a);
        vs_cpu = vs_cpu.max(b);
    }
    assert!(
        vs_nemo <= 1e-4 && vs_cpu <= 1e-4,
        "vs NeMo {vs_nemo:e}, vs CPU {vs_cpu:e}"
    );
}

/// Three speaker slots stepped in an uneven pattern, each with its own
/// targets, against three independent CPU streams: batching must not mix
/// slots, and an unstepped slot keeps its caches.
///
/// The bound is looser than the NeMo-conditioned test's on purpose. Slots
/// 1 and 2 get a silent target with an active background, which is ill
/// conditioned: on an M4 Max the CPU's own output moves 8.5e-5 of RMS for
/// a relative 1e-7 (about one ulp) input perturbation there, and Metal sits
/// at 1.7e-4 (1.1e-4 with precise math), against 1.2e-5 and 1.1e-5 for a
/// speaking target. A slot mix-up is orders of magnitude above 1e-3.
#[test]
fn metal_encoder_batches_speakers_without_mixing_caches() {
    let Some(asr) = gated_model() else { return };
    let steps = fixture_steps(&asr);
    let mut metal = MultitalkerEncoderMetal::new(&asr).unwrap();
    let mut streams: Vec<SpeakerStream> = (0..3).map(|_| asr.new_stream()).collect();
    let mut cache_lens = [0usize; 3];
    let pattern: [&[usize]; 6] = [&[0, 1], &[0, 2], &[0, 1, 2], &[1], &[2, 0], &[0, 1, 2]];
    let mut worst = 0.0f32;
    for (k, active) in pattern.iter().enumerate() {
        let (pre, target, background) = &steps[k];
        // Distinct conditioning per slot: shifted targets, swapped roles.
        let conditions: Vec<(Vec<f32>, Vec<f32>)> = active
            .iter()
            .map(|&s| match s {
                0 => (target.clone(), background.clone()),
                1 => (background.clone(), target.clone()),
                _ => (
                    target.iter().map(|v| 1.0 - v).collect(),
                    vec![1.0; target.len()],
                ),
            })
            .collect();
        let inputs: Vec<Vec<f32>> = conditions
            .iter()
            .map(|(t, b)| asr.apply_speaker_kernels(pre, t, b))
            .collect();
        let views: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
        let lens: Vec<usize> = active.iter().map(|&s| cache_lens[s]).collect();
        let gpu = metal
            .stream_layers(active, &views, pre.rows, pre.valid, &lens)
            .unwrap();
        for ((&s, (t, b)), got) in active.iter().zip(&conditions).zip(&gpu) {
            let cpu = asr.step(pre, t, b, &mut streams[s], false).unwrap();
            let e = scaled(got, &cpu.encoded);
            eprintln!("step {k} slot {s}: scaled vs CPU {e:e}");
            worst = worst.max(e);
            cache_lens[s] = (cache_lens[s] + pre.rows).min(70);
        }
    }
    assert!(worst <= 1e-3, "worst scaled error {worst:e}");
}
