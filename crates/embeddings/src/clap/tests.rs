//! CLAP parity tests against `testdata/clap` (written by
//! `tools/gen_clap_fixtures.py`). The default suite needs no download;
//! the `#[ignore]`d tests read the real checkpoint from
//! `$TURBOSPARK_CLAP_DIR` or `~/models/embeddings/clap-htsat-unfused`.

use std::collections::HashMap;
use std::f64::consts::PI;

use serde_json::Value;

use super::*;
use crate::similarity::cosine_similarity;
use crate::testutil::{max_abs_diff, model_dir, read_json, read_npy, testdata};

const FAMILY: &str = "clap";

// ------------------------------------------------------------ synthesis

/// Mirrors `xorshift_noise` in the generator: an exact u32 stream mapped
/// to `k * 2^-23 - 1`.
fn xorshift_noise(n: usize, seed: u32) -> Vec<f64> {
    let mut x: u32 = if seed != 0 { seed } else { 0x9E37_79B9 };
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            f64::from(x >> 8) * 2f64.powi(-23) - 1.0
        })
        .collect()
}

/// Mirrors `synth_clip` in the generator, same arithmetic order.
fn synth_clip(spec: &Value) -> Vec<f32> {
    let n = spec["samples"].as_u64().unwrap() as usize;
    let sr = f64::from(SAMPLE_RATE);
    let seed = spec.get("seed").and_then(Value::as_u64).unwrap_or(0) as u32;
    let t = |i: usize| i as f64 / sr;
    let x: Vec<f64> = match spec["kind"].as_str().unwrap() {
        "tone" => (0..n)
            .map(|i| {
                0.3 * (2.0 * PI * 440.0 * t(i)).sin() + 0.15 * (2.0 * PI * 1320.0 * t(i)).sin()
            })
            .collect(),
        "chirp_noise" => {
            let dur = n as f64 / sr;
            let noise = xorshift_noise(n, seed);
            (0..n)
                .map(|i| {
                    let tt = t(i);
                    let phase = 2.0 * PI * (200.0 * tt + (6000.0 - 200.0) * tt * tt / (2.0 * dur));
                    0.25 * phase.sin() + 0.05 * noise[i]
                })
                .collect()
        }
        "burst" => {
            let noise = xorshift_noise(n, seed);
            (0..n)
                .map(|i| 0.2 * noise[i] * (-4.0 * t(i)).exp())
                .collect()
        }
        "long" => {
            let noise = xorshift_noise(n, seed);
            (0..n)
                .map(|i| {
                    let env = 0.5 + 0.5 * (2.0 * PI * 0.25 * t(i)).sin();
                    0.2 * (2.0 * PI * 330.0 * t(i)).sin() * env + 0.05 * noise[i]
                })
                .collect()
        }
        other => panic!("unknown clip kind {other}"),
    };
    x.into_iter().map(|v| v as f32).collect()
}

fn clips() -> Vec<Value> {
    read_json(FAMILY, "clips.json")["clips"]
        .as_array()
        .unwrap()
        .clone()
}

fn real_feature_extractor() -> ClapFeatureExtractor {
    // The real checkpoint's preprocessor_config.json values (the fixture
    // generator asserts rand_trunc + repeatpad and uses this file).
    let json = serde_json::json!({
        "feature_size": 64, "sampling_rate": 48000, "hop_length": 480,
        "max_length_s": 10, "fft_window_size": 1024, "frequency_min": 50,
        "frequency_max": 14000, "top_db": null, "truncation": "rand_trunc",
        "padding": "repeatpad", "n_fft": 1024, "nb_max_frames": 1000
    });
    ClapFeatureExtractor::new(FeatureConfig::from_json(&json).unwrap()).unwrap()
}

#[test]
fn synthesis_matches_generator() {
    for spec in clips() {
        let pcm = synth_clip(&spec);
        let head: Vec<f32> = spec["head"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(&pcm[..head.len()], &head[..], "{}", spec["name"]);
        let sum: f64 = pcm.iter().map(|&v| f64::from(v)).sum();
        let want = spec["checksum_sum"].as_f64().unwrap();
        assert!(
            (sum - want).abs() < 1e-6,
            "{}: sum {sum} vs {want}",
            spec["name"]
        );
    }
}

// ------------------------------------------------------------ features

/// Every clip through the f64 frontend vs `ClapFeatureExtractor`. The
/// long clip is cropped at the offset the reference's seeded
/// `np.random.randint` drew.
#[test]
fn feature_extractor_matches_reference() {
    let fe = real_feature_extractor();
    for spec in clips() {
        let name = spec["name"].as_str().unwrap();
        let pcm = synth_clip(&spec);
        let offset = spec["crop_offset"].as_u64().unwrap() as usize;
        let mel = fe.extract(&pcm, LongAudioCrop::Offset(offset)).unwrap();
        let (shape, want) = read_npy(FAMILY, &format!("features/{name}.npy"));
        assert_eq!(shape, vec![mel.frames, mel.mels], "{name}");
        assert_eq!(
            mel.is_longer,
            spec["is_longer"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(mel.crop_offset, offset);
        let diff = max_abs_diff(&mel.data, &want);
        eprintln!("features {name}: max |dB diff| = {diff:.3e}");
        // The reference is f64 numpy rounded to f32 at the end; the only
        // divergence is FFT and mel-sum rounding order in f64. Measured 0.0
        // (bit-exact) on every clip; the bound is about one f32 ulp at
        // 100 dB (7.6e-6) so a platform libm difference does not trip it.
        assert!(diff <= 1e-5, "{name}: {diff}");
    }
}

/// Measures the shared f32 frontend (`turbospark_audio` STFT and Slaney
/// filterbank) against the same goldens. Not the production path; this
/// pins how far it is so the README number stays honest.
#[test]
fn f32_frontend_measurement() {
    use turbospark_audio::mel::{mel_filterbank, MelScale};
    use turbospark_audio::stft::{stft, StftOptions};
    let fe = real_feature_extractor();
    let n = 1024;
    let window: Vec<f32> = (0..n)
        .map(|k| (0.5 - 0.5 * (2.0 * PI * k as f64 / n as f64).cos()) as f32)
        .collect();
    let bank = mel_filterbank(64, n, SAMPLE_RATE, 50.0, Some(14_000.0), MelScale::Slaney).unwrap();
    let mut worst = 0.0f32;
    let mut worst_bank = 0.0f64;
    for (a, b) in bank.weights.iter().zip(fe.filterbank()) {
        worst_bank = worst_bank.max((f64::from(*a) - b).abs());
    }
    for spec in clips() {
        let name = spec["name"].as_str().unwrap();
        let pcm = synth_clip(&spec);
        let offset = spec["crop_offset"].as_u64().unwrap() as usize;
        let max = fe.config().max_samples();
        let clip: Vec<f32> = if pcm.len() > max {
            pcm[offset..offset + max].to_vec()
        } else {
            let mut c = Vec::with_capacity(max);
            for _ in 0..max / pcm.len() {
                c.extend_from_slice(&pcm);
            }
            c.resize(max, 0.0);
            c
        };
        let opts = StftOptions {
            fft_size: n,
            hop: 480,
            window: window.clone(),
            center: true,
        };
        let frames = stft(&clip, &opts).unwrap();
        let mut data = Vec::with_capacity(frames.len() * 64);
        for frame in &frames {
            let power: Vec<f32> = frame.iter().map(|c| c.re.hypot(c.im).powi(2)).collect();
            for v in bank.project(&power).unwrap() {
                data.push(10.0 * v.max(1e-10).log10());
            }
        }
        let (_, want) = read_npy(FAMILY, &format!("features/{name}.npy"));
        let diff = max_abs_diff(&data, &want);
        eprintln!("f32 frontend {name}: max |dB diff| = {diff:.3e}");
        worst = worst.max(diff);
    }
    eprintln!("f32 filterbank max |w diff| = {worst_bank:.3e}");
    // Loose: documents the gap rather than gating a path we do not ship.
    assert!(worst < 1.0, "{worst}");
}

// ------------------------------------------------------------ tiny model

fn tiny_model() -> ClapModel {
    ClapModel::load(&testdata(FAMILY, "tiny")).unwrap()
}

/// Largest |a - b| relative to the golden's scale (max |b|, at least 1).
fn scaled_diff(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    max_abs_diff(got, want) / scale
}

// Tiny-model tolerance, relative to each tensor's max magnitude. The
// tensors are fp32 through identical math; the residual is accumulation
// order (torch's BLAS and conv vs sequential sums here) and the erf
// approximation (abs error < 1.5e-7), compounded over 6 Swin blocks or 2
// RoBERTa layers with LayerNorms re-amplifying it. Measured worst 1.0e-6
// (audio_a final norm); the bound leaves 10x headroom.
const TINY_TOL: f32 = 1e-5;

#[test]
fn tiny_audio_tower_matches_reference() {
    let model = tiny_model();
    for case in ["audio_a", "audio_b"] {
        let (shape, input) = read_npy(FAMILY, &format!("tiny/{case}_input.npy"));
        let mut got: HashMap<String, Vec<f32>> = HashMap::new();
        let embed = model
            .audio_embedding_with(&input, shape[0], shape[1], &mut |name, v| {
                got.insert(name.to_string(), v.to_vec());
            })
            .unwrap();
        got.insert("embed".into(), embed);
        let meta = read_json(FAMILY, "tiny/goldens.json");
        let names: Vec<String> = meta["audio"][case]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        for name in names {
            let (_, want) = read_npy(FAMILY, &format!("tiny/{case}_{name}.npy"));
            let have = got.get(&name).unwrap_or_else(|| panic!("no {name}"));
            let d = scaled_diff(have, &want);
            eprintln!("tiny {case} {name}: scaled max diff {d:.3e}");
            assert!(d <= TINY_TOL, "{case} {name}: {d}");
        }
    }
}

#[test]
fn tiny_text_tower_matches_reference() {
    let model = tiny_model();
    let meta = read_json(FAMILY, "tiny/goldens.json");
    for (key, entry) in meta["text"].as_object().unwrap() {
        let ids: Vec<u32> = entry["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mut got: HashMap<String, Vec<f32>> = HashMap::new();
        let embed = model
            .text_embedding_with(&ids, &mut |name, v| {
                got.insert(name.to_string(), v.to_vec());
            })
            .unwrap();
        got.insert("embed".into(), embed);
        for name in entry.as_object().unwrap().keys().filter(|k| *k != "ids") {
            let (_, want) = read_npy(FAMILY, &format!("tiny/{key}_{name}.npy"));
            let d = scaled_diff(&got[name], &want);
            eprintln!("tiny {key} {name}: scaled max diff {d:.3e}");
            assert!(d <= TINY_TOL, "{key} {name}: {d}");
        }
    }
}

#[test]
fn tiny_logits_match_reference() {
    let model = tiny_model();
    let meta = read_json(FAMILY, "tiny/goldens.json");
    let scale = meta["logit_scale_a_exp"].as_f64().unwrap() as f32;
    assert!((model.audio_logit_scale() - scale).abs() <= 1e-6 * scale);
    let mut texts = Vec::new();
    for i in 0..3 {
        texts.extend(read_npy(FAMILY, &format!("tiny/text{i}_embed.npy")).1);
    }
    let (_, want) = read_npy(FAMILY, "tiny/logits_per_audio.npy");
    let mut got = Vec::new();
    for case in ["audio_a", "audio_b"] {
        let (shape, input) = read_npy(FAMILY, &format!("tiny/{case}_input.npy"));
        let a = model
            .audio_embedding_with(&input, shape[0], shape[1], &mut |_, _| {})
            .unwrap();
        got.extend(model.logits_per_audio(&a, &texts).unwrap());
    }
    let d = max_abs_diff(&got, &want);
    eprintln!("tiny logits: max diff {d:.3e}");
    // Measured 3.4e-6 at scale 10.
    assert!(d <= 1e-5 * scale, "{d}");
}

#[test]
fn tiny_refuses_bad_inputs() {
    let model = tiny_model();
    // More frames than spec_width (256 for the tiny config).
    assert!(model
        .audio_embedding_with(&vec![0.0; 257 * 16], 257, 16, &mut |_, _| {})
        .is_err());
    assert!(model
        .audio_embedding_with(&vec![0.0; 10 * 8], 10, 8, &mut |_, _| {})
        .is_err());
    // Out-of-vocab id, empty, and longer than the 40-row position table.
    assert!(model.text_embedding(&[0, 100, 2]).is_err());
    assert!(model.text_embedding(&[]).is_err());
    assert!(model.text_embedding(&[5; 39]).is_err());
    assert!(model.text_embedding(&[5; 38]).is_ok());
}

// ------------------------------------------------------------ config

#[test]
fn config_refuses_unsupported_values() {
    let base = read_json(FAMILY, "tiny/config.json");
    assert!(ClapConfig::from_json(&base).is_ok());
    let with = |path: &[&str], v: Value| {
        let mut c = base.clone();
        let mut node = &mut c;
        for p in &path[..path.len() - 1] {
            node = node.get_mut(*p).unwrap();
        }
        node[path[path.len() - 1]] = v;
        ClapConfig::from_json(&c)
    };
    assert!(with(&["audio_config", "enable_fusion"], Value::Bool(true)).is_err());
    assert!(with(&["audio_config", "hidden_act"], "relu".into()).is_err());
    assert!(with(&["audio_config", "hidden_size"], 64.into()).is_err());
    // Resolution 16 is not a multiple of window 3: needs window padding.
    assert!(with(&["audio_config", "window_size"], 3.into()).is_err());
    // Last stage resolution 4 below window 8.
    assert!(with(&["audio_config", "window_size"], 8.into()).is_err());
    assert!(with(&["audio_config", "spec_size"], 72.into()).is_err());
    assert!(with(
        &["text_config", "position_embedding_type"],
        "relative_key".into()
    )
    .is_err());
    assert!(with(&["projection_hidden_act"], "gelu".into()).is_err());

    // Defaults fill in what a sparse config omits.
    let sparse = ClapConfig::from_json(&serde_json::json!({})).unwrap();
    assert_eq!(sparse.audio.depths, vec![2, 2, 6, 2]);
    assert_eq!(sparse.text.vocab_size, 50265);
    let g = sparse.audio.geometry().unwrap();
    assert_eq!((g.freq_ratio, g.spec_width, g.grid), (4, 1024, 64));
    let shifts: Vec<usize> = g.stages.iter().map(|s| s.shift).collect();
    assert_eq!(shifts, vec![4, 4, 4, 0]);

    let fe = |k: &str, v: Value| {
        let mut j = serde_json::json!({"truncation": "rand_trunc", "padding": "repeatpad"});
        j[k] = v;
        FeatureConfig::from_json(&j)
    };
    assert!(fe("padding", "repeatpad".into()).is_ok());
    assert!(fe("truncation", "fusion".into()).is_err());
    assert!(fe("padding", "pad".into()).is_err());
    assert!(fe("top_db", 80.into()).is_err());
    assert!(fe("fft_window_size", 1000.into()).is_err());
    assert!(fe("frequency_max", 30000.into()).is_err());
    // The class default truncation is "fusion", so an absent field fails.
    assert!(FeatureConfig::from_json(&serde_json::json!({})).is_err());
}

// ------------------------------------------------------------ real checkpoint

fn real_dir() -> Option<std::path::PathBuf> {
    model_dir("TURBOSPARK_CLAP_DIR", "clap-htsat-unfused")
}

fn cos_min(got: &[f32], want: &[f32], d: usize) -> f32 {
    got.chunks_exact(d)
        .zip(want.chunks_exact(d))
        .map(|(a, b)| cosine_similarity(a, b))
        .fold(f32::INFINITY, f32::min)
}

#[test]
#[ignore = "needs the real laion/clap-htsat-unfused checkpoint"]
fn real_checkpoint_parity() {
    let Some(dir) = real_dir() else { return };
    let clap = Clap::load(&dir).unwrap();
    let golden = read_json(FAMILY, "real/goldens.json");
    let d = clap.model().config().projection_dim;

    let mut audio = Vec::new();
    for spec in clips() {
        let pcm = synth_clip(&spec);
        let offset = spec["crop_offset"].as_u64().unwrap() as usize;
        audio.extend(
            clap.embed_audio_with(&pcm, LongAudioCrop::Offset(offset))
                .unwrap(),
        );
    }
    let (_, want_audio) = read_npy(FAMILY, "real/audio_embeds.npy");
    let audio_diff = max_abs_diff(&audio, &want_audio);
    let audio_cos = cos_min(&audio, &want_audio, d);
    eprintln!("real audio embeds: max |diff| {audio_diff:.3e}, min cosine {audio_cos:.9}");

    let labels: Vec<&str> = golden["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut text = Vec::new();
    for l in &labels {
        text.extend(clap.embed_text(l).unwrap());
    }
    let (_, want_text) = read_npy(FAMILY, "real/text_embeds.npy");
    let text_diff = max_abs_diff(&text, &want_text);
    let text_cos = cos_min(&text, &want_text, d);
    eprintln!("real text embeds: max |diff| {text_diff:.3e}, min cosine {text_cos:.9}");

    let want_logits: Vec<f32> = golden["logits_per_audio"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
        })
        .collect();
    let mut logits = Vec::new();
    for a in audio.chunks_exact(d) {
        logits.extend(clap.model().logits_per_audio(a, &text).unwrap());
    }
    let logit_diff = max_abs_diff(&logits, &want_logits);
    eprintln!("real logits_per_audio: max |diff| {logit_diff:.3e}");

    let want_probs = golden["probs"].as_array().unwrap();
    for (i, a) in audio.chunks_exact(d).enumerate() {
        let ranked = clap.classify(a, &labels).unwrap();
        let wp: Vec<f32> = want_probs[i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let want_rank: Vec<&str> = rank_descending(&wp)
            .into_iter()
            .map(|j| labels[j])
            .collect();
        let got_rank: Vec<&str> = ranked.iter().map(|(l, _)| l.as_str()).collect();
        let pdiff = ranked
            .iter()
            .map(|(l, p)| (p - wp[labels.iter().position(|x| x == l).unwrap()]).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "clip {i}: top {:?} p={:.6}, max |prob diff| {pdiff:.3e}, ranking equal: {}",
            ranked[0].0,
            ranked[0].1,
            got_rank == want_rank
        );
        assert_eq!(got_rank, want_rank, "clip {i} ranking");
    }
    // Measured on Apple M4 Max: audio 2.2e-7 / cos 1.0, text 5.0e-7 / cos
    // 0.99999994, logits 1.9e-5 (scale 18.66). Bounds leave ~10x headroom.
    assert!(
        audio_cos >= 0.999_999 && audio_diff <= 5e-6,
        "audio {audio_diff} {audio_cos}"
    );
    assert!(
        text_cos >= 0.999_999 && text_diff <= 5e-6,
        "text {text_diff} {text_cos}"
    );
    assert!(logit_diff <= 2e-4, "logits {logit_diff}");
}

#[test]
#[ignore = "needs the real laion/clap-htsat-unfused tokenizer.json"]
fn real_tokenizer_ids() {
    let Some(dir) = real_dir() else { return };
    let tok = crate::tokenize::TextTokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let golden = read_json(FAMILY, "real/goldens.json");
    for probe in golden["tokenizer_probes"].as_array().unwrap() {
        let text = probe["text"].as_str().unwrap();
        let want: Vec<u32> = probe["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(tok.encode(text, true).unwrap(), want, "{text:?}");
    }
}

#[test]
#[ignore = "timing on the real checkpoint; run with --release"]
fn real_timing() {
    let Some(dir) = real_dir() else { return };
    let t0 = std::time::Instant::now();
    let clap = Clap::load(&dir).unwrap();
    eprintln!("load: {:.3} s", t0.elapsed().as_secs_f64());
    let spec = clips()
        .into_iter()
        .find(|c| c["name"] == "chirp_noise_10s")
        .unwrap();
    let pcm = synth_clip(&spec);
    for run in 0..3 {
        let t = std::time::Instant::now();
        let mel = clap
            .feature_extractor()
            .extract(&pcm, LongAudioCrop::Head)
            .unwrap();
        let tf = t.elapsed().as_secs_f64();
        clap.model().audio_embedding(&mel).unwrap();
        let ta = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        clap.embed_text("a dog barking").unwrap();
        let tt = t.elapsed().as_secs_f64();
        eprintln!("run {run}: features {tf:.3} s, 10 s clip total {ta:.3} s, text {tt:.4} s");
    }
}
