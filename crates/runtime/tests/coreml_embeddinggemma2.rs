//! Real-model gates for the Core ML EmbeddingGemma 2 host. Ignored by
//! default.
//!
//! Model: `TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL` (default
//! `~/models/coreml/embeddinggemma-2-coreml`), FluidInference
//! `embeddinggemma-2-coreml` at `3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4`.
//!
//! The reference vectors in `testdata/coreml_embeddinggemma2/` come from
//! `tools/ref_embeddinggemma2_coreml.py`: the same packages run through
//! coremltools on the CPU with independently written preprocessing
//! (transformers' Gemma4 image processor for images). Matching them checks
//! this host's tokenization, table lookup, image and audio preprocessing,
//! and marshalling.
//!
//!     cargo test --release -p turbospark-runtime --test coreml_embeddinggemma2 -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use turbospark_coreml::ComputeUnits;
use turbospark_runtime::coreml::{EmbeddingGemma2CoreMl, EmbeddingGemma2Options};

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var_os("TURBOSPARK_EMBEDDINGGEMMA2_COREML_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("models/coreml/embeddinggemma-2-coreml")
        });
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!("skipping: {} not found", p.display());
        None
    }
}

fn testdata(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/coreml_embeddinggemma2")
        .join(name)
}

fn reference() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(testdata("reference.json")).unwrap()).unwrap()
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn cpu() -> EmbeddingGemma2Options {
    EmbeddingGemma2Options {
        text_units: ComputeUnits::CpuOnly,
        media_units: ComputeUnits::CpuOnly,
    }
}

/// Minimum cosine to the coremltools reference on the same CPU path.
/// Measured on an M4 Max, macOS 26.6: text 0.999999 to 1.000000 (five
/// cases, identical token ids), image 0.999999 (budget 70) and 1.000000
/// (280), audio 1.000000. Same package, same compute units, same inputs.
const SAME_PATH_MIN: f32 = 0.9999;

#[test]
#[ignore = "needs the Core ML EmbeddingGemma 2 checkpoint"]
fn text_matches_coremltools_reference() {
    let Some(dir) = model_dir() else { return };
    let m = EmbeddingGemma2CoreMl::open(&dir, &cpu()).unwrap();
    for case in reference()["texts"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let task = case["task"].as_str();
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            m.token_ids(text, task).unwrap(),
            ids,
            "token ids for {text:?}"
        );
        let got = &m.embed_texts(&[text], task, None).unwrap()[0];
        let c = cosine(got, &floats(&case["embedding"]));
        eprintln!("text {task:?} ({} ids): cosine {c:.6}", ids.len());
        assert!(c >= SAME_PATH_MIN, "{text:?}: cosine {c}");
    }
    // Matryoshka: the leading 256 values, re-normalized.
    let full = &m.embed_texts(&["hello"], None, None).unwrap()[0];
    let small = &m.embed_texts(&["hello"], None, Some(256)).unwrap()[0];
    assert_eq!(small.len(), 256);
    let norm: f32 = small.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
    assert!(cosine(&full[..256], small) > 0.99999);
    assert!(m.embed_texts(&["hello"], None, Some(300)).is_err());
    assert!(m
        .embed_texts(&["hello"], Some("no_such_task"), None)
        .is_err());
}

#[test]
#[ignore = "needs the Core ML EmbeddingGemma 2 checkpoint"]
fn image_matches_reference_preprocessing() {
    let Some(dir) = model_dir() else { return };
    let m = EmbeddingGemma2CoreMl::open(&dir, &cpu()).unwrap();
    let img = turbospark_vision_io::decode_image_file(&testdata("pattern.png")).unwrap();
    for case in reference()["images"].as_array().unwrap() {
        let budget = case["budget"].as_u64().unwrap() as usize;
        let got = m
            .embed_image(&img.data, img.width, img.height, Some(budget), None)
            .unwrap();
        let c = cosine(&got, &floats(&case["embedding"]));
        eprintln!("image budget {budget}: cosine {c:.6}");
        assert!(c >= SAME_PATH_MIN, "budget {budget}: cosine {c}");
    }
}

#[test]
#[ignore = "needs the Core ML EmbeddingGemma 2 checkpoint"]
fn audio_matches_reference() {
    let Some(dir) = model_dir() else { return };
    let m = EmbeddingGemma2CoreMl::open(&dir, &cpu()).unwrap();
    let w = audio::wav::read_wav_f32(&testdata("chirp.wav")).unwrap();
    assert_eq!((w.sample_rate, w.channels), (16_000, 1));
    let windows = m.embed_audio(&w.samples, None).unwrap();
    assert_eq!(windows.len(), 1);
    let case = &reference()["audio"][0];
    let c = cosine(&windows[0].embedding, &floats(&case["embedding"]));
    eprintln!("audio: cosine {c:.6}");
    assert!(c >= SAME_PATH_MIN, "audio cosine {c}");
}

/// The default placement (text on the Neural Engine) against the CPU
/// reference. The first run compiles every text function for the Neural
/// Engine, which takes minutes. Reports the cosines; the bound is the
/// measured value with headroom.
#[test]
#[ignore = "needs the Core ML EmbeddingGemma 2 checkpoint; first run compiles for the Neural Engine"]
fn neural_engine_text_tracks_cpu() {
    let Some(dir) = model_dir() else { return };
    let started = std::time::Instant::now();
    let m = EmbeddingGemma2CoreMl::open(&dir, &EmbeddingGemma2Options::default()).unwrap();
    eprintln!(
        "opened with {:?} in {:?}",
        EmbeddingGemma2Options::default(),
        started.elapsed()
    );
    let mut worst = 1.0f32;
    for case in reference()["texts"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let started = std::time::Instant::now();
        let got = &m.embed_texts(&[text], case["task"].as_str(), None).unwrap()[0];
        let elapsed = started.elapsed();
        let c = cosine(got, &floats(&case["embedding"]));
        eprintln!("ane text: cosine {c:.6} in {elapsed:?}");
        worst = worst.min(c);
    }
    assert!(worst >= ANE_MIN, "worst Neural Engine cosine {worst}");
}

/// Neural Engine text against the CPU reference. Measured worst 0.999582
/// over the five cases (M4 Max, macOS 26.6); the card reports >= 0.9984
/// against PyTorch fp32. The bound sits below both.
const ANE_MIN: f32 = 0.998;

// int8 derivative: lorelaiassistant/embeddinggemma-2-coreml-int8 at
// `45446c9c1bac44874191b6311dd4c26e0af313aa`, the pinned FluidInference
// packages weight-quantized (text and vision only). Its vectors are compared
// with the fp16 reference above: they are close to it, not equal, so the
// bound is a quantization budget, not a same-path check.

fn int8_dir() -> Option<PathBuf> {
    let p = std::env::var_os("TURBOSPARK_EMBEDDINGGEMMA2_COREML_INT8_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("models/coreml/embeddinggemma-2-coreml-int8")
        });
    if p.is_dir() {
        Some(p)
    } else {
        eprintln!("skipping: {} not found", p.display());
        None
    }
}

/// Opens the int8 install on `units` (text and media) and checks every
/// reference text and image against the fp16 vectors. Non-finite output is
/// refused inside `embed_texts` / `embed_image`, so a NaN placement fails.
fn int8_tracks_fp16(units: ComputeUnits) {
    let Some(dir) = int8_dir() else { return };
    let opts = EmbeddingGemma2Options {
        text_units: units,
        media_units: units,
    };
    let started = std::time::Instant::now();
    let m = EmbeddingGemma2CoreMl::open(&dir, &opts).unwrap();
    eprintln!("int8 {units:?}: opened in {:?}", started.elapsed());
    let r = reference();
    let mut worst_text = 1.0f32;
    for case in r["texts"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let task = case["task"].as_str();
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            m.token_ids(text, task).unwrap(),
            ids,
            "token ids for {text:?}"
        );
        let got = &m.embed_texts(&[text], task, None).unwrap()[0];
        let c = cosine(got, &floats(&case["embedding"]));
        eprintln!(
            "int8 {units:?} text {task:?} ({} ids): cosine {c:.6}",
            ids.len()
        );
        worst_text = worst_text.min(c);
    }
    let img = turbospark_vision_io::decode_image_file(&testdata("pattern.png")).unwrap();
    let mut worst_image = 1.0f32;
    for case in r["images"].as_array().unwrap() {
        let budget = case["budget"].as_u64().unwrap() as usize;
        let got = m
            .embed_image(&img.data, img.width, img.height, Some(budget), None)
            .unwrap();
        let c = cosine(&got, &floats(&case["embedding"]));
        eprintln!("int8 {units:?} image budget {budget}: cosine {c:.6}");
        worst_image = worst_image.min(c);
    }
    // No audio tower in this install: a clear error, not a crash.
    let err = m.embed_audio(&vec![0.01f32; 16_000], None).unwrap_err();
    eprintln!("int8 audio refused: {err}");
    assert!(
        worst_text >= INT8_TEXT_MIN,
        "worst int8 text cosine {worst_text}"
    );
    assert!(
        worst_image >= INT8_IMAGE_MIN,
        "worst int8 image cosine {worst_image}"
    );
}

/// int8 against the fp16 reference, M4 Max, macOS 26.5, one run per
/// placement over the five texts and two image budgets: text worst 0.999337
/// (CPU), 0.999006 (Neural Engine), 0.999097 (GPU); image worst 0.999565,
/// 0.999581, 0.999593. The card reports min 0.9991 (queries) and 0.9989
/// (images) at 256-d. The bounds sit below both.
const INT8_TEXT_MIN: f32 = 0.998;
const INT8_IMAGE_MIN: f32 = 0.998;

#[test]
#[ignore = "needs the int8 Core ML EmbeddingGemma 2 install"]
fn int8_cpu_tracks_fp16() {
    int8_tracks_fp16(ComputeUnits::CpuOnly);
}

#[test]
#[ignore = "needs the int8 Core ML EmbeddingGemma 2 install; first run compiles for the Neural Engine"]
fn int8_neural_engine_tracks_fp16() {
    int8_tracks_fp16(ComputeUnits::CpuAndNeuralEngine);
}

/// Measured separately: FluidAudio reports int8 Core ML programs aborting
/// the process on the GPU, which would take the other tests with it.
#[test]
#[ignore = "needs the int8 Core ML EmbeddingGemma 2 install"]
fn int8_gpu_tracks_fp16() {
    int8_tracks_fp16(ComputeUnits::CpuAndGpu);
}
