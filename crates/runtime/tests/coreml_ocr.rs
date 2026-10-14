//! Real-model smoke gate for the Core ML Unlimited OCR pipeline. Ignored by
//! default.
//!
//! Model: `TURBOSPARK_UNLIMITED_OCR_COREML_MODEL` (default
//! `~/models/coreml/unlimited-ocr-coreml`): the Core ML parts of
//! FluidInference `unlimited-ocr-mlx-coreml` at
//! `4568571eacd864e0a253c609b3c9d0e00450cebc` (both `.mlpackage`s,
//! `embeddings.f16`, `tokenizer.json`, `config.json`).
//!
//! Pages are the rendered text in `testdata/coreml_ocr/`
//! (`tools/gen_ocr_pages.py`); the gate compares the plain-text output to
//! the known lines by character error rate.
//!
//!     cargo test --release -p turbospark-runtime --test coreml_ocr -- --ignored --nocapture

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use turbospark_runtime::coreml::{OcrOptions, OcrPage, UnlimitedOcrCoreMl};

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var_os("TURBOSPARK_UNLIMITED_OCR_COREML_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("models/coreml/unlimited-ocr-coreml")
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
        .join("testdata/coreml_ocr")
        .join(name)
}

fn expected_pages() -> Vec<String> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(testdata("pages.json")).unwrap()).unwrap();
    v["pages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            p["lines"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_str().unwrap())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// Whitespace-collapsed character error rate (Levenshtein over chars).
fn cer(reference: &str, hyp: &str) -> f64 {
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let (r, h): (Vec<char>, Vec<char>) = (
        norm(reference).chars().collect(),
        norm(hyp).chars().collect(),
    );
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rc) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hc) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(rc != hc))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len().max(1) as f64
}

/// Measured 0.000 on both pages, single and multi prompt (M4 Max, macOS
/// 26.6, GPU decoder). The bound leaves room for a one-word slip.
const MAX_CER: f64 = 0.02;

#[test]
#[ignore = "needs the Core ML Unlimited OCR checkpoint (about 4.1 GB)"]
fn reads_one_and_two_rendered_pages() {
    let Some(dir) = model_dir() else { return };
    let opts = OcrOptions {
        max_new_tokens: Some(800),
        ..OcrOptions::default()
    };
    let ocr = UnlimitedOcrCoreMl::open(&dir, &opts).unwrap();
    let want = expected_pages();
    let pages = [
        OcrPage::from_file(&testdata("page1.png")).unwrap(),
        OcrPage::from_file(&testdata("page2.png")).unwrap(),
    ];

    let one = ocr.read(&pages[..1], "single", |_| true).unwrap();
    let c = cer(&want[0], &one.plain_text());
    eprintln!("single page: cer {c:.3}, {:?}", one.stats);
    assert!(one.stats.stopped_at_eos, "single page did not finish");
    assert!(c <= MAX_CER, "single page cer {c}: {}", one.text);

    let two = ocr.read(&pages, "multi", |_| true).unwrap();
    let text = two.plain_text();
    let got: Vec<&str> = text.split("\n\n").collect();
    eprintln!("two pages: {:?}", two.stats);
    assert_eq!(got.len(), 2, "expected two pages: {}", two.text);
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        let c = cer(w, g);
        eprintln!("page {}: cer {c:.3}", i + 1);
        assert!(c <= MAX_CER, "page {} cer {c}: {g}", i + 1);
    }

    // Cancellation from the progress callback stops the read.
    assert!(ocr.read(&pages[..1], "single", |_| false).is_err());
    // Unknown prompts and too many pages are refused before any work.
    assert!(ocr.read(&pages[..1], "nope", |_| true).is_err());
    let many: Vec<OcrPage> = (0..ocr.max_pages() + 1).map(|_| pages[0].clone()).collect();
    assert!(ocr.read(&many, "multi", |_| true).is_err());
}
