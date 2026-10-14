//! Reads page images with the Core ML Unlimited OCR export.
//!
//! Usage: `cargo run --release -p turbospark-runtime --example ocr_coreml --
//! <model_dir> <page.png> [page2.png ...]`
//!
//! The model directory holds the Core ML parts of FluidInference
//! `unlimited-ocr-mlx-coreml`. One page uses the `single` prompt, several
//! the `multi` prompt. `OCR_MAX_NEW_TOKENS` caps the output. macOS 15+.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use turbospark_runtime::coreml::{OcrEvent, OcrOptions, OcrPage, UnlimitedOcrCoreMl};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .ok_or("usage: ocr_coreml <model_dir> <page> [page ...]")?;
    let pages: Vec<PathBuf> = args.map(PathBuf::from).collect();
    if pages.is_empty() {
        return Err("give at least one page image".into());
    }
    let opts = OcrOptions {
        max_new_tokens: std::env::var("OCR_MAX_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok()),
        ..OcrOptions::default()
    };
    let started = std::time::Instant::now();
    let ocr = UnlimitedOcrCoreMl::open(Path::new(&model_dir), &opts)?;
    eprintln!(
        "opened in {:?} (max {} pages)",
        started.elapsed(),
        ocr.max_pages()
    );
    let pages = pages
        .iter()
        .map(|p| OcrPage::from_file(p))
        .collect::<Result<Vec<_>, _>>()?;
    let prompt = if pages.len() == 1 { "single" } else { "multi" };
    let result = ocr.read(&pages, prompt, |e| {
        if let OcrEvent::Prefilled { tokens, capacity } = e {
            eprintln!("prefilled {tokens} tokens (bucket {capacity})");
        }
        true
    })?;
    let s = &result.stats;
    eprintln!(
        "encode {:.0} ms, prefill {:.0} ms, decode {} tokens in {:.0} ms ({:.1} ms/token), eos {}",
        s.encode_ms,
        s.prefill_ms,
        s.output_tokens,
        s.decode_ms,
        s.decode_ms / s.output_tokens.max(1) as f64,
        s.stopped_at_eos
    );
    println!("{}", result.text);
    Ok(())
}
