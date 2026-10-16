//! One measured run of a Core ML pipeline, for
//! `crates/runtime/scripts/benchmark_coreml.py`.
//!
//! ```text
//! coreml_backend_probe parakeet <model-dir> <wav> <coreml_gpu|coreml_ane|coreml_cpu|cpu_redux>
//! coreml_backend_probe embed <model-dir> <cpu|gpu|ane>
//! coreml_backend_probe ocr <model-dir> <page.png> [page.png ...]
//! ```
//!
//! Each process opens one arm, runs one discarded warmup, then one measured
//! pass, and prints a `coreml_probe` metric line on stderr plus a digest
//! line on stdout, so the script can check that every arm produced the same
//! output. `cpu_redux` opens the CPU port (`audio::stt::parakeet`) on a
//! `moondream/parakeet-redux` directory instead of the Core ML export.
//! `coreml_gpu` on an encoder not verified on the GPU (an int8 one) is
//! refused at open; `TURBOSPARK_COREML_ALLOW_QUANTIZED_GPU=1` lets it run,
//! and only ever in this child process, because Core ML can abort there.
//! macOS-only.

#![cfg(target_os = "macos")]

use std::path::Path;
use std::time::Instant;

use turbospark_coreml::ComputeUnits;
use turbospark_runtime::coreml::{
    EmbeddingGemma2CoreMl, EmbeddingGemma2Options, OcrOptions, OcrPage, ParakeetCoreMl,
    ParakeetCoreMlOptions, UnlimitedOcrCoreMl,
};

type Res<T> = Result<T, String>;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn digest(s: &str) -> String {
    // FNV-1a: only equality across arms matters, not cryptographic strength.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

fn parakeet(dir: &Path, wav: &Path, arm: &str) -> Res<()> {
    let wave = audio::wav::read_wav_f32(wav).map_err(|e| e.to_string())?;
    let pcm = audio::conversion::to_mono_resampled(
        &wave,
        16_000,
        &audio::conversion::MonoResampleStrategy::SincHann(Default::default()),
    )
    .map_err(|e| e.to_string())?;
    let opened = Instant::now();
    let run: Box<dyn Fn() -> Res<String>> = match arm {
        "cpu_redux" => {
            let m = audio::stt::parakeet::ParakeetTdt::open(dir).map_err(|e| e.to_string())?;
            let pcm = pcm.clone();
            Box::new(move || m.transcribe(&pcm, 16_000).map_err(|e| e.to_string()))
        }
        "coreml_gpu" | "coreml_ane" | "coreml_cpu" => {
            let opts = ParakeetCoreMlOptions {
                encoder_units: match arm {
                    "coreml_ane" => ComputeUnits::CpuAndNeuralEngine,
                    "coreml_cpu" => ComputeUnits::CpuOnly,
                    _ => ComputeUnits::CpuAndGpu,
                },
                ..ParakeetCoreMlOptions::default()
            };
            let m = ParakeetCoreMl::open(dir, &opts)?;
            let pcm = pcm.clone();
            Box::new(move || m.transcribe(&pcm, 16_000))
        }
        other => return Err(format!("unknown parakeet arm {other}")),
    };
    let open_ms = ms(opened);
    let t = Instant::now();
    let warm = run()?;
    let warmup_ms = ms(t);
    let t = Instant::now();
    let text = run()?;
    let measured_ms = ms(t);
    if warm != text {
        return Err("warmup and measured transcripts differ".into());
    }
    let audio_s = pcm.len() as f64 / 16_000.0;
    println!("DIGEST: {}", digest(&text));
    eprintln!(
        "coreml_probe task=parakeet arm={arm} open_ms={open_ms:.2} warmup_ms={warmup_ms:.2} \
         measured_ms={measured_ms:.2} audio_s={audio_s:.3} rtfx={:.2}",
        audio_s / (measured_ms / 1e3)
    );
    Ok(())
}

/// Fixed text set: 32 short sentences and 8 of about 100 tokens, so both
/// the `embed_32` and `embed_128` functions are timed.
fn texts() -> Vec<String> {
    let short = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
        "Bananas are a good source of potassium.",
    ];
    let long = "Lighthouses guided ships along dangerous coasts for centuries, \
                and their keepers tended oil lamps through long winter nights. ";
    let mut v: Vec<String> = (0..32).map(|i| format!("{} ({i})", short[i % 4])).collect();
    v.extend((0..8).map(|i| format!("{}({i})", long.repeat(4))));
    v
}

fn embed(dir: &Path, arm: &str) -> Res<()> {
    let units = match arm {
        "cpu" => ComputeUnits::CpuOnly,
        "gpu" => ComputeUnits::CpuAndGpu,
        "ane" => ComputeUnits::CpuAndNeuralEngine,
        other => return Err(format!("unknown embed arm {other}")),
    };
    let opts = EmbeddingGemma2Options {
        text_units: units,
        ..EmbeddingGemma2Options::default()
    };
    let opened = Instant::now();
    let m = EmbeddingGemma2CoreMl::open(dir, &opts)?;
    let open_ms = ms(opened);
    let texts = texts();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let t = Instant::now();
    let warm = m.embed_texts(&refs, Some("document"), None)?;
    let warmup_ms = ms(t);
    let t = Instant::now();
    let v = m.embed_texts(&refs, Some("document"), None)?;
    let measured_ms = ms(t);
    // Digest of the rounded vectors: placements differ in the last bits,
    // so equality is checked at 2 decimals and cosine is reported apart.
    let rounded: String = v
        .iter()
        .flat_map(|e| e.iter().map(|x| format!("{:.2},", x)))
        .collect();
    let min_cos = warm
        .iter()
        .zip(&v)
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>())
        .fold(1.0f32, f32::min);
    println!("DIGEST: {}", digest(&rounded));
    eprintln!(
        "coreml_probe task=embed arm={arm} open_ms={open_ms:.2} warmup_ms={warmup_ms:.2} \
         measured_ms={measured_ms:.2} texts={} texts_per_s={:.1} warm_min_cos={min_cos:.6}",
        texts.len(),
        texts.len() as f64 / (measured_ms / 1e3)
    );
    Ok(())
}

fn ocr(dir: &Path, pages: &[String]) -> Res<()> {
    let pages = pages
        .iter()
        .map(|p| OcrPage::from_file(Path::new(p)))
        .collect::<Res<Vec<_>>>()?;
    let prompt = if pages.len() == 1 { "single" } else { "multi" };
    let opened = Instant::now();
    let m = UnlimitedOcrCoreMl::open(dir, &OcrOptions::default())?;
    let open_ms = ms(opened);
    let t = Instant::now();
    let warm = m.read(&pages, prompt, |_| true)?;
    let warmup_ms = ms(t);
    let t = Instant::now();
    let r = m.read(&pages, prompt, |_| true)?;
    let measured_ms = ms(t);
    if warm.text != r.text {
        return Err("warmup and measured OCR output differ".into());
    }
    let s = &r.stats;
    println!("DIGEST: {}", digest(&r.text));
    eprintln!(
        "coreml_probe task=ocr arm=gpu open_ms={open_ms:.2} warmup_ms={warmup_ms:.2} \
         measured_ms={measured_ms:.2} pages={} encode_ms={:.2} prefill_ms={:.2} decode_ms={:.2} \
         tokens={} decode_tok_s={:.2}",
        s.pages,
        s.encode_ms,
        s.prefill_ms,
        s.decode_ms,
        s.output_tokens,
        s.output_tokens as f64 / (s.decode_ms / 1e3)
    );
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: coreml_backend_probe parakeet <dir> <wav> <arm> | embed <dir> <arm> | ocr <dir> <page>...";
    match args.first().map(String::as_str) {
        Some("parakeet") if args.len() == 4 => {
            parakeet(Path::new(&args[1]), Path::new(&args[2]), &args[3])
        }
        Some("embed") if args.len() == 3 => embed(Path::new(&args[1]), &args[2]),
        Some("ocr") if args.len() >= 3 => ocr(Path::new(&args[1]), &args[2..]),
        _ => Err(usage.into()),
    }
}
