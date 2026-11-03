//! The real Unlimited-OCR-2 decoder against mlx-vlm, given mlx-vlm's own prompt
//! ids and fused image rows, so nothing here can be a tower or tokenizer bug.
//!
//! ```sh
//! TURBOSPARK_UOCR2_INSTALL_DIR=~/models/Unlimited-OCR-2-4bit.gturbo \
//! TURBOSPARK_UOCR2_DUMP=... \
//!   cargo test -p turbospark-runtime --test unlimited_ocr2_real --release -- --ignored --nocapture
//! ```
//!
//! The dump comes from `scripts/unlimited_ocr2_decoder_dump.py`.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use half::f16;
use turbospark_runtime::{LogitProducer, RealForwardRunner};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

fn read_f16(path: PathBuf) -> Vec<f16> {
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn ints(v: &serde_json::Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect()
}

fn top5(x: &[f16]) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[b].to_f32().total_cmp(&x[a].to_f32()));
    idx.truncate(5);
    idx.into_iter().map(|i| i as i32).collect()
}

#[test]
#[ignore = "needs the real install and an mlx-vlm dump"]
fn the_decoder_reads_the_page_mlx_vlm_reads() {
    let install = env_dir("TURBOSPARK_UOCR2_INSTALL_DIR");
    let dump = env_dir("TURBOSPARK_UOCR2_DUMP");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dump.join("reference.json")).unwrap()).unwrap();
    let ids = ints(&reference["ids"]);
    let want = ints(&reference["tokens"]);
    let rows = read_f16(dump.join("image_rows.f16"));

    let arch = model_io::unlimited_ocr2_12l();
    let vocab = arch.vocab_size as usize;
    let mut runner = RealForwardRunner::open(&install, arch).expect("open");
    runner
        .set_unlimited_ocr2_prompt(&ids, rows)
        .expect("prompt");
    LogitProducer::reset(&mut runner);
    runner
        .set_unlimited_ocr2_prompt(&ids, read_f16(dump.join("image_rows.f16")))
        .expect("prompt");

    let mut logits = vec![f16::from_f32(0.0); vocab];
    let started = std::time::Instant::now();
    for (position, &token) in ids.iter().enumerate() {
        runner
            .produce(token, position, &mut logits)
            .expect("prefill");
    }
    println!(
        "prefill {} tokens in {:.2}s",
        ids.len(),
        started.elapsed().as_secs_f64()
    );

    let started = std::time::Instant::now();
    let mut got = Vec::new();
    let mut position = ids.len();
    let mut top_mismatch = 0;
    let mut wide_mismatch = 0;
    let mut logit_gap = 0.0f32;
    let (mut gap_sum, mut gap_count) = (0.0f64, 0u64);
    let allocations = runner.gpu_buffer_allocations();
    for (step, top) in reference["top"].as_array().unwrap().iter().enumerate() {
        let want_top = ints(&top["ids"]);
        let ours = top5(&logits);
        let want_logits: Vec<f32> = top["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        if ours[0] != want_top[0] {
            top_mismatch += 1;
            // Near-ties flip on FP16 rounding: the reference's own top logits
            // are quantized to 0.25 at this magnitude.
            if want_logits[0] - want_logits[1] > 0.5 {
                wide_mismatch += 1;
            }
            println!(
                "step {step}: ours {} reference {}, reference margin {:.3}, our margin {:.3}",
                ours[0],
                want_top[0],
                want_logits[0] - want_logits[1],
                logits[ours[0] as usize].to_f32() - logits[want_top[0] as usize].to_f32(),
            );
        }
        for (id, want_logit) in want_top.iter().zip(&want_logits) {
            let gap = (logits[*id as usize].to_f32() - want_logit).abs();
            logit_gap = logit_gap.max(gap);
            gap_sum += f64::from(gap);
            gap_count += 1;
        }
        // Teacher-forced on the reference's token, so one disagreement cannot
        // hide every step after it.
        let forced = want[step];
        got.push(ours[0]);
        if forced == 1 {
            break;
        }
        runner
            .produce(forced, position, &mut logits)
            .expect("decode");
        position += 1;
    }
    println!(
        "decode {} tokens at {:.1} tok/s",
        got.len(),
        got.len() as f64 / started.elapsed().as_secs_f64()
    );
    println!(
        "argmax mismatches {top_mismatch}/{}, top-5 logit error mean {:.4} worst {logit_gap}",
        got.len(),
        gap_sum / gap_count as f64
    );
    assert_eq!(
        runner.gpu_buffer_allocations(),
        allocations,
        "the decode hot path allocated a Metal buffer"
    );
    let tok = tokenizer::MfTokenizer::load_from_dir(&install).expect("tokenizer");
    println!("{}", tok.decode(&got, false));
    // Measured on one page, so a sentinel and not a quality claim. A wrong route
    // or wrong rope moves logits by whole units across most steps.
    assert_eq!(
        wide_mismatch, 0,
        "greedy disagrees with mlx-vlm off a near-tie"
    );
    assert!(
        top_mismatch * 20 <= got.len(),
        "{top_mismatch}/{} near-ties is more than FP16 rounding explains",
        got.len()
    );
    assert!(
        gap_sum / (gap_count as f64) < 0.2,
        "top-5 logits drift from the reference by {}",
        gap_sum / gap_count as f64
    );
}
