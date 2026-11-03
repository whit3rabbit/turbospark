//! The real Unlimited-OCR-2 tower against mlx-vlm, stage by stage, then the
//! whole pipeline on our own preprocessing and tower. See
//! `unlimited_ocr2_real.rs` for the environment and the dump.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use half::f16;
use turbospark_runtime::{LogitProducer, RealForwardRunner};
use turbospark_vision_io::{decode_image_file, uocr2_gundam_page};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

fn read_f16(path: PathBuf) -> Vec<f32> {
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}

/// Cosine, and the worst row's cosine: a mean can hide one bad row.
fn cosine(a: &[f32], b: &[f32], row: usize) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|v| v.is_finite()), "non-finite rows");
    let cos = |x: &[f32], y: &[f32]| {
        let (mut d, mut n1, mut n2) = (0.0f64, 0.0f64, 0.0f64);
        for (p, q) in x.iter().zip(y) {
            d += f64::from(*p) * f64::from(*q);
            n1 += f64::from(*p) * f64::from(*p);
            n2 += f64::from(*q) * f64::from(*q);
        }
        d / (n1.sqrt() * n2.sqrt()).max(1e-30)
    };
    let worst = a
        .chunks_exact(row)
        .zip(b.chunks_exact(row))
        .map(|(x, y)| cos(x, y))
        .fold(1.0f64, f64::min);
    (cos(a, b), worst)
}

fn runner() -> RealForwardRunner {
    RealForwardRunner::open(
        &env_dir("TURBOSPARK_UOCR2_INSTALL_DIR"),
        model_io::unlimited_ocr2_12l(),
    )
    .expect("open")
}

#[test]
#[ignore = "needs the real install and an mlx-vlm dump"]
fn preprocessing_matches_pil() {
    let dump = env_dir("TURBOSPARK_UOCR2_DUMP");
    let image = decode_image_file(&dump.join("page.png")).unwrap();
    let page = uocr2_gundam_page(&image.data, image.width, image.height);
    assert_eq!(page.grid, (3, 4));
    let close = |got: &[f32], want: &[f32], what: &str| {
        assert_eq!(got.len(), want.len(), "{what}");
        // Upstream rounds its pixels to BF16 (`astype(mx.bfloat16)`), so ours
        // take the same rounding before comparing. Ours stay finer in the
        // tower. What remains is the FP16 dump's own rounding.
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (compute::bf16_to_f32(compute::f32_to_bf16(*a)) - b).abs())
            .fold(0.0f32, f32::max);
        println!("{what}: worst pixel error {worst}");
        assert!(worst < 1e-3, "{what}: worst pixel error {worst}");
    };
    close(&page.base, &read_f16(dump.join("pixels_base.f16")), "base");
    let flat: Vec<f32> = page.crops.iter().flatten().copied().collect();
    close(&flat, &read_f16(dump.join("pixels_crops.f16")), "crops");
}

#[test]
#[ignore = "needs the real install and an mlx-vlm dump"]
fn each_tower_stage_matches_mlx_vlm() {
    let dump = env_dir("TURBOSPARK_UOCR2_DUMP");
    let mut runner = runner();
    let crops = read_f16(dump.join("pixels_crops.f16"));
    for (label, pixels, side, dim_grid) in [
        (
            "base",
            read_f16(dump.join("pixels_base.f16")),
            1024usize,
            16usize,
        ),
        ("crop", crops[..640 * 640 * 3].to_vec(), 640, 10),
    ] {
        let started = std::time::Instant::now();
        let stages = runner.unlimited_ocr2_view_stages(&pixels, side).unwrap();
        println!("{label}: tower {:.2}s", started.elapsed().as_secs_f64());
        let as_f32 = |v: &[f16]| v.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
        for (stage, ours, row) in [
            ("patch", as_f32(&stages.patch), 768),
            ("sam", as_f32(&stages.sam), 1024),
            ("clip", as_f32(&stages.clip), 1024),
            ("projected", as_f32(&stages.projected), 1280),
        ] {
            let want = read_f16(dump.join(format!("stage_{label}_{stage}.f16")));
            let (mean, worst) = cosine(&ours, &want, row);
            println!("{label} {stage}: cosine {mean:.8}, worst row {worst:.8}");
            assert!(mean > 0.999, "{label} {stage}: cosine {mean}");
        }
        assert_eq!(stages.projected.len(), dim_grid * dim_grid * 1280);
    }
}

#[test]
#[ignore = "needs the real install and an mlx-vlm dump"]
fn the_whole_pipeline_reads_the_page() {
    let dump = env_dir("TURBOSPARK_UOCR2_DUMP");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dump.join("reference.json")).unwrap()).unwrap();
    let ids: Vec<i32> = reference["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let image = decode_image_file(&dump.join("page.png")).unwrap();
    let page = uocr2_gundam_page(&image.data, image.width, image.height);
    let mut runner = runner();
    let started = std::time::Instant::now();
    let rows = runner.encode_unlimited_ocr2_page(&page).unwrap();
    println!("page encoded in {:.2}s", started.elapsed().as_secs_f64());
    let want_rows = read_f16(dump.join("image_rows.f16"));
    let ours: Vec<f32> = rows.iter().map(|x| x.to_f32()).collect();
    let (mean, worst) = cosine(&ours, &want_rows, 1280);
    println!("fused rows: cosine {mean:.8}, worst row {worst:.8}");
    assert!(mean > 0.999, "fused rows cosine {mean}");

    runner.set_unlimited_ocr2_prompt(&ids, rows).unwrap();
    LogitProducer::reset(&mut runner);
    let vocab = 129_280;
    let mut logits = vec![f16::from_f32(0.0); vocab];
    for (position, &token) in ids.iter().enumerate() {
        runner.produce(token, position, &mut logits).unwrap();
    }
    // Free running: this is what a caller gets.
    let mut got = Vec::new();
    let mut position = ids.len();
    for _ in 0..400 {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
            .map(|(i, _)| i as i32)
            .unwrap();
        got.push(next);
        if next == 1 {
            break;
        }
        runner.produce(next, position, &mut logits).unwrap();
        position += 1;
    }
    let tok =
        tokenizer::MfTokenizer::load_from_dir(&env_dir("TURBOSPARK_UOCR2_INSTALL_DIR")).unwrap();
    let text = tok.decode(&got, false);
    println!("{text}");
    assert_eq!(*got.last().unwrap(), 1, "never reached EOS");
    for line in [
        "Quarterly Report on Sparse Attention",
        "Pages parsed: 1,284. Mean latency: 41.7 ms. Errors: 3.",
        "reviewed on 14 March 2026",
    ] {
        assert!(text.contains(line), "missing {line:?}");
    }
}
