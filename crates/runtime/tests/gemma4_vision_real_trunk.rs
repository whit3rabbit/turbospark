//! The real Gemma 4 install answering an image prompt, against mlx-vlm
//! (`docs/GEMMA4_VISION_PHASE0.md`).
//!
//! Needs an install repacked WITH its tower
//! (`gemma4_checkpoint_network::repacks_the_real_gemma4_checkpoint_with_its_vision_tower`)
//! and a dump from `scripts/gemma4_vision_trunk_dump.py`:
//!
//! ```sh
//! TURBOSPARK_GEMMA4_VISION_INSTALL_DIR=... TURBOSPARK_GEMMA4_TRUNK_DUMP=... \
//!   cargo test -p turbospark-runtime --test gemma4_vision_real_trunk --release -- --ignored --nocapture
//! ```
//!
//! The prompt is mlx-vlm's own token ids, so tokenization is not under test;
//! the image is the same RGB bytes, preprocessed by `vision-io`, whose parity
//! with the reference processor is held elsewhere. What is under test is the
//! tower on the install, the injection and the image-span mask.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

use half::f16;
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::{gemma4_preprocess, ImageSpan, Rgb8Image, GEMMA4_DEFAULT_SOFT_TOKENS};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

fn read_f32(path: PathBuf) -> Vec<f32> {
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn log_softmax(x: &[f32]) -> Vec<f64> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = x.iter().map(|&v| (v as f64 - max).exp()).sum();
    x.iter().map(|&v| v as f64 - max - sum.ln()).collect()
}

/// KL(reference || ours) in nats, and the top-k index lists.
fn compare(ours: &[f32], reference: &[f32]) -> (f64, Vec<usize>, Vec<usize>) {
    let (lp, lq) = (log_softmax(reference), log_softmax(ours));
    let kl = lp.iter().zip(&lq).map(|(p, q)| p.exp() * (p - q)).sum();
    let top = |x: &[f32]| {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&a, &b| x[b].total_cmp(&x[a]));
        idx.truncate(5);
        idx
    };
    (kl, top(ours), top(reference))
}

fn argmax(x: &[f16]) -> i32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.to_f32().total_cmp(&b.1.to_f32()))
        .map(|(i, _)| i as i32)
        .unwrap()
}

#[test]
#[ignore = "needs the real Gemma 4 install with its tower and an mlx-vlm trunk dump"]
fn the_real_install_answers_an_image_prompt_like_mlx_vlm() {
    let install = env_dir("TURBOSPARK_GEMMA4_VISION_INSTALL_DIR");
    let dump = env_dir("TURBOSPARK_GEMMA4_TRUNK_DUMP");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dump.join("manifest.json")).unwrap()).unwrap();
    let ids: Vec<i32> = manifest["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let image_id = manifest["image_token_id"].as_i64().unwrap() as i32;
    let vocab = manifest["vocab"].as_u64().unwrap() as usize;

    let arch = turbospark_repack::peek_manifest_arch(&install).expect("install manifest");
    assert_eq!(arch.vision.tower, model_io::VisionTowerKind::Gemma4);
    assert_eq!(arch.vision.image_token_id as i32, image_id);
    let mut runner = RealForwardRunner::open(&install, arch).expect("install opens");

    let hw = &manifest["image_hw"];
    let (h, w) = (
        hw[0].as_u64().unwrap() as usize,
        hw[1].as_u64().unwrap() as usize,
    );
    let rgb = std::fs::read(dump.join("image.rgb")).unwrap();
    let image = Rgb8Image::new(w, h, rgb).expect("image bytes");
    let patches = gemma4_preprocess(&image, GEMMA4_DEFAULT_SOFT_TOKENS).expect("preprocess");
    let started = std::time::Instant::now();
    let rows = runner.encode_image_gemma4(&patches).expect("tower encodes");
    eprintln!(
        "tower: {} patches -> {} soft tokens in {:.0} ms",
        patches.num_patches(),
        patches.soft_tokens(),
        started.elapsed().as_secs_f64() * 1e3
    );

    let start = ids
        .iter()
        .position(|&t| t == image_id)
        .expect("an image span");
    let len = ids[start..].iter().take_while(|&&t| t == image_id).count();
    assert_eq!(
        len,
        patches.soft_tokens(),
        "span length vs the tower's soft tokens"
    );
    assert_eq!(
        ids.iter().filter(|&&t| t == image_id).count(),
        len,
        "one image"
    );

    let span = ImageSpan { start, len };
    let prefill = |runner: &mut RealForwardRunner| -> Vec<f16> {
        runner.reset();
        runner
            .set_prompt_vision_gemma4(std::slice::from_ref(&rows), &[span], ids.len())
            .expect("image rows set");
        let mut logits = vec![f16::from_f32(0.0); vocab];
        let mut offset = 0;
        while offset < ids.len() {
            let take = (ids.len() - offset).min(128);
            runner
                .prefill_chunk_with_status(
                    &ids[offset..offset + take],
                    offset,
                    &mut logits,
                    offset + take == ids.len(),
                )
                .expect("prefill");
            offset += take;
        }
        logits
    };
    let to_f32 = |x: &[f16]| -> Vec<f32> { x.iter().map(|v| v.to_f32()).collect() };
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += f64::from(*x) * f64::from(*y);
            na += f64::from(*x).powi(2);
            nb += f64::from(*y).powi(2);
        }
        d / (na.sqrt() * nb.sqrt())
    };

    // Free-running greedy, for the text.
    let started = std::time::Instant::now();
    let mut logits = prefill(&mut runner);
    eprintln!(
        "prefill: {} tokens in {:.0} ms",
        ids.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
    let last = to_f32(&logits);
    assert!(last.iter().all(|v| v.is_finite()), "non-finite logits");
    let greedy_n = manifest["runs"]["hf_mask"]["greedy"]
        .as_array()
        .unwrap()
        .len();
    let mut generated = Vec::new();
    let mut token = argmax(&logits);
    for step in 0..greedy_n {
        generated.push(token);
        runner
            .produce(token, ids.len() + step, &mut logits)
            .expect("decode");
        token = argmax(&logits);
    }
    eprintln!("ours greedy: {generated:?}");

    // Per reference: the last prompt position, then every decode step
    // TEACHER-FORCED with that reference's own tokens. Each step reads the
    // image keys the prefill cached, so this measures the mask 25 times
    // rather than once, and a near-tie cannot derail it.
    let mut summary = Vec::new();
    let runs: Vec<String> = manifest["runs"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    for run in runs.iter().map(String::as_str) {
        let reference = read_f32(dump.join(format!("{run}.logits.f32")));
        let steps = read_f32(dump.join(format!("{run}.steps.f32")));
        // Every run is forced on the same tokens, so the comparisons below
        // are position for position across rules.
        let tokens: Vec<i32> = manifest["forced_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let mut logits = prefill(&mut runner);
        let (kl0, top_ours, top_ref) = compare(&to_f32(&logits), &reference);
        let mut kls = vec![kl0];
        let mut coss = vec![cosine(&to_f32(&logits), &reference)];
        for (i, &t) in tokens.iter().enumerate() {
            runner
                .produce(t, ids.len() + i, &mut logits)
                .expect("decode");
            let want = &steps[i * vocab..(i + 1) * vocab];
            let ours = to_f32(&logits);
            kls.push(compare(&ours, want).0);
            coss.push(cosine(&ours, want));
        }
        let mean_kl = kls.iter().sum::<f64>() / kls.len() as f64;
        let mut sorted = kls.clone();
        sorted.sort_by(f64::total_cmp);
        let median_kl = sorted[sorted.len() / 2];
        let worst = kls
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, k)| (i, *k))
            .unwrap();
        eprintln!(
            "[{run}] median KL {median_kl:.3e}; worst position {} (KL {:.3e}); per position {:?}",
            worst.0,
            worst.1,
            kls.iter().map(|k| format!("{k:.1e}")).collect::<Vec<_>>()
        );
        let max_kl = kls.iter().cloned().fold(0f64, f64::max);
        let min_cos = coss.iter().cloned().fold(1f64, f64::min);
        let free: Vec<i32> = manifest["runs"][run]["greedy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        let agree = generated
            .iter()
            .zip(&free)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "[{run}] last position top5 ours {top_ours:?} ref {top_ref:?}; over {} positions \
             mean KL {mean_kl:.3e}, max KL {max_kl:.3e}, min cosine {min_cos:.6}; free greedy \
             agrees for {agree}/{greedy_n}; ref text {:?}",
            kls.len(),
            manifest["runs"][run]["text"]
        );
        summary.push((run.to_string(), median_kl, top_ours[0] == top_ref[0]));
    }
    // The bar. The decision point's top-1 matches the transformers-rule run,
    // and that run is the CLOSEST of the four mask rules by median KL over
    // the forced positions. The median, not the mean: one near-tie position
    // (where the rules' own greedy paths part) dominates every mean, and
    // mlx-vlm's cached decode is itself only consistent with its one-pass
    // forward to a mean KL of 5e-3 to 2e-2 (docs/GEMMA4_VISION_PHASE0.md).
    let hf = summary
        .iter()
        .find(|r| r.0 == "hf_mask")
        .expect("hf_mask run");
    assert!(
        hf.2,
        "the top-1 next token differs from mlx-vlm with the transformers mask"
    );
    for other in summary.iter().filter(|r| r.0 != "hf_mask") {
        assert!(
            hf.1 < other.1,
            "median KL to the transformers rule {:.3e} is not below {} {:.3e}",
            hf.1,
            other.0,
            other.1
        );
    }
}
