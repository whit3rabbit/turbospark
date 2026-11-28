//! The `gemma4_unified` embedder on a synthetic install: the real config
//! parsers and repack walk, the real open, the GPU embedder against an
//! independent CPU one, and the rows through Gemma 4's span prefill
//! (`docs/GEMMA4_UNIFIED_VISION_PHASE0.md`).
//!
//! The CPU side reads the tensors under their published names and cuts the
//! 48 px patches from the image itself, so a wrong table column, patch order,
//! epsilon or normalization step shows as a numeric disagreement. It cannot
//! see whether the reference was read correctly: no real weights run here.
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{build_synthetic_gemma4_unified_install, SyntheticTowerTensor};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::{gemma4_expand_image_placeholders, Gemma4Patches, ImageSpan};

const VOCAB: i64 = 128;
const EMBED: usize = 128;
const TEXT: usize = 64;
const PATCH: usize = 48 * 48 * 3;
/// Two model patches across, three down: x and y cannot be confused.
const WIDTH: usize = 96;
const HEIGHT: usize = 144;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn open() -> (RealForwardRunner, BTreeMap<String, SyntheticTowerTensor>) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir: PathBuf = std::env::temp_dir().join(format!(
        "turbospark-gemma4-unified-{}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (arch, tower) = build_synthetic_gemma4_unified_install(&dir, VOCAB, 6, 8, "tiny-unified")
        .expect("synthetic install builds");
    let runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16).expect("install opens");
    let tower = tower.into_iter().map(|t| (t.name.clone(), t)).collect();
    (runner, tower)
}

/// Patch 0 is flat but for a few samples, so its variance is near zero and the
/// LayerNorm epsilon decides its scale. The rest is textured.
fn image(seed: usize) -> Vec<u8> {
    let mut rgb: Vec<u8> = (0..WIDTH * HEIGHT * 3)
        .map(|i| ((i * 7 + i / 31 + seed * 53) % 251) as u8)
        .collect();
    for r in 0..48 {
        for c in 0..48 * 3 {
            rgb[r * WIDTH * 3 + c] = 128;
        }
    }
    rgb[0] = 129;
    rgb
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let d = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(d) {
        let mean = (row.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64) as f32;
        let var = (row
            .iter()
            .map(|&v| f64::from((v - mean) * (v - mean)))
            .sum::<f64>()
            / d as f64) as f32;
        out.extend((0..d).map(|i| (row[i] - mean) / (var + 1e-5).sqrt() * w[i] + b[i]));
    }
    out
}

fn linear(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
    x.chunks_exact(k)
        .flat_map(|row| {
            w.chunks_exact(k)
                .map(|col| row.iter().zip(col).map(|(a, b)| a * b).sum::<f32>())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn reference(rgb: &[u8], tower: &BTreeMap<String, SyntheticTowerTensor>) -> Vec<f32> {
    let get = |name: &str| tower[name].values.as_slice();
    let ours = |name: &str| get(&format!("vision_embedder.{name}"));
    let mut pixels = Vec::new();
    let mut positions = Vec::new();
    for y in 0..HEIGHT / 48 {
        for x in 0..WIDTH / 48 {
            for r in 0..48 {
                let at = ((y * 48 + r) * WIDTH + x * 48) * 3;
                // The tower takes FP16 pixels in [0, 1], uncentered.
                pixels.extend(
                    rgb[at..at + 48 * 3]
                        .iter()
                        .map(|&v| f16::from_f32(f32::from(v) / 255.0).to_f32()),
                );
            }
            positions.push((x, y));
        }
    }
    assert_eq!(pixels.len(), positions.len() * PATCH);
    let h = layer_norm(&pixels, ours("patch_ln1.weight"), ours("patch_ln1.bias"));
    let mut h = linear(&h, ours("patch_dense.weight"), PATCH);
    for (i, v) in h.iter_mut().enumerate() {
        *v += ours("patch_dense.bias")[i % EMBED];
    }
    let mut h = layer_norm(&h, ours("patch_ln2.weight"), ours("patch_ln2.bias"));
    // `[positions, 2, EMBED]`.
    let table = ours("pos_embedding");
    for (row, &(x, y)) in h.chunks_exact_mut(EMBED).zip(&positions) {
        for (c, v) in row.iter_mut().enumerate() {
            *v += table[x * 2 * EMBED + c] + table[(y * 2 + 1) * EMBED + c];
        }
    }
    let mut h = layer_norm(&h, ours("pos_norm.weight"), ours("pos_norm.bias"));
    for row in h.chunks_exact_mut(EMBED) {
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / EMBED as f32 + 1e-6).sqrt();
        for v in row {
            *v /= rms;
        }
    }
    linear(&h, get("embed_vision.embedding_projection.weight"), EMBED)
}

fn encode(runner: &mut RealForwardRunner, rgb: &[u8]) -> Vec<u16> {
    let patches = Gemma4Patches::from_rgb(rgb, WIDTH, HEIGHT).expect("patchifies");
    assert_eq!(patches.soft_tokens(), 6);
    runner.encode_image_gemma4(&patches).expect("encodes")
}

#[test]
fn the_gpu_embedder_matches_the_cpu_reference() {
    let (mut runner, tower) = open();
    let rgb = image(0);
    let got: Vec<f32> = encode(&mut runner, &rgb)
        .into_iter()
        .map(|b| f16::from_bits(b).to_f32())
        .collect();
    let want = reference(&rgb, &tower);
    assert_eq!(got.len(), 6 * TEXT);
    assert_eq!(want.len(), got.len());
    let peak = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(peak > 0.1, "degenerate reference");
    for (row, (g, w)) in got.chunks(TEXT).zip(want.chunks(TEXT)).enumerate() {
        let worst = g
            .iter()
            .zip(w)
            .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(
            worst < 0.01 * peak,
            "soft token {row} differs by {worst} against a peak of {peak}"
        );
    }
    // A released embedder reopens to the same rows.
    runner.release_vision_tower();
    let again = encode(&mut runner, &rgb);
    assert_eq!(again.len(), got.len());
    assert_eq!(f16::from_bits(again[0]).to_f32(), got[0]);
}

/// `[boi] image x 6 [eoi]` between text, from the install's own ids.
#[test]
fn an_encoded_image_reaches_the_logits_through_the_span_prefill() {
    let (mut runner, _) = open();
    let first = encode(&mut runner, &image(0));
    let second = encode(&mut runner, &image(1));
    assert_ne!(first, second, "the embedder ignored the pixels");
    let ids = runner.gemma4_image_ids().expect("a gemma4-trunk tower");
    assert_eq!((ids.boi, ids.image, ids.eoi), (1, 0, 2));
    let rendered = [5, 9, ids.image, 7, 3];
    let (tokens, spans) = gemma4_expand_image_placeholders(&rendered, ids, &[6]).unwrap();
    assert_eq!(spans, vec![ImageSpan { start: 3, len: 6 }]);

    let mut run = |rows: &[u16], chunk: usize| {
        runner.reset();
        runner
            .set_prompt_vision_gemma4(&[rows.to_vec()], &spans, tokens.len())
            .expect("rows set");
        let mut logits = vec![f16::from_f32(0.0); VOCAB as usize];
        for (i, part) in tokens.chunks(chunk).enumerate() {
            let at = i * chunk;
            runner
                .prefill_chunk_with_status(part, at, &mut logits, at + part.len() == tokens.len())
                .expect("chunk succeeds");
        }
        logits.iter().map(|v| v.to_f32()).collect::<Vec<f32>>()
    };
    let mut other = first.clone();
    other[2 * TEXT + 3] = f16::from_f32(3.0).to_bits();
    let whole = run(&first, tokens.len());
    assert!(whole.iter().all(|v| v.is_finite()));
    assert_eq!(whole, run(&first, 2), "the chunk boundary moved the logits");
    assert_ne!(whole, run(&other, 2), "one row did not reach the logits");
    assert_ne!(whole, run(&second, 2), "a second image answered the same");
}

/// The sliding ring is sized for a default image on this tower too: a span
/// past one prefill chunk (128) must be accepted up to the 280 token budget.
#[test]
fn the_kv_ring_holds_a_default_size_image_span() {
    let (mut runner, _) = open();
    let span = ImageSpan { start: 1, len: 280 };
    let rows = vec![0u16; 280 * TEXT];
    runner
        .set_prompt_vision_gemma4(&[rows], &[span], 282)
        .expect("a 280-token span fits");
}
