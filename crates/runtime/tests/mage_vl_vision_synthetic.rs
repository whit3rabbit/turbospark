//! The Mage-ViT tower of `mage_vl` on a synthetic install: the real config
//! parsers and repack walk, the real open, the GPU tower against an
//! independent CPU one, and its rows through a qwen3 trunk with 1-D
//! positions (`docs/MAGE_VL_VISION_PHASE0.md`).
//!
//! The CPU side reads the tensors under their published names and does its
//! own normalization, patch cutting, rope and attention, so a wrong patch
//! order, rope pairing, activation, epsilon or a skipped `layernorm_pre`
//! shows as a numeric disagreement. It cannot see whether the reference was
//! read correctly: no real weights run here.
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use turbospark_repack::{build_synthetic_mage_vl_install, SyntheticTowerTensor};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::{
    preprocess, splice_and_walk, PreprocessParams, PreprocessedImage, Rgb8Image, VisionSpecialIds,
};

const H: usize = 64;
const HEADS: usize = 2;
const HEAD: usize = 32;
const TEXT: usize = 64;
/// Six patches across, four down, so x and y cannot be confused. Both are
/// multiples of 32 inside the budget, so nothing is resized.
const WIDTH: usize = 96;
const HEIGHT: usize = 64;
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

static COUNTER: AtomicU64 = AtomicU64::new(0);

type Tower = BTreeMap<String, SyntheticTowerTensor>;

fn open() -> (RealForwardRunner, Tower, PreprocessParams) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir: PathBuf =
        std::env::temp_dir().join(format!("turbospark-mage-vl-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (arch, tower) = build_synthetic_mage_vl_install(&dir, None).expect("builds");
    let json = std::fs::read_to_string(dir.join("preprocessor_config.json")).unwrap();
    let params = PreprocessParams::from_preprocessor_config_json(&json).unwrap();
    let runner = RealForwardRunner::open_with_options(&dir, arch, 4096, 16).expect("opens");
    let tower = tower.into_iter().map(|t| (t.name.clone(), t)).collect();
    (runner, tower, params)
}

fn rgb(seed: usize) -> Vec<u8> {
    (0..WIDTH * HEIGHT * 3)
        .map(|i| ((i * 7 + i / 31 + (i / 288) * 11 + seed * 97) % 251) as u8)
        .collect()
}

fn prepare(data: Vec<u8>, params: &PreprocessParams) -> PreprocessedImage {
    let image = Rgb8Image {
        data,
        width: WIDTH,
        height: HEIGHT,
    };
    preprocess(&image, params).expect("preprocesses")
}

/// What an FP16 buffer keeps.
fn stored(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|v| f16::from_f32(v).to_f32()).collect()
}

fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, k: usize) -> Vec<f32> {
    let n = w.len() / k;
    stored(
        x.chunks_exact(k)
            .flat_map(|row| {
                (0..n).map(move |o| {
                    b.map_or(0.0, |b| b[o])
                        + row.iter().zip(&w[o * k..]).map(|(a, w)| a * w).sum::<f32>()
                })
            })
            .collect(),
    )
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let d = w.len();
    stored(
        x.chunks_exact(d)
            .flat_map(|row| {
                let mean = row.iter().sum::<f32>() / d as f32;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                (0..d).map(move |i| (row[i] - mean) / (var + 1e-6).sqrt() * w[i] + b[i])
            })
            .collect(),
    )
}

/// The exact GELU, `erf` by its Maclaurin series.
fn gelu(x: Vec<f32>) -> Vec<f32> {
    stored(
        x.into_iter()
            .map(|v| {
                let z = (f64::from(v) / std::f64::consts::SQRT_2).clamp(-5.0, 5.0);
                let (mut term, mut sum) = (z, z);
                for n in 1..120 {
                    term *= -z * z / f64::from(n);
                    sum += term / f64::from(2 * n + 1);
                }
                let erf = (sum * std::f64::consts::FRAC_2_SQRT_PI).clamp(-1.0, 1.0);
                (f64::from(v) * 0.5 * (1.0 + erf)) as f32
            })
            .collect(),
    )
}

fn reference(rgb: &[u8], tower: &Tower) -> Vec<f32> {
    let get = |name: &str| tower[&format!("vision_tower.{name}")].values.as_slice();
    // Patches in 2x2 merge-window order, each `(row, column, channel)`.
    let mut pixels = Vec::new();
    let mut positions = Vec::new();
    for wy in 0..HEIGHT / 32 {
        for wx in 0..WIDTH / 32 {
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (row, col) = (2 * wy + dy, 2 * wx + dx);
                for y in 0..16 {
                    for x in 0..16 {
                        let at = ((row * 16 + y) * WIDTH + col * 16 + x) * 3;
                        pixels.extend(
                            (0..3).map(|c| (f32::from(rgb[at + c]) / 255.0 - MEAN[c]) / STD[c]),
                        );
                    }
                }
                positions.push((row as f32, col as f32));
            }
        }
    }
    let seq = positions.len();
    let patches = linear(
        &stored(pixels),
        get("embeddings.patch_embedding.weight"),
        None,
        768,
    );
    let mut x = layer_norm(
        &patches,
        get("layernorm_pre.weight"),
        get("layernorm_pre.bias"),
    );
    // Half a head: 4 zeros for time (frame 0), 6 for the row, 6 for the column.
    let angles: Vec<[f32; HEAD / 2]> = positions
        .iter()
        .map(|&(row, col)| {
            let mut a = [0.0f32; HEAD / 2];
            for j in 0..6 {
                let inv = 1.0 / 10_000f32.powf(j as f32 / 6.0);
                a[4 + j] = row * inv;
                a[10 + j] = col * inv;
            }
            a
        })
        .collect();
    let rotate = |t: Vec<f32>| -> Vec<f32> {
        let mut out = t.clone();
        for (i, v) in out.iter_mut().enumerate() {
            let (token, lane) = (i / H, i % HEAD);
            let turned = if lane % 2 == 0 { -t[i + 1] } else { t[i - 1] };
            let angle = angles[token][lane % (HEAD / 2)];
            *v = t[i] * angle.cos() + turned * angle.sin();
        }
        stored(out)
    };
    for block in 0..2 {
        let at = |name: &str| get(&format!("encoder.layers.{block}.{name}"));
        let normed = layer_norm(&x, at("layer_norm1.weight"), at("layer_norm1.bias"));
        let part = |i: usize| {
            linear(
                &normed,
                &at("self_attn.qkv.weight")[i * H * H..][..H * H],
                Some(&at("self_attn.qkv.bias")[i * H..][..H]),
                H,
            )
        };
        let (q, k, v) = (rotate(part(0)), rotate(part(1)), part(2));
        let mut attn = vec![0f32; seq * H];
        for head in 0..HEADS {
            let off = |s: usize| s * H + head * HEAD;
            for i in 0..seq {
                let scores: Vec<f32> = (0..seq)
                    .map(|j| {
                        (0..HEAD)
                            .map(|e| q[off(i) + e] * k[off(j) + e])
                            .sum::<f32>()
                            / (HEAD as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::MIN, f32::max);
                let total: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                for (j, score) in scores.iter().enumerate() {
                    let p = (score - max).exp() / total;
                    for e in 0..HEAD {
                        attn[off(i) + e] += p * v[off(j) + e];
                    }
                }
            }
        }
        let proj = linear(
            &stored(attn),
            at("self_attn.proj.weight"),
            Some(at("self_attn.proj.bias")),
            H,
        );
        x = stored(x.iter().zip(&proj).map(|(a, b)| a + b).collect());
        let normed = layer_norm(&x, at("layer_norm2.weight"), at("layer_norm2.bias"));
        let mid = gelu(linear(
            &normed,
            at("mlp.fc1.weight"),
            Some(at("mlp.fc1.bias")),
            H,
        ));
        let down = linear(
            &mid,
            at("mlp.fc2.weight"),
            Some(at("mlp.fc2.bias")),
            at("mlp.fc1.bias").len(),
        );
        x = stored(x.iter().zip(&down).map(|(a, b)| a + b).collect());
    }
    let normed = layer_norm(&x, get("merger.ln_q.weight"), get("merger.ln_q.bias"));
    let mid = gelu(linear(
        &normed,
        get("merger.mlp.0.weight"),
        Some(get("merger.mlp.0.bias")),
        4 * H,
    ));
    linear(
        &mid,
        get("merger.mlp.2.weight"),
        Some(get("merger.mlp.2.bias")),
        4 * H,
    )
}

#[test]
fn the_gpu_tower_matches_the_cpu_reference() {
    let (mut runner, tower, params) = open();
    let data = rgb(0);
    let image = prepare(data.clone(), &params);
    assert_eq!(image.resized, (HEIGHT, WIDTH));
    let got = runner.encode_image(&image, &params).expect("encodes");
    assert_eq!((got.merged_tokens, got.out_hidden), (6, TEXT));
    let got: Vec<f32> = got
        .rows
        .iter()
        .map(|&b| f16::from_bits(b).to_f32())
        .collect();
    let want = reference(&data, &tower);
    assert_eq!(want.len(), got.len());
    let peak = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(peak > 0.1, "degenerate reference");
    let worst = got
        .iter()
        .zip(&want)
        .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    eprintln!("worst {worst} against peak {peak}");
    // Correct reads about 0.0016 of the peak; a quick-GELU block MLP 0.0065.
    assert!(worst < 0.003 * peak, "{worst} against a peak of {peak}");
}

/// The front ends hand over the mRoPE walk. This trunk has 1-D positions, so
/// the result must equal a plain flat injection, including the text after the
/// image, and the rows must reach the logits.
#[test]
fn the_rows_reach_a_qwen3_trunk_at_sequential_positions() {
    let (mut runner, _, params) = open();
    let image = prepare(rgb(0), &params);
    let first = runner.encode_image(&image, &params).expect("encodes");
    let second = runner
        .encode_image(&prepare(rgb(1), &params), &params)
        .expect("encodes");
    assert_ne!(first.rows, second.rows, "the tower ignored the pixels");

    let special = VisionSpecialIds {
        vision_start: 100,
        image_pad: 102,
    };
    let walked = splice_and_walk(&[5, 9, 100, 102, 101, 7, 3], &[image.grid], special, 2).unwrap();
    assert_eq!(walked.ids.len(), 12);
    // Without this the two injections below could not differ.
    assert_ne!(walked.positions.rope_delta, 0);

    let mut run = |flat: bool, other: bool, chunk: usize| {
        let e = if other { &second } else { &first };
        runner.reset();
        if flat {
            runner.set_prompt_vision_flat(
                std::slice::from_ref(&e.rows),
                &walked.positions.spans,
                walked.ids.len(),
            )
        } else {
            runner.set_prompt_vision(std::slice::from_ref(e), &walked.positions, walked.ids.len())
        }
        .expect("rows set");
        let mut logits = vec![f16::from_f32(0.0); 128];
        for (i, part) in walked.ids.chunks(chunk).enumerate() {
            let at = i * chunk;
            runner
                .prefill_chunk_with_status(
                    part,
                    at,
                    &mut logits,
                    at + part.len() == walked.ids.len(),
                )
                .expect("chunk succeeds");
        }
        logits.iter().map(|v| v.to_f32()).collect::<Vec<f32>>()
    };
    let walk = run(false, false, walked.ids.len());
    assert!(walk.iter().all(|v| v.is_finite()));
    assert_eq!(walk, run(true, false, walked.ids.len()));
    assert_eq!(walk, run(false, false, 4), "chunking moved the logits");
    assert_ne!(walk, run(false, true, 4), "pixels did not reach the logits");
}
