//! LLaVA 1.6 on a synthetic checkpoint: the real config parse, reader and
//! sidecar writer, the real attach onto a llama trunk, the GPU tower and
//! packing against an independent CPU computation, and the injection into the
//! batched prefill (`docs/LLAVA_NEXT_VISION_PHASE0.md`).
//!
//! The CPU side reads the SOURCE tensors under their published names (separate
//! q, k and v) and packs the run with its own hand-written geometry, so a wrong
//! fusion order, tensor mapping, epsilon, activation, tile order, unpad or
//! newline shows up as a numeric disagreement. It cannot see whether the
//! reference was read correctly, and it takes the preprocessed pixels as given:
//! no real weights run here.
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use foundation::LogitValue as F16;
use turbospark_repack::{SyntheticTowerTensor, SYNTHETIC_NEXT_HEADS, SYNTHETIC_NEXT_HIDDEN};
use turbospark_runtime::{ChunkedPrefillRunner, LogitProducer, RealForwardRunner};
use turbospark_vision_io::{
    clip_expand_image_placeholders, clip_preprocess, ClipPreprocessParams, Rgb8Image,
};

const VOCAB: i64 = 256;
/// The synthetic llama trunk's width.
const OUT: usize = 64;
const H: usize = SYNTHETIC_NEXT_HIDDEN;

struct Tower(BTreeMap<String, SyntheticTowerTensor>);

impl Tower {
    fn get(&self, name: &str) -> &[f32] {
        &self.0[name].values
    }

    /// `rows x [k] @ w[n, k]^T + bias`.
    fn linear(&self, x: &[f32], k: usize, name: &str, bias: bool) -> Vec<f32> {
        let w = self.get(&format!("{name}.weight"));
        let n = w.len() / k;
        let b = bias.then(|| self.get(&format!("{name}.bias")));
        let mut out = Vec::new();
        for row in x.chunks_exact(k) {
            for o in 0..n {
                let dot: f32 = row
                    .iter()
                    .zip(&w[o * k..(o + 1) * k])
                    .map(|(a, b)| a * b)
                    .sum();
                out.push(dot + b.map_or(0.0, |b| b[o]));
            }
        }
        out
    }

    fn layer_norm(&self, x: &[f32], name: &str) -> Vec<f32> {
        let w = self.get(&format!("{name}.weight"));
        let b = self.get(&format!("{name}.bias"));
        let mut out = Vec::new();
        for row in x.chunks_exact(H) {
            let mean: f32 = row.iter().sum::<f32>() / H as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / H as f32;
            for i in 0..H {
                out.push((row[i] - mean) / (var + 1e-5).sqrt() * w[i] + b[i]);
            }
        }
        out
    }

    /// One tile's 576 patch rows to its `[576, OUT]` projected rows.
    fn tile(&self, pixels: &[f32]) -> Vec<f32> {
        let base = "vision_tower.vision_model.";
        let patches = self.linear(
            pixels,
            588,
            &format!("{base}embeddings.patch_embedding"),
            false,
        );
        let mut x = self
            .get(&format!("{base}embeddings.class_embedding"))
            .to_vec();
        x.extend(patches);
        let table = self.get(&format!("{base}embeddings.position_embedding.weight"));
        assert_eq!(x.len(), table.len());
        for (v, p) in x.iter_mut().zip(table) {
            *v += p;
        }
        let mut x = self.layer_norm(&x, &format!("{base}pre_layrnorm"));
        let head = H / SYNTHETIC_NEXT_HEADS;
        let rows = x.len() / H;
        // Layer 2 is in the file and must not run.
        for layer in 0..2 {
            let p = format!("{base}encoder.layers.{layer}.");
            let n = self.layer_norm(&x, &format!("{p}layer_norm1"));
            let q = self.linear(&n, H, &format!("{p}self_attn.q_proj"), true);
            let k = self.linear(&n, H, &format!("{p}self_attn.k_proj"), true);
            let v = self.linear(&n, H, &format!("{p}self_attn.v_proj"), true);
            let mut attn = vec![0.0f32; rows * H];
            let mut scores = vec![0.0f32; rows];
            for h in 0..SYNTHETIC_NEXT_HEADS {
                let at = h * head;
                for i in 0..rows {
                    let qi = &q[i * H + at..i * H + at + head];
                    for (j, s) in scores.iter_mut().enumerate() {
                        let kj = &k[j * H + at..j * H + at + head];
                        *s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>()
                            / (head as f32).sqrt();
                    }
                    let max = scores.iter().copied().fold(f32::MIN, f32::max);
                    let mut sum = 0.0;
                    for s in &mut scores {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for (j, e) in scores.iter().enumerate() {
                        for d in 0..head {
                            attn[i * H + at + d] += e / sum * v[j * H + at + d];
                        }
                    }
                }
            }
            let o = self.linear(&attn, H, &format!("{p}self_attn.out_proj"), true);
            for (a, b) in x.iter_mut().zip(&o) {
                *a += b;
            }
            let n = self.layer_norm(&x, &format!("{p}layer_norm2"));
            let mut up = self.linear(&n, H, &format!("{p}mlp.fc1"), true);
            for v in &mut up {
                // quick_gelu.
                *v /= 1.0 + (-1.702 * *v).exp();
            }
            let inter = up.len() / rows;
            let down = self.linear(&up, inter, &format!("{p}mlp.fc2"), true);
            for (a, b) in x.iter_mut().zip(&down) {
                *a += b;
            }
        }
        // The class row is dropped; the projector's GELU is the exact one.
        let mut mid = self.linear(&x[H..], H, "multi_modal_projector.linear_1", true);
        for v in &mut mid {
            *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
        }
        self.linear(&mid, OUT, "multi_modal_projector.linear_2", true)
    }
}

/// Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - poly * (-x * x).exp()).copysign(x)
}

fn image(seed: usize) -> Rgb8Image {
    // 448 wide by 336 high: a 1 x 2 grid whose map is cropped on both sides.
    let (width, height) = (448, 336);
    let data = (0..width * height * 3)
        .map(|i| ((i / 3 / 37 * 41 + i % 3 * 83 + seed * 91 + i / 4001 * 29) % 256) as u8)
        .collect();
    Rgb8Image {
        width,
        height,
        data,
    }
}

struct Setup {
    root: PathBuf,
    tower: Tower,
    runner: RealForwardRunner,
}

fn setup() -> Setup {
    let root = std::env::temp_dir().join(format!(
        "turbospark-llava-next-runtime-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let (trunk, sidecar) = (root.join("trunk"), root.join("tower"));
    std::fs::create_dir_all(&trunk).expect("trunk dir");
    std::fs::create_dir_all(&sidecar).expect("sidecar dir");
    // The parser, reader and writer the catalog uses.
    let tensors =
        turbospark_repack::build_synthetic_llava_next_sidecar(&sidecar, OUT, 200).expect("sidecar");
    turbospark_repack::build_synthetic_dense_llama_install(&trunk, VOCAB, 2, "tiny-llama")
        .expect("trunk");
    let arch = turbospark_repack::peek_manifest_arch(&trunk).expect("peeks");
    let mut runner = RealForwardRunner::open(&trunk, arch).expect("opens");
    assert!(!runner.is_llava_next_vision(), "a text trunk has no tower");
    runner.attach_vision_sidecar(&sidecar).expect("attaches");
    Setup {
        root,
        tower: Tower(tensors.into_iter().map(|t| (t.name.clone(), t)).collect()),
        runner,
    }
}

#[test]
fn the_tower_and_packing_match_the_cpu_and_reach_the_batched_prefill() {
    let Setup {
        root,
        tower,
        mut runner,
    } = setup();
    assert!(runner.is_llava_next_vision());
    assert_eq!(runner.clip_image_ids(), Some((200, 200)));
    let json = std::fs::read_to_string(runner.vision_dir().join("preprocessor_config.json"))
        .expect("processor config");
    let params = ClipPreprocessParams::from_preprocessor_config_json(&json).expect("params");
    let v = runner.vision_config();
    params
        .check_against_tower(v.patch_size, v.num_position_embeddings)
        .expect("agrees with the tower");

    let patches = clip_preprocess(&image(1), &params).expect("preprocesses");
    // 576 base rows, a 24 x 32 map and 24 newlines.
    let tokens = 576 + 24 * 32 + 24;
    assert_eq!(patches.num_patches(), tokens);
    assert_eq!(params.grid_for(448, 336), (tokens, 1));
    assert_eq!(patches.pixels.len(), 3 * 576 * 588);
    let got: Vec<f32> = runner
        .encode_image_clip(&patches)
        .expect("encodes")
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    assert_eq!(got.len(), tokens * OUT);

    // Independent packing: base, then the 24 x 48 grid map with its columns
    // 8..40 kept and a newline after each row.
    let tiles: Vec<Vec<f32>> = patches
        .pixels
        .chunks_exact(576 * 588)
        .map(|t| tower.tile(t))
        .collect();
    let newline = tower.get("image_newline");
    let mut want = tiles[0].clone();
    for y in 0..24 {
        for x in 8..40 {
            let row = y * 24 + x % 24;
            want.extend_from_slice(&tiles[1 + x / 24][row * OUT..(row + 1) * OUT]);
        }
        want.extend_from_slice(newline);
    }
    assert_eq!(want.len(), got.len());
    let rms = (want.iter().map(|w| w * w).sum::<f32>() / want.len() as f32).sqrt();
    for (row, (g, w)) in got.chunks(OUT).zip(want.chunks(OUT)).enumerate() {
        let worst = g
            .iter()
            .zip(w)
            .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(worst < 0.05 * rms, "row {row}: {worst} against rms {rms}");
    }
    // The fixture must separate rows, or a wrong tile order would still pass.
    let apart = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
    };
    assert!(
        apart(&tiles[1], &tiles[2]) > 0.1 * rms,
        "degenerate tiles: {}",
        apart(&tiles[1], &tiles[2])
    );
    assert!(apart(&tiles[0], &tiles[1]) > 0.1 * rms);

    // The same tower must refuse a LLaVA-1.5 style (untiled) image.
    let mut flat = patches.clone();
    flat.anyres = None;
    assert!(runner.encode_image_clip(&flat).is_err());

    // Injection through the batched prefill.
    let rows: Vec<u16> = got
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    let (ids, spans) =
        clip_expand_image_placeholders(&[3, 7, 200, 5, 9], 200, 200, &[tokens]).expect("expands");
    let run = |runner: &mut RealForwardRunner, rows: &[u16], chunk: usize| -> Vec<f32> {
        runner.reset();
        runner
            .set_prompt_vision_flat(&[rows.to_vec()], &spans, ids.len())
            .expect("rows set");
        let mut logits = vec![F16::from_f32(0.0); VOCAB as usize];
        for (i, part) in ids.chunks(chunk).enumerate() {
            let at = i * chunk;
            runner
                .prefill_chunk_with_status(part, at, &mut logits, at + part.len() == ids.len())
                .expect("chunk succeeds");
        }
        logits.iter().map(|v| v.to_f32()).collect()
    };
    let whole = run(&mut runner, &rows, ids.len());
    assert!(whole.iter().all(|v| v.is_finite()));
    assert_eq!(
        whole,
        run(&mut runner, &rows, 200),
        "the chunk boundary moved the logits"
    );
    let other = runner
        .encode_image_clip(&clip_preprocess(&image(2), &params).expect("preprocesses"))
        .expect("encodes");
    assert_ne!(whole, run(&mut runner, &other, 200), "the image is ignored");
    let _ = std::fs::remove_dir_all(&root);
}

/// LLaVA-1.5 on the same weights: its single-crop path still matches the CPU,
/// returns the 576 patch rows and nothing packed, and refuses anyres patches.
#[test]
fn llava_1_5_still_encodes_one_crop_and_refuses_tiles() {
    let root = std::env::temp_dir().join(format!(
        "turbospark-llava-15-runtime-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let (trunk, sidecar, next) = (root.join("trunk"), root.join("old"), root.join("next"));
    for dir in [&trunk, &sidecar, &next] {
        std::fs::create_dir_all(dir).expect("dir");
    }
    let tensors = turbospark_repack::build_synthetic_llava_15_sidecar(&sidecar, OUT, 200)
        .expect("1.5 sidecar");
    let tower = Tower(tensors.into_iter().map(|t| (t.name.clone(), t)).collect());
    turbospark_repack::build_synthetic_llava_next_sidecar(&next, OUT, 200).expect("1.6 sidecar");
    turbospark_repack::build_synthetic_dense_llama_install(&trunk, VOCAB, 2, "tiny-llama")
        .expect("trunk");
    let arch = turbospark_repack::peek_manifest_arch(&trunk).expect("peeks");
    let mut runner = RealForwardRunner::open(&trunk, arch).expect("opens");
    runner.attach_vision_sidecar(&sidecar).expect("attaches");
    assert!(!runner.is_llava_next_vision());
    assert_eq!(runner.clip_image_ids(), Some((200, 200)));

    let params = |dir: &std::path::Path| {
        let json = std::fs::read_to_string(dir.join("preprocessor_config.json")).expect("config");
        ClipPreprocessParams::from_preprocessor_config_json(&json).expect("params")
    };
    let patches = clip_preprocess(&image(1), &params(&sidecar)).expect("preprocesses");
    assert_eq!(patches.num_patches(), 576);
    assert!(patches.anyres.is_none());
    let got: Vec<f32> = runner
        .encode_image_clip(&patches)
        .expect("encodes")
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    let want = tower.tile(&patches.pixels);
    assert_eq!(got.len(), want.len());
    let rms = (want.iter().map(|w| w * w).sum::<f32>() / want.len() as f32).sqrt();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 0.05 * rms, "{g} against {w}, rms {rms}");
    }

    let tiled = clip_preprocess(&image(1), &params(&next)).expect("preprocesses");
    let err = runner
        .encode_image_clip(&tiled)
        .expect_err("1.5 refuses tiles");
    assert!(err.to_string().contains("LLaVA-1.5"), "{err}");
    let _ = std::fs::remove_dir_all(&root);
}
