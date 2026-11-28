//! The Idefics3 / SmolVLM tower on a synthetic install: the real repack walk,
//! the real open, the GPU forward pass against an independent CPU one, and the
//! injection into a llama prefill (`docs/IDEFICS3_VISION_PHASE0.md`).
//!
//! The CPU side reads the SOURCE tensors under their published names (separate
//! q, k and v), so a wrong fusion order, tensor mapping, epsilon, activation
//! or shuffle order shows up as a numeric disagreement. It cannot see whether
//! the reference was read correctly: no real weights run here.
#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

use foundation::LogitValue as F16;
use turbospark_repack::{build_synthetic_idefics3_install, SyntheticTowerTensor};
use turbospark_runtime::{LogitProducer, RealForwardRunner};
use turbospark_vision_io::{idefics3_preprocess, Idefics3Params, ImageSpan, Rgb8Image};

const VOCAB: i64 = 256;
const LAYERS: i64 = 2;
const H: usize = 16;
const HEADS: usize = 2;
const MLP: usize = 24;
const SIDE: usize = 4;
const SCALE: usize = 2;
const OUT: usize = 64;

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
                out.push((row[i] - mean) / (var + 1e-6).sqrt() * w[i] + b[i]);
            }
        }
        out
    }

    /// One frame's patch rows to its `[(SIDE / SCALE)^2, OUT]` features.
    fn frame(&self, pixels: &[f32]) -> Vec<f32> {
        let patch_dim = pixels.len() / (SIDE * SIDE);
        let mut x = self.linear(
            pixels,
            patch_dim,
            "vision_model.embeddings.patch_embedding",
            true,
        );
        for (v, p) in x
            .iter_mut()
            .zip(self.get("vision_model.embeddings.position_embedding.weight"))
        {
            *v += p;
        }
        let head = H / HEADS;
        for blk in 0..2 {
            let p = format!("vision_model.encoder.layers.{blk}.");
            let n = self.layer_norm(&x, &format!("{p}layer_norm1"));
            let q = self.linear(&n, H, &format!("{p}self_attn.q_proj"), true);
            let k = self.linear(&n, H, &format!("{p}self_attn.k_proj"), true);
            let v = self.linear(&n, H, &format!("{p}self_attn.v_proj"), true);
            let rows = SIDE * SIDE;
            let mut attn = vec![0.0f32; rows * H];
            for h in 0..HEADS {
                for i in 0..rows {
                    let scores: Vec<f32> = (0..rows)
                        .map(|j| {
                            (0..head)
                                .map(|d| q[i * H + h * head + d] * k[j * H + h * head + d])
                                .sum::<f32>()
                                / (head as f32).sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f32::MIN, f32::max);
                    let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    for (j, e) in exp.iter().enumerate() {
                        for d in 0..head {
                            attn[i * H + h * head + d] += e / sum * v[j * H + h * head + d];
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
            assert_eq!(up.len(), rows * MLP);
            for v in &mut up {
                // The tanh GELU.
                let t = (0.797_884_6 * (*v + 0.044715 * *v * *v * *v)).tanh();
                *v = 0.5 * *v * (1.0 + t);
            }
            let down = self.linear(&up, MLP, &format!("{p}mlp.fc2"), true);
            for (a, b) in x.iter_mut().zip(&down) {
                *a += b;
            }
        }
        let x = self.layer_norm(&x, "vision_model.post_layernorm");
        // The reference's shuffle: block row offset outermost.
        let merged = SIDE / SCALE;
        let mut shuffled = Vec::new();
        for y in 0..merged {
            for z in 0..merged {
                for dy in 0..SCALE {
                    for dx in 0..SCALE {
                        let at = ((y * SCALE + dy) * SIDE + z * SCALE + dx) * H;
                        shuffled.extend_from_slice(&x[at..at + H]);
                    }
                }
            }
        }
        self.linear(
            &shuffled,
            H * SCALE * SCALE,
            "connector.modality_projection.proj",
            false,
        )
    }
}

fn image(seed: usize) -> Rgb8Image {
    // 36x36 against longest edge 36 and tile 18: 2x2 tiles plus the global.
    let data = (0..36 * 36 * 3)
        .map(|i| ((i * 37 + seed * 91 + i / 7) % 256) as u8)
        .collect();
    Rgb8Image {
        width: 36,
        height: 36,
        data,
    }
}

#[test]
fn the_tower_matches_a_cpu_forward_pass_and_reaches_the_trunk() {
    let dir = std::env::temp_dir().join(format!("turbospark-idefics3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (_, tensors) =
        build_synthetic_idefics3_install(&dir, VOCAB, LAYERS, "tiny-idefics3").expect("builds");
    let tower = Tower(tensors.into_iter().map(|t| (t.name.clone(), t)).collect());
    let arch = turbospark_repack::peek_manifest_arch(&dir).expect("peeks");
    assert_eq!(arch.vision.tower, model_io::VisionTowerKind::SigLipNaViT);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("opens");
    assert_eq!(runner.idefics3_image_ids(), Some(200));

    let params = Idefics3Params {
        longest_edge: 36,
        tile: 18,
        split: true,
        mean: [0.5; 3],
        std: [0.5; 3],
    };
    let tiles = idefics3_preprocess(&image(1), 4, &params).expect("preprocesses");
    assert_eq!(tiles.frames, 5);
    let got = runner.encode_image_idefics3(&tiles).expect("encodes");
    let per_frame = (SIDE / SCALE) * (SIDE / SCALE) * OUT;
    assert_eq!(got.len(), tiles.frames * per_frame);

    let patch_len = tiles.pixels.len() / tiles.frames;
    let want: Vec<f32> = tiles
        .pixels
        .chunks_exact(patch_len)
        .flat_map(|frame| tower.frame(frame))
        .collect();
    assert_eq!(want.len(), got.len());
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        let g = half::f16::from_bits(*g).to_f32();
        assert!(g.is_finite());
        dot += f64::from(g) * f64::from(*w);
        aa += f64::from(g) * f64::from(g);
        bb += f64::from(*w) * f64::from(*w);
        worst = worst.max((g - w).abs());
    }
    let cosine = dot / (aa.sqrt() * bb.sqrt());
    let rms = (bb / want.len() as f64).sqrt() as f32;
    println!("cosine {cosine}, worst abs {worst}, rms {rms}");
    assert!(cosine > 0.9999, "cosine {cosine}");
    assert!(worst < 0.05 * rms, "worst {worst} against rms {rms}");

    // Injection: the rows replace the placeholder embeddings of a llama
    // prefill, one span per frame.
    let token = |rows: &[u16], runner: &mut RealForwardRunner| -> Vec<u16> {
        let tokens = per_frame / OUT;
        let mut ids = vec![3i32];
        let mut spans = Vec::new();
        for _ in 0..tiles.frames {
            ids.push(7);
            spans.push(ImageSpan {
                start: ids.len(),
                len: tokens,
            });
            ids.extend(std::iter::repeat_n(200, tokens));
        }
        let frames: Vec<Vec<u16>> = rows.chunks_exact(per_frame).map(<[u16]>::to_vec).collect();
        runner
            .set_prompt_vision_flat(&frames, &spans, ids.len())
            .expect("the map validates");
        let mut head = vec![F16::from_f32(0.0); VOCAB as usize];
        for (position, &id) in ids.iter().enumerate() {
            runner.produce(id, position, &mut head).expect("produces");
        }
        runner.clear_prompt_vision();
        runner.reset();
        head.into_iter().map(|v| v.to_bits()).collect()
    };
    let other = idefics3_preprocess(&image(2), 4, &params).expect("preprocesses");
    let other_rows = runner.encode_image_idefics3(&other).expect("encodes");
    let first = token(&got, &mut runner);
    let second = token(&other_rows, &mut runner);
    assert_ne!(first, second, "the image rows must reach the trunk");
    let _ = std::fs::remove_dir_all(&dir);
}
