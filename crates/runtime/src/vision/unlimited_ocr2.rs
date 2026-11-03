//! Unlimited-OCR-2's tower: SAM ViT-B, then CLIP-L reading SAM's output as its
//! patch embeddings, then a linear projector over both.
//!
//! ```text
//! s  = patch_embed(pixels) + resized(sam.pos_embed)
//! for block: s += attn(norm1(s)); s += mlp(norm2(s))   // eps 1e-6, exact GELU
//!            // attn is windowed (14) with zero padding, global on 2/5/8/11,
//!            // and carries a decomposed relative-position bias
//! f  = net_3(net_2(norm(conv3(norm(conv1(s))))))       // [g, g, 1024]
//! c  = pre_layrnorm([cls ; f] + resized(pos))          // eps 1e-5
//! for block: c += attn(ln1(c)); c += mlp(ln2(c))       // quick_gelu
//! y  = projector([c[1:] ; f])                          // [g*g, 1280]
//! ```
//!
//! Upstream recommends FP32 here for an MXFP4 trunk. This runs FP16, like every
//! tower in this port, and is gated stage by stage against mlx-vlm
//! (`tests/unlimited_ocr2_tower_real.rs`). The weights are `uocr2_tower.bin`,
//! read on the first image and held for the runner's life, and the scratch is
//! per view.

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use turbospark_vision_io::{uocr2_grid, Uocr2Page, UOCR2_BASE_SIZE, UOCR2_CROP_SIZE};

use crate::real_forward_types::RealForwardError;
use crate::real_forward_utils::f16_slice_to_le_bytes;

const WINDOW: usize = 14;
const SAM_DIM: usize = 768;
const SAM_HEADS: usize = 12;
const SAM_DEPTH: usize = 12;
const SAM_MLP: usize = 3072;
const SAM_GLOBAL: [usize; 4] = [2, 5, 8, 11];
const SAM_EPS: f32 = 1e-6;
const NECK: usize = 256;
const CLIP_DIM: usize = 1024;
const CLIP_HEADS: usize = 16;
const CLIP_DEPTH: usize = 24;
const CLIP_MLP: usize = 4096;
const CLIP_EPS: f32 = 1e-5;
const CLIP_SOURCE_GRID: usize = 16;
const OUT_DIM: usize = 1280;

fn gpu_err(e: gpu::GpuError) -> RealForwardError {
    RealForwardError::Gpu(e)
}

fn unsupported(detail: String) -> RealForwardError {
    RealForwardError::Unsupported(detail)
}

/// Intermediate rows of a view, for the parity gate.
#[derive(Debug, Default, Clone)]
pub struct Uocr2Stages {
    pub patch: Vec<f16>,
    pub sam: Vec<f16>,
    pub clip: Vec<f16>,
    pub projected: Vec<f16>,
}

struct TableSet {
    sam_pos: gpu::MetalBuffer,
    clip_pos: gpu::MetalBuffer,
    /// `(rel_h, rel_w)` per block, at that block's side.
    rel: Vec<(gpu::MetalBuffer, gpu::MetalBuffer)>,
}

pub(crate) struct Uocr2Tower {
    weights: gpu::MetalBuffer,
    bytes: Vec<u16>,
    index: HashMap<String, (u64, Vec<u64>)>,
    tables: HashMap<usize, TableSet>,
}

impl Uocr2Tower {
    pub(crate) fn open(context: &gpu::MetalContext, dir: &Path) -> Result<Self, RealForwardError> {
        let read_err = |what: &str, e: String| unsupported(format!("reading {what}: {e}"));
        let index_bytes = std::fs::read(dir.join("uocr2_tower.json"))
            .map_err(|e| read_err("uocr2_tower.json", e.to_string()))?;
        let root: serde_json::Value = serde_json::from_slice(&index_bytes)
            .map_err(|e| read_err("uocr2_tower.json", e.to_string()))?;
        let mut index = HashMap::new();
        for (name, entry) in root.as_object().into_iter().flatten() {
            let offset = entry["offset"].as_u64();
            let shape: Option<Vec<u64>> = entry["shape"]
                .as_array()
                .and_then(|s| s.iter().map(serde_json::Value::as_u64).collect());
            match (offset, shape) {
                (Some(offset), Some(shape)) => {
                    index.insert(name.clone(), (offset, shape));
                }
                _ => return Err(read_err("uocr2_tower.json", format!("bad entry {name}"))),
            }
        }
        let raw = std::fs::read(dir.join("uocr2_tower.bin"))
            .map_err(|e| read_err("uocr2_tower.bin", e.to_string()))?;
        let weights = context.new_buffer_with_data(&raw);
        let bytes = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Self {
            weights,
            bytes,
            index,
            tables: HashMap::new(),
        })
    }

    fn entry(&self, name: &str) -> Result<&(u64, Vec<u64>), RealForwardError> {
        self.index
            .get(name)
            .ok_or_else(|| RealForwardError::MissingTensor(format!("uocr2_tower/{name}")))
    }

    fn view(&self, name: &str) -> Result<(&gpu::MetalBuffer, u64), RealForwardError> {
        Ok((&self.weights, self.entry(name)?.0))
    }

    fn at(
        &self,
        name: &str,
        byte_offset: u64,
    ) -> Result<(&gpu::MetalBuffer, u64), RealForwardError> {
        Ok((&self.weights, self.entry(name)?.0 + byte_offset))
    }

    fn host(&self, name: &str) -> Result<Vec<f32>, RealForwardError> {
        let (offset, shape) = self.entry(name)?;
        let n: u64 = shape.iter().product();
        let start = (*offset / 2) as usize;
        Ok(self.bytes[start..start + n as usize]
            .iter()
            .map(|b| f16::from_bits(*b).to_f32())
            .collect())
    }

    fn upload(context: &gpu::MetalContext, v: &[f32]) -> gpu::MetalBuffer {
        let rows: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
        context.new_buffer_with_data(&f16_slice_to_le_bytes(&rows))
    }

    /// The tables a view of `grid` patches per side needs, built once per size.
    fn ensure_tables(
        &mut self,
        context: &gpu::MetalContext,
        grid: usize,
    ) -> Result<(), RealForwardError> {
        use compute::uocr2_tower::{bicubic_resize_antialias, rel_pos_table};
        if self.tables.contains_key(&grid) {
            return Ok(());
        }
        let source = self.host("sam.pos_embed")?;
        let sam_pos = bicubic_resize_antialias(&source, 64, SAM_DIM, grid);
        let clip = self.host("clip.embeddings.position_embedding.weight")?;
        let clip_grid = grid / 4;
        let mut clip_pos = clip[..CLIP_DIM].to_vec();
        clip_pos.extend(bicubic_resize_antialias(
            &clip[CLIP_DIM..],
            CLIP_SOURCE_GRID,
            CLIP_DIM,
            clip_grid,
        ));
        let head_dim = SAM_DIM / SAM_HEADS;
        let mut rel = Vec::with_capacity(SAM_DEPTH);
        for block in 0..SAM_DEPTH {
            let side = if SAM_GLOBAL.contains(&block) {
                grid
            } else {
                WINDOW
            };
            let table = |axis: &str| -> Result<gpu::MetalBuffer, RealForwardError> {
                let values = self.host(&format!("sam.blocks.{block}.attn.rel_pos_{axis}"))?;
                Ok(Self::upload(
                    context,
                    &rel_pos_table(&values, head_dim, side),
                ))
            };
            rel.push((table("h")?, table("w")?));
        }
        let set = TableSet {
            sam_pos: Self::upload(context, &sam_pos),
            clip_pos: Self::upload(context, &clip_pos),
            rel,
        };
        self.tables.insert(grid, set);
        Ok(())
    }

    /// One view through both towers and the projector. `capture` keeps its
    /// intermediate rows.
    pub(crate) fn run_view(
        &mut self,
        context: &mut gpu::MetalContext,
        pixels: &[f32],
        side: usize,
        capture: Option<&mut Uocr2Stages>,
    ) -> Result<Vec<f16>, RealForwardError> {
        if pixels.len() != side * side * 3 {
            return Err(unsupported(format!(
                "a {side} view needs {} values, got {}",
                side * side * 3,
                pixels.len()
            )));
        }
        let grid = side / 16;
        self.ensure_tables(context, grid)?;
        let this = &*self;
        gpu::autorelease_pool(|| this.run_view_inner(context, pixels, side, grid, capture))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_view_inner(
        &self,
        context: &mut gpu::MetalContext,
        pixels: &[f32],
        side: usize,
        grid: usize,
        mut capture: Option<&mut Uocr2Stages>,
    ) -> Result<Vec<f16>, RealForwardError> {
        let tables = &self.tables[&grid];
        let hw = grid * grid;
        let padded = grid.div_ceil(WINDOW) * WINDOW;
        let halfs = |context: &gpu::MetalContext, n: usize| context.new_output_buffer(n as u64 * 2);

        // ---- SAM ----
        let pixel_rows: Vec<f16> = pixels.iter().map(|v| f16::from_f32(*v)).collect();
        let pixel_buffer = context.new_buffer_with_data(&f16_slice_to_le_bytes(&pixel_rows));
        let seq_max = padded * padded;
        let x = halfs(context, hw * SAM_DIM);
        let normed = halfs(context, hw * SAM_DIM);
        let windows = halfs(context, seq_max * SAM_DIM);
        let (q, k, v) = (
            halfs(context, seq_max * SAM_DIM),
            halfs(context, seq_max * SAM_DIM),
            halfs(context, seq_max * SAM_DIM),
        );
        let attn = halfs(context, seq_max * SAM_DIM);
        let projected = halfs(context, seq_max * SAM_DIM);
        let mlp = halfs(context, hw * SAM_MLP);
        let mlp_out = halfs(context, hw * SAM_DIM);
        let head_dim = SAM_DIM / SAM_HEADS;
        let sam_scale = (head_dim as f32).powf(-0.5);

        let pass = context.begin_pass_labeled("uocr2 sam embed");
        gpu::encode_vision_conv2d(
            context,
            &pass,
            (&pixel_buffer, 0),
            self.view("sam.patch_embed.proj.weight")?,
            Some(self.view("sam.patch_embed.proj.bias")?),
            (&x, 0),
            gpu::VisionConvShape {
                h: side as u32,
                w: side as u32,
                c_in: 3,
                c_out: SAM_DIM as u32,
                kh: 16,
                kw: 16,
                stride: 16,
                pad: 0,
            },
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_residual_add(
            context,
            &pass,
            (&x, 0),
            (&tables.sam_pos, 0),
            (hw * SAM_DIM) as u32,
        )
        .map_err(gpu_err)?;
        pass.commit_and_wait();
        if let Some(c) = capture.as_mut() {
            c.patch = gpu::read_half_buffer(&x, hw * SAM_DIM);
        }

        for block in 0..SAM_DEPTH {
            let name = |suffix: &str| format!("sam.blocks.{block}.{suffix}");
            let global = SAM_GLOBAL.contains(&block);
            let attn_side = if global { grid } else { WINDOW };
            let seq = if global { hw } else { seq_max };

            let pass = context.begin_pass_labeled("uocr2 sam norm1");
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&x, 0),
                self.view(&name("norm1.weight"))?,
                self.view(&name("norm1.bias"))?,
                (&normed, 0),
                hw as u32,
                SAM_DIM as u32,
                SAM_EPS,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();

            // Partition on the host: the padded rows are zeros, as upstream's
            // `mx.pad` makes them, and they attend and are attended to.
            let attn_input: &gpu::MetalBuffer = if global {
                &normed
            } else {
                let rows = gpu::read_half_buffer(&normed, hw * SAM_DIM);
                let mut out = vec![f16::from_f32(0.0); seq_max * SAM_DIM];
                for t in 0..seq_max {
                    let (win, local) = (t / (WINDOW * WINDOW), t % (WINDOW * WINDOW));
                    let per_side = padded / WINDOW;
                    let y = (win / per_side) * WINDOW + local / WINDOW;
                    let x_at = (win % per_side) * WINDOW + local % WINDOW;
                    if y < grid && x_at < grid {
                        let src = (y * grid + x_at) * SAM_DIM;
                        out[t * SAM_DIM..(t + 1) * SAM_DIM]
                            .copy_from_slice(&rows[src..src + SAM_DIM]);
                    }
                }
                gpu::write_buffer_bytes(&windows, 0, &f16_slice_to_le_bytes(&out));
                &windows
            };

            let pass = context.begin_pass_labeled("uocr2 sam attn");
            let row_bytes = (SAM_DIM * SAM_DIM * 2) as u64;
            let bias_bytes = (SAM_DIM * 2) as u64;
            let qkv = name("attn.qkv.weight");
            let qkv_bias = name("attn.qkv.bias");
            for (part, out) in [&q, &k, &v].into_iter().enumerate() {
                gpu::encode_vision_matmul(
                    context,
                    &pass,
                    (attn_input, 0),
                    self.at(&qkv, part as u64 * row_bytes)?,
                    Some(self.at(&qkv_bias, part as u64 * bias_bytes)?),
                    (out, 0),
                    seq as u32,
                    SAM_DIM as u32,
                    SAM_DIM as u32,
                )
                .map_err(gpu_err)?;
            }
            let (rel_h, rel_w) = &tables.rel[block];
            gpu::encode_vision_attention_relpos(
                context,
                &pass,
                (&q, 0),
                (&k, 0),
                (&v, 0),
                (rel_h, 0),
                (rel_w, 0),
                (&attn, 0),
                seq as u32,
                SAM_HEADS as u32,
                head_dim as u32,
                attn_side as u32,
                sam_scale,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&attn, 0),
                self.view(&name("attn.proj.weight"))?,
                Some(self.view(&name("attn.proj.bias"))?),
                (&projected, 0),
                seq as u32,
                SAM_DIM as u32,
                SAM_DIM as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();

            let residual: &gpu::MetalBuffer = if global {
                &projected
            } else {
                let rows = gpu::read_half_buffer(&projected, seq_max * SAM_DIM);
                let mut out = vec![f16::from_f32(0.0); hw * SAM_DIM];
                for t in 0..seq_max {
                    let (win, local) = (t / (WINDOW * WINDOW), t % (WINDOW * WINDOW));
                    let per_side = padded / WINDOW;
                    let y = (win / per_side) * WINDOW + local / WINDOW;
                    let x_at = (win % per_side) * WINDOW + local % WINDOW;
                    if y < grid && x_at < grid {
                        let dst = (y * grid + x_at) * SAM_DIM;
                        out[dst..dst + SAM_DIM]
                            .copy_from_slice(&rows[t * SAM_DIM..(t + 1) * SAM_DIM]);
                    }
                }
                gpu::write_buffer_bytes(&windows, 0, &f16_slice_to_le_bytes(&out));
                &windows
            };

            let pass = context.begin_pass_labeled("uocr2 sam mlp");
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&x, 0),
                (residual, 0),
                (hw * SAM_DIM) as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&x, 0),
                self.view(&name("norm2.weight"))?,
                self.view(&name("norm2.bias"))?,
                (&normed, 0),
                hw as u32,
                SAM_DIM as u32,
                SAM_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&normed, 0),
                self.view(&name("mlp.lin1.weight"))?,
                Some(self.view(&name("mlp.lin1.bias"))?),
                (&mlp, 0),
                hw as u32,
                SAM_DIM as u32,
                SAM_MLP as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_gelu(
                context,
                &pass,
                (&mlp, 0),
                (hw * SAM_MLP) as u32,
                gpu::GeluKind::Erf,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&mlp, 0),
                self.view(&name("mlp.lin2.weight"))?,
                Some(self.view(&name("mlp.lin2.bias"))?),
                (&mlp_out, 0),
                hw as u32,
                SAM_MLP as u32,
                SAM_DIM as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&x, 0),
                (&mlp_out, 0),
                (hw * SAM_DIM) as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
        }

        // ---- neck, then the two downsamples ----
        let g2 = grid / 2;
        let g4 = grid / 4;
        let n1 = halfs(context, hw * NECK);
        let n2 = halfs(context, hw * NECK);
        let d2 = halfs(context, g2 * g2 * 512);
        let features = halfs(context, g4 * g4 * CLIP_DIM);
        let conv = |h: usize, c_in: usize, c_out: usize, k: u32, stride: u32, pad: u32| {
            gpu::VisionConvShape {
                h: h as u32,
                w: h as u32,
                c_in: c_in as u32,
                c_out: c_out as u32,
                kh: k,
                kw: k,
                stride,
                pad,
            }
        };
        let pass = context.begin_pass_labeled("uocr2 sam neck");
        gpu::encode_vision_conv2d(
            context,
            &pass,
            (&x, 0),
            self.view("sam.neck.0.weight")?,
            None,
            (&n1, 0),
            conv(grid, SAM_DIM, NECK, 1, 1, 0),
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_layer_norm(
            context,
            &pass,
            (&n1, 0),
            self.view("sam.neck.1.weight")?,
            self.view("sam.neck.1.bias")?,
            (&n2, 0),
            hw as u32,
            NECK as u32,
            SAM_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_conv2d(
            context,
            &pass,
            (&n2, 0),
            self.view("sam.neck.2.weight")?,
            None,
            (&n1, 0),
            conv(grid, NECK, NECK, 3, 1, 1),
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_layer_norm(
            context,
            &pass,
            (&n1, 0),
            self.view("sam.neck.3.weight")?,
            self.view("sam.neck.3.bias")?,
            (&n2, 0),
            hw as u32,
            NECK as u32,
            SAM_EPS,
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_conv2d(
            context,
            &pass,
            (&n2, 0),
            self.view("sam.net_2.weight")?,
            None,
            (&d2, 0),
            conv(grid, NECK, 512, 3, 2, 1),
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_conv2d(
            context,
            &pass,
            (&d2, 0),
            self.view("sam.net_3.weight")?,
            None,
            (&features, 0),
            conv(g2, 512, CLIP_DIM, 3, 2, 1),
        )
        .map_err(gpu_err)?;
        pass.commit_and_wait();
        let tokens = g4 * g4;
        let sam_rows = gpu::read_half_buffer(&features, tokens * CLIP_DIM);
        if let Some(c) = capture.as_mut() {
            c.sam = sam_rows.clone();
        }

        // ---- CLIP, over [cls ; SAM features] ----
        let seq = tokens + 1;
        let cls = self.host("clip.embeddings.class_embedding")?;
        let mut embed: Vec<f16> = cls.iter().map(|v| f16::from_f32(*v)).collect();
        embed.extend_from_slice(&sam_rows);
        let clip_x = context.new_buffer_with_data(&f16_slice_to_le_bytes(&embed));
        let clip_n = halfs(context, seq * CLIP_DIM);
        let (cq, ck, cv) = (
            halfs(context, seq * CLIP_DIM),
            halfs(context, seq * CLIP_DIM),
            halfs(context, seq * CLIP_DIM),
        );
        let clip_attn = halfs(context, seq * CLIP_DIM);
        let clip_out = halfs(context, seq * CLIP_DIM);
        let clip_mlp = halfs(context, seq * CLIP_MLP);
        let clip_head = CLIP_DIM / CLIP_HEADS;
        let clip_scale = (clip_head as f32).powf(-0.5);
        let pass = context.begin_pass_labeled("uocr2 clip embed");
        gpu::encode_vision_residual_add(
            context,
            &pass,
            (&clip_x, 0),
            (&tables.clip_pos, 0),
            (seq * CLIP_DIM) as u32,
        )
        .map_err(gpu_err)?;
        gpu::encode_vision_layer_norm(
            context,
            &pass,
            (&clip_x, 0),
            self.view("clip.pre_layrnorm.weight")?,
            self.view("clip.pre_layrnorm.bias")?,
            (&clip_n, 0),
            seq as u32,
            CLIP_DIM as u32,
            CLIP_EPS,
        )
        .map_err(gpu_err)?;
        pass.commit_and_wait();
        gpu::write_buffer_bytes(
            &clip_x,
            0,
            &f16_slice_to_le_bytes(&gpu::read_half_buffer(&clip_n, seq * CLIP_DIM)),
        );

        for block in 0..CLIP_DEPTH {
            let name = |suffix: &str| format!("clip.transformer.blocks.{block}.{suffix}");
            let pass = context.begin_pass_labeled("uocr2 clip block");
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&clip_x, 0),
                self.view(&name("layer_norm1.weight"))?,
                self.view(&name("layer_norm1.bias"))?,
                (&clip_n, 0),
                seq as u32,
                CLIP_DIM as u32,
                CLIP_EPS,
            )
            .map_err(gpu_err)?;
            let row_bytes = (CLIP_DIM * CLIP_DIM * 2) as u64;
            let bias_bytes = (CLIP_DIM * 2) as u64;
            let qkv = name("self_attn.qkv_proj.weight");
            let qkv_bias = name("self_attn.qkv_proj.bias");
            for (part, out) in [&cq, &ck, &cv].into_iter().enumerate() {
                gpu::encode_vision_matmul(
                    context,
                    &pass,
                    (&clip_n, 0),
                    self.at(&qkv, part as u64 * row_bytes)?,
                    Some(self.at(&qkv_bias, part as u64 * bias_bytes)?),
                    (out, 0),
                    seq as u32,
                    CLIP_DIM as u32,
                    CLIP_DIM as u32,
                )
                .map_err(gpu_err)?;
            }
            gpu::encode_vision_attention(
                context,
                &pass,
                (&cq, 0),
                (&ck, 0),
                (&cv, 0),
                (&clip_attn, 0),
                seq as u32,
                CLIP_HEADS as u32,
                clip_head as u32,
                clip_scale,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&clip_attn, 0),
                self.view(&name("self_attn.out_proj.weight"))?,
                Some(self.view(&name("self_attn.out_proj.bias"))?),
                (&clip_out, 0),
                seq as u32,
                CLIP_DIM as u32,
                CLIP_DIM as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&clip_x, 0),
                (&clip_out, 0),
                (seq * CLIP_DIM) as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_layer_norm(
                context,
                &pass,
                (&clip_x, 0),
                self.view(&name("layer_norm2.weight"))?,
                self.view(&name("layer_norm2.bias"))?,
                (&clip_n, 0),
                seq as u32,
                CLIP_DIM as u32,
                CLIP_EPS,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&clip_n, 0),
                self.view(&name("mlp.fc1.weight"))?,
                Some(self.view(&name("mlp.fc1.bias"))?),
                (&clip_mlp, 0),
                seq as u32,
                CLIP_DIM as u32,
                CLIP_MLP as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_quick_gelu(context, &pass, (&clip_mlp, 0), (seq * CLIP_MLP) as u32)
                .map_err(gpu_err)?;
            gpu::encode_vision_matmul(
                context,
                &pass,
                (&clip_mlp, 0),
                self.view(&name("mlp.fc2.weight"))?,
                Some(self.view(&name("mlp.fc2.bias"))?),
                (&clip_out, 0),
                seq as u32,
                CLIP_MLP as u32,
                CLIP_DIM as u32,
            )
            .map_err(gpu_err)?;
            gpu::encode_vision_residual_add(
                context,
                &pass,
                (&clip_x, 0),
                (&clip_out, 0),
                (seq * CLIP_DIM) as u32,
            )
            .map_err(gpu_err)?;
            pass.commit_and_wait();
        }

        // ---- the projector over [CLIP past cls ; SAM] ----
        let clip_rows = gpu::read_half_buffer(&clip_x, seq * CLIP_DIM);
        if let Some(c) = capture.as_mut() {
            c.clip = clip_rows.clone();
        }
        let mut fused = Vec::with_capacity(tokens * (CLIP_DIM + CLIP_DIM));
        for t in 0..tokens {
            fused.extend_from_slice(&clip_rows[(t + 1) * CLIP_DIM..(t + 2) * CLIP_DIM]);
            fused.extend_from_slice(&sam_rows[t * CLIP_DIM..(t + 1) * CLIP_DIM]);
        }
        let fused_buffer = context.new_buffer_with_data(&f16_slice_to_le_bytes(&fused));
        let out = halfs(context, tokens * OUT_DIM);
        let pass = context.begin_pass_labeled("uocr2 projector");
        gpu::encode_vision_matmul(
            context,
            &pass,
            (&fused_buffer, 0),
            self.view("projector.weight")?,
            Some(self.view("projector.bias")?),
            (&out, 0),
            tokens as u32,
            (CLIP_DIM * 2) as u32,
            OUT_DIM as u32,
        )
        .map_err(gpu_err)?;
        pass.commit_and_wait();
        let rows = gpu::read_half_buffer(&out, tokens * OUT_DIM);
        if let Some(c) = capture {
            c.projected = rows.clone();
        }
        Ok(rows)
    }

    /// A page's fused rows, in upstream's order: local tiles with a newline per
    /// grid row, then the global view with a newline per row, then the view
    /// separator. The row count equals [`Uocr2Page::image_tokens`].
    pub(crate) fn encode_page(
        &mut self,
        context: &mut gpu::MetalContext,
        page: &Uocr2Page,
    ) -> Result<Vec<f16>, RealForwardError> {
        let newline: Vec<f16> = self
            .host("image_newline")?
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let separator: Vec<f16> = self
            .host("view_separator")?
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let mut rows = Vec::with_capacity(page.image_tokens() * OUT_DIM);
        if !page.crops.is_empty() {
            let q = uocr2_grid(UOCR2_CROP_SIZE);
            let (cols, grid_rows) = page.grid;
            let mut tiles = Vec::with_capacity(page.crops.len());
            for crop in &page.crops {
                tiles.push(self.run_view(context, crop, UOCR2_CROP_SIZE, None)?);
            }
            for tile_row in 0..grid_rows {
                for y in 0..q {
                    for col in 0..cols {
                        let tile = &tiles[tile_row * cols + col];
                        rows.extend_from_slice(&tile[y * q * OUT_DIM..(y + 1) * q * OUT_DIM]);
                    }
                    rows.extend_from_slice(&newline);
                }
            }
        }
        let base = self.run_view(context, &page.base, UOCR2_BASE_SIZE, None)?;
        let g = uocr2_grid(UOCR2_BASE_SIZE);
        for y in 0..g {
            rows.extend_from_slice(&base[y * g * OUT_DIM..(y + 1) * g * OUT_DIM]);
            rows.extend_from_slice(&newline);
        }
        rows.extend_from_slice(&separator);
        Ok(rows)
    }
}
