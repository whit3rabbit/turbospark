//! The vision tower end to end on a synthetic install: does the streamed
//! block loop compute what `compute::vision` says it should, and
//! does it hold the memory it claims to (ROADMAP M-V4, stage 1).
//!
//! **This file exists so the real-checkpoint parity run is not the thing that
//! finds the wiring holes** -- `crates/repack` Gotcha 8's rule, one milestone
//! on. It costs a tenth of a second and makes every structural assertion the
//! 15 GB install would otherwise make minutes at a time.
//!
//! # What it can and cannot see
//!
//! It CAN see: that the 27-block streaming shape runs at all, that the
//! composition matches an independent CPU reference reading the SAME bytes,
//! that residency is two slots plus one page of scratch and does not grow
//! across pages, and that a malformed install is refused by name.
//!
//! It CANNOT see whether the tower is the RIGHT function for the real
//! checkpoint: the fixture's weights are untrained, so a convention this port
//! and the reference share would agree here and disagree with mlx-vlm. That
//! is `vision_tower_parity.rs`'s job, and it is the reason this file's
//! CPU-vs-GPU case is a wiring check rather than a correctness claim.
//!
//! # The fixture's dimensions are mutually indivisible on purpose
//!
//! `tiny_vision_config` gives hidden 64, intermediate 96, merger input 256,
//! out_hidden 128 and a 16-row position table, where the real tower has
//! several shapes that are multiples of one another (hidden 1152 against a
//! merger input of 4608). A transposed read or a swapped role changes a byte
//! COUNT here and is caught by construction.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use turbospark_repack::{
    build_synthetic_qwen_gdn_dense_install_with_vision_streamed, tiny_vision_config,
    VISION_BLOCK_ROLES, VISION_RESIDENT_TENSORS,
};
use turbospark_runtime::RealForwardRunner;
use turbospark_vision_io::{GridThw, PreprocessParams, PreprocessedImage};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const VOCAB: i64 = 256;
const LAYERS: i64 = 4;
const BITS: u32 = 1;
/// The image grid. 4x4 patches, so 16 patches and 4 merged tokens at the
/// fixture's merge size of 2 -- more than one merge window in both axes, so a
/// row-major-versus-window-order mistake in the merger changes the answer.
const GRID_H: usize = 4;
const GRID_W: usize = 4;

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("turbospark-vision-mv4-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn build() -> (PathBuf, model_io::ArchConfig) {
    let dir = temp_dir();
    let arch = build_synthetic_qwen_gdn_dense_install_with_vision_streamed(
        &dir,
        VOCAB,
        LAYERS,
        "qwen35-vision-mv4",
        BITS,
    )
    .expect("the streamed writer builds a dense install with a vision tower");
    (dir, arch)
}

/// The fixture's preprocessing parameters, matching `tiny_vision_config`.
///
/// Built by hand rather than parsed, because the fixture ships no
/// `preprocessor_config.json`; the pixel bounds are unused here since nothing
/// resizes.
fn params() -> PreprocessParams {
    let v = tiny_vision_config();
    PreprocessParams {
        patch_size: v.patch_size as usize,
        temporal_patch_size: v.temporal_patch_size as usize,
        merge_size: v.spatial_merge_size as usize,
        in_channels: v.in_channels as usize,
        min_pixels: 1,
        max_pixels: 1 << 20,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
    }
}

/// A deterministic page at the fixture's geometry.
///
/// Values are bounded to roughly what a normalized image carries, so the
/// tower's FP16 activations stay in range rather than saturating on a
/// contrived input (the trap `synthetic_gguf`'s Q8_0 fixture records).
fn image(params: &PreprocessParams) -> PreprocessedImage {
    let grid = GridThw::new(1, GRID_H, GRID_W);
    let dim = params.patch_dim();
    let patch_rows: Vec<f32> = (0..grid.patches() * dim)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.8)
        .collect();
    PreprocessedImage {
        merged_tokens: grid.merged_tokens(params.merge_size),
        patch_rows,
        grid,
        resized: (GRID_H * params.patch_size, GRID_W * params.patch_size),
    }
}

// ---------------------------------------------------------------------------
// Reading the install's own bytes, so the reference runs on the same weights
// ---------------------------------------------------------------------------

fn fp16_slice(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// The nine resident vision tensors, by their install names.
fn resident_tensors(dir: &Path) -> std::collections::HashMap<String, Vec<f32>> {
    let path = dir.join("model_weights.bin");
    let index = model_io::load_resident_index(&path).expect("resident index");
    let bytes = std::fs::read(&path).expect("read model_weights.bin");
    let mut out = std::collections::HashMap::new();
    for tail in VISION_RESIDENT_TENSORS {
        let name = format!("vision.{tail}");
        let e = index.entries.get(&name).expect("vision resident tensor");
        let start = e.file_offset as usize;
        out.insert(
            tail.to_string(),
            fp16_slice(&bytes[start..start + e.size_bytes as usize]),
        );
    }
    out
}

/// Every block's twelve roles, in `VISION_BLOCK_ROLES` order.
fn block_tensors(dir: &Path) -> Vec<Vec<Vec<f32>>> {
    let layout = model_io::load_packed_layout_from(
        dir,
        model_io::PACKED_VISION_DIR,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("packed_vision layout");
    let layer = &layout.layers[0];
    let blob_path = dir.join(model_io::PACKED_VISION_DIR).join(&layer.file);
    let bytes = std::fs::read(&blob_path).expect("read packed_vision blobs");

    let mut blocks = vec![Vec::new(); layer.experts.len()];
    for entry in &layer.experts {
        let mut roles = Vec::with_capacity(VISION_BLOCK_ROLES.len());
        for (role, _) in VISION_BLOCK_ROLES {
            let sub = entry.sub_tensors.get(role).expect("role present");
            let start = (entry.offset + sub.offset) as usize;
            roles.push(fp16_slice(&bytes[start..start + sub.size as usize]));
        }
        blocks[entry.expert] = roles;
    }
    blocks
}

// ---------------------------------------------------------------------------
// The CPU reference tower
// ---------------------------------------------------------------------------

/// Round through FP16 and back.
///
/// Every value the GPU sees is FP16 storage, so the reference has to start
/// from the SAME numbers. Comparing against the f32 originals would fold the
/// storage rounding into the parity bound and hide a real error of the same
/// size -- the discipline `crates/gpu/tests/vision_block_parity.rs` states.
fn quantize(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect()
}

struct Ref {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    inter: usize,
    merge: usize,
    out_hidden: usize,
    eps: f32,
    /// Which GELU the BLOCK's MLP uses. `false` is the correct one (tanh);
    /// `true` is what `the_gelu_choice_is_invisible_at_this_bound` runs to
    /// demonstrate that this file cannot tell the two apart.
    block_erf: bool,
}

impl Ref {
    fn norm_rows(&self, x: &[f32], w: &[f32], b: &[f32], rows: usize, d: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * d);
        for r in 0..rows {
            out.extend(compute::vision::layer_norm(
                &x[r * d..(r + 1) * d],
                w,
                b,
                self.eps,
            ));
        }
        out
    }

    fn rope(&self, buf: &[f32], freqs: &[f32], seq: usize) -> Vec<f32> {
        let half = self.head_dim / 2;
        let mut out = Vec::with_capacity(buf.len());
        for t in 0..seq {
            let row = &freqs[t * half..(t + 1) * half];
            for h in 0..self.heads {
                let base = (t * self.heads + h) * self.head_dim;
                out.extend(compute::vision::rope_vision_2d(
                    &buf[base..base + self.head_dim],
                    row,
                ));
            }
        }
        out
    }

    /// One block, straight off `Qwen3VLMoEVisionBlock.__call__`.
    fn block(&self, x: &[f32], w: &[Vec<f32>], freqs: &[f32], seq: usize) -> Vec<f32> {
        let h = self.hidden;
        let mut stream = x.to_vec();

        let n1 = self.norm_rows(&stream, &w[0], &w[1], seq, h);
        // q, k, v at rows 0, h, 2h of the fused weight and the matching
        // thirds of the fused bias -- the same partition the reference's
        // `reshape(seq, 3, heads, head_dim)` makes.
        let mut qkv = Vec::with_capacity(3);
        for i in 0..3 {
            qkv.push(compute::vision::matmul_bias(
                &n1,
                &w[2][i * h * h..(i + 1) * h * h],
                Some(&w[3][i * h..(i + 1) * h]),
                seq,
                h,
                h,
            ));
        }
        // Q AND K ONLY.
        let q = self.rope(&qkv[0], freqs, seq);
        let k = self.rope(&qkv[1], freqs, seq);
        let attn = compute::vision::bidirectional_attention(
            &q,
            &k,
            &qkv[2],
            seq,
            self.heads,
            self.head_dim,
            compute::vision::attention_scale(self.head_dim),
        );
        let proj = compute::vision::matmul_bias(&attn, &w[4], Some(&w[5]), seq, h, h);
        for (s, p) in stream.iter_mut().zip(&proj) {
            *s += p;
        }

        let n2 = self.norm_rows(&stream, &w[6], &w[7], seq, h);
        let mut h1 = compute::vision::matmul_bias(&n2, &w[8], Some(&w[9]), seq, h, self.inter);
        // TANH in the block, ERF in the merger.
        h1 = if self.block_erf {
            compute::vision::gelu_erf(&h1)
        } else {
            compute::vision::gelu_tanh_vision(&h1)
        };
        let out = compute::vision::matmul_bias(&h1, &w[10], Some(&w[11]), seq, self.inter, h);
        for (s, o) in stream.iter_mut().zip(&out) {
            *s += o;
        }
        stream
    }

    fn merger(
        &self,
        x: &[f32],
        r: &std::collections::HashMap<String, Vec<f32>>,
        seq: usize,
    ) -> Vec<f32> {
        // Over `hidden` PER PATCH ROW, then the reshape. The reference builds
        // its PatchMerger with `use_postshuffle_norm=False`.
        let normed = self.norm_rows(
            x,
            &r["merger.norm.weight"],
            &r["merger.norm.bias"],
            seq,
            self.hidden,
        );
        let wide = self.hidden * self.merge * self.merge;
        let merged = seq / (self.merge * self.merge);
        let m1 = compute::vision::matmul_bias(
            &normed,
            &r["merger.linear_fc1.weight"],
            Some(&r["merger.linear_fc1.bias"]),
            merged,
            wide,
            wide,
        );
        let m1 = compute::vision::gelu_erf(&m1);
        compute::vision::matmul_bias(
            &m1,
            &r["merger.linear_fc2.weight"],
            Some(&r["merger.linear_fc2.bias"]),
            merged,
            wide,
            self.out_hidden,
        )
    }
}

/// The whole reference tower on the install's own weights.
fn cpu_tower(dir: &Path, image: &PreprocessedImage, params: &PreprocessParams) -> Vec<f32> {
    cpu_tower_with(dir, image, params, false)
}

fn cpu_tower_with(
    dir: &Path,
    image: &PreprocessedImage,
    params: &PreprocessParams,
    block_erf: bool,
) -> Vec<f32> {
    let v = tiny_vision_config();
    let r = Ref {
        hidden: v.hidden_size as usize,
        heads: v.num_heads as usize,
        head_dim: (v.hidden_size / v.num_heads) as usize,
        inter: v.intermediate_size as usize,
        merge: v.spatial_merge_size as usize,
        out_hidden: v.out_hidden_size as usize,
        eps: 1e-6,
        block_erf,
    };
    let resident = resident_tensors(dir);
    let blocks = block_tensors(dir);
    let seq = image.grid.patches();
    let side = (v.num_position_embeddings as f64).sqrt() as usize;

    let rows = quantize(&image.patch_rows);
    let mut x = compute::vision::matmul_bias(
        &rows,
        &resident["patch_embed.proj.weight"],
        Some(&resident["patch_embed.proj.bias"]),
        seq,
        params.patch_dim(),
        r.hidden,
    );

    // The position blend, host-side on both sides of the comparison; the
    // difference is that the engine narrows the blended row to FP16 before
    // the add, so the reference does too.
    let table = turbospark_vision_io::pos_embed_weights(image.grid, side, params).expect("pos");
    let pos_src = &resident["pos_embed.weight"];
    for patch in 0..seq {
        for d in 0..r.hidden {
            let mut acc = 0.0f32;
            for corner in 0..4 {
                let row = table.indices[corner][patch];
                acc += table.weights[corner][patch] * pos_src[row * r.hidden + d];
            }
            x[patch * r.hidden + d] += half::f16::from_f32(acc).to_f32();
        }
    }

    let freqs = quantize(
        &turbospark_vision_io::rope::vision_rope_freq_rows_default(image.grid, r.head_dim, params)
            .expect("rope rows"),
    );
    for w in &blocks {
        x = r.block(&x, w, &freqs, seq);
    }
    r.merger(&x, &resident, seq)
}

fn embedding_f32(rows: &[u16]) -> Vec<f32> {
    rows.iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect()
}

/// [`image`] at an arbitrary grid, for the row-tiling cases below that need a
/// second `seq` distinct from the fixture's default 4x4/16-patch page.
///
/// `gh` and `gw` must each be a multiple of the fixture's merge size (2), the
/// same constraint [`image`]'s fixed 4x4 already satisfies.
fn image_with_grid(params: &PreprocessParams, gh: usize, gw: usize) -> PreprocessedImage {
    let grid = GridThw::new(1, gh, gw);
    let dim = params.patch_dim();
    let patch_rows: Vec<f32> = (0..grid.patches() * dim)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.8)
        .collect();
    PreprocessedImage {
        merged_tokens: grid.merged_tokens(params.merge_size),
        patch_rows,
        grid,
        resized: (gh * params.patch_size, gw * params.patch_size),
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// The tower runs and its output is finite.
///
/// Finiteness FIRST and separately, because NaN reads as a perfect score on
/// every rank and top-k instrument downstream and hashes as stably as any
/// other bit pattern (AGENTS.md Gotcha 59). A digest taken before this check
/// would freeze a broken tower.
#[test]
fn the_tower_runs_and_returns_finite_rows() {
    let (dir, arch) = build();
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    assert!(runner.has_vision_tower(), "the fixture declares a tower");

    let out = runner.encode_image(&img, &p).expect("encode");
    let v = tiny_vision_config();
    assert_eq!(out.merged_tokens, img.merged_tokens);
    assert_eq!(out.out_hidden, v.out_hidden_size as usize);
    assert_eq!(out.rows.len(), out.merged_tokens * out.out_hidden);
    let values = embedding_f32(&out.rows);
    assert!(
        values.iter().all(|v| v.is_finite()),
        "the tower produced a non-finite row"
    );
    assert!(
        values.iter().any(|v| *v != 0.0),
        "an all-zero embedding would pass every finiteness check and mean nothing"
    );
}

/// The GPU tower and the CPU reference agree, reading the SAME install bytes.
///
/// The only assertion here that compares against something computed
/// independently. Untrained weights make this a WIRING check -- q/k/v
/// slicing, rope reaching q and k but not v, both residual adds, the merger's
/// norm width, the two GELUs -- and not a claim that the arithmetic matches
/// the published model. The real-checkpoint gate is what says that.
///
/// The bound is relative to the output's own RMS rather than absolute,
/// because the tower's scale is set by untrained weights and an absolute
/// number here would be a magic constant. **Measured at 8.4e-3** (worst
/// absolute 9.9e-4 against an RMS of 0.1175) against a bar of 2.5e-2, so the
/// headroom is a factor of THREE and not the order of magnitude a reader
/// might assume. Both sides are deterministic, so that number does not
/// wander; what it reflects is FP16 storage plus two different reduction
/// orders through two blocks and a 256-wide merger GEMM.
///
/// Three is enough because the wiring mistakes it exists to catch are not
/// small perturbations. Measured, one mutation at a time: rotating `v` as
/// well as q and k reads 1.36, swapping q and v in the fused projection reads
/// 0.73, and norming the merger's wide row instead of its patch rows reads
/// 2.61 -- 29 to 104 times the bar, against a clean run's 0.34 of it.
///
/// **The GELU choice is the exception and it is stated as its own case
/// below**: both directions SURVIVE this bound.
#[test]
fn the_gpu_tower_agrees_with_the_cpu_reference() {
    let (dir, arch) = build();
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    let got = embedding_f32(&runner.encode_image(&img, &p).expect("encode").rows);
    let want = cpu_tower(&dir, &img, &p);

    assert_eq!(got.len(), want.len(), "output shapes differ");
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    assert!(rms > 1e-3, "reference output is degenerate: rms {rms}");
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst / rms < 2.5e-2,
        "GPU and CPU towers disagree: worst {worst} against rms {rms} (relative {})",
        worst / rms
    );
}

/// **THIS FILE CANNOT SEE WHICH GELU EITHER STAGE USES, and that is recorded
/// rather than fixed.**
///
/// `crates/gpu` Gotcha 9 states the same limitation for the whole-BLOCK
/// parity test; this is it one layer up, and it had to be measured rather
/// than inherited. Both mutations survive: swapping the block's tanh for the
/// merger's erf leaves all eight cases green, and so does the reverse.
///
/// The reason is arithmetic. The two forms agree to about 3e-4, while a
/// tower's output carries the accumulated FP16 error of two norms, five
/// GEMMs and an attention per block -- 8.4e-3 relative on this fixture, more
/// than an order of magnitude above the signal. No tightening of the bound
/// fixes that, because the bound is measuring the storage rather than the
/// choice.
///
/// So the assertion is the LIMITATION itself: the two towers differ, and by
/// less than the parity case's own bar. A reader cannot then assume the
/// composition test covers the choice. What does cover it is the pair of
/// per-kernel cases in `crates/gpu/tests/vision_parity.rs` plus the one-line
/// call site in each stage.
#[test]
fn the_gelu_choice_is_invisible_at_this_bound() {
    let (dir, _) = build();
    let p = params();
    let img = image(&p);
    let tanh = cpu_tower_with(&dir, &img, &p, false);
    let erf = cpu_tower_with(&dir, &img, &p, true);

    let rms = (tanh.iter().map(|v| v * v).sum::<f32>() / tanh.len() as f32).sqrt();
    let worst = tanh
        .iter()
        .zip(&erf)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 0.0,
        "the two GELUs must be different functions, or this case proves nothing"
    );
    assert!(
        worst / rms < 2.5e-2,
        "the two GELUs differ by {} of the output RMS, which is ABOVE the parity bar -- if this \
         ever fires, the composition test has become able to see the choice and this case is \
         the thing to delete",
        worst / rms
    );
}

/// Residency is two slots plus one page of scratch, and the page's scratch is
/// what the sizing formula predicts.
///
/// Two independent numbers rather than one: `vision_last_scratch_bytes` is
/// what the allocator was asked for and `vision_scratch_bytes_for` is what a
/// caller budgeting a page would compute, so a formula that drifted from the
/// allocation reddens here instead of understating a budget silently.
#[test]
fn the_tower_holds_two_slots_and_one_page_of_scratch() {
    let (dir, arch) = build();
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    assert_eq!(
        runner.vision_slot_bytes(),
        None,
        "the tower is opened lazily, so a runner that has seen no image holds none of it"
    );

    runner.encode_image(&img, &p).expect("encode");

    let layout = model_io::load_packed_layout_from(
        &dir,
        model_io::PACKED_VISION_DIR,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("layout");
    let stride = layout.layers[0].expert_stride;
    assert_eq!(
        runner.vision_slot_bytes(),
        Some(turbospark_runtime::vision::VISION_SLOTS as u64 * stride),
        "the slot cache is exactly VISION_SLOTS blobs, whatever the depth"
    );
    let seq = img.grid.patches();
    assert_eq!(
        runner.vision_last_scratch_bytes(),
        runner.vision_scratch_bytes_for(seq),
        "the scratch sizing formula and the allocation disagree"
    );
}

/// A second page allocates no more Metal buffers than the first.
///
/// The Gotcha 17 shape: a per-block allocation leak is invisible to
/// correctness and grows with the number of pages, which is exactly what a
/// bulk-OCR loop does. Counting BUFFERS rather than bytes is what catches a
/// buffer created inside the block loop; the scratch itself is allocated per
/// page by design and both pages pay the same amount.
#[test]
fn a_second_page_allocates_no_more_buffers_than_the_first() {
    let (dir, arch) = build();
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");

    runner.encode_image(&img, &p).expect("first page");
    let after_first = runner.gpu_buffer_allocations();
    runner.encode_image(&img, &p).expect("second page");
    let after_second = runner.gpu_buffer_allocations();
    runner.encode_image(&img, &p).expect("third page");
    let after_third = runner.gpu_buffer_allocations();

    let second = after_second - after_first;
    let third = after_third - after_second;
    assert_eq!(
        second, third,
        "per-page buffer count is not steady: {second} then {third}"
    );
    // The scratch is 13 buffers; anything much above that means something
    // inside the 2-block loop is allocating.
    assert!(
        second <= 16,
        "a page allocated {second} buffers, which is more than one scratch set"
    );
}

/// The same page twice produces the same bytes.
///
/// Nothing in the tower depends on hidden state -- there is no cache whose
/// contents could reach the output -- and this is what says so rather than
/// leaving it to the design. It is the shape
/// `gguf_nondeterminism_probe` exists for on the decode side, where the
/// answer was NO for two months (AGENTS.md Gotcha 27).
#[test]
fn the_same_page_encodes_to_the_same_bytes_twice() {
    let (dir, arch) = build();
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    let first = runner.encode_image(&img, &p).expect("first").rows;
    let second = runner.encode_image(&img, &p).expect("second").rows;
    assert_eq!(first, second, "the tower is not deterministic across pages");
}

/// A text-only install refuses an image by name.
#[test]
fn an_install_without_a_tower_refuses_an_image() {
    let dir = temp_dir();
    let arch = turbospark_repack::build_synthetic_qwen_gdn_dense_install(
        &dir,
        VOCAB,
        LAYERS,
        "qwen35-textonly",
    )
    .expect("text-only install");
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");
    assert!(!runner.has_vision_tower());
    let Err(err) = runner.encode_image(&img, &p) else {
        panic!("a text-only install must refuse an image rather than returning no rows");
    };
    let text = err.to_string();
    assert!(
        text.contains("no vision tower"),
        "the refusal must name the missing component: {text}"
    );
}

/// Preprocessing that disagrees with the install is refused by name.
///
/// Both directions matter and both are silent otherwise: a wrong patch
/// dimension makes the patch-embedding GEMM read the wrong `k` (a correctly
/// shaped product over the wrong data), and a wrong merge size makes the
/// merger reshape the stream into rows that mix patches from different
/// tokens.
#[test]
fn preprocessing_that_disagrees_with_the_install_is_refused() {
    let (dir, arch) = build();
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");

    let mut wrong_patch = p.clone();
    wrong_patch.patch_size += 1;
    let Err(err) = runner.encode_image(&img, &wrong_patch) else {
        panic!("a patch dimension the tower cannot take must be refused");
    };
    assert!(
        err.to_string().contains("patch rows"),
        "refusal must name the patch rows: {err}"
    );

    let mut wrong_merge = p.clone();
    wrong_merge.merge_size += 1;
    let Err(err) = runner.encode_image(&img, &wrong_merge) else {
        panic!("a merge size the tower does not declare must be refused");
    };
    assert!(
        err.to_string().contains("merge size"),
        "refusal must name the merge size: {err}"
    );
}

/// The role and prefix tables this crate restates agree with the writer's.
///
/// `crates/repack` is a DEV-dependency here, so the production code carries
/// its own copy of both rather than inverting the dependency direction to
/// read a few strings. This is what stops the copy from drifting.
#[test]
fn the_role_and_prefix_tables_match_the_writers() {
    let (dir, _) = build();
    let index = model_io::load_resident_index(&dir.join("model_weights.bin")).expect("index");
    for tail in VISION_RESIDENT_TENSORS {
        assert!(
            index.entries.contains_key(&format!("vision.{tail}")),
            "the walk writes vision.{tail} under a prefix this crate does not expect"
        );
    }
    let layout = model_io::load_packed_layout_from(
        &dir,
        model_io::PACKED_VISION_DIR,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("layout");
    let mut writer: Vec<&str> = VISION_BLOCK_ROLES.iter().map(|(r, _)| *r).collect();
    writer.sort_unstable();
    let mut written: Vec<&str> = layout.layers[0].experts[0]
        .sub_tensors
        .keys()
        .map(|k| k.as_str())
        .collect();
    written.sort_unstable();
    assert_eq!(
        writer, written,
        "the roles the walk writes are not the roles this crate resolves"
    );
}

// ---------------------------------------------------------------------------
// Row-tiling the MLP (Part B1)
// ---------------------------------------------------------------------------

/// Row-tiling `fc1 -> gelu -> fc2` must not change the tower's output.
///
/// The fixture's page is 16 patches; a tile of 4 forces FOUR loop iterations
/// where the shipped default (2,048) takes exactly one, so this is the
/// multi-tile arm the extreme real page (64,516 patches, 32 tiles at the
/// default) exercises in miniature. `set_vision_mlp_tile_rows` mutates the
/// tower IN PLACE, so the same runner and the same install bytes back both
/// runs -- only the tile size differs between them.
#[test]
fn row_tiling_the_mlp_does_not_change_the_towers_output() {
    let (dir, arch) = build();
    let p = params();
    let img = image(&p);
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");

    let baseline = runner
        .encode_image(&img, &p)
        .expect("baseline (untiled)")
        .rows;
    runner.set_vision_mlp_tile_rows(4);
    let tiled = runner.encode_image(&img, &p).expect("tiled").rows;

    assert_eq!(
        baseline, tiled,
        "tiling the MLP into 4-row chunks changed the tower's output; fc1 -> gelu -> fc2 is \
         row-independent and must produce identical bytes at any tile size"
    );
}

/// The scratch-bytes prediction and the real allocation agree at two
/// different page sizes, one smaller than its tile and one spanning several.
///
/// `vision_scratch_bytes_for` and `vision_last_scratch_bytes` are two
/// independent numbers (Gotcha 25's residency case, one level up): the first
/// is what a caller budgeting a page would compute and the second is what the
/// allocator actually asked for, so a formula that drifted from the
/// allocation reddens here rather than understating a budget silently. Both
/// read `VisionTower::mlp_tile_rows` at call time, so the equality holds
/// under an override too, not only at the shipped default.
#[test]
fn scratch_bytes_matches_the_allocation_at_two_seq_values() {
    let (dir, arch) = build();
    let p = params();
    let mut runner = RealForwardRunner::open(&dir, arch).expect("open");

    // Case 1: the fixture's default 16-patch page against the SHIPPED
    // default tile (2,048) -- seq is far smaller than the tile, one
    // iteration.
    let small = image(&p);
    runner.encode_image(&small, &p).expect("small page");
    let seq = small.grid.patches();
    assert_eq!(
        runner.vision_last_scratch_bytes(),
        runner.vision_scratch_bytes_for(seq),
        "scratch_bytes disagrees with the real allocation at seq={seq}, default tile"
    );

    // Case 2: the SAME 16-patch page, now several tiles wide relative to an
    // overridden tile of 4 (4 tiles). Confirms the formula tracks an
    // overridden tile, not just the shipped constant.
    runner.set_vision_mlp_tile_rows(4);
    runner.encode_image(&small, &p).expect("small page, tiled");
    assert_eq!(
        runner.vision_last_scratch_bytes(),
        runner.vision_scratch_bytes_for(seq),
        "scratch_bytes disagrees with the real allocation at seq={seq}, tile=4"
    );
    // And the tiled allocation must actually be SMALLER than the untiled
    // one's h1 term would have been at this seq -- otherwise the override
    // reached nothing and this case would pass by accident.
    let v = tiny_vision_config();
    let untiled_h1 = 2 * seq * v.intermediate_size as usize;
    let tiled_h1 = 2 * 4 * v.intermediate_size as usize;
    assert!(
        tiled_h1 < untiled_h1,
        "the tile override must shrink h1 below the untiled size for this case to mean anything"
    );

    // Case 3: a LARGER page (32 patches) against the shipped default tile
    // (2,048) -- a second, genuinely different seq, still smaller than the
    // tile.
    let large = image_with_grid(&p, 4, 8);
    let mut runner2 = RealForwardRunner::open(&dir, arch_for_grid()).expect("open second runner");
    let seq2 = large.grid.patches();
    runner2.encode_image(&large, &p).expect("large page");
    assert_eq!(
        runner2.vision_last_scratch_bytes(),
        runner2.vision_scratch_bytes_for(seq2),
        "scratch_bytes disagrees with the real allocation at seq={seq2}, default tile"
    );
}

/// Same fixture as [`build`], for the second runner
/// [`scratch_bytes_matches_the_allocation_at_two_seq_values`] opens against a
/// fresh directory.
fn arch_for_grid() -> model_io::ArchConfig {
    let (_, arch) = build();
    arch
}
