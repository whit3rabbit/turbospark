//! The item 10 verification loop: does the gated-DeltaNet convention hold on
//! EVERY layer and EVERY tensor carrying a V-head axis, and does undoing it
//! make the real Qwen GGUF install generate sense?
//!
//! Two things `gguf_qwen_core_probe.rs` cannot do. That probe reads LAYER 0
//! and only BF16 tensors, which is where all three F32-sourced transforms
//! were characterized; one layer is not evidence about thirty, and BF16 is
//! not evidence about the five QUANTIZED tensors on the same axis
//! (`gguf_qwen_quant_probe.rs` found those after fixing the first three left
//! the model still generating gibberish). And it only measures; it cannot
//! answer the question the item turns on, which is whether coherent text
//! comes out the other end.
//!
//! **Why patch instead of repack.** These tensors sit at a fixed
//! `file_offset`/`size_bytes`, every transform preserves length, and
//! `RealForwardRunner::open` runs no receipt or SHA-256 check (`model_io` has
//! a verifier; the open path never calls it). So the whole-model coherence
//! gate costs seconds instead of the 23 minutes a streamed repack costs. The
//! repack is the LAST step, proving the walk writes what this wrote, not the
//! search loop.
//!
//! Read-only by default. `MREFRUST_QWEN_PATCH=1` is what writes, and it is
//! IDEMPOTENT: a tensor is patched only if the transform agrees with the MLX
//! install better than the bytes already on disk do, so a second run is a
//! no-op rather than a double permutation.
//!
//!   MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!   MREFRUST_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
//!     cargo test -p mrefrust-repack --test gguf_qwen_convention_patch --release -- --ignored --nocapture

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use compute::{
    dequantize_int4_affine, dequantize_int8_affine, dequantize_q4_k, dequantize_q6_k,
    dequantize_q8_0, pearson, Int4AffineRow, Int8AffineRow,
};
use model_io::{ArchConfig, ResidentIndex, ResidentIndexEntry};

const BF16: u8 = 1;

fn install(var: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(var).unwrap_or_else(|| panic!("{var} must point at a Qwen 3.6 install")),
    )
}

struct Weights {
    bytes: Vec<u8>,
    index: ResidentIndex,
}

impl Weights {
    fn open(dir: &Path) -> Self {
        let path = dir.join("model_weights.bin");
        Self {
            index: model_io::load_resident_index(&path).expect("resident index"),
            bytes: std::fs::read(&path).expect("weights"),
        }
    }

    fn get(&self, name: &str) -> Option<(&ResidentIndexEntry, &[u8])> {
        let e = self.index.entries.get(name)?;
        let at = e.file_offset as usize;
        Some((e, &self.bytes[at..at + e.size_bytes as usize]))
    }

    /// One logical row, dequantized, whatever the dtype. Every layout here
    /// stores a row as a contiguous byte run, which is also why the transform
    /// below is a byte move and never a requantization.
    fn row(&self, e: &ResidentIndexEntry, bytes: &[u8], row: usize) -> Vec<f32> {
        let cols = e.shape.1 as usize;
        let u16s = |b: &[u8]| -> Vec<u16> {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let groups = cols / 64;
        let at = |o: u64, n: usize| &self.bytes[o as usize..o as usize + n];
        match e.dtype {
            4 | 5 => {
                let rb = if e.dtype == 4 { cols / 2 } else { cols };
                let w = bytes[row * rb..(row + 1) * rb].to_vec();
                let s = u16s(at(e.scale_offset + (row * groups * 2) as u64, groups * 2));
                let b = u16s(at(e.bias_offset + (row * groups * 2) as u64, groups * 2));
                if e.dtype == 4 {
                    dequantize_int4_affine(
                        &Int4AffineRow {
                            packed: w,
                            scales: s,
                            biases: b,
                        },
                        cols,
                    )
                } else {
                    dequantize_int8_affine(
                        &Int8AffineRow {
                            packed: w,
                            scales: s,
                            biases: b,
                        },
                        cols,
                    )
                }
            }
            BF16 => u16s(&bytes[row * cols * 2..(row + 1) * cols * 2])
                .into_iter()
                .map(compute::bf16_to_f32)
                .collect(),
            6 => {
                let rb = cols / 32 * compute::Q8_0_BLOCK_BYTES;
                dequantize_q8_0(&bytes[row * rb..(row + 1) * rb], cols)
            }
            7 => {
                let rb = cols / 256 * compute::Q4_K_BLOCK_BYTES;
                dequantize_q4_k(&bytes[row * rb..(row + 1) * rb], cols)
            }
            8 => {
                let rb = cols / 256 * compute::Q6_K_BLOCK_BYTES;
                dequantize_q6_k(&bytes[row * rb..(row + 1) * rb], cols)
            }
            d => panic!("no dequant path for dtype {d}"),
        }
    }
}

/// Every tensor on the V-head axis, as `(canonical suffix, base, span)` in
/// units of the axis, plus whether that axis is the COLUMNS. The same table
/// `gguf_checkpoint.rs::v_head_axis` owns, restated here against install
/// bytes.
///
/// DELIBERATELY A SECOND EXPRESSION rather than a call into the walk: that
/// code takes a GGUF header and a `RangeSource`, and this file has neither.
/// The duplication is safe because every result is scored against the MLX
/// install, an independent third party -- if this arithmetic were wrong the
/// comparison rejects it rather than agreeing with a matching bug.
fn axes(arch: &ArchConfig) -> Vec<(&'static str, usize, usize, bool)> {
    let la = &arch.linear_attention;
    let v_at = 2 * la.num_k_heads as usize * la.key_head_dim as usize;
    let w = la.value_head_dim as usize;
    vec![
        ("linear_attn.A_log", 0, 1, false),
        ("linear_attn.dt_bias", 0, 1, false),
        ("linear_attn.in_proj_a.weight", 0, 1, false),
        ("linear_attn.in_proj_b.weight", 0, 1, false),
        ("linear_attn.conv1d.weight", v_at, w, false),
        ("linear_attn.in_proj_qkv.weight", v_at, w, false),
        ("linear_attn.in_proj_z.weight", 0, w, false),
        ("linear_attn.out_proj.weight", 0, w, true),
    ]
}

fn permute<T: Copy>(data: &mut [T], base: usize, span: usize, heads: usize) {
    let src = data.to_vec();
    for h in 0..heads {
        let to = if h < heads / 2 {
            2 * h
        } else {
            2 * (h - heads / 2) + 1
        };
        data[base + to * span..base + (to + 1) * span]
            .copy_from_slice(&src[base + h * span..base + (h + 1) * span]);
    }
}

/// The candidate bytes for one tensor, or `None` when the transform cannot
/// apply -- which is how an ALREADY-PATCHED `A_log` is recognized: its values
/// are logs by then, so some are positive and `ln(-x)` has nowhere to go.
fn candidate(
    e: &ResidentIndexEntry,
    bytes: &[u8],
    suffix: &str,
    (base, span, columns): (usize, usize, bool),
    heads: usize,
) -> Option<Vec<u8>> {
    let (rows, cols) = (e.shape.0 as usize, e.shape.1.max(1) as usize);
    if e.dtype == BF16 {
        assert!(!columns, "no BF16 tensor takes the column axis");
        let mut v: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        if suffix.ends_with("A_log") {
            for x in v.iter_mut() {
                if !x.is_finite() || *x >= 0.0 {
                    return None;
                }
                *x = (-*x).ln();
            }
        }
        permute(&mut v, base * cols, span * cols, heads);
        return Some(
            v.iter()
                .flat_map(|x| compute::f32_to_bf16(*x).to_le_bytes())
                .collect(),
        );
    }
    let mut out = bytes.to_vec();
    let row_bytes = bytes.len() / rows;
    if columns {
        let head_bytes = row_bytes * span / cols;
        assert_eq!(row_bytes * span % cols, 0, "a head is not a whole block");
        for r in 0..rows {
            permute(
                &mut out[r * row_bytes..(r + 1) * row_bytes],
                row_bytes * base / cols,
                head_bytes,
                heads,
            );
        }
    } else {
        permute(&mut out, base * row_bytes, span * row_bytes, heads);
    }
    Some(out)
}

/// How well a candidate agrees with the MLX install, as a mean |correlation|
/// over one representative row per V head.
///
/// A CORRELATION and not an equality, because the two installs hold different
/// quantizations of the same trained weights everywhere but the BF16 core.
/// Rows that are constant score 0.0 by `pearson`'s own contract and are
/// skipped rather than averaged in, which is AGENTS.md Gotcha 30: this
/// checkpoint really does carry all-zero rows.
fn agreement(
    mlx: &Weights,
    gguf: &Weights,
    name: &str,
    bytes: &[u8],
    (base, span, columns): (usize, usize, bool),
    heads: usize,
) -> f32 {
    let (me, mb) = mlx.get(name).expect("mlx tensor");
    let (ge, _) = gguf.get(name).expect("gguf tensor");
    // A rank-1 tensor has one VALUE per head, and a correlation over one
    // element is not a number. Correlate the whole vector instead, which is
    // what makes a permutation of it visible at all.
    if ge.shape.1 <= 1 {
        let f = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(2)
                .map(|c| compute::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        return pearson(&f(bytes), &f(mb)).abs();
    }
    let width = ge.shape.1.max(1) as usize / heads.max(1);
    let (mut sum, mut n) = (0.0f32, 0usize);
    for h in 0..heads {
        let (a, b) = if columns {
            // One row carries every head; compare the head's column slice.
            let (g, m) = (gguf.row(ge, bytes, 0), mlx.row(me, mb, 0));
            (
                g[h * width..(h + 1) * width].to_vec(),
                m[h * width..(h + 1) * width].to_vec(),
            )
        } else {
            let r = base + h * span;
            (gguf.row(ge, bytes, r), mlx.row(me, mb, r))
        };
        let c = pearson(&a, &b).abs();
        if c != 0.0 {
            sum += c;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

#[test]
#[ignore = "needs both Qwen installs"]
fn the_convention_holds_on_every_layer_and_optionally_patches_the_install() {
    let mlx_dir = install("MREFRUST_QWEN36_INSTALL_DIR");
    let gguf_dir = install("MREFRUST_QWEN36_GGUF_INSTALL_DIR");
    let (mlx, gguf) = (Weights::open(&mlx_dir), Weights::open(&gguf_dir));
    let arch = model_io::known_architecture(model_io::ModelFamily::Qwen36);
    let heads = arch.linear_attention.num_v_heads as usize;
    let table = axes(&arch);

    let patch = std::env::var_os("MREFRUST_QWEN_PATCH").is_some();
    println!(
        "{} layers, {heads} V heads, {} tensors on the axis; mode: {}",
        arch.num_layers,
        table.len(),
        if patch { "PATCH" } else { "verify only" }
    );

    // Collected and written in one pass at the end, so a mid-run panic cannot
    // leave the install half-transformed.
    let mut writes: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut norms_worst = 0.0f32;
    let mut per_tensor: Vec<(&str, f32, f32, usize)> = table
        .iter()
        .map(|(s, _, _, _)| (*s, 0.0f32, 0.0f32, 0usize))
        .collect();

    for layer in 0..arch.num_layers {
        let at = |suffix: &str| format!("language_model.model.layers.{layer}.{suffix}");

        // The control, and the answer to "was layer 0 special?". These are
        // the tensors the probe proved BIT-IDENTICAL, so anything but 0.0
        // means the convention gap is wider than the axis.
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "linear_attn.norm.weight",
        ] {
            let name = at(suffix);
            let (Some((_, a)), Some((_, b))) = (mlx.get(&name), gguf.get(&name)) else {
                continue;
            };
            for (x, y) in a.iter().zip(b) {
                norms_worst = norms_worst.max(if x == y { 0.0 } else { 1.0 });
            }
        }

        for (i, (suffix, base, span, columns)) in table.iter().enumerate() {
            let name = at(suffix);
            let (Some((ge, gb)), Some(_)) = (gguf.get(&name), mlx.get(&name)) else {
                continue;
            };
            let shape = (*base, *span, *columns);
            let Some(cand) = candidate(ge, gb, suffix, shape, heads) else {
                continue; // already transformed; `A_log` is the only such case
            };
            let asis = agreement(&mlx, &gguf, &name, gb, shape, heads);
            let moved = agreement(&mlx, &gguf, &name, &cand, shape, heads);
            per_tensor[i].1 += asis;
            per_tensor[i].2 += moved;
            per_tensor[i].3 += 1;
            if patch && moved > asis {
                writes.push((ge.file_offset, cand));
            }
        }
    }

    println!(
        "  norms across all layers: {}",
        if norms_worst == 0.0 {
            "bit-identical"
        } else {
            "DIFFER"
        }
    );
    println!("  mean |corr| against the MLX install, per tensor:");
    for (suffix, asis, moved, n) in &per_tensor {
        if *n == 0 {
            println!("    {suffix}: already in the MLX convention, or absent");
            continue;
        }
        let (a, m) = (asis / *n as f32, moved / *n as f32);
        let verdict = if a > 0.99 {
            "already de-interleaved on disk"
        } else {
            "needs the de-interleave"
        };
        println!("    {suffix}: as-is {a:.5}   de-interleaved {m:.5}   ({n} layers, {verdict})");
        // One of the two readings has to be the MLX convention. Which one
        // depends on whether this install has been patched already, so the
        // assertion is on the DISJUNCTION rather than on the transform always
        // winning -- otherwise a second run of the patcher reddens.
        assert!(
            a > 0.99 || m > a,
            "{suffix}: neither the bytes on disk nor the de-interleave agree with the MLX install"
        );
    }
    assert_eq!(norms_worst, 0.0, "the norms must stay bit-identical");

    if !patch {
        println!("verify only; set MREFRUST_QWEN_PATCH=1 to write these back");
        return;
    }
    let path = gguf_dir.join("model_weights.bin");
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("open for patch");
    for (offset, bytes) in &writes {
        f.seek(SeekFrom::Start(*offset)).expect("seek");
        f.write_all(bytes).expect("write");
    }
    f.sync_all().expect("sync");
    println!("patched {} tensors in {}", writes.len(), path.display());
}
