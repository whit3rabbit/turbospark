//! THROWAWAY probe: does the V-head convention reach the QUANTIZED
//! gated-DeltaNet tensors too?
//!
//! `gguf_qwen_core_probe.rs` can only see BF16 resident tensors, so it found
//! the three F32-sourced ones (`A_log`, `dt_bias`, `conv1d.weight`) and was
//! structurally blind to the rest of the layer. After those three were fixed
//! the real Qwen GGUF install still generated gibberish, which says the
//! convention is a property of the V-HEAD AXIS rather than of those three
//! tensors, and five more tensors carry that axis:
//!
//! ```text
//! linear_attn.in_proj_qkv.weight   rows [q | k | v], v region moves
//! linear_attn.in_proj_z.weight     rows = value_dim
//! linear_attn.in_proj_a.weight     rows = num_v_heads
//! linear_attn.in_proj_b.weight     rows = num_v_heads
//! linear_attn.out_proj.weight      COLUMNS = value_dim
//! ```
//!
//! `in_proj_a` is the cheapest decisive one: 32 rows, one per V head, so the
//! permutation can be RECOVERED rather than guessed by correlating every GGUF
//! row against every MLX row and reading the argmax. Two different
//! quantizations of the same trained weights, so this is a correlation and
//! never an equality (the discipline `gguf_q4_k_network.rs` established).
//!
//!   TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!   TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
//!     cargo test -p turbospark-repack --test gguf_qwen_quant_probe --release -- --ignored --nocapture

use std::path::{Path, PathBuf};

use compute::{
    dequantize_int4_affine, dequantize_int8_affine, dequantize_q4_k, dequantize_q6_k,
    dequantize_q8_0, pearson, Int4AffineRow, Int8AffineRow,
};
use model_io::{ResidentIndex, ResidentIndexEntry};

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

    fn entry(&self, name: &str) -> &ResidentIndexEntry {
        self.index
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("no resident entry {name}"))
    }

    fn slice(&self, at: u64, len: usize) -> &[u8] {
        &self.bytes[at as usize..at as usize + len]
    }

    /// One logical output row, dequantized, whatever the tensor's dtype.
    ///
    /// Every dtype here stores a row as a CONTIGUOUS byte run (affine packs
    /// nibbles or bytes with planar scale/bias companions; a GGUF block tiles
    /// along the fastest-varying dim), which is also why a row permutation is
    /// a byte-range permutation and needs no dequantization to apply.
    fn row(&self, name: &str, row: usize) -> Vec<f32> {
        let e = self.entry(name);
        let cols = e.shape.1 as usize;
        let u16s = |b: &[u8]| -> Vec<u16> {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let groups = cols / 64;
        match e.dtype {
            // INT4 / INT8 affine, the MLX install's shapes.
            4 | 5 => {
                let (row_bytes, packed) = if e.dtype == 4 {
                    (cols / 2, true)
                } else {
                    (cols, false)
                };
                let w = self.slice(e.file_offset + (row * row_bytes) as u64, row_bytes);
                let s = u16s(self.slice(e.scale_offset + (row * groups * 2) as u64, groups * 2));
                let b = u16s(self.slice(e.bias_offset + (row * groups * 2) as u64, groups * 2));
                if packed {
                    dequantize_int4_affine(
                        &Int4AffineRow {
                            packed: w.to_vec(),
                            scales: s,
                            biases: b,
                        },
                        cols,
                    )
                } else {
                    dequantize_int8_affine(
                        &Int8AffineRow {
                            packed: w.to_vec(),
                            scales: s,
                            biases: b,
                        },
                        cols,
                    )
                }
            }
            1 => u16s(self.slice(e.file_offset + (row * cols * 2) as u64, cols * 2))
                .into_iter()
                .map(compute::bf16_to_f32)
                .collect(),
            6 => {
                let rb = cols / 32 * compute::Q8_0_BLOCK_BYTES;
                dequantize_q8_0(self.slice(e.file_offset + (row * rb) as u64, rb), cols)
            }
            7 => {
                let rb = cols / 256 * compute::Q4_K_BLOCK_BYTES;
                dequantize_q4_k(self.slice(e.file_offset + (row * rb) as u64, rb), cols)
            }
            8 => {
                let rb = cols / 256 * compute::Q6_K_BLOCK_BYTES;
                dequantize_q6_k(self.slice(e.file_offset + (row * rb) as u64, rb), cols)
            }
            d => panic!("{name}: no dequant path for dtype {d}"),
        }
    }
}

/// The V-head map settled on the F32 tensors: GGUF head `h` is MLX head
/// `MAP(h)`.
fn expected(h: usize, heads: usize) -> usize {
    if h < heads / 2 {
        2 * h
    } else {
        2 * (h - heads / 2) + 1
    }
}

#[test]
#[ignore = "needs both Qwen installs"]
fn which_quantized_gdn_tensors_carry_the_v_head_permutation() {
    let mlx = Weights::open(&install("TURBOSPARK_QWEN36_INSTALL_DIR"));
    let gguf = Weights::open(&install("TURBOSPARK_QWEN36_GGUF_INSTALL_DIR"));
    let arch = model_io::known_architecture(model_io::ModelFamily::QwenGdnMoe);
    let la = &arch.linear_attention;
    let heads = la.num_v_heads as usize;

    // `in_proj_a` and `in_proj_b`: one output row per V head, so the map is
    // recoverable outright.
    for suffix in [
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
    ] {
        let name = format!("language_model.model.layers.0.{suffix}");
        let (me, ge) = (mlx.entry(&name), gguf.entry(&name));
        println!(
            "{suffix}\n  mlx dtype {} shape {:?} | gguf dtype {} shape {:?}",
            me.dtype, me.shape, ge.dtype, ge.shape
        );
        if me.shape.0 as usize != heads {
            println!("  not one row per V head, skipped");
            continue;
        }
        let mlx_rows: Vec<Vec<f32>> = (0..heads).map(|r| mlx.row(&name, r)).collect();
        let mut map = Vec::with_capacity(heads);
        let mut diag = 0.0f32;
        for h in 0..heads {
            let g = gguf.row(&name, h);
            let (mut best, mut best_at) = (f32::MIN, usize::MAX);
            for (j, m) in mlx_rows.iter().enumerate() {
                let c = pearson(&g, m).abs();
                if c > best {
                    best = c;
                    best_at = j;
                }
            }
            diag = diag.max(pearson(&g, &mlx_rows[h]).abs());
            map.push(best_at);
        }
        let identity = (0..heads).collect::<Vec<_>>();
        let candidate = (0..heads).map(|h| expected(h, heads)).collect::<Vec<_>>();
        println!("  recovered map: {map:?}");
        println!(
            "  identity? {}   V-head de-interleave? {}   best same-index corr {diag:.5}",
            map == identity,
            map == candidate
        );
    }

    // The three big ones. Their V axis is thousands of rows (or columns), so
    // instead of recovering a map this compares ONE representative row per
    // head against the two candidates: same index, or de-interleaved.
    for (suffix, base_rows, span_rows) in [
        (
            "linear_attn.in_proj_qkv.weight",
            2 * la.num_k_heads as usize * la.key_head_dim as usize,
            la.value_head_dim as usize,
        ),
        (
            "linear_attn.in_proj_z.weight",
            0,
            la.value_head_dim as usize,
        ),
    ] {
        let name = format!("language_model.model.layers.0.{suffix}");
        let (me, ge) = (mlx.entry(&name), gguf.entry(&name));
        println!(
            "{suffix}\n  mlx dtype {} shape {:?} | gguf dtype {} shape {:?}",
            me.dtype, me.shape, ge.dtype, ge.shape
        );
        let (mut same, mut moved) = (0.0f32, 0.0f32);
        for h in 0..heads {
            // The head's first row, which is enough to identify a head.
            let g = gguf.row(&name, base_rows + h * span_rows);
            same += pearson(&g, &mlx.row(&name, base_rows + h * span_rows)).abs();
            moved += pearson(
                &g,
                &mlx.row(&name, base_rows + expected(h, heads) * span_rows),
            )
            .abs();
        }
        let n = heads as f32;
        println!(
            "  mean |corr| same index {:.5}   de-interleaved {:.5}",
            same / n,
            moved / n
        );
    }

    // `out_proj` takes the V axis on its COLUMNS, so one output row carries
    // every head and the comparison is per column-slice.
    let name = "language_model.model.layers.0.linear_attn.out_proj.weight".to_string();
    let (me, ge) = (mlx.entry(&name), gguf.entry(&name));
    println!(
        "linear_attn.out_proj.weight\n  mlx dtype {} shape {:?} | gguf dtype {} shape {:?}",
        me.dtype, me.shape, ge.dtype, ge.shape
    );
    let width = la.value_head_dim as usize;
    let (mut same, mut moved) = (0.0f32, 0.0f32);
    const ROWS: usize = 8;
    for r in 0..ROWS {
        let (g, m) = (gguf.row(&name, r), mlx.row(&name, r));
        for h in 0..heads {
            let gs = &g[h * width..(h + 1) * width];
            same += pearson(gs, &m[h * width..(h + 1) * width]).abs();
            let e = expected(h, heads);
            moved += pearson(gs, &m[e * width..(e + 1) * width]).abs();
        }
    }
    let n = (ROWS * heads) as f32;
    println!(
        "  mean |corr| same index {:.5}   de-interleaved {:.5}",
        same / n,
        moved / n
    );
}
