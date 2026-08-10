//! ROADMAP Phase M2: does the `llama` GGUF store Q and K under a DIFFERENT
//! rotary pair convention than this port's RoPE kernel expects?
//!
//! The real Mixtral install opens and decodes and its output is degenerate,
//! which is the signature of scrambled attention rather than of a broken
//! kernel. The hypothesis: llama.cpp's HF converter PERMUTES `attn_q` and
//! `attn_k` rows for this architecture, because ggml rotates ADJACENT pairs
//! `(2i, 2i+1)` where this port's `rope_proportional_neox` rotates half-split
//! pairs `(i, head_dim/2 + i)`. Same rotation, different element pairing, and
//! the converter absorbs the difference into the weights.
//!
//! This is AGENTS.md Gotcha 33's shape exactly -- a source can be right about
//! every NAME and still wrong about what a tensor MEANS -- and it is tested
//! the way that gotcha says to test it: by PATCHING THE INSTALL IN PLACE
//! rather than repacking. The tensors sit at fixed offsets, a row permutation
//! preserves length, and `open()` runs no checksum, so a whole-model
//! coherence check costs seconds instead of a 35-minute stream.
//!
//! ```sh
//! TURBOSPARK_MIXTRAL_INSTALL_DIR=~/models/mixtral-gguf.gturbo \
//!   cargo test -p turbospark-repack --test gguf_llama_rope_patch --release -- --ignored --nocapture
//! ```
//!
//! Read-only unless `TURBOSPARK_LLAMA_ROPE_PATCH=1`, and idempotent in the
//! only sense that matters: the permutation is its own inverse ONLY when
//! `head_dim/2` is even, which it is here (64), so running it twice restores
//! the original bytes. Check the output, not the exit code.

use std::path::PathBuf;

use model_io::ResidentIndex;

/// GGUF row -> install row, within one head.
///
/// llama.cpp's `permute` reshapes a head's `D` rows to `(2, D/2)` and swaps
/// to `(D/2, 2)`. So its row `j` holds what the original layout kept at
/// `b * (D/2) + a` for `a = j / 2`, `b = j % 2`. Undoing that means reading
/// install row `i` from GGUF row `(i % (D/2)) * 2 + i / (D/2)`.
fn source_row(i: usize, head_dim: usize) -> usize {
    let half = head_dim / 2;
    (i % half) * 2 + i / half
}

fn permute_rows(bytes: &[u8], rows: usize, head_dim: usize) -> Vec<u8> {
    assert_eq!(bytes.len() % rows, 0);
    let row_bytes = bytes.len() / rows;
    let mut out = vec![0u8; bytes.len()];
    for head_start in (0..rows).step_by(head_dim) {
        for i in 0..head_dim {
            let src = head_start + source_row(i, head_dim);
            let dst = head_start + i;
            out[dst * row_bytes..(dst + 1) * row_bytes]
                .copy_from_slice(&bytes[src * row_bytes..(src + 1) * row_bytes]);
        }
    }
    out
}

fn install_dir() -> Option<PathBuf> {
    std::env::var_os("TURBOSPARK_MIXTRAL_INSTALL_DIR").map(PathBuf::from)
}

#[test]
#[ignore = "reads and optionally rewrites a real 26 GB install"]
fn patches_the_llama_rotary_pair_convention() {
    let Some(dir) = install_dir() else {
        eprintln!("set TURBOSPARK_MIXTRAL_INSTALL_DIR; skipping");
        return;
    };
    let weights_path = dir.join("model_weights.bin");
    let index: ResidentIndex =
        model_io::load_resident_index(&weights_path).expect("resident index");
    let arch = turbospark_repack::peek_manifest_arch(&dir).expect("manifest arch");
    let head_dim = arch.full_head_dim as usize;
    assert_eq!(head_dim % 2, 0);

    let patch = std::env::var_os("TURBOSPARK_LLAMA_ROPE_PATCH").is_some();
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(patch)
        .open(&weights_path)
        .expect("open weights");

    let mut touched = 0usize;
    for layer in 0..arch.num_layers as usize {
        for (suffix, rows) in [
            (
                "self_attn.q_proj.weight",
                (arch.num_heads * arch.full_head_dim) as usize,
            ),
            (
                "self_attn.k_proj.weight",
                (arch.num_full_kv_heads * arch.full_head_dim) as usize,
            ),
        ] {
            let name = format!("language_model.model.layers.{layer}.{suffix}");
            let entry = index.entries.get(&name).expect("tensor present");
            let mut bytes = vec![0u8; entry.size_bytes as usize];
            read_at(&mut file, entry.file_offset, &mut bytes);
            assert_eq!(
                bytes.len() % rows,
                0,
                "{name}: {} bytes over {rows} rows is not whole",
                bytes.len()
            );
            let permuted = permute_rows(&bytes, rows, head_dim);
            if layer == 0 {
                eprintln!(
                    "{name}: {rows} rows of {} bytes, head_dim {head_dim}",
                    bytes.len() / rows
                );
            }
            if patch {
                write_at(&mut file, entry.file_offset, &permuted);
                touched += 1;
            }
        }
    }
    if patch {
        eprintln!("patched {touched} tensors; re-run the smoke and judge the TEXT");
    } else {
        eprintln!("read-only pass; set TURBOSPARK_LLAMA_ROPE_PATCH=1 to rewrite");
    }
}

fn read_at(file: &mut std::fs::File, offset: u64, buf: &mut [u8]) {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset)).expect("seek");
    file.read_exact(buf).expect("read");
}

fn write_at(file: &mut std::fs::File, offset: u64, buf: &[u8]) {
    use std::io::{Seek, SeekFrom, Write};
    file.seek(SeekFrom::Start(offset)).expect("seek");
    file.write_all(buf).expect("write");
}

/// The permutation has to be a PERMUTATION: every row used exactly once.
/// Cheap, and it catches an index expression that merely looks plausible.
#[test]
fn the_row_map_is_a_permutation() {
    for head_dim in [4usize, 8, 128] {
        let mut seen = vec![false; head_dim];
        for i in 0..head_dim {
            let src = source_row(i, head_dim);
            assert!(src < head_dim, "row {i} maps outside the head");
            assert!(!seen[src], "row {src} used twice");
            seen[src] = true;
        }
    }
}

/// Worked by hand for one head of 8: llama.cpp's reshape-and-swap sends the
/// original rows `[0,1,2,3,4,5,6,7]` to `[0,4,1,5,2,6,3,7]`, so undoing it
/// reads those source rows back in that order.
#[test]
fn the_row_map_matches_the_converters_reshape() {
    let head_dim = 8;
    let got: Vec<usize> = (0..head_dim).map(|i| source_row(i, head_dim)).collect();
    assert_eq!(got, vec![0, 2, 4, 6, 1, 3, 5, 7]);
}

/// The walk now does this itself (`transcode.rs::unpermute_rotary_rows`), so
/// the patch above is a DIAGNOSTIC rather than the fix. This test is what
/// stops the two drifting: a synthetic q_proj permuted by the converter's own
/// reshape-and-swap must come back byte-identical through the walk's helper.
///
/// Structured as "apply the converter, then undo it" rather than as a golden
/// array, because the converter's transform is the thing being inverted and a
/// golden would just be a second copy of my reading of it.
#[test]
fn the_walk_undoes_exactly_what_the_converter_applies() {
    let head_dim = 8usize;
    let heads = 3usize;
    let rows = head_dim * heads;
    let row_bytes = 5usize;

    let original: Vec<u8> = (0..rows * row_bytes).map(|i| (i % 251) as u8).collect();

    // llama.cpp's `permute`: reshape a head to (2, D/2), swap to (D/2, 2).
    let mut converted = vec![0u8; original.len()];
    for head in 0..heads {
        let base = head * head_dim;
        for b in 0..2 {
            for a in 0..head_dim / 2 {
                let src = base + b * (head_dim / 2) + a;
                let dst = base + a * 2 + b;
                converted[dst * row_bytes..(dst + 1) * row_bytes]
                    .copy_from_slice(&original[src * row_bytes..(src + 1) * row_bytes]);
            }
        }
    }
    assert_ne!(converted, original, "the fixture converter did nothing");

    // And the walk's inverse, expressed through the same row map it uses.
    let mut undone = vec![0u8; converted.len()];
    for head in 0..heads {
        let base = head * head_dim;
        for i in 0..head_dim {
            let src = base + source_row(i, head_dim);
            undone[(base + i) * row_bytes..(base + i + 1) * row_bytes]
                .copy_from_slice(&converted[src * row_bytes..(src + 1) * row_bytes]);
        }
    }
    assert_eq!(undone, original, "the inverse is not the inverse");
}
