//! Checks the Q4_K reference against REAL published bytes (ROADMAP Phase G
//! Stage 2, item 6), the same way `gguf_fused_gate_network.rs` checked Q8_0.
//!
//! Why this exists on top of the unit tests: a decoder and the fixture
//! quantizer that feeds it are written from one mental model, so they can
//! agree while both are wrong about the format. Nothing in `crates/compute`
//! can catch that. Correlating a dequantized Q4_K expert row out of the real
//! `Qwen3.6-35B-A3B-Q4_K_M.gguf` against the SAME row in the independently
//! produced MLX-derived `.gturbo` install can: the two sides share no code
//! and no input, and a sign, scale, sub-scale-unpacking or nibble-ordering
//! error cannot correlate highly with an INT4-affine repack of the same
//! trained weights.
//!
//! It is a correlation and not an equality on purpose. Q4_K here against
//! INT4-affine there are two different quantizations, so they agree in
//! direction and not to the bit.
//!
//! COSTS A FEW KB, NOT A DOWNLOAD. Q4_K superblocks tile along the
//! fastest-varying dim, which is `hidden`, so one output row is
//! `hidden / 256 * 144` CONTIGUOUS bytes: the whole check is the header plus
//! two small ranged reads.
//!
//! ```sh
//! TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
//!   cargo test -p turbospark-repack --test gguf_q4_k_network --release -- --ignored --nocapture
//! ```

use std::io::{Read, Seek, SeekFrom};

use compute::{dequantize_int4_affine, dequantize_q4_k, pearson, Int4AffineRow, Q4_K_BLOCK_BYTES};
use turbospark_repack::{fetch_gguf_header, HttpRangeSource, RangeSource};

const QWEN36_Q4_K_M: &str =
    "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf";

/// ggml type id for Q4_K.
const GGML_Q4_K: u32 = 12;

/// Consecutive output rows read from each side. Only some are usable: this
/// checkpoint carries ALL-ZERO rows inside a routed expert, and both sides
/// agree exactly on which (measured 2026-08-08: rows 0, 8, 255, 256 and 300
/// of layer 0 expert 0's gate are real, rows 1, 2, 3, 16, 64, 128 and 511 are
/// zero on the GGUF side AND on the install side). Two readers doing
/// completely different arithmetic cannot agree on a zero set by accident, so
/// the zeros are the model rather than a bug in either. `pearson` returns 0.0
/// on a constant input by design, so they are skipped rather than averaged in.
const ROWS: usize = 16;

/// Usable (non-constant) rows needed before the result means anything.
const MIN_USABLE: usize = 4;

/// Read `count` consecutive Q4_K output rows starting at logical row `first`
/// of expert 0, and dequantize each.
fn gguf_rows(
    source: &dyn RangeSource,
    tensor_start: u64,
    hidden: usize,
    first: usize,
    count: usize,
) -> Vec<Vec<f32>> {
    let row_bytes = (hidden / 256 * Q4_K_BLOCK_BYTES) as u64;
    let start = tensor_start + first as u64 * row_bytes;
    let bytes = source
        .read_range(start, start + count as u64 * row_bytes)
        .expect("ranged read of the routed tensor");
    (0..count)
        .map(|r| {
            let base = r * row_bytes as usize;
            dequantize_q4_k(&bytes[base..base + row_bytes as usize], hidden)
        })
        .collect()
}

/// Read `count` consecutive INT4-affine output rows of `role` ("gate" or
/// "up") from expert 0 of layer 0 of the MLX-derived install.
fn install_rows(
    dir: &std::path::Path,
    role: &str,
    hidden: usize,
    first: usize,
    count: usize,
) -> Vec<Vec<f32>> {
    let layout = model_io::load_packed_experts_layout(
        dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("packed_experts/layout.json");
    let entry = layout.expert(0, 0);
    let file = layout.layers[0].file.clone();
    let mut f = std::fs::File::open(dir.join("packed_experts").join(&file)).expect("layer blob");

    let sub = |k: &str| {
        entry
            .sub_tensors
            .get(k)
            .unwrap_or_else(|| panic!("expert 0 has no sub-tensor {k}"))
    };
    let groups = hidden / compute::quant::GROUP_SIZE;
    let packed_row = hidden / 2;

    let mut read_at = |offset: u64, len: usize| {
        let mut buf = vec![0u8; len];
        f.seek(SeekFrom::Start(entry.offset + offset)).unwrap();
        f.read_exact(&mut buf).unwrap();
        buf
    };

    (first..first + count)
        .map(|r| {
            let packed = read_at(sub(role).offset + (r * packed_row) as u64, packed_row);
            let raw_s = read_at(
                sub(&format!("{role}_scales")).offset + (r * groups * 2) as u64,
                groups * 2,
            );
            let raw_b = read_at(
                sub(&format!("{role}_biases")).offset + (r * groups * 2) as u64,
                groups * 2,
            );
            let to_u16 = |v: Vec<u8>| -> Vec<u16> {
                v.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            };
            dequantize_int4_affine(
                &Int4AffineRow {
                    packed,
                    scales: to_u16(raw_s),
                    biases: to_u16(raw_b),
                },
                hidden,
            )
        })
        .collect()
}

/// Mean correlation over the rows named in `usable`.
fn mean_corr(a: &[Vec<f32>], b: &[Vec<f32>], usable: &[usize]) -> f32 {
    usable.iter().map(|&r| pearson(&a[r], &b[r])).sum::<f32>() / usable.len() as f32
}

fn is_constant(row: &[f32]) -> bool {
    row.iter().all(|&v| v == row[0])
}

#[test]
#[ignore = "network: reads a few KB off a 20 GB remote checkpoint, plus the local install"]
fn a_real_q4_k_expert_dequantizes_to_the_installed_one() {
    let Ok(dir) = std::env::var("TURBOSPARK_QWEN36_INSTALL_DIR") else {
        println!("(skipped: TURBOSPARK_QWEN36_INSTALL_DIR is unset)");
        return;
    };
    let dir = std::path::PathBuf::from(shellexpand_home(&dir));

    let source = HttpRangeSource::new(QWEN36_Q4_K_M);
    let header = fetch_gguf_header(&source).expect("GGUF header");

    // Qwen does NOT fuse gate and up (that is Gemma's trick), so the routed
    // gate is a tensor of its own. Q4_K_M mixes types per tensor, so which
    // one is Q4_K is read off the file rather than assumed: whichever of
    // gate/up is Q4_K is the one measured, and the other role is the control.
    let (name, role, control) = ["gate", "up"]
        .iter()
        .find_map(|r| {
            let n = format!("blk.0.ffn_{r}_exps.weight");
            let info = header.tensors.get(&n)?;
            (info.ggml_type == GGML_Q4_K).then(|| {
                let other = if *r == "gate" { "up" } else { "gate" };
                (n, r.to_string(), other.to_string())
            })
        })
        .unwrap_or_else(|| {
            panic!(
                "no Q4_K routed gate/up tensor; routed tensors are {:?}",
                sample(&header)
            )
        });

    let info = &header.tensors[&name];
    let dims = info.dims.clone();
    assert_eq!(
        dims.len(),
        3,
        "expected [hidden, ffn, experts], got {dims:?}"
    );
    let hidden = dims[0] as usize;
    assert_eq!(
        hidden % 256,
        0,
        "hidden {hidden} is not a whole number of Q4_K superblocks"
    );
    let (tensor_start, _) = header
        .absolute_range(&name)
        .expect("tensor is in the table")
        .expect("tensor has a byte size");

    println!("{name}: dims (as stored) {dims:?}, hidden {hidden}, ggml type Q4_K");

    let gguf = gguf_rows(&source, tensor_start, hidden, 0, ROWS);
    let matched = install_rows(&dir, &role, hidden, 0, ROWS);
    let mismatched = install_rows(&dir, &control, hidden, 0, ROWS);

    // Both sides must be non-constant for a correlation to say anything, and
    // agreeing on WHICH rows those are is itself a check: the two readers
    // index completely differently (a 1152-byte Q4_K run against a 1024-byte
    // INT4 run plus scale and bias planes), so a shared zero set is the data.
    let usable: Vec<usize> = (0..ROWS)
        .filter(|&r| !is_constant(&gguf[r]) && !is_constant(&matched[r]))
        .collect();
    let gguf_zero: Vec<usize> = (0..ROWS).filter(|&r| is_constant(&gguf[r])).collect();
    let install_zero: Vec<usize> = (0..ROWS).filter(|&r| is_constant(&matched[r])).collect();
    assert_eq!(
        gguf_zero, install_zero,
        "the two sides disagree about which rows are constant, which is a reader bug on one of them"
    );
    assert!(
        usable.len() >= MIN_USABLE,
        "only {} of {ROWS} rows carry weights; the sample proves nothing",
        usable.len()
    );

    let c_matched = mean_corr(&gguf, &matched, &usable);
    let c_mismatched = mean_corr(&gguf, &mismatched, &usable);
    println!(
        "{} usable of {ROWS} rows of layer 0 expert 0 (constant rows on both sides: {:?})",
        usable.len(),
        gguf_zero
    );
    println!("  GGUF Q4_K {role} vs install {role} {c_matched:+.4}");
    println!("  GGUF Q4_K {role} vs install {control} {c_mismatched:+.4}  (control)");

    // The control is not decoration. A dequantizer with a systematic error
    // could still track the row's coarse structure, so the claim is that the
    // MATCHED pairing is high AND the unrelated matrix is not, which no
    // common-mode error produces.
    assert!(
        c_matched > 0.95,
        "Q4_K dequant does not reproduce the installed weights: {c_matched:+.4}"
    );
    assert!(
        c_mismatched.abs() < 0.2,
        "the control correlates too: {c_mismatched:+.4}, so this proves nothing"
    );
}

fn sample(header: &turbospark_repack::GgufHeader) -> Vec<&String> {
    header
        .tensors
        .keys()
        .filter(|n| n.contains("exps"))
        .take(6)
        .collect()
}

fn shellexpand_home(s: &str) -> String {
    match s.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => s.to_string(),
    }
}
