//! Settles [`FUSED_GATE_FIRST`] against the real files (ROADMAP Phase G
//! Stage 2, first use of the Q8_0 reference).
//!
//! Gemma 4's routed gate and up arrive fused in one GGUF tensor
//! (`ffn_gate_up_exps`, `[hidden, 2 * ffn, experts]`) where MLX keeps them
//! apart. Both halves have identical shapes and nothing in the file
//! distinguishes them, so Stage 1 could only assume an order. Getting it
//! backwards swaps two unrelated matrices inside every routed expert, which
//! is not a crash and not a NaN: the model keeps generating fluent text that
//! is simply wrong. That is why this runs BEFORE any generated output is
//! judged by eye, and why it is an assertion rather than a printout.
//!
//! The measurement is a correlation, not an equality. The two sides are
//! different quantizations (Q8_0 here, INT4-affine there) of the same trained
//! weights, so they agree in direction and not to the bit. Gate and up are
//! unrelated matrices, so the separation is not subtle.
//!
//! COSTS A FEW KB, NOT A DOWNLOAD. One output row is `hidden` elements, which
//! at Q8_0 is `hidden / 32 * 34` contiguous bytes, so the whole check is the
//! header plus two small ranged reads. There is no need to pull the 27 GB
//! file to answer this question.
//!
//! ```sh
//! MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!   cargo test -p mrefrust-repack --test gguf_fused_gate_network --release -- --ignored --nocapture
//! ```

use std::io::{Read, Seek, SeekFrom};

use compute::{dequantize_int4_affine, dequantize_q8_0, pearson, Int4AffineRow, Q8_0_BLOCK_BYTES};
use mrefrust_repack::{fetch_gguf_header, HttpRangeSource, RangeSource, FUSED_GATE_FIRST};

const GEMMA4_Q8_0: &str = "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q8_0.gguf";

/// Output rows sampled from each half. One row would settle it; four makes a
/// single unlucky row unable to decide the question on its own.
const ROWS: usize = 4;

/// Read `count` consecutive Q8_0 output rows starting at logical row `first`
/// of expert 0, and dequantize each. Rows are contiguous byte runs because
/// blocks tile along the fastest-varying dim, which is `hidden`.
fn gguf_rows(
    source: &dyn RangeSource,
    tensor_start: u64,
    hidden: usize,
    first: usize,
    count: usize,
) -> Vec<Vec<f32>> {
    let row_bytes = (hidden / 32 * Q8_0_BLOCK_BYTES) as u64;
    let start = tensor_start + first as u64 * row_bytes;
    let bytes = source
        .read_range(start, start + count as u64 * row_bytes)
        .expect("ranged read of the fused tensor");
    (0..count)
        .map(|r| {
            let base = r * row_bytes as usize;
            dequantize_q8_0(&bytes[base..base + row_bytes as usize], hidden)
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

/// Mean correlation of two equally-ordered row sets.
fn mean_corr(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().zip(b).map(|(x, y)| pearson(x, y)).sum::<f32>() / a.len() as f32
}

#[test]
#[ignore = "network: reads a few KB off a 27 GB remote checkpoint, plus the local install"]
fn the_first_half_of_the_fused_tensor_is_the_gate() {
    let Ok(dir) = std::env::var("MREFRUST_GEMMA4_INSTALL_DIR") else {
        println!("(skipped: MREFRUST_GEMMA4_INSTALL_DIR is unset)");
        return;
    };
    let dir = std::path::PathBuf::from(shellexpand_home(&dir));

    let source = HttpRangeSource::new(GEMMA4_Q8_0);
    let header = fetch_gguf_header(&source).expect("GGUF header");

    let name = "blk.0.ffn_gate_up_exps.weight";
    let info = header
        .tensors
        .get(name)
        .or_else(|| header.tensors.get("blk.0.ffn_gate_up_exps"))
        .unwrap_or_else(|| panic!("no fused routed tensor; names are {:?}", sample(&header)));
    let dims = info.dims.clone();
    assert_eq!(
        dims.len(),
        3,
        "expected [hidden, 2 * ffn, experts], got {dims:?}"
    );
    let (hidden, two_ffn) = (dims[0] as usize, dims[1] as usize);
    assert_eq!(two_ffn % 2, 0, "fused dim {two_ffn} is not even");
    let ffn = two_ffn / 2;
    let (tensor_start, _) = header
        .absolute_range(&info_name(&header, name))
        .expect("tensor is in the table")
        .expect("tensor has a byte size");

    println!("fused tensor dims (as stored) {dims:?} -> hidden {hidden}, ffn {ffn}");

    // Half A is rows [0, ffn); half B is rows [ffn, 2 * ffn).
    let half_a = gguf_rows(&source, tensor_start, hidden, 0, ROWS);
    let half_b = gguf_rows(&source, tensor_start, hidden, ffn, ROWS);
    let mlx_gate = install_rows(&dir, "gate", hidden, 0, ROWS);
    let mlx_up = install_rows(&dir, "up", hidden, 0, ROWS);

    let a_gate = mean_corr(&half_a, &mlx_gate);
    let a_up = mean_corr(&half_a, &mlx_up);
    let b_gate = mean_corr(&half_b, &mlx_gate);
    let b_up = mean_corr(&half_b, &mlx_up);

    println!("mean Pearson over {ROWS} rows of layer 0 expert 0:");
    println!("  first half  vs install gate {a_gate:+.4}   vs install up {a_up:+.4}");
    println!("  second half vs install gate {b_gate:+.4}   vs install up {b_up:+.4}");

    // The claim is not "a_gate is high" on its own -- a systematic bias in
    // either dequantizer could lift every pairing at once. It is that the
    // matching pairing beats the mismatched one by a wide margin on BOTH
    // halves, which no common-mode error produces.
    let gate_first = a_gate > a_up && b_up > b_gate;
    let up_first = a_up > a_gate && b_gate > b_up;
    assert!(
        gate_first || up_first,
        "the two halves do not agree on an order; correlations are ambiguous"
    );
    let margin = if gate_first {
        (a_gate - a_up).min(b_up - b_gate)
    } else {
        (a_up - a_gate).min(b_gate - b_up)
    };
    assert!(
        margin > 0.5,
        "separation is too weak to settle anything: margin {margin:.4}"
    );

    assert_eq!(
        FUSED_GATE_FIRST, gate_first,
        "FUSED_GATE_FIRST is {FUSED_GATE_FIRST} but the real files say gate_first = {gate_first}"
    );
    println!("FUSED_GATE_FIRST = {FUSED_GATE_FIRST} confirmed, margin {margin:.4}");
}

/// The tensor table's key, whichever of the two spellings the converter used.
fn info_name(header: &mrefrust_repack::GgufHeader, preferred: &str) -> String {
    if header.tensors.contains_key(preferred) {
        preferred.to_string()
    } else {
        "blk.0.ffn_gate_up_exps".to_string()
    }
}

fn sample(header: &mrefrust_repack::GgufHeader) -> Vec<&String> {
    header
        .tensors
        .keys()
        .filter(|n| n.contains("exps"))
        .take(6)
        .collect()
}

/// `~` in an env var is not expanded by the shell when the value is quoted,
/// and every other env-gated test in this workspace is run with a literal
/// `~/models/...`.
fn shellexpand_home(s: &str) -> String {
    match s.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => s.to_string(),
    }
}
