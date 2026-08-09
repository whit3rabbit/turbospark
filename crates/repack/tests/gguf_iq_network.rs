//! Checks the IQ references against REAL published bytes (ROADMAP Phase S
//! step 2), the same way `gguf_q4_k_network.rs` checked Q4_K and
//! `gguf_fused_gate_network.rs` checked Q8_0.
//!
//! This is Phase S's decisive checkpoint, and it is deliberately the step
//! before any MSL is written. `crates/compute`'s unit tests hold the decoders
//! against ggml, which is a strong check of the LAYOUT, but both sides of it
//! come from one source: if ggml and this port agreed on a wrong reading of
//! what a tensor MEANS, nothing there would notice. Correlating a
//! dequantized routed expert out of the real
//! `gemma-4-26B-A4B-it-UD-Q3_K_M.gguf` against the same expert in the
//! independently produced MLX-derived `.gturbo` install can: the two sides
//! share no code, no format and no input, and they are two different
//! quantizations of the same trained weights.
//!
//! Two questions at once, because the candidate puts a different type in
//! each half of a routed expert:
//!
//! - `ffn_gate_up_exps` is IQ3_XXS, the codebook type, and it is FUSED, so
//!   this also re-settles [`FUSED_GATE_FIRST`] for this file. That was
//!   measured on the Q8_0 build; the split here happens at whole output rows
//!   of an IQ3_XXS tensor, and which half is the gate is a property of the
//!   converter run, not of the format. Getting it backwards swaps two
//!   unrelated matrices in every expert and generates fluent, wrong text.
//! - `ffn_down_exps` is IQ4_NL.
//!
//! IQ4_XS is NOT covered here, and the reason is the file: it appears on
//! exactly one tensor (`blk.29.ffn_gate_up_exps.weight`), which the last case
//! reads for that reason.
//!
//! COSTS A FEW KB, NOT A DOWNLOAD. Blocks tile along the fastest-varying dim,
//! so one output row is a contiguous byte run: 2816 / 256 * 98 = 1078 bytes
//! for an IQ3_XXS gate row and 704 / 32 * 18 = 396 for an IQ4_NL down row.
//! The whole check is the header plus a handful of small ranged reads.
//!
//! AGENTS.md Gotcha 30 before reading a low number: this family of
//! checkpoints carries ALL-ZERO rows inside a routed expert, `pearson`
//! returns 0.0 on a constant input by design, and a pooled correlation over a
//! fixed range therefore divides a good result by the number of dead rows.
//! Every case here correlates PER ROW, selects the non-constant ones, and
//! asserts both sides agree on which those are.
//!
//! ```sh
//! TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
//!   cargo test -p turbospark-repack --test gguf_iq_network --release -- --ignored --nocapture
//! ```

use std::io::{Read, Seek, SeekFrom};

use compute::{
    dequantize_int4_affine, dequantize_iq3_xxs, dequantize_iq4_nl, dequantize_iq4_xs, pearson,
    Int4AffineRow, IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMS, IQ4_NL_BLOCK_BYTES,
    IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMS,
};
use turbospark_repack::{fetch_gguf_header, GgufHeader, HttpRangeSource, RangeSource};

/// The Phase S candidate. An unsloth "UD" (Unsloth Dynamic) imatrix build, so
/// its block types vary per tensor and its name says nothing about them: this
/// file called `Q3_K_M` contains no Q3_K at all.
const GEMMA4_UD_Q3_K_M: &str = "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf";

/// ggml type ids for the three types under test.
const GGML_IQ3_XXS: u32 = 18;
const GGML_IQ4_NL: u32 = 20;
const GGML_IQ4_XS: u32 = 23;

/// Consecutive output rows read from each side. Larger than the Q8_0 probe's
/// four because more of them are dead here: at 16 the Q4_K sibling found only
/// 5 usable, and this is the same base model's expert table.
const ROWS: usize = 16;

/// Usable (non-constant) rows needed before the result means anything.
const MIN_USABLE: usize = 4;

/// A decoded output row is `elems` values, which on disk is a CONTIGUOUS run
/// of `elems / block_elems * block_bytes` bytes, because every block layout in
/// the format tiles along the fastest-varying dimension. That is what makes
/// this test a few KB rather than a 12 GB download.
struct BlockKind {
    name: &'static str,
    ggml_type: u32,
    block_elems: usize,
    block_bytes: usize,
    dequant: fn(&[u8], usize) -> Vec<f32>,
}

const IQ3_XXS: BlockKind = BlockKind {
    name: "IQ3_XXS",
    ggml_type: GGML_IQ3_XXS,
    block_elems: IQ3_XXS_BLOCK_ELEMS,
    block_bytes: IQ3_XXS_BLOCK_BYTES,
    dequant: dequantize_iq3_xxs,
};

const IQ4_NL: BlockKind = BlockKind {
    name: "IQ4_NL",
    ggml_type: GGML_IQ4_NL,
    block_elems: IQ4_NL_BLOCK_ELEMS,
    block_bytes: IQ4_NL_BLOCK_BYTES,
    dequant: dequantize_iq4_nl,
};

const IQ4_XS: BlockKind = BlockKind {
    name: "IQ4_XS",
    ggml_type: GGML_IQ4_XS,
    block_elems: IQ4_XS_BLOCK_ELEMS,
    block_bytes: IQ4_XS_BLOCK_BYTES,
    dequant: dequantize_iq4_xs,
};

/// Read `count` consecutive output rows starting at logical row `first` of
/// expert 0, and dequantize each.
fn gguf_rows(
    source: &dyn RangeSource,
    kind: &BlockKind,
    tensor_start: u64,
    elems: usize,
    first: usize,
    count: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(
        elems % kind.block_elems,
        0,
        "{elems} is not a whole number of {}-element {} blocks",
        kind.block_elems,
        kind.name
    );
    let row_bytes = (elems / kind.block_elems * kind.block_bytes) as u64;
    let start = tensor_start + first as u64 * row_bytes;
    let bytes = source
        .read_range(start, start + count as u64 * row_bytes)
        .expect("ranged read of the routed tensor");
    (0..count)
        .map(|r| {
            let base = r * row_bytes as usize;
            (kind.dequant)(&bytes[base..base + row_bytes as usize], elems)
        })
        .collect()
}

/// Read `count` consecutive INT4-affine output rows of `role` from expert 0
/// of layer `layer` of the MLX-derived install. Verbatim in shape from
/// `gguf_q4_k_network.rs`, widened by a layer argument for the IQ4_XS case.
fn install_rows(
    dir: &std::path::Path,
    layer: usize,
    role: &str,
    elems: usize,
    first: usize,
    count: usize,
) -> Vec<Vec<f32>> {
    let layout = model_io::load_packed_experts_layout(
        dir,
        model_io::PACKED_EXPERTS_LAYOUT_DEFAULT_MAX_BYTES,
    )
    .expect("packed_experts/layout.json");
    let entry = layout.expert(layer, 0);
    let file = layout.layers[layer].file.clone();
    let mut f = std::fs::File::open(dir.join("packed_experts").join(&file)).expect("layer blob");

    let sub = |k: &str| {
        entry
            .sub_tensors
            .get(k)
            .unwrap_or_else(|| panic!("expert 0 has no sub-tensor {k}"))
    };
    let groups = elems / compute::quant::GROUP_SIZE;
    let packed_row = elems / 2;

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
                elems,
            )
        })
        .collect()
}

fn is_constant(row: &[f32]) -> bool {
    row.iter().all(|&v| v == row[0])
}

/// The rows both sides call non-constant, having first asserted they agree on
/// which those are.
///
/// That agreement is itself the check, and it is stronger than it looks: the
/// two readers index completely differently (a 1078-byte IQ3_XXS codebook run
/// against a 1408-byte INT4 run plus two scale planes), so they cannot land
/// on the same zero set by accident. Where they agree, the zeros are the
/// model. See AGENTS.md Gotcha 30 for the 45 minutes this cost the first time.
fn usable_rows(gguf: &[Vec<f32>], install: &[Vec<f32>]) -> Vec<usize> {
    let gguf_zero: Vec<usize> = (0..gguf.len()).filter(|&r| is_constant(&gguf[r])).collect();
    let install_zero: Vec<usize> = (0..install.len())
        .filter(|&r| is_constant(&install[r]))
        .collect();
    assert_eq!(
        gguf_zero, install_zero,
        "the two sides disagree about which rows are constant, which is a reader bug on one of them"
    );
    let usable: Vec<usize> = (0..gguf.len())
        .filter(|&r| !is_constant(&gguf[r]))
        .collect();
    assert!(
        usable.len() >= MIN_USABLE,
        "only {} of {} rows carry weights; the sample proves nothing",
        usable.len(),
        gguf.len()
    );
    println!(
        "  {} usable of {} rows (constant on BOTH sides: {gguf_zero:?})",
        usable.len(),
        gguf.len()
    );
    usable
}

fn mean_corr(a: &[Vec<f32>], b: &[Vec<f32>], usable: &[usize]) -> f32 {
    usable.iter().map(|&r| pearson(&a[r], &b[r])).sum::<f32>() / usable.len() as f32
}

/// Locate a tensor, assert its block type, and return its dims and absolute
/// start. Reading the type off the file rather than assuming it is the point
/// of a mixed checkpoint: `UD-Q3_K_M` puts a different one on layer 29.
fn tensor(header: &GgufHeader, name: &str, kind: &BlockKind) -> (Vec<u64>, u64) {
    let info = header
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("{name} is not in the tensor table"));
    assert_eq!(
        info.ggml_type, kind.ggml_type,
        "{name} is ggml type {}, not {}",
        info.ggml_type, kind.name
    );
    let (start, _) = header
        .absolute_range(name)
        .expect("tensor is in the table")
        .expect("tensor has a byte size");
    (info.dims.clone(), start)
}

fn install_dir() -> Option<std::path::PathBuf> {
    let Ok(dir) = std::env::var("TURBOSPARK_GEMMA4_INSTALL_DIR") else {
        println!("(skipped: TURBOSPARK_GEMMA4_INSTALL_DIR is unset)");
        return None;
    };
    Some(std::path::PathBuf::from(shellexpand_home(&dir)))
}

/// The codebook type, on the fused gate/up tensor, which settles the decoder
/// and the half-ordering in one measurement.
#[test]
#[ignore = "network: reads a few KB off a 12 GB remote checkpoint, plus the local install"]
fn a_real_iq3_xxs_expert_dequantizes_to_the_installed_one() {
    let Some(dir) = install_dir() else { return };
    let source = HttpRangeSource::new(GEMMA4_UD_Q3_K_M);
    let header = fetch_gguf_header(&source).expect("GGUF header");

    let name = "blk.0.ffn_gate_up_exps.weight";
    let (dims, start) = tensor(&header, name, &IQ3_XXS);
    assert_eq!(
        dims.len(),
        3,
        "expected [hidden, 2 * ffn, experts]: {dims:?}"
    );
    let (hidden, two_ffn) = (dims[0] as usize, dims[1] as usize);
    assert_eq!(two_ffn % 2, 0);
    let ffn = two_ffn / 2;
    println!("{name}: dims (as stored) {dims:?}, hidden {hidden}, ffn {ffn}, IQ3_XXS");

    // Half A is rows [0, ffn); half B is rows [ffn, 2 * ffn). Nothing in the
    // file distinguishes them.
    let half_a = gguf_rows(&source, &IQ3_XXS, start, hidden, 0, ROWS);
    let half_b = gguf_rows(&source, &IQ3_XXS, start, hidden, ffn, ROWS);
    let mlx_gate = install_rows(&dir, 0, "gate", hidden, 0, ROWS);
    let mlx_up = install_rows(&dir, 0, "up", hidden, 0, ROWS);

    let usable = usable_rows(&half_a, &mlx_gate);
    let a_gate = mean_corr(&half_a, &mlx_gate, &usable);
    let a_up = mean_corr(&half_a, &mlx_up, &usable);
    let b_gate = mean_corr(&half_b, &mlx_gate, &usable);
    let b_up = mean_corr(&half_b, &mlx_up, &usable);
    println!("  first half  vs install gate {a_gate:+.4}   vs install up {a_up:+.4}");
    println!("  second half vs install gate {b_gate:+.4}   vs install up {b_up:+.4}");

    // Two claims in one table, and the second is what makes the first mean
    // something. The decoder is right only if the MATCHED pairings are high;
    // a systematic decode error that still tracked a row's coarse structure
    // would raise all four, and the crossed pair is what refuses that.
    assert!(
        a_gate > 0.95 && b_up > 0.95,
        "IQ3_XXS dequant does not reproduce the installed weights: \
         first-half/gate {a_gate:+.4}, second-half/up {b_up:+.4}"
    );
    assert!(
        a_up.abs() < 0.2 && b_gate.abs() < 0.2,
        "the crossed pairings correlate too ({a_up:+.4}, {b_gate:+.4}), so this proves nothing"
    );
    // Derived from the numbers rather than hardcoded, so this says "the file
    // agrees with the constant" and not "the constant is true".
    let measured_gate_first = a_gate > b_gate;
    assert_eq!(
        measured_gate_first,
        turbospark_repack::FUSED_GATE_FIRST,
        "this file puts the gate in the {} half; FUSED_GATE_FIRST says the {}",
        if measured_gate_first {
            "first"
        } else {
            "second"
        },
        if turbospark_repack::FUSED_GATE_FIRST {
            "first"
        } else {
            "second"
        }
    );
}

/// IQ4_NL, on the down projection. A different block shape (32 elements, no
/// superblock, no per-sub-block scale) reading a different matrix.
#[test]
#[ignore = "network: reads a few KB off a 12 GB remote checkpoint, plus the local install"]
fn a_real_iq4_nl_expert_dequantizes_to_the_installed_one() {
    let Some(dir) = install_dir() else { return };
    let source = HttpRangeSource::new(GEMMA4_UD_Q3_K_M);
    let header = fetch_gguf_header(&source).expect("GGUF header");

    let name = "blk.0.ffn_down_exps.weight";
    let (dims, start) = tensor(&header, name, &IQ4_NL);
    assert_eq!(dims.len(), 3, "expected [ffn, hidden, experts]: {dims:?}");
    let ffn = dims[0] as usize;
    println!("{name}: dims (as stored) {dims:?}, row is {ffn} elements, IQ4_NL");

    let gguf = gguf_rows(&source, &IQ4_NL, start, ffn, 0, ROWS);
    let mlx_down = install_rows(&dir, 0, "down", ffn, 0, ROWS);
    // The control here is the GATE, read at the same row indices. It is a
    // different matrix of a different width, so it is compared over the
    // overlap only -- which is enough to say "not this one".
    let mlx_gate = install_rows(&dir, 0, "gate", ffn, 0, ROWS);

    let usable = usable_rows(&gguf, &mlx_down);
    let matched = mean_corr(&gguf, &mlx_down, &usable);
    let control = mean_corr(&gguf, &mlx_gate, &usable);
    println!("  GGUF IQ4_NL down vs install down {matched:+.4}");
    println!("  GGUF IQ4_NL down vs install gate {control:+.4}  (control)");

    assert!(
        matched > 0.95,
        "IQ4_NL dequant does not reproduce the installed weights: {matched:+.4}"
    );
    assert!(
        control.abs() < 0.2,
        "the control correlates too: {control:+.4}, so this proves nothing"
    );
}

/// IQ4_XS, which this file carries on exactly ONE tensor: layer 29's fused
/// gate/up. That is the whole reason this case exists, and the reason it is
/// separate rather than folded into the first: a mixed checkpoint's odd layer
/// is exactly the thing a probe anchored to `blk.0.` cannot see, and this
/// port has already paid for that once (the Qwen manifest slot probe, which
/// read its types off `blk.0.` on a model whose layer 0 has no `attn_q`).
#[test]
#[ignore = "network: reads a few KB off a 12 GB remote checkpoint, plus the local install"]
fn the_one_iq4_xs_tensor_dequantizes_to_the_installed_one() {
    let Some(dir) = install_dir() else { return };
    let source = HttpRangeSource::new(GEMMA4_UD_Q3_K_M);
    let header = fetch_gguf_header(&source).expect("GGUF header");

    // Asserted rather than searched for: if a future re-upload moves the odd
    // layer, this should say so loudly rather than quietly probing layer 0.
    let odd: Vec<&String> = header
        .tensors
        .iter()
        .filter(|(_, i)| i.ggml_type == GGML_IQ4_XS)
        .map(|(n, _)| n)
        .collect();
    assert_eq!(
        odd,
        vec!["blk.29.ffn_gate_up_exps.weight"],
        "the IQ4_XS tensor set moved; the layer-29 assumption needs re-checking"
    );

    let name = "blk.29.ffn_gate_up_exps.weight";
    let (dims, start) = tensor(&header, name, &IQ4_XS);
    let (hidden, ffn) = (dims[0] as usize, dims[1] as usize / 2);
    println!("{name}: dims (as stored) {dims:?}, hidden {hidden}, ffn {ffn}, IQ4_XS");

    let half_a = gguf_rows(&source, &IQ4_XS, start, hidden, 0, ROWS);
    let mlx_gate = install_rows(&dir, 29, "gate", hidden, 0, ROWS);
    let mlx_up = install_rows(&dir, 29, "up", hidden, 0, ROWS);

    let usable = usable_rows(&half_a, &mlx_gate);
    let matched = mean_corr(&half_a, &mlx_gate, &usable);
    let control = mean_corr(&half_a, &mlx_up, &usable);
    println!("  first half vs install gate {matched:+.4}");
    println!("  first half vs install up   {control:+.4}  (control)");

    assert!(
        matched > 0.95,
        "IQ4_XS dequant does not reproduce the installed weights: {matched:+.4}"
    );
    assert!(
        control.abs() < 0.2,
        "the control correlates too: {control:+.4}, so this proves nothing"
    );
}

fn shellexpand_home(s: &str) -> String {
    match s.strip_prefix("~/") {
        Some(rest) => format!("{}/{rest}", std::env::var("HOME").unwrap_or_default()),
        None => s.to_string(),
    }
}
