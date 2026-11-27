//! Stage A of the GGUF multiply decision (docs/designs/speedup-kern.md,
//! section 3): price the existing GGUF block GEMVs (Q4_K, Q8_0) on REAL
//! quantized bytes, against the INT4 GEMV at the same shapes.
//!
//! `gemv_format_price.rs` walks zero weights, and its rate is the number the
//! design doc's 0.30 to 0.36 price came from. This probe reads the bytes of
//! real GGUF tensors, binds them where the kernel reads them, and cycles
//! through every tensor in the manifest so the footprint is not one warm
//! matrix. A zero-weight control of the same block format runs beside the
//! real arm, so any effect of the data is measured here rather than assumed.
//!
//! Inputs are TSV manifests written from GGUF headers. Columns: file, tensor
//! name, ggml type, cols (ne0), rows (ne1), absolute byte offset, byte count.
//! Both variables must be set:
//!
//! ```sh
//! TURBOSPARK_BENCH_Q4K_MANIFEST=/path/manifest_q4k.tsv \
//! TURBOSPARK_BENCH_Q8_MANIFEST=/path/manifest_q8.tsv \
//!   cargo test -p turbospark-gpu --test gguf_real_bytes_price --release -- --ignored --nocapture
//! ```
//!
//! The RATIO against the INT4 arm is the deliverable. Absolute GB/s moves
//! with clocks and power state, so compare runs only within one process.
//! Every arm runs back to back, after a discarded warmup round.

#![cfg(target_os = "macos")]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use half::f16;
use turbospark_gpu::{
    autorelease_pool, encode_dequant_int4_gemv_resident, encode_dequant_q4_k_gemm_resident_choose,
    encode_dequant_q4_k_gemv_resident, encode_dequant_q4_k_gemv_resident_wide,
    encode_dequant_q4_k_gemv_resident_wide_unroll, encode_dequant_q8_0_gemv_resident,
    Int4ResidentMatrix, KQuantBatchType, KQuantResidentMatrix, MetalContext, PassEncoder,
    Q4KResidentMatrix, Q8_0ResidentMatrix,
};

/// Weight bytes each timed command buffer walks, as in gemv_format_price.rs.
const TARGET_BYTES: u64 = 1 << 31;
/// Byte alignment of each tensor inside the shared weights buffer.
const ALIGN: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Block {
    Q4K,
    Q8,
}

#[derive(Clone, Debug)]
struct Tensor {
    file: String,
    name: String,
    cols: usize,
    rows: usize,
    file_offset: u64,
    nbytes: usize,
}

/// Reads the manifest named by `var` and keeps the rows of ggml type `ty`.
fn read_manifest(var: &str, ty: &str) -> Vec<Tensor> {
    let path = std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a manifest path"));
    let text = std::fs::read_to_string(&path).expect("read manifest");
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.is_empty()) {
        let c: Vec<&str> = line.split('\t').collect();
        assert_eq!(c.len(), 7, "manifest line must have 7 columns: {line}");
        if c[2] != ty {
            continue;
        }
        out.push(Tensor {
            file: c[0].to_string(),
            name: c[1].to_string(),
            cols: c[3].parse().expect("cols"),
            rows: c[4].parse().expect("rows"),
            file_offset: c[5].parse().expect("offset"),
            nbytes: c[6].parse().expect("nbytes"),
        });
    }
    assert!(!out.is_empty(), "no {ty} tensors in {path}");
    out
}

/// Reads one tensor's raw bytes and checks they look like quantized data:
/// a plausible first block scale, and mostly nonzero bytes.
fn read_tensor(t: &Tensor) -> Vec<u8> {
    let mut f = File::open(&t.file).expect("open gguf");
    f.seek(SeekFrom::Start(t.file_offset)).expect("seek");
    let mut buf = vec![0u8; t.nbytes];
    f.read_exact(&mut buf).expect("read tensor");
    let d = f16::from_le_bytes([buf[0], buf[1]]).to_f32();
    assert!(
        d.is_finite() && d.abs() > 1e-8 && d.abs() < 1e3,
        "{}: implausible first block scale {d}",
        t.name
    );
    let nonzero = buf.iter().filter(|&&b| b != 0).count();
    assert!(
        nonzero * 2 > buf.len(),
        "{}: mostly zero bytes, not real weights",
        t.name
    );
    buf
}

fn align_up(x: usize) -> usize {
    x.div_ceil(ALIGN) * ALIGN
}

/// One arm: the byte layout of every tensor inside a single weights buffer.
struct Arm {
    label: &'static str,
    kind: ArmKind,
    blob: metal::Buffer,
    offsets: Vec<usize>,
    footprint: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum ArmKind {
    Block(Block),
    Int4,
    /// The KERN-04 Stage A unit-wide loads arm (Q4_K only so far).
    Q4KWide,
    /// Stage B: the wide GEMV over this many superblocks per iteration.
    Q4KWideUnroll(u32),
    /// Stage B: the batched verify GEMM at this width, wide or current.
    Q4KGemm {
        batch: usize,
        wide: bool,
    },
}

/// INT4 bytes for a matrix: packed nibbles, then BF16 scales and biases, one
/// per 64 weights. Same bytes per weight as the INT4 probe in gemv_format_price.rs.
fn int4_bytes(rows: usize, cols: usize) -> usize {
    rows * cols / 2 + rows * cols / 64 * 4
}

/// Builds an arm. `real` copies each tensor's bytes in; otherwise (control or
/// INT4) the buffer is zeroed and only the layout matters.
fn build_arm(
    context: &MetalContext,
    label: &'static str,
    kind: ArmKind,
    tensors: &[Tensor],
    real: Option<&[Vec<u8>]>,
) -> Arm {
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cursor = 0usize;
    for t in tensors {
        offsets.push(cursor);
        let bytes = match kind {
            ArmKind::Int4 => int4_bytes(t.rows, t.cols),
            ArmKind::Block(_)
            | ArmKind::Q4KWide
            | ArmKind::Q4KWideUnroll(_)
            | ArmKind::Q4KGemm { .. } => t.nbytes,
        };
        cursor = align_up(cursor + bytes);
    }
    let mut blob = vec![0u8; cursor];
    if let Some(data) = real {
        for (i, bytes) in data.iter().enumerate() {
            blob[offsets[i]..offsets[i] + bytes.len()].copy_from_slice(bytes);
        }
    }
    Arm {
        label,
        kind,
        blob: context.new_buffer_with_data(&blob),
        offsets,
        footprint: cursor,
    }
}

fn encode_one(
    context: &mut MetalContext,
    pass: &PassEncoder,
    arm: &Arm,
    t: &Tensor,
    off: usize,
    x: &metal::Buffer,
    y: &metal::Buffer,
) {
    let off = off as u64;
    match arm.kind {
        ArmKind::Int4 => {
            let scales = off + (t.rows * t.cols / 2) as u64;
            let biases = scales + (t.rows * t.cols / 64 * 2) as u64;
            encode_dequant_int4_gemv_resident(
                context,
                pass,
                &Int4ResidentMatrix {
                    buffer: &arm.blob,
                    weights_offset: off,
                    scales_offset: scales,
                    biases_offset: biases,
                    rows: t.rows,
                    cols: t.cols,
                },
                (x, 0),
                (y, 0),
            )
            .expect("encode int4");
        }
        ArmKind::Block(Block::Q4K) => {
            encode_dequant_q4_k_gemv_resident(
                context,
                pass,
                &Q4KResidentMatrix {
                    buffer: &arm.blob,
                    weights_offset: off,
                    rows: t.rows,
                    cols: t.cols,
                },
                (x, 0),
                (y, 0),
            )
            .expect("encode q4_k");
        }
        ArmKind::Q4KWide => {
            encode_dequant_q4_k_gemv_resident_wide(
                context,
                pass,
                &Q4KResidentMatrix {
                    buffer: &arm.blob,
                    weights_offset: off,
                    rows: t.rows,
                    cols: t.cols,
                },
                (x, 0),
                (y, 0),
            )
            .expect("encode q4_k wide");
        }
        ArmKind::Q4KWideUnroll(unroll) => {
            encode_dequant_q4_k_gemv_resident_wide_unroll(
                context,
                pass,
                &Q4KResidentMatrix {
                    buffer: &arm.blob,
                    weights_offset: off,
                    rows: t.rows,
                    cols: t.cols,
                },
                (x, 0),
                (y, 0),
                unroll,
            )
            .expect("encode q4_k wide unroll");
        }
        ArmKind::Q4KGemm { batch, wide } => {
            encode_dequant_q4_k_gemm_resident_choose(
                context,
                pass,
                &KQuantResidentMatrix {
                    buffer: &arm.blob,
                    weights_offset: off,
                    rows: t.rows,
                    cols: t.cols,
                    kind: KQuantBatchType::Q4K,
                },
                (x, 0),
                (y, 0),
                batch,
                wide,
            )
            .expect("encode q4_k gemm");
        }
        ArmKind::Block(Block::Q8) => {
            encode_dequant_q8_0_gemv_resident(
                context,
                pass,
                &Q8_0ResidentMatrix {
                    buffer: &arm.blob,
                    weights_offset: off,
                    rows: t.rows,
                    cols: t.cols,
                },
                (x, 0),
                (y, 0),
            )
            .expect("encode q8_0");
        }
    }
}

/// Walks the tensors in order, cycling, until TARGET_BYTES of weights have been
/// read in one command buffer. Returns (GPU seconds, bytes walked).
fn walk(
    context: &mut MetalContext,
    arm: &Arm,
    tensors: &[Tensor],
    x: &metal::Buffer,
    y: &metal::Buffer,
) -> (f64, u64) {
    autorelease_pool(|| {
        let pass = context.begin_pass();
        let mut walked = 0u64;
        let mut k = 0usize;
        while walked < TARGET_BYTES {
            let i = k % tensors.len();
            encode_one(context, &pass, arm, &tensors[i], arm.offsets[i], x, y);
            walked += tensors[i].nbytes as u64;
            k += 1;
        }
        (pass.commit_and_wait_with_gpu_time(), walked)
    })
}

fn run(block: Block, manifest_var: &str, ty: &str) {
    let tensors = read_manifest(manifest_var, ty);
    let real: Vec<Vec<u8>> = tensors.iter().map(read_tensor).collect();
    let mut context = MetalContext::new().expect("Metal device");

    let real_arm = build_arm(
        &context,
        "real bytes",
        ArmKind::Block(block),
        &tensors,
        Some(&real),
    );
    let zero_arm = build_arm(
        &context,
        "zero control",
        ArmKind::Block(block),
        &tensors,
        None,
    );
    let int4_arm = build_arm(&context, "INT4 same shapes", ArmKind::Int4, &tensors, None);
    // The wide arm reads the real arm's buffer: same bytes, no second copy.
    let wide_arm = (block == Block::Q4K).then(|| Arm {
        label: "wide loads",
        kind: ArmKind::Q4KWide,
        blob: real_arm.blob.clone(),
        offsets: real_arm.offsets.clone(),
        footprint: real_arm.footprint,
    });
    let mut arms = vec![real_arm, zero_arm, int4_arm];
    arms.extend(wide_arm);

    let max_cols = tensors.iter().map(|t| t.cols).max().unwrap();
    let max_rows = tensors.iter().map(|t| t.rows).max().unwrap();
    let x = context.new_output_buffer((max_cols * 2) as u64);
    let y = context.new_output_buffer((max_rows * 2) as u64);

    let mut best = vec![0.0f64; arms.len()];
    // Round 0 is warmup. Arms interleave each round so thermal drift hits all alike.
    for round in 0..6 {
        for (i, arm) in arms.iter().enumerate() {
            let (seconds, walked) = walk(&mut context, arm, &tensors, &x, &y);
            if round > 0 {
                best[i] = best[i].max(walked as f64 / seconds / 1e9);
            }
        }
    }

    let total_mb: f64 = tensors.iter().map(|t| t.nbytes as f64).sum::<f64>() / 1e6;
    println!(
        "{:?}: {} tensors, {:.1} MB of quantized weights, {:.1} MB footprint per arm",
        block,
        tensors.len(),
        total_mb,
        arms[0].footprint as f64 / 1e6
    );
    for (i, arm) in arms.iter().enumerate() {
        println!(
            "  {:<16} {:6.1} GB/s  ({:.3}x INT4 same shapes)",
            arm.label,
            best[i],
            best[i] / best[2]
        );
    }
    println!(
        "  real vs zero control: {:.3}x (near 1.0 means the data does not change the rate)",
        best[0] / best[1]
    );
    if arms.len() > 3 {
        println!(
            "  wide loads vs current on the same real bytes: {:.3}x (the bench rule asks 1.25x)",
            best[3] / best[0]
        );
    }
}

#[test]
#[ignore = "benchmark: needs a real Metal device and two GGUF manifests, reports rather than asserts"]
fn stage_a_real_bytes_price() {
    run(Block::Q4K, "TURBOSPARK_BENCH_Q4K_MANIFEST", "Q4_K");
    run(Block::Q8, "TURBOSPARK_BENCH_Q8_MANIFEST", "Q8_0");
}

/// Stage B for Q4_K (docs/designs/speedup-kern.md section 3), run only after
/// Stage A cleared 1.25x: the unroll arms of the wide GEMV at one row, and the
/// batched verify GEMM, current against wide, at B = 2, 4 and 8. Every arm
/// reads the SAME real bytes (one shared buffer), and arms interleave each
/// round. GB/s counts weight bytes once per dispatch, so a batched arm's rate
/// is how fast it moves the matrix while serving B tokens.
#[test]
#[ignore = "benchmark: needs a real Metal device and a Q4_K manifest, reports rather than asserts"]
fn stage_b_q4k_arms() {
    let tensors = read_manifest("TURBOSPARK_BENCH_Q4K_MANIFEST", "Q4_K");
    let real: Vec<Vec<u8>> = tensors.iter().map(read_tensor).collect();
    let mut context = MetalContext::new().expect("Metal device");
    let base = build_arm(
        &context,
        "current",
        ArmKind::Block(Block::Q4K),
        &tensors,
        Some(&real),
    );
    let share = |label: &'static str, kind: ArmKind| Arm {
        label,
        kind,
        blob: base.blob.clone(),
        offsets: base.offsets.clone(),
        footprint: base.footprint,
    };
    let mut arms = vec![
        share("wide", ArmKind::Q4KWide),
        share("wide u2", ArmKind::Q4KWideUnroll(2)),
        share("wide u4", ArmKind::Q4KWideUnroll(4)),
    ];
    for batch in [2usize, 4, 8] {
        let label_cur: &'static str = Box::leak(format!("gemm B={batch} current").into_boxed_str());
        let label_wide: &'static str = Box::leak(format!("gemm B={batch} wide").into_boxed_str());
        arms.push(share(label_cur, ArmKind::Q4KGemm { batch, wide: false }));
        arms.push(share(label_wide, ArmKind::Q4KGemm { batch, wide: true }));
    }
    arms.insert(0, base);

    let max_cols = tensors.iter().map(|t| t.cols).max().unwrap();
    let max_rows = tensors.iter().map(|t| t.rows).max().unwrap();
    let x = context.new_output_buffer((8 * max_cols * 2) as u64);
    let y = context.new_output_buffer((8 * max_rows * 2) as u64);

    let mut best = vec![0.0f64; arms.len()];
    for round in 0..6 {
        for (i, arm) in arms.iter().enumerate() {
            let (seconds, walked) = walk(&mut context, arm, &tensors, &x, &y);
            if round > 0 {
                best[i] = best[i].max(walked as f64 / seconds / 1e9);
            }
        }
    }
    println!(
        "Q4K stage B: {} tensors, {:.1} MB of real Q4_K bytes, one shared buffer",
        tensors.len(),
        arms[0].footprint as f64 / 1e6
    );
    for (i, arm) in arms.iter().enumerate() {
        println!(
            "  {:<20} {:6.1} GB/s  ({:.3}x current GEMV)",
            arm.label,
            best[i],
            best[i] / best[0]
        );
    }
    for (k, batch) in [2usize, 4, 8].iter().enumerate() {
        let cur = best[4 + 2 * k];
        let wide = best[5 + 2 * k];
        println!("  verify B={batch}: wide / current {:.3}x", wide / cur);
    }
}
