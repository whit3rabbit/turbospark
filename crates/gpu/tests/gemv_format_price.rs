//! Pricing probe, not a gate: weight GB/s of the GGUF block GEMVs (Q4_K,
//! Q8_0) and the 2-bit affine GEMV against the INT4 affine one at the same
//! shapes, in one process. It answers whether a format's kernel is far enough
//! below the bandwidth the INT4 kernel reaches to be worth rewriting.
//!
//! Zero weights are fine for a rate: every kernel reads every byte whatever
//! it holds. The ratios are the output; absolute GB/s moves with clocks.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test gemv_format_price --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use turbospark_gpu::{
    autorelease_pool, encode_dequant_int2_gemv_resident, encode_dequant_int4_gemv_resident,
    encode_dequant_q4_k_gemv_resident, encode_dequant_q8_0_gemv_resident, int2_row_bytes,
    q4_k_row_bytes, q8_0_row_bytes, Int2ResidentMatrix, Int4ResidentMatrix, MetalContext,
    PassEncoder, Q4KResidentMatrix, Q8_0ResidentMatrix,
};

/// Bytes of weight data each timed command buffer walks.
const TARGET_BYTES: u64 = 1 << 31;

#[derive(Clone, Copy, Debug)]
enum Format {
    Int4,
    Q4K,
    Q8,
    Int2,
}

impl Format {
    /// Weight bytes per row, companions included.
    fn row_bytes(self, cols: usize) -> usize {
        match self {
            Format::Int4 => cols / 2 + cols / 64 * 4,
            Format::Q4K => q4_k_row_bytes(cols),
            Format::Q8 => q8_0_row_bytes(cols),
            Format::Int2 => int2_row_bytes(cols) + cols / 128 * 4,
        }
    }
}

fn encode(
    context: &mut MetalContext,
    pass: &PassEncoder,
    format: Format,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    (rows, cols): (usize, usize),
) {
    match format {
        Format::Int4 => {
            let scales_offset = (rows * cols / 2) as u64;
            encode_dequant_int4_gemv_resident(
                context,
                pass,
                &Int4ResidentMatrix {
                    buffer: w,
                    weights_offset: 0,
                    scales_offset,
                    biases_offset: scales_offset + (rows * cols / 64 * 2) as u64,
                    rows,
                    cols,
                },
                (x, 0),
                (y, 0),
            )
        }
        Format::Int2 => {
            let scales_offset = (rows * int2_row_bytes(cols)) as u64;
            encode_dequant_int2_gemv_resident(
                context,
                pass,
                &Int2ResidentMatrix {
                    buffer: w,
                    weights_offset: 0,
                    scales_offset,
                    biases_offset: scales_offset + (rows * cols / 128 * 2) as u64,
                    rows,
                    cols,
                    group_size: 128,
                },
                (x, 0),
                (y, 0),
            )
        }
        Format::Q4K => encode_dequant_q4_k_gemv_resident(
            context,
            pass,
            &Q4KResidentMatrix {
                buffer: w,
                weights_offset: 0,
                rows,
                cols,
            },
            (x, 0),
            (y, 0),
        ),
        Format::Q8 => encode_dequant_q8_0_gemv_resident(
            context,
            pass,
            &Q8_0ResidentMatrix {
                buffer: w,
                weights_offset: 0,
                rows,
                cols,
            },
            (x, 0),
            (y, 0),
        ),
    }
    .expect("encode");
}

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn block_format_gemv_rate_against_int4() {
    let mut context = MetalContext::new().expect("Metal device");
    let formats = [Format::Int4, Format::Q4K, Format::Q8, Format::Int2];
    for (rows, cols) in [(5120usize, 5120usize), (1024, 5120), (17408, 5120)] {
        let x = context.new_output_buffer((cols * 2) as u64);
        let y = context.new_output_buffer((rows * 2) as u64);
        let weights: Vec<_> = formats
            .iter()
            .map(|f| context.new_output_buffer((rows * f.row_bytes(cols)) as u64))
            .collect();
        // GB/s of weight bytes (ratios of these are what a rewrite can win
        // in time only after dividing by bytes per element, printed too).
        let mut best = [0.0f64; 4];
        // Round 0 is warmup; arms interleave so thermal drift hits all alike.
        for round in 0..6 {
            for (i, &format) in formats.iter().enumerate() {
                let bytes = (rows * format.row_bytes(cols)) as u64;
                let repeats = (TARGET_BYTES / bytes).max(1);
                let seconds = autorelease_pool(|| {
                    let pass = context.begin_pass();
                    for _ in 0..repeats {
                        encode(
                            &mut context,
                            &pass,
                            format,
                            &weights[i],
                            &x,
                            &y,
                            (rows, cols),
                        );
                    }
                    pass.commit_and_wait_with_gpu_time()
                });
                if round > 0 {
                    best[i] = best[i].max((bytes * repeats) as f64 / seconds / 1e9);
                }
            }
        }
        for (i, &format) in formats.iter().enumerate() {
            let bits = format.row_bytes(cols) as f64 * 8.0 / cols as f64;
            let gmac = best[i] / (bits / 8.0);
            let int4_bits = Format::Int4.row_bytes(cols) as f64 * 8.0 / cols as f64;
            println!(
                "{rows:>6} x {cols}  {format:?}: {:6.1} GB/s ({:.2}x INT4), {bits:.2} bits/weight, \
                 {gmac:6.1} G weights/s ({:.2}x INT4)",
                best[i],
                best[i] / best[0],
                gmac / (best[0] / (int4_bits / 8.0)),
            );
        }
    }
}
