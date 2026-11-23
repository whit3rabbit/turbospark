#![cfg(target_os = "macos")]
//! Not a gate: GPU time and bandwidth of `inkling2_bf16_gemv_multi`, Inkling
//! 2's decode BF16 GEMV, at the Mini's decode shapes, to tell how far it sits
//! from the memory system. About 192 MB of distinct matrices are cycled so
//! the weights come from DRAM, not the system cache, and a primer pass brings
//! the device up first (a cold device runs 3-6x slow). Both the one-row and
//! the two-row form run, so their rates compare directly.
//!
//! cargo test --release -p turbospark-gpu --test inkling2_gemv_bandwidth -- --ignored --nocapture

use turbospark_gpu::{
    encode_inkling2_bf16_gemv_multi, encode_inkling2_bf16_gemv_multi2, Inkling2GemvSegment,
    MetalContext,
};

const REPS: usize = 4;

#[test]
#[ignore = "benchmark: needs a real Metal device, reports rather than asserts"]
fn bf16_gemv_multi_bandwidth_at_mini_decode_shapes() {
    let mut context = MetalContext::new().expect("Metal device");
    // (rows, cols): q / gate_attn (3072 x 2304), k / v (1024 x 2304),
    // o-proj (2304 x 3072), unembed (201,024 x 2304, once per token).
    for (rows, cols, label) in [
        (3072usize, 2304usize, "q"),
        (1024, 2304, "k"),
        (2304, 3072, "o"),
        (201_024, 2304, "unembed"),
    ] {
        let mat = rows * cols * 2;
        let copies = (192usize << 20).div_ceil(mat).clamp(1, 256);
        let w = context.new_buffer_with_data(&vec![0x3cu8; mat * copies]);
        let x = context.new_buffer_with_data(&vec![0x3cu8; cols * 2]);
        let y = context.new_buffer_with_data(&vec![0u8; rows * 2]);
        for two in [false, true] {
            let run = |context: &mut MetalContext| -> f64 {
                let pass = context.begin_pass();
                for _ in 0..REPS {
                    for c in 0..copies {
                        let seg = Inkling2GemvSegment {
                            w_offset: (c * mat) as u64,
                            rows: rows as u32,
                            cols: cols as u32,
                            x: 0,
                            y: (&y, 0),
                        };
                        if two {
                            encode_inkling2_bf16_gemv_multi2(
                                context,
                                &pass,
                                &w,
                                &[(&x, 0)],
                                &[seg],
                            )
                            .unwrap();
                        } else {
                            encode_inkling2_bf16_gemv_multi(context, &pass, &w, &[(&x, 0)], &[seg])
                                .unwrap();
                        }
                    }
                }
                pass.commit_and_wait_with_gpu_time()
            };
            run(&mut context); // primer
            let t = (0..3).map(|_| run(&mut context)).fold(f64::MAX, f64::min);
            let n = (REPS * copies) as f64;
            println!(
                "{label:8} {rows:6}x{cols} {}: {:8.2} us/dispatch {:6.1} GB/s ({copies} copies)",
                if two { "2-row" } else { "1-row" },
                t / n * 1e6,
                mat as f64 / (t / n) / 1e9
            );
        }
    }
}
