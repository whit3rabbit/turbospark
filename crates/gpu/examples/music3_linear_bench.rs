//! Bit-exactness oracle and timing for the Music 3 packed linear kernels.
//!
//! `hashes` prints an FNV-1a hash of the output bits for a grid of affine
//! bit widths, group sizes, row counts and dtypes. Run it before and after a
//! kernel change: any difference in any line means the arithmetic changed.
//! `timing` prints mean wall time for the real checkpoint shapes.
//!
//! cargo run --release -p turbospark-gpu --example music3_linear_bench -- hashes|timing

#[cfg(target_os = "macos")]
fn main() {
    use std::time::Instant;
    use turbospark_gpu::{Music3DType, Music3Device, Music3Encoding};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*: deterministic across runs and machines.
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }
    // Checkpoint scales and offsets are bf16 values stored as f32.
    fn bf16_trunc(v: f32) -> f32 {
        f32::from_bits(v.to_bits() & 0xffff_0000)
    }
    fn pack(codes: &[u32], rows: usize, cols: usize, bits: u32) -> Vec<u8> {
        let stride = (cols * bits as usize).div_ceil(32);
        let mut out = vec![0u32; rows * stride];
        for r in 0..rows {
            for c in 0..cols {
                let bit = c * bits as usize;
                let i = r * stride + bit / 32;
                let shift = bit % 32;
                out[i] |= codes[r * cols + c] << shift;
                if shift + bits as usize > 32 {
                    out[i + 1] |= codes[r * cols + c] >> (32 - shift);
                }
            }
        }
        out.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    fn fnv(values: &[f32]) -> u64 {
        values.iter().fold(0xcbf29ce484222325u64, |h, v| {
            (h ^ u64::from(v.to_bits())).wrapping_mul(0x100000001b3)
        })
    }
    struct Case {
        weight: turbospark_gpu::Music3Weight,
        input_dim: usize,
        output_dim: usize,
        bias: Vec<f32>,
        seed: u64,
    }
    fn build(
        device: &Music3Device,
        bits: u32,
        group: usize,
        input_dim: usize,
        output_dim: usize,
        seed: u64,
    ) -> Case {
        let mut rng = Rng(seed | 1);
        let codes: Vec<u32> = (0..output_dim * input_dim)
            .map(|_| (rng.next() as u32) & ((1u32 << bits) - 1))
            .collect();
        let groups = output_dim * input_dim / group;
        let scales: Vec<f32> = (0..groups)
            .map(|_| bf16_trunc(0.002 + 0.01 * rng.unit()))
            .collect();
        let offsets: Vec<f32> = (0..groups)
            .map(|_| bf16_trunc(-0.05 + 0.1 * rng.unit()))
            .collect();
        let weight = device
            .load_weight(
                &[output_dim, input_dim],
                &pack(&codes, output_dim, input_dim, bits),
                Music3Encoding::Affine {
                    bits,
                    group_size: group,
                },
                &scales,
                &offsets,
                &[],
            )
            .unwrap();
        let bias = (0..output_dim)
            .map(|_| bf16_trunc(rng.unit() - 0.5))
            .collect();
        Case {
            weight,
            input_dim,
            output_dim,
            bias,
            seed,
        }
    }
    fn input(case: &Case, rows: usize) -> Vec<f32> {
        let mut rng = Rng(case.seed ^ 0x9e3779b97f4a7c15 ^ rows as u64);
        (0..rows * case.input_dim)
            .map(|_| bf16_trunc(2.0 * rng.unit() - 1.0))
            .collect()
    }

    let mode = std::env::args().nth(1).unwrap_or_else(|| "hashes".into());
    let device = Music3Device::new().unwrap();
    match mode.as_str() {
        "hashes" => {
            for bits in [2u32, 3, 4, 5, 6, 8] {
                for group in [32usize, 64] {
                    // 512 inputs divides every group size and bit width used.
                    let case = build(&device, bits, group, 512, 96, 7 + u64::from(bits));
                    for dtype in [Music3DType::F32, Music3DType::F16, Music3DType::Bf16] {
                        for rows in [1usize, 2, 3, 5, 9, 12, 13, 15, 16, 40] {
                            for with_bias in [false, true] {
                                let x = input(&case, rows);
                                let out = case
                                    .weight
                                    .linear_typed(
                                        &x,
                                        with_bias.then_some(case.bias.as_slice()),
                                        rows,
                                        case.input_dim,
                                        case.output_dim,
                                        dtype,
                                    )
                                    .unwrap();
                                println!(
                                    "bits={bits} group={group} {dtype:?} rows={rows} bias={with_bias} {:016x}",
                                    fnv(&out)
                                );
                            }
                        }
                    }
                }
            }
            // Tall and ragged shapes: tile tails in rows, outputs and K, plus
            // the split-K parts the dispatcher chooses for small tiles.
            for bits in [4u32, 8] {
                for (input_dim, output_dim) in [(576usize, 100usize), (640, 33), (1024, 256)] {
                    let case = build(
                        &device,
                        bits,
                        64,
                        input_dim,
                        output_dim,
                        31 + u64::from(bits),
                    );
                    for dtype in [Music3DType::F16, Music3DType::Bf16] {
                        for rows in [14usize, 16, 17, 31, 32, 33, 64, 65, 100, 200] {
                            let x = input(&case, rows);
                            let out = case
                                .weight
                                .linear_typed(
                                    &x,
                                    Some(&case.bias),
                                    rows,
                                    input_dim,
                                    output_dim,
                                    dtype,
                                )
                                .unwrap();
                            println!(
                                "tall bits={bits} {input_dim}x{output_dim} {dtype:?} rows={rows} {:016x}",
                                fnv(&out)
                            );
                        }
                    }
                }
            }
            // A decode-sized shape at the real width, hashed for the 4-bit case.
            let case = build(&device, 4, 64, 4096, 1024, 99);
            for rows in [2usize, 6, 14] {
                let x = input(&case, rows);
                let out = case
                    .weight
                    .linear_typed(&x, None, rows, 4096, 1024, Music3DType::Bf16)
                    .unwrap();
                println!("wide bits=4 rows={rows} {:016x}", fnv(&out));
            }
        }
        "timing" => {
            for (rows, input_dim, output_dim) in [
                // Smallest real projection: fixed per-call cost, not kernel time.
                (2usize, 64usize, 64usize),
                (2, 4096, 1024),
                (2, 4096, 4096),
                (2, 4096, 12288),
                (2, 12288, 4096),
                (2, 4096, 6144),
                (6, 4096, 4096),
                (14, 4096, 4096),
                (12, 4096, 6144),
                // DiT-sized: the shapes that dominate multi-step flow.
                (690, 2048, 2048),
                (690, 2048, 16384),
                (690, 8192, 2048),
            ] {
                let case = build(&device, 4, 64, input_dim, output_dim, 5);
                let x = input(&case, rows);
                let run = || {
                    case.weight
                        .linear_typed(&x, None, rows, input_dim, output_dim, Music3DType::Bf16)
                        .unwrap()
                };
                run();
                run();
                let iterations = 10;
                let started = Instant::now();
                let mut last = Vec::new();
                for _ in 0..iterations {
                    last = run();
                }
                let mean_ms = started.elapsed().as_secs_f64() * 1e3 / f64::from(iterations);
                // Packed 4-bit weight bytes the kernel must read at minimum.
                let gigabytes = (input_dim * output_dim / 2) as f64 / 1e9;
                println!(
                    "rows={rows} in={input_dim} out={output_dim} mean_ms={mean_ms:.3} \
                     weight_GBps={:.1} hash={:016x}",
                    gigabytes / (mean_ms / 1e3),
                    fnv(&last)
                );
            }
        }
        "conv_hashes" | "conv_timing" => {
            use half::bf16;
            // (ic, oc, kernel, stride, pad, dilation, transpose, len)
            type ConvCase = (usize, usize, usize, usize, usize, usize, bool, usize);
            let hash_cases: Vec<ConvCase> = vec![
                (96, 96, 7, 1, 3, 1, false, 1000),
                (96, 96, 7, 1, 9, 3, false, 1000),
                (96, 96, 7, 1, 27, 9, false, 1000),
                (96, 96, 1, 1, 0, 1, false, 513),
                (128, 100, 7, 1, 3, 1, false, 300),
                (40, 33, 7, 1, 3, 1, false, 77),
                (192, 96, 16, 8, 4, 1, true, 37),
                (96, 50, 4, 2, 1, 1, true, 101),
                (64, 100, 16, 8, 4, 1, true, 29),
                (130, 70, 8, 4, 2, 1, true, 55),
            ];
            let timing_cases: Vec<ConvCase> = vec![
                (192, 192, 7, 1, 3, 1, false, 176384),
                (192, 192, 1, 1, 0, 1, false, 176384),
                (384, 192, 8, 4, 2, 1, true, 44096),
                (1536, 768, 16, 8, 4, 1, true, 689),
            ];
            let cases = if mode == "conv_hashes" {
                hash_cases
            } else {
                timing_cases
            };
            for (ic, oc, kernel, stride, pad, dilation, transpose, len) in cases {
                let mut rng = Rng(77 + (ic * 31 + oc) as u64);
                let shape = if transpose {
                    [ic, oc, kernel]
                } else {
                    [oc, ic, kernel]
                };
                let count = shape.iter().product::<usize>();
                let bytes: Vec<u8> = (0..count)
                    .flat_map(|_| {
                        bf16::from_f32(0.2 * rng.unit() - 0.1)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect();
                let weight = device
                    .load_weight(&shape, &bytes, Music3Encoding::Bf16, &[], &[], &[])
                    .unwrap();
                let x: Vec<f32> = (0..ic * len)
                    .map(|_| bf16_trunc(2.0 * rng.unit() - 1.0))
                    .collect();
                let bias: Vec<f32> = (0..oc).map(|_| bf16_trunc(rng.unit() - 0.5)).collect();
                for dtype in [Music3DType::F16, Music3DType::Bf16] {
                    if mode == "conv_timing" && dtype == Music3DType::F16 {
                        continue;
                    }
                    let run = || {
                        weight
                            .convolution_typed(
                                &x,
                                Some(&bias),
                                ic,
                                oc,
                                kernel,
                                stride,
                                pad,
                                dilation,
                                transpose,
                                dtype,
                            )
                            .unwrap()
                    };
                    let mut last = run();
                    let mut mean_ms = 0.0;
                    if mode == "conv_timing" {
                        let started = Instant::now();
                        let iterations = 3;
                        for _ in 0..iterations {
                            last = run();
                        }
                        mean_ms = started.elapsed().as_secs_f64() * 1e3 / f64::from(iterations);
                    }
                    println!(
                        "conv ic={ic} oc={oc} k={kernel} s={stride} p={pad} d={dilation} t={transpose} len={len} {dtype:?} mean_ms={mean_ms:.2} {:016x}",
                        fnv(&last)
                    );
                }
            }
        }
        other => panic!("unknown mode {other}; use hashes, timing, conv_hashes or conv_timing"),
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("music3_linear_bench requires macOS and a Metal device");
}
