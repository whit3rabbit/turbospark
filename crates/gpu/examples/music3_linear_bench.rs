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
                (2usize, 4096usize, 4096usize),
                (2, 4096, 12288),
                (2, 12288, 4096),
                (2, 4096, 6144),
                (6, 4096, 4096),
                (14, 4096, 4096),
                (12, 4096, 6144),
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
        other => panic!("unknown mode {other}; use hashes or timing"),
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("music3_linear_bench requires macOS and a Metal device");
}
