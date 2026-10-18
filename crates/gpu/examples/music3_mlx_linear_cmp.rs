//! Bit-for-bit diff of the Music 3 Metal linear kernels against MLX 0.32.3.
//!
//! Reads a directory written by `crates/audio/tools/gen_music3_mlx_linear_cases.py`
//! (real depth-decoder weights, a bf16 input block, and MLX's own output at row
//! counts 1 to 16) and runs the same operation through `Music3Weight::linear_typed`.
//! No model code, sampling or tokenizer is involved, so a mismatch here is the
//! kernel and nothing else. Prints one line per (case, rows) and exits nonzero
//! when `--require-exact` is given and any value differs.
//!
//! cargo run --release -p turbospark-gpu --example music3_mlx_linear_cmp -- DIR [--require-exact]

#[cfg(target_os = "macos")]
fn main() {
    use std::{fs, path::Path, process::ExitCode};
    use turbospark_gpu::{Music3DType, Music3Device, Music3Encoding};

    fn read_f32(path: &Path) -> Vec<f32> {
        fs::read(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn run() -> ExitCode {
        let mut args = std::env::args().skip(1);
        let dir = std::path::PathBuf::from(args.next().expect("directory required"));
        let require_exact = args.any(|a| a == "--require-exact");
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("cases.json")).unwrap()).unwrap();
        assert_eq!(manifest["mlx_version"], "0.32.3");
        let device = Music3Device::new().unwrap();
        let mut total_bad = 0usize;
        for case in manifest["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let (k, n) = (
                case["in"].as_u64().unwrap() as usize,
                case["out"].as_u64().unwrap() as usize,
            );
            let weight_bytes = fs::read(dir.join(format!("{name}.w.bin"))).unwrap();
            let weight = if case["kind"] == "dense" {
                device
                    .load_weight(&[n, k], &weight_bytes, Music3Encoding::Bf16, &[], &[], &[])
                    .unwrap()
            } else {
                let scales = read_f32(&dir.join(format!("{name}.scales.f32")));
                let offsets = read_f32(&dir.join(format!("{name}.biases.f32")));
                device
                    .load_weight(
                        &[n, k],
                        &weight_bytes,
                        Music3Encoding::Affine {
                            bits: case["quant"]["bits"].as_u64().unwrap() as u32,
                            group_size: case["quant"]["group_size"].as_u64().unwrap() as usize,
                        },
                        &scales,
                        &offsets,
                        &[],
                    )
                    .unwrap()
            };
            let x = read_f32(&dir.join(format!("{name}.x.f32")));
            let rows_list: Vec<usize> = case["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            for &rows in &rows_list {
                let expected = read_f32(&dir.join(format!("{name}.y{rows}.f32")));
                let got = weight
                    .linear_typed(&x[..rows * k], None, rows, k, n, Music3DType::Bf16)
                    .unwrap();
                assert_eq!(got.len(), expected.len());
                let bad = got
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let worst = got
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                total_bad += bad;
                println!(
                    "{name:<18} {k}x{n} rows={rows:<2} differing={bad}/{} max_abs={worst:.5}",
                    got.len()
                );
            }
        }
        for conv in manifest["convs"].as_array().into_iter().flatten() {
            let name = conv["name"].as_str().unwrap();
            let get = |key: &str| conv[key].as_u64().unwrap() as usize;
            let (ic, oc, kernel) = (get("in"), get("out"), get("kernel"));
            let (stride, pad, dilation) = (get("stride"), get("pad"), get("dilation"));
            let transpose = conv["transpose"].as_bool().unwrap();
            let shape = if transpose {
                [ic, oc, kernel]
            } else {
                [oc, ic, kernel]
            };
            let weight = device
                .load_weight(
                    &shape,
                    &fs::read(dir.join(format!("{name}.w.bin"))).unwrap(),
                    Music3Encoding::Bf16,
                    &[],
                    &[],
                    &[],
                )
                .unwrap();
            let bias = conv["bias"]
                .as_bool()
                .unwrap()
                .then(|| read_f32(&dir.join(format!("{name}.bias.f32"))));
            for len in conv["lens"].as_array().unwrap() {
                let len = len.as_u64().unwrap() as usize;
                let x = read_f32(&dir.join(format!("{name}.x{len}.f32")));
                let run = |bias: Option<&[f32]>| {
                    weight
                        .convolution_typed(
                            &x,
                            bias,
                            ic,
                            oc,
                            kernel,
                            stride,
                            pad,
                            dilation,
                            transpose,
                            Music3DType::Bf16,
                        )
                        .unwrap()
                };
                let diff = |got: &[f32], file: &str| -> (usize, f32) {
                    let expected = read_f32(&dir.join(file));
                    assert_eq!(got.len(), expected.len(), "{file}");
                    let bad = got
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let worst = got
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    (bad, worst)
                };
                let got = run(bias.as_deref());
                let (bad, worst) = diff(&got, &format!("{name}.y{len}.f32"));
                if bad > 0 && bad <= 4 {
                    // Output is channel-major [oc, outlen]: show where the few misses sit.
                    let expected = read_f32(&dir.join(format!("{name}.y{len}.f32")));
                    let outlen = got.len() / oc;
                    for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
                        if a.to_bits() != b.to_bits() {
                            println!(
                                "    differs at channel {} time {}: got {a} expected {b}",
                                i / outlen,
                                i % outlen
                            );
                        }
                    }
                }
                // The convolution alone, so a bias-add difference shows separately.
                let (raw_bad, _) = diff(&run(None), &format!("{name}.raw{len}.f32"));
                total_bad += bad;
                println!(
                    "{name:<18} {} {ic}->{oc} k{kernel} s{stride} d{dilation} len={len:<3} differing={bad}/{} max_abs={worst:.5} (conv alone: {raw_bad})",
                    if transpose { "convT" } else { "conv " },
                    got.len()
                );
            }
        }
        println!("total differing values: {total_bad}");
        if require_exact && total_bad != 0 {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }
    std::process::exit(if run() == ExitCode::SUCCESS { 0 } else { 1 });
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("music3_mlx_linear_cmp requires macOS and a Metal device");
}
