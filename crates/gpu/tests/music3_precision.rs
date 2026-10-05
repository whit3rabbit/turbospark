#![cfg(target_os = "macos")]
use turbospark_gpu::{Music3DType, Music3Device};

#[test]
fn native_norm_rounds_normalized_values_before_affine() {
    let device = Music3Device::new().unwrap();
    let x = [0.984375, 2.46875, -0.70703125, 0.2001953125];
    let w = [1.203125, 0.90234375, 1.796875, 0.69921875];
    // Independent MLX 0.32.3 golden values distinguish both rounding boundaries.
    assert_eq!(
        device
            .rms_norm(&x, &w, 1, 4, 1e-6, Music3DType::Bf16)
            .unwrap(),
        [0.859375, 1.6171875, -0.91796875, 0.1015625]
    );
    let b = [0.10009765625, 0.2001953125, 0.30078125, 0.400390625];
    assert_eq!(
        device
            .layer_norm(&x, &w, Some(&b), 1, 4, 1e-5, Music3DType::Bf16)
            .unwrap(),
        [0.35546875, 1.5390625, -1.9296875, 0.078125]
    );
}

#[test]
fn native_rope_preserves_contraction_order_and_rejects_position_truncation() {
    let device = Music3Device::new().unwrap();
    let mut x = vec![0.0; 128];
    x[9] = -0.65625;
    x[73] = 1.03125;
    let y = device
        .rope(&x, 1, 1, 1, 128, 7, 1e6, Music3DType::Bf16)
        .unwrap();
    // Independent MLX output at a cancellation tie in the full checkpoint.
    assert_eq!(y[73], 0.00116729736328125);
    assert!(device
        .rope(&x, 1, 1, 1, 128, 1usize << 32, 1e6, Music3DType::Bf16)
        .is_err());
}

#[test]
fn native_affine_short_groups_do_not_read_neighboring_columns() {
    use turbospark_gpu::Music3Encoding;
    let device = Music3Device::new().unwrap();
    for group in [1, 2] {
        let groups = 4 / group;
        let w = device
            .load_weight(
                &[1, 4],
                &0x1111u32.to_le_bytes(),
                Music3Encoding::Affine {
                    bits: 4,
                    group_size: group,
                },
                &vec![1.0; groups],
                &vec![0.0; groups],
                &[],
            )
            .unwrap();
        assert_eq!(
            w.linear_typed(&[1.0, 2.0, 3.0, 4.0], None, 1, 4, 1, Music3DType::Bf16)
                .unwrap(),
            [10.0]
        );
    }
}

fn values(c: &serde_json::Value, key: &str) -> Vec<f32> {
    c[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn n(c: &serde_json::Value, key: &str) -> usize {
    c[key].as_u64().unwrap() as usize
}
fn packed_bytes(c: &serde_json::Value, dtype: &str) -> Vec<u8> {
    use half::{bf16, f16};
    if dtype == "float32" {
        values(c, "weight")
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect()
    } else if dtype == "bf16" {
        values(c, "weight")
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect()
    } else if dtype == "float16" {
        values(c, "weight")
            .iter()
            .flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes())
            .collect()
    } else {
        c["weight"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|x| (x.as_u64().unwrap() as u32).to_le_bytes())
            .collect()
    }
}
#[test]
fn independent_mlx_native_operation_fixtures() {
    use turbospark_gpu::Music3Encoding;
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../audio/testdata/minimax_music3/precision/ops.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(fixtures["provenance"]["mlx_version"], "0.32.3");
    let device = Music3Device::new().unwrap();
    let mut failed = Vec::new();
    let mut count = 0;
    for c in fixtures["cases"].as_array().unwrap() {
        let dtype = match c["dtype"].as_str().unwrap() {
            "float32" => Music3DType::F32,
            "float16" => Music3DType::F16,
            _ => Music3DType::Bf16,
        };
        let actual = match c["op"].as_str().unwrap() {
            "rotary_tables" => device
                .rotary_tables(
                    n(c, "seq"),
                    n(c, "rotary_dim"),
                    c["theta"].as_f64().unwrap() as f32,
                )
                .map(|(cos, sin)| [cos, sin].concat()),
            "rms_norm" => device.rms_norm(
                &values(c, "input"),
                &values(c, "weight"),
                n(c, "rows"),
                n(c, "cols"),
                c["eps"].as_f64().unwrap() as f32,
                dtype,
            ),
            "layer_norm" => device.layer_norm(
                &values(c, "input"),
                &values(c, "weight"),
                Some(&values(c, "bias")),
                n(c, "rows"),
                n(c, "cols"),
                c["eps"].as_f64().unwrap() as f32,
                dtype,
            ),
            "linear" => {
                let enc = c["encoding"].as_str().unwrap();
                let encoding = match enc {
                    "float32" => Music3Encoding::F32,
                    "float16" => Music3Encoding::F16,
                    "bf16" => Music3Encoding::Bf16,
                    "mxfp8" => Music3Encoding::MxFp8,
                    "mxfp4" => Music3Encoding::MxFp4,
                    "nvfp4" => Music3Encoding::NvFp4,
                    _ => Music3Encoding::Affine {
                        bits: n(c, "bits") as u32,
                        group_size: n(c, "group_size"),
                    },
                };
                let scales = if c["scales"].is_array() {
                    values(c, "scales")
                } else {
                    vec![]
                };
                let offsets = if c["offsets"].is_array() {
                    values(c, "offsets")
                } else {
                    vec![]
                };
                let blocks: Vec<u8> = if c["scale_dtype"] == "uint8" {
                    scales.iter().map(|s| *s as u8).collect()
                } else {
                    vec![]
                };
                let w = device
                    .load_weight(
                        &[n(c, "output_dim"), n(c, "input_dim")],
                        &packed_bytes(c, enc),
                        encoding,
                        if blocks.is_empty() { &scales } else { &[] },
                        &offsets,
                        &blocks,
                    )
                    .unwrap();
                let product = w
                    .linear_typed(
                        &values(c, "input"),
                        None,
                        n(c, "rows"),
                        n(c, "input_dim"),
                        n(c, "output_dim"),
                        dtype,
                    )
                    .unwrap();
                let expected_product = if c["product"].is_array() {
                    values(c, "product")
                } else {
                    product.clone()
                };
                let max = product
                    .iter()
                    .zip(&expected_product)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                if max > 0.0 {
                    eprintln!(
                        "{} producterr={max} first {:?}",
                        c["name"],
                        product
                            .iter()
                            .zip(&expected_product)
                            .enumerate()
                            .filter(|(_, (a, b))| a != b)
                            .take(5)
                            .collect::<Vec<_>>()
                    );
                }
                w.linear_typed(
                    &values(c, "input"),
                    Some(&values(c, "bias")),
                    n(c, "rows"),
                    n(c, "input_dim"),
                    n(c, "output_dim"),
                    dtype,
                )
            }
            "convolution" => {
                let ic = n(c, "input_channels");
                let oc = n(c, "output_channels");
                let k = n(c, "kernel");
                let trans = c["transpose"].as_bool().unwrap();
                let shape = if trans {
                    vec![ic, oc, k]
                } else {
                    vec![oc, ic, k]
                };
                let w = device
                    .load_weight(
                        &shape,
                        &packed_bytes(c, "bf16"),
                        Music3Encoding::Bf16,
                        &[],
                        &[],
                        &[],
                    )
                    .unwrap();
                w.convolution_typed(
                    &values(c, "input"),
                    Some(&values(c, "bias")),
                    ic,
                    oc,
                    k,
                    n(c, "stride"),
                    n(c, "padding"),
                    n(c, "dilation"),
                    trans,
                    dtype,
                )
            }
            "attention" => device.attention_typed(
                &values(c, "q"),
                &values(c, "k"),
                &values(c, "v"),
                n(c, "batch"),
                n(c, "queries"),
                n(c, "keys"),
                n(c, "heads"),
                n(c, "kv_heads"),
                n(c, "dim"),
                c["time_major"].as_bool().unwrap(),
                c["causal"].as_bool().unwrap(),
                n(c, "offset"),
                dtype,
            ),
            "rope" => device.rope(
                &values(c, "input"),
                n(c, "batch"),
                n(c, "seq"),
                n(c, "heads"),
                n(c, "dim"),
                n(c, "offset"),
                c["theta"].as_f64().unwrap() as f32,
                dtype,
            ),
            "snake" => {
                let shape = c["input_shape"].as_array().unwrap();
                device.snake(
                    &values(c, "input"),
                    &values(c, "alpha"),
                    shape[1].as_u64().unwrap() as usize,
                    shape[2].as_u64().unwrap() as usize,
                    dtype,
                )
            }
            "normal_from_uniform" => device.normal_from_uniform(&values(c, "input"), dtype),
            _ => continue,
        }
        .unwrap();
        count += 1;
        let expected = values(c, "expected");
        assert_eq!(actual.len(), expected.len());
        let err = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        let tolerance = if dtype == Music3DType::F32 { 2e-6 } else { 0.0 };
        if err > tolerance {
            let differences = actual.iter().zip(&expected).filter(|(a, e)| a != e).count();
            failed.push(format!(
                "{} max={err} differing={differences}/{}",
                c["name"],
                actual.len()
            ));
        }
    }
    assert!(count >= 27, "fixture coverage: {count}");
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
