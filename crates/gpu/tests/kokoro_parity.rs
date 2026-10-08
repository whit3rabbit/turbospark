#![cfg(target_os = "macos")]
//! Real Metal against independently generated pinned MLX 0.31.2 tensors.
//! Regenerate with tests/reference/generate_kokoro.py, never during a test.
use serde_json::Value;
use turbospark_gpu::{Music3Device, Music3Encoding};
fn values(row: &Value, key: &str) -> Vec<f32> {
    row[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn number(row: &Value, key: &str) -> usize {
    row[key].as_u64().unwrap() as usize
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/kokoro_mlx_0_31_2.json")).unwrap()
}
fn precision_fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/kokoro_precision_mlx_0_31_2.json")).unwrap()
}
#[test]
fn kokoro_matched_input_bert_attention_preserves_pinned_f32_arithmetic() {
    let fixture = precision_fixture();
    assert_eq!(fixture["mlx_version"], "0.31.2");
    let row = &fixture["attention"];
    let device = Music3Device::new().expect("real Metal device required");
    let actual = device
        .kokoro_attention_f32(
            &values(row, "q"),
            &values(row, "k"),
            &values(row, "v"),
            number(row, "seq"),
            number(row, "dim"),
        )
        .unwrap();
    // These real BERT inputs cause waveform failure despite passing 1e-5 fixtures.
    close(&actual, &values(row, "expected"), 0.0);
}
#[test]
fn kokoro_matched_input_bert_layernorm_preserves_pinned_f32_arithmetic() {
    let fixture = precision_fixture();
    let row = &fixture["norm"];
    let device = Music3Device::new().expect("real Metal device required");
    let actual = device
        .kokoro_layer_norm_f32(
            &values(row, "x"),
            &values(row, "weight"),
            Some(&values(row, "bias")),
            number(row, "rows"),
            number(row, "cols"),
            1e-12,
        )
        .unwrap();
    close(&actual, &values(row, "expected"), 0.0);
}
#[test]
fn kokoro_matched_input_gelu_preserves_pinned_f32_arithmetic() {
    let fixture = precision_fixture();
    let row = &fixture["gelu"];
    let device = Music3Device::new().expect("real Metal device required");
    let x = values(row, "x");
    let actual = device.kokoro_gelu_f32(&x).unwrap();
    let expected = values(row, "expected");
    for i in (0..x.len()).filter(|&i| actual[i] != expected[i]).take(12) {
        println!(
            "GELU input {}, expected {} ({}), actual {} ({})",
            x[i],
            expected[i],
            expected[i].to_bits(),
            actual[i],
            actual[i].to_bits()
        );
    }
    close(&actual, &expected, 0.0);
}
#[test]
fn kokoro_matched_input_weight_norm_preserves_pinned_f32_arithmetic() {
    let fixture = precision_fixture();
    let device = Music3Device::new().expect("real Metal device required");
    for row in fixture["folds"].as_array().unwrap() {
        let shape: Vec<usize> = row["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let actual = device
            .kokoro_weight_norm_f32(
                &values(row, "x"),
                &values(row, "g"),
                shape[0],
                shape[1],
                shape[2],
            )
            .unwrap();
        close(&actual, &values(row, "expected"), 0.0);
    }
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max);
    assert!(actual.iter().all(|v| v.is_finite()));
    assert!(
        maximum <= tolerance,
        "maximum error {maximum} exceeds {tolerance}"
    );
}
#[test]
fn kokoro_f32_linear_conv_and_lstm_match_independent_mlx() {
    let fixture = fixture();
    assert_eq!(fixture["mlx_version"], "0.31.2");
    let device = Music3Device::new().expect("real Metal device required");
    for row in fixture["operators"].as_array().unwrap() {
        let x = values(row, "x");
        let bias = values(row, "bias");
        let seq = number(row, "seq");
        let ic = number(row, "ic");
        let (w, shape) = match row["op"].as_str().unwrap() {
            "linear" => (values(row, "w"), vec![number(row, "oc"), ic]),
            "conv" => {
                let oc = number(row, "oc");
                let groups = number(row, "groups");
                let kernel = number(row, "kernel");
                (
                    values(row, "w"),
                    if row["transpose"].as_bool().unwrap() {
                        vec![ic, oc / groups, kernel]
                    } else {
                        vec![oc, ic / groups, kernel]
                    },
                )
            }
            "lstm" => (
                values(row, "wh"),
                vec![4 * number(row, "hidden"), number(row, "hidden")],
            ),
            _ => panic!("unknown reference op"),
        };
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = device
            .load_weight(&shape, &bytes, Music3Encoding::F32, &[], &[], &[])
            .unwrap();
        let output = match row["op"].as_str().unwrap() {
            "linear" => weight
                .linear_f32(&x, Some(&bias), seq, ic, number(row, "oc"))
                .unwrap(),
            "conv" => weight
                .convolution_f32_grouped(
                    &x,
                    Some(&bias),
                    ic,
                    number(row, "oc"),
                    number(row, "kernel"),
                    number(row, "stride"),
                    number(row, "padding"),
                    number(row, "dilation"),
                    row["transpose"].as_bool().unwrap(),
                    number(row, "groups"),
                )
                .unwrap(),
            "lstm" => weight
                .lstm_recurrence(
                    &values(row, "projection"),
                    number(row, "hidden"),
                    row["backward"].as_bool().unwrap(),
                )
                .unwrap(),
            _ => unreachable!(),
        };
        close(&output, &values(row, "expected"), 1e-5);
    }
    println!("real Metal: independent MLX F32 linear, ordinary/grouped/transpose convolution and both recurrent directions passed");
}
#[test]
fn kokoro_device_shapes_refuse_invalid_groups_hidden_and_indices() {
    let device = Music3Device::new().unwrap();
    let bytes: Vec<u8> = [1.0f32; 12].iter().flat_map(|v| v.to_le_bytes()).collect();
    let conv = device
        .load_weight(&[4, 1, 3], &bytes, Music3Encoding::F32, &[], &[], &[])
        .unwrap();
    assert!(conv
        .convolution_f32_grouped(&[0.0; 20], None, 4, 4, 3, 2, 0, 1, true, 0)
        .is_err());
    assert!(conv
        .convolution_f32_grouped(&[0.0; 20], None, 4, 4, 3, 2, 0, 1, true, 3)
        .is_err());
    assert!(conv.lstm_recurrence(&[0.0; 12], 0, false).is_err());
    assert!(conv.lstm_recurrence(&[0.0; 12], usize::MAX, false).is_err());
    let emb = device
        .load_weight(&[3, 4], &bytes, Music3Encoding::F32, &[], &[], &[])
        .unwrap();
    assert!(emb.embedding(&[-1], 4).is_err());
    assert!(emb.embedding(&[3], 4).is_err());
    assert!(emb.linear_f32(&[0.0; 4], None, usize::MAX, 4, 3).is_err());
}

#[test]
fn kokoro_scan_stft_and_gaussian_match_mlx_0_31_2() {
    let fixture = fixture();
    let device = Music3Device::new().unwrap();
    for row in fixture["scans"].as_array().unwrap() {
        let actual = device
            .cumulative_sum_f32(
                &values(row, "x"),
                number(row, "rows"),
                number(row, "columns"),
            )
            .unwrap();
        close(&actual, &values(row, "expected"), 1e-6);
    }
    let row = &fixture["stft"];
    close(
        &device
            .stft20_magnitude_phase(&values(row, "signal"), &values(row, "window"), 5)
            .unwrap(),
        &values(row, "expected"),
        1e-5,
    );
    for row in fixture["rng"].as_array().unwrap() {
        close(
            &device
                .normal_from_uniform(
                    &values(row, "normal_uniform"),
                    turbospark_gpu::Music3DType::F32,
                )
                .unwrap(),
            &values(row, "noise"),
            5e-6,
        );
    }
    assert!(device.cumulative_sum_f32(&[0.0], 0, 1).is_err());
    assert!(device
        .stft20_magnitude_phase(&[0.0; 300], &[0.0; 20], 4)
        .is_err());
}
