#![cfg(target_os = "macos")]
//! Independent pinned MLX inputs covering arithmetic missed by coarse gates.
use serde_json::Value;
use turbospark_gpu::{Music3Device, Music3Encoding};
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/kokoro_contracts_mlx_0_31_2.json")).unwrap()
}
fn values(v: &Value, k: &str) -> Vec<f32> {
    v[k].as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn n(v: &Value, k: &str) -> usize {
    v[k].as_u64().unwrap() as usize
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(a.iter().all(|x| x.is_finite()));
    assert_eq!(max, 0.0, "pinned arithmetic maximum {max}");
}
fn weight(d: &Music3Device, v: &Value, shape: &[usize]) -> turbospark_gpu::Music3Weight {
    let bytes: Vec<u8> = values(v, "w")
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    d.load_weight(shape, &bytes, Music3Encoding::F32, &[], &[], &[])
        .unwrap()
}
#[test]
fn kokoro_splitk_gemv_and_remainders() {
    let f = fixture();
    assert_eq!(f["mlx_version"], "0.31.2");
    let d = Music3Device::new().unwrap();
    for v in f["linear"].as_array().unwrap() {
        let a = weight(&d, v, &[n(v, "oc"), n(v, "ic")])
            .linear_f32(
                &values(v, "x"),
                Some(&values(v, "bias")),
                n(v, "rows"),
                n(v, "ic"),
                n(v, "oc"),
            )
            .unwrap();
        close(&a, &values(v, "expected"));
    }
}
#[test]
fn kokoro_precise_layernorm() {
    let f = fixture();
    let v = &f["layernorm"];
    let d = Music3Device::new().unwrap();
    close(
        &d.kokoro_layer_norm_f32(
            &values(v, "x"),
            &values(v, "w"),
            Some(&values(v, "bias")),
            n(v, "rows"),
            n(v, "cols"),
            1e-12,
        )
        .unwrap(),
        &values(v, "expected"),
    );
}
#[test]
fn kokoro_explicit_normalization_layout_and_wide_reductions() {
    let f = fixture();
    let d = Music3Device::new().unwrap();
    for v in f["norm"].as_array().unwrap() {
        println!(
            "norm rows {} cols {} columns {}",
            v["rows"], v["cols"], v["columns"]
        );
        close(
            &d.kokoro_normalize_f32(
                &values(v, "x"),
                n(v, "rows"),
                n(v, "cols"),
                1e-5,
                v["columns"].as_bool().unwrap(),
            )
            .unwrap(),
            &values(v, "expected"),
        );
    }
}
#[test]
fn kokoro_recurrent_gemv_and_unaries() {
    let f = fixture();
    let d = Music3Device::new().unwrap();
    for v in f["lstm"].as_array().unwrap() {
        close(
            &weight(&d, v, &[4 * n(v, "hidden"), n(v, "hidden")])
                .lstm_recurrence(
                    &values(v, "projection"),
                    n(v, "hidden"),
                    v["backward"].as_bool().unwrap(),
                )
                .unwrap(),
            &values(v, "expected"),
        );
    }
}
#[test]
fn kokoro_channel_tap_order_and_depthwise_mma() {
    let f = fixture();
    let d = Music3Device::new().unwrap();
    for v in f["conv"].as_array().unwrap() {
        let trans = v["transpose"].as_bool().unwrap();
        let shape = if trans {
            vec![n(v, "ic"), n(v, "oc") / n(v, "groups"), n(v, "kernel")]
        } else {
            vec![n(v, "oc"), n(v, "ic") / n(v, "groups"), n(v, "kernel")]
        };
        close(
            &weight(&d, v, &shape)
                .convolution_f32_grouped(
                    &values(v, "x"),
                    Some(&values(v, "bias")),
                    n(v, "ic"),
                    n(v, "oc"),
                    n(v, "kernel"),
                    n(v, "stride"),
                    n(v, "padding"),
                    1,
                    trans,
                    n(v, "groups"),
                )
                .unwrap(),
            &values(v, "expected"),
        );
    }
}
#[test]
fn kokoro_precise_source_sine_and_tanh() {
    let f = fixture();
    let v = &f["unary"];
    let d = Music3Device::new().unwrap();
    for (sine, key) in [(true, "sin"), (false, "tanh")] {
        close(
            &d.kokoro_unary_f32(&values(v, "x"), sine).unwrap(),
            &values(v, key),
        );
    }
}
#[test]
fn kokoro_precise_normal_transform() {
    let f = fixture();
    let v = &f["normal"];
    let d = Music3Device::new().unwrap();
    close(
        &d.kokoro_normal_from_uniform_f32(&values(v, "x")).unwrap(),
        &values(v, "expected"),
    );
}
