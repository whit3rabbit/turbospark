#![cfg(target_os = "macos")]
//! F32 stages over dense 16-bit weights: the tiled kernels are an opt-in per
//! weight (`Music3Weight::tiled_for_f32`, set for Music 3 Turbo), and every
//! other weight and dtype keeps its kernel.
//!
//! ```sh
//! cargo test -p turbospark-gpu --test music3_tiled_f32
//! ```
use half::f16;
use turbospark_gpu::{Music3DType, Music3Device, Music3Encoding, Music3Weight};

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// An F16 `[out, inn]` weight and the f32 values it holds.
fn f16_weight(
    device: &Music3Device,
    out: usize,
    inn: usize,
    seed: u64,
) -> (Music3Weight, Vec<f32>) {
    let held: Vec<f32> = values(out * inn, seed)
        .into_iter()
        .map(|v| f16::from_f32(v).to_f32())
        .collect();
    let bytes: Vec<u8> = held
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let weight = device
        .load_weight(&[out, inn], &bytes, Music3Encoding::F16, &[], &[], &[])
        .unwrap();
    (weight, held)
}

#[test]
fn f32_stages_take_tiled_kernels_only_on_opted_in_16_bit_weights() {
    let device = Music3Device::new().unwrap();
    let (out, inn) = (128, 256);
    let f32 = Music3DType::F32;
    let (plain, _) = f16_weight(&device, out, inn, 1);
    let (tiled, _) = f16_weight(&device, out, inn, 1);
    let tiled = tiled.tiled_for_f32();
    for rows in [32, 173] {
        assert_eq!(
            plain.linear_kernel(rows, inn, out, f32).unwrap(),
            "music3_linear"
        );
        assert_eq!(
            tiled.linear_kernel(rows, inn, out, f32).unwrap(),
            "music3_linear_tiled_32x32"
        );
    }
    assert_eq!(
        tiled.linear_kernel(8, inn, out, f32).unwrap(),
        "music3_linear_tiled_16x64"
    );
    // A single row stays a per-element GEMV either way.
    assert_eq!(
        tiled.linear_kernel(1, inn, out, f32).unwrap(),
        "music3_linear"
    );
    // 16-bit stages are planned as before.
    for dtype in [Music3DType::Bf16, Music3DType::F16] {
        for rows in [1, 8, 32, 173] {
            assert_eq!(
                tiled.linear_kernel(rows, inn, out, dtype).unwrap(),
                plain.linear_kernel(rows, inn, out, dtype).unwrap(),
                "{dtype:?} rows {rows}"
            );
        }
    }
    // An F32-encoded weight ignores the flag: it is the float32 reference.
    let bytes: Vec<u8> = values(out * inn, 2)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let dense = device
        .load_weight(&[out, inn], &bytes, Music3Encoding::F32, &[], &[], &[])
        .unwrap()
        .tiled_for_f32();
    assert_eq!(
        dense.linear_kernel(32, inn, out, f32).unwrap(),
        "music3_linear"
    );
}

/// The tiled F32 product is a float32 product: within float32 accumulation
/// error of an f64 reference, as the scalar kernel is. BF16 stages over the
/// same weight are unchanged bit for bit.
#[test]
fn tiled_f32_linear_stays_within_float32_error_and_leaves_bf16_bits() {
    let device = Music3Device::new().unwrap();
    let (rows, out, inn) = (173, 192, 2048);
    let (plain, held) = f16_weight(&device, out, inn, 3);
    let (tiled, _) = f16_weight(&device, out, inn, 3);
    let tiled = tiled.tiled_for_f32();
    let x = values(rows * inn, 4);
    let bias = values(out, 5);
    let f32 = Music3DType::F32;
    let scalar = plain
        .linear_typed(&x, Some(&bias), rows, inn, out, f32)
        .unwrap();
    let fast = tiled
        .linear_typed(&x, Some(&bias), rows, inn, out, f32)
        .unwrap();
    let mut differs = 0;
    for r in 0..rows {
        for o in 0..out {
            let mut want = bias[o] as f64;
            let mut magnitude = (bias[o] as f64).abs();
            for k in 0..inn {
                let p = x[r * inn + k] as f64 * held[o * inn + k] as f64;
                want += p;
                magnitude += p.abs();
            }
            // Float32 summation bound for K terms, with headroom.
            let bound = 2.0 * inn as f64 * f32::EPSILON as f64 * magnitude;
            let i = r * out + o;
            for (got, what) in [(fast[i], "tiled"), (scalar[i], "scalar")] {
                assert!(
                    (got as f64 - want).abs() <= bound,
                    "{what} [{r},{o}]: {got} vs {want} (bound {bound})"
                );
            }
            differs += (fast[i].to_bits() != scalar[i].to_bits()) as usize;
        }
    }
    // The opt-in is observable: a different accumulation order.
    assert!(differs > 0, "tiled and scalar F32 outputs are identical");
    let bf = Music3DType::Bf16;
    let want = plain
        .linear_typed(&x, Some(&bias), rows, inn, out, bf)
        .unwrap();
    let got = tiled
        .linear_typed(&x, Some(&bias), rows, inn, out, bf)
        .unwrap();
    assert!(got
        .iter()
        .zip(&want)
        .all(|(g, w)| g.to_bits() == w.to_bits()));
}
