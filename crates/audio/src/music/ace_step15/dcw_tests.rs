//! DCW parity against upstream's `DCWCorrector` over pytorch-wavelets
//! (`testdata/ace_step15_dcw/`, from `tools/gen_ace_step15_dcw_fixtures.py`).

use std::path::Path;

use super::dcw::{Dcw, DcwMode};
use super::tests::{read_npy, rel_err};

/// Measured on M4 Max: worst case 1.1e-7 relative (f32 rounding of the
/// filter bank) while the correction itself moves values by 2e-2 to
/// 2e-1. Dropping the zero-pad pair, swapping the band schedules, or
/// skipping the reconstruction scale fail it by 1e-2 or more.
const TOL: f32 = 1e-6;

#[test]
fn dcw_matches_upstream() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/ace_step15_dcw");
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mode = match c["mode"].as_str().unwrap() {
            "low" => DcwMode::Low,
            "high" => DcwMode::High,
            "double" => DcwMode::Double,
            "pix" => DcwMode::Pix,
            other => panic!("unknown mode {other}"),
        };
        let dcw = Dcw {
            mode,
            scaler: c["scaler"].as_f64().unwrap() as f32,
            high_scaler: c["high_scaler"].as_f64().unwrap() as f32,
        };
        let (shape, mut x) = read_npy(&dir.join(format!("{name}_x.npy")));
        let (_, y) = read_npy(&dir.join(format!("{name}_y.npy")));
        let (_, want) = read_npy(&dir.join(format!("{name}_out.npy")));
        let before = x.clone();
        dcw.apply(
            &mut x,
            &y,
            shape[0],
            shape[1],
            c["t"].as_f64().unwrap() as f32,
        )
        .unwrap();
        let e = rel_err(&x, &want);
        eprintln!(
            "dcw {name}: {e:.2e} (change {:.2e})",
            rel_err(&before, &want)
        );
        assert!(e <= TOL, "{name}: relative error {e:.3e}");
    }
}

#[test]
fn dcw_rejects_mismatched_shapes_and_short_circuits() {
    let dcw = Dcw::TURBO;
    let mut x = vec![1.0f32; 6];
    assert!(dcw.apply(&mut x, &[0.0; 5], 3, 2, 0.5).is_err());
    let off = Dcw {
        scaler: 0.0,
        high_scaler: 0.0,
        ..Dcw::TURBO
    };
    assert!(!off.is_active());
    let mut y = vec![0.25f32, -1.0, 3.0];
    off.apply(&mut y, &[9.0, 9.0, 9.0], 3, 1, 0.5).unwrap();
    assert_eq!(y, vec![0.25, -1.0, 3.0]);
}
