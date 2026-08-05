//! Fast Walsh-Hadamard transform reference. Ported from
//! `Support/Reference/Primitives/Wht.swift`.

/// Normalized Walsh-Hadamard transform. `x.len()` must be a power of two.
pub fn wht(x: &[f32]) -> Vec<f32> {
    let d = x.len();
    assert!(d > 0 && (d & (d - 1)) == 0, "D must be a power of two");
    if d == 1 {
        return x.to_vec();
    }
    walsh(x)
}

fn walsh(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 1 {
        return x.to_vec();
    }
    let half = n / 2;
    let lo = walsh(&x[..half]);
    let hi = walsh(&x[half..]);
    let inv_sqrt2 = 1.0 / 2f32.sqrt();
    let mut y = vec![0f32; n];
    for i in 0..half {
        y[i] = (lo[i] + hi[i]) * inv_sqrt2;
        y[half + i] = (lo[i] - hi[i]) * inv_sqrt2;
    }
    y
}
