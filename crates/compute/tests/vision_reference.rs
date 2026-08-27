//! The vision tower's FP32 reference kernels.
//!
//! These are the ground truth `crates/gpu`'s Metal kernels are checked
//! against, so a wrong function here is not caught by anything downstream --
//! the GPU parity test would agree with it. Each case therefore pins the
//! function against an INDEPENDENT statement of what it computes (a
//! hand-worked value, an algebraic invariant, or a second formulation)
//! rather than against a recomputation of the same expression.

use turbospark_compute::vision::{
    attention_scale, bidirectional_attention, gelu_erf, gelu_tanh_vision, layer_norm, matmul_bias,
    rope_vision_2d,
};

// ---------------------------------------------------------------- LayerNorm

#[test]
fn layer_norm_produces_zero_mean_unit_variance_before_the_affine() {
    // The defining property, stated independently of the implementation:
    // with an identity affine the output must have mean 0 and variance 1.
    let x = [3.0f32, -1.5, 7.25, 0.5, -4.0, 2.0, 11.0, -0.25];
    let weight = [1.0f32; 8];
    let bias = [0.0f32; 8];
    let y = layer_norm(&x, &weight, &bias, 0.0);

    let mean: f32 = y.iter().sum::<f32>() / 8.0;
    let var: f32 = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 8.0;
    assert!(mean.abs() < 1e-6, "mean {mean}");
    assert!((var - 1.0).abs() < 1e-5, "var {var}");
}

#[test]
fn layer_norm_subtracts_the_mean_where_rms_norm_does_not() {
    // The one-line difference from every other norm in this workspace, and
    // the reason reaching for `rms_norm` here produces plausible output. On
    // a mean-shifted input the two disagree structurally: LayerNorm is
    // INVARIANT to the shift and RMSNorm is not.
    let base = [1.0f32, -2.0, 3.0, -4.0];
    let shifted: Vec<f32> = base.iter().map(|v| v + 50.0).collect();
    let weight = [1.0f32; 4];
    let bias = [0.0f32; 4];

    let a = layer_norm(&base, &weight, &bias, 1e-6);
    let b = layer_norm(&shifted, &weight, &bias, 1e-6);
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-4, "shift changed the result: {x} vs {y}");
    }

    let r = turbospark_compute::rms_norm(&shifted, &weight, 1e-6);
    assert!(
        r.iter().zip(&b).any(|(x, y)| (x - y).abs() > 0.5),
        "rms_norm and layer_norm agree on a mean-shifted input; \
         the fixture cannot tell them apart"
    );
}

#[test]
fn layer_norm_applies_weight_then_bias_in_that_order() {
    // `w * x_hat + b`, not `w * (x_hat + b)`. The two coincide at `w == 1`,
    // which is exactly the fixture a careless test would use.
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let weight = [3.0f32; 4];
    let bias = [10.0f32; 4];
    let scaled = layer_norm(&x, &weight, &bias, 0.0);
    let plain = layer_norm(&x, &[1.0; 4], &[0.0; 4], 0.0);
    for (got, hat) in scaled.iter().zip(&plain) {
        assert!((got - (3.0 * hat + 10.0)).abs() < 1e-5, "{got}");
    }
}

#[test]
fn layer_norm_eps_is_inside_the_square_root() {
    // `sqrt(var + eps)`, not `sqrt(var) + eps`.
    //
    // A CONSTANT input cannot tell these apart, which is the fixture the
    // obvious version of this test reaches for: there `x - mean` is exactly
    // zero, so both spellings return zero however different the divisor is.
    // The eps has to be comparable to the variance AND the numerator has to
    // be nonzero, so this uses a large eps against an ordinary spread --
    // legitimate because eps is a runtime argument to the kernel, and the
    // question is the formula's shape rather than the shipped value.
    let x = [1.0f32, -1.0, 1.0, -1.0]; // mean 0, var 1
    let y = layer_norm(&x, &[1.0; 4], &[0.0; 4], 1.0);
    // Inside:  1 / sqrt(1 + 1) = 0.70710678
    // Outside: 1 / (sqrt(1) + 1) = 0.5
    assert!(
        (y[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
        "{y:?}: eps looks like it is outside the square root"
    );
}

#[test]
fn layer_norm_is_finite_on_a_constant_input() {
    // The separate property the case above cannot also carry: at zero
    // variance the divisor is `sqrt(eps)` rather than zero, so nothing
    // divides by zero and a flat patch normalizes to its bias.
    let y = layer_norm(&[7.0f32; 6], &[1.0; 6], &[0.5; 6], 1e-6);
    assert!(y.iter().all(|v| v.is_finite()), "{y:?}");
    assert!(y.iter().all(|v| (v - 0.5).abs() < 1e-6), "{y:?}");
}

// --------------------------------------------------------------------- GELU

#[test]
fn the_two_gelus_are_different_functions() {
    // GUARDS EVERY OTHER GELU CASE. The tanh approximation and the exact erf
    // form agree to ~3e-4 at their worst, so a fixture drawn from a small
    // range, or a tolerance set at 1e-3, cannot tell them apart -- and the
    // tower uses BOTH, tanh in each block's MLP and erf in the merger.
    let xs: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
    let tanh = gelu_tanh_vision(&xs);
    let erf = gelu_erf(&xs);
    let worst = tanh
        .iter()
        .zip(&erf)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 1e-4,
        "the two GELUs differ by at most {worst}; this fixture cannot \
         discriminate them and no test using it proves anything"
    );
    // And they are CLOSE, which is why the guard above is needed at all.
    assert!(
        worst < 1e-2,
        "worst {worst} is too large to be these two GELUs"
    );
}

#[test]
fn gelu_erf_matches_hand_computed_values() {
    // Independent values: `0.5 * x * (1 + erf(x / sqrt(2)))` evaluated at
    // points where erf is tabulated. Not a recomputation of the same code.
    for (x, want) in [
        (0.0f32, 0.0f32),
        // Written to the digits f32 actually holds: clippy refuses a literal
        // carrying more, and a longer one would be the same number anyway.
        (1.0, 0.841_344_7),    // 0.5 * (1 + erf(1 / sqrt(2)))
        (-1.0, -0.158_655_25), // = 1.0 - 0.8413447, by the odd symmetry
        (2.0, 1.954_499_7),
        (-2.0, -0.045_500_264),
        (3.0, 2.995_950_2),
    ] {
        let got = gelu_erf(&[x])[0];
        assert!(
            (got - want).abs() < 1e-5,
            "gelu_erf({x}) = {got}, want {want}"
        );
    }
}

#[test]
fn gelu_erf_saturates_to_identity_and_to_zero() {
    // The two asymptotes, which no tabulated point covers.
    for x in [8.0f32, 20.0, 100.0] {
        let got = gelu_erf(&[x])[0];
        assert!((got - x).abs() < 1e-4, "gelu_erf({x}) = {got}, want ~{x}");
        let neg = gelu_erf(&[-x])[0];
        assert!(neg.abs() < 1e-4, "gelu_erf({}) = {neg}, want ~0", -x);
    }
}

// --------------------------------------------------------------------- RoPE

#[test]
fn rope_vision_2d_pairs_halves_not_neighbours() {
    // Half-split (NeoX): element `i` rotates with `i + head_dim/2`. Adjacent
    // pairing is the other convention in this workspace and produces a
    // finite, wrong result. A freq row that is ZERO everywhere except one
    // slot makes the pairing directly visible.
    let head: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
    let mut freqs = vec![0.0f32; 4];
    freqs[0] = std::f32::consts::FRAC_PI_2; // quarter turn on pair (0, 4)

    let out = rope_vision_2d(&head, &freqs);
    // (a, b) = (1, 5) rotated by 90 degrees -> (-5, 1).
    assert!((out[0] - -5.0).abs() < 1e-5, "{out:?}");
    assert!((out[4] - 1.0).abs() < 1e-5, "{out:?}");
    // Everything else untouched: a zero angle is the identity.
    for i in [1usize, 2, 3, 5, 6, 7] {
        assert!((out[i] - head[i]).abs() < 1e-5, "index {i} moved: {out:?}");
    }
}

#[test]
fn rope_vision_2d_preserves_each_pairs_norm() {
    // A rotation is orthogonal, so `a^2 + b^2` is invariant per pair. This
    // catches a sign error or a swapped sin/cos that a single hand-worked
    // angle can miss.
    let head: Vec<f32> = (0..72).map(|i| ((i * 37) % 19) as f32 - 9.0).collect();
    let freqs: Vec<f32> = (0..36).map(|i| i as f32 * 0.213 - 2.0).collect();
    let out = rope_vision_2d(&head, &freqs);
    for i in 0..36 {
        let before = head[i] * head[i] + head[i + 36] * head[i + 36];
        let after = out[i] * out[i] + out[i + 36] * out[i + 36];
        assert!(
            (before - after).abs() < 1e-3 * before.max(1.0),
            "pair {i}: {before} -> {after}"
        );
    }
}

#[test]
fn rope_vision_2d_composes_by_angle_addition() {
    // Rotating by `a` then by `b` equals rotating by `a + b`. An independent
    // statement of what the function is, which no single evaluation gives.
    let head: Vec<f32> = (0..8).map(|i| (i as f32 * 1.7).sin()).collect();
    let a: Vec<f32> = (0..4).map(|i| 0.3 + i as f32 * 0.11).collect();
    let b: Vec<f32> = (0..4).map(|i| -0.7 + i as f32 * 0.29).collect();
    let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let twice = rope_vision_2d(&rope_vision_2d(&head, &a), &b);
    let once = rope_vision_2d(&head, &sum);
    for (x, y) in twice.iter().zip(&once) {
        assert!((x - y).abs() < 1e-5, "{twice:?} vs {once:?}");
    }
}

// ---------------------------------------------------------------- attention

#[test]
fn bidirectional_attention_lets_the_first_patch_see_the_last() {
    // THE PROPERTY THAT SEPARATES THIS FROM `causal_attention`, and the one
    // whose absence degrades an image rather than corrupting it. Position 0
    // is given a query that matches ONLY the last key; under a causal mask
    // it could not reach it and would return position 0's own value.
    let (seq, heads, head_dim) = (4usize, 1usize, 2usize);
    let mut q = vec![0.0f32; seq * heads * head_dim];
    let mut k = vec![0.0f32; seq * heads * head_dim];
    let mut v = vec![0.0f32; seq * heads * head_dim];
    // Keys: position j points along a direction unique to j.
    for j in 0..seq {
        k[j * head_dim] = if j == seq - 1 { 20.0 } else { -20.0 };
        k[j * head_dim + 1] = 0.0;
        v[j * head_dim] = j as f32;
        v[j * head_dim + 1] = 100.0 + j as f32;
    }
    q[0] = 1.0; // aligns with the last key alone

    let out = bidirectional_attention(&q, &k, &v, seq, heads, head_dim, 1.0);
    assert!(
        (out[0] - (seq - 1) as f32).abs() < 1e-3,
        "position 0 did not reach the last value: {out:?}"
    );
}

#[test]
fn bidirectional_attention_weights_sum_to_one() {
    // With every value row set to the same constant, the output must be that
    // constant at every position and head -- a direct statement that the
    // softmax normalizes, independent of what the scores are.
    let (seq, heads, head_dim) = (5usize, 3usize, 4usize);
    let n = seq * heads * head_dim;
    let q: Vec<f32> = (0..n).map(|i| ((i * 13) % 7) as f32 - 3.0).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i * 29) % 11) as f32 - 5.0).collect();
    let v = vec![2.5f32; n];
    let out = bidirectional_attention(&q, &k, &v, seq, heads, head_dim, 0.5);
    for (i, &value) in out.iter().enumerate() {
        assert!((value - 2.5).abs() < 1e-4, "slot {i} = {value}");
    }
}

#[test]
fn bidirectional_attention_keeps_heads_independent() {
    // Perturbing head 0's query must not move head 1's output. A stride
    // error in the `[seq, heads, head_dim]` indexing is the ordinary way to
    // break this, and it produces finite, plausible output.
    let (seq, heads, head_dim) = (4usize, 2usize, 3usize);
    let n = seq * heads * head_dim;
    let base_q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    let scale = attention_scale(head_dim);

    let before = bidirectional_attention(&base_q, &k, &v, seq, heads, head_dim, scale);
    let mut poked = base_q.clone();
    for i in 0..seq {
        poked[(i * heads) * head_dim] += 5.0; // head 0 only
    }
    let after = bidirectional_attention(&poked, &k, &v, seq, heads, head_dim, scale);

    let mut head1_moved = false;
    let mut head0_moved = false;
    for i in 0..seq {
        for d in 0..head_dim {
            let h0 = (i * heads) * head_dim + d;
            let h1 = (i * heads + 1) * head_dim + d;
            if (before[h0] - after[h0]).abs() > 1e-5 {
                head0_moved = true;
            }
            if (before[h1] - after[h1]).abs() > 1e-5 {
                head1_moved = true;
            }
        }
    }
    assert!(head0_moved, "the perturbation did not reach head 0 at all");
    assert!(
        !head1_moved,
        "head 1 moved when only head 0's query changed"
    );
}

#[test]
fn bidirectional_attention_survives_the_towers_real_activation_scale() {
    // `docs/VISION_PHASE0.md` item 3 measures absmax 8,384 at block 26. An
    // unshifted `exp` overflows well before that, so the max-subtraction is
    // load-bearing on real data rather than on a contrived input.
    let (seq, heads, head_dim) = (6usize, 2usize, 8usize);
    let n = seq * heads * head_dim;
    let q: Vec<f32> = (0..n).map(|i| 8000.0 + (i as f32)).collect();
    let k: Vec<f32> = (0..n).map(|i| 8000.0 - (i as f32)).collect();
    let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let out = bidirectional_attention(&q, &k, &v, seq, heads, head_dim, attention_scale(head_dim));
    assert!(
        out.iter().all(|v| v.is_finite()),
        "overflowed: {:?}",
        &out[..8]
    );
}

#[test]
fn attention_scale_is_the_inverse_square_root() {
    assert!((attention_scale(72) - 1.0 / 72.0f32.sqrt()).abs() < 1e-7);
    assert!((attention_scale(64) - 0.125).abs() < 1e-7);
}

// ------------------------------------------------------------------- matmul

#[test]
fn matmul_bias_reads_the_weight_row_major_by_output() {
    // A `nn.Linear` weight is `[n_out, k]`. Reading it `[k, n_out]` instead
    // transposes every projection while keeping every length check happy,
    // so this pins the orientation with an asymmetric shape and hand-worked
    // values rather than with a square fixture.
    let a = [1.0f32, 2.0, 3.0]; // m = 1, k = 3
    let b = [
        1.0f32, 0.0, 0.0, // output 0 selects a[0]
        0.0, 10.0, 0.0, // output 1 selects 10 * a[1]
    ]; // n = 2, k = 3
    let out = matmul_bias(&a, &b, None, 1, 3, 2);
    assert_eq!(out, vec![1.0, 20.0]);
}

#[test]
fn matmul_bias_adds_the_bias_per_output_column() {
    let a = [1.0f32, 1.0];
    let b = [1.0f32, 1.0, 2.0, 2.0]; // n = 2, k = 2
    let out = matmul_bias(&a, &b, Some(&[100.0, -100.0]), 1, 2, 2);
    assert_eq!(out, vec![102.0, -96.0]);
}

#[test]
fn matmul_bias_keeps_rows_independent() {
    // Two rows through the same weight must equal two separate single-row
    // calls. Catches an accumulator that is not reset between rows.
    let (m, k, n) = (3usize, 5usize, 4usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.31).sin()).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.77).cos()).collect();
    let bias: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let batched = matmul_bias(&a, &b, Some(&bias), m, k, n);
    for row in 0..m {
        let single = matmul_bias(&a[row * k..(row + 1) * k], &b, Some(&bias), 1, k, n);
        for col in 0..n {
            let got = batched[row * n + col];
            assert!((got - single[col]).abs() < 1e-5, "row {row} col {col}");
        }
    }
}

#[test]
fn matmul_bias_runs_at_the_towers_real_projection_shapes() {
    // The four shapes the tower actually dispatches, at one row, so a
    // dimension swapped between `k` and `n` fails here rather than at a
    // Metal buffer-size check three milestones later.
    for (k, n, label) in [
        (1536usize, 1152usize, "patch_embed"),
        (1152, 3456, "qkv"),
        (1152, 4304, "mlp.fc1"),
        (4304, 1152, "mlp.fc2"),
        (4608, 4608, "merger.fc1"),
        (4608, 5120, "merger.fc2"),
    ] {
        let a: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let out = matmul_bias(&a, &b, None, 1, k, n);
        assert_eq!(out.len(), n, "{label}");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{label} produced non-finite"
        );
    }
}
