//! The Mamba2-style selective state-space recurrence behind the falcon_h1
//! hybrid backbone.
//!
//! Port of `mlx_audio/lm/models/ssm.py` at the pinned mlx-audio commit.
//! The reference dispatches between a Metal initial-scan kernel, a chunked
//! SSD "surrogate attention" matmul form (`ssm_attn`, `step = 256`), and a
//! Metal single-step kernel; all three implement the same linear
//! recurrence, which is what this port computes once:
//!
//! ```text
//! dt_t = clip(softplus(dt + dt_bias), limit)      # per head
//! S_t  = exp(A * dt_t) * S_(t-1) + dt_t * x_t (outer) B_t
//! y_t  = (< S_t , C_t >) + D (x) x_t
//! ```
//!
//! with `A = -exp(A_log)`, head `h` reading the state group
//! `h / (heads / groups)`, and the first state starting from zero when the
//! caller passes `None`. The masked and length-truncated variants stay
//! unported: single-batch falcon_h1 builds neither (`create_ssm_mask`
//! returns `None` without left padding and `ArraysCache.lengths` is
//! `None`). The chunked and Metal-kernel forms differ from this
//! recurrence only in f32 summation order, which the fixture gate
//! absorbs; the recurrence is chunk invariant by construction.

/// Port of `compute_dt`: `clip(softplus(dt + dt_bias), limit)` per entry.
/// `softplus` is MLX's `log1p(exp(x))` in f32; `dt` is `[len, heads]`
/// row-major and `dt_bias` `[heads]`.
pub(crate) fn compute_dt(dt: &[f32], dt_bias: &[f32], limit: (f32, f32)) -> Vec<f32> {
    dt.iter()
        .zip(dt_bias.iter().cycle())
        .map(|(&raw, &bias)| {
            let x = raw + bias;
            // softplus(x) = ln(1 + exp(x)) = ln_1p(exp(x)).
            let sp = x.exp().ln_1p();
            sp.max(limit.0).min(limit.1)
        })
        .collect()
}

/// One selective-scan pass over `len` positions.
///
/// * `x` is `[len, heads, head_dim]` row-major (`heads * head_dim` wide).
/// * `a_log`, `d`, `dt_bias` are `[heads]`.
/// * `b`, `c` are `[len, groups, d_state]`.
/// * `dt` is `[len, heads]`.
/// * `state` is the optional `[heads, head_dim, d_state]` carry-in.
///
/// Returns `(y, next_state)`: `y` is `[len, heads * head_dim]` and
/// `next_state` is `[heads, head_dim, d_state]`, both matching the
/// reference buffers the fixtures pin step by step.
pub(crate) fn ssm_update(
    x: &[f32],
    a_log: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    dt: &[f32],
    dt_bias: &[f32],
    head_dim: usize,
    d_state: usize,
    state: Option<&[f32]>,
    time_step_limit: (f32, f32),
) -> (Vec<f32>, Vec<f32>) {
    let heads = a_log.len();
    assert_eq!(d.len(), heads);
    assert_eq!(dt_bias.len(), heads);
    assert_eq!(dt.len(), x.len() / head_dim);
    let len = x.len() / (heads * head_dim);
    let groups = b.len() / (len * d_state);
    assert!(groups > 0 && heads % groups == 0);
    let repeats = heads / groups;
    let mut state = match state {
        Some(state) => {
            assert_eq!(state.len(), heads * head_dim * d_state);
            state.to_vec()
        }
        None => vec![0.0; heads * head_dim * d_state],
    };
    let dt = compute_dt(dt, dt_bias, time_step_limit);
    let mut y = vec![0.0f32; len * heads * head_dim];
    for t in 0..len {
        for h in 0..heads {
            let group = h / repeats;
            // exp(A dt) with A = -exp(a_log); the product is always
            // negative, so this is a decay in (0, 1].
            let decay = (-a_log[h].exp() * dt[t * heads + h]).exp();
            let input_scale = dt[t * heads + h];
            let b_row = (t * groups + group) * d_state;
            let c_row = (t * groups + group) * d_state;
            for feature in 0..head_dim {
                let cell = (h * head_dim + feature) * d_state;
                let value = x[(t * heads + h) * head_dim + feature];
                let mut acc = 0.0f32;
                for s in 0..d_state {
                    let next = decay * state[cell + s] + value * input_scale * b[b_row + s];
                    state[cell + s] = next;
                    acc += next * c[c_row + s];
                }
                y[(t * heads + h) * head_dim + feature] = acc + value * d[h];
            }
        }
    }
    (y, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The recurrence run step by step with `state` chaining equals one
    /// multi-position call from zero (the reference split between the
    /// initial kernel and the update kernel).
    #[test]
    fn stepped_calls_equal_one_multi_position_call() {
        let heads = 3;
        let head_dim = 2;
        let groups = 1;
        let _ = groups;
        let d_state = 4;
        let len = 5;
        let mut x = Vec::new();
        for t in 0..len {
            for h in 0..heads {
                for f in 0..head_dim {
                    x.push(0.1 * (t * heads + h) as f32 + 0.05 * f as f32);
                }
            }
        }
        let a_log: Vec<f32> = (0..heads).map(|h| 0.2 * h as f32 - 0.3).collect();
        let b: Vec<f32> = (0..len * groups * d_state)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
            .collect();
        let c: Vec<f32> = (0..len * groups * d_state)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.15)
            .collect();
        let d: Vec<f32> = (0..heads).map(|h| 0.5 + 0.1 * h as f32).collect();
        let dt: Vec<f32> = (0..len * heads).map(|i| 0.05 * (i % 4) as f32).collect();
        let dt_bias: Vec<f32> = (0..heads).map(|h| -0.4 + 0.2 * h as f32).collect();
        let limit = (0.0, f32::INFINITY);

        let (whole, whole_state) = ssm_update(
            &x, &a_log, &b, &c, &d, &dt, &dt_bias, head_dim, d_state, None, limit,
        );

        // Step by step through one-position calls, carrying the state.
        let mut chained_state = None;
        let mut chained_rows = Vec::new();
        for t in 0..len {
            let row = &x[t * heads * head_dim..(t + 1) * heads * head_dim];
            let (y, next) = ssm_update(
                row,
                &a_log,
                &b[t * groups * d_state..(t + 1) * groups * d_state],
                &c[t * groups * d_state..(t + 1) * groups * d_state],
                &d,
                &dt[t * heads..(t + 1) * heads],
                &dt_bias,
                head_dim,
                d_state,
                chained_state.as_deref(),
                limit,
            );
            chained_state = Some(next);
            chained_rows.extend_from_slice(&y);
        }
        let max = chained_rows
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-6, "stepped vs whole diff {max}");
        assert_eq!(chained_state.as_deref().unwrap().len(), whole_state.len());
    }

    /// Zero `dt` input keeps the state and returns the `D * x` skip
    /// connection only (softplus of a very negative input underflows the
    /// exp toward 0, which is the identity dynamics the fixtures bound).
    #[test]
    fn vanishing_dt_is_identity_dynamics() {
        let heads = 2;
        let _ = heads;
        let head_dim = 1;
        let groups = 1;
        let _ = groups;
        let d_state = 3;
        let x = [1.0, -2.0];
        let a_log = [0.0, 0.0];
        let b = [1.0, 0.5, 0.25, 1.0, 0.5, 0.25];
        let c = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let d = [2.0, 3.0];
        // softplus(x + -40) is about exp(x - 40): far below f32 resolution.
        let dt = [-40.0, -40.0];
        let dt_bias = [0.0, 0.0];
        let initial = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let (y, state) = ssm_update(
            &x,
            &a_log,
            &b,
            &c,
            &d,
            &dt,
            &dt_bias,
            head_dim,
            d_state,
            Some(&initial),
            (0.0, f32::INFINITY),
        );
        for (got, want) in state.iter().zip(&initial) {
            assert!((got - want).abs() < 1e-6, "state {got} vs {want}");
        }
        // Zero dt keeps the state, so y is <state, C> + D * x:
        // head 0: (7 + 8 + 9) * 1 + 1 * 2 = 26; head 1: 33 - 6 = 27.
        assert!((y[0] - 26.0).abs() < 1e-5, "y0 {y:?}");
        assert!((y[1] - 27.0).abs() < 1e-5, "y1 {y:?}");
    }
}
