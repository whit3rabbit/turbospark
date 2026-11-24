use super::*;

/// Deterministic test words (xorshift32).
fn words(n: usize, mut seed: u32) -> Vec<u32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        })
        .collect()
}

/// The k-bit symbol at stream position `i`, read bit by bit (no window).
fn symbol(w: &[u32], i: usize, k: usize) -> u32 {
    (0..k).fold(0, |acc, b| {
        let bit = i * k + b;
        acc | ((w[bit / 32] >> (bit % 32)) & 1) << b
    })
}

/// The generic compiler's view: columns in order, `state = (state << k | sym) & 0xFFF`
/// from 0, with column `j`'s symbol at stream position `n - 1 - j`.
fn shift_register_states(w: &[u32], n: usize, k: usize) -> Vec<u32> {
    let mut state = 0u32;
    (0..n)
        .map(|j| {
            state = ((state << k) | symbol(w, n - 1 - j, k)) & 0xFFF;
            state
        })
        .collect()
}

#[test]
fn window_matches_the_shift_register_for_every_k() {
    let n = 96;
    for k in 1..=4usize {
        let w = words(n * k / 32, 0x1234_5678 + k as u32);
        let reg = shift_register_states(&w, n, k);
        for (j, &want) in reg.iter().enumerate() {
            assert_eq!(window(&w, (n - 1 - j) * k), want, "k {k} column {j}");
        }
    }
}

#[test]
fn hash_index_constants() {
    assert_eq!(table_index(LamqCode::Ht2, 1), 316);
    assert_eq!(table_index(LamqCode::Ht3, 1), 632);
    assert_eq!(table_index(LamqCode::Ht2, 4095), 434);
    assert_eq!(table_index(LamqCode::Ht3, 4095), 869);
    assert_eq!(table_index(LamqCode::Ht2, 0x5A5), 30);
    assert_eq!(table_index(LamqCode::Rtr3, 0x5A5), 0x5A5);
}

/// A table whose entry `e`, element `t` is `e * 10 + t`, so a decoded value
/// names the index and the element that produced it.
fn tagged_table(code: LamqCode) -> Vec<f32> {
    let g = code.g();
    (0..code.table_rows() * g)
        .map(|i| ((i / g) * 10 + i % g) as f32)
        .collect()
}

#[test]
fn trellis_rows_take_the_state_of_the_last_column_of_each_group() {
    let n = 96;
    for code in [
        LamqCode::Ht2,
        LamqCode::Ht3,
        LamqCode::Rtr1,
        LamqCode::Rtr2,
        LamqCode::Rtr3,
        LamqCode::Rtr4,
    ] {
        let k = code.k() as usize;
        let g = code.g();
        let w = words(n * k / 32, 0xBEEF + k as u32);
        let reg = shift_register_states(&w, n, k);
        let row = decode_trellis_row(code, &w, n, &tagged_table(code));
        for j in (0..n).step_by(g) {
            let e = table_index(code, reg[j + g - 1]);
            for t in 0..g {
                assert_eq!(
                    row[j + t],
                    (e * 10 + t) as f32,
                    "{} column {}",
                    code.name(),
                    j + t
                );
            }
        }
    }
}

#[test]
fn lm6_packs_four_columns_in_three_bytes_low_first() {
    // codes 1, 2, 3, 4: v = 1 | 2 << 6 | 3 << 12 | 4 << 18 = 0x103081.
    assert_eq!(
        level_indices(LamqCode::Lm6, &[0x81, 0x30, 0x10], 4),
        vec![1, 2, 3, 4]
    );
    assert_eq!(level_indices(LamqCode::Lm8, &[7, 255], 2), vec![7, 255]);
}

#[test]
fn row_bytes_match_the_release_shapes() {
    // From the 5.4 GB release: rtr4 embed bucket (3072, 288) u32, ht2 q_proj
    // 3072 x 576 bytes, rtr1 72 words at 2304, lm6 1728 bytes per row.
    assert_eq!(LamqCode::Rtr4.row_bytes(2304), Some(288 * 4));
    assert_eq!(LamqCode::Ht2.row_bytes(2304), Some(576));
    assert_eq!(LamqCode::Rtr1.row_bytes(2304), Some(72 * 4));
    assert_eq!(LamqCode::Ht3.row_bytes(1152), Some(432));
    assert_eq!(LamqCode::Lm6.row_bytes(2304), Some(1728));
    assert_eq!(LamqCode::Lm8.row_bytes(2304), Some(2304));
    // Widths the packing cannot hold.
    assert_eq!(LamqCode::Rtr1.row_bytes(48), None);
    assert_eq!(LamqCode::Ht2.row_bytes(18), None);
    assert_eq!(LamqCode::Lm6.row_bytes(6), None);
}

#[test]
fn ht1_and_level_codes_have_no_expert_fid() {
    assert_eq!(LamqCode::from_fid(0), None);
    assert_eq!(LamqCode::from_name("ht1"), None);
    assert_eq!(LamqCode::from_fid(1), Some(LamqCode::Ht2));
    assert_eq!(LamqCode::from_fid(3), Some(LamqCode::Rtr1));
    assert_eq!(LamqCode::from_fid(6), Some(LamqCode::Rtr4));
    assert_eq!(LamqCode::from_fid(7), None);
}

fn assert_hadamard(h: &[f64], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let d: f64 = (0..n).map(|t| h[i * n + t] * h[j * n + t]).sum();
            let want = if i == j { n as f64 } else { 0.0 };
            assert_eq!(d, want, "row {i} . row {j} of order {n}");
        }
    }
}

#[test]
fn paley_blocks_are_hadamard_with_the_upstream_border() {
    for q in [11, 71] {
        let p = q + 1;
        let h = paley(q);
        assert_hadamard(&h, p);
        assert!(h[..p].iter().all(|&v| v == 1.0), "row 0 is all ones");
        assert!(
            (1..p).all(|i| h[i * p] == -1.0),
            "column 0 below the corner is -1"
        );
        assert!((0..p).all(|d| h[d * p + d] == 1.0), "unit diagonal");
    }
    // Not symmetric: the output rotation's transposes matter.
    let h = paley(11);
    assert!((0..12).any(|i| (0..12).any(|j| h[i * 12 + j] != h[j * 12 + i])));
}

#[test]
fn factorization_of_every_release_width() {
    for (n, na, nb) in [
        (2304, 32, 72),
        (1152, 16, 72),
        (3072, 256, 12),
        (384, 32, 12),
        (1024, 32, 32),
    ] {
        let r = Rht::new(n).unwrap();
        assert_eq!((r.na, r.nb), (na, nb), "width {n}");
    }
    // 40 = 2 x Paley(19) is admissible; 10 (5 x 2, 10 x 1) has no split and
    // is refused rather than guessed.
    assert_eq!(Rht::new(40).map(|r| (r.na, r.nb)), Some((2, 20)));
    assert!(Rht::new(10).is_none());
}

fn signs(n: usize, seed: u32) -> Vec<i8> {
    words(n, seed)
        .into_iter()
        .map(|w| if w & 1 == 0 { 1 } else { -1 })
        .collect()
}

/// `kron(Sylvester(32), Paley(11))` built here, not by `Rht`.
fn kron_384() -> Vec<f64> {
    let (a, b) = (sylvester(32), paley(11));
    let n = 384;
    let mut h = vec![0.0; n * n];
    for i in 0..32 {
        for p in 0..32 {
            for j in 0..12 {
                for q in 0..12 {
                    h[(i * 12 + j) * n + p * 12 + q] = a[i * 32 + p] * b[j * 12 + q];
                }
            }
        }
    }
    h
}

#[test]
fn rotations_are_the_dense_kron_with_signs_first_in_and_last_out() {
    let n = 384;
    let h = kron_384();
    assert_hadamard(&h, n);
    assert_eq!(h, Rht::new(n).unwrap().dense());
    let s = signs(n, 99);
    let x: Vec<f32> = words(n, 7)
        .into_iter()
        .map(|w| (w % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    let r = Rht::new(n).unwrap();
    let inv = 1.0 / (n as f64).sqrt();
    let got_in = r.apply_in(&x, &s);
    let got_out = r.apply_out(&x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), &s);
    for i in 0..n {
        let want_in: f64 = (0..n)
            .map(|j| h[i * n + j] * f64::from(s[j]) * f64::from(x[j]))
            .sum::<f64>()
            * inv;
        let want_out: f64 =
            (0..n).map(|j| h[j * n + i] * f64::from(x[j])).sum::<f64>() * inv * f64::from(s[i]);
        assert!((got_in[i] - want_in).abs() < 1e-12, "in {i}");
        assert!((got_out[i] - want_out).abs() < 1e-12, "out {i}");
    }
}

#[test]
fn round_trip_at_every_release_width() {
    for n in [2304, 1152, 3072, 1024, 384] {
        let r = Rht::new(n).unwrap();
        let s = signs(n, n as u32);
        let x: Vec<f32> = words(n, 3)
            .into_iter()
            .map(|w| (w % 1001) as f32 / 500.0 - 1.0)
            .collect();
        let back = r.apply_out(&r.apply_in(&x, &s), &s);
        let err = x
            .iter()
            .zip(&back)
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-12, "width {n}: max err {err}");
    }
}

#[test]
fn validate_refuses_bad_shapes_and_signs() {
    let n = 384;
    let codes = vec![0u8; 2 * LamqCode::Rtr2.row_bytes(n).unwrap()];
    let scales = [0x3C00u16; 2];
    let s = signs(n, 1);
    let table = vec![0.0f32; 4096];
    let ok = LamqLinear {
        code: LamqCode::Rtr2,
        m: 2,
        n,
        codes: &codes,
        scales: &scales,
        signs_in: &s,
        signs_out: None,
        table: &table,
    };
    assert!(ok.validate().is_ok());
    assert!(LamqLinear { m: 3, ..ok }.validate().is_err());
    assert!(LamqLinear {
        table: &table[..4095],
        ..ok
    }
    .validate()
    .is_err());
    assert!(
        LamqLinear {
            signs_out: Some(&s[..2]),
            ..ok
        }
        .validate()
        .is_ok(),
        "height 2, two signs"
    );
    assert!(
        LamqLinear {
            signs_out: Some(&s[..3]),
            ..ok
        }
        .validate()
        .is_err(),
        "three signs for two rows"
    );
    let mut zero = s.clone();
    zero[5] = 0;
    assert!(LamqLinear {
        signs_in: &zero,
        ..ok
    }
    .validate()
    .is_err());
    // rtr2 at 2304 fits; ht2 at 2306 does not.
    assert!(LamqLinear {
        code: LamqCode::Ht2,
        n: 386,
        ..ok
    }
    .validate()
    .is_err());
}

#[test]
fn forward_is_scale_times_decoded_row_dot_rotated_input() {
    let (m, n) = (4, 384);
    let code = LamqCode::Rtr3;
    let w = words(m * n * 3 / 32, 42);
    let codes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scales = [0x3C00u16, 0x4000, 0xBC00, 0x3800]; // 1, 2, -1, 0.5
    let table: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) / 1024.0).collect();
    let s = signs(n, 5);
    let lin = LamqLinear {
        code,
        m,
        n,
        codes: &codes,
        scales: &scales,
        signs_in: &s,
        signs_out: None,
        table: &table,
    };
    lin.validate().unwrap();
    let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
    let xr = Rht::new(n).unwrap().apply_in(&x, &s);
    let y = lin.forward(&x);
    let words_per_row = n * 3 / 32;
    for (r, scale) in [1.0, 2.0, -1.0, 0.5].into_iter().enumerate() {
        let row = decode_trellis_row(
            code,
            &w[r * words_per_row..(r + 1) * words_per_row],
            n,
            &table,
        );
        let want: f64 = row
            .iter()
            .zip(&xr)
            .map(|(&a, &b)| f64::from(a) * b)
            .sum::<f64>()
            * scale;
        assert!((y[r] - want).abs() < 1e-9, "row {r}");
    }
}

#[test]
fn code_ids_round_trip_and_match_upstream_fids() {
    for c in [
        LamqCode::Ht2,
        LamqCode::Ht3,
        LamqCode::Rtr1,
        LamqCode::Rtr2,
        LamqCode::Rtr3,
        LamqCode::Rtr4,
        LamqCode::Lm6,
        LamqCode::Lm8,
    ] {
        assert_eq!(LamqCode::from_id(c.id()), Some(c));
        if c.is_trellis() {
            assert_eq!(
                LamqCode::from_fid(i32::from(c.id())),
                Some(c),
                "trellis ids are the upstream fids"
            );
        }
    }
    assert_eq!(LamqCode::from_id(0), None);
    assert_eq!(LamqCode::from_id(7), None);
}

#[test]
fn matrix_aux_round_trips_with_and_without_out_rot_and_levels() {
    let s_in = signs(384, 1);
    let s_out = signs(12, 2);
    let levels: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 16.0).collect();
    let a = encode_matrix_aux(LamqCode::Lm6, 384, 12, &s_in, Some(&s_out), Some(&levels));
    // 16 + 384 + 12 = 412, already a multiple of 4, then 64 levels.
    assert_eq!(a.len(), 412 + 256);
    let p = parse_matrix_aux(&a).unwrap();
    let h = p.header;
    assert_eq!(
        (h.code, h.out_rot, h.n, h.m),
        (Some(LamqCode::Lm6), true, 384, 12)
    );
    assert_eq!(
        (p.signs_in, p.signs_out, p.levels),
        (s_in.clone(), Some(s_out), Some(levels))
    );

    let a = encode_matrix_aux(LamqCode::Ht2, 384, 3, &s_in, None, None);
    assert_eq!(a.len(), 400);
    let p = parse_matrix_aux(&a).unwrap();
    assert_eq!(
        (p.header.out_rot, p.signs_out, p.levels),
        (false, None, None)
    );
}

#[test]
fn aux_header_refusals() {
    let s_in = signs(384, 1);
    let good = encode_matrix_aux(LamqCode::Rtr4, 384, 2, &s_in, None, None);
    let bad = |f: &dyn Fn(&mut Vec<u8>), why: &str| {
        let mut b = good.clone();
        f(&mut b);
        assert!(parse_matrix_aux(&b).is_err(), "{why}");
    };
    bad(&|b| b[0] = b'X', "magic");
    bad(&|b| b[2] = 2, "version");
    bad(&|b| b[3] = 9, "kind");
    bad(&|b| b[4] = 7, "code id");
    bad(&|b| b[4] = 0, "matrix without a code");
    bad(&|b| b[5] = 2, "unknown flag");
    bad(&|b| b[6] = 1, "reserved byte");
    bad(&|b| b[20] = 0, "zero sign");
    bad(&|b| b.push(0), "trailing byte");
    bad(
        &|b| {
            b.truncate(b.len() - 4);
        },
        "short aux",
    );
    let order = LamqAuxHeader {
        kind: LamqKind::Order,
        code: None,
        out_rot: false,
        n: 5,
        m: 0,
    };
    assert_eq!(LamqAuxHeader::parse(&order.encode()).unwrap(), order);
    assert!(
        parse_matrix_aux(&order.encode()).is_err(),
        "an order entry is not a matrix"
    );
    let mut coded_order = order.encode();
    coded_order[4] = 1;
    assert!(
        LamqAuxHeader::parse(&coded_order).is_err(),
        "a code on an order entry"
    );
}
