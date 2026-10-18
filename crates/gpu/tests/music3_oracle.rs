//! Bit-exactness oracle for the Music 3 affine linear kernels, as a test.
//!
//! `examples/music3_linear_bench.rs` hashes the output bits of every affine
//! bit width, group size, row count and dtype, and of the real 691-row DiT
//! shapes. Those hashes used to be diffed by hand before and after a kernel
//! change, which is easy to skip. This test runs the same code and compares it
//! with the checked-in golden files, so `cargo test` fails the moment an
//! "exact" kernel change alters one bit.
//!
//! If a change is meant to alter the arithmetic, regenerate the goldens and say
//! why in the commit:
//!
//!     cargo run --release -p turbospark-gpu --example music3_linear_bench -- hashes \
//!         > crates/gpu/tests/golden/music3_hashes.txt
//!     cargo run --release -p turbospark-gpu --example music3_linear_bench -- dit_hashes \
//!         > crates/gpu/tests/golden/music3_dit_hashes.txt
//!
//! Dense bf16/f16 kernels are not covered here: they are checked against MLX
//! itself by `examples/music3_mlx_linear_cmp.rs` (see docs/audio/minimax_music3.md).

#![cfg(target_os = "macos")]

#[path = "../examples/music3_linear_bench.rs"]
#[allow(dead_code)]
mod bench;

fn lines_of(text: &str) -> Vec<String> {
    text.lines().map(str::to_owned).collect()
}

fn check(mode: &str, golden: &str) {
    let mut got = Vec::new();
    bench::run(mode, &mut |line| got.push(line));
    let want = lines_of(golden);
    assert_eq!(
        got.len(),
        want.len(),
        "{mode}: {} lines, golden has {}",
        got.len(),
        want.len()
    );
    let differing: Vec<_> = got
        .iter()
        .zip(&want)
        .filter(|(a, b)| a != b)
        .take(5)
        .collect();
    assert!(
        differing.is_empty(),
        "{mode}: kernel output bits changed (first differences, got vs golden): {differing:#?}"
    );
}

#[test]
fn affine_linear_hashes_match_the_golden_file() {
    check("hashes", include_str!("golden/music3_hashes.txt"));
}

#[test]
fn dit_shape_hashes_match_the_golden_file() {
    check("dit_hashes", include_str!("golden/music3_dit_hashes.txt"));
}
