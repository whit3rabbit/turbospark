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
//!     cargo run --release -p turbospark-gpu --example music3_linear_bench -- conv_hashes \
//!         > crates/gpu/tests/golden/music3_conv_hashes.txt
//!
//! The convolution golden pins the tiled conv's bits (K order, split-K, bias) so a
//! restructuring of its loaders is checked without needing MLX. Dense bf16/f16
//! linear kernels are not covered here: they are checked against MLX
//! itself by `examples/music3_mlx_linear_cmp.rs` (see docs/audio/minimax_music3.md).

#![cfg(target_os = "macos")]

#[path = "../examples/music3_linear_bench.rs"]
#[allow(dead_code)]
mod bench;

/// The GPU the affine goldens were generated on.
///
/// The kernels reproduce MLX's choices, and several of those depend on the
/// device (`mlx_splitk_partitions`, the quantized split-K limit by chip name).
/// The first CI run of this oracle, on a different runner GPU, matched the
/// DiT-shape and convolution goldens but not the affine grid, so the grid's
/// bits are a property of the device as well as of the kernels. It is compared
/// only on the device that produced it. Set `TURBOSPARK_MUSIC3_ORACLE_ANY_DEVICE=1`
/// to compare anywhere, for example to regenerate and inspect on a new chip.
const AFFINE_GOLDEN_DEVICE: &str = "Apple M4 Max";

fn affine_golden_applies() -> bool {
    if std::env::var_os("TURBOSPARK_MUSIC3_ORACLE_ANY_DEVICE").is_some() {
        return true;
    }
    let name = std::env::var("TURBOSPARK_MUSIC3_ORACLE_DEVICE")
        .unwrap_or_else(|_| turbospark_gpu::Music3Device::new().unwrap().device_name());
    if name == AFFINE_GOLDEN_DEVICE {
        return true;
    }
    eprintln!("skipping the affine grid: its golden is for {AFFINE_GOLDEN_DEVICE}, this is {name}");
    false
}

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

// The goldens were generated on an M4 Max. The GitHub macOS runner's GPU
// produces different bits for the bits=2 split-K rows, so this arm is red there
// while the DiT and conv arms match. Run it on the golden hardware with
// `cargo test -p turbospark-gpu --test music3_oracle -- --ignored`.
#[test]
#[ignore = "goldens are M4 Max bits; the CI runner's GPU differs on bits=2 split-K rows"]
fn affine_linear_hashes_match_the_golden_file() {
    if !affine_golden_applies() {
        return;
    }
    check("hashes", include_str!("golden/music3_hashes.txt"));
}

#[test]
fn dit_shape_hashes_match_the_golden_file() {
    check("dit_hashes", include_str!("golden/music3_dit_hashes.txt"));
}

#[test]
fn convolution_hashes_match_the_golden_file() {
    check("conv_hashes", include_str!("golden/music3_conv_hashes.txt"));
}
