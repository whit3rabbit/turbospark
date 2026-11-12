# gpu examples

## Overview
Developer harnesses for the Music 3 packed linear kernels. Not public API.

## Interface
- [music3_linear_bench.rs:main]: `hashes` prints an FNV-1a hash of output bits over a grid of bit widths, group sizes, row counts and dtypes; `timing` prints mean wall time for real checkpoint shapes. Run: `cargo run --release -p turbospark-gpu --example music3_linear_bench -- hashes|timing`.
- [music3_mlx_linear_cmp.rs:main]: bit-for-bit diff of the Music 3 Metal linear kernels against MLX 0.32.3 cases written by `crates/audio/tools/gen_music3_mlx_linear_cases.py`. Optional `--require-exact` after the case directory fails on any differing value.

## Invariants
- Any hash difference before and after a kernel change means the arithmetic changed [music3_linear_bench.rs:main] (module doc).
- No model code, sampling or tokenizer is involved, so a mismatch isolates the kernel [music3_mlx_linear_cmp.rs:main] (module doc).

## Side Effects
- Requires a Metal device; the comparison reads a case directory from disk [music3_mlx_linear_cmp.rs:main].

## Verification
- Run: `cargo build -p turbospark-gpu --examples`
