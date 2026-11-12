# ops

## Overview
Holds the retained naive reference kernels used to prove that the optimized `ops` kernels in `src/ops.rs` are bitwise identical. Compiled only under `cfg(test)`.

## Interface
- `kernel_parity.rs` is declared as `#[cfg(test)] mod kernel_parity` in [../ops.rs]; it exports nothing publicly.
- Reference functions: `matmul_reference`, `linear_reference`, `conv1d_reference`, `conv_transpose1d_reference`, `sdpa_reference`, `conv2d_reference`, the rope (`rope_tables_reference`, `rope_neox_reference`, `rope_interleaved_reference`), `split_qkv_reference` and layernorm (`layernorm_rows_reference`, `layernorm_affine_free_reference`, `layernorm_channels_reference`) references; helper `assert_bits_eq` [kernel_parity.rs].

## Invariants
- Reference loops must stay verbatim: golden tensors for every model were recorded against their reduction order, and optimized kernels must match bit for bit [kernel_parity.rs:assert_bits_eq].
- Test inputs mix magnitudes and exact zeros so a changed summation order or zero-skip shows in the bits [kernel_parity.rs:matmul_is_bitwise_identical_to_reference].

## Side Effects
- None.

## Verification
- Run: `cargo test -p turbospark-audio ops::kernel_parity`
