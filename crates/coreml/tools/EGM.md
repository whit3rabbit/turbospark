# coreml tools

## Overview
Python scripts (run with an isolated interpreter, `coremltools==9.0`, `numpy==2.2.6`) that generate Core ML programs and fixtures for the opt-in ANE+Steel prefill split and the `tslab` ANE benchmarks.

## Interface
- [gen_ane_prefill_programs.py:main]: writes F16-input-plane matmul programs per projection class and token count (`--classes`, `--tokens`, `--rows`).
- [bake_ane_prefill_programs.py:main]: bakes real INT4 weights from a `.gturbo` install's `model_weights.bin` into uint4 constants (`baked_program`, `bake_classes`). Modes: `--tensors` (sweep.json) versus `--layers`/`--classes` (manifest.json); `--lut` (4-bit indices into 16 fp16 values), `--shared-f16`, `--with-f16`, `--share`, `--rows`.
- [bake_ane_vision_programs.py:main]: bakes a vision tower block's qkv/proj/fc1/fc2 projections with seeded random F16 weights and no biases, as timing and placement probes only (`--config`, `--tokens`, `--roles`).
- [gen_ane_sweep_programs.py:main]: writes programs and `sweep.json` for `tslab kernel --bench ane_matmul` (`--shapes`, `--tokens`, `--shares`, `--experts` with `--expert-dims`, `--const-int4` with `--const-block`; const-int4 weights are random).
- [gen_coreml_fixtures.py:main]: writes tiny fixture packages under `crates/coreml/testdata/` (f32, f16, int32 argmax, multifunction, stateful accumulator).

## Invariants
- Baked INT4 uses the resident affine layout: row-major codes low nibble first, then BF16 scales and biases per 64 columns, `w = q * scale + bias` [bake_ane_prefill_programs.py:Install.rows] (unpacking), consumed by [bake_ane_prefill_programs.py:baked_program].
- A tensor is refused unless the fp16 scale and offset reproduce the BF16 values within 1e-5 absolute or 0.05 of the group scale; scales under 6.1e-5 are subnormal and can fail; a zero scale is refused, and `--lut` refuses a table more than 0.05 level off [bake_ane_prefill_programs.py:fp16_mapping].
- Sweep shape widths must be multiples of 32 halfs, and rows must be multiples of 32; only the ANE row count is rounded down to 32 [gen_ane_sweep_programs.py:main].
- Prefill programs slice K at 2048 and sum partial products in F16 [gen_ane_prefill_programs.py:build].
- Fixture expected outputs are closed forms computed by Rust tests; the manifest records only provenance and digests [gen_coreml_fixtures.py:digest_tree].

## Side Effects
- The program generators write into an output directory argument [gen_ane_prefill_programs.py:main]. [gen_coreml_fixtures.py:main] takes none: it writes to the fixed `crates/coreml/testdata/`, deletes any existing package of the same name first, and rewrites `manifest.json`.
- The bake script reads the install's weight file from disk [bake_ane_prefill_programs.py:main].

## Verification
- Run: `uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 crates/coreml/tools/gen_ane_prefill_programs.py --help` (the script imports `coremltools` before parsing args, so bare `python3` fails)
