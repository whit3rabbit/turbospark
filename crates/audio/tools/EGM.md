# tools

## Overview
Python tools that regenerate Rust parity fixtures for the codec, TTS, STS, and LM families against the pinned mlx-audio reference, plus a MegaTTS3 checkpoint converter and Music 3 reference probes.

## Interface
- `gen_<model>_fixtures.py` for each codec and TTS family, for example `gen_snac_fixtures.py`, `gen_mimi_fixtures.py`, `gen_dac_fixtures.py`, `gen_megatts3_*_fixtures.py`, `gen_minimax_music3_fixtures.py`, `gen_music3_precision_fixtures.py`.
- `convert_megatts3.py` (PyTorch to safetensors) with its test `test_convert_megatts3.py` (`python3 -I crates/audio/tools/test_convert_megatts3.py`).
- Reference probes: `probe_minimax_music3_reference.py`, `time_minimax_music3_reference.py`, `gen_minimax_music3_bench_tree.py`, `gen_music3_mlx_linear_cases.py`, `dbg_chatterbox_traces.py`.

## Invariants
- Generators run from inside the mlx-audio checkout at the pinned commit using the project venv; they write into `crates/audio/testdata/<model>/` (for example `OUT` in [gen_snac_fixtures.py]).
- `gen_snac_fixtures.py --real` records goldens only from locally cached checkpoints (`local_files_only`, no network).

## Side Effects
- Overwrite fixtures under `crates/audio/testdata/`; read local model caches [gen_snac_fixtures.py:OUT].

## Verification
- Run: `python3 -I crates/audio/tools/test_convert_megatts3.py`
