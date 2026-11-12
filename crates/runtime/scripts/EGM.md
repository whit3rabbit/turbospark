# Runtime Benchmark Scripts

## Overview
Two Python 3 drivers that benchmark Core ML pipelines against CPU or MLX arms. Each sample runs the matching release example binary in a fresh process under `/usr/bin/time -l` for peak footprint, rotates arm order per run, and records 1-minute load and power source so a run is only quotable on a quiet, AC-powered host.

## Interface
- `benchmark_coreml.py` [benchmark_coreml.py]: tasks `parakeet|embed|ocr`; flags `--coreml --cpu --wav --model --pages --arms --runs --output`; drives `target/release/examples/coreml_backend_probe`
- `benchmark_embeddinggemma3.py` [benchmark_embeddinggemma3.py]: EmbeddingGemma 300m on cpu, mlx and coreml_{ane,gpu,cpu,all}; flags `--mlx-dir --coreml-dir --arms --runs --output`; drives `target/release/examples/embeddinggemma3_probe`
- `__pycache__/` holds compiled bytecode, not source.

## Invariants
- The example binary must be built first (`cargo build --release -p turbospark-runtime --example ...`); the path is relative to the repo root [benchmark_coreml.py:BINARY] [benchmark_embeddinggemma3.py:BINARY]
- Cross-arm output agreement is reported (`digests_agree`), not enforced [benchmark_coreml.py]
- `quiet_host` is false when any sample saw load of 4 or more or battery power [benchmark_embeddinggemma3.py]
- All arms embed identical token ids tokenized once by the CPU port [benchmark_embeddinggemma3.py]

## Side Effects
- Spawns the probe binary and `/usr/bin/time -l`, and runs `pmset -g batt` [benchmark_embeddinggemma3.py:subprocess.run]
- Writes JSON results to `--output` or stdout, and temporary ids/texts/embedding JSON files [benchmark_embeddinggemma3.py]
- No network access.

## Verification
- Run: `python3 crates/runtime/scripts/benchmark_coreml.py --help`
- Run: `python3 crates/runtime/scripts/benchmark_embeddinggemma3.py --help`
