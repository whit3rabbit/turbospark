# tslab tools

## Overview
Standalone helper scripts for tslab. `trace_tables.py` summarizes a Metal System Trace captured by `tslab trace` into one JSON object (evaluator item L5).

## Interface
- `python3 -I trace_tables.py TRACE.trace PID` [tools/trace_tables.py:main]: prints JSON with `encoders` (count, window, GPU busy percent, idle gaps, mean encoder time) [tools/trace_tables.py:encoders] and `shaders` (per-kernel sample share) [tools/trace_tables.py:shaders].
- `export` runs `xcrun xctrace export` for a table schema [tools/trace_tables.py:export]; `rows`, `pid_of`, `num`, `label`, `percentile` are internal helpers.

## Invariants
- Standard library only; xctrace `id`/`ref` value compression is resolved, and nothing from the trace is executed [tools/trace_tables.py:rows].
- Shader samples cover about a tenth of GPU time (an xctrace observation from the module docstring, not computed), so treat shares as comparable between traces, not absolute microseconds [tools/trace_tables.py:shaders].
- `encoders` counts only intervals on the `Compute` channel for the given pid [tools/trace_tables.py:encoders].

## Side Effects
- Spawns `xcrun xctrace export` as a subprocess [tools/trace_tables.py:export]. Writes nothing to disk itself.

## Verification
- Run: `cargo test -p turbospark-lab` (no Rust test covers this script; check with `python3 -I trace_tables.py <trace> <pid>`).
