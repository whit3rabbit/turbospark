# tslab output contract (`tslab/1`)

`tslab schema` prints the authoritative JSON Schema. This page is the short
form.

## The `--json` envelope

With `--json`, stdout is exactly one line, one object. Tables and progress go
to stderr.

```json
{"schema": "tslab/1", "command": "ab", "ok": true, "exit_code": 0,
 "decision": "keep_pending_confirm",
 "records": [{"id": "1792930520663-50321-19", "kind": "verdict", "ledger": "/.../ab/ledger.jsonl"}],
 "result": {}, "warnings": [], "error": "only when ok is false"}
```

- `ok: false` means the command failed. `error` says why, and `exit_code` is 1.
- `ok: true` with a non-zero `exit_code` is a decision, not an error.
- `records` lists every ledger line the command appended, oldest first.
- `result` is the command's payload. For commands without one of their own
  it is `{"by_kind": {kind: [records]}}`. Run records are cut down to their
  headline fields, and the ledger has the rest by id.

Exit codes: 0 ok or keep, 1 error, 2 usage, 3 discard (also `hash-diff` when
an output changed), 4 inconclusive (the host was busy: do not revert, fix the
host and rerun).

## Ledger records

One JSON object per line in `<lab>/<experiment>/ledger.jsonl`, where `<lab>`
is `$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else
`~/.turbospark/lab`. Every record has `id`, `kind`, `ts_ms` and, from version
1 on, `schema`. Lines are append-only. A correction is a new line.

| Kind | Holds |
|---|---|
| `run` | One worker process: arm, env, host samples, `eligible`, `reasons`, `totals`, `engaged`, `spec_stats`, `audio` |
| `profile_summary` | Per-arm medians over clean runs |
| `null_calibration` | The noise floor (`floor`) for a workload and metric |
| `verdict` | A paired A/B: `verdict.{verdict, median, ci_low, ci_high, improvements}`, `memory` |
| `kernel_curve` | One kernel bench case: `result` is a `BenchResult` |
| `hash_diff`, `dispatch_probe`, `standin`, `trace`, `round_cost` | What their command measured |
| `experiment`, `confirm`, `gate`, `experiment_start` | run-experiment's records and decision |
| `workflow_step`, `workflow_run` | A workflow's steps and summary |

A run is `eligible: false` when the host guard rejected it, the worker
refused, an unquotable switch was set, or its engagement lines contradict its
arm. `reasons` says which.

## BenchResult (inside `kernel_curve.result`)

```json
{"bench": "int4_matmul", "case": "17408x5120",
 "subject": {"rows": 17408, "cols": 5120, "roles": ["mlp.gate_proj", "mlp.up_proj"], "count": 128},
 "x": {"name": "m", "unit": "rows", "scale": "log2"},
 "layout": "mlx_affine_int4_g64_bf16",
 "arms": [{"label": "scalar_gemm",
           "points": [{"x": 8, "ms": 0.41, "gb_s": 210.0, "c_of_m": 0.37, "out_fnv": "0c7f089a7e72e924"}],
           "fit": {"a": 0.21, "b": 0.025, "max_rel_residual": 0.03}},
          {"label": "steel_m32", "unsupported": "this shape does not fit the tile"}],
 "break_even": [{"arm": "scalar_gemm", "against": "gemv_per_row", "above_x": 1.4}],
 "notes": []}
```

`c_of_m` is the cost of M rows in one pass over M one-row passes. Below 1,
batching pays.

## Series (from `tslab export`)

```json
{"schema": "tslab/1", "id": "...", "kind": "kernel_ms", "experiment": "e",
 "source_kind": "kernel_curve", "source_schema": 1, "model": "q.gturbo",
 "family": "qwen35", "chip": "Apple M4 Max", "rev": "c4d6ee5e11",
 "eligible": true, "quotable": true, "group": "int4_matmul 17408x5120",
 "label": "scalar_gemm", "x": {"name": "m", "unit": "rows", "scale": "log2"},
 "y": {"name": "ms", "unit": "ms"}, "points": [{"x": 8, "y": 0.41}], "refs": {}}
```

Kinds: `kernel_ms`, `kernel_c_of_m`, `profile_totals`, `profile_buckets`,
`verdict` (points are pair improvements, and `refs` holds `median`, `ci_low`,
`ci_high`, `min_effect`, `null_floor`), `dispatch_probe`, `round_cost`,
`audio_stages`, `trace_shares`, `standin`. Series sharing a `group` belong on
one chart. A point with `dirty: true` came from a run the host guard
rejected.
