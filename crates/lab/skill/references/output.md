# tslab output contract (`tslab/1`)

`tslab schema` prints the authoritative JSON Schema. This page is the short
form.

## Contents

- [The `--json` envelope](#the---json-envelope)
- [Ledger records](#ledger-records)
- [BenchResult](#benchresult-inside-kernel_curveresult)
- [Series](#series-from-tslab-export)
- [The dashboard API](#the-dashboard-api-tslab-serve)

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

Exit codes: 0 ok or keep, 1 error (an unknown command or bad flag too), 2
bare `tslab` with no arguments, 3 discard, 4 inconclusive (the
host was busy: do not revert, fix the host and rerun). Some commands use the
codes for their own outcome, with `ok: true`:

- 1: `sanity`, `quant-ab` and `release-gate` when a check failed. Read `ok`
  to tell a failed check from an error.
- 3: `hash-diff`, `anchor` and `image-stage` when an output changed;
  `serve-bench` and `soak` when a request or the soak failed.

## Ledger records

One JSON object per line in `<lab>/<experiment>/ledger.jsonl`, where `<lab>`
is `$TURBOSPARK_LAB_DIR`, else `$TURBOSPARK_HOME/lab`, else
`~/.turbospark/lab`. Every record has `id`, `kind`, `ts_ms` and, from version
1 on, `schema`. Lines are append-only. A correction is a new line.

| Kind | Holds |
|---|---|
| `run` | One worker process: arm, env, host samples (with `low_power` and `memory_pressure`), `eligible`, `reasons`, `totals` (with `prefill_j`, `decode_j`, `run_j`, joules per token and `energy_model` when the host meters), `engaged`, `spec_stats`, `ane_stats`, `controlled_contention`, `audio`, `open_pressure` (compressor and swap movement while the install opened), `synthetic_accept` |
| `profile_summary` | Per-arm medians over clean runs, each phase with a `moe_bound` class and shares, plus `prefill_j`/`decode_j` when metered |
| `null_calibration` | The noise floor (`floor`) for a workload and metric |
| `verdict` | A paired A/B: `verdict.{verdict, median, ci_low, ci_high, improvements}`, `memory`, `null_floor`, `noninferiority`, `baseline_route.{default_missing, default_extra}`, `methodology`, `buckets`; from a sweep also `sweep_id` and `family` |
| `sweep` | `tslab sweep`: `sweep_id`, `family`, `baseline`, `metric`, `floor`, `baseline_route`, `false_wins_upper_bound`, `candidates[].{arm, verdict, memory, clean_pairs, dirty_pairs, refused, rejected, secondary_guard, buckets, expert_hit_rate, methodology, verdict_record}`, `best.{arm, median}` (the largest median among WINs not rejected, else null), `stopped` |
| `kernel_curve` | One kernel bench case: `result` is a `BenchResult` |
| `vision_encode` | One `vision_encode` bench run: per page the stage medians, patches/s, `stream_share` and peak footprint |
| `hash_diff`, `dispatch_probe`, `standin`, `trace`, `round_cost`, `ane_plan` | What their command measured |
| `chain_capture` | `capture.{case, depth, prompt_tokens, match_len, stream_digest}` and, from this build on, `capture.prompt_ids` and `capture.stream` (token ids; absent on older captures) |
| `depth_replay`, `hybrid_replay` | Controller settings, or the lookup-plus-head hybrid against the head alone, priced over captures: `results[].{label, env, case}` with per-case `head` and `hybrid` walks and `delta_committed_per_round`, `delta_tok_s` |
| `tree_replay` | Speculative acceptance: tree against chain at equal verify rows |
| `expert_census` | `tslab expert-census`: `summary.phases.{prefill,decode}.{reuse, concentration, union, slots_fit, sub_batches, largest_group}` over one capture run (`capture` is the file) |
| `projection` | `tslab project`: `projection.{ratio, improvement_throughput, improvement_time, coverage, floor, clears_floor, cases, skipped, notes}` and the `inputs` it read |
| `calibration` | `tslab calibrate`: `calibration.{predicted, measured, error, miss, sign_flip, reason}` and the projection and verdict it joined |
| `experiment`, `confirm`, `gate`, `experiment_start` | run-experiment's records and decision. `quality.decode` holds the forced-trace summary (`overall` and per prompt: KL mean, median, p99, flips, flips by reference margin, PPL per arm, paired NLL delta and SE). `gate` also holds `noise_arms` and `floor.{family, mean_kl, bits_identical, record}` |
| `sanity`, `noise`, `floor`, `quant_ab`, `launch_count`, `idle_gap`, `spec_rounds`, `image_stage`, `anchor`, `release_gate` | What the command of that name measured |
| `serve_bench`, `soak` | Server rows. They carry `ineligible` (not `reasons`), `target`, `protocol`, `host`, `provenance` and a `methodology` string, so server rows never merge with in-process ones. `serve_bench`: `counts.{requests, failed, capped, accounting_mismatch}`, `server_memory`, `server_metrics`, `warmup`, then `by_size`, `requests`, `burst` (closed loop) or `rounds`, `goodput_req_s`, `aggregate_tok_s` (open loop). `soak`: `passed`, `failure`, `completed`, `failed`, `disconnected`, `per_bucket`, `long_prefill`, `drain_probe`, `process` |
| `workflow_step`, `workflow_run` | A workflow's steps and summary |

A run is `eligible: false` when the host guard rejected it, the worker
refused, an unquotable knob was set, or its engagement lines contradict its
arm. `reasons` says which. Each run also carries `methodology` (`id` is kind,
version and a hash of the workload with its prompt files' bytes: records with
different ids are not comparable; `kind` is `timing`, `quality`,
`forced_decode`, `idle_gap` or `spec_probe`, at version 3), `file_age_s` (the
install's newest file), and `ballast_gib` and `cold` when the arm set them.
Provenance carries `toolchain_id` (the Xcode and Metal versions) beside the
revision and binary hash.

## BenchResult (inside `kernel_curve.result`)

```json
{"bench": "int4_matmul", "case": "17408x5120",
 "subject": {"rows": 17408, "cols": 5120, "roles": ["mlp.gate_proj", "mlp.up_proj"], "count": 128},
 "x": {"name": "m", "unit": "rows", "scale": "log2"},
 "layout": "mlx_affine_int4_g64_bf16",
 "arms": [{"label": "scalar_gemm",
           "points": [{"x": 8, "ms": 0.41, "gb_s": 210.0, "c_of_m": 0.37, "out_fnv": "0c7f089a7e72e924", "row_fnv": "5e1a0c33f2a7b910"}],
           "fit": {"a": 0.21, "b": 0.025, "max_rel_residual": 0.03}},
          {"label": "steel_m32", "unsupported": "this shape does not fit the tile"}],
 "break_even": [{"arm": "scalar_gemm", "against": "gemv_per_row", "above_x": 1.4}],
 "notes": []}
```

`c_of_m` is the cost of M rows in one pass over M one-row passes. Below 1,
batching pays. `out_fnv` and `row_fnv` (the first output row alone) appear
under `--hash`. The `kernel_curve` record also carries `exactness_hints`, one
per arm: `{arm, against, bit_exact_vs_first, row_invariant}`, each `true`,
`false` or null when too little was hashed to say.

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
`ci_high`, `min_effect`, `null_floor`), `sweep` (one point per candidate's
median on a category axis, an inconclusive candidate at zero and `dirty`),
`dispatch_probe`, `round_cost`, `audio_stages`, `trace_shares`, `standin`,
`calibration` (predicted against
measured improvement, one point per calibration). Series sharing a `group` belong on
one chart. A point with `dirty: true` came from a run the host guard
rejected. Each series carries its `methodology`. When one group mixes
methodology ids, it is split and each part renamed `<group> [<id>]` (or
`[unrecorded]`): points measured differently never share a chart.

## The dashboard API (`tslab serve`)

Every route is a GET and returns `{"schema": "tslab/1", ...}`:

| Route | Returns |
|---|---|
| `/api/experiments` | `experiments[]` (name, last activity, counts by kind, models, families, chips, revs, clean and dirty runs) and `totals` by kind |
| `/api/experiment/{name}` | the summary, `notes`, and that experiment's `verdicts`, `sweeps`, `profiles` rows |
| `/api/ledger/{name}` | the raw ledger records |
| `/api/series?exp=a,b[&kind=K]` | the experiments' series, the shape `tslab export --format json` writes |
| `/api/verdicts[?family=&model=&metric=&exp=]` | the verdict board rows |
| `/api/sweeps[?family=&model=]` | the sweep rows with `best` |
| `/api/family/{family}` | the profile plus the lab's rows for that family |
| `/api/record/{exp}/{id}` | one ledger record |

A bad experiment name or record id is a 400, a missing one a 404, both as
`{"ok": false, "error": ...}`.

