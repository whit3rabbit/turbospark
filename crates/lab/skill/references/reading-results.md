# Reading tslab results

## Verdicts

| Verdict | Means | Do |
|---|---|---|
| `WIN` | At least 3 clean pairs, median at or above the effect floor, bootstrap 95% interval above zero, and every pair agreeing (or sign-test p at most 0.05 from 6 pairs) | Report it with its conditions. Under run-experiment, run `confirm` once |
| `LOSS` | The same, the other way | Report it |
| `NOISE` | Enough clean pairs, no effect at the floor | Not a reason to rerun until it wins. A rerun is a new hypothesis only if something changed |
| `INCONCLUSIVE` | Too few clean pairs, usually a busy host | Fix the host (the run lists the causes), then rerun the same thing. Never revert on it |

The effect floor is the largest of 2%, the requested `--min-effect`, and the
null floor measured for that workload. Run `ab --null` once per workload
first, or the floor is only the 2% default.

## Sweeps

A `sweep` record holds one verdict per candidate, each judged on its own
clean pairs against the shared baseline under the same floor. `best` is the
candidate with the largest median among the WINs, and null when none won: a
NOISE candidate is never "best", however its median reads. A refused
candidate carries `refused` and no pairs. An inconclusive candidate makes the
command exit 4 like `ab` does; fix the host and rerun the sweep, which
re-judges every candidate.

## Dirty runs

A run is ledgered whatever happened. `eligible: false` and its `reasons`
say why it does not count:

- `foreign CPU over ...`: another process was busy (a build, an editor's
  background `cargo check`, Spotlight). Stop it and rerun.
- `not on AC power`: plug in, or pass `--allow-battery` and say so.
- `... perturbs timing; run is not quotable`: the arm set a profiler, a trace,
  the probe or a stand-in. Expected for those runs.
- `not engaged: ...`: the arm set a knob whose lane never ran, or a kill
  knob whose lane still ran. The arm measured something other than its
  name. Find out why before anything else.
- `worker status refused`: the install cannot run that arm. `record.refusal`
  says what.

## Quoting a number

Quote only clean, quotable runs, and always with: the ledger experiment and
record id, the model, the workload (prompt, decode tokens, chunk), the chip,
the power source, the page-cache class, and the number of clean runs or
pairs. Never quote tok/s from a trace, probe, stand-in or dispatch-profile
run. A kernel bench is a GPU-time microbenchmark, not throughput.

## Phase buckets and blocks

On routed MoE installs `profile` splits a token into buckets (layer GPU wait,
router readback, expert I/O, bind, retire, final wait). Dense flows fill only
the final wait and the remainder, and the table says which buckets read zero.
For a dense install, price blocks with `--standin` and dispatches with
`--dispatch-probe`. Both are rankings, not throughput.

## Kernel curves

Read `c_of_m` first: below 1, M rows in one pass beat M one-row passes.
`break_even` gives the M above which an arm beats the per-row GEMV. Check
`layout` against the install's own weights: an INT4 curve on a GGUF or MXFP4
install times a layout the install does not run.
