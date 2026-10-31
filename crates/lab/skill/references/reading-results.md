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

## The compare view

`tslab serve` and `/compare?exp=a,b` put several experiments' series on one
chart per (kind, group): the same kernel shape, the same profile buckets,
the sweeps and the verdict forests, with labels prefixed by the experiment.
A shared axis makes a difference visible, not significant: the verdict
board under the charts holds the paired results, and two experiments' bars
on one chart were not paired with each other. Hollow marks stay what they
are, runs the host guard rejected.

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

## Check the output before the timings

A forward pass that decodes garbage can still run fast and bit-exact between
arms. On a MoE install it can run faster: degenerate routing hits a few
experts, so cache hit rates near 99% and batching seams that look like wins
are symptoms, not results (qwen4exp, 2026-10-30, before the RMSNorm fix).
Read a few decoded tokens through the chat template before trusting a
family's first sweep, and re-measure after any forward-pass fix.

## Memory pressure

Each run's `host` block carries `compressions_delta` and
`decompressions_delta`, the macOS compressor's page counts over the sampled
span. Swap alone misses pressure: a 96-slot qwen4exp cache pushed about 5
million pages through the compressor per run while swap barely moved, and
it lost 18% on decode with GPU busy time flat. When a slot, residency or
footprint arm loses, compare these deltas across arms before reading the
loss as compute. Other sessions' memory moves them too, so compare arms
within one interleaved command.

## Pricing a route against the model run

A route that swaps one kernel for another (a lane in `tslab knobs`) is
priced twice: the kernel part is the bench's per-projection time difference
times the projection count, and the observed part is the paired model run.
If the observed delta is larger than the priced part, report both and say the
difference is unattributed; a per-op-class profile of the two arms is what
attributes it. Do not quote the larger number alone.

A kernel bench's `exactness` line (bit-equal, row invariant) is read off its
synthetic weights. It can say bit-equal for an arm that changes bits on the
real install, so take the lane's class from the install's KL or digest gate.

A quiet host before a run does not cover builds that start during it. Expect
to repeat pairs, count only clean ones, and say how many of the runs were
clean when you quote the rest as a reading.

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
