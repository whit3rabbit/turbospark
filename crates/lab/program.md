# program.md: the kernel autoresearch loop

Instructions for a Claude Code session that iterates on Metal and ANE
kernel work against `tslab`, the fixed evaluator in this crate. You propose
and implement changes. `tslab` measures and decides. You never decide a
result yourself, and you never edit the evaluator.

Read before the first iteration: `docs/SPEEDUP_PLAYBOOK.md` (how wins were
found and qualified, the kernel checklist, traps), `docs/KAPITSA_KERNELS.md`
("Where we were already ahead" and the decision table: do not re-propose
what it declines), `docs/MLX_SERVE_KERNELS.md` (its "Negative results"
list, and the evaluator additions L1 to L7 that this crate does not have
yet), `docs/BATCHED_PREFILL.md` for prefill drift bounds, and the AGENTS.md
files of every crate you touch.

## Setup (once per loop)

1. Work in your own worktree on a `lab/<tag>` branch cut from a clean
   `main`. One loop at a time on this machine: a release worktree build is
   about 7 GB, and two measuring sessions ruin each other's numbers.

   ```sh
   git worktree add .claude/worktrees/lab-<tag> -b lab/<tag> main
   ```

2. Install the trusted evaluator from that clean tree BEFORE any edit.
   `run-experiment` refuses a `tslab` that is not under `<lab>/bin/`, so the
   binary that grades you cannot contain your changes.

   ```sh
   cargo build --release -p turbospark-lab
   LAB=${TURBOSPARK_LAB_DIR:-${TURBOSPARK_HOME:-$HOME/.turbospark}/lab}
   REV=$(git rev-parse --short HEAD)
   mkdir -p "$LAB/bin/$REV"
   cp target/release/tslab target/release/tslab-worker "$LAB/bin/$REV/"
   TSLAB="$LAB/bin/$REV/tslab"
   ```

   Copy BOTH binaries. `profile`, `ab` and `kernel` use the `tslab-worker`
   beside `tslab` and stop with "worker not found" without it.
   `run-experiment` builds its own `tslab-worker` from your tree, so the
   candidate code is what gets measured.

   If the lab is not on `main` yet (it lives on a branch you merged in), the
   integrity check would list every lab file as a touched evaluator input.
   Pass the commit that holds the lab as the base to `gate`, `run-experiment`
   and `confirm`: `--base <that merge commit>`.

3. Find where time goes, on AC power with a quiet host:

   ```sh
   $TSLAB profile --model <install> --prompt protocol:long-synthesis --decode 64 \
       --experiment <tag>-profile
   $TSLAB kernel --model <install> --experiment <tag>-kernel
   ```

   The largest prefill or decode bucket, and any kernel far from its
   bandwidth or compute peak, is where a hypothesis should start.

4. Calibrate noise for the workload your specs will use. The effect floor
   for later A/Bs in the same experiment comes from this:

   ```sh
   $TSLAB ab --model <install> --prompt protocol:long-synthesis --null --experiment <name>
   ```

## One iteration

1. **Hypothesis.** One sentence: what gets faster, by which mechanism, and
   which bucket or curve says so. If the playbook or the kapitsa page already
   records the idea as a dead end, pick another.
2. **Spec.** Write `<name>.spec.json` (template below). Declare the exactness
   class honestly: `bit_exact` if output bits cannot change, otherwise the
   family's KL class (`dense_kl`, or `moe_kl` for routed MoE).
3. **Implement behind a new switch.** Add `TURBOSPARK_<NAME>`, read once
   through a `OnceLock`, default OFF, so the old path stays one variable
   away. The spec's candidate arm sets it, and the baseline does not.
4. **Prove the primitive.** Add a NEW test file (existing tests and their
   tolerances are off limits) that pins the change at its declared class:
   `to_bits()` equality for bit-exact work, a reference comparison for
   KL-bounded work. Break the kernel on purpose once and confirm the test
   fails, then restore it. List the test in `required_tests`.

   `required_tests` run with `cargo test --release`, and so should you while
   iterating: a model-opening test took 1,111 s in the debug profile and under
   a second in release. A switch read once through a `OnceLock` cannot be
   flipped inside one test binary, so set it ONCE for the whole file (a
   `std::sync::Once` that calls `set_var` before any runner is built) and
   compare against a reference path that needs no switch, such as the
   sequential loop. Tests that compare two runs of the new path with each other
   are self-relative and survive a broken kernel: only the comparison against
   the old path has teeth, which is what the mutation check is for.
5. **Commit** the iteration on `lab/<tag>` (only your paths), so a discard
   is one revert.
6. **Evaluate:** `$TSLAB run-experiment --spec <name>.spec.json`. It checks
   the diff, builds the worker, runs the required tests and the quality
   gate, runs the paired A/B under the time budget, checks memory, and
   prints `DECISION:`.
7. **Act on the decision.** The exit code says it too: 0 keep, 3 discard,
   4 inconclusive, 1 error.
   - `inconclusive`: do NOT revert. Too few clean pairs means nothing judged
     the idea, usually because the host was busy. Read the `reason` and the
     "Causes, by runs affected" list `tslab` prints, fix the condition (stop
     the offenders, free disk, wait), and re-run the same spec. After three
     inconclusive runs in a row, say so and stop rather than loop.
   - `discard`: `git revert --no-edit HEAD` (your own commit only). Never
     `reset --hard` or `clean`: other sessions share this repository.
   - `keep_pending_confirm`: run `$TSLAB confirm --spec <name>.spec.json`
     once. It measures the same arms on the held-out `confirm_workload`.
     `keep` leaves the commit in place, and `discard` reverts it as above.
8. **Note it.** Append one line to `<lab>/<name>/notes.md`: hypothesis, ledger
   id, decision, and the bucket or curve that moved. The ledger holds the
   numbers, and the note says why you tried it.

Then the next iteration. Stop an idea after three discards in a row and
move to a different bucket.

## Rules

- **The evaluator is fixed.** You may edit `crates/gpu/src/**` (shaders
  included), `crates/runtime/src/**`, and add new test files and docs.
  `run-experiment` refuses a diff that touches `crates/lab/**`, the shared
  gate math (`crates/bench/src/{logit_compare,quality_corpus,memory}.rs`),
  `crates/bench/prompts/**`, `docs/BENCHMARKS.md`, any `AGENTS.md`,
  `CLAUDE.md` or `Cargo.toml`, `Cargo.lock`, CI, or any existing test file.
- **Never widen a tolerance or flip a default.** The loop proposes. Making a
  switch the default, freezing a row in `docs/BENCHMARKS.md`, and pushing are
  human decisions.
- **No numbers without conditions.** Quote a result only with its ledger id,
  model, workload, chip, power source, and page-cache class.
- **One measuring session per machine.** `tslab` holds a machine lock. If it
  is taken, wait. Do not run model processes beside a measurement.
- **Keep ASCII,** no emojis or em dashes, in code, comments and notes.

## Checks for a new model or family

Collect these in the ledger before any speed or speculation claim about a
family the lab has not measured. Each was a surprise on the dense `qwen35`
family, the readings are in `docs/KAPITSA_KERNELS.md`, and none of them is a
number to copy: a new family has its own.

1. **The verify cost curve, per shape.** `tslab kernel --bench int4_matmul
   --rows 1,2,4,8,9` at the install's own shapes, and read the M-row pass
   against the one-row step. If the pass grows with the rows, only the
   shortest speculative blocks can pay, and comparing drafters is premature:
   fix the curve first. Routed MoE needs its expert-union cost measured too,
   because extra rows bring in extra experts.
2. **The small projections.** The shape list is derived from `ArchConfig`, and
   a shape missing from it cannot be seen. Check that the short ones are there
   (attention K/V, the gated-DeltaNet gate projections). A kernel with a
   row-count threshold can serve the large shapes and leave the small ones on
   an older, slower kernel, which costs launch latency a bandwidth model never
   predicts.
3. **Where a round goes.** Set `TURBOSPARK_SPEC_STATS=1` in the arm and read
   `ms_per_round` (draft, verify, accept, rollback, commit, rewind, bonus).
   Different drafters at the same block can cost the same per round, which
   means the lever is the verify and not the drafter. A rollback that is a
   small fraction of a round needs no further work.
4. **Acceptance, per position.** `NAME~dflash:7` or `NAME~mtp:4` arms give
   committed tokens per round and the per-position curve. Derive the best
   fixed depth from the curve and the measured round cost before building an
   adaptive controller. When verify is flat in rows, a row pays once its
   chance of being reached passes a few percent, and a fixed deep block was
   within a couple of percent of the best per-request choice.
5. **Which attention kernel each scratch uses.** The trunk, the MTP head and a
   block drafter each build their own attention scratch. One can default to
   the barrier decode kernel while another uses the GQA-shared one, and the
   cost of the older one grows with the span. Compare the draft phase at a
   short and at a multi-thousand-token prompt.
6. **Prefill on the install actually served.** Drafter and head variants of
   one family can take different prefill drivers (chunked or token by token).
   Measure prefill tokens per second for the plain, the MTP and the block
   drafter installs separately, at a short and a long prompt. A decode win on
   an install whose prefill is an order of magnitude slower is a latency
   regression for any request with a real prompt.
7. **Memory per install variant.** Peak footprint for each, and what each
   drafter's state costs. When comparing against another engine, say how each
   one counts memory (mapped weights against allocated buffers).
8. **The exactness class.** Declare it in the spec. A change that must keep
   decode and verify on one arithmetic has to move both. A switch that is
   global moves every family: gate the family you changed and name the ones you
   did not.
9. **Controls.** Every arm sets its variables itself in a cleared
   environment. Never use a forced-kernel variable as the baseline, because it
   also forces the trunk. Null-calibrate. Do not time on a host that is
   swapping: one stalled run here took 22 minutes, and the lab's host rules
   exist for that reason.
10. **Comparing with another engine.** Count tokens with one tokenizer, never
    stream chunks. Render the prompt the same way and compare the generated
    lengths first: different chat templates or thinking behavior make the
    texts differ, and acceptance on different text says little. Keep time to
    first token apart from decode, and know whether the other engine reuses a
    prefix across runs.
11. **What earlier testing did not cover.** Sampled (temperature above zero)
    speculative throughput, decode at contexts of 16K and beyond, concurrent
    requests, and output quality against the other engine. Say so when a claim
    needs one of them.
12. **Would a tree verify pay.** `tslab tree-replay --model M --drafter
    mtp|dflash` decodes greedily, drafts a lattice after every token, and
    replays the loop for the chain, a best-first tree (kapitsa's KFlash) and
    a per-round oracle at the same verify rows. It is exact on the count and
    models no cost, so it only answers whether tree verify is worth costing:
    a mean tree gain under 5% committed/round at the served block
    (`--serve-block`, default the block that pays today) says no. A gain at
    a deeper block counts only once a cheaper verify makes that block pay.
    Under `TURBOSPARK_INT4_MMA8` DFlash2 serves depth 7, so pass
    `--serve-block 7` for that question. An MTP head
    caps the gain at one token per round (siblings are leaves); a DFlash2
    lattice does not. The result and its reading for qwen38 are in
    `docs/KAPITSA_KERNELS.md` (KFlash section).

Workloads under `crates/lab/workloads/` are for `--prompt messages:PATH`.
`copy-heavy.messages.json` is a quote-back task (rename one variable in a short
function and return all of it), the shape where lookup drafting was expected to
pay and where a block drafter commits nearly the whole block. The 8-row tile
(`TURBOSPARK_INT4_MMA8`) is not on `main` yet, so the kernel bench has no arm
for it; add one when it lands, next to the scalar GEMM and the Steel tiles.

## Spec template

```json
{
  "name": "qwen36-fused-router",
  "hypothesis": "Fusing the router GEMV into the attention cb removes one wait per layer. The router readback bucket is N% of prefill in profile ledger id <id>.",
  "model": "/Users/me/models/qwen36.gturbo",
  "knob_env": "TURBOSPARK_FUSED_ROUTER",
  "baseline": {"name": "off"},
  "candidate": {"name": "on", "env": {"TURBOSPARK_FUSED_ROUTER": "1"}},
  "workload": {"prompts": [{"protocol": "long-synthesis"}], "decode_tokens": 64},
  "confirm_workload": {"prompts": [{"protocol": "medium-review"}, {"corpus": 2000}], "decode_tokens": 64},
  "primary_metric": "prefill_tok_s",
  "min_effect": 0.02,
  "exactness": "moe_kl",
  "required_tests": ["-p turbospark-gpu --test fused_router_parity"],
  "time_budget_min": 45,
  "min_pairs": 3,
  "max_pairs": 6
}
```

Fields: `primary_metric` is `prefill_tok_s`, `decode_tok_s`, `prefill_s` or
`peak_bytes`. Arms may also carry `"open": {"slots": {"fixed": 32},
"residency": "streamed"}`. Prompts are `{"protocol": id}`,
`{"messages_file": path}`, `{"text_file": path}` or `{"corpus": tokens}`.
`quality_bounds` and `memory_allowance_bytes` may only tighten the defaults.

Arms can decode through the speculative loop: `"speculation": {"drafter":
"mtp", "block": 2}` in the arm (or the workload), or `NAME~mtp:2` on the
command line (`~dflash:7`, `~mtp:4a` for adaptive depth). Set
`TURBOSPARK_SPEC_STATS=1` in the arm's env to get committed tokens per round
and ms per round by phase in the record. For a speculation-only change,
declare `bit_exact`: the gate then compares generated tokens against the
baseline arm.

## What tslab does not measure yet

Say so in the hypothesis rather than measure around it:

- **Energy.** The ANE prefill split's case rests partly on modeled energy
  (IOReport, about 0.65x), and tslab has no energy meter. Use
  `scripts/power.sh` for joules until one exists.
- **Multi-turn, concurrent and mixed workloads.** Time to first token on a
  cached follow-up, aggregate tok/s and p95 inter-token latency over several
  streams, and decode stalls while another request prefills are all
  single-stream blind spots here.
- **Sampled speculation.** Speculative arms run greedy. Exact rejection
  sampling at temperature above 0 is not exercised.
- **Prefill and decode split inside a speculative arm.** The speculative
  loop owns its prefill, so such a record carries one combined phase
  report, plus wall seconds for each half.
- **Tree verify cost.** `tslab tree-replay` counts what a tree would
  accept; no tree kernel exists here, so its overhead (masked attention,
  per-branch recurrent state, KV compaction) and its c(M) are unmeasured.
- **Kernel benches beyond two.** `tslab kernel` covers INT4 matmul and
  decode attention. The MoE routed pair, the MMA8 verify tile, multi-row
  prefill attention and a bit-hash corpus over outputs are not ported.
- **Phase buckets on dense families.** `PhaseCounters` is filled by the
  MoE flows. On dense Qwen3.8 every named bucket reads 0, so a profile there
  gives wall times and spec-stats only. Use `TURBOSPARK_DISPATCH_PROFILE=1`
  (unquotable) to rank kernels, or attribute by skipping a kernel behind a
  switch.
- **Per-round milliseconds on this base.** `ms/round` in `[spec-stats]`
  comes from uncommitted runtime work. Until it lands, speculative records
  carry committed tokens per round and per-position acceptance, not draft
  and verify times.
- **Dispatch-level shares.** `TURBOSPARK_DISPATCH_PROFILE=1` output is
  kept as text in the record, unparsed, and such runs are never quotable.
- **ANE arms need an `ane-prefill` worker.** The worker refuses an ANE arm
  it was not built for, rather than measuring Steel under the label. The
  split also needs `--chunk 2048`.

## Reading a verdict

- `WIN`: at least 3 clean pairs, median improvement at or above the effect
  floor (the larger of 2%, the spec's `min_effect`, and the null run's
  floor), the bootstrap CI above zero, and every pair agreeing (or a
  sign-test p of 0.05 or less from 6 pairs).
- `NOISE`: enough clean pairs, no effect at the floor. Not a reason to
  re-run until it wins: a re-run of the same spec is a new hypothesis only
  if something changed.
- `INCONCLUSIVE`: too few clean pairs (busy host, refused arm, budget). Fix
  the condition, not the threshold. `run-experiment` records it as the
  decision `inconclusive`, NOT `discard`, so nothing is reverted.
  Before each pair `tslab` waits up to `--quiet-wait-min` (default 5) for the
  host to settle, and stops early, naming the offenders, if it does not; it also
  refuses to start under `--min-free-gb` (default 10) of free disk. Pass `0` to
  either to turn it off.
- Dirty runs stay in the ledger with their reasons (`tslab ledger
  --experiment <name> --kind run`).
