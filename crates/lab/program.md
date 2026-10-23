# program.md: the kernel autoresearch loop

Instructions for a Claude Code session that iterates on Metal and ANE
kernel work against `tslab`, the fixed evaluator in this crate. You propose
and implement changes. `tslab` measures and decides. You never decide a
result yourself, and you never edit the evaluator.

Read before the first iteration: `docs/SPEEDUP_PLAYBOOK.md` (how wins were
found and qualified, the kernel checklist, traps), `docs/KAPITSA_KERNELS.md`
("Where we were already ahead" and the decision table: do not re-propose
what it declines), `docs/BATCHED_PREFILL.md` for prefill drift bounds, and
the AGENTS.md files of every crate you touch.

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
   cargo build --release -p turbospark-lab --bin tslab
   LAB=${TURBOSPARK_LAB_DIR:-${TURBOSPARK_HOME:-$HOME/.turbospark}/lab}
   REV=$(git rev-parse --short HEAD)
   mkdir -p "$LAB/bin/$REV" && cp target/release/tslab "$LAB/bin/$REV/tslab"
   TSLAB="$LAB/bin/$REV/tslab"
   ```

   `run-experiment` builds `tslab-worker` from your tree itself, so the
   candidate code is what gets measured.

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
5. **Commit** the iteration on `lab/<tag>` (only your paths), so a discard
   is one revert.
6. **Evaluate:** `$TSLAB run-experiment --spec <name>.spec.json`. It checks
   the diff, builds the worker, runs the required tests and the quality
   gate, runs the paired A/B under the time budget, checks memory, and
   prints `DECISION:`.
7. **Act on the decision.**
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
- **Kernel benches beyond two.** `tslab kernel` covers INT4 matmul and
  decode attention. The MoE routed pair, the MMA8 verify tile, multi-row
  prefill attention and a bit-hash corpus over outputs are not ported.
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
  the condition, not the threshold.
- Dirty runs stay in the ledger with their reasons (`tslab ledger
  --experiment <name> --kind run`).
