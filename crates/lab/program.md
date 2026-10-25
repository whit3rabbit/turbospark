# program.md: the kernel autoresearch loop

Instructions for a Claude Code session that iterates on Metal and ANE
kernel work against `tslab`, the fixed evaluator in this crate. You propose
and implement changes. `tslab` measures and decides. You never decide a
result yourself, and you never edit the evaluator.

Read before the first iteration: `docs/SPEEDUP_PLAYBOOK.md` (how wins were
found and qualified, the kernel checklist, traps), `docs/KAPITSA_KERNELS.md`
("Where we were already ahead" and the decision table: do not re-propose
what it declines), `docs/MLX_SERVE_KERNELS.md` (its "Negative results"
list, and the evaluator additions L1 to L7 this crate now carries),
`docs/BATCHED_PREFILL.md` for prefill drift bounds, and the AGENTS.md files
of every crate you touch. `skill/SKILL.md` beside this file is the short
form for an agent: the `--json` contract and which command answers what.

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

3. Find where time goes, on AC power with a quiet host. Start with what
   the install supports, then the survey:

   ```sh
   $TSLAB capabilities --model <install>
   $TSLAB workflow new-family --model <install> --experiment <tag>-survey
   ```

   `new-family` runs this page's family checks that apply to the install
   (profile, hashed kernel curves, null floor, dispatch probe, speculative
   round costs, verify ladder, tree replay) and writes
   `<lab>/<tag>-survey/report.html`. Any step also runs alone, for example:

   ```sh
   $TSLAB profile --model <install> --prompt protocol:long-synthesis --decode 64 \
       --experiment <tag>-profile
   $TSLAB kernel --model <install> --hash --experiment <tag>-kernel
   $TSLAB trace --model <install> --prompt protocol:short-explanation --decode 64 \
       --experiment <tag>-trace
   ```

   The largest prefill or decode bucket, any kernel far from its bandwidth
   or compute peak, and any trace with low GPU busy inside decode
   (launch-bound) is where a hypothesis should start. On a dense family the
   phase buckets are not filled: price blocks with `profile --standin
   attn,mlp,gdn` and dispatches with `profile --dispatch-probe 0,4,8`.

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

## Pinned settings from the compared engines

`crates/lab/pinned/` holds a verified, embedded reference for each kernel
the engines this repository compares against ships (`nace-ai/kapitsa`,
`jundot/omlx`, `ddalcu/mlx-serve`): the repo and 40-hex revision each entry
was read at, the models it serves, its hardcoded settings traceable to a
`path:line`, its dispatch gates, and the numbers the engine itself reports
(provenance `upstream_reported`, always with conditions). `tslab pinned`
lists the set (`--repo`, `--class`, `--id`, `--json`) and loads a directory
variant (`--dir`); the schema, field rules, and sweep workflow are in
`crates/lab/pinned/README.md`.

Use it at the hypothesis and spec stages:

- When an idea mirrors an upstream mechanism, take the setting values, the
  gates that route to it, and its cost curves from the pinned entry, not
  from memory or a half-remembered topic doc. The entry's `notes` already
  record where those docs and the source disagree.
- Each entry's `lab_mapping` says where a setting meets our surfaces: a
  `TURBOSPARK_*` knob, a `tslab kernel` bench axis, the workload's
  `speculation.block` field, or `none yet`. `none yet` means the axis has
  to be built before the sweep can run, which is part of the hypothesis's
  cost.
- To sweep against an upstream baseline: copy the entry's JSON, change only
  what the sweep varies, and load it with `tslab pinned --dir DIR` (or
  `PinnedSet::load_dir` in code). A same-id copy replaces the embedded
  baseline for that process, so the tweaked file keeps the original id and
  every reader rule still applies.

## Checks for a new model or family

Collect these in the ledger before any speed or speculation claim about a
family the lab has not measured. Each was a surprise on the dense `qwen35`
family, the readings are in `docs/KAPITSA_KERNELS.md`, and none of them is a
number to copy: a new family has its own. `tslab workflow new-family` runs
the measurable ones and skips, with the reason ledgered, those the install
cannot run.

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
pay and where a block drafter commits nearly the whole block.

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

Fields: `primary_metric` is `prefill_tok_s`, `decode_tok_s`, `prefill_s`,
`peak_bytes` or `followup_ttft_s` (or, for an audio workload, `audio_run_s`
or `rtf`). Arms may also carry `"open": {"slots":
{"fixed": 32}, "residency": "streamed"}`. Prompts are `{"protocol": id}`,
`{"messages_file": path}`, `{"text_file": path}`, `{"corpus": tokens}` or
`{"chat": path}`.
`quality_bounds` and `memory_allowance_bytes` may only tighten the defaults.
A spec may not declare `moe_kl` on an install that routes no experts.

An arm may list `"expect_engaged": ["my_lane"]`: the run must print
`[engaged] my_lane` (`foundation::engaged!("my_lane")` where the new lane
runs) or it is ineligible. Registered knobs need no list; their lines are in
`knobs.rs`. Use it for the switch under test, so an arm that silently ran
the old path cannot read as a NOISE verdict.

An audio spec sets `"workload": {"prompts": [], "audio": {"task": "music",
...}}` (see `workloads/audio/`), declares `bit_exact` (output bytes equal) or
`text_exact` (transcript equal), and uses an audio metric.

Arms can decode through the speculative loop: `"speculation": {"drafter":
"mtp", "block": 2}` in the arm (or the workload), or `NAME~mtp:2` on the
command line (`~dflash:7`, `~mtp:4a` for adaptive depth). Set
`TURBOSPARK_SPEC_STATS=1` in the arm's env to get committed tokens per round
and ms per round by phase in the record. For a speculation-only change,
declare `bit_exact`: the gate then compares generated tokens against the
baseline arm.

Chats (`{"chat": path}`, `chat:<path>` on the command line) run a
conversation: an opening turn, optionally with a long document in front of
it, then each follow-up, the model's replies carried back without their
reasoning. Prefix reuse is on for the whole chat, so `followup_ttft_s`
(mean time to first token after the first turn, restore included) is what
a client waits for. `crates/lab/chats/long-doc.chat.json` is a 6000-token
document and four follow-ups. Replies stop at end-of-turn, so a chat's
decode work can differ between arms that change output; a chat workload is
judged on generated tokens and must declare `bit_exact`. That gate is
`gate`, `run-experiment` and `confirm`. `tslab ab` reports `followup_ttft_s`
but compares no tokens, so an `ab` result on a chat says nothing about
whether the arms produced the same text.

A reply cut inside the reasoning goes back empty, which is what a client
that drops reasoning sends. With a small `--decode` on a thinking model most
replies end there, so the re-rendered reply is empty and the experiment
prices only the new message. Give `--decode` room to finish the thinking
when the re-rendered reply is what you are measuring. For a template with no
thinking channel, set `"reasoning": false` in the chat file (default `true`):
the whole reply is sent back instead of nothing.

The fit check renders the opening turn only. A later turn that overflows
`--context` fails mid-run, after the warm pass, with the turn named. The
knob that targets chats is `gdn_turn_snapshot`:

```sh
$TSLAB ab --model <install> --prompt chat:crates/lab/chats/long-doc.chat.json \
    --decode 32 --context 16384 --baseline off~mtp:2 --candidate on~mtp:2:gdn_turn_snapshot=1 \
    --metric followup_ttft_s --experiment <name>
```

## What tslab does not measure yet

Say so in the hypothesis rather than measure around it:

- **Energy.** The ANE prefill split's case rests partly on modeled energy
  (IOReport, about 0.65x), and tslab has no energy meter. Use
  `scripts/power.sh` for joules until one exists.
- **Concurrent and mixed workloads.** Aggregate tok/s and p95 inter-token
  latency over several streams, and decode stalls while another request
  prefills, are single-stream blind spots here. Chats cover one client's
  follow-up time to first token, and only with the reasoning dropped from
  history.
- **Sampled speculation.** Speculative arms run greedy. Exact rejection
  sampling at temperature above 0 is not exercised.
- **Prefill and decode split inside a speculative arm.** The speculative
  loop owns its prefill, so such a record carries one combined phase
  report, plus wall seconds for each half.
- **Tree verify cost.** `tslab tree-replay` counts what a tree would
  accept; no tree kernel exists here, so its overhead (masked attention,
  per-branch recurrent state, KV compaction) and its c(M) are unmeasured.
- **Kernels without a bench.** MLA and compressed attention, TurboQuant
  attention, and GGUF, MXFP4, 1-bit and 2-bit matmuls are not benched;
  `tslab capabilities` lists what an install's benches skip and why. The
  `moe_routed` bench models the expert union as uniform routing, an upper
  end; real routing is skewed.
- **Phase buckets on dense families.** `PhaseCounters` is filled by the
  MoE flows only. Dense profiles print which buckets read zero. Stand-ins
  (L4) and the dispatch probe (L1) are wired for dense FFN layers in the
  qwen and llama decode loops only; other flows print no engagement line,
  so their arms are refused rather than mismeasured.
- **Traces.** `tslab trace` attaches for a short window (60 s at most:
  longer recordings were killed while saving here), and a traced run is
  never quotable. Shader shares come from samples, so compare shares
  between traces, not times. `TURBOSPARK_DISPATCH_PROFILE=1` text is still
  kept unparsed and unquotable.
- **Audio.** Worker arms run MiniMax Music 3 and Whisper; Kokoro,
  MegaTTS3, OmniVoice, Moonshine and Qwen3-ASR are kernel-bench only. The
  Whisper arm and the `whisper_linear` bench have not run against a real
  Whisper install on this host. Audio has no KL bound: its gates are
  output identity only.
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
