---
title: Verify a Change Before Handoff
description: The contributor gate runbook. Which commands to run, how to decide which real-model gates a change owes, and what a green run of each one proves.
---

## Goal

Hand off with the tree green and the real-model gates your change actually
needs already run. "Green" alone is not the goal: a one-family numerics
regression has shipped through a fully green workspace suite before (see
Pitfalls), so the work here is deciding which additional gates the change
owes and running them per family touched.

## Prerequisites

- A built workspace: `cargo build --workspace` passes.
- A real `.gturbo` install for each family your change touches, because the
  gates that catch numerics regressions are gated on an install path via an
  environment variable. Gemma 4's gate needs roughly 14.6 GB on disk.
- macOS with a Metal-capable device for the real-model gates and smokes.

## Steps

### 1. Run the four workspace commands

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

`fmt --check` and `clippy --workspace --tests` are tree-wide. If one goes red
naming paths you did not touch, another session's in-flight edit is the likely
cause: check mtimes before debugging, and use the per-file form for your own
files while the tree is contested:

```sh
rustfmt --check --edition 2021 <your files>
```

### 2. Classify the change

Decide whether the change touches the **decode path, the output head, the KV
cache, or a Metal encode loop**. If it does, steps 3 and 4 are mandatory.

"Touches" includes **moving** the code. A pure refactor of a family flow is a
numerics change until a real model says otherwise: commit `5279c88` split
`real_forward_qwen.rs` into `families/qwen/`, moved no math on purpose, picked
up a Gemma-style sandwich norm on the way, and shipped a Qwen whose
reference-answer perplexity read 255,409 against a frozen 6.2536, with the
whole workspace suite green (`crates/runtime/CLAUDE.md` Gotcha 11).

### 3. Run the smoke pair, greedy AND sampled

```sh
cargo build --release -p turbospark-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy. Catches broken math.
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. Sampled, at the CLI defaults (T=0.2, top-k 64, top-p 0.95). Catches
#    distribution bugs greedy cannot see.
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

Greedy alone is not a smoke test. It is `argmax`, and argmax is invariant
under every monotone transform of the distribution, so greedy stays
byte-identical to correct through bugs that destroy sampling entirely. The
sampled run must stay coherent for the whole run and reach `EndOfTurn` on a
short question.

Use `--messages-file`, never a bare `--prompt`, on an instruction-tuned
model: the bare form omits the chat template and babbles, which is not a
decode bug.

### 4. Run the per-family memory oracle

One target per family, always `--release` (otherwise the tok/s numbers are
meaningless) and always with the install env var set:

```sh
# Gemma 4 (~10 min)
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture
```

Every family follows the same shape. Targets and env vars, from
`crates/bench/tests/`:

| Family | Env var | Test target (oracle) |
|---|---|---|
| gemma4 | `TURBOSPARK_GEMMA4_INSTALL_DIR` | `memory_oracle` |
| qwen36 | `TURBOSPARK_QWEN36_INSTALL_DIR` | `qwen36_memory_oracle` |
| qwen38 (dense) | `TURBOSPARK_QWEN38_INSTALL_DIR` | `qwen38_memory_oracle` |
| qwen3moe | `TURBOSPARK_QWEN3MOE_INSTALL_DIR` | `qwen3moe_memory_oracle` |
| qwen4exp | `TURBOSPARK_QWEN4EXP_INSTALL_DIR` | `qwen4exp_memory_oracle` |
| gptoss | `TURBOSPARK_GPTOSS_INSTALL_DIR` | `gptoss_memory_oracle` |
| museglimmer | `TURBOSPARK_MUSEGLIMMER_INSTALL_DIR` | `museglimmer_memory_oracle` |
| mistral (dense llama) | `TURBOSPARK_MISTRAL_INSTALL_DIR` | `mistral_memory_oracle` |

Run the oracle for **each family the change touches**, not once for the
workspace. The targets are separate binaries on purpose: one model per
process, because the assertion is against a whole-session peak and a second
model opened in the same process is measured against the high-water mark the
first one left (`crates/bench/tests/oracle_common/mod.rs`). Do not add a
second `#[test]` to an oracle file.

What one run asserts (`oracle_common/mod.rs`):

- **Peak `phys_footprint`** at or under the per-chip ceiling. Memory sizing
  does not depend on the chip, so an unknown chip is still asserted against
  the loosest documented ceiling for that family.
- **Steady state**: replaying an already-warm case stops growing the
  footprint (4 replay rounds, 8 MiB slack). This is the scale-free leak check
  the ceiling cannot see.
- **Validity**: every measured case stops with `endOfTurn`, not `maxTokens`.
- **Decode tok/s floor**, per case, only when the chip brand matches a
  baseline row. On an unknown chip tok/s is reported but not asserted.

### 5. Run the per-family quality gate

Same shape, roughly a minute per family:

```sh
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
```

Quality-gate targets sit beside the oracles (`quality_gate` for gemma4,
`qwen36_quality_gate`, `qwen3moe_quality_gate`, `qwen4exp_quality_gate`, and
so on), each reading the same env var as its oracle sibling.

What one run asserts (`crates/bench/tests/quality_common/mod.rs`):

1. **Reference-answer perplexity** within a 2% relative tolerance
   (`PERPLEXITY_REL_TOLERANCE`) of the recorded row for this chip.
2. **Greedy golden digest** matches the frozen row.
3. **Sampled golden digest** matches the frozen row. This is the arm that
   sees distribution bugs greedy cannot.
4. **Determinism**: two warm greedy runs in one session produce identical
   output. The one assertion that holds on any chip, with or without a row.
5. **Constrained working set**: the install reopened at 8 expert-cache slots;
   its greedy digest must EQUAL the 16-slot one, and its decode throughput
   must stay at or above 0.5x the 16-slot baseline.

A chip with no recorded row still runs and still asserts determinism;
everything else is measured and printed for you to freeze.

### 6. Explain any digest mismatch

If the change could move numerics (a kernel, the head, quantization, the
sampler), the quality gate is **additive** to the three gates above (greedy
smoke, sampled smoke, oracle), not a replacement for any of them.

A digest mismatch is not automatically a failure: reduce order legitimately
changes bytes. But it is never allowed to pass unexplained, and the
perplexity number is the tiebreak. The gate's detection floor is calibrated
(`crates/bench/tests/quality_sensitivity.rs`): shifting one quantization
level in 0.195% of routed-expert bytes moves perplexity +37.7%, 0.0122% moves
it +10.5%, while 0.0015% moves it +0.54% and is missed.

Before re-freezing a row that stopped matching, prove it is not a regression
(`crates/bench/CLAUDE.md` Gotcha 24):

1. `git stash push -- <the one file>`, rebuild, re-run, `git stash pop`:
   rules your own uncommitted diff in or out immediately.
2. `git bisect` to **and including** the commit that froze the digest, not
   just to a suspect commit.
3. Rule out the install's own files (mtimes) and the toolchain.
4. Inspect the new output for coherence, not just a healthy perplexity.
5. Confirm the new value is stable across several fresh runs.

### 7. Mutation-check every new test

Every new test is mutation-checked before it is believed, and the loop is
seconds:

```sh
cp <test-file> /tmp/<test-file>.bak
perl -0pi -e 's/A/B/' <test-file>
cargo test -p <crate> --test <target>
cp /tmp/<test-file>.bak <test-file>
```

Assert each mutation reddens ONLY its own case. Three rules:

- **Assert the mutation applied.** `cargo fmt` wraps and re-indents match
  arms and long calls, so a pattern written from your draft stops matching
  the file on disk, silently: perl reports nothing when a substitution finds
  no target. Substitute through a helper that fails when the old text is
  absent.
- **Assert it applied where you meant.** Presence is not uniqueness: a
  pattern written at one indent level is a substring of the same line at a
  deeper one, so a first-match replace can hit a sibling call site. Assert
  the old text occurs exactly once, or anchor on a neighbouring line.
- **A survivor whose mutation did apply is a missing test, not a weak one.**
  Ask what actually calls the mutated function. A fixture-feeding test module
  can be green while the real caller goes unguarded; the guard then belongs
  in a different file with a differently-shaped fixture.

## Verify

Green means, per gate:

- **Workspace commands**: all four pass, and any red path they name is one
  you touched.
- **Smoke pair**: greedy output coherent; sampled output coherent for the
  whole run and reaches `EndOfTurn` on a short question.
- **Memory oracle**: session peak under the ceiling at the stated context
  window (the run prints both), replays stop growing, every case ends with
  `endOfTurn`, and per-case tok/s sits above the chip row's floor (for
  example the Gemma 4 row on the development chip: ceiling 2,300 MiB, floor
  25.0 tok/s, source recorded as this port's own measurement).
- **Quality gate**: reference-answer perplexity within 2% of the frozen row,
  greedy and sampled digests match it, determinism holds, and the 8-slot
  digest equals the 16-slot one.
- **A skip is not a pass.** The gate targets are `#[ignore]`d and read an
  env var: a run without the variable set prints "skipping" and returns
  green having asserted nothing. Check that your gate run actually ran.

## Pitfalls

- **A green workspace suite cannot see a one-family numerics regression.**
  `5279c88` shipped a broken Qwen with every workspace test green; only the
  family's quality gate caught it, in 77 seconds, and it was not run. Gates
  run per family touched.
- **Contested tree.** This repo is routinely worked by more than one session
  at once. A compile error in a crate you did not touch is probably not
  yours; check mtimes, do not "fix" another session's half-finished edit,
  and re-run `git status` immediately before staging.
- **Cold GPU.** The first timed run after a build executes on a cold GPU at
  low DVFS clocks, up to 53% slower. Discard at least one warmup run before
  recording any throughput number.
- **The flakiest assertion in the quality gate** is the constrained arm's
  throughput ratio, and its variance sits inside what one machine produces
  in one session (0.27x to 0.85x against a 0.50x floor on a correct tree
  while Gatekeeper verified freshly built test binaries). If it fails, read
  the printed digests first: if they match the frozen row, no numerics moved
  and the failure is throughput alone. Re-run on a settled machine, and
  `git stash push` your diff to compare at HEAD.
- **Power source.** Run formal gates and any throughput comparison on AC
  power; battery thermal throttling distorts tok/s in both directions.
- **Unknown chip.** The oracle asserts memory on any chip but only asserts
  tok/s where a baseline row exists. A green oracle on an unlisted chip says
  nothing about throughput.
