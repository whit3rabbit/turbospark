---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e01"
title: "Running the tests"
summary: "cargo test --workspace runs the standing suite (~1,435 tests, macOS). The pre-handoff gate adds build, fmt --check, clippy --tests"
tags: ["testing", "day-one", "ci"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e09", "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e08", "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e07"]
source: "docs/TESTING.md, docs/DEVELOPMENT.md, .github/workflows/ci.yml"
created: "2026-09-04"
updated: "2026-09-04"
---

## How do I run the tests?

```sh
cargo test --workspace
```

Runs the whole standing suite on macOS, including every Metal test (needs a
real Metal-capable device and `xcrun -sdk macosx metal`). On Linux
`crates/gpu` compiles to nothing, so the same command stays green there too,
just narrower.

The full pre-handoff gate (what should be green before you call something
done) is four commands, not one:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

`make check` runs three of the four (fmt-check + clippy + test-debug).
`make` with no target is `make check`.

Re-count the test total before quoting it anywhere: as of 2026-08-29 it was
~1,435 passing plus 147 `#[ignore]`d, and both numbers have previously sat
stale for weeks (once by more than 2x) because nothing goes red when a count
in prose rots. Re-derive with `cargo test --workspace` (first number) and
`for d in crates/*/tests; do grep -rh '#\[ignore' $d/*.rs; done | wc -l`
(second).

## Don't

- Don't assume `fmt --check` / `clippy` failures are yours. Both are
  tree-wide and this repo is routinely worked by more than one session at
  once. Read the paths named in the failure before debugging.
- Don't kill a `cargo test --workspace` that looks hung at 0% CPU right
  after a build. That's `syspolicyd` (Gatekeeper) verifying each freshly
  built test binary on first execution, ~30s each across dozens of binaries.
  Sample `ps` twice 20s apart. A different binary name each time means it's
  advancing, not stuck. A cold full-workspace run can take ~75 minutes on a
  laptop for this reason (see [[machine-specific-test-notes]]).
- Don't treat a green `cargo test --workspace` as proof a decode-path,
  output-head, KV-cache, or Metal-encode-loop change is safe. Those need the
  real-model smoke tests too (see [[real-model-smoke-tests]]). A pure file
  split once shipped a model whose reference perplexity read 255,409 against
  a frozen 6.2536 with the whole workspace suite green.
- Don't run the `#[ignore]`d tests as part of routine verification. They're
  opt-in: checkpoint downloads, per-family memory oracles and quality gates,
  cross-engine KL dumps, real-install behaviour gates. `cargo test
  --workspace -- --ignored` runs all of them at once, but three download
  many gigabytes. See [[ignored-tests-and-real-model-gates]].

## Why this shape

The four-command gate exists because none of the three (build, test, fmt)
implies the others catch what clippy does, and vice versa. `make check`
deliberately runs `test-debug` (not release) because that's the fast loop.
Release-mode gates are for the model-backed tests, which need real
throughput numbers.
