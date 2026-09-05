# Development

Setting up, building, testing, and running this repository. Start here on day
one. The pages linked below are the depth.

This tree is two halves that build in one direction. A Rust workspace
(`crates/`) is the engine. A C ABI over it (`crates/ffi`) is compiled to a
static library, which two SwiftPM packages link: `swift/TurboSpark` is the
binding and `swift/TurboSparkApp` is the macOS app. **Nothing Swift here builds
until the Rust half has been built and STAGED**, and that staging step is the
single most common way a first build fails. It has its own section below.

Keep code, comments, and docs ASCII: no emojis and no em dashes (project
rule).

## Prerequisites

| What | Version | Why | Check |
|---|---|---|---|
| macOS on Apple Silicon | 13.0+ for the binding, 14.0+ for the app | The engine is Metal; there is no CPU decode path and no Intel build | `sw_vers -productVersion` |
| Rust | stable, 1.82 or newer | `rust-toolchain.toml` pins the channel and the `rustfmt` + `clippy` components; `Cargo.toml` sets the MSRV | `cargo --version` |
| Xcode command line tools | any recent | The `metal` compiler, which `crates/gpu` needs to build its shaders | `xcrun -sdk macosx metal --version` |
| Swift | 5.9 or newer | Both `Package.swift` files declare `swift-tools-version: 5.9` | `swift --version` |

Two things that are NOT prerequisites and are worth knowing:

- **A model install is not needed to build or to run the standing test
  suites.** Everything gated on a real checkpoint is `#[ignore]`d or reads an
  environment variable and skips. See [`docs/MODELS.md`](MODELS.md) for
  getting one when you want it.
- **Full Xcode is not required**, only the command line tools, unless you
  want Instruments or the simulator. `xcode-select -p` shows which you have.

### On a machine that is not a Mac

Most of the workspace does not build off macOS, by design rather than by
neglect: `crates/runtime` declares `model_io`, `gpu`, `compute` and
`streaming` under a `cfg(target_os = "macos")` dependency block, so anything
depending on it inherits that. Nine crates are portable and are checked as a
group:

```sh
cargo check --target x86_64-unknown-linux-gnu \
  -p turbospark-core -p turbospark-compute -p turbospark-model-io \
  -p turbospark-streaming -p turbospark-selection -p turbospark-invocation \
  -p turbospark-window-fit -p turbospark-gpu -p turbospark-vision-io
```

Run that after touching a `cfg`, a dependency table, or anything `unsafe`.
`turbospark-gpu` is in the list on purpose despite being Metal-only: it must
reduce to an empty shell off macOS, and it has silently failed to twice
(`AGENTS.md` Gotcha 8).

## The `Makefile` is the front door

Every target below runs from the repository root. `make` with no target is
`make check`.

| Target | What it does |
|---|---|
| `make build` / `build-release` | `cargo build --workspace`, debug or release |
| `make test` / `test-release` | `cargo test --workspace` |
| `make fmt` / `fmt-check` / `clippy` | the three lint gates |
| `make check` | `fmt-check` + `clippy` + `test-debug`. **This is the handoff gate.** |
| `make swift-lib` | builds `crates/ffi` and STAGES it into the SwiftPM package |
| `make swift-test` | the binding's own suite, no model, about a second |
| `make swift-test-real MODEL=...` | the same plus the end-to-end arm, minutes |
| `make swift-app-build` / `swift-app-release` | build the macOS app |
| `make swift-app` | build and run the macOS app |
| `make app-bundle` / `dmg` | the release artifacts, into `dist/` |
| `make install` / `uninstall` | the four CLI binaries into `~/.local/bin` |
| `make clean` | `clean-cargo` + `clean-swift` + `clean-dist` |

Read the `Makefile` itself when a target surprises you: several carry the
reasoning for why they are shaped as they are, and `swift-test-real`'s
`MODEL` / `BLOCKED` / `IMAGE` variables each gate a different install SHAPE
rather than being three ways to say the same thing.

## Building

### The Rust workspace

```sh
cargo build --workspace          # or: make build
```

Nothing special. Binaries land in `target/debug/` or `target/release/`, and
`cargo run -p turbospark-cli --bin turbospark-check -- --help` works in place
of an installed binary throughout [`docs/CLI.md`](CLI.md).

### The staging step, which is where first builds fail

```sh
make swift-lib
```

This builds `crates/ffi` for `aarch64-apple-darwin` and COPIES two files --
`libturbospark_ffi.a` and `turbospark.h` -- into
`swift/TurboSpark/Sources/CTurboSpark/`. It is a copy rather than a reference
because a SwiftPM target may not reach outside its own directory, and both
copies are gitignored.

Three consequences, all of which have cost somebody an afternoon:

- **`swift build` fails with a missing-header error until this has run
  once.** That is the honest ordering, not a bug.
- **A fresh git worktree starts without them**, because gitignored files are
  not carried into one. Run `make swift-lib` in the worktree, or copy the
  directory across.
- **SwiftPM does not treat the archive as a build input**, so a rebuilt `.a`
  under unchanged Swift sources triggers no relink at all --
  `scripts/swift-lib.sh` therefore `touch`es every Swift source in both
  packages. That touch is load-bearing and is also why anything going through
  `make` pays a full Swift rebuild.

### The Swift binding and the app

```sh
make swift-app-build             # debug
make swift-app-release           # release
```

**Iterating on SwiftUI alone, skip `make`, and call SwiftPM directly:**

```sh
cd swift/TurboSparkApp && swift build
```

`make swift-app*` depends on `swift-lib`, which touches every Swift file, so
going through `make` recompiles the whole app every time. Use `make` when the
Rust side moved and SwiftPM when it did not.

**Build and test from `swift/TurboSparkApp` itself, never a subdirectory of
it.** The package passes `-L../TurboSpark/Sources/CTurboSpark` as an unsafe
linker flag, which is resolved against the current directory. From anywhere
else it fails with `linker command failed` and no error line above it. This
is easy to do by accident because the shell keeps its working directory
between commands.

## Testing

### The standing gate

Four commands, and everything here should be green before a handoff:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

`make check` runs three of the four. Two notes on reading them:

- `fmt --check` and `clippy` are TREE-WIDE, so read the paths they name
  before assuming a red one is yours. This tree is routinely worked by more
  than one session at once. `rustfmt --check --edition 2021 <your files>` is
  the per-file form.
- A cold `cargo test --workspace` on macOS can appear hung at 0% CPU for a
  long time. That is `syspolicyd` verifying each freshly built test binary on
  first execution. Sample `ps` twice. A different binary name each time means
  it is advancing.

### The Swift suites

Three of them, and they are separate:

```sh
make swift-test                            # the binding's ABI surface
cd swift/TurboSparkApp && swift test       # the app's own suite, seconds
make swift-test-real MODEL=~/models/gemma4.gturbo   # end to end, minutes
```

`make swift-test` is the only thing in this repository that can check the
hand-written `turbospark.h`: the Rust tests reach the same function bodies
through the rlib and pass against a wrong declaration.

Two mechanics worth having before you need them:

- `swift test --filter SuiteName/testCaseName` runs ONE case in a fraction of
  a second, against seconds for the whole suite. That is what makes a
  mutation check (change the fix, confirm the test reddens, restore) cheap
  enough to do per assertion, which this repository expects for every new
  test.
- `swift test` runs TWO harnesses and prints two summaries. swift-testing's
  `Test run with 0 tests in 0 suites passed` is not the result and a `| tail`
  lands on exactly that. Read `Executed N tests, with M failures`.

### What is not in the standing gate

The `#[ignore]`d tests are opt-in: the checkpoint downloads, the per-family
memory oracles and quality gates, the cross-engine dumps, and the real-install
behaviour gates. When a change could move memory or decode throughput, run an
oracle. When it could move numerics, run a quality gate.
[`docs/TESTING.md`](TESTING.md) is the home for the gating conventions and
the test-writing rules, and [`docs/BENCHMARKING.md`](BENCHMARKING.md) for the
benchmark modes.

Anything touching the decode path, the output head, the KV cache, or a Metal
encode loop additionally needs the real-model smoke tests in `AGENTS.md`, all
of them, per model family the change touches. **A pure refactor counts**: one
file split shipped a model whose reference perplexity read 255,409 against a
frozen 6.2536 with the whole workspace suite green.

### What CI runs, and what it does not

`.github/workflows/ci.yml` has two jobs and the difference matters:

- `verify` runs on every PR. It builds and tests the Rust workspace on macOS
  and checks the portable subset on Linux. **It compiles no Swift at all.**
- `package-macos` builds the app bundle and the DMG. It is push-to-main only.

So a Swift break reaches you on a push to main rather than on the PR. That
job's SDK is also not the one on your machine. `swift/CLAUDE.md` Gotcha 45
has the three failure modes this has already produced. The short version:
build the app locally before merging anything that touches Swift, and read
`make app-bundle` as the closest local approximation of that job.

## Running

```sh
# The CLI, against a model install. Every flag: docs/CLI.md
cargo run --release -p turbospark-cli --bin turbospark-check -- \
  --model ~/models/gemma4.gturbo --prompt "hi"

# The HTTP server (OpenAI, Anthropic and Ollama-shaped routes)
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo

# The macOS app
make swift-app
```

`--prompt` on an instruction-tuned model babbles: that is the chat template
missing, not a decode bug. `--messages-file` applies it.

A `swift run` build is a bare executable rather than an `.app` bundle. Two
things follow. Its `@AppStorage` preferences live in a different domain from
the installed app's, and it cannot be driven by UI automation, because it has
no bundle identifier. `make app-bundle` is the only way to get a real bundle
out of this tree, and [`docs/RELEASE.md`](RELEASE.md) covers what it
assembles.

Environment variables that change runtime behaviour -- profiling seams,
dispatch profiles, the phase report -- are catalogued in
[`docs/ENV.md`](ENV.md).

## Cleaning

```sh
make clean-cargo                 # target/, the big one
make clean-swift                 # both .build trees AND the staged header
make clean-dist                  # dist/
make clean                       # all three
```

`clean-swift` removes the staged files, so the next Swift build needs
`make swift-lib` again.

## When something fails

| Symptom | Cause |
|---|---|
| `turbospark.h` not found, or a missing-module error | `make swift-lib` has not run in this checkout or worktree |
| `linker command failed` with no error above it | `swift build` run from a subdirectory of the package |
| `ld: warning: search path 'Sources/CTurboSpark' not found` | Expected. The library's own flag resolved against the app's root; the link then succeeds on the app's copy |
| Swift changes appear to have no effect | SwiftPM relinked the previous `.a`; run `make swift-lib` |
| `cargo test` hung at 0% CPU after a build | Gatekeeper verifying test binaries; see above |
| A compile error in a crate you did not touch | Probably not yours. Check mtimes before debugging it |
| CI red on Swift, green locally | The packaging job's SDK, not your change. `swift/CLAUDE.md` Gotcha 45 |

## Where to go next

- [`AGENTS.md`](../AGENTS.md) -- the working guide: conventions, the
  verification policy in full, and the gotcha list. `CLAUDE.md` is a symlink
  to it.
- [`swift/README.md`](../swift/README.md) and
  [`swift/CLAUDE.md`](../swift/CLAUDE.md) -- the Swift half in detail.
- [`docs/TESTING.md`](TESTING.md), [`docs/BENCHMARKING.md`](BENCHMARKING.md),
  [`docs/RELEASE.md`](RELEASE.md), [`docs/CLI.md`](CLI.md),
  [`docs/MODELS.md`](MODELS.md), [`docs/ENV.md`](ENV.md).
- Each crate has its own `CLAUDE.md` with the architecture and the gotchas
  that live there. Read it before changing that crate.
