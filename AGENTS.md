# AGENTS.md

Conventions, gotchas, and commands for working in this clean-room Rust
workspace. Derived from the clean code under this root. Keep all code, comments,
and docs ASCII: no emojis and no em dashes (project rule).

## Stack

- Language: Rust, edition 2021, MSRV 1.82 (see `rust-toolchain.toml` and
  `[workspace.package] rust-version`).
- Toolchain pin: stable, with the `rustfmt` and `clippy` components.
- Build system: cargo, resolver "2".
- License: MIT.

## Build, test, dev commands

```sh
# Build every crate in the workspace.
cargo build --workspace

# Run the whole test suite.
cargo test --workspace

# Run one crate only (crates are core and compute today).
cargo test -p core
cargo test -p compute

# Formatting check (must stay clean; enforced in verification).
cargo fmt --check

# Apply formatting.
cargo fmt

# Lint the workspace and its tests (must stay clean).
cargo clippy --workspace --tests
```

The shared foundation crate is named `core` and the compute crate is `compute`.
Add each new crate directory to the `members` list in the root `Cargo.toml` as
it lands, and keep the member list in sync with the directories under
`crates/`.

## Gotchas

1. The `core` crate collides with the standard library `core` in the extern
   prelude. Downstream crates must depend on it under an alias, for example
   `foundation = { package = "core", path = "../core" }`, and refer to it as
   `foundation`. Integration tests that live outside any `extern crate core`
   context can use the bare `core::` path, but library code should prefer the
   alias to stay unambiguous.

2. Runtime configuration numeric setters abort construction by panicking when a
   value is outside its documented allowed set. This is an intentional fatal
   precondition failure, not a recoverable error: there is no `Result`-returning
   variant and no clamping. Callers that must not abort on bad input should
   validate the value first, or contain the panic with
   `std::panic::catch_unwind`. The allowed sets are exposed as the const arrays
   in `crates/core/src/runtime_config.rs`; read from them rather than
   re-hardcoding the literals.

3. The half-precision logit element is backed by the maintained `half` crate
   (version 2, MIT OR Apache-2.0) because the native `f16` type is unstable on
   the stable toolchain. Do not hand-roll IEEE-754 binary16 storage or
   arithmetic.

4. Token ids cross crate boundaries as signed 32-bit integers
   (`pub type TokenId = i32`). Keep that interchange width when wiring
   downstream crates.

5. `Cargo.lock` is committed on purpose. This workspace targets a command-line
   binary, so the lockfile stays in version control for reproducible builds.
   Do not delete or gitignore it.

6. The cargo build output lives in `/target` and is gitignored. It is large;
   never commit it.

## Layout

- `crates/core`: shared primitives (token id, logit value, logits view) and the
  public runtime configuration with its allowed value sets and builder.
- `crates/compute`: destination-selected compute strategy skeleton. Concrete
  forward-pass and attention compute land in later slices.

## Verification policy

Every change should keep these green before handoff:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

Numerics parity with any upstream implementation is explicitly out of scope;
only the structural and configuration contracts are exercised by the tests.
