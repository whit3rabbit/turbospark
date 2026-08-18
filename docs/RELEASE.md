# Release process

How a version of turbospark gets from a commit on `main` to a GitHub
Release, a set of crates.io publishes, and an updated Homebrew cask. Read
this before cutting a release; `.github/workflows/release.yml` is the
executable version of this document and should never drift from it.

## What ships

Every crate under `crates/` shares one version number
(`workspace.package.version` in the root `Cargo.toml`). All of them publish
to crates.io except `turbospark-bench` (`publish = false`; it's a dev
harness that reads local model installs, no use off this machine).

Publish order matters: `cargo publish` refuses a crate until every internal
path dependency it declares (`path = "../x", version = "..."`) is already
live on the registry at that exact version. The order below is the one
`cargo publish --workspace --dry-run` itself resolved when this doc was
written (verified locally, not assumed):

1. `turbospark-core`: no internal deps
2. `turbospark-window-fit`: no internal deps
3. `turbospark-compute`: needs core
4. `turbospark-invocation`: needs core
5. `turbospark-model-io`: needs core
6. `turbospark-selection`: needs core
7. `turbospark-tokenizer`: needs core
8. `turbospark-repack`: needs core, compute, model-io
9. `turbospark-catalog`: needs core, model-io, repack, tokenizer
10. `turbospark-streaming`: needs core, model-io
11. `turbospark-gpu`: needs core; macOS target-deps on model-io
12. `turbospark-runtime`: needs core, selection, tokenizer; macOS target-deps on gpu, model-io, compute, streaming
13. `turbospark-cli`: needs invocation, runtime, selection, tokenizer, repack, model-io, window-fit
14. `turbospark-server`: needs core, runtime, selection, tokenizer; macOS target-deps on repack

`.github/workflows/release.yml`'s `publish-crates` job hardcodes this same
order and publishes one crate at a time (`cargo publish -p <crate>`) rather
than a single `cargo publish --workspace` call. That's deliberate, not
stylistic: `--workspace`'s behavior when a crate in the middle of the chain
is already published (partial retry after a prior failed run) isn't
documented, and per Cargo's own changelog "`cargo publish` is still
non-atomic at this time. If there is a server side error during the
publish, the workspace will be left in a partially published state." A
per-crate loop that treats "already uploaded" as success for that one crate
and moves on gives a re-run of the same tag correct resume behavior no
matter what the batch mode does internally.

If a new crate is added to the workspace, or an existing crate grows a new
internal dependency, update this list AND the `ORDER` array in
`release.yml`'s `publish-crates` job together. Re-derive the order with:

```sh
cargo publish --workspace --dry-run --allow-dirty
```

A dry run only actually resolves interdependencies once earlier crates
are really on the registry. On a clean, nothing-published-yet repo it
will fail partway through; on this repo, with everything already
published, it is the authoritative check.

## Cutting a release

1. **Update `CHANGELOG.md`.** Move everything under `## [Unreleased]` to a
   new `## [X.Y.Z] - YYYY-MM-DD` section, dated the day you're releasing.
   Leave a fresh empty `## [Unreleased]` section above it for the next
   round of changes. CI enforces that this section exists and matches the
   tag (`check-version` job); a release cannot ship without it.
2. **Bump the version.** Edit `workspace.package.version` in the root
   `Cargo.toml` to `X.Y.Z`, matching the changelog heading. Then bump every
   internal path dependency's `version = "..."` field in every crate's
   `Cargo.toml` to the same `X.Y.Z`; `check-version` asserts this too
   (`grep -n 'path = "\.\./' crates/*/Cargo.toml`, checked against the tag).
   A `cargo build --workspace` after the bump will fail loudly if any
   `version.workspace = true` reference was missed.
3. **Commit and merge to `main`**, e.g. `chore(release): vX.Y.Z`.
4. **Tag it**: `git tag vX.Y.Z && git push origin vX.Y.Z`. The tag is what
   triggers `release.yml` (`on.push.tags: ['v[0-9]*']`); nothing runs on
   the commit itself.
5. **Watch the Actions run.** In order: `check-version` (fails fast if the
   tag, the workspace version, or the changelog entry disagree) ->
   `build-macos` (release binaries, zipped) -> `release` (GitHub Release,
   with the changelog section for this version as the body and
   `generate_release_notes: true` appending the auto-generated PR/commit
   list below it) -> `publish-crates` (crates.io, skipped with a notice if
   `CARGO_REGISTRY_TOKEN` isn't set) -> `update-homebrew-cask` (skipped with
   a notice if `HOMEBREW_TAP_TOKEN` isn't set).

## Prerequisites

- **Local cargo >= 1.90** if you ever want to run `cargo publish
  --workspace` by hand outside CI. Multi-package publishing stabilized in
  Cargo 1.90 (2025-09-18). `rust-toolchain.toml` here pins `channel =
  "stable"`, so a `rustup update` keeps you current; the workspace's
  declared `rust-version = "1.82"` is the MSRV floor for *consumers*
  compiling this code, not the cargo binary you invoke `publish` with.
- **`CARGO_REGISTRY_TOKEN`** repo secret: a crates.io API token
  (`cargo login` locally, or generate one at
  https://crates.io/settings/tokens). Without it, `publish-crates` no-ops
  with a `::notice::` rather than failing, so a release can still ship
  binaries and a GitHub Release without touching crates.io.
- **`HOMEBREW_TAP_TOKEN`** repo secret: a PAT with push access to
  `whit3rabbit/homebrew-tap`, only needed for the cask update. Same
  no-op-with-notice behavior when unset.
- **`cargo login`** if publishing manually (not through CI): the token
  needs publish rights on every crate name below, which is automatic on
  first publish (crates.io grants the publishing account ownership) and
  matters afterward for yanking or adding co-owners.

## Homebrew tap

`update-homebrew-cask` is the last job in `release.yml`. It runs after
`release` (needs the GitHub Release's zip asset to already exist) and is
gated on the `HOMEBREW_TAP_TOKEN` repo secret exactly like `publish-crates`
is gated on `CARGO_REGISTRY_TOKEN`. Set, it pushes; unset, it no-ops with
`::notice::` and the rest of the release still ships.

Confirmed by reading the job, not assumed: **the CLI is one of the three
binaries the cask installs.** The `build-macos` job zips `turbospark-check`
and `turbospark-model` (both `[[bin]]` targets of the one `turbospark-cli`
crate, so `cargo build -p turbospark-cli` produces both) plus
`turbospark-server` into one `turbospark-${VER}-macos-arm64.zip` GitHub
Release asset; `update-homebrew-cask` downloads that exact asset, hashes
it, and writes a cask with a `binary` stanza for each of the three.
`brew install --cask turbospark` (from `whit3rabbit/homebrew-tap`)
therefore puts all three on `PATH`. There is no separate per-binary cask:
one cask, one zip, every binary.

This is Apple-Silicon-only by construction: `build-macos` builds only
`aarch64-apple-darwin`, and the rendered cask hardcodes
`depends_on arch: :arm64`. There's no Intel Mac path today; if one is ever
added it needs a second `build-*` job producing an `x86_64` zip and a
second `sha256`/`url` stanza in the cask (Homebrew casks support per-arch
`sha256`/`url` blocks), not a second cask.

The generated cask is checked for Ruby syntax validity
(`ruby -c homebrew-tap/Casks/turbospark.rb`) before commit, but not for
Homebrew's own `brew audit`/`brew style` conventions; those need `brew`
installed, which this job doesn't do. Worth adding if the cask ever grows
past this template, not urgent while it's five static lines plus two
interpolated values.

## Local dry run before tagging

Do this before pushing a tag, not after. A bad tag is cheap to delete; a
half-published crates.io chain is not:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests

# Per-crate metadata/build check, in the order above. Requires the
# PREVIOUS release to already be on the registry (see the dry-run caveat
# above) -- this is a sanity check on your working tree, not a substitute
# for watching the real `publish-crates` job.
cargo publish --workspace --dry-run --allow-dirty
```

`--allow-dirty` is for the dry run only. A real `cargo publish` (including
the one CI runs) refuses on an uncommitted working tree by design; commit
everything (the changelog move, the version bump) before tagging.

## If something goes wrong mid-release

- **Wrong changelog entry, tag not pushed yet**: fix the commit, re-tag.
- **Tag already pushed, `check-version` failed**: delete the tag
  (`git push --delete origin vX.Y.Z && git tag -d vX.Y.Z`), fix the
  problem, re-tag. Nothing downstream ran, so this is fully reversible.
- **`publish-crates` partially failed** (say, crate 6 of 14 failed for a
  transient registry reason): fix the cause if it's a real error, then
  re-run the same workflow run (or push the same tag's SHA again; tags
  are immutable once other jobs succeeded, so re-triggering `release.yml`
  needs a `workflow_dispatch` re-run of the failed job from the Actions
  UI). The per-crate loop treats every already-published crate as a
  no-op and continues, so a re-run correctly resumes rather than
  re-attempting crates 1-5.
- **A bad version got published for real**: crates.io has no delete, only
  yank (`cargo yank -p <crate> --vers X.Y.Z`), which prevents new
  dependents from resolving to it without removing it for existing lock
  files. Ship a fixed `X.Y.(Z+1)` through this same process; don't try to
  reuse the yanked version number.
