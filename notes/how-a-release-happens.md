---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e03"
title: "How a release happens"
summary: "Move CHANGELOG.md Unreleased section to a dated heading, bump workspace version, tag vX.Y.Z, push. release.yml does the rest"
tags: ["release", "day-one", "ci", "deploy"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e12"]
source: "docs/RELEASE.md, .github/workflows/release.yml"
created: "2026-09-04"
updated: "2026-09-04"
---

## How does a release or deploy happen?

There is no continuous deploy. A release is triggered by pushing a version
tag, and the changelog IS the release notes (not documentation of them):

1. Move everything under `## [Unreleased]` in `CHANGELOG.md` to a new
   `## [X.Y.Z] - YYYY-MM-DD` heading, dated today. Leave a fresh empty
   `## [Unreleased]` above it.
2. Bump `workspace.package.version` in the root `Cargo.toml` to `X.Y.Z`,
   then bump every internal path dependency's `version = "..."` field in
   every crate's own `Cargo.toml` to match. `cargo build --workspace` fails
   loudly if a `version.workspace = true` reference was missed.
3. Commit and merge to `main` (e.g. `chore(release): vX.Y.Z`).
4. `git tag vX.Y.Z && git push origin vX.Y.Z`. The tag triggers
   `release.yml` (`on.push.tags: ['v[0-9]*']`). Nothing runs on the commit
   itself.
5. Watch Actions: `check-version` -> `build-macos` (CLI zip + app DMG, one
   build) -> `release` (GitHub Release) -> `publish-crates` (crates.io) ->
   `update-homebrew-cask` (both casks).

`check-version` fails the run if the tag, workspace version, and
changelog heading disagree, before anything is built.

## What ships

Every tagged release produces three assets, Apple Silicon only:
`turbospark-<ver>-macos-arm64.zip` (the three CLI binaries),
`TurboSpark-<ver>-arm64.dmg` (the app bundle with those same binaries
inside), and `SHA256SUMS`. Both come from **one** `build-macos` job on
purpose, so the loose CLI binaries and the ones inside the app are
guaranteed byte-identical under one version number.

Crates publish to crates.io one at a time in a fixed dependency order
(`turbospark-core` first, `turbospark-server` last, `turbospark-bench`
never), not via one `cargo publish --workspace` call. A re-run of the same
tag resumes correctly, since an already-published crate is treated as a
no-op.

## Don't

- Don't expect `publish-crates` or `update-homebrew-cask` to fail a release
  over a missing secret. `CARGO_REGISTRY_TOKEN` and `HOMEBREW_TAP_TOKEN` are
  each optional. Unset, the job no-ops with a `::notice::` and the rest of
  the release still ships.
- Don't try to reuse a bad version number. crates.io has no delete, only
  yank. Ship a fixed `X.Y.(Z+1)` through the same process instead.
- Don't assume the app is notarized. It's ad-hoc signed only, which means a
  Homebrew install gets Gatekeeper's "cannot be verified." See
  [[gatekeeper-quarantine-and-notarization]].
- Don't re-tag to fix a `check-version` failure once the tag is pushed.
  Delete it first (`git push --delete origin vX.Y.Z && git tag -d vX.Y.Z`),
  fix the problem, re-tag. Nothing downstream ran yet.

## Local dry run before tagging

```sh
cargo build --workspace && cargo test --workspace && cargo fmt --check && cargo clippy --workspace --tests
cargo publish --workspace --dry-run --allow-dirty   # needs the PREVIOUS version already live
make dmg && ls -l dist/
```

A bad tag is cheap to delete. A half-published crates.io chain is not.
