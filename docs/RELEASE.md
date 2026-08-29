# Release process

How a version of turbospark gets from a commit on `main` to a GitHub
Release, a set of crates.io publishes, and two updated Homebrew casks. Read
this before cutting a release; `.github/workflows/release.yml` is the
executable version of this document and should never drift from it.

## The three artifacts

Every tagged release produces exactly these, all Apple Silicon:

| Asset | Contains | Installed by |
|---|---|---|
| `turbospark-<ver>-macos-arm64.zip` | `turbospark-check`, `turbospark-model`, `turbospark-server` | `brew install --cask whit3rabbit/tap/turbospark-cli`, or by hand |
| `TurboSpark-<ver>-arm64.dmg` | `TurboSpark.app`, with those same three binaries inside it | `brew install --cask whit3rabbit/tap/turbospark` |
| `SHA256SUMS` | one line per asset above | nothing; it is for humans |

**Both artifacts come out of ONE `build-macos` job**, and that is load
bearing rather than a tidiness point: the CLI binaries ship twice, loose in
the zip and inside the app bundle, and two jobs building them separately
could ship two different builds under one version number. In one job they
are the same files, copied twice.

## What ships to crates.io

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
   `build-macos` (the CLI zip and the app DMG, both from one build) ->
   `release` (GitHub Release, with the changelog section for this version
   as the body and `generate_release_notes: true` appending the
   auto-generated PR/commit list below it) -> `publish-crates` (crates.io,
   skipped with a notice if `CARGO_REGISTRY_TOKEN` isn't set) ->
   `update-homebrew-cask` (both casks, skipped with a notice if
   `HOMEBREW_TAP_TOKEN` isn't set).

The changelog step is first for a reason: `check-version` greps
`CHANGELOG.md` for `^## \[X.Y.Z\]` and fails the whole run without it, and
the `release` job then `awk`s that section out as the Release body. So the
changelog is not documentation of the release, it IS the release notes, and
a tag pushed without it never reaches the build.

## The app bundle and the DMG

`swift build` emits a bare executable and a resource bundle side by side;
there is no Xcode project here (`swift/CLAUDE.md` Gotcha 12). Two scripts
close that gap, and CI calls exactly these, so a local `make dmg` and a
release build the same thing:

```sh
make app-bundle   # scripts/make-app-bundle.sh -> dist/TurboSpark.app
make dmg          # scripts/make-dmg.sh        -> dist/TurboSpark-<ver>-arm64.dmg
```

`make-app-bundle.sh` builds all three halves (the FFI staticlib via
`scripts/swift-lib.sh`, the CLI and server via cargo, the app via SwiftPM),
then assembles:

```
TurboSpark.app/Contents/
+-- Info.plist                 # written by the script; the ONLY copy in the tree
+-- PkgInfo
+-- MacOS/
|   +-- TurboSparkApp          # CFBundleExecutable
|   +-- turbospark-check       # the CLI, where the cask links it from
|   +-- turbospark-model
|   \-- turbospark-server
\-- Resources/
    \-- TurboSparkApp_TurboSparkApp.bundle   # Bundle.module's resources
```

Three facts about that layout that are decisions rather than defaults:

- **Every bundle-identity fact is invented in that script**, because no
  Info.plist existed before it. `CFBundleIdentifier` is
  `com.whit3rabbit.turbospark` and `LSMinimumSystemVersion` is `14.0`,
  which MUST match `platforms: [.macOS(.v14)]` in
  `TurboSparkApp/Package.swift` and `depends_on macos: ">= :sonoma"` in
  both casks. Three files, one number.
- **The identifier moves the user's preferences domain.** A bare
  `swift run` build writes `~/Library/Preferences/TurboSparkApp.plist`; the
  bundle writes `com.whit3rabbit.turbospark.plist` (verified by launching
  it, not inferred). So `@AppStorage` settings -- appearance, text size,
  language -- do not carry across from a development run to an installed
  app. The JSON stores under
  `~/Library/Application Support/TurboSpark/` are unaffected: those paths
  are hardcoded, not bundle-derived.
- **The resource bundles are copied by glob, not by name.** A dependency
  that grows resources emits a second `.bundle` next to the executable, and
  copying only ours would drop it silently. The script fails if the glob
  matches nothing.

`make-dmg.sh` wraps the bundle with an `/Applications` symlink, then
**mounts the image it just built and checks it**: main executable present,
all three CLI binaries present, a resource `.bundle` in `Contents/Resources`,
and `codesign --verify --strict` on the mounted copy. That check is the
reason the script exists rather than a one-line `hdiutil` call in the
workflow -- `hdiutil create` exits 0 over a staging directory missing the
resource bundle, and the symptom is an app that launches with no provider
logos and no built-in prompts, on a user's machine, weeks later.

`.github/workflows/ci.yml`'s `package-macos` job runs both scripts on every
push to `main` (not on PRs, where it would add minutes to a Rust-only
change). Without it, a tag push would be the first thing to exercise the
packaging path since the last release.

## Gatekeeper: the app is signed but NOT notarized

`make-app-bundle.sh` signs ad-hoc (`codesign --sign -`), which is the
minimum an arm64 binary needs to execute at all. It is not a Developer ID
signature and there is no notarization step, because this repo holds no
signing identity.

The consequence is concrete and worth stating rather than discovering:
Homebrew applies the quarantine attribute to cask downloads by default, so
`brew install --cask turbospark` puts an app in `/Applications` that
Gatekeeper refuses to open with "cannot be verified". The three ways
around it, in the order to suggest them:

```sh
brew install --cask --no-quarantine whit3rabbit/tap/turbospark
xattr -dr com.apple.quarantine /Applications/TurboSpark.app
# or: right-click the app in Finder and choose Open
```

To fix it properly, in the order the steps have to happen: get a Developer
ID Application certificate, put it in the CI keychain, set
`CODESIGN_IDENTITY` (the script already reads it and defaults to `-`), then
add a `xcrun notarytool submit --wait` plus `xcrun stapler staple` pair
after `make-dmg.sh`. Only the last two are new work; the signing seam is
already there.

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

## Homebrew tap: two casks

`update-homebrew-cask` is the last job in `release.yml`. It runs after
`release` (it downloads the published assets, so they have to exist) and is
gated on the `HOMEBREW_TAP_TOKEN` repo secret exactly like `publish-crates`
is gated on `CARGO_REGISTRY_TOKEN`. Set, it pushes; unset, it no-ops with
`::notice::` and the rest of the release still ships. It writes BOTH casks
into `whit3rabbit/homebrew-tap` in one commit:

| Cask | Asset | Installs |
|---|---|---|
| `turbospark` | the DMG | `/Applications/TurboSpark.app` plus the three CLI commands on `PATH` |
| `turbospark-cli` | the zip | the three CLI commands, nothing else |

```sh
brew install --cask whit3rabbit/tap/turbospark        # app + CLI
brew install --cask whit3rabbit/tap/turbospark-cli    # CLI only
brew uninstall --cask turbospark                      # removes both the app and the commands
brew uninstall --zap --cask turbospark                # ...and the app's settings and chat archive
```

**Why two casks and not one with a flag: Homebrew has no such flag.** A
cask carries exactly one `url`, and "with the app" and "without it" are
different downloads. The alternative considered and rejected was shipping
the app in the zip too and letting the user drag it, which gives up the
`/Applications` install and the uninstall that this whole exercise is for.

**The full cask does not ship a second copy of the CLI.** Its `binary`
stanzas point INTO the installed bundle
(`#{appdir}/TurboSpark.app/Contents/MacOS/turbospark-check`, and so on),
which is the standard cask pattern for an app that carries commands. Two
consequences: the app and the commands are guaranteed to be the same build,
and `brew uninstall` removes the app and unlinks all three in one step,
because the symlinks are part of the same cask.

**`conflicts_with` is declared on both casks, not one.** Homebrew does not
infer the reverse edge, and without both the second install would try to
link three command names the first already owns.

**`zap` deliberately does not list `~/.turbospark`.** That is where
multi-gigabyte model installs live, and a zap is not the place to silently
delete them. It trashes the app's own state: the JSON stores under
`~/Library/Application Support/TurboSpark/`, the preferences plist, and the
saved application state. Note `zap` only runs under
`brew uninstall --zap`; a plain uninstall leaves all of it.

This is Apple-Silicon-only by construction: `build-macos` builds only
`aarch64-apple-darwin`, and both casks hardcode `depends_on arch: :arm64`.
There's no Intel Mac path today; if one is ever added it needs a second
`build-*` job producing an `x86_64` zip and a second `sha256`/`url` stanza
in each cask (Homebrew casks support per-arch `sha256`/`url` blocks), not
more casks.

Each generated cask is checked for Ruby syntax validity (`ruby -c`) before
commit, but not against Homebrew's own `brew audit` / `brew style`
conventions; those need `brew` installed, which this job doesn't do. That
was a fair trade while the cask was five static lines; the app cask is
bigger now, and `brew audit --cask` on a macOS runner is the obvious next
hardening step.

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

# The packaging half, same scripts CI runs. Minutes, and it takes the
# version from the root Cargo.toml, so run it AFTER the version bump and
# the DMG name matches what the cask will point at.
make dmg && ls -l dist/
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
