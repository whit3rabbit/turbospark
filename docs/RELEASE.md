# Release process

How a version of TurboSpark gets from a commit on `main` to a GitHub
Release, updated Homebrew casks, and optional on-demand crates.io publishes.
Read this before cutting a release. `.github/workflows/release.yml` and
`.github/workflows/publish-crates.yml` are the executable versions of this
document and should never drift from it.

## Architecture: App Releases vs. Crate Publishes

TurboSpark separates user-facing application releases from Rust crate publishes
on purpose:

1. **Desktop App & CLI releases** (`.github/workflows/release.yml`):
   - Triggered by pushing a version tag (`vX.Y.Z`).
   - Builds `TurboSpark.app`, the arm64 DMG, and the standalone CLI zip.
   - Creates a GitHub Release with release notes extracted from `CHANGELOG.md`.
   - Generates Sparkle signatures for automatic in-app updates.
   - Updates the Homebrew casks in `whit3rabbit/homebrew-tap`.
2. **Rust Crate Publishes** (`.github/workflows/publish-crates.yml`):
   - Triggered **on demand** via GitHub Actions `workflow_dispatch`.
   - Allows publishing all crates in topological dependency order, or publishing
     a specific crate individually.
   - Supports a `dry_run` mode to test packaging without uploading to crates.io.

### Why this is decoupled

- **Release Frequency**: The macOS app iterates quickly with UI improvements,
  dialog refinements, localization updates, and bugfixes. Bumping and publishing
  16 Rust crates to crates.io for a Swift-only change creates excessive version
  churn on the registry.
- **Permanent Registry State**: Crates cannot be deleted from crates.io once
  published (they can only be yanked). Publishing on demand preserves clean,
  intentional crate version histories.
- **Failure Isolation**: An intermittent crates.io timeout or rate limit never
  stalls or breaks a user-facing app release or Homebrew update.

## The three release artifacts

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

## Cutting an App Release

Follow these steps in order when releasing a new version of the app and CLI:

1. **Update `CHANGELOG.md`.**
   Move everything under `## [Unreleased]` to a new `## [X.Y.Z] - YYYY-MM-DD`
   section, dated the day you are releasing. Leave a fresh empty
   `## [Unreleased]` section above it for the next round of changes.
   CI enforces that this section exists and matches the tag (`check-version` job);
   a release cannot ship without it because CI extracts this section directly
   as the GitHub Release body.
2. **Bump the version.**
   Edit `workspace.package.version` in the root `Cargo.toml` to `X.Y.Z`,
   matching the changelog heading. Then bump every internal path dependency's
   `version = "..."` field in every crate's `Cargo.toml` to the same `X.Y.Z`;
   `check-version` asserts this too (`grep -n 'path = "\.\./' crates/*/Cargo.toml`).
   Run `cargo build --workspace` to ensure all workspace references agree.
3. **Commit and merge to `main`**, e.g. `chore(release): vX.Y.Z`.
4. **Tag it**: `git tag vX.Y.Z && git push origin vX.Y.Z`. The tag is what
   triggers `release.yml` (`on.push.tags: ['v[0-9]*']`); nothing runs on
   the commit itself.
5. **Watch the Actions run.** In order:
   - `check-version`: verifies the tag matches `Cargo.toml` and `CHANGELOG.md`.
   - `build-macos`: compiles the app, CLI binaries, DMG, and updates `appcast.xml`.
   - `release`: creates the GitHub Release with DMG, zip, and release notes.
   - `update-homebrew-cask`: updates both casks in `whit3rabbit/homebrew-tap`.

## Publishing Crates to crates.io

Rust crates are published on demand via `.github/workflows/publish-crates.yml`.

### Publish Order

Publish order matters: `cargo publish` refuses a crate until every internal
path dependency it declares (`path = "../x", version = "..."`) is already
live on the registry at that exact version.

The topological dependency order across the 16 publishable crates is:

1. `turbospark-core`: no internal deps
2. `turbospark-window-fit`: no internal deps
3. `turbospark-vision-io`: needs core
4. `turbospark-compute`: needs core
5. `turbospark-invocation`: needs core
6. `turbospark-model-io`: needs core
7. `turbospark-selection`: needs core
8. `turbospark-tokenizer`: needs core
9. `turbospark-repack`: needs core, compute, model-io
10. `turbospark-catalog`: needs core, model-io, repack, tokenizer
11. `turbospark-streaming`: needs core, model-io
12. `turbospark-gpu`: needs core; macOS target-deps on model-io
13. `turbospark-image`: needs core, compute, model-io, tokenizer; macOS target-deps on gpu
14. `turbospark-runtime`: needs core, selection, tokenizer; macOS target-deps on gpu, model-io, compute, streaming, vision-io
15. `turbospark-cli`: needs invocation, runtime, selection, tokenizer, repack, model-io, window-fit, vision-io, image
16. `turbospark-server`: needs core, runtime, selection, tokenizer, vision-io; macOS target-deps on repack

`turbospark-bench` (`publish = false`) is an internal dev harness and is never
published. `turbospark-ffi` (`publish = false`) is the C ABI static library
for the native macOS app and is linked by SwiftPM rather than published to
crates.io.

### How to trigger crate publishing

In GitHub Actions:
1. Navigate to **Actions** -> **Publish Crates**.
2. Click **Run workflow**.
3. Choose whether to run in `dry_run` mode (recommended first).
4. Select `all` to publish all crates in the topological order above, or select
   a specific crate if only that crate had changes.
5. Click **Run workflow**.

If a new crate is added to the workspace, or an existing crate grows a new
internal dependency, update the list above and the `ALL_CRATES` array in
`.github/workflows/publish-crates.yml`. Re-derive the order with:

```sh
cargo publish --workspace --dry-run --allow-dirty
```

## The app bundle and the DMG

`swift build` emits a bare executable and a resource bundle side by side;
there is no Xcode project here (`swift/AGENTS.md` Gotcha 12). Two scripts
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
and `codesign --verify --deep --strict` on the mounted copy. That check is the
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

## In-app updates: Sparkle, fed from GitHub Releases

The app updates itself through Sparkle 2 (`SUPublicEDKey` + `SUFeedURL` in
the bundle's Info.plist, both written by `make-app-bundle.sh`). The update
archive IS the release DMG -- Sparkle 2 mounts DMGs natively -- and the feed
is the `appcast.xml` asset each release attaches, reached through the
evergreen `https://github.com/whit3rabbit/turbospark/releases/latest/download/appcast.xml`.
No feed host beyond GitHub Releases; `release.yml`'s `build-macos` job
generates and uploads it via `scripts/make-sparkle-appcast.sh`.
The runtime detection path and UI contract are documented in
[`swift/docs/AUTO_UPDATE.md`](../swift/docs/AUTO_UPDATE.md).

Trust is EdDSA, not Apple. The app is ad-hoc signed, so there is no Team ID
for Sparkle's Apple-identity comparison; what makes an update verifiable is
the `sparkle:edSignature` over the DMG, checked against the committed public
key. This is why a Developer ID is not a prerequisite for the updater. The
DMG a brew user installs and the DMG Sparkle installs are the same artifact,
and the Gatekeeper quarantine caveat above applies to both equally -- it is
unchanged by the updater.

One-time bootstrap (done 2026-09-20):

```sh
# generate_keys lives in the Sparkle tools tarball the script downloads
./bin/generate_keys            # created the key in the login keychain, printed SUPublicEDKey
./bin/generate_keys -x ~/path/to/sparkle-ed25519.key   # export a copy
gh secret set SPARKLE_ED_PRIVATE_KEY < ~/path/to/sparkle-ed25519.key
```

The public half is the literal in `make-app-bundle.sh` (public on purpose:
local bundles and ci.yml must never need the secret). The private half lives
in the login keychain and as the `SPARKLE_ED_PRIVATE_KEY` repo secret. Lose
both and updates stop being signable -- Sparkle's key rotation
documentation covers recovery, which requires a Developer ID DMG, so treat
the key as release-critical.

The secret is required. `build-macos` fails before publishing when it is
missing. The feed URL resolves through the latest release, so publishing a
release without `appcast.xml` would turn update checks into a 404 for every
installed app.

For Homebrew installs, the `turbospark` cask declares `auto_updates true`:
the app replaces its own bundle, and brew defers to it instead of offering
an `upgrade` that would fight the updater over the same `/Applications`
path. `turbospark-cli` keeps no updater -- `brew upgrade turbospark-cli`
remains its only update path (the CLIs inside the app bundle update together
with the app, so a cask `upgrade` after an in-app update would at worst
relink three binaries from the newer bundle).

Local two-version smoke (the real gate; proves an ad-hoc app survives a
Sparkle install + relaunch on current macOS). The app binary is identical
across the two versions -- only the Info.plist version differs -- so the
"update" is real Sparkle work with throwaway version numbers:

```sh
KEY=~/.turbospark/keys/sparkle-ed25519-private.key   # keychain export (bootstrap above)
TOOLS=/tmp/turbospark-sparkle-tools                  # Sparkle tarball extracted

# 1. The "new" release: DMG + its localhost-signed feed.
scripts/make-app-bundle.sh --version 9.9.2
scripts/make-dmg.sh --version 9.9.2
mkdir -p /tmp/update-smoke && cp dist/TurboSpark-9.9.2-arm64.dmg /tmp/update-smoke/
SPARKLE_ED_KEY_FILE="$KEY" SPARKLE_TOOLS_DIR="$TOOLS" \
  scripts/make-sparkle-appcast.sh --version 9.9.2 \
  --dist /tmp/update-smoke --download-url-prefix http://127.0.0.1:8765/
(cd /tmp/update-smoke && python3 -m http.server 8765) &

# 2. The "old" app, stamped down, launched with the feed override and the
# unattended smoke driver. The driver forces a check even if SULastCheckTime
# says the normal interval is not due.
scripts/make-app-bundle.sh --version 9.9.1 --skip-build
TURBOSPARK_UPDATE_SILENT=1 \
TURBOSPARK_UPDATE_FEED_URL=http://127.0.0.1:8765/appcast.xml \
  dist/TurboSpark.app/Contents/MacOS/TurboSparkApp &

# 3. Assert the handover and inspect the complete funnel:
test "$(plutil -extract CFBundleShortVersionString raw \
  dist/TurboSpark.app/Contents/Info.plist)" = "9.9.2"
sed -n '1,200p' /tmp/turbospark-update-smoke.log
# The log must end with installed and relaunched=true.
```

(`SPARKLE_TOOLS_DIR` avoids the tarball re-download; loopback is exempt from
ATS, so plain http on 127.0.0.1 needs no Info.plist exception.) The
ad-hoc-signing edge this smoke exists for: Sparkle verifies the new bundle's
code signature and there is no stable identity to match, so if current macOS
refuses the handover it happens HERE, not on a user's machine.

## Prerequisites

- **Local cargo >= 1.90** if you ever want to run `cargo publish
  --workspace` by hand outside CI. Multi-package publishing stabilized in
  Cargo 1.90 (2025-09-18). `rust-toolchain.toml` here pins `channel =
  "stable"`, so a `rustup update` keeps you current; the workspace's
  declared `rust-version = "1.82"` is the MSRV floor for *consumers*
  compiling this code, not the cargo binary you invoke `publish` with.
- **`CARGO_REGISTRY_TOKEN`** repo secret: a crates.io API token
  (`cargo login` locally, or generate one at
  https://crates.io/settings/tokens). Required for running the live
  `publish-crates.yml` workflow (can still run with `dry_run: true` without it).
- **`HOMEBREW_TAP_TOKEN`** repo secret: a PAT with push access to
  `whit3rabbit/homebrew-tap`, only needed for the cask update. Same
  no-op-with-notice behavior when unset.
- **`SPARKLE_ED_PRIVATE_KEY`** repo secret: the Sparkle EdDSA private key
  (see the in-app updates section). Without it `build-macos` fails before a
  GitHub Release can replace the last valid `appcast.xml`.
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

**`auto_updates true` is on the app cask only.** The app self-updates
through Sparkle (previous section), and the stanza is what keeps `brew
outdated`/`brew upgrade` from fighting that updater over the same
`/Applications` path. The CLI cask gets no updater and no stanza: `brew
upgrade turbospark-cli` is its update path, by design.

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
- **Crate publish partially failed** (say, crate 6 of 16 failed for a
  transient registry reason): fix the cause, then re-run the
  `publish-crates.yml` workflow dispatch. The per-crate loop treats every
  already-published crate as a no-op and continues, so a re-run correctly
  resumes rather than re-attempting earlier crates. You can also re-run
  for just the specific failed crate using the workflow's input selector.
- **A bad version got published for real**: crates.io has no delete, only
  yank (`cargo yank -p <crate> --vers X.Y.Z`), which prevents new
  dependents from resolving to it without removing it for existing lock
  files. Ship a fixed `X.Y.(Z+1)` through this same process; don't try to
  reuse the yanked version number.
