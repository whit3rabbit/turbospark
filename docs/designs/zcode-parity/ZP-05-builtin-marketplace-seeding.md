# ZP-05: Builtin marketplace seeding and CDN zip source

Finding 4.2 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[bundled-plugins bootstrap](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/bootstrap/src/app/bundled-plugins.ts)
and the
[plugin CDN zip source doc](https://github.com/zai-org/ZCode/blob/main/docs/plugin-cdn-zip-source.md).

* Author: whit3rabbit
* Status: Implemented (seeding half; `cdnZip` deferred)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Give the app an official marketplace identity: builtin plugins seeded from
the app bundle at startup, and a CDN zip source that installs sha256
verified archives. ZCode merges both into one official marketplace row;
the Swift side gains the same capability without any server of its own
(seeding is bundle-local; the CDN is a static host of the publisher's
choosing).

## The Problem

The Swift app has no builtin plugin registry. `MarketplaceSource` is
url/github/git/directory only. The docs list builtin registry and seed
dirs as out of scope. Consequences: bundled content plugins cannot exist
(the fleet gap in ZP-20), installs always come from user-added
marketplaces, and uninstalling a builtin cannot stick because restart
would re-seed it unless suppression persists.

## Goals

* A builtin marketplace row exists out of the box, named
  `turbospark-plugins-builtin`.
* Builtin plugins are seeded from bundle resources at startup under a
  seed lock, with a seed marker per plugin.
* Uninstalling a builtin leaves a persistent suppression record so
  restart does not re-seed it; Install restores it by clearing the
  record.
* A new source kind `cdnZip` downloads a zip from a URL, verifies
  sha256 against a declared digest, stages, and installs through the
  existing atomic installer.
* Digest mismatch refuses the install with a named error; no partial
  install.

## Non-Goals

* No first-party CDN or hosted catalog. TurboSpark publishes no service.
* No plugin auto-update (still out of scope per swift/docs/SWIFT_PLUGINS.md;
  update remains manual reinstall, see finding 4.9 which is Partial).
* No new manifest formats.

## Open Questions

* **What ships as builtins on day one**: the ZCode fleet has no
  equivalent here until ZP-18/ZP-20 land. Day-one candidates: ZP-20's
  guide skills pack and ZP-18's device-control plugins once written.
  Decide at implementation time; the mechanism must not depend on the
  answer.
* **CDN zip manifest shape**: a single JSON listing url + sha256 per
  plugin, or one zip per plugin with the digest in the marketplace row.
  Lean one-zip-per-plugin with digest in the row, matching the existing
  per-plugin install ledger.

## Critical User Journeys

### CUJ 1: seed, uninstall, restore

* **Given**: a bundle ships one builtin plugin (guide skills).
* **When**: the app launches, seeds it under the seed lock, and the user
  later uninstalls it from the plugins pane.
* **Then**: restart does not re-seed it (suppression record); choosing
  Install from the builtin row clears the suppression and reinstalls
  from bundle resources.

### CUJ 2: CDN zip install

* **Given**: a marketplace row declares a `cdnZip` source with a sha256
  digest.
* **When**: the user installs the plugin.
* **Then**: the zip downloads, digest verifies, staging uses the
  existing stage-move-rollback installer, and the ledger records the
  source kind.

## Technical Plan (Architecture)

```text
PluginManager startup
  |
  +--> BuiltinPluginSeeder
         bundle resource dir (Plugins/Builtin/<id>/...)
         seed lock (per-profile file lock)
         per-plugin seed marker in the ledger
         suppression records in settings.json
  |
  +--> MarketplaceSource.cdnZip(url, sha256)
         downloader (reuse ranged-download client patterns
         from crates/repack ranged_download for bounds safety)
         digest check -> stage -> atomic move -> ledger
```

Seeding copies bundle resources into the per-profile plugin root, then
records a seed marker. Suppression records are settings-keyed by plugin
id. The `cdnZip` source reuses the existing atomic
stage-move-rollback installer; only download and verification are new.

## Alternatives Considered

### Alternative A: seed by symlink to bundle resources

* **Pros**: no copy cost.
* **Cons**: plugin code that writes next to itself breaks; uninstall
  semantics get muddy (delete a bundle path? no).
* **Reason for Rejection**: Rejected by author: copy keeps uninstall and
  suppression semantics clean.

### Alternative B: git source with a bundled mirror

* **Pros**: reuses the existing git source end to end.
* **Cons**: shipping a git checkout in the bundle is heavy and brittle
  across updates.
* **Reason for Rejection**: Rejected by author: bundle zip resources are
  lighter and digest-verifiable.

## Non-Functional Requirements

* **Security**: CDN installs require the declared sha256; mismatch is a
  named error with no partial state. Seeding never runs code; it only
  stages files that pass through the existing trust gates (plugin hooks
  still need SHA-256 trust approval per the existing plugin trust model).
* **Storage**: seeded copies live in the per-profile plugin root, so
  profiles stay isolated.
* **Rollout**: the builtin row always exists; with no builtins declared
  it renders as an empty official segment in the marketplace sheet.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/BuiltinPluginSeeder.swift` (Create)
   * **Type / Scope**: seeding.
   * **Proposed Changes**: bundle resource enumeration, seed lock, seed
     markers, suppression records, install/restore paths.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/MarketplaceSource.swift` (Modify)
   * **Type / Scope**: source model.
   * **Proposed cases**: add `builtin` and `cdnZip(url, sha256)` kinds
     with digest verification data.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/State/PluginMarketplaceManager.swift` (Modify)
   * **Type / Scope**: manager.
   * **Proposed Changes**: builtin row injection, cdnZip download and
     digest check before staging, ledger source-kind recording.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Components/PluginMarketplaceSheet.swift` (Modify)
   * **Type / Scope**: UI.
   * **Proposed Changes**: official segment rendering for the builtin
     row; digest-mismatch error surfacing.
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/BuiltinPluginSeederTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: seed idempotence, suppression prevents
     re-seed, restore clears suppression, cdnZip digest match and
     mismatch paths, partial-download refusal.
   * Mutation checks: skip the digest check and assert the mismatch test
     fails; drop the seed lock and assert concurrent-seed test fails.
6. `swift/docs/SWIFT_PLUGINS.md` (Modify)
   * **Type / Scope**: docs.
   * **Proposed Changes**: remove builtin registry/seed dirs from the
     out-of-scope list; document the builtin row, seeding, suppression,
     and the cdnZip source.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for seeder tests plus existing plugin
  tests.
* **Manual**: seed, uninstall, restart (no re-seed), restore; install
  from a cdnZip row against a local static server with a good and a bad
  digest.

--------------------------------------------------------------------------------

## Implementation notes (2026-11-14)

Shipped: `State/BuiltinPluginSeeder.swift`, `BuiltinPluginSeederTests`, and the
`SWIFT_PLUGINS.md` section. Applied EGM review findings: B1 (copy semantics
stated technically: the bundle is a signed read-only `.app`, so symlinks and
in-place writes are wrong), M3 (seeding is a `directory`-source install through
the existing installer, not a second path), M4 (one state record per builtin
reconciled with the ledger every launch; flock seed lock, no stale-lock case;
version bump reinstalls; corrupt state fails closed), M6 (hooks stay untrusted
until approved; seeding copies files and never runs code), M7 (the builtin row is
app-owned and never persisted, hidden when the bundle is empty), M8 (failure
modes in the SWIFT_PLUGINS.md section and tests).

Deviations from the draft:

* **`cdnZip` and `MarketplaceSource.builtin`/`cdnZip` cases are NOT built**
  (review B2, M1, M2, M5; INDEX says ZP-05 is a later re-delivery path). The
  app has no zip extractor, so it needs either a new dependency or an `unzip`
  subprocess, both of which need an explicit decision. A digest in the same
  row as the URL only catches corruption, so a trust anchor (signed catalog or
  pinned digests shipped in the bundle) must be chosen too. No consumer exists
  (TurboSpark hosts no CDN). Constraints for a future delivery: HTTPS only,
  redirects off or same-host, streamed download with a byte cap and a
  streaming sha256, required lowercase 64-hex digest (malformed refuses),
  extraction refusing `..`, absolute paths, symlinks and over-cap archives,
  all staged into a temp dir before the existing atomic installer.
* **No new source enum case at all.** The builtin row is injected as a
  non-persisted `.directory` entry in `AppModel.marketplaceSources`, avoiding
  Codable downgrade risk.
* **No builtin ships day one** (ZP-07, ZP-18 and ZP-20 shipped as bundled
  skills instead). `Package.swift` is therefore untouched; the first builtin
  adds `.copy("BuiltinPlugins")` and runs `make swift-app-build`.
* **No UI change**: no official-segment styling and no digest error surface
  (no digest yet). The row appears as an ordinary marketplace once a builtin
  exists. No new localized strings.
* **Integrity**: the bundle code signature is the anchor; the seeder adds
  symlink, escape, and size-cap refusals (2,000 files, 64 MiB per plugin).
* **Mutation checks run**: bypassing the lock, the suppression check, and the
  symlink check each fail their tests.

--------------------------------------------------------------------------------

*Drafted with EGM*
