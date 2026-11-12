# ZP-05: Builtin marketplace seeding and CDN zip source

Finding 4.2 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity targets: ZCode's
[bundled-plugins bootstrap](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/bootstrap/src/app/bundled-plugins.ts)
and the
[plugin CDN zip source doc](https://github.com/zai-org/ZCode/blob/main/docs/plugin-cdn-zip-source.md).

* Author: whit3rabbit
* Status: Draft
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

*Drafted with EGM*
