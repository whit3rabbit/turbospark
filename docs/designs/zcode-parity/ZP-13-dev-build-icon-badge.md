# ZP-13: Dev-build icon badge

Finding 1.16 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[dev-icon-badge doc](https://github.com/zai-org/ZCode/blob/main/docs/dev-icon-badge.md).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Composite a diagonal "DEV" ribbon onto the Dock and window icon at
startup when running an unpackaged development build, so a dev build
next to an installed release is distinguishable at a glance.

## The Problem

A locally built TurboSpark.app and an installed release carry the same
icon. The window title does not distinguish them either, so during
release testing it is easy to interact with the wrong one. This matters
for this repo specifically: release procedure requires launching the
built bundle for verification
(`scripts/verify-app-bundle.sh`), and confusing a dev build with the
installed app can recreate state under test.

## Goals

* When the running bundle is detected as unpackaged (debug build or not
  launched from `/Applications`-style install with a missing release
  marker), the Dock icon gains a blue diagonal DEV ribbon composited at
  startup.
* Failure to composite falls back to the original icon; the badge never
  blocks startup.
* Packaged release builds never take the badge path (no compositing, no
  detection cost beyond the check).

## Non-Goals

* No badge on settings-window icons or document icons; Dock plus the
  app's own window representation is the surface.
* No red or environment-specific variants in v1.

## Open Questions

* **Packaged detection rule**: options: (a) debug configuration check
  (`#if DEBUG` build flag); (b) bundle marker file presence (release
  packaging writes a marker during `make app-bundle`); (c) install
  location check. ZCode checks "running unpackaged". Lean (a) plus (b):
  debug builds always badge; release builds badge only if the packaging
  marker is absent, catching a manually-run non-installed release build.

## Critical User Journeys

### CUJ 1: dev build is distinguishable

* **Given**: a debug build launched from Xcode or `swift run`.
* **When**: the app finishes launching.
* **Then**: the Dock icon shows the DEV ribbon; the installed release in
  the same Dock shows none.

### CUJ 2: failure fallback

* **Given**: icon compositing fails (icon resource missing).
* **When**: the badge path runs.
* **Then**: the original icon is kept, a single log line records the
  failure, and startup proceeds unaffected.

## Technical Plan (Architecture)

```text
App launch (TurboSparkApp init)
  DevBadgePolicy: isUnpackagedBuild? (Open Question rule)
       | yes
       v
DevIconBadger
  load app icon (bundle icon or Assets)
  composite diagonal ribbon (Core Graphics, off main thread)
  NSApp.applicationIconImage = composited
  Dock icon updates automatically
  failure: log + keep original
```

The ribbon is drawn in Core Graphics: diagonal band, rotated label, the
design system's accent blue for the band, white bold label, at icon
resolution with safe margins.

## Alternatives Considered

### Alternative A: suffix the process name instead

* **Pros**: trivial.
* **Cons**: does not help Dock recognition, which is the actual failure
  mode; names get truncated.
* **Reason for Rejection**: Rejected by author: the Dock icon is the
  surface people actually look at.

### Alternative B: separate dev asset catalog icon

* **Pros**: no runtime compositing.
* **Cons**: requires maintaining a second icon asset and swapping
  catalogs per build config; runtime compositing keeps one source icon.
* **Reason for Rejection**: Rejected by author: one icon source of
  truth.

## Non-Functional Requirements

* **Performance**: compositing once at launch, off the main thread,
  before first window display when possible; sub-100 ms budget.
* **Safety**: the badge path must never throw into startup; wrap in
  full error handling with fallback to the original icon.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/App/DevIconBadger.swift` (Create)
   * **Type / Scope**: detection and compositing.
   * **Proposed Changes**: unpackaged-build detection per the policy,
     Core Graphics ribbon compositing, `NSApp.applicationIconImage`
     assignment, failure fallback and logging.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Type / Scope**: launch hook.
   * **Proposed Changes**: call the badger during app init, after basic
     state but before window display.
3. `Makefile` (Modify)
   * **Type / Scope**: packaging.
   * **Proposed Changes**: `app-bundle` target writes the release marker
     into the bundle if the Open Question lands on marker-based
     detection.
4. `swift/TurboSparkApp/Tests/TurboSparkAppTests/DevIconBadgerTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: policy decision per build configuration and
     marker presence, composited image dimensions match the source,
     ribbon pixels present (corner sampling), fallback keeps the
     original icon on injected failure.
   * Mutation checks: invert the policy and assert the decision test
     fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the badger tests.
* **Manual**: launch the dev build (badged) beside the installed release
  (not badged); run `make app-bundle` and confirm the packaged build is
  not badged.

--------------------------------------------------------------------------------

*Drafted with EGM*
