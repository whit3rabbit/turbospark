# App/Updates

## Overview
Owns the Sparkle 2 auto-updater for the whole app. One `@MainActor` singleton is created once and shared by the app-menu item and the General settings pane. The update feed is the `appcast.xml` GitHub release asset; trust comes from the EdDSA signature (`SUPublicEDKey` in the bundle Info.plist), not from code-signing identity, because the app is ad-hoc signed [SparkleUpdateController.swift:SparkleUpdateController]. The directory also holds a harness-only user driver that lets a two-version smoke test run download, verify, install and relaunch unattended.

## Interface
- `SparkleUpdateController.shared`: the singleton `ObservableObject` [SparkleUpdateController.swift:SparkleUpdateController].
- `start()`: idempotent; call sites may call it unconditionally [SparkleUpdateController.swift:start].
- `checkForUpdates()`: manual check; logs and returns if the updater is missing or `canCheckForUpdates` is false [SparkleUpdateController.swift:checkForUpdates].
- Published state: `automaticallyChecksForUpdates` (read/write, mirrors Sparkle's preference) and `canCheckForUpdates` (private set) [SparkleUpdateController.swift:automaticallyChecksForUpdates].
- Static environment helpers: `isBundledApp`, `displayVersion` (CFBundleShortVersionString or "unknown"), `feedURLOverride` (env `TURBOSPARK_UPDATE_FEED_URL`), `silentUpdateMode` (env `TURBOSPARK_UPDATE_SILENT == "1"` and a feed override) [SparkleUpdateController.swift:isBundledApp].
- `SPUUpdaterDelegate` conformance: returns the override feed, logs found, not-found, abort, install and user-choice events [SparkleUpdateController.swift:SparkleUpdateController].
- `AutoInstallUserDriver`: harness-only `SPUUserDriver` that answers every prompt with install/allow; `harnessLog` and `appendHarnessLog(_:to:)` are the log helpers [SparkleUpdateController.swift:AutoInstallUserDriver].
- Child contracts: none (no subdirectories).

## Invariants
- Sparkle starts only when `Bundle.main.bundleURL.pathExtension == "app"`. A bare `swift run` executable has no Info.plist or SUFeedURL and a different preferences domain, so `start()` logs and returns [SparkleUpdateController.swift:start].
- `start()` is a no-op once `updater` is set, so the updater is never created twice [SparkleUpdateController.swift:start].
- The class is `@MainActor` because Sparkle must be created on the main thread. The KVO callbacks hop back with `MainActor.assumeIsolated` [SparkleUpdateController.swift:attach].
- The `automaticallyChecksForUpdates` setter writes to Sparkle only when the value differs, preventing a KVO feedback loop with the observation that sets it [SparkleUpdateController.swift:automaticallyChecksForUpdates].
- `silentUpdateMode` requires both the env flag and a feed override. The scheduler is disabled only during `start()` and restored to the saved preference afterward; on a start failure it is restored before returning [SparkleUpdateController.swift:start].
- `feedURLString(for:)` returns the override, or nil to use the bundle's SUFeedURL. EdDSA still gates what installs, so an override can only serve signed updates [SparkleUpdateController.swift:feedURLString].
- `AutoInstallUserDriver` replies to Sparkle on a later main-queue turn (50 ms); replying synchronously starves Sparkle's handoff [SparkleUpdateController.swift:replyLater]. It declines to send a system profile [SparkleUpdateController.swift:show].
- The harness driver is never used unless `silentUpdateMode` is true.

## Side Effects
- Starts Sparkle (`SPUStandardUpdaterController` or `SPUUpdater`), which schedules network checks of the appcast feed, downloads, installs and may relaunch the app [SparkleUpdateController.swift:start].
- Sparkle persists its own preferences (automatic-check flag, last-check time); this code writes only through `updater.automaticallyChecksForUpdates` [SparkleUpdateController.swift:automaticallyChecksForUpdates].
- `NSLog` output for lifecycle and delegate events [SparkleUpdateController.swift:start].
- Reads process environment variables `TURBOSPARK_UPDATE_FEED_URL` and `TURBOSPARK_UPDATE_SILENT` [SparkleUpdateController.swift:feedURLOverride].
- Harness mode only: truncates then appends to `/tmp/turbospark-update-smoke.log` via `FileHandle` [SparkleUpdateController.swift:AutoInstallUserDriver].
- Installs two KVO observations on the updater [SparkleUpdateController.swift:attach].

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
