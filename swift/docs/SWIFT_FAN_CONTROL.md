# Swift fan control (ThermalForge)

Added 2026-09-06 (`State/FanController.swift`). The status bar's fan
control talks to the `thermalforge` CLI, which forwards to a root
LaunchDaemon over `/tmp/thermalforge.sock`. This page is the home for why
the quit-restore is load-bearing rather than a courtesy.

Read this before touching `FanController.swift` or `keepFansPinnedOnQuit`.

## A hold outlives this app

The watchdog inside ThermalForge covers only its OWN menu bar app, so once
this process pins the fans, nothing restores them but an explicit
`thermalforge auto` -- quit without the restore and the machine sits at
full RPM indefinitely. `restoreOnQuitIfNeeded` runs from the app delegate's
`applicationWillTerminate`, not a view observer (the same reasoning as
Gotcha 25 in `swift/docs/SWIFT_MODEL_HUB.md`: a quit-time action must not
depend on a view still being alive to observe it). `keepFansPinnedOnQuit`
defaults to FALSE, i.e. restore.

## Three facts the implementation rests on

`status` reads the SMC directly and needs NO daemon, so the RPM readout
works even when control cannot. Pinned state is OBSERVABLE, not
app-tracked: `mode` reads "manual" against "auto" in the status JSON, so a
hold set from a terminal shows up here, and an unpin from a terminal shows
up too. And the daemon refuses connections for a short window after a
restart (observed: "Failed to connect to daemon socket", clearing within a
minute), which is why every control command retries once.

## PATH resolution on an app launched from Finder

An app launched from Finder or the Dock inherits a minimal PATH
(`/usr/bin:/bin`) that never contains a Homebrew prefix, so resolving the
binary by PATH alone hides the feature from exactly the users who have it
installed. `locateExecutable` falls back to the known install locations
(`/usr/local/bin`, `/opt/homebrew/bin`) for that reason.
`scripts/power.sh`'s bare `thermalforge` keeps working only because a
shell session has the full PATH.

## Test-host isolation

`FanController.shared` uses an empty executable path and no poll interval
under XCTest, detected through `AppStorageRoot.isRunningTests`. AppModel
settings initialization can therefore touch the singleton without locating
or polling real hardware. Normal app launches retain PATH discovery and
polling. Explicit controller instances still accept fixture executables
and intervals, so action and timer tests exercise the same implementation.
