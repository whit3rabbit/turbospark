# OC-07: First-run onboarding and startup splash

Findings: Section 7, "First-run onboarding - Missing in Swift" and
"Animated splash and startup screens - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[onboarding](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/renderer/onboarding/index.tsx)
and
[splash](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/renderer/startup/splash.tsx).

* Author: whit3rabbit
* Status: Implemented (2026-11-14)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

A branded splash shown while app state loads, and a sequenced first-run
flow for genuinely new installs: welcome, pick or install a first
model, optional import hand-off, and a permissions primer that asks for
nothing at launch. OpenCode gates its onboarding on a first-launch
condition and uses the splash as the hand-off boundary; the TurboSpark
adaptation keeps both inside the main window and defers to the
[desktop-shell-hardening](../../../.kiro/specs/desktop-shell-hardening/brief.md)
import wizard for any migration step.

## The Problem

There is no persisted first-launch flag, tour, or checklist. The first
run surface is the static welcome hero
([WelcomeCharacterView.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Components/WelcomeCharacterView.swift))
plus install gating
([ModelInstallGate.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelInstallGate.swift)).
The main window appears directly with no branded loading state, and a
new user gets no sequenced path from launch to first prompt.

## Goals

* **Splash**: an in-window overlay with the wordmark and a quiet pulse
  while app state loads. It dismisses when the main surface is ready.
  It honors Reduce Motion (no pulse, instant dismiss) and uses theme
  colors only.
* **First-run detection**: a persisted completion flag, checked before
  the hero renders; a skipped or completed flow never replays.
* **Flow steps** (each skippable, progress visible, back allowed):
  1. Welcome: the existing hero plus one sentence of orientation.
  2. First model: routed to the existing install hub entry when no
     installed model fits; skipped when one is already installed.
  3. Import hand-off: shown only when the desktop-shell-hardening
     import wizard exists; until then this step is absent, not a
     placeholder.
  4. Permissions primer: describes what the Audio workspace will ask
     for and when (mic and screen recording are requested only later,
     from Allow buttons, per the app's permission rules). Nothing is
     requested during onboarding.
* **Completion**: sets the flag, opens a new chat with the composer
  focused. Relaunch never replays the flow.

## Non-Goals

* No account, sign-in, or remote step of any kind (offline policy).
* No animated wordmark outline tracing in v1; a pulse plus fade meets
  the brand moment without bespoke motion work.
* No forced sequencing: every step is skippable; skipping all steps is
  a valid first run.
* No permission prompts inside the flow (rule: access is asked for,
  never taken; only the Audio surfaces' Allow buttons request).

## Open Questions

* **Flag location**: profile-scoped (inside the encrypted profile) or
  a plain app-level flag. Lean: app-level, because the flow should run
  before any vault unlock gate for a genuinely fresh machine.
* **Splash dismissal trigger**: app-state-ready signal from
  persistence load versus a fixed minimum display time. Lean: state
  ready, with a 400 ms floor so the mark does not flash.

**Resolved 2026-11-14** (implementation took each lean):

* **Flag location**: app-level. `FirstRunStore` writes `first_run.json`
  under `AppStorageRoot.machineRoot` (shared by every profile) through
  `AppJSONStore`, so a corrupt file is quarantined, not overwritten. The
  flow mounts inside `RootView`, which exists only after the vault gate
  has opened a profile, so a vault-locked user never sees it. A fresh Mac
  has no vault, so "before the unlock gate" and "after it" coincide for
  the one audience that sees the flow.
* **Splash dismissal**: state-ready with a 400 ms floor (`SplashPolicy`).
  Reduce Motion dismisses the moment state is ready with no floor and
  never draws the mark. A 3 s cap dismisses regardless, so a ready signal
  that never arrives cannot cover the window. "State ready" is
  `ProfileVaultCoordinator.state != .unlocking`: a locked vault is ready,
  because the unlock screen is what the splash hands off to.

## Critical User Journeys

### CUJ 1: fresh machine

* **Given**: first launch, no installed model, no flag set.
* **When**: the app opens.
* **Then**: splash, then the flow; the user installs a model (or
  skips), reads the permissions primer, and lands in a chat. A
  relaunch shows the hero directly.

### CUJ 2: skip everything

* **Given**: the flow visible.
* **When**: the user presses Skip on every step.
* **Then**: the flag is set and the app lands in a usable chat with an
  empty-state prompt to pick a model.

### CUJ 3: returning user

* **Given**: the flag set.
* **When**: any launch.
* **Then**: splash only, then the last state as today.

## Technical Plan (Architecture)

```text
Launch
  FirstRunCoordinator: flag check (app-level store)
     | fresh                         | returning
     v                               v
  SplashOverlay (always)         SplashOverlay (always)
     v
  FirstRunFlowView (step model, skippable)
     - welcome / first-model / import / primer
     v
  set flag -> existing main surface
```

The splash is an overlay on the existing window, not a second window:
no new scene, no chrome band change, and the vault unlock gate keeps
its place. The flow is a step machine with a value-type step list so
steps are injectable and testable.

## Non-Functional Requirements

* **Performance**: the splash adds no artificial delay; the flow is
  rendered only on first run.
* **Localization**: all flow copy and splash accessibility labels in 21
  languages in the same change.
* **Accessibility**: full keyboard operation (Skip always reachable,
  focus lands on the primary action of each step); VoiceOver announces
  step progress; the pulse is decorative and hidden from assistive
  tech.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/App/FirstRunCoordinator.swift` (Create)
   * **Rationale**: flag persistence, detection, and the step model.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/App/SplashOverlayView.swift` (Create)
   * **Rationale**: the overlay view with the pulse, Reduce Motion
     handling, and dismissal signal.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/App/FirstRunFlowView.swift` (Create)
   * **Rationale**: the step UI: welcome, model entry, import slot,
     primer. Each step its own subview file if it passes the 400-line
     guidance.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Rationale**: launch gating order: splash, then flow, then main
     surface; keeps the vault gate composition correct.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Installation/ModelInstallGate.swift` (Modify)
   * **Rationale**: expose the "no installed model" signal for the flow
     step, reusing the existing gate logic rather than re-deriving it.
6. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
7. `swift/TurboSparkApp/Tests/TurboSparkAppTests/FirstRunCoordinatorTests.swift` (Create)
   * **Rationale**: detection per flag state, step list content per
     machine state (model present removes the step; import spec absent
     removes the step), completion sets the flag, skip sets the flag.
     Mutation check: remove the "absent step" rule and assert the step
     list test fails.
8. `swift/TurboSparkApp/Tests/TurboSparkAppTests/FirstRunNoPermissionPromptTests.swift` (Create)
   * **Rationale**: a source-scan test in the
     `RecordingPermissionsTests` style asserting the flow sources never
     call the prompt APIs.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suites plus localization gates and
  `make design-check`.
* **Manual**: fresh-state run (flag removed), skip path, returning-user
  path. Confirm no permission prompt appears at any point in the flow.
  Run the DESIGN.md section 9 checklist on the flow and splash.

--------------------------------------------------------------------------------

## Implementation record (2026-11-14)

Files (all under `swift/TurboSparkApp`, `Sources/TurboSparkApp/App/FirstRun/`
unless noted):

* `FirstRunCoordinator.swift`: `FirstRunRecord`, `FirstRunStore`,
  `FirstRunDecision`, `FirstRunStep`, `FirstRunFlow.steps(...)`,
  `FirstRunCoordinator` (the step machine).
* `FirstRunFlowView.swift`, `FirstRunStepViews.swift`: the flow UI.
* `FirstRunHost.swift`: the modifier that mounts the flow in `RootView` and
  applies its exit (new chat with composer focus, or Discover).
* `SplashOverlayView.swift`: `SplashPolicy` and the overlay, mounted in the
  main `Window` scene in `TurboSparkApp.swift`.
* `App/RootView.swift`: one `.modifier(FirstRunHost(model:))` line.
* `Localization/Localizable.xcstrings`: 14 new keys in 21 languages.
* Tests: `FirstRunCoordinatorTests.swift` (also holds `SplashPolicyTests`),
  `FirstRunNoPermissionPromptTests.swift`.

Decisions and deviations from the draft as written:

* **Existing users are never forced through the flow.** The flag is new, so
  an upgrader has none. `FirstRunDecision.shouldRun` therefore also requires
  no chat with a message and no installed model. This also keeps a lost or
  quarantined flag file from replaying the flow for someone with history.
  Consequence: a Mac whose model library was filled by the CLI before the
  first launch skips the flow entirely. The "model present removes the
  first-model step" rule still applies live: a model that finishes
  installing while the flow is open drops the step.
* **No change to `ModelInstallGate.swift`** (draft target file 5). The review
  was right that the gate is a per-install verdict function with no
  installed-model state; the flow reads `AppModel.installed` instead.
* **Import step is absent.** `FirstRunStep.importHandoff` exists only so the
  step list can be tested; `FirstRunFlow.importWizardAvailable` is false and
  the step is in no list. The desktop-shell-hardening spec is not in this
  checkout. Turning the flag on requires a real view in `FirstRunFlowView`.
* **Browse Models ends the flow** (flag set, lands in Discover). The
  primer is not shown on that path; every Allow sheet states its own reason
  where it asks. Nothing downloads from the flow: choosing a model is the
  existing Discover section, by the person's own click.
* **Skip semantics**: Skip moves to the next step and writes nothing; Skip
  Setup (header, also Escape) and Get Started write the flag. Quitting
  mid-flow writes nothing, so the flow shows again on relaunch unless the
  person has gained history or a model by then. Back is always allowed.
* **Files live in `App/FirstRun/`**, not flat in `App/`, following the review
  note (N1).
* **Splash covers slow loads only** (follow-up, user decision). It draws
  only when state is still not ready `SplashPolicy.appearDelay` (0.4 s)
  after launch, stays at least `minimumOnScreen` (0.4 s) once drawn, never
  draws under Reduce Motion, and the 3 s `maximumDisplay` cap still ends the
  wait. Because the profile state is ready on the first frame in practice,
  it normally never draws.
* **Flow does not trap keyboard shortcuts underneath it.** The overlay is
  marked modal for VoiceOver, but menu chords still reach the window while it
  is open. Harmless (the flow ends in a new chat either way) and left as is.
* **No settings row, no new shortcut.** Nothing here is a setting or a menu
  command, so `SettingsControlCatalog` and the OC-01 catalog are untouched.
  Return (default action) and Escape (cancel action) are fixed
  sheet-convention keys, not rebindable commands.
* **Help > Show Welcome Tour** (follow-up, user decision) replays the flow
  through `FirstRunCoordinator.replay`, posted as `.showWelcomeTour`. It runs
  for existing users (bypasses `FirstRunDecision`), is ignored while a flow is
  open, never writes or clears the flag, and ends without opening a new chat
  (`FirstRunExit.stay`). **Deferred: the OC-01 catalog entry.** The request was
  a catalog command with no default chord, but `KeyboardCommand.defaultChord`
  is non-optional and OC-01 states there is no unbound state (restore/clear
  returns a command to its default), so an unbound, rebindable command needs
  an optional default through the override map, the pane rows and capture
  field, and other drafts' tests. The item carries no chord for now.

Verification run: `swift build`; focused `swift test` over
`FirstRunCoordinatorTests`, `SplashPolicyTests`,
`FirstRunNoPermissionPromptTests`, `LocalizationParityTests`,
`LocalizationTests`, `LocalizationAndAccessibilityTests`,
`RecordingPermissionsTests`, `DesignTokenRatchetTests` and
`KeyboardShortcutCatalogTests` (green); `make compile-strings`,
`make design-check` and `generate-settings-catalog.py --check` (green).
Mutation checks (all killed): removing the import-absent rule, dropping
the history/model terms of the existing-user rule, and planting a
microphone prompt call in a flow file. Not run: the DESIGN.md section 9
live-app pass (text sizes, themes, VoiceOver, RTL, Reduce Motion) and a
fresh-state launch of the app; both remain open.

*Drafted with EGM*
