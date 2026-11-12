# OC-07: First-run onboarding and startup splash

Findings: Section 7, "First-run onboarding - Missing in Swift" and
"Animated splash and startup screens - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[onboarding](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/renderer/onboarding/index.tsx)
and
[splash](https://github.com/anomalyco/opencode/blob/dev/packages/desktop/src/renderer/startup/splash.tsx).

* Author: whit3rabbit
* Status: Draft
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

*Drafted with EGM*
