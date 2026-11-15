# OC-01: Keybind customization

Finding: Section 3, "Keybind customization - Missing in Swift" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[keybinds page](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/settings/keybinds/keybinds.tsx)
and
[override migration](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/settings/keybinds/migration.ts).

* Author: whit3rabbit
* Status: Implemented (see Implementation record)
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Let the user rebind and unbind menu command chords. A persisted override
store keyed by stable command ids drives the SwiftUI menu, the alternate
chord bridge, and the read-only shortcuts pane, which gains a per-row
record-and-assign control. OpenCode's shortcuts page rebinding is the
model, adapted to a native macOS menu instead of a web palette.

## The Problem

The Keyboard Shortcuts pane states "Shortcuts are fixed. There is no
rebinding in this version."
([KeyboardShortcutsSettingsPaneView.swift](swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutsSettingsPaneView.swift)).
The chord inventory is real and test-pinned
([KeyboardShortcutCatalog.swift](swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutCatalog.swift),
held by `KeyboardShortcutCatalogTests`), and hidden alternate chords live
in
[AlternateShortcutBridge.swift](swift/TurboSparkApp/Sources/TurboSparkApp/App/AlternateShortcutBridge.swift).
None of it is user-writable. A user who wants Cmd+Shift+R for retry, or
who conflicts with a system-wide chord from another app, cannot move a
binding.

## Goals

* Every catalog command gets a stable string id; the catalog entries,
  menu items, and alternate bridge buttons all derive from it.
* The Shortcuts pane gains a per-row capture control: press a chord to
  assign, a clear button to unbind (restore the default), and a reset-all
  action.
* Overrides persist across launches in a dedicated JSON store through the
  existing `AppJSONStore` path (quarantine on corruption, no settings.json
  growth).
* Duplicate-chord assignment is detected at capture time and refused with
  the name of the command that already holds the chord.
* Overrides apply live: the menu, the alternate bridge, and the catalog
  pane reflect the store without relaunch.
* Migration on rename: when a command id changes, an overrides-migration
  table rekeys the stored override, matching OpenCode's migration step.

## Non-Goals

* No rebinding of system-reserved chords (text editing chords, Cmd+Tab,
  Cmd+Q, Cmd+W, Cmd+,): the capture control refuses them with a reason.
* No per-context chord profiles or temporary "chord sets"; one global
  override layer.
* No in-view gestures beyond chords (no key-holder sequences).

## Open Questions

Both were resolved with the draft's own recommendation (see the
implementation record).

* **Override scope** (decided): menu commands only; the alternate bridge
  stays a mirror, never an override source.
* **Conflict policy** (decided): refuse duplicates outright, naming the
  holder.

## Critical User Journeys

### CUJ 1: rebind a command

* **Given**: the Shortcuts pane with the row for "Check for Updates".
* **When**: the user starts capture and presses Cmd+Shift+U.
* **Then**: the row shows the new chord, the menu item carries it
  immediately, and the row that previously held it (if any) was refused
  with its holder named.

### CUJ 2: unbind and restore

* **Given**: an overridden command row.
* **When**: the user presses the row's clear control.
* **Then**: the default chord returns everywhere and the override is
  removed from the store.

### CUJ 3: corrupted store

* **Given**: a hand-edited overrides file that fails to decode.
* **When**: the app launches.
* **Then**: the file is quarantined by the `AppJSONStore` contract,
  defaults apply, and a toast says settings were reset, matching the
  established corruption behavior.

## Technical Plan (Architecture)

```text
KeybindOverrideStore (AppJSONStore-backed)
  id -> KeyChord?   (nil = unbound)
        |
        v
KeyboardCommandId (stable ids)
  KeyboardShortcutCatalog entries carry ids
        |
        +--> TurboSparkApp Commands: .keyboardShortcut(resolved(id))
        +--> AlternateShortcutBridge: mirrors resolved(id)
        +--> Shortcuts pane rows: capture UI + clear + reset
```

Capture uses a local `NSEvent` monitor while the recorder is active
(chords, including function keys, are not fully reachable through
SwiftUI text-input APIs on macOS 14), then stores a normalized
`KeyChord` (key code plus modifier set, device-independent flags).

The catalog test keeps working: `KeyboardShortcutCatalogTests` asserts
the default inventory; a new test asserts that with a clean store the
resolved chord equals the default, and with a stored override the
resolved chord equals the override. Mutation check: flip the resolution
order (override ignored) and the override test must fail.

## Non-Functional Requirements

* **Latency**: resolution is a dictionary read; no launch cost.
* **Localization**: capture control labels, refusal reasons, and reset
  copy ship in all 21 catalog languages in the same change.
* **Accessibility**: the capture control is keyboard-operable (Space to
  start capture, Escape to cancel); VoiceOver announces "recording" and
  the assigned chord as the row's accessibility value.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeybindOverrideStore.swift` (Create)
   * **Rationale**: the override model, normalization, persistence, and
     migration table. Owns the only writable copy of chord state.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutCatalog.swift` (Modify)
   * **Rationale**: add stable ids to entries and a `resolvedChord(for:)`
     lookup that consults the store. The catalog remains the single
     inventory.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeyboardShortcutsSettingsPaneView.swift` (Modify)
   * **Rationale**: per-row capture control, clear and reset actions,
     duplicate-refusal display. Uses the vocabulary controls; the pane
     shell already renders its title.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/App/KeybindCaptureField.swift` (Create)
   * **Rationale**: the recorder view with the local event monitor,
     reserved-chord refusal, and VoiceOver contract. Split out to keep
     the pane near the 400-line guidance.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/App/TurboSparkApp.swift` (Modify)
   * **Rationale**: Commands read chords through the catalog resolution
     instead of fixed shortcuts.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/App/AlternateShortcutBridge.swift` (Modify)
   * **Rationale**: mirror resolved chords; keep the disabled-state
     mirroring contract.
7. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
8. `swift/TurboSparkApp/Tests/TurboSparkAppTests/KeybindOverrideStoreTests.swift` (Create)
   * **Rationale**: normalization, persistence round trip, quarantine on
     corrupt file, migration rekeying, duplicate refusal, resolution
     order with a mutation check.
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/KeyboardShortcutCatalogTests.swift` (Modify)
   * **Rationale**: default inventory unchanged with a clean store;
     overrides resolve.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: `swift test` filtered to the keybind and catalog suites,
  plus the ten localization gates (`make design-check`,
  `make compile-strings`, the parity suites from DESIGN.md section 9).
* **Manual**: rebind a menu command and confirm the menu, the bridge
  chord, and the pane agree; restart and confirm persistence; reset all
  and confirm defaults. Keyboard-only pass and VoiceOver pass per
  DESIGN.md section 9.

--------------------------------------------------------------------------------

## Implementation record

Implemented 2026-11-14 on branch `fix/b6-tool-edge-cases`. Not pushed.

### Decisions

Resolved with the draft's recommendation, plus the review's blockers and
majors where they changed behavior (EGM review under
`docs/egm-reviews/opencode2-parity/OC-01/`):

* **Override scope**: menu commands only. A command is rebindable only if
  exactly one menu item carries its chord. Generate Response, Cancel
  Generation (their chords are also bound on the Images, Audio and Humanize
  run and stop buttons), Add Files or Photos (composer button) and Start
  Recording (record bar button) are listed as Fixed and reserved for conflict
  checks. The bridge alternates stay fixed mirrors and are reserved too
  (review B2).
* **Conflict policy**: refuse, naming the holder (a command, or the command a
  hidden alternate belongs to). No "unassign the other" confirmation.
* **Unbind** (review M1): there is no unbound state. The row's clear control
  restores the default, so `id -> KeyChord?` became `id -> KeyChord`. Restoring
  is itself refused if another command has since taken the default chord.
* **Migration table** (review M6): dropped. Ids are introduced here, so there
  is nothing to rekey yet. The file carries a schema `version`, a malformed
  entry is dropped alone (lossy decode), ids this build does not know are kept
  in the file untouched, and ids are never renamed.
* **Key model** (review M3): a chord is the character SwiftUI's
  `.keyboardShortcut` takes (lowercased, Shift in the modifier set) or a named
  key (space, return, tab, delete, arrows), not a key code, so it follows the
  keyboard layout it was recorded on. Escape cancels recording and is never
  bindable. A chord needs Command or Control.
* **Reserved chords** (review M9): one static list in `KeyChord.isReserved`
  (text editing, quit, close, hide, minimize, settings, switcher, Spotlight,
  screenshots, Mission Control arrows, full screen, lock). A system shortcut
  the user rebound in System Settings cannot be detected; a chord the system
  takes never reaches the menu.
* **Ownership** (review M10): the catalog holds ids and defaults with no store
  dependency; `KeybindMap` (pure value) layers overrides; `KeybindOverrideStore`
  (one `@MainActor ObservableObject`, the `AppearanceManager` pattern) persists
  and publishes. The menu `Commands`, the pane and the Help sheet observe it,
  so changes apply live.
* **Persistence** (review M5): `keybind_overrides.json` under `AppStorageRoot`
  through `AppJSONStore` (quarantine on corruption, latched read error to the
  existing toast). The toast reads "Keyboard Shortcuts failed to decode,
  starting empty", naming the store, not "settings". A hand-edited duplicate,
  reserved or fixed-command entry is dropped on load.
* **Capture**: a local `NSEvent` key-down monitor that returns nil (so the
  chord is captured instead of firing its menu item), removed on capture,
  Escape, row disappearance and app deactivation. Return or Space on the
  focused button starts recording (review N8).

### Files

Created: `App/KeyChord.swift` (chord value, capture mapping, reserved list;
split from the store draft's file list), `App/KeybindOverrideStore.swift`,
`App/KeybindCaptureField.swift`, `Tests/.../KeybindOverrideStoreTests.swift`.
Modified: `KeyboardShortcutCatalog.swift` (ids, typed defaults, `sections(using:)`),
`KeyboardShortcutsSettingsPaneView.swift`, `TurboSparkApp.swift`,
`AlternateShortcutBridge.swift`, `ChatDialogSheets.swift` (Help sheet shows
resolved chords), `Localizable.xcstrings` (20 keys, 21 languages; the obsolete
"Shortcuts are fixed" key removed), `DESIGN.md` section 7,
`swift/docs/KEYBOARD_SHORTCUTS.md`, `App/EGM.md`.

### Deferred

* Function keys as chords: `KeyEquivalent` has no named function keys.
* Making Generate, Cancel, Add Files and Start Recording rebindable: needs the
  in-window buttons to read the same resolution.
* Rebinding the alternate bridge chords (decided against in v1).
* VoiceOver pass and keyboard-only pass on a running app (not run here).


--------------------------------------------------------------------------------

*Drafted with EGM*
