# Stage 6: Comprehension report (OC-01)

Target: docs/designs/opencode2-parity/OC-01-keybind-customization.md (full file). Reviewer: egm-comprehension, read the document alone. Completed.

## What this proposes

Let users rebind and unbind keyboard chords for the app's menu commands. Each command gets a stable string id. A persisted JSON override store maps id to chord (nil means unbound). The menu, the hidden alternate-chord bridge, and the Shortcuts settings pane all read the resolved chord (override, else default) from that store. The pane gets a per-row capture control, a clear control, and reset-all. Duplicate and system-reserved chords are refused at capture time. A migration table rekeys stored overrides when a command id is renamed. The design is modeled on OpenCode's keybind settings.

## How the existing system works, per this document

The Shortcuts pane is read-only ("Shortcuts are fixed."). A catalog file holds the test-pinned chord inventory. A separate AlternateShortcutBridge holds hidden alternate chords that mirror menu items. Nothing is user-writable. A generic JSON store (AppJSONStore) exists with quarantine-on-corruption and a toast on reset. The document does not describe how the menu currently gets its chords (apparently fixed `.keyboardShortcut` values in TurboSparkApp.swift).

## Findings

CO-1 (Major). Lines 18, 43, 84-89, 103. "Unbind" contradicts itself: the clear button is "unbind (restore the default)" and CUJ 2 returns the default, yet the store has "nil = unbound". Unclear whether a command can be left with no chord and which control does it.

CO-2 (Major). Lines 78-82, 48-49. CUJ 1 Then mixes two outcomes: the new chord is shown and the previous holder "was refused". Per the goals a duplicate is refused, so the chord would not be assigned. Also unclear whether the refusal path is in this journey.

CO-3 (Major). Lines 70-72 vs 48-49. Conflict policy is listed as an Open Question while the Goals and CUJ 1 already commit to refusal.

CO-4 (Major). Lines 63-69, 156. "Override scope" never explains what an alternate chord is or whether alternates follow an override, stay default, or vanish. The primary-to-alternate relationship is undefined.

CO-5 (Major). Lines 26-33, 40, 106-110, 114. The doc never says how a chord is associated with a command today. `KeyChord` and `resolved(id)` are defined only by name, and what "derive from it" changes is unclear.

CO-6 (Major). Lines 52-54, 119-123, 161-164. The migration table has no stated location, format, owner, or run time. "Matching OpenCode's migration step" is unexplained; the linked file is external.

CO-7 (Major). Lines 55-58, 114-115, 34-37. "Text editing chords" is not enumerated, so the refusal rule is not implementable from the doc. Conflicts with another app's system-wide chord (the stated pain) are not addressed.

CO-8 (Major). Lines 91-97. "Settings were reset" toast is undefined, and misleading if only keybinds reset. Unclear whether only overrides or all settings reset.

CO-9 (Nit). Lines 26-28, 33, 147, 151. `KeyboardShortcutCatalogTests`, `AppJSONStore`, "vocabulary controls", "pane shell", "400-line guidance" are undefined and unlinked.

CO-10 (Nit). Lines 129, 159-160. "21 catalog languages", "ten localization gates", DESIGN.md section 9 are not listed or linked.

CO-11 (Nit). Lines 3-8, 20. The OPENCODE2_COMPARISON and OpenCode links must be opened to learn what "Missing in Swift" and "the model" mean. "Web palette" is unexplained.

CO-12 (Nit). Line 61. "Key-holder sequences" is not a standard term.

CO-13 (Nit). Lines 114-117. macOS 14 as minimum is unsourced. How the NSEvent monitor is installed and removed, and how capturing Cmd+Shift+U avoids firing the existing menu item, are unstated.

CO-14 (Nit). Lines 125-132. "Latency" is a claim, not a requirement. VoiceOver output for an unbound row is not stated.

CO-15 (Nit). Lines 134-167, 141-143. Ownership statements conflict: the store "owns the only writable copy" while the catalog is "the single inventory"; dependency direction (catalog consults store, store keyed by catalog ids) reads as circular. Pane current size is not stated.

CO-16 (Nit). Line 123. The mutation check is not stated as automated or a one-time author check.
