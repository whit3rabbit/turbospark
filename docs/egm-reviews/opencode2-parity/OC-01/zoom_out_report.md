# Stage 5: Initial zoom out report (OC-01)

Target: docs/designs/opencode2-parity/OC-01-keybind-customization.md (full file, 182 lines). Reviewers: egm-zoom-out, egm-inventive-catalyst. Both completed.

## Zoom-out (egm-zoom-out)

Context sufficiency passes. All named Swift files exist; the pane string at KeyboardShortcutsSettingsPaneView.swift line 13 is verified. Reuse of `AppJSONStore` (quarantine, latched toasts) is correct.

ZO-1 (Major). Lines 40-41, 102-112, 141-144. The doc treats the catalog as the source of bindings. `KeyboardShortcutRow` is display-only: `label`, pre-formatted glyph string `keys`, `altKeys` string, `id` = label. No key or modifier data exists. Real bindings are hardcoded `.keyboardShortcut(...)` calls (19 in TurboSparkApp.swift, plus composer, search overlay, AudioRecordingPanel). The catalog is a test-pinned mirror. Making it the source needs a typed `KeyChord` with display strings derived from it. That is the true scope and the doc does not state it.

ZO-2 (Major). Lines 47-49, 69-70. Duplicate detection covers only what the store knows. Chords bound outside menu commands exist (Cmd+K palette arrows, Cmd+Return and Cmd+. in the composer, Start Recording). "Menu commands only" leaves those rows unrebindable or silently broken. Alternates stay hardwired in the bridge (AlternateShortcutBridge.swift lines 26-47) and can collide with user chords (Cmd+B, Ctrl+2). The doc must say what the pane shows for non-rebindable rows and that conflict checks cover every inventory chord, alternates included.

ZO-3 (Major). Lines 22-23, 41-44, 80-82, 107. "Unbind (restore the default)" contradicts `nil = unbound`. CUJ 2 only restores the default; a command can never be left with no chord. CUJ 1 says the row that "previously held" the chord "was refused", but the refusal applies to the capture attempt. Decide whether unbind exists; if yes it needs its own control and a nil-versus-absent state, if not drop `KeyChord?`.

ZO-4 (Major, YAGNI). Lines 52-53, 138-140, 163. The migration-on-rename table has no use: ids are introduced in this change, so no legacy ids exist. Cut the goal and table. Add a schema `version` and decode tolerantly (existing TolerantDecoding.swift) so a later rename cannot crash a launch. Drop "migration rekeying" from the tests.

ZO-5 (Major). Lines 114-117, 125-132, 148-152. SwiftUI `.keyboardShortcut` takes `KeyEquivalent` plus `EventModifiers`, not a key code. Storing key code plus modifiers needs a layout-dependent mapping, fragile on non-US keyboards. Store character plus modifiers. Function keys need an explicit mapping or exclusion. The monitor must swallow the captured event so recording does not trigger the menu command.

ZO-6 (Major). Lines 50-51, 119-123, 164-167. "Live apply" needs `Commands` to observe the store, which means the store instance must be held at App level. The doc does not say where it lives, nor whether it is vault-independent. It should be an app-level preference, not profile data (Check for Updates sits outside the vault gate).

ZO-7 (Nit). Lines 55-58, 125-132. The reserved-chord list needs one named source, a short static set in the store or capture file.

ZO-8 (Nit). Lines 63-72 vs 47-49. The conflict policy is already decided in Goals ("refused outright") yet listed as open. Same for override scope, which finding ZO-2 shows is insufficient.

ZO-9 (Nit). Lines 129, 159-160, 173-174. "21 languages" and "ten gates" dominate a medium feature. Confirm the new string count is small and reuse existing accessibility vocabulary where it exists.

ZO-10 (Nit). Lines 149-152. Splitting the capture field "to keep the pane near 400 lines" is wrong: the pane is 86 lines. Justify the split by the event-monitor lifecycle instead.

Reuse check: follow the `DisabledItemStore` / `ImageStyleFavoriteStore` / `AppearanceStore` pattern (Archive struct, migrated flag). INDEX line 79 says OC-04 chords join this store, so the id namespace must not be limited to catalog rows.

Verdict: do not advance until ZO-1 to ZO-3 are resolved. ZO-1 raises scope from medium to a refactor of every `.keyboardShortcut` call site.

## Inventive catalyst (egm-inventive-catalyst)

Baseline-only read; section 3 of the comparison doc was not read. Code spot checks: catalog row `id` = `label` (KeyboardShortcutCatalog.swift:28), bridge lines 26-47 hardcode chords, line 47 derives the chord from `section.shortcutKey` with `.control`.

IC-1 (Major). Lines 38-53, 99-123, 127. The doc assumes live reactivity and a layout-independent key-code chord are both achievable, but SwiftUI shortcuts take `KeyEquivalent`. Fix: make the store an `@Observable` source of truth observed by `Commands`, the bridge and the pane; define `KeyChord` as `KeyEquivalent` plus `EventModifiers`, convert at capture via `charactersIgnoringModifiers`; add a spike with a Dvorak or AZERTY layout test and function keys, or list unsupported chords in Open Questions. Liabilities: character chords shift with layout (state this), a single observable store invalidates the whole `Commands` body, a conversion table must be maintained.

IC-2 (Major). Lines 44, 107, CUJ 2. "Unbind" is conflated with "restore default". Fix: tri-state override (`default | chord | unbound`), two row controls (Unbind, Restore default), split CUJ 1 into a successful rebind and a refused duplicate, and settle the conflict policy. Move-on-conflict with an undo toast is an option, behind confirmation. Liabilities: schema version, tests per state, extra strings.

IC-3 (Nit). Goal 6 and migration table. Speculative: stable ids exist to avoid renames. Ship a `schemaVersion` with an empty table. State how unknown stored ids are handled (retain but ignore) and add a catalog test that fails if an id disappears without a migration entry.

IC-4 (Nit). Line 109. The bridge binds control-plus-digit chords derived from `AppNavigationSection` with no menu counterpart (not fully verified). Open Question (a) should say whether they are rebindable.

IC-5 (Nit). Non-Goals. A static reserved list does not cover macOS system shortcuts (Mission Control, Spotlight), although the Problem section motivates the feature with system-wide conflicts.

IC-6 (Nit). Line 128. No copy is planned for modifier-set VoiceOver naming beyond the one accessibility bullet.
