# Stage 7: Critic report (OC-01)

Target: docs/designs/opencode2-parity/OC-01-keybind-customization.md (full file). Reviewers: egm-design and egm-structural (feature-spec-standard), egm-style (no standard). One round, no remediation, convergence not assessed. All three completed.

## Design (DC)

DC-1 (Blocker). Whole document, Open Questions lines 63-72. Alternatives Rule failed: no rejected alternatives. Missing: AppKit key equivalents instead of SwiftUI `.keyboardShortcut`; an existing recorder library; settings.json or UserDefaults (rejected at line 46 with no reason); a pane-only remap. Add an Alternatives Considered section with technical rejections. Do not attribute rejections to OpenCode maintainers without a link.

DC-2 (Blocker). Lines 40-41, 106-107, 141-143. The stable-id premise misreads the catalog. `KeyboardShortcutRow.id` is `label` (a display string), `keys` is a glyph string such as "⌘ N". The catalog drives nothing; real chords are about 19 hard-coded `.keyboardShortcut` calls in TurboSparkApp.swift (lines 331-534) plus bridge calls. The work is a data-model rewrite and a refactor of every command site, not "add ids and a lookup". Needs a data-model section: id enum, `KeyChord` type, default storage, glyph renderer, ids for `AppNavigationSection` rows.

DC-3 (Blocker). Lines 42, 65-70, 137-158. Scope of chords outside the menu. Start Recording is bound on the AudioRecordingPanel button; `keyboardShortcut` appears in 59 files (134 occurrences) including Cmd+K palette arrows, composer, sheets, Escape/Return defaults. "Menu commands only" contradicts "every catalog command" and breaks duplicate detection (silent collisions with unrebindable in-view chords). Define the rebindable set and how user chords are checked against fixed ones.

DC-4 (Major). Lines 50-51, 69-70, 110, 156-158. The alternate bridge: unspecified when the primary is rebound to an alternate (Cmd+N vs Cmd+Shift+O), when the primary is unbound, or when a user wants to remove an alternate (Cmd+B, Ctrl+N). Duplicate detection must include alternates.

DC-5 (Major). Lines 43-44, 86-89, 102-103. Unbind versus restore default conflict; nil versus absent key encoding undefined. Fix: separate controls, specify encoding (tri-state).

DC-6 (Major). Lines 77-82. CUJ 1 inconsistent. Split into a successful rebind and a refused duplicate; add CUJs for reserved chord, unbind, non-US layout.

DC-7 (Major). Lines 114-118, 129-131. SwiftUI shortcuts take `KeyEquivalent`, not a key code; layout conversion (AZERTY, Dvorak) and special keys unaddressed. An NSEvent local monitor cannot see system-intercepted chords. Reserved list is examples only (add Cmd+C/V/X/A/Z, Cmd+H, Cmd+M). Escape cancels capture yet is also a bindable chord. No minimum macOS version.

DC-8 (Major). Lines 45-47, 50-51, 95-97, 52-53. Persistence and live apply thin: how the store reports failure into the latched-error path; no schema version, migration format or location; `Commands` must observe the store and live update needs a verification step; stale ids and hand-edited duplicate chords not handled.

DC-9 (Major). Whole document. No Risks and Mitigations section. Missing: VoiceOver, Voice Control and Full Keyboard Access chord conflicts; a user stranding themselves by unbinding Keyboard Shortcuts or Settings access (needs a protected set or escape hatch); per-profile versus global storage (encrypted per-profile storage exists); NSEvent monitor lifecycle; regression risk across about 19 call sites with no per-site test (the existing test pins display strings, not live shortcuts).

DC-10 (Nit). Lines 119-123, 161-167. Mutation check only for resolution order; add duplicate detection and reserved refusal. Add an assertion that each command site uses `resolved(id)`. Give exact `swift test --filter` suites.

DC-11 (Nit). Lines 128-129, 172-174. "21 languages", "ten gates" unverified here; DESIGN.md section 9 reference may be wrong (section 7 is "Menus, commands, and shortcuts"). Verify the `Localizable.xcstrings` path and the 400-line source.

DC-12 (Nit). Lines 5-8, 52-53, 67-69. OpenCode behaviors ("dedupes in the registry", "migration step") asserted from unfetched links. Cite what the files do or soften. A short rollout or success-criteria note would help.

## Structural (ST)

ST-1 (Major). No Alternatives Considered section (standard's Alternatives Rule). Same gap as DC-1.

ST-2 (Major). No Risks and Mitigations section; risks scattered (CUJ 3 lines 91-97, lines 57-58, 38-40, 52-53); monitor lifecycle and unbound or removed-id behavior absent. Same gap as DC-9.

ST-3 (Nit). Lines 18, 43-44, 84-89, 103. "Unbind" maps to two behaviors across sections.

ST-4 (Nit). Lines 3-12. Metadata bullets are unlabeled and mix a prose "Finding:" label with a bullet list.

ST-5 (Nit). Lines 28-33, 138-166. Swift source links are repo-root-relative and do not resolve from docs/designs/opencode2-parity/ (needs `../../../`); the plan paths are unlinked backticks.

ST-6 (Nit). Lines 63-72. Open Questions precede the architecture they depend on, and the conflict-policy question is settled by Goals line 48, CUJ 1 and verification (refuse outright). Resolve or mark consistently.

## Style (SY)

SY-1 (Major). Lines 50-51, 117, 127. Uncalibrated claims: live apply has no stated mechanism, "no launch cost" is unmeasured, the macOS 14 text-input limitation is uncited.

SY-2 (Major). Lines 43, 102, 84-89. Clear control described as both unbind and restore default.

SY-3 (Major). Lines 78-82. CUJ 1 Then clause garbled (same as DC-6).

SY-4 (Major). Lines 52-54. Migration goal has no trigger or table format; ids are introduced in this change so none exist to migrate. Drop or defer for v1.

SY-5 (Major). Lines 65-72, 109-111, 156-158. Override scope and conflict policy listed as open but already decided in plan and Goals. Mark decided.

SY-6 (Nit). Lines 18-21. "OpenCode's shortcuts page rebinding is the model" is vague; name the adopted behaviors.

SY-7 (Nit). Lines 128-129, 173. "21 languages" and "ten gates" unsourced locally; link the source.

SY-8 (Nit). Lines 130-132. "Space to start capture" may conflict with focus-activation conventions; verify.

SY-9 (Nit). Lines 149-152. "400-line guidance" uncited (the pane is 86 lines).

SY-10 (Nit). Lines 119-123. Reword "mutation check" to "test must fail when resolution order is flipped".

SY-11 (Nit). Line 61. "Key-holder sequences" is jargon; say "multi-key sequences".

SY-12 (Nit). Lines 175-178. DESIGN.md section 9 cited by number, not linked.
