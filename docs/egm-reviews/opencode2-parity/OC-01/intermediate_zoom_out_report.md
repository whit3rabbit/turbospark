# Stage 8: Intermediate zoom out report (OC-01)

Target: docs/designs/opencode2-parity/OC-01-keybind-customization.md (full file, 182 lines). Reviewer: egm-zoom-out, given zoom_out_report.md, comprehension_report.md and critic_report.md. Completed. The reviewer did not open OPENCODE2_COMPARISON.md and took the "catalog is display-only" claim from Stage 5. It confirmed `addLocalMonitorForEvents` appears nowhere in swift/ (no recorder to reuse) and that TolerantDecoding.swift exists.

Verdict: the root problem is valid, but the central premise (the catalog is the source of bindings) is wrong. Fix the premise and the scope statement before advancing. Most other findings reduce to three defects: undefined unbind semantics, undefined scope, missing sections.

## Part A. Keep (removes ambiguity or corrects a false premise)

S8-A1 (Blocker). Lines 40-41, 102-112, 141-144. Re-anchor the premise: the catalog is a display mirror (`id` = label, glyph strings); real chords are about 19 hard-coded `.keyboardShortcut` calls. State that this is a data-model change: a typed `KeyChord` (KeyEquivalent plus EventModifiers) with defaults stored once and glyphs derived from it. No helper-function enumeration. Otherwise the "medium" size is false. (ZO-1, DC-2, CO-5)

S8-A2 (Blocker). Lines 42, 63-70, 137-158. Define the rebindable set in one sentence: menu commands only. All other chords (composer, palette arrows, Start Recording, bridge alternates, ctrl+digit navigation) appear as fixed rows in the pane and are reserved for conflict checks. Alternates are fixed mirrors. This is the cheapest answer to a question the doc leaves open. (ZO-2, DC-3, DC-4, CO-4, IC-4)

S8-A3 (Major). Lines 18, 43, 84-89, 102-103. Pick the YAGNI option: no unbind; drop `KeyChord?` and "nil = unbound". Reject the tri-state with two controls (IC-2, DC-5). Caveat: if "free this chord" is part of the stated pain (conflicts with other apps), unbind is in scope and needs the tri-state; make that decision explicitly.

S8-A4 (Major). Lines 78-82. CUJ 1 gets one outcome (assign succeeds); add one refusal CUJ (duplicate with holder named, plus reserved-chord refusal). (CO-2, DC-6, SY-3)

S8-A5 (Major). Lines 63-72 vs 47-49. Resolve both Open Questions as decided (refuse duplicates; menu-only) and remove the section. (ZO-8, CO-3, SY-5, ST-6)

S8-A6 (Major). Lines 114-117, 125-132. Store character plus modifiers, not key code; one sentence that chords follow the active layout; say how the monitor consumes the captured event; Escape cancels capture and is not bindable; name the supported key set, excluding function keys unless mapped. Reject a mandatory Dvorak/AZERTY spike. (ZO-5, IC-1, DC-7)

S8-A7 (Major). Lines 50-51, 119-123. Live apply: state that one app-level `@Observable` store, not profile or vault data, lives at App level and `Commands` observes it. Follow the `AppearanceStore` / `DisabledItemStore` / `ImageStyleFavoriteStore` pattern. (ZO-6, IC-1, DC-8)

S8-A8 (Major). Lines 91-97. The CUJ 3 toast must say keyboard shortcuts were reset, not "settings"; reuse the `AppJSONStore` latched-toast path. (CO-8, DC-8)

S8-A9 (Major). Lines 55-58. Name one static reserved list (Cmd+C/V/X/A/Z, Cmd+H, Cmd+M, Cmd+Tab, Cmd+Q, Cmd+W, Cmd+,). State that macOS system shortcuts (Mission Control, Spotlight) are not detectable by the monitor and are a known limitation. (ZO-7, CO-7, DC-7, IC-5)

S8-A10 (Major, narrowed from DC-9). Reset-all must be reachable by mouse and menu. Without unbind the stranding risk disappears. Reject the protected-set mechanism and the VoiceOver/Voice Control/Full Keyboard Access research as scope-adding.

S8-A11 (Major). Add short Alternatives Considered and Risks and Mitigations sections. Alternatives: settings.json or UserDefaults (reason: existing `AppJSONStore` quarantine), pane-only remap (does not solve the problem), AppKit key equivalents (mixes with SwiftUI Commands). Risks: monitor lifecycle, regression across about 19 call sites, layout dependence, id loss on catalog change. Do not invent a recorder-library alternative that was not evaluated. (DC-1, DC-9, ST-1, ST-2)

S8-A12 (Major). Lines 52-54, 138-140, 163. Drop the migration table and its tests: ids are created in this change. Replace Goal 6 with a schema `version`, tolerant decode via `TolerantDecoding.swift`, unknown ids ignored not deleted. Drop or link the "matching OpenCode's migration step" claim. (ZO-4, SY-4, IC-3, CO-6, DC-8, DC-12)

## Part B. Reject as scope-adding or noise

IC-2 tri-state and move-on-conflict undo toast; DC-9 protected-command set; IC-1 conversion-table maintenance and layout spike (one-sentence disclosure instead); IC-6 modifier-set VoiceOver copy beyond one bullet. DC-10 per-site `resolved(id)` assertion only as a grep or review-time check, but keep extra mutation checks for duplicate and reserved refusal. ST-4, SY-6, SY-11, CO-11, CO-12, CO-14, CO-16 are optional one-line wording fixes. Undefined terms and unverified numbers (CO-9, CO-10, DC-11, SY-7, SY-9, SY-12) fold into one linking and verification pass.

## Part C. Other items

S8-C1 (Major). Lines 148-152. The "400-line" justification for splitting `KeybindCaptureField.swift` is false (pane is 86 lines); justify by event-monitor lifecycle. The split is acceptable. (ZO-10, SY-9)

S8-C2 (Major). Lines 138-144. Dependency direction is circular as written. Settle: catalog (ids and defaults) does not depend on the store; the store holds overrides keyed by id; a thin resolver combines them and is what Commands, bridge and pane read. Resolves CO-15.

S8-C3 (Nit). INDEX lines 79-80 say OC-04 chords join this store. Keep the id namespace as plain strings, not an enum closed over catalog rows.

S8-C4 (Nit). Lines 125-132. "Latency: no launch cost" is a claim; delete or soften.

S8-C5 (Nit). Lines 130-132. "Space to start capture" may collide with focus-activation conventions; default to Return or click.

S8-C6 (Nit). Lines 114-117. State the minimum deployment target from Package.swift or drop the macOS 14 clause.

Reuse: `AppJSONStore`, `TolerantDecoding.swift`, the app-level store pattern. `KeybindCaptureField.swift` is the one place new machinery is warranted.

Net direction: add one data-model paragraph, one scope sentence, remove unbind, add Alternatives and Risks sections, version the file. Remove the migration table and tests, Open Questions, the 400-line justification, the Latency claim.
