# Consolidated review report: ZP-16 command center palette

Target: docs/designs/zcode-parity/ZP-16-command-center-palette.md (full file, 181 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 13, Nit 8.

## Blocker

B1. Whole doc (no section after line 128). No Risks and Mitigations section, required by the standard. Fix: add a short section (overlay regression covered by the chat-search parity test, command execution bypassing menu enablement, stale or truncated file index served from cache, history local only). [ST-1, DC-8, S8 scope note: keep it to a few lines]
B2. Lines 111-126, 117. Both alternatives are "Rejected by author" with no technical reason; "ZCode merged them for a reason" is unsupported; Alternative B ("NSPredicate") is a strawman. Fix: state concrete reasons (help search cannot return chats or inject composer references; two surfaces duplicate rendering and shortcut handling); optionally add "extend the overlay in place with prefixes only". [ST-2, DC-2, SY-7, SY-9, ZO-10, IC-10, CO-15, S8-10]

## Major

M1. Lines 35-36, 142-145, 175. "Three scope tabs" lists four. Fix: "four tabs (All plus three provider scopes)"; correct "all three scopes". [ZO-6, IC-1, CO-1, DC-1, ST-3, SY-1, S8-1]
M2. Lines 19-22 (and INDEX 43). Cited kiro spec `chat-runtime-qol` does not exist in the repo. Fix: link the real spec or say "supersedes an earlier deferral, per author decision 2026-11-11". [ZO-4, CO-10, DC-3, ST-5, S8-2]
M3. Lines 44, 156-159. Shortcut wiring: TurboSparkApp.swift (Cmd+K at ~459/463) and RootView.swift (overlay host) missing from Target Files; "pre-filtered to commands" has no mechanism; Cmd+Shift+K (Clear Chat History) adjacent. Fix: add both files, add an optional payload on `.showChatSearch` that pre-types `>`, define whether the prefix shows. [ZO-1, IC-4, CO-2, DC-4, SY-6, S8-3]
M4. Lines 152-155, Target Files. AppModel+SlashParity.swift:88 also posts `.showChatSearch`; changing the notification can break it. Fix: keep the no-payload default unchanged or add the file. [S8-N-1]
M5. Lines 38-40, 62-68, 96-105, 144-151. Command provider has no data source: menu actions have no registry, enablement and run-versus-insert undefined, commands with arguments unhandled. Fix: slash commands plus plugin commands plus a small fixed action table sharing the menu enabled predicate; disabled is a no-op; argument commands insert into the composer. [ZO-2, IC-7, CO-5, DC-5, S8-5]
M6. Lines 78-85, 107-109. All-scope merge and ranking undefined; "no new scoring claims" contradicts the plan; chat muscle memory at risk; per-keystroke cost unstated. Fix: fixed sections (Chats, Commands, Files), native order, per-section cap, first chat row pre-selected, state debounce and threading, delete the parenthetical. [ZO-3, IC-2, IC-3, IC-5, CO-4, DC-10, SY-4, S8-4]
M7. Lines 41-42, 70-76, 132-134. File index: cap (2000, depth 8) hides files, `entries(for:)` walks on cold or stale cache so "no new I/O on open" and 100 ms are false, no project means empty scope, directories included. Fix: read `cachedEntries` on open, refresh in background, state cap and empty-project behavior, mark 100 ms as a target. [ZO-5, IC-6, IC-9, CO-7, DC-6, S8-6]
M8. Lines 70-76. `@file:` handoff is a new composer seam, and edge cases are undefined (composer unmounted or disabled, directory chosen, attach failure). Fix: name the insert entry point (autocomplete controller) and list its file; define each case. [ZO-5, DC-7, S8-6]
M9. Lines 43, 169. History has no store, key, schema or privacy treatment, and is speculative for the stated problem. Fix: name the store and workspace key in one line with the 20 cap, or defer history. [ZO-7, CO-3, CO-18, DC-9, SY-14, SY-15, S8 scope note]
M10. Lines 54-58, 91. Home view undecided, no default, overlaps history. Fix: record lean (c) as the default, mark non-blocking, decide with history. [CO-3, DC-12, ST-4, SY-5, IC-11, S8-7]
M11. Lines 89-105, 142-155. Router location ambiguous, "common scored row type" undefined, provider shape missing; parser lives in the view file. Fix: put a SwiftUI-free parser reusing `ChatSearch.Query` in PaletteProviders.swift; say providers are synchronous reads of in-memory or cached state; keep the protocol small. [CO-8, ZO-8, ZO-9, ZO-10, SY-15, S8-N-3, S8-N-4]
M12. Lines 85, 152-155. Fate of ChatSearchOverlayView.swift unclear ("Modify" versus "fully replaced"). Fix: pick replace-in-place or delete-and-create. [CO-9, ST-13, S8-N-2]
M13. Lines 38-40, 98, 132-134. Plugin commands: "when plugins are enabled" undefined; skills versus commands and nested `<plugin>:<ns>:<name>` namespaces unaddressed; no-I/O claim unchecked. Fix: state PluginManager list is in memory and which entries appear. [CO-6, IC-12, ZO-2]
M14. Lines 160-177. Verification gaps: no tests for plugin namespace or Cmd+Shift+P, no empty-state or failure CUJ, prefix rules (first character only, deleting returns to All, literal `@`/`#`) unstated, no measurement for 100 ms, no release-evidence gate. Fix: add one test line each, one CUJ for Cmd+Shift+P, one-sentence prefix rule; skip harness and audit. [ST-6, ST-8, ST-9, DC-11, DC-14, DC-17, CO-11, CO-12, S8 scope note]

## Nit

N1. Line 26. "Roughly 40" slash commands unsourced (registry has 39). Fix: state the count. [CO-14, ST-7, SY-2, S8-9]
N2. Lines 135-136, 174. "21 languages" versus INDEX "ten localization parity gates". Fix: cite the gate by name. [ZO-11, CO-12, DC-16, ST-14, SY-3, S8-8]
N3. Lines 165-166. Mutation checks mis-nested, detail-level, need a clarifying clause. Fix: move to Verification Plan or delete. [ZO-12, CO-17, ST-12, SY-13, S8-11]
N4. Lines 3-7. ZCode not described as the reference product; links use `blob/main`; provenance docs unlinked; ZCode tabs renamed silently. Fix: one clause, pin to `a60ba6a7`. [CO-13, ST-11, DC-13]
N5. Lines 82-85, 17-19, 49-50. Wording: "plain-Enter", requirement inside a Then clause, wordy overview, non-goal never proposed. Fix: rephrase. [SY-10, SY-11, SY-12, ZO-9]
N6. Lines 130-136. Accessibility generic; help entries and "single-source rule" undefined. Fix: name tab switching, Esc, focus return; define rule. [DC-15, CO-16]
N7. Lines 60-138. Section order versus the EGM template unconfirmed; Alternatives line "Reason for Rejection: Rejected by author" repeats label. Fix: confirm order, drop repeat. [ST-10, SY-9]
N8. Stale-entry handling. Dead history entries, disabled plugins, missing files unhandled. Fix: validate lazily at render. [IC-8]

## Traceability

* zoom_out_report.md: ZO-1 to ZO-12 (zoom-out), IC-1 to IC-12 (inventive catalyst). Feeds B2, M1-M13, N1-N8.
* comprehension_report.md: CO-1 to CO-18. Feeds M1-M3, M5-M13, N1-N6.
* critic_report.md: DC-1 to DC-17, ST-1 to ST-14, SY-1 to SY-15. Feeds B1, B2 and all Majors.
* intermediate_zoom_out_report.md: S8-1 to S8-11, N-1 to N-4. Confirms every Stage 5 finding open; recommends rejecting scope-adding asks (full risks section, new alternatives, full history spec, prefix grammar, extra CUJs, measurement harness, locale handling, full protocol shape) and keeping ambiguity-removal fixes.
* Merge notes: the Stage 5 reports and the Stage 7 reports raised the tab-count, spec-citation, shortcut and All-ranking issues independently; each is one finding here. ST-2 rated the alternatives Blocker; zoom-out rated it Major; the higher rating is kept.
