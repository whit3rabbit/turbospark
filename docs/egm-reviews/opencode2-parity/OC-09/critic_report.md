# Stage 7: Critic report (OC-09)

Target: docs/designs/opencode2-parity/OC-09-sources-panel.md (full file)
One round only; convergence not assessed. Reviewers: egm-design (DC, feature-spec-standard), egm-structural (ST, feature-spec-standard), egm-style (SY, no standard). All completed.

## Design critic (DC)

DC-1 (Blocker). Whole doc, lines 60-66. No Alternatives Considered. Popover vs right panel is a lean with no rationale. Missing: derive sources lazily from archived `toolCalls`/`toolResults` at render time (no schema change); store the typed payload on `AppToolResult`; per-tool-card lists only.
DC-2 (Major). Lines 93-94, 135-137. "Structured result payload (already exists)" is false at the turn level: `AppToolRegistry.swift:441` flattens with `formatMarkdown()` into `AppToolResult.output` (String). Message holds `[AppToolCall]` and `[AppToolResult]` only. Item 5 targets `ToolPresentation.swift`, which only registers display names. Fetch fields (title, domain) not shown to exist. Fix the seam and file list.
DC-3 (Major). Lines 47-48, 98-101, 106-108. Persistence underspecified. `AppChatMessage` has a hand-written tolerant decoder (AppChat.swift:381-407); an intolerant new field discards the whole archive (swift/CLAUDE.md Gotcha 13). Require decodeIfPresent or lossy decode, CodingKeys and init updates, an old-archive decode test. "Write cost unchanged" unsupported (archive rewritten whole). Caps have no numbers.
DC-4 (Major). Lines 18, 40-41, 96-98. Turn and reply semantics undefined: an agent loop emits several assistant messages per user turn; which carries the chip, when the collector runs, cancel, error, retry, edit, variant swap (`alternates` are flattened copies), cross-search dedupe, and callID survival for "reveal".
DC-5 (Major). Lines 55-58, 114-115. Security and privacy: URL scheme (javascript:, file:, custom) on open; user path is `WebKitBrowserEngine.navigateAsUser` (line 438) called from BrowserPaneView, separate from the agent `authorizeBrowserNavigation` ladder (AppModel+Browser.swift:38); name it and require http/https only. IDN lookalikes and long titles. Query strings may retain tokens, now a second durable copy in the archive and exports. Export and delete purge unstated.
DC-6 (Nit). No Risks and Mitigations section, no success metric.
DC-7 (Nit). Lines 62-66. Fetch "marked differently" undefined and conflicts with glyph-plus-count chip.
DC-8 (Nit). Lines 28-31. Links repo-relative; should be `../../../swift/...`.
DC-9 (Nit). Lines 44-46. Three actions, CUJ 2 covers only open; reveal has no behavior when card is collapsed or truncated.
DC-10 (Nit). Lines 116-120. Accessibility and plural handling generic; rows as single elements holding three actions needs a pattern.
DC-11 (Nit). Lines 140-143. Test plan lacks old-archive decode, cap overflow, odd-scheme URL, multi-message agent loop.
DC-12 (Nit). Line 96. Hook point not named (MessageRowView.swift and AppModel+Tools.swift were located, not read).

## Structural (ST)

ST-1 (Blocker). Alternatives Considered section absent (lines 60-68). Same as DC-1.
ST-2 (Major). Risks and Mitigations absent; risk content scattered (lines 56-58, 106-108, 114-115). Unaddressed: hostile titles and URLs, archive back-compat, edit/regenerate/delete, untyped payload (line 136).
ST-3 (Major). Lines 16-34. No motivation (trust, verification) or success measure; no definition of turn, AppChat, browser pane.
ST-4 (Major). Lines 38-39 vs 63-66, 44-46. Goals vs open questions inconsistent: fetch inclusion, no source-kind field in the model, three actions but only open has a CUJ, and "reveal" needs a source-to-tool-call link the record (title, URL, domain) lacks.
ST-5 (Major). Lines 96-98 vs 129-130. TurnSourceCollector has no file; chip placement unclear when a turn has no final reply row (interrupted, tool-only).
ST-6 (Major). Lines 60-66 vs 42-46. Popover committed in Goals yet reopened as an open question; INDEX convention says list defensible alternatives with rationale.
ST-7 (Nit). Lines 116, 138, 148, 151. "21 languages", localization gates, DESIGN.md section 9 unlinked.
ST-8 (Nit). Lines 28-31. Link paths dead. Same as DC-8.
ST-9 (Nit). Lines 22, 109. Headers differ from standard names ("Problem Statement", "Proposed Architecture").
ST-10 (Nit). Lines 18-23. Overview duplicates the Technical Plan flow.
ST-11 (Nit). Verification lacks a CUJ-to-test map; CUJ 2 has no test.

## Style (SY)

SY-1 (Major). Lines 28-31, 126-140. Dead link paths. Same as DC-8.
SY-2 (Major). Lines 62-66 vs 38-46. Decision asserted and deferred; name who decides and when.
SY-3 (Major). Lines 106-108. Performance claim without numbers or measurement; give cap values or mark TBD.
SY-4 (Major). Lines 112-115. Design assertions written as requirements; make testable (popover opens without a disk read; collector and popover make no network request, with a test).
SY-5 (Nit). Lines 21-23. Upstream behavior unsourced.
SY-6 (Nit). Lines 32-35. Absence claim unverified; cite the 2026-10-31 audit.
SY-7 (Nit). Lines 96-98, 135-137. "already exists" vs "if not typed" conflict.
SY-8 (Nit). Lines 44-46. Ordinal phrasing; use three bullets.
SY-9 (Nit). Lines 116-117, 138-139. "21 languages" repeated.
SY-10 (Nit). Lines 118-120. "Standard focus handling" vague; name focus return and Escape.
SY-11 (Nit). Lines 149-152. DESIGN.md section 9 unlinked.
