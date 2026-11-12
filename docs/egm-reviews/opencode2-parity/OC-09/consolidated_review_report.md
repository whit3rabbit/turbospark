# Consolidated review report: OC-09 per-turn sources panel

Target: docs/designs/opencode2-parity/OC-09-sources-panel.md (full file, 156 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (one spawn round hit the concurrent-subagent limit and succeeded on retry).
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 1, Major 8, Nit 10.

## Blocker

B1. Lines 60-66 (section absent). No Alternatives Considered; the standard makes this pass/fail. Popover vs right panel is a lean without rationale, and the obvious competitor (derive sources at render time from archived tool calls and results vs store new metadata) is never weighed. Fix: add a short Alternatives section (S8 accepts a two-line note: derive over store, popover over side panel). [DC-1, ST-1, ZO-1, IC-1; S8 downgrades to a note]

## Major

M1. Lines 92-108, 126-131, 140-143. Stores a second copy of data the archive already holds (`toolCalls`, `toolResults`); adds a collector hook, new Codable field and write-cost concern. Fix: derive sources at render time; verify truncation or `archivalOutput` cannot drop URLs first. [ZO-1, IC-1, IC-3, DC-3, CO-6, S8-1]
M2. Lines 93-96, 135-137. "Structured result payload (already exists)" is false at the turn level: the registry flattens to markdown in `AppToolResult.output`; item 5 contradicts it and targets `ToolPresentation.swift`, which only registers names. Fix: choose parse-markdown or an optional typed field on `AppToolResult`; drop item 5. [ZO-2, IC-1, DC-2, CO-2, SY-7, S8-2]
M3. Lines 18, 40-41, 96-98. "Turn" and the chip's owning row undefined across multi-message agent loops, cancel, error, variants. Fix: chip on the final assistant message, union of tool results since the last user message. [ZO-3, CO-1, DC-4, ST-5, IC-3, S8-3]
M4. Lines 38-39, 64-66. "Source" undefined: failed or redirected fetches, pre vs post sanitize, synthetic Tavily direct-answer item (WebSearchExecutor lines 743-749), no dedupe key, no cap numbers. Fix: one-sentence definition plus dedupe key and numeric cap; make fetch inclusion a decision. [IC-2, CO-3, CO-4, ST-4, SY-3, S8-4]
M5. Lines 44-46, 60-66. v1 scope: three row actions plus domain grouping and counts; reveal needs a first-seen call id and scroll plumbing not designed; Goals and open questions conflict on popover and fetch marking. Fix: plain list with open and copy; cut reveal and grouping; make popover a decision. [ZO-5, IC-4, CO-4, CO-7, DC-7, DC-9, ST-4, ST-6, SY-2, S8-5]
M6. Lines 55-58, 44-46, CUJ 2, 114-115. Open path unnamed: user path is `WebKitBrowserEngine.navigateAsUser` via BrowserPaneView, distinct from agent `authorizeBrowserNavigation`; no scheme restriction. Fix: name it, allow http and https only. [ZO-4, DC-5, S8-6]
M7. Lines 47-48, 98-101, 106-108. If a stored field is kept: tolerant decoder hazard (AppChat.swift:381-407, Gotcha 13), no old-archive decode test, unsupported write-cost claim. Fix: moot under M1; otherwise decodeIfPresent plus old-archive test. [DC-3, CO-6, SY-3]
M8. Whole doc. No Risks and Mitigations section and no success measure. Fix: S8 rejects a full section as ceremony; at minimum fold the scheme, truncation and old-archive risks into the Technical Plan. [DC-6, ST-2, ST-3; S8 disputes]
M9. Lines 112-115. Performance and privacy NFRs are assertions, not testable. Fix: "popover opens with no disk read"; "collection path makes no network request"; note that opening a source is a user-initiated request. [SY-4, ZO-7, S8]

## Nit

N1. Lines 28-31, 126-140. Swift links repo-root relative and dead from this folder. Fix: `../../../swift/...`. [CO-8, DC-8, ST-8, SY-1, S8-7]
N2. Lines 3-8, 21-23, 32-35. Upstream renderer behavior and absence claim unsourced. Fix: one sentence plus audit citation (2026-10-31). [CO-9, ZO-9, SY-5, SY-6]
N3. Lines 116, 138-139. "21 languages" repeated. Fix: point to INDEX constraint. [CO-10, SY-9]
N4. Lines 148-151. DESIGN.md section 9, localization gates, browser pane, tool card unlinked or undefined. Fix: link or gloss. [CO-5, ST-7, SY-11]
N5. Lines 140-143. Test plan lacks multi-message loop, non-http scheme, old-archive decode (if field kept); CUJ 2 unmapped. [DC-11, ST-11]
N6. Lines 60-66. Open questions carry leans; resolve or name who decides. [ZO-8, SY-2]
N7. Lines 116-120. Accessibility generic (focus return, Escape, plural forms). Fix: state pattern or defer to DESIGN.md. [DC-10, SY-10; S8 deems moot]
N8. Lines 22, 109, 18-23. Section names differ from standard; Overview duplicates plan. [ST-9, ST-10; S8 skips]
N9. Lines 44-46. Ordinal phrasing for actions. [SY-8]
N10. Line 96, 126-131. Hook point not named; check Components/ for reusable rows. [DC-12, ZO-6]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-9, IC-1 to IC-5.
- comprehension_report.md: CO-1 to CO-10.
- critic_report.md: DC-1 to DC-12, ST-1 to ST-11, SY-1 to SY-11.
- intermediate_zoom_out_report.md: S8-1 to S8-7. Recommends rejecting the Alternatives blocker severity, Risks section, memoization, backfill, IDN and export-purge asks; net file list shrinks from seven to about four.
- Merge note: severity of B1 and M8 differs between Stage 7 reviewers (Blocker or Major) and Stage 8 (reject or downgrade); the higher Stage 7 severity is kept here and the dispute noted.
