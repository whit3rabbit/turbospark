# Stage 8: Intermediate zoom out report (OC-09)

Target: docs/designs/opencode2-parity/OC-09-sources-panel.md (full file). Reviewer: egm-zoom-out (completed). Target was not edited, so all Stage 5 to 7 findings are still open. The reviewer relied on earlier stages for code facts and did not re-run code searches.

Verdict: fits the INDEX "medium, app-local" slot. The main problem is that the design stores a second copy of data the chat archive already holds. Most findings disappear once that is reversed.

## Carry forward (apply)

S8-1 (Major, altitude). Lines 92-108, 126-131, 140-143. Derive sources at render time from the turn's archived tool calls and results instead of storing metadata. Removes the collector hook, the new Codable field, the write-cost paragraph and the round-trip test; CUJ 3 holds by construction. Caveat: verify that truncation or `archivalOutput` cannot drop URLs. [ZO-1, IC-1, IC-3, DC-1, DC-3, CO-6]
S8-2 (Major). Lines 96, 135-137. Name the extraction seam: parse stored markdown, or one optional typed field on `AppToolResult` emitted by the web executor (never on the message). Delete item 5 unless reveal survives. [ZO-2, IC-1, DC-2, CO-2, SY-7]
S8-3 (Major). Lines 38-41, 96-98. Define turn and chip owner: the final assistant message of the loop, unioning tool results since the last user message; interrupted or tool-only turns use the last assistant row. [ZO-3, CO-1, DC-4, ST-5]
S8-4 (Major). Lines 38-39, 64-66. Define "source": post-sanitize, successful search item or fetch, final URL, excluding the synthetic Tavily item; state the dedupe key and a numeric cap. Make fetch inclusion a decision. [IC-2, CO-3, CO-4, ST-4]
S8-5 (Major). Lines 44-46, 60-66. Trim v1: one plain list, open and copy per row; cut reveal-in-tool-card, domain grouping and counts; make popover the decision with a flip condition. [ZO-5, IC-4, CO-7, DC-9, DC-7, ST-6]
S8-6 (Major). Lines 44-46, 56-58, CUJ 2. Tie open to `navigateAsUser` (user path), not `authorizeBrowserNavigation`; allow http and https only. [ZO-4, DC-5]
S8-7 (Nit). Lines 28-31, 126-140. Dead links; use `../../../swift/...`. [CO-8, DC-8, ST-8, SY-1]

Keep as small ambiguity fixes: link or gloss browser pane, tool card, DESIGN.md section 9, localization gates [CO-5, ST-7, SY-11]; one pointer for "21 languages" [CO-10, SY-9]; testable privacy and performance wording [SY-4, ZO-7]; one sentence on upstream renderer plus audit citation [CO-9, ZO-9, SY-5, SY-6]; test plan cases for multi-message loop and non-http scheme, CUJ 2 mapping [DC-11, ST-11]; resolve open-question leans [ZO-8, SY-2].

## Reject (scope or ceremony)

- Alternatives Considered as a Blocker and full section [DC-1, ST-1]: replace with a two-line note (derive over store; popover over side panel). Note: this lowers severity relative to the structural and design reviewers; consolidated report keeps it as Major.
- Risks and Mitigations section and success metric [DC-6, ST-2].
- Memoization, backfill, global per-chat cap, write-cost measurement [IC-1, IC-5, SY-3].
- IDN lookalike handling, export and delete purge, query-string scrubbing [DC-5 extras].
- Accessibility elaboration [DC-10, SY-10], header renames and Overview trim [ST-9, ST-10], Components reuse search [ZO-6], ordinal phrasing [SY-8].

Net: file list drops from seven to about four (derivation file, chip in MessageRowView, popover view, strings and tests); AppModel+Tools.swift, ToolPresentation.swift and any AppChat schema change drop out.
