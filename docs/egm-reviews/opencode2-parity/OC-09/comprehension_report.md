# Stage 6: Comprehension report (OC-09)

Target: docs/designs/opencode2-parity/OC-09-sources-panel.md (full file)
Reviewer: egm-comprehension (completed). Context: INDEX.md and OPENCODE2_COMPARISON.md (lines 1-136 only; the reviewer could not read the Section 4 finding text).

## What this proposes

When an assistant reply used web search or web fetch results, the app records each source (title, URL, domain) and removes duplicates within that turn. The reply's transcript row shows a chip with a glyph and a count. Clicking the chip opens a popover that groups sources by domain. Each row offers three actions: open the URL in the app's browser pane, copy the URL, or jump to the tool card that produced it. The source list is stored in the reply message's metadata and saved with the chat archive, so old replies keep their chips after relaunch. No plugin system is needed, only a metadata model, a collector, and two views.

## How the existing system works, per this document

The app has agent web tools (a search executor and a tools file). Their results appear only as tool cards through a presentation file. The document says a "structured result payload" already exists, but item 5 of the file list hedges with "if it is not already typed". Chats are Codable objects (`AppChat`) with per-message metadata, written whole to an archive. The app also has a browser pane with its own navigation and approval rules, and a permission ladder for the web tools. There is no per-turn aggregation of sources today.

## Findings

CO-1 (Major). Goals, Architecture (lines 38-41, 96-98). "Turn" and "the reply row that follows the turn" are undefined. Not stated: whether a turn has several tool steps or several assistant messages, which message carries the data, or what happens to a cancelled, errored or streaming turn at "turn completion".

CO-2 (Major). Lines 96 vs 135. "Structured result payload (already exists)" conflicts with item 5 ("if it is not already typed"). The fields a search or fetch result yields are never listed; a fetch may have no title and no fallback is given.

CO-3 (Major). Lines 38-39, 106-108, 140-143. Dedupe key not stated (exact URL, normalized URL, domain plus path). "Capped per turn" and "bounded per chat" give no numbers; "tens of entries" is an estimate. The tests check behavior that is not defined.

CO-4 (Major). Lines 42-46, 60-66, CUJ 1. Goals commit to a popover; Open Questions leave popover versus right artifact panel open. "Right artifact panel" is not defined. Fetch inclusion is open, yet Goals and CUJ 1 say "web results" and "searches". "Marked differently" is never described.

CO-5 (Major). Undefined project terms: browser pane, "existing navigation policy" and "approval rules" (which, where), permission ladder (INDEX names `AppToolPermissionEngine`, this doc does not), AppModel turn hook, tool card, chat archive, DESIGN.md section 9, `make design-check`, localization gates, `make swift-lib`, "app scheme build".

CO-6 (Major). Lines 47-48, 84-88, 106-108. Persistence compatibility is not addressed: no statement on archives lacking the new field, optional versus required, backfill of old turns. "Write cost does not change materially" has no basis.

CO-7 (Major). Lines 42-46, 74-76. Row-action detail missing: "URL path short form" undefined; reveal has no CUJ and no behavior when the card is collapsed or off screen; overflow count has no display; domain counts appear in CUJ 1 but not Goals; failed or blocked tool calls and a refused navigation are not covered.

CO-8 (Nit). Lines 28-31. Swift links are written repo-root relative and will not resolve from docs/designs/opencode2-parity/ (INDEX links use `../../`).

CO-9 (Nit). Lines 3-8. Upstream renderer and PR 58307 are only linked; one sentence of what it does would help.

CO-10 (Nit). Line 116. "21 languages" is stated without the list; the constraint lives in INDEX only.

Overall: readable as an outline, not self-contained. Turn and message model, payload shape, and dedupe and cap rules are missing or contradictory.
