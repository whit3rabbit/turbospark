# Stage 8: Intermediate zoom out report, ZP-19 web image search

Target: full file (168 lines), unedited, so all Stage 5-7 findings remain open. Reviewer egm-zoom-out, completed. Earlier code findings (ZO-1/2/3, DC-1/2/4) were relied on, not re-run.

Verdict: the draft claims reuse that does not match the code and hides a second tool. The right design is smaller: Brave-only on day one, fetch as an extension of the existing web fetch path. Most earlier findings dissolve with that cut.

## Blockers

- S8-B1 Lines 39-42, 68, 91-93. `image_fetch` is an undefined hidden tool (CO-1, DC-2, ZO-3, IC-3, ST-4). WebFetchExecutor rejects image MIME types. Fix (keep): one subsection; extend WebFetchExecutor/HttpRequestExecutor to save an image URL to a workspace path, no new tool; state destination, ladder rung, and that `AppToolSandbox.isPrivateOrMetadataHost` and the connected-address check apply; add the file to the target list. Reject a standalone image_fetch with magic-byte sniffing and its own test matrix unless the author picks the new-tool option (medium; re-size INDEX).
- S8-B2 Lines 36-38, 53-59, 82-85, 95-97. "MCP first, then Brave" is false (DC-1, ZO-1, CO-3, CO-9, ST-6, SY-5). Fix (keep, a deletion): day one is Brave images only; remove MCP from Goals and diagram; note an installed MCP image tool is already callable via the normal MCP path. Reject the ImageSearchProvider protocol (IC-2), Exa/Tavily/SearXNG variants, MCP schema and discovery rules.
- S8-B3 Lines 49, 118-121, 144-148. Settings pane and Brave key storage do not exist (ZO-2, IC-4, DC-4, CO-5). Fix (keep): cite the real file and store or state this work creates a minimal key field and Keychain entry; one shared key; opt-in, not read at launch. Reject per-backend settings UI.

## Major

- S8-M1 Lines 95-97, 118. "Reuse wholesale" overstates (DC-3, IC-1, CO-2, SY-3). Fix: name shared helpers with paths and visibility changes; list image-row rules (http(s) only, no data: URIs, length caps, control chars stripped, private hosts dropped at parse, titles untrusted). Reject a full shared-helper refactor (IC-1).
- S8-M2 Lines 22-29, 95-97. Overview and Problem misdescribe the stack (IC-1, SY-2, SY-3). Fix: rewrite accurately, cite files and defaults, scope the grep claim.
- S8-M3 Lines 66, 122-123. Timeout and cap contradict text defaults and CUJ 1 (DC-5, ZO-6, IC-4b, SY-4, IC-8). Fix: inherit about 25 s and the text cap, state the clamp, one default count; drop "tunable".
- S8-M4 Lines 140-143. Ladder positions unstated, file list incomplete (ZO-4, DC-9, ST-5, SY-8). Fix: name the AppToolPermissionEngine tier per tool; add AppToolCatalog.swift, ToolPresentation.swift, ToolCallNormalization.swift, AppHookToolNameAliases.swift, AppToolRegistry.swift.
- S8-M5 Lines 71-77. CUJ 2 either/or (CO-4, DC-8, SY-6, IC-7). Fix: adopt the text tool's availability rule, cite it, one testable Then clause.
- S8-M6 Lines 61-69. CUJ 1 not verifiable at ZP-19 landing because ZP-09 is later (IC-5, CO-6). Fix: end CUJ 1 at image saved to a workspace path with source retained; move pptx embed to a ZP-09 note; gloss and link ZP-09.
- S8-M7 No Risks section (ST-1, DC-7, IC-6). Fix: 4-5 bullets with mitigations (title injection, Brave quota or 429, expired or hotlink URLs, query leaves the machine, licensing). Reject license field and safesearch options.
- S8-M8 Lines 99-114. Alternatives lack concrete reasons (DC-6, ST-2, ST-3, SY-1, SY-9). Fix: replace "rejected by author" with trade-offs; cite or narrow the ZCode claim; add "image mode on web_search". Reject further per-provider alternatives.

## Nit

- S8-N1 Line 125. Reconcile "21 languages" with INDEX "ten gates" (DC-11, CO-10, ST-8).
- S8-N2 Lines 1-10. Add Sources block; re-size only if new-tool fetch is chosen (ST-7, IC-9, DC-13).
- S8-N3 Lines 33-35. Keep domain filters only if the text tool has them; state where filtering runs (ZO-7, CO-7).
- S8-N4 Lines 149-155. Retarget tests at behavior; move mutation checks to Verification; add design-check and localization gates; reject extra rebinding matrix (ZO-8, DC-10, SY-12, ST-9, ST-10).
- S8-N5 Lines 3-6. "Parity target" to "Reference implementation"; say whether ZCode argument names are adopted (DC-12, SY-11).
- S8-N6 Gloss "ladder", "basename", "mutation checks"; link or delete "attachment cache rules" and "per-origin rules" (CO-8, CO-11, SY-10, SY-13).
