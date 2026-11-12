# Stage 5: Initial zoom out report, ZP-19 web image search

Target: docs/designs/zcode-parity/ZP-19-web-image-search.md (full file, 168 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom out (ZO)

Verdict: scope is small and sound, but the "reuse wholesale" claim does not match the code and the fetch half is under-specified.

- ZO-1 (Major) Lines 36-38, 95-97, 131-134. The claimed precedence "configured MCP servers first, then direct API" does not exist in WebSearchExecutor.swift, which hardcodes Exa, Parallel, Brave, SearXNG and Tavily and does not call user MCP servers. An MCP image backend has no precedent and needs a new bridge. Fix: ship Brave images only on day one, or let the agent call an installed MCP image tool through the normal MCP path; resolve Open Question 1 in the doc.
- ZO-2 (Major) Lines 118-121, 134-148. `SearchSettingsPaneView.swift` does not exist; `braveApiKey` appears only as a parameter in WebSearchExecutor.swift; no settings pane or Keychain storage found. Claims "already the text path's provider" and "existing Keychain-backed storage" are unsupported. Fix: name the real file or state that this work adds key storage and UI (INDEX "small" sizing is then understated).
- ZO-3 (Major) Lines 39-43, 66-69, 91-93. `image_fetch` is a hidden new tool absent from target files, schema, ladder and tests. WebFetchExecutor.fetch throws "Unsupported fetched image content type" for image MIME types. The attachment intake is not shown to download bytes, nor how bytes reach a generated pptx path. Fix: decide the fetch interface (extend WebFetchExecutor with save-to-path, or an existing downloader), add it to the plan, say how ZP-09 references the file.
- ZO-4 (Major) Lines 140-143 vs 49-51. Only image_search gets a ladder position; the fetch step writes a file and contacts arbitrary hosts. INDEX requires a position per agent-facing tool. Fix: state both.
- ZO-5 (Nit) Lines 20-21, 95-97. Sibling versus extension not justified; shared helpers (sanitize, caps) not named; fetchLimited and validateResponseSize are private. Fix: one sentence naming shared helpers and visibility change.
- ZO-6 (Nit) Lines 122-123. 90 s timeout copied from a hosted service would block the turn; "tunable" is an unneeded knob. Fix: reuse the web search timeout.
- ZO-7 (Nit) Lines 33-35, 144-148. Allowed/blocked domains and per-backend settings UI add surface. Fix: keep only if the text tool has them.
- ZO-8 (Nit) Lines 149-155. Tests are code-level and omit fetch tests (non-image MIME refusal, size cap, redirect to private host). Fix: retarget at fetch.
- Verified: WebSearchExecutor.swift, WebTools.swift, AppToolRegistry+Vocabulary.swift exist; AppToolSandbox.isPrivateOrMetadataHost is the guard; no existing image_search in app sources.

## Inventive catalyst (IC)

No Blocker. Resolve IC-1, IC-3, IC-4 before implementation.

- IC-1 (Major) Lines 22-29, 82-95, 135-139. Design misdescribes the existing stack (enum of static functions; no MCP backend) and assumes reuse costs nothing; sanitize is tied to WebSearchResultItem. Fix: extract a shared helper layer, add the refactor to target files, restate Overview and Problem.
- IC-2 (Major) Lines 31-37, 53-59. Backend set unexamined; Brave-only fallback leaves CUJ 2 users with an error; SearXNG, Exa, Tavily have image options. Fix: narrow ImageSearchProvider protocol with an ordered list; decide the Open Question now.
- IC-3 (Major) Lines 39-42, 91-92, 118. image_fetch and "attachment URL intake" unspecified; WebFetchExecutor.isImageAttachment and HttpRequestExecutor.refuseConnectedPrivateAddress suggest an existing path. Fix: name the file and function, or add image_fetch with its own schema, ladder position and tests.
- IC-4 (Major) Lines 144-148. SearchSettingsPaneView.swift does not exist; no Brave key store found. Fix: cite the real pane and store, or state it is created, with localization and design-check obligations.
- IC-4b (Nit) Line 122. 90 s fixed timeout. Fix: short default (15-20 s), per-backend override, cancellation.
- IC-5 (Major) Lines 24-26, 59-60, 130-139. CUJ 1 depends on ZP-09, sequenced after ZP-19, so it cannot be verified when ZP-19 lands. Fix: make "find a photo, show it" the primary CUJ; pptx embed becomes a ZP-09 acceptance test; say how the source travels with the image.
- IC-6 (Major) Lines 31-45, 116-125, 140-143. Provenance is treated as trust: no licensing handling, no prompt-injection sanitization of titles and alt text, private-host image URLs only blocked at fetch, query leaves the machine. Fix: sanitize all strings, drop non-http(s) and private-host URLs at parse time, add license field or "unknown", add hostile-title test.
- IC-7 (Nit) Lines 71-77. CUJ 2 "listed or hidden" is undecided. Fix: pick the text tool's availability rule and name where it lives.
- IC-8 (Nit) Lines 154-155. Mutation checks lack concrete targets; default 8 versus "count 5"; no max clamp stated. Fix: name field and cap constant, state clamp.
- IC-9 (Nit) Lines 1-10. INDEX sizes this "small"; IC-1 to IC-4 show it is larger. Fix: re-size or defer the MCP backend.
