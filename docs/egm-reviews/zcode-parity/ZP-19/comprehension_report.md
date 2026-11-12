# Stage 6: Comprehension report, ZP-19 web image search

Target: full file (168 lines). Reviewer: egm-comprehension, completed.

## What this proposes

Add an `image_search` tool the agent can call to find reference photos. It takes a query, an optional count and optional allowed/blocked domains, and returns rows of title, source page, image URL and dimensions where available. It is a new `ImageSearchExecutor`. The executor tries MCP servers that expose an image_search tool first, then falls back to the Brave image API with the user's own key. Results hold only URLs and metadata. A separate "image_fetch" step downloads bytes through the existing attachment URL intake, which has SSRF guards. The goal is parity with ZCode's hosted image-search plugin so the ZP-09 office skills can embed cited photos in slides.

## How the existing system works, per this document

The app has no image search; the Images pane searches only the user's generated library. A `WebSearchExecutor` does text search with zero-config MCP backends first and a Brave API path second, and provides sanitization, size caps, timeouts and SSRF defenses that the new executor reuses. An "attachment URL intake" path downloads bytes with SSRF guards and an attachment cache. A permission ladder has risk levels. A Keychain-backed credential store exists, and a settings pane hosts the web search backends. How these connect to a user-visible `image_fetch` tool is not explained (CO-1).

## Findings

- CO-1 (Blocker) Lines 39-42, 68, 91-93, 131-156. `image_fetch` has no definition or owner: new tool or informal name for an existing mechanism? No file list entry, no schema, ladder position or output (context, disk or attachment). Fix: state which, give signature, ladder position and output; add to target files or name the existing entry point.
- CO-2 (Major) Lines 28-29, 95-97, 118-119, 144-147. WebSearchExecutor and its guards are used as a foundation without description, path or defaults; unclear whether the 90 s timeout and cap of 8 replace or inherit text values. Fix: add an inline description with path and the applicable values.
- CO-3 (Major) Lines 36-38, 55-59, 82-85. MCP backend selection undefined: discovery, schema mapping, failure and fall-through behavior. Fix: short subsection.
- CO-4 (Major) Lines 71-77, 138. CUJ 2 is an unresolved either/or (listed as unconfigured or hidden); the text tool's behavior is not stated. Fix: state it and pick one.
- CO-5 (Major) Lines 47-49, 118-121, 144-147. Credential rules and settings flow undefined; same Brave key for text and image or separate? Does a text key cover the image endpoint? Fix: say so, restate or link rules.
- CO-6 (Major) Lines 3-5, 16-18, 69, 122. ZP-09 and the 90 s ZCode figure depend on other documents without gloss or citation. Fix: one-line gloss with link; cite 9.10.
- CO-7 (Major) Lines 33-35, 43, 119-121. Row schema and "provenance tag" undefined; domain filtering owner unclear (Brave); "no user identifiers" has no mechanism. Fix: define fields and where filtering runs.
- CO-8 (Nit) Lines 131-148, 154. "Ladder" not named (AppToolPermissionEngine), "basename" unexplained, "mutation checks" unglossed.
- CO-9 (Nit) Lines 36-37, 53-59. Open Question half-answered with no decision criteria; conflicts with Goals listing MCP first. Fix: record day-one default.
- CO-10 (Nit) Lines 124-125. "21 languages" versus INDEX "ten localization parity gates" not reconciled.
- CO-11 (Nit) Lines 51, 119. "Attachment cache rules" and "per-origin rules" named without content.
