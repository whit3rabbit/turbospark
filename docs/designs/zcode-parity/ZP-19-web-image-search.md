# ZP-19: Web image search

Finding 9.10 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[image-search plugin](https://github.com/zai-org/ZCode/blob/main/apps/zcode-cli/packages/image-search-plugin/README.md)
(hosted MCP `image_search` server).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Give the agent an `image_search` tool so it can find reference photos
for documents, slides, and posters (the ZP-09 office skills' companion,
mirroring how ZCode's office plugins delegate to image search).
TurboSpark implements it as a local tool over a configurable search
backend, not a hosted ZCode service.

## The Problem

The app has no image search: greps for imagesearch/image_search match
nothing. The Images pane searches only the user's generated library.
When ZP-09's document skills need a photo, or a user asks for "a photo
of a red-tailed hawk for the slide", the agent cannot fetch references.
The existing `WebSearchExecutor` already solves the text twin of this
problem with zero-config MCP search backends and a Brave API path.

## Goals

* An `image_search` agent tool: query, optional count and
  allowed/blocked domains, returning result rows (title, source page,
  image URL, dimensions when the backend provides them).
* Backend precedence mirroring WebSearchExecutor: configured MCP image
  search servers first, then a direct API backend (Brave image search
  API when the user has a key; it is already the text path's provider).
* Downloads stay out of the model context by default: results are URLs
  plus metadata; an explicit companion `image_fetch` step (existing
  attachment URL intake path) pulls bytes when needed, keeping the
  existing SSRF guards.
* Results are provenance-tagged with source pages so the agent can cite
  them.

## Non-Goals

* No bundled search API keys; the Brave path requires the user's own
  key (consistent with the existing WebSearchExecutor posture).
* No ZCode-hosted JWT endpoint; no first-party service.
* No caching of image bytes beyond the existing attachment cache rules.

## Open Questions

* **MCP image-search ecosystem**: whether mainstream MCP image-search
  servers exist with a stable tool schema. If none is mature, day one
  ships the Brave-image-API backend only and the MCP backend lands as
  generic "any server exposing image_search" support. Author decides at
  implementation.

## Critical User Journeys

### CUJ 1: find a reference photo

* **Given**: the user asks for a slide with a hawk photo.
* **When**: the agent calls image_search("red-tailed hawk", count 5).
* **Then**: five rows return with source pages; the agent picks one,
  fetches it through the attachment path (with the SSRF guards), and
  embeds it in the generated pptx (ZP-09) with a cited source.

### CUJ 2: no backend configured

* **Given**: no MCP image servers and no Brave key.
* **When**: the tool is called.
* **Then**: a named configuration error explains both setup paths; the
  tool is still listed but documents itself as unconfigured (or is
  hidden, matching the text search tool's availability behavior).

## Technical Plan (Architecture)

```text
ImageSearchExecutor (new, sibling of WebSearchExecutor)
  backends in precedence:
    1. MCP servers exposing an image_search tool
    2. Brave Image Search API (user key)
       |
       v
  sanitize + cap results, attach provenance
       |
       v
  model consumes rows; bytes only via the existing
  URL attachment intake (SSRF-guarded)
```

The executor reuses WebSearchExecutor's sanitization, size caps,
timeouts, and SSRF defenses wholesale; only result-shape parsing
differs.

## Alternatives Considered

### Alternative A: return image bytes inline to the model

* **Pros**: fewer round trips.
* **Cons**: blows context with bytes; bypasses the attachment intake's
  guards; ZCode's office plugins also separate search from fetch.
* **Reason for Rejection**: Rejected by author: rows-plus-fetch keeps
  context bounded and guards centralized.

### Alternative B: browser-based scraping search

* **Pros**: no API key.
* **Cons**: fragile, slower, and drives the embedded browser through
  pages hostile to automation.
* **Reason for Rejection**: Rejected by author: unreliable.

## Non-Functional Requirements

* **Security**: the fetch path inherits the existing SSRF defenses and
  per-origin rules; search requests carry no user identifiers; the API
  key stays in the Keychain under the existing credential rules (opt-in,
  never at launch).
* **Performance**: 90 second timeout ceiling like ZCode's hosted
  server, tunable; result caps (default 8) protect context.
* **Localization**: tool-facing strings and any settings copy in the
  catalog for 21 languages.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Web/ImageSearchExecutor.swift` (Create)
   * **Type / Scope**: executor.
   * **Proposed Changes**: backend precedence, result parsing and caps,
     provenance tagging, reuse of WebSearchExecutor guards.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Web/WebTools.swift` (Modify)
   * **Type / Scope**: tool schema.
   * **Proposed Changes**: `image_search` schema (query, count,
     allowed/blocked domains), availability rule matching the text
     search tool's unconfigured behavior.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Registry/AppToolRegistry+Vocabulary.swift` (Modify)
   * **Type / Scope**: registration.
   * **Proposed Changes**: ladder entry (low risk: read-only web
     query), mirroring the web search tool.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Components/SearchSettingsPaneView.swift` (Modify; basename, the pane hosting web search backends)
   * **Type / Scope**: settings.
   * **Proposed Changes**: image backend configuration beside the text
     search configuration (Brave key flows through the existing
     Keychain-backed storage).
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/ImageSearchExecutorTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: result parsing per backend shape, caps,
     provenance presence, unconfigured error, SSRF guard inheritance
     (private-address fetch refusal), key never logged.
   * Mutation checks: strip provenance and assert the tagging test
     fails; weaken a cap and assert the cap test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the executor tests plus existing web
  search tests.
* **Manual**: with a Brave key configured, run searches from chat;
  verify row provenance and the fetch path's guard behavior against a
  private-address URL.

--------------------------------------------------------------------------------

*Drafted with EGM*
