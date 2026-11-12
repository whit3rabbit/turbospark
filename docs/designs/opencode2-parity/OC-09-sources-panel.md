# OC-09: Per-turn sources panel

Finding: Section 4, "Sources (per-turn web sources panel) - Missing in
Swift" in [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[sources renderer](https://github.com/anomalyco/opencode/blob/dev/packages/gui-plugins/src/sources/renderer.tsx),
from [PR 58307](https://github.com/anomalyco/opencode/pull/58307)
("show the web sources a reply used").

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

When a reply used web search or fetch results, its transcript row gains
a sources chip with a count; clicking it opens a sources list for that
turn with domains, titles, and URLs, and each entry can open in the
existing browser pane. OpenCode collects citations per turn and shows
them in a side tab; the TurboSpark version needs message metadata plus
a view, no plugin host.

## The Problem

Web search exists as an agent tool
([WebSearchExecutor.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Web/WebSearchExecutor.swift),
[WebTools.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Web/WebTools.swift)),
but results surface only through the tool-call presentation in
[ToolPresentation.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolPresentation.swift).
There is no per-turn citation collection and no dedicated view: a user
cannot see which sources a specific reply relied on without opening
every tool card in the turn.

## Goals

* Source records captured when a turn's tool calls include web results:
  title, URL, and domain, deduplicated per turn.
* The reply row that follows the turn shows a sources chip (glyph plus
  count) only when sources exist.
* The chip opens a sources popover: grouped by domain, one row per
  source with title, domain, and the URL path short form.
* A row action opens the URL in the app's browser pane (existing
  navigation policy applies); a second action copies the URL; a third
  reveals it in the transcript's tool card.
* Source records persist with the chat archive so old turns keep their
  chips after relaunch.

## Non-Goals

* No inline markdown citation markers rewritten into the reply text.
* No ranking or "influence" scoring of sources.
* No scraping or content prefetch of source pages beyond what the web
  tools already fetched.
* No change to the web tools' permission ladder: opening a source URL
  from the popover is a user action and follows the browser pane's
  existing approval rules.

## Open Questions

* **Surface shape**: popover from the chip (lean, v1) versus a Sources
  mode in the right artifact panel for long source lists.
* **Fetch inclusion**: whether plain URL fetches count as sources or
  only search results. Lean: both, marked differently, because a fetched
  page the reply drew on is a source.

## Critical User Journeys

### CUJ 1: see what a reply used

* **Given**: a reply whose turn ran two searches.
* **When**: the user clicks the reply's sources chip.
* **Then**: a popover lists the deduplicated sources grouped by domain
  with counts.

### CUJ 2: open a source

* **Given**: the sources popover open.
* **When**: the user picks "open in browser pane".
* **Then**: the browser pane navigates under its existing rules and the
  popover closes.

### CUJ 3: persisted history

* **Given**: a chat with sourced replies, relaunched.
* **When**: the user scrolls to an older sourced reply.
* **Then**: its chip and list are intact.

## Technical Plan (Architecture)

```text
Web tools (search / fetch) results
        |  structured result payload (already exists)
        v
TurnSourceCollector (AppModel turn hook)
  collects title/url/domain from the turn's web tool calls,
  dedupes, stores on the turn's reply message metadata
        v
AppChat message metadata (Codable, archived with the chat)
        v
SourcesChipView on the reply row -> SourcesPopoverView
  open-in-browser via the existing browser pane routing
```

Metadata is small (tens of entries per turn at most, bounded per
chat), so the whole-archive write cost does not change materially.
Records are capped per turn with the overflow recorded as a count.

## Non-Functional Requirements

* **Performance**: collection is O(tool results) per turn; the popover
  renders from already-resident metadata.
* **Privacy**: only URLs the tools already fetched are recorded;
  nothing extra leaves the machine.
* **Localization**: chip, popover, and action copy in 21 languages in
  the same change.
* **Accessibility**: the chip announces the count; the popover is
  keyboard-navigable with standard focus handling; rows are single
  combined elements.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppChatTurnSources.swift` (Create)
   * **Rationale**: the metadata model, dedupe, caps, and the collector
     hook input; pure and headlessly testable.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Tools.swift` (Modify)
   * **Rationale**: collect from web tool results at turn completion.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/MessageRowView.swift` (Modify)
   * **Rationale**: the sources chip on reply rows with counts.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/SourcesPopoverView.swift` (Create)
   * **Rationale**: the popover: domain groups, rows, row actions.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/ToolPresentation.swift` (Modify)
   * **Rationale**: expose the structured result payload the collector
     reads, if it is not already typed.
6. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
7. `swift/TurboSparkApp/Tests/TurboSparkAppTests/AppChatTurnSourcesTests.swift` (Create)
   * **Rationale**: collection from mixed search and fetch results,
     dedupe, caps, persistence round trip through the chat archive.
     Mutation check: drop dedupe and assert the duplicate test fails.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suite plus localization gates and
  `make design-check`.
* **Manual**: a turn with searches, chip count and popover contents,
  open in browser pane, relaunch persistence, and the DESIGN.md
  section 9 checklist on the popover.

--------------------------------------------------------------------------------

*Drafted with EGM*
