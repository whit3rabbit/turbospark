# OC-09: Per-turn sources panel

Finding: Section 4, "Sources (per-turn web sources panel) - Missing in
Swift" in [OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[sources renderer](https://github.com/anomalyco/opencode/blob/dev/packages/gui-plugins/src/sources/renderer.tsx),
from [PR 58307](https://github.com/anomalyco/opencode/pull/58307)
("show the web sources a reply used").

* Author: whit3rabbit
* Status: Implemented (reveal in tool card shipped 2026-11-14; manual app checklist not run)
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

## Resolved decisions (implementation, 2026-11-14)

* **Surface shape**: popover from the chip, per the draft's lean. The
  right-column claimant is not used; a long list scrolls inside the
  popover (cap below).
* **Fetch inclusion**: both, per the lean. A successful fetch is a source
  marked "Fetched page" (document glyph); a search row is "Search result"
  (magnifier glyph). A page both searched and fetched shows as fetched and
  keeps the search title. The label is also in the VoiceOver text.
* **Derive, do not store** (review M1, M7): sources are computed at render
  time from the turn's archived `toolCalls` and `toolResults`. There is no
  new `AppChatMessage` field, collector hook or archive migration, so old
  chats get chips for free and CUJ 3 holds by construction. Plan items 2
  (`AppModel+Tools.swift`) and 5 (`ToolPresentation.swift`) are dropped.
* **Typed payload** (review M2): the registry flattens results to markdown,
  so the search parser reads `N. [title](url)` rows (the exact
  `WebSearchOutput.formatMarkdown` shape) and a fetch takes its URL from the
  call arguments. No change to `AppToolResult`.
* **Turn and chip owner** (review M3): a turn is the existing
  `TurnCollapseModel` turn (opened by a user row). The chip sits on the
  turn's final assistant row that has content and no tool calls, the row a
  collapsed turn keeps visible. A turn that ends on a tool call has no chip.
* **Source definition** (review M4): a http or https URL from a successful
  call (a failed, denied or interrupted call records nothing). Dedupe key:
  lowercased scheme and host, port, path without trailing slash, query kept,
  fragment dropped. Cap: 50 listed per turn, the rest counted and shown as
  "N more sources not shown". The Tavily direct-answer row (home page URL)
  is skipped. A redirected fetch records the requested URL, since the
  executor does not return the final one.
* **Open path** (review M6): `AppModel.openSourceInBrowserPane` uses
  `WebKitBrowserEngine.navigateAsUser`, the pane's user path, not agent
  `authorizeBrowserNavigation`. Closed pane: a new user tab, then the pane
  opens. Open pane: the visible tab navigates (the pane model does not
  follow a tab created behind it). Disabled with an explanation when the
  browser feature is off.
* **Scope cut** (review M5): the "reveal in tool card" row action was
  deferred in the first pass and shipped in the follow-up below. Domain
  grouping and counts are kept.
* **No new shortcut, setting, dependency or CI change.**

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

## Implementation record (2026-11-14)

Files:

* `State/AppChatTurnSources.swift` (new): `TurnSource`, `TurnSources`
  (collect, parse, dedupe, cap, groups, reply-row choice). Pure.
* `State/AppModel+TurnSources.swift` (new): open-in-browser-pane and copy.
* `Generation/SourcesPopoverView.swift` (new): chip and popover.
* `Generation/MessageRowView.swift`, `Generation/ChatTranscriptView.swift`:
  `sources:` input on the row; the transcript derives it per turn, only for
  a turn that made a web call.
* `Localization/Localizable.xcstrings`: 12 new keys in 21 languages (the
  chip count and overflow strings avoid plural variations by wording).
* `Tests/.../AppChatTurnSourcesTests.swift` (new).

Deferred then: reveal in the transcript's tool card (done, next record).
The manual app checklist (DESIGN.md section 9: text sizes, themes,
VoiceOver, Reduce Motion, RTL) was not run from this session.

## Implementation record: reveal in the tool card (2026-11-14)

Still derived, still nothing stored. Design:

* `TurnSource.origin` (`TurnSourceOrigin`: message id, call id) is set when
  `TurnSources.collect` first adds a source and is never overwritten by a
  later duplicate (a fetch of a searched page upgrades the kind, keeps the
  first call).
* `AppModel.revealToolCard(for:)` checks the call is still in the selected
  chat's transcript (else a toast and `false`), removes the enclosing turn's
  anchor from `collapsedTurnAnchors`, publishes
  `toolCardReveal: ToolCardReveal(messageID, callID, token)` and posts a
  VoiceOver announcement. Not persisted.
* `ChatTranscriptView` scrolls to the message row (`TSMotion.animate`, so
  Reduce Motion is honored) after one `Task.yield`, because an opened turn's
  rows do not exist until the next pass.
* `ToolGroupView` opens the group that holds the call; `ToolCallCardView`
  expands for it. The card is the only consumer
  (`consumeToolCardReveal(forCallID:)`, gated by a token high-water mark), so
  a card recreated by the lazy stack does not re-expand after the reader
  collapsed it, and a card created after the request still acts once.
* The popover row gets a third icon button, "Reveal in transcript" (the
  popover closes first). It is a real button: keyboard focusable, VoiceOver
  label and help text from the same key. Disabled for a source with no origin.
* 3 new strings in 21 languages (button, announcement, stale-call toast).
* Tests: origin first-wins, per-source calls, turn anchor lookup, reveal
  opens a collapsed turn and bumps the token, reveal consumed once and only
  by the named call, refusal when the call is gone. Mutation checks: copying
  the later duplicate's origin over the first, and dropping the pending
  check in `consumeToolCardReveal`, each fail exactly one named test.

Remaining: scrolling lands on the message ROW, not the card inside a long
multi-call row (the card is not an addressable id and a lazy stack cannot
scroll to an uninstantiated child); no highlight pulse on the revealed card.
The manual DESIGN.md section 9 pass (text sizes, themes, VoiceOver, RTL) was
not run from this session.

--------------------------------------------------------------------------------

*Drafted with EGM*
