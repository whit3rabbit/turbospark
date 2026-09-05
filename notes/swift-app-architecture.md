---
uuid: "f872646f-6cbf-44b4-a2bd-85a13de114f8"
title: "TurboSparkApp: AppModel structure and window layout"
summary: "State lives in three whole-file JSON archives with tolerant decoding required. AppModel is split by subview/extension past 400 lines, not grown as one file"
tags: ["swift", "app", "architecture"]
source: "swift/CLAUDE.md"
depends_on: ["50158806-656a-4a09-8bc1-39dc3801244b"]
created: "2026-09-05"
updated: "2026-09-05"
---

## How is TurboSparkApp's state organized, and what breaks it?

All app state lives in three JSON files under
`~/Library/Application Support/TurboSpark/` (`settings.json`,
`chats_archive.json`, `projects_archive.json`), each written whole with
`.atomic` on every mutation. `AppModel` itself, and every view over
~400 lines (`ChatSidebarView`, `ModelHubView`, and others), is split into
dedicated subviews, domain extensions, and type files rather than grown as
one file. `AppModel` is `@MainActor`, so a generation turn's `Task`
inherits that and every streamed `.content` chunk mutates a `@Published`
property on the main thread, one `objectWillChange` per event.

## Don't

- Don't let a store's `load()` swallow a decode failure silently. This
  already happened for real (2026-08-29): `AppChatMessage` gained two
  non-optional fields, one message written before they existed threw
  `keyNotFound`, `load()` returned the empty default, and the app opened
  with an empty chat list while a real conversation sat intact on disk,
  one keystroke from being overwritten for good. Every field added to a
  persisted struct needs `decodeIfPresent` plus a default, and `load()`
  reports the error to stderr before falling back.
- Don't blame the decode engine for a slow-feeling transcript before
  measuring the renderer. `ResponseMarkdownRenderer` and
  `ChatMessageMarkdownView` re-run per token because every chunk mutates a
  `@Published` property on the main actor. That's where per-token cost
  lives, not in generation.
- Don't add a fifth rail section without checking
  `testEverySectionHasAUniqueTitleAndShortcut`. `railButton` builds its
  hover tooltip from `title` and `shortcutKey` for every case in
  `allCases`, so a missing or duplicate one is a blank or wrong tooltip
  that only a screenshot would catch.
- Don't assume the right column can show two things at once. It's one
  slot: `previewAttachment != nil` takes it from the inspector, which is
  why `.toggleInspector` closes the preview first.
- Don't let a UI element imply certainty it doesn't have. `ModelHubView`
  shipped three badges/filters that read as working and were caught only
  by writing the first unit test over the code: an unconditional
  "verified" checkmark on every row, a capability filter matching alias
  substrings so `.conversational` silently returned the whole catalog, and
  a format label restated by hand per family (wrong on a majority of
  shipped rows). Derive a display value from the data the row actually
  carries, never restate it per branch.
- Don't render a `.unknown` measurement's zero as a reading. An unsized
  model showed "Expert Slots: 16 slots" arrived at by ignorance (`Auto`
  dividing by no expert stride). Gate every such cell on `fitIsKnown`.

See [[swift-app-architecture-2]] for lifecycle and naming traps
(`activeLoadGuard`, server/session lifetime, the two "guardrails"), and
grep this repo's Swift comments for `(state#N)` to find the fix rationale
behind a numbered defect in `swift/CLAUDE.md`'s ledger.
